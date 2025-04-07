"""
cache.py

This module provides functions to compute and store the key/value (kv) caches
of documents using a scoring model (e.g., ScoringWrapper) and later use these
cached representations to score new queries.
"""

import torch
from transformers import PreTrainedTokenizer
import os
from transformers.cache_utils import Cache, DynamicCache
from tqdm import tqdm

from transformers.cache_utils import DynamicCache  # needed for type checking
import time

def move_cache_to_cpu(batch_pkv):
    # Use a dedicated CUDA stream for asynchronous transfer
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        if hasattr(batch_pkv, "key_cache"):
            batch_pkv.key_cache = [tensor.to("cpu", non_blocking=True) for tensor in batch_pkv.key_cache]
            batch_pkv.value_cache = [tensor.to("cpu", non_blocking=True) for tensor in batch_pkv.value_cache]
        elif isinstance(batch_pkv, tuple):
            keys_cpu = tuple(tensor.to("cpu", non_blocking=True) for tensor in batch_pkv[0])
            values_cpu = tuple(tensor.to("cpu", non_blocking=True) for tensor in batch_pkv[1])
            batch_pkv = (keys_cpu, values_cpu)
    # Optionally synchronize if subsequent operations depend on the data
    stream.synchronize()
    return batch_pkv

def move_cache_to_gpu(cache_chunk, device):
    # Use a dedicated CUDA stream for asynchronous transfer to GPU
    stream = torch.cuda.Stream(device)
    with torch.cuda.stream(stream):
        if isinstance(cache_chunk, DynamicCache):
            cache_chunk.key_cache = [tensor.to(device, non_blocking=True) for tensor in cache_chunk.key_cache]
            cache_chunk.value_cache = [tensor.to(device, non_blocking=True) for tensor in cache_chunk.value_cache]
        elif isinstance(cache_chunk, tuple):
            keys_gpu = tuple(tensor.to(device, non_blocking=True) for tensor in cache_chunk[0])
            values_gpu = tuple(tensor.to(device, non_blocking=True) for tensor in cache_chunk[1])
            cache_chunk = (keys_gpu, values_gpu)
    stream.synchronize()
    return cache_chunk

def get_documents_cache(model, documents: dict, tokenizer: PreTrainedTokenizer, device: torch.device, batch_size: int = 8):
    """
    Process multiple documents in batch to compute and aggregate their kv-caches.
    """
    doc_ids = list(documents.keys())
    all_docs = [documents[doc_id] for doc_id in doc_ids]
    
    # Tokenize all documents at once. Add special tokens, pad, and truncate as needed.
    input_ids, token_type_ids, attention_mask = model.prepare_documents_input(all_docs, tokenizer)
    num_docs = input_ids.size(0)
    
    all_pkv = []           # list to store past_key_values from each batch
    
    # Process documents in batches
    for i in tqdm(range(0, num_docs, batch_size), desc="Computing document batch caches", leave=False):
        batch_input_ids = input_ids[i:i+batch_size].to(device)
        batch_attention_mask = attention_mask[i:i+batch_size].to(device)
        batch_token_type_ids = token_type_ids[i:i+batch_size].to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                token_type_ids=batch_token_type_ids,
                use_cache=True,
                return_dict=True,
            )
        batch_pkv = outputs.get("past_key_values", None)
        if batch_pkv is None:
            raise ValueError("The model did not return past_key_values. Ensure that use_cache=True is working.")
        
        # Offload the cache to CPU and store it in the list
        batch_pkv = move_cache_to_cpu(batch_pkv)

        all_pkv.append(batch_pkv)
    
    # Aggregate caches from all batches
    # If the cache is a DynamicCache (has key_cache and value_cache), use its helper method:
    if hasattr(all_pkv[0], "key_cache") and hasattr(all_pkv[0], "value_cache"):
        aggregated_cache = DynamicCache.from_batch_splits(all_pkv)
    else:
        # Legacy tuple format: tuple of length num_layers,
        # where each element is a tuple: (key_tensor, value_tensor)
        num_layers = len(all_pkv[0])
        agg_keys = []
        agg_values = []
        for layer_idx in range(num_layers):
            keys = [batch_cache[layer_idx][0] for batch_cache in all_pkv]
            values = [batch_cache[layer_idx][1] for batch_cache in all_pkv]
            agg_keys.append(torch.cat(keys, dim=0))
            agg_values.append(torch.cat(values, dim=0))
        aggregated_cache = (tuple(agg_keys), tuple(agg_values))

    return aggregated_cache

def score_with_cache(model, kv_caches, query, tokenizer: PreTrainedTokenizer, device: torch.device, batch_size: int = 8):
    """
    Scores a query using the pre-computed document caches.
    The query encoding is appended right after the document cache.
    """
    if isinstance(kv_caches, DynamicCache):
        full_batch_size = kv_caches.key_cache[0].size(0)
        split_caches = kv_caches.batch_split(full_batch_size, batch_size)
    else:
        full_batch_size = kv_caches[0][0].size(0)
        split_caches = []
        for i in range(0, full_batch_size, batch_size):
            split_key = tuple(tensor[i : i + batch_size] for tensor in kv_caches[0])
            split_value = tuple(tensor[i : i + batch_size] for tensor in kv_caches[1])
            split_caches.append((split_key, split_value))

    logits_list = []
    total_score_time = 0.0

    # Retrieve max context length from model configuration.
    max_context_length = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", None)
    if max_context_length is None:
        raise ValueError("Model config must define a context window length (n_positions or max_position_embeddings).")
    
    # Process each cache split.
    for cache_chunk in split_caches:
        if isinstance(cache_chunk, DynamicCache):
            current_batch = cache_chunk.key_cache[0].size(0)
            # Get document sequence length from the cache dimensions.
            doc_seq_len = cache_chunk.key_cache[0].size(2)
        else:
            current_batch = cache_chunk[0][0].size(0)
            doc_seq_len = cache_chunk[0][0].size(2)

        # Move cache to GPU.
        cache_chunk = move_cache_to_gpu(cache_chunk, device)

        # Prepare the query inputs for the current batch.
        # We replicate the query so that each document gets the same query.
        queries = [query] * current_batch
        query_input_ids, query_token_type_ids, query_attention_mask = model.prepare_query_input(queries, tokenizer)
        # query_input_ids: (B, query_seq_len)
        # query_attention_mask: (B, query_seq_len)
        # query_token_type_ids: (B, query_seq_len)
        query_input_ids = query_input_ids.to(device)
        query_token_type_ids = query_token_type_ids.to(device)
        query_attention_mask = query_attention_mask.to(device)

        # Determine the actual query length (number of non-padded tokens) per batch element.
        # Assuming all queries have the same length; use the first element.
        query_len = int(query_attention_mask.sum(dim=1)[0].item())

        # If document cache and query exceed max context, truncate the document cache.
        if doc_seq_len + query_len > max_context_length:
            tokens_to_drop = (doc_seq_len + query_len) - max_context_length
            new_doc_len = doc_seq_len - tokens_to_drop

            # Truncate document cache along the sequence length dimension (dim=2).
            # Truncate from the beginning of the document cache because there is a [SEP] token in the middle.
            if isinstance(cache_chunk, DynamicCache):
                cache_chunk.key_cache = [
                    torch.cat([
                        tensor[:, :, :new_doc_len - 1, :],
                        tensor[:, :, doc_seq_len - 1:doc_seq_len, :]  # append the [SEP] token
                    ], dim=2).clone() if new_doc_len > 1 else tensor[:, :, doc_seq_len - 1:doc_seq_len, :].clone()
                    for tensor in cache_chunk.key_cache
                ]
                cache_chunk.value_cache = [
                    torch.cat([
                        tensor[:, :, :new_doc_len - 1, :],
                        tensor[:, :, doc_seq_len - 1:doc_seq_len, :]
                    ], dim=2).clone() if new_doc_len > 1 else tensor[:, :, doc_seq_len - 1:doc_seq_len, :].clone()
                    for tensor in cache_chunk.value_cache
                ]
            else:
                new_keys = []
                new_values = []
                for key_tensor in cache_chunk[0]:
                    if new_doc_len > 1:
                        new_key = torch.cat([
                            key_tensor[:, :, :new_doc_len - 1, :],
                            key_tensor[:, :, doc_seq_len - 1:doc_seq_len, :]
                        ], dim=2).clone()
                    else:
                        new_key = key_tensor[:, :, doc_seq_len - 1:doc_seq_len, :].clone()
                    new_keys.append(new_key)
                for value_tensor in cache_chunk[1]:
                    if new_doc_len > 1:
                        new_value = torch.cat([
                            value_tensor[:, :, :new_doc_len - 1, :],
                            value_tensor[:, :, doc_seq_len - 1:doc_seq_len, :]
                        ], dim=2).clone()
                    else:
                        new_value = value_tensor[:, :, doc_seq_len - 1:doc_seq_len, :].clone()
                    new_values.append(new_value)
                cache_chunk = (tuple(new_keys), tuple(new_values))
            # Update document sequence length after truncation.
            doc_seq_len = doc_seq_len - tokens_to_drop

        # Build the full attention mask: document part (all ones) concatenated with query's attention mask.
        # Document part: (current_batch, doc_seq_len)
        doc_attention_mask = torch.ones((current_batch, doc_seq_len), dtype=torch.long, device=device)
        full_attention_mask = torch.cat([doc_attention_mask, query_attention_mask], dim=1)  # (B, doc_seq_len+query_seq_len)
        
        # Set cache_position so that new query tokens are positioned right after the document tokens.
        # cache_position: tensor of shape (query_len,) with values [doc_seq_len, doc_seq_len+1, ..., doc_seq_len+query_len-1]
        cache_position = torch.arange(doc_seq_len, doc_seq_len + query_len, device=device)
        
        start = time.time()
        with torch.no_grad():
            outputs = model(
                input_ids=query_input_ids,            # (B, query_seq_len)
                token_type_ids=query_token_type_ids,  # (B, query_seq_len)
                attention_mask=full_attention_mask,   # (B, doc_seq_len+query_seq_len)
                past_key_values=cache_chunk,
                cache_position=cache_position,        # (query_len,)
                use_cache=True,
                return_dict=True
            )
        elapsed = time.time() - start
        total_score_time += elapsed

        logits_list.append(outputs["logits"])

        # Optionally offload the cache back to CPU.
        cache_chunk = move_cache_to_cpu(cache_chunk)

    return torch.cat(logits_list, dim=0).squeeze(-1).tolist(), total_score_time

def save_kv_cache(kv_cache_dict: dict, filename: str):
    """
    Save the kv-cache dictionary to a file.
    """
    torch.save(kv_cache_dict, filename)
    print(f"kv-cache dictionary saved to {filename}")

def load_kv_cache_from_file(filename: str) -> dict:
    """
    Load the kv-cache dictionary from a file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cache file {filename} does not exist.")
    kv_cache_dict = torch.load(filename, map_location="cpu")
    return kv_cache_dict

def build_and_save_documents_cache(model, documents: dict, tokenizer: PreTrainedTokenizer, device: torch.device, filename: str, batch_size: int = 8):
    """
    Convert the kv-cache dict to doc_id: kv_cache dict and save it to a file.
    """

    # List of document IDs and corresponding texts.
    doc_ids = list(documents.keys())
    all_docs = [documents[doc_id] for doc_id in doc_ids]
    kv_cache_dict = {}

    # Tokenize all documents at once.
    # This function is assumed to exist on your model. If not, see below for a dummy implementation.
    input_ids, token_type_ids, attention_mask = model.prepare_documents_input(all_docs, tokenizer)

    num_docs = input_ids.size(0)

    # Process documents in batches.
    for i in range(0, num_docs, batch_size):
        # Get the current batch of document IDs.
        batch_doc_ids = doc_ids[i:i+batch_size]
        # Slice and move inputs to the target device.
        batch_input_ids = input_ids[i:i+batch_size].to(device)
        batch_attention_mask = attention_mask[i:i+batch_size].to(device)
        batch_token_type_ids = token_type_ids[i:i+batch_size].to(device)

        # Run the forward pass with caching enabled.
        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            token_type_ids=batch_token_type_ids,
            use_cache=True,
            return_dict=True,
        )
        batch_past_key_values = outputs.get("past_key_values", None)

        if batch_past_key_values is None:
            raise ValueError("The model did not return past_key_values. Ensure that use_cache=True is working.")

        # For each document in the batch, extract its cache from the batch dimension.
        # If the KV cache is in tuple format:
        if isinstance(batch_past_key_values, (tuple, list)):
            # Each element in the tuple corresponds to a layer and is a tuple (key, value)
            for j, doc_id in enumerate(batch_doc_ids):
                doc_cache = []
                for layer_cache in batch_past_key_values:
                    # layer_cache[0] and layer_cache[1] are tensors of shape
                    # (batch_size, num_heads, seq_length, head_dim)
                    key = layer_cache[0][j:j+1].cpu()
                    value = layer_cache[1][j:j+1].cpu()
                    doc_cache.append((key, value))
                kv_cache_dict[doc_id] = tuple(doc_cache)
        # Else, if using a DynamicCache instance:
        elif hasattr(batch_past_key_values, "key_cache") and hasattr(batch_past_key_values, "value_cache"):
            for j, doc_id in enumerate(batch_doc_ids):
                # Extract j-th element from each layer's cache.
                doc_key_cache = [k[j:j+1].cpu() for k in batch_past_key_values.key_cache]
                doc_value_cache = [v[j:j+1].cpu() for v in batch_past_key_values.value_cache]
                kv_cache_dict[doc_id] = {"key_cache": doc_key_cache, "value_cache": doc_value_cache}
        else:
            raise TypeError("Unrecognized KV cache format.")

    return kv_cache_dict


def score_with_cache_from_file(model, cache_filename: str, query, tokenizer: PreTrainedTokenizer, device: torch.device, batch_size: int = 8):
    pass