import time
import torch
from transformers import LlamaTokenizer
from MLA_CDR.configuration_mla import MLAConfig
from MLA_CDR.modeling_mla import MLAForSequenceScoring
from cache import (
    get_documents_cache,
    move_cache_to_cpu,
    move_cache_to_gpu,
    score_with_cache,
)

def compute_cache_size(cache):
    """
    Compute the total memory footprint (in bytes) of the key/value cache.
    Supports both DynamicCache (with key_cache/value_cache attributes)
    and legacy tuple format.
    """
    total_bytes = 0
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for tensor in cache.key_cache:
            total_bytes += tensor.numel() * tensor.element_size()
        for tensor in cache.value_cache:
            total_bytes += tensor.numel() * tensor.element_size()
    elif isinstance(cache, tuple):
        for layer in cache:
            key, value = layer
            total_bytes += key.numel() * key.element_size()
            total_bytes += value.numel() * value.element_size()
    else:
        print("Warning: Unrecognized cache format.")
    return total_bytes

def measure_ttft_no_cache(model, full_input, device):
    """
    Measure time-to-first-token (TTFT) for processing the full input (document + query)
    without caching.
    """
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        _ = model(full_input, use_cache=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start_time
    return elapsed

def test_cache_vs_noncache_batch_scoring(model, tokenizer, device, batch_size=8):
    """
    Compare non-cached (full concatenated input) vs. cached scoring on a batch of documents.
    Also, measure the time spent for cache generation and movement between GPU and CPU,
    and report the cache size.
    """
    # Define sample texts.
    doc_text = "word " * 512    # ~512 tokens per document
    query_text = "query " * 15   # ~15 tokens per query

    # --- Non-cached inference (full input) ---
    # Tokenize document and query.
    doc_tokens = tokenizer.encode(doc_text, return_tensors="pt").to(device)
    query_tokens = tokenizer.encode(query_text, return_tensors="pt").to(device)
    if batch_size > 1:
        doc_tokens = doc_tokens.repeat(batch_size, 1)
        query_tokens = query_tokens.repeat(batch_size, 1)
    full_input = torch.cat((doc_tokens, query_tokens), dim=1)
    # Warm-up.
    with torch.no_grad():
        _ = model(full_input, use_cache=True)
    # Timed run.
    noncache_time = measure_ttft_no_cache(model, full_input, device)

    # --- Cached inference ---
    # Build a dict of documents for batch scoring.
    documents = {f"doc_{i}": doc_text for i in range(batch_size)}
    
    # Warm-up cache generation.
    with torch.no_grad():
        _ = get_documents_cache(model, documents, tokenizer, device, batch_size=batch_size)
    
    # Measure cache generation time.
    start = time.time()
    cache = get_documents_cache(model, documents, tokenizer, device, batch_size=batch_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    cache_gen_time = time.time() - start

    # Measure movement: GPU → CPU.
    start = time.time()
    cache_cpu = move_cache_to_cpu(cache)
    if device.type == "cuda":
        torch.cuda.synchronize()
    cpu_move_time = time.time() - start

    # Measure movement: CPU → GPU.
    start = time.time()
    cache_gpu = move_cache_to_gpu(cache_cpu, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    gpu_move_time = time.time() - start

    # Warm-up cached scoring.
    with torch.no_grad():
        _ = score_with_cache(model, cache_gpu, query_text, tokenizer, device, batch_size=batch_size)
    
    # Measure cached scoring time (query inference using precomputed cache).
    start = time.time()
    scores, cache_forward_time = score_with_cache(model, cache_gpu, query_text, tokenizer, device, batch_size=batch_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_cached_time = time.time() - start

    # Compute overhead in the score_with_cache function (if any).
    movement_overhead_in_score = total_cached_time - cache_forward_time

    # Compute cache size.
    cache_size_bytes = compute_cache_size(cache)
    cache_size_mb = cache_size_bytes / (1024 * 1024)

    # --- Report results ---
    print("=== Batch Scoring Speed Comparison ===")
    print(f"Batch size: {batch_size}")
    print(f"Non-cached inference time (full input): {noncache_time*1000:.2f} ms")
    print(f"Cache generation time:               {cache_gen_time*1000:.2f} ms")
    print(f"Cache move to CPU time:              {cpu_move_time*1000:.2f} ms")
    print(f"Cache move to GPU time:              {gpu_move_time*1000:.2f} ms")
    print(f"Cached query inference (forward):    {cache_forward_time*1000:.2f} ms")
    print(f"Total cached scoring time:           {total_cached_time*1000:.2f} ms")
    print(f"Movement overhead in score_with_cache:{movement_overhead_in_score*1000:.2f} ms")
    print(f"Cache size:                          {cache_size_mb:.2f} MB\n")

def main():
    # Device selection: GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the tokenizer.
    tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({"sep_token": "[SEP]"})
    if "[SCORE]" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["[SCORE]"]})
    
    # Initialize the model.
    print("Initializing MLA model for pretraining...")
    config = MLAConfig(vocab_size=len(tokenizer))
    model = MLAForSequenceScoring(config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    model.eval()  # Set model to evaluation mode.

    # Test for various batch sizes.
    for batch_size in [1, 8, 16]:
        test_cache_vs_noncache_batch_scoring(model, tokenizer, device, batch_size=batch_size)

if __name__ == "__main__":
    main()
