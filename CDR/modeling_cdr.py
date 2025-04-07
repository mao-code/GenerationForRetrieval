from transformers import PreTrainedModel, AutoConfig
from transformers.cache_utils import Cache, DynamicCache
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm

class ScoringWrapper(PreTrainedModel):
    # Define config_class to use AutoConfig
    config_class = AutoConfig

    def __init__(self, config, decoder):
        super().__init__(config)

        # Store the base decoder model (e.g., GPT2Model, OPTModel, BloomModel)
        self.decoder = decoder

        # Add token type embeddings (e.g., 2 types: document and query/special tokens)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)

        # Add a score head to output a single value from the last token's hidden state
        self.score_head = nn.Linear(config.hidden_size, 1)

        # Initialize weights for the new layers
        self.post_init()

    def forward(       
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get token embeddings from the base model
        token_embeds = self.decoder.get_input_embeddings()(input_ids)

        # Handle position embeddings (optional, depending on the model)
        # position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)

        # position_embeds = 0
        # if hasattr(self.decoder, 'get_position_embeddings'):
        #     position_embeds = self.decoder.get_position_embeddings()(position_ids)

        # Add token type embeddings (default to 0 if not provided)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        # Combine all embeddings
        inputs_embeds = token_embeds + token_type_embeds # + position_embeds

        # Pass through the base model to get hidden states
        outputs = self.decoder(
            input_ids=None,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs
        )
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Compute the position of [SCORE] for each sequence
        # If pad to the right, sum of attention_mask gives the unpadded length; [SCORE] is at length - 1
        # score_positions = attention_mask.sum(dim=1) - 1  # Shape: (batch_size,) 

        # Ensure positions are within bounds
        # score_positions = torch.clamp(score_positions, min=0, max=hidden_states.size(1) - 1)

        # If pad to the left, the position of [SCORE] is at the last position of the sequence
        score_positions = hidden_states.size(1) - 1

        # Extract the hidden state of [SCORE] for each sequence in the batch
        batch_indices = torch.arange(hidden_states.size(0))  # [0, 1, 2, ..., batch_size-1]
        score_hidden = hidden_states[batch_indices, score_positions]  # Shape: (batch_size, hidden_size)
        logits = self.score_head(score_hidden).squeeze(-1)  # [batch_size]

        # If labels are provided, compute the loss (e.g., for binary classification)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        else:
            return {
                "logits": logits, 
                "loss": loss, 
                "past_key_values": outputs.past_key_values,
                "hidden_states": outputs.hidden_states, 
                "attentions": outputs.attentions
            }
        
    def prepare_input(self, documents: list, queries: list, tokenizer):
        """        
        Prepares batched inputs for the model by truncating only the document tokens if needed.
        Padding is done to the longest sequence in the batch.
        """
        # Retrieve the maximum allowed sequence length from the model's config.
        max_config_length = getattr(self.config, 'n_positions', None) or getattr(self.config, 'max_position_embeddings', None)
        if max_config_length is None:
            raise ValueError(
                "The model's configuration does not specify a maximum sequence length. "
                "Please use a model that defines 'n_positions' or 'max_position_embeddings'."
            )

        input_ids_list = []
        token_type_ids_list = []
        
        # Process each document-query pair.
        for document, query in zip(documents, queries):
            # Encode without adding special tokens.
            doc_ids = tokenizer.encode(document, add_special_tokens=False)
            query_ids = tokenizer.encode(query, add_special_tokens=False)
            score_id = tokenizer.convert_tokens_to_ids("[SCORE]")
            
            # Reserve tokens for [SEP] and [SCORE] plus all query tokens.
            reserved_tokens = 2 + len(query_ids)
            available_doc_length = max_config_length - reserved_tokens
            if available_doc_length < 0:
                raise ValueError("max_config_length is too small to accommodate the query and required special tokens.")
            
            # Truncate document tokens if needed.
            truncated_doc_ids = doc_ids[:available_doc_length]

            # Build the input sequence: document tokens + [SEP] + query tokens + [SCORE]
            input_ids = truncated_doc_ids + [tokenizer.sep_token_id] + query_ids + [score_id]
            input_ids_list.append(input_ids)
            
            # Create token type IDs: 0 for document part (including [SEP]), 1 for query part (including [SCORE])
            doc_part_length = len(truncated_doc_ids) + 1
            query_part_length = len(query_ids) + 1
            token_type_ids = [0] * doc_part_length + [1] * query_part_length
            token_type_ids_list.append(token_type_ids)
        
        # Determine the maximum sequence length in the current batch.
        batch_max_length = max(len(ids) for ids in input_ids_list)
        
        # Pad each sequence and corresponding token types to the batch maximum.
        padded_input_ids = []
        padded_token_type_ids = []
        attention_masks_list = []
        
        for ids, tt_ids in zip(input_ids_list, token_type_ids_list):
            pad_length = batch_max_length - len(ids)
            padded_ids = [tokenizer.pad_token_id] * pad_length + ids
            padded_tt_ids = [0] * pad_length + tt_ids
            # Create attention mask: 1 for original tokens, 0 for pad tokens.
            attention_mask = [0] * pad_length + [1] * len(ids)
            
            padded_input_ids.append(padded_ids)
            padded_token_type_ids.append(padded_tt_ids)
            attention_masks_list.append(attention_mask)
        
        # Convert lists to tensors.
        input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
        token_type_ids_tensor = torch.tensor(padded_token_type_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_masks_list, dtype=torch.long)
        
        return input_ids_tensor, token_type_ids_tensor, attention_mask_tensor

    # TODO: Pad should Pad to the end of the input sequence, not the doc part. Can be optimized in the future.
    def prepare_documents_input(self, documents: list, tokenizer):
        """
        Prepares batched document inputs.
        Each document is tokenized (with a trailing [SEP]) and then dynamically padded
        to the maximum document length within the batch.
        """

        # Append [SEP] to each document
        doc_sequences = [doc + " [SEP]" for doc in documents]
        inputs = tokenizer(
            doc_sequences,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        ) # shape: (batch_size, doc_max_len)
        
        input_ids = inputs["input_ids"]
        token_type_ids = torch.zeros_like(input_ids)  # All 0 for doc_ids and [SEP]
        attention_mask = inputs["attention_mask"]

        return input_ids, token_type_ids, attention_mask

    def prepare_query_input(self, queries: list, tokenizer):
        """
        Prepares batched query inputs.
        """
        
        # Append [SCORE] to each query
        query_sequences = [query + " [SCORE]" for query in queries]
        inputs = tokenizer(
            query_sequences,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        ) # shape: (batch_size, query_max_len)
        
        input_ids = inputs["input_ids"]
        token_type_ids = torch.ones_like(input_ids)  # All 1 for query_ids and [SCORE]
        attention_mask = inputs["attention_mask"]
        
        return input_ids, token_type_ids, attention_mask

    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.decoder.set_input_embeddings(value)

    def get_position_embeddings(self):
        if hasattr(self.decoder, 'get_position_embeddings'):
            return self.decoder.get_position_embeddings()
        else:
            return None