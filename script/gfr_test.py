# Example usage for inference:

# Assume you have a pretrained tokenizer that provides these attributes/methods.
# For example, using Hugging Face's BertTokenizer:
#
#   from transformers import BertTokenizer
#   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
# And your model is an instance of GFRModel.

import torch
from GFR.configuration_GFR import GFRConfig
from GFR.modeling_GFR import GFRModel
from transformers import AutoTokenizer

config = GFRConfig(
    vocab_size=32000,
    hidden_size=256,         # smaller hidden size for demo purposes
    num_hidden_layers=4,       # fewer layers for faster training in demo
    max_position_embeddings=512,
)

# Instantiate the model.
model = GFRModel(config)

# Use the tokenizer from Zamba
tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba-7B-v1")

# Example document and query.
document = "This is an example document discussing machine learning techniques and applications."
query = "What are the applications of machine learning?"

# Prepare the input tensors.
input_ids, token_type_ids = model.prepare_input(document, query, tokenizer, max_length=512)

# Set the model to evaluation mode.
model.eval()
with torch.no_grad():
    # Forward pass through the model to get the relevance score.
    relevance_score = model(input_ids=input_ids, token_type_ids=token_type_ids)
    print("Predicted relevance score:", relevance_score.item())