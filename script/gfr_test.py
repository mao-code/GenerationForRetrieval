import torch
from GFR.configuration_GFR import GFRConfig
from GFR.modeling_GFR import GFRModel
from transformers import AutoTokenizer, BertTokenizer

import logging

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

from torchinfo import summary

def main():
    logging.info("Initializing GFR model for inference...")
    # Use the tokenizer from Zamba (did not have CLS and SEP token ids)
    # tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba-7B-v1") # Llama fast tokenizer
    # Load the pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    config = GFRConfig(
        vocab_size=len(tokenizer), 
        hidden_size=256,         # smaller hidden size for demo purposes
        num_hidden_block=3,
        num_layers_per_block=8,
        max_position_embeddings=512,
    )

    # Instantiate the model.
    model = GFRModel(config)
    model.to('cuda')
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    summary(
        model,
        input_size=(1, 32),  
        dtypes=[torch.long],
    )
    logging.info("Model initialized successfully.")

    # Example document and query.
    document = "This is an example document discussing machine learning techniques and applications."
    query = "What are the applications of machine learning?"

    # Prepare the input tensors.
    logging.info("Preparing input tensors...")
    input_ids, token_type_ids = model.prepare_input([document], [query], tokenizer, max_length=512)
    input_ids = input_ids.to('cuda')
    token_type_ids = token_type_ids.to('cuda')

    # Set the model to evaluation mode.
    logging.info("Running inference...")
    model.eval()
    with torch.no_grad():
        # Forward pass through the model to get the relevance score.
        relevance_score, model_output = model(input_ids=input_ids, token_type_ids=token_type_ids, return_dict=True)
        print("Predicted relevance score:", relevance_score.item())

if __name__ == '__main__':
    main()

    """
    python -m script.gfr_test
    """