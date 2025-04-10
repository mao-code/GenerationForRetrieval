# Test GFR2 Causal LM and Scoring Model
# Use initial model (haven't been trained yet)

import torch
from GFR2.configuration_GFR2 import GFR2Config
from GFR2.modeling_GFR2 import GFR2Model, GFR2ModelForCausalLM, GFR2ModelForSequenceScoring
from GFR2.tokenizer_utils import get_tokenizer
import logging

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

from torchinfo import summary

def main():
    logging.info("Initializing GFR2 model for inference...")
    tokenizer = get_tokenizer()
    config = GFR2Config(
        vocab_size=len(tokenizer), 
        num_hidden_blocks=3,        # N GFR2Blocks,
        num_layers_per_block=9,     # 9 layers per block
    )

    # Instantiate the GFR2 model for causal language modeling.
    model_for_causallm = GFR2ModelForCausalLM(config).to('cuda')
    num_params = sum(p.numel() for p in model_for_causallm.parameters())
    print(f"Number of parameters for causal LM: {num_params}")

    # Instantiate the GFR2 model for sequence scoring.
    model_for_scoring = GFR2ModelForSequenceScoring(config).to('cuda')
    num_params = sum(p.numel() for p in model_for_scoring.parameters())
    print(f"Number of parameters for sequence scoring model: {num_params}")

    # Test the causal LM
    logging.info("Testing causal LM...")
    model_for_causallm.eval()
    with torch.no_grad():
        # Generate a sequence from the model.
        input_ids = tokenizer("The meaning of life is", return_tensors="pt").input_ids.to('cuda')
        outputs = model_for_causallm.generate(input_ids=input_ids, max_length=50)
        print("Generated sequence:", tokenizer.decode(outputs[0], skip_special_tokens=True))

    # Prepare the input tensors.
    logging.info("Testing sequence scoring model...")
    document = "This is an example document discussing machine learning techniques and applications."
    query = "What are the applications of machine learning?"

    input_ids, token_type_ids, attention_mask = model_for_scoring.prepare_input([document], [query], tokenizer, max_length=512)
    input_ids = input_ids.to('cuda')
    token_type_ids = token_type_ids.to('cuda')
    attention_mask = attention_mask.to('cuda')

    model_for_scoring.eval()
    with torch.no_grad():
        # Forward pass through the model to get the relevance score.
        res = model_for_scoring(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, return_dict=True)
        print("Predicted relevance score:", res["logits"].item())

if __name__ == '__main__':
    main()

    """
    python -m script.gfr_test
    """