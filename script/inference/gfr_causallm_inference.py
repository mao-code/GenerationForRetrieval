import argparse
import torch
from transformers import LlamaTokenizer
from GFR.modeling_GFR import GFRForCausalLM

def main():
    parser = argparse.ArgumentParser(description="Inference script for GFR model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model directory (e.g., 'gfr_causal_lm_final_finewebedu_v2')."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The meaning of life is",
        help="Text prompt to generate output from."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="Maximum total length (prompt + generated tokens)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference ('cuda' or 'cpu')."
    )
    # If you saved your tokenizer along with the model, you can load it from model_path.
    parser.add_argument(
        "--tokenizer_source",
        type=str,
        default="huggyllama/llama-7b",
        help="Tokenizer source. Use model_path if you saved the tokenizer with your model."
    )
    args = parser.parse_args()

    # Load the tokenizer.
    # If you saved the tokenizer with your model, you can replace args.tokenizer_source with args.model_path.
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_source)

    # Load the model
    model = GFRForCausalLM.from_pretrained(args.model_path)
    model.eval()  # switch to evaluation mode

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model.to(device)

    # Tokenize the input prompt and move the tensors to the same device as the model.
    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Generate text from the prompt.
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=args.max_length, use_cache=True)

    # Decode and print the generated tokens.
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated text:", generated_text)

if __name__ == "__main__":
    main()