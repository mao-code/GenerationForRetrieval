from Mamba2_CDR.configuration_mamba2 import Mamba2Config
from Mamba2_CDR.modeling_mamba2 import Mamba2ForCausalLM
from Mamba2_CDR.tokenizer_utils import get_tokenizer
import torch

def main():
    tokenizer = get_tokenizer()
    config = Mamba2Config(
        num_heads=128,
        head_dim=64,
        vocab_size=len(tokenizer),
        hidden_size=4096,
        state_size=128,
        num_hidden_layers=4,

        expand=2,
        conv_kernel=4,
        n_groups=8,
        chunk_size=256
    )
    mamba2 = Mamba2ForCausalLM(config).to('cuda')
    num_params = sum(p.numel() for p in mamba2.parameters())
    print(f"Number of parameters: {num_params}")

    # Test the model
    mamba2.eval()
    with torch.no_grad():
        input_ids = tokenizer("The meaning of life is", return_tensors="pt").input_ids.to('cuda')
        model_output = mamba2(input_ids=input_ids, return_dict=True)
        cache = model_output.cache_params
        print("Cache shape:", cache.shape)

        # Generate a sequence from the model.
        outputs = mamba2.generate(input_ids=input_ids, max_length=50)
        print("Generated sequence:", tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()