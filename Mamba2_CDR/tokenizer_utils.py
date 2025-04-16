from transformers import AutoTokenizer

def get_tokenizer():
    model_id = 'mistralai/Mamba-Codestral-7B-v0.1'
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # if tokenizer.sep_token is None:
    #     tokenizer.add_special_tokens({"sep_token": "[SEP]"})
    # if "[SCORE]" not in tokenizer.get_vocab():
    #     tokenizer.add_special_tokens({"additional_special_tokens": ["[SCORE]"]})

    return tokenizer