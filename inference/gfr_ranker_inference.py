import torch
from transformers import LlamaTokenizer
from GFR.modeling_GFR import GFRForSequenceScoring

def main():
    # Path to the saved GFRForSequenceScoring model checkpoint.
    checkpoint_path = "path/to/your/model/checkpoint"

    tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({"sep_token": "[SEP]"})
    if "[SCORE]" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["[SCORE]"]})

    # Load the pre-trained sequence scoring model.
    model = GFRForSequenceScoring.from_pretrained(checkpoint_path)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    # Define example documents and queries.
    documents = [
        "Deep learning has revolutionized many industries, including computer vision and natural language processing.",
        "The stock market witnessed unprecedented growth last year due to various economic factors."
    ]
    queries = [
        "How has deep learning impacted computer vision?",
        "What factors contributed to the stock market growth?"
    ]

    # Use the model's prepare_input function to tokenize and format inputs.
    input_ids, token_type_ids, attention_mask = model.prepare_input(
        documents, 
        queries, 
        tokenizer, 
        max_length=1024 # Maximum sequence length for the model. Please refer to the training settings.
    )

    input_ids = input_ids.to(device)
    token_type_ids = token_type_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

    logits = outputs["logits"]  # shape: (batch_size, 1)
    relevancy_scores = logits.squeeze(-1)  # shape: (batch_size,)

    for doc, query, score in zip(documents, queries, relevancy_scores):
        print("Document: ", doc)
        print("Query:    ", query)
        print("Relevancy Score: {:.4f}".format(score.item()))
        print("-" * 50)

if __name__ == "__main__":
    main()