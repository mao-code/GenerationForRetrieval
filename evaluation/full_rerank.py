import argparse
import logging
import torch
from transformers import LlamaTokenizer
from GFR.modeling_GFR import GFRForSequenceScoring
from utils import load_dataset
from script.utils import evaluate_full_retrieval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("gfr_test_results.log"),
        logging.StreamHandler()
    ]
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_path", type=str, required=True,
                        help="Path to the saved model checkpoint directory")
    parser.add_argument("--datasets", type=str, nargs='+', default=["msmarco"],
                        help="List of test datasets (e.g., msmarco hotpotqa nq)")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2,
                        help="Evaluation batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load tokenizer.
    tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({"sep_token": "[SEP]"})
    if "[SCORE]" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["[SCORE]"]})

    # Load the saved model.
    logging.info(f"Loading model from {args.saved_model_path}...")
    model = GFRForSequenceScoring.from_pretrained(args.saved_model_path)
    model.to(device)
    model.eval()

    # Evaluate on each test dataset.
    for dataset in args.datasets:
        logging.info(f"Loading dataset: {dataset} (test split)")
        corpus_test, queries_test, qrels_test = load_dataset(dataset, split="test")
        
        logging.info("Evaluating model on test data...")
        test_metrics = evaluate_full_retrieval(
            model, corpus_test, queries_test, qrels_test,
            tokenizer, device, batch_size=args.per_device_eval_batch_size
        )
        logging.info(f"Test metrics for {dataset}: {test_metrics}")


if __name__ == "__main__":
    main()

    """
    python -m script.benchmarks.full_rerank.gfr \
    --saved_model_path ./gfr_finetune_ckpts_200m_msmarco/checkpoint-1000 \
    --datasets msmarco \
    --per_device_eval_batch_size 64
    """