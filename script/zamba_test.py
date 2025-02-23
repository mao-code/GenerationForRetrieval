from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os

import argparse
import logging
from zamba.zamba_evaluator import ZambaEvaluator
import json

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BEIR dataset using Zamba LLM")
    parser.add_argument(
        '--dataset',
        type=str,
        default='nq',
        choices=['nq', 'hotpotqa', 'fiqa'],
        help='Dataset to evaluate (options: nq, hotpotqa, fiqa)'
    )
    parser.add_argument('--model', type=str, default='Zyphra/Zamba-7B-v1', help='Model to evaluate')
    parser.add_argument(
        '--score_type',
        type=str,
        default='binary',
        choices=['binary', 'ordered'],
        help="Scoring type: 'binary' (yes/no) or 'ordered' (high/mid/low)"
    )
    parser.add_argument(
        '--num_test_samples',
        type=int,
        default=100,
        help='Number of test samples to evaluate. -1 for all samples.'
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    dataset = args.dataset

    # Set up dataset directory
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = os.path.join(out_dir, dataset)

    if not os.path.exists(data_path):
        print(f"Dataset '{dataset}' not found locally. Downloading...")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        data_path = util.download_and_unzip(url, out_dir)
    else:
        print(f"Dataset '{dataset}' found locally. Skipping download.")

    # Load the BEIR dataset
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    
    # Initialize ZambaEvaluator with the chosen model and scoring mode
    evaluator = ZambaEvaluator(model_name=args.model, score_type=args.score_type)
    
    # Evaluate using Zamba
    results = evaluator.evaluate(corpus, queries, num_test_samples=args.num_test_samples)
    
    # Save the evaluation results to a file
    results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{dataset}_zamba_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_file}")

if __name__ == '__main__':
    main()

    """
    python -m script.zamba_test \
    --dataset nq \
    --model Zyphra/Zamba-7B-v1 \
    --score_type binary \
    --num_test_samples 100
    """