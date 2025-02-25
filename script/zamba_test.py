import argparse
import json
import logging
import os
import pathlib

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

# Import your custom ZambaEvaluator
from zamba.zamba_evaluator import ZambaEvaluator

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BEIR dataset using retrieval + Zamba re-ranking")
    parser.add_argument(
        '--dataset',
        type=str,
        default='nq',
        choices=['nq', 'hotpotqa', 'fiqa'],
        help='Dataset to evaluate (options: nq, hotpotqa, fiqa)'
    )
    parser.add_argument('--model', type=str, default='Zyphra/Zamba-7B-v1', help='Zamba model to evaluate')
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
    parser.add_argument(
        '--top_k_retrieval',
        type=int,
        default=100,
        help='Number of documents to retrieve per query in the first stage.'
    )
    parser.add_argument(
        '--top_k_rerank',
        type=int,
        default=50,
        help='Number of top documents (from retrieval) to re-rank with Zamba.'
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

    # Load BEIR dataset (corpus, queries, qrels)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    if args.num_test_samples > 0:
        queries = dict(list(queries.items())[:args.num_test_samples])
        qrels = dict(list(qrels.items())[:args.num_test_samples])

    # === Stage 1: Initial Retrieval using Snowflake Sentence Embeddings ===
    # Initialize the embedding model (using the Snowflake model)
    embed_model = models.SentenceBERT("Snowflake/snowflake-arctic-embed-l")
    retriever = DRES(embed_model, batch_size=256)
    top_k_retrieval = args.top_k_retrieval  # e.g., retrieve top 100 docs per query

    logging.info("Performing initial retrieval using Snowflake embeddings...")
    retrieval_results = retriever.search(corpus, queries, top_k_retrieval, score_function="cos_sim")

    # === Stage 2: Re-ranking with ZambaEvaluator ===
    # Initialize ZambaEvaluator (the re-ranker)
    zamba = ZambaEvaluator(model_name=args.model, score_type=args.score_type)

    # For each query, re-rank only the top_k_rerank retrieved documents using Zamba
    reranked_results = {}
    top_k_rerank = args.top_k_rerank
    for qid, doc_scores in retrieval_results.items():
        reranked_results[qid] = {}
        # Sort the documents by their retrieval scores and take the top_k_rerank documents
        sorted_doc_ids = [
            doc_id for doc_id, _ in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top_k_rerank]
        ]
        for doc_id in sorted_doc_ids:
            # Combine title and text for a more complete document context.
            doc = corpus[doc_id]
            document_text = f"{doc.get('title', '')}\n{doc.get('text', '')}"
            # Ask Zamba to score the document for the given query.
            score, answer, yes_logit = zamba.ask_zamba(queries[qid], document_text)
            # Here we use the logit for "yes" as a continuous re-ranking score.
            reranked_results[qid][doc_id] = yes_logit
            logging.info(f"Query {qid} | Doc {doc_id} | Zamba score: {yes_logit:.4f} | Answer: {answer}")

    # === Stage 3: Evaluation ===
    k_values = [1, 3, 5, 10, 100]
    metrics = zamba.evaluate_metrics(qrels, reranked_results, k_values=k_values)

    # Save the re-ranked results and evaluation metrics.
    results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{dataset}_zamba_reranked_results.json")
    metrics_file = os.path.join(results_dir, f"{dataset}_zamba_reranked_metrics.json")

    with open(results_file, "w") as f:
        json.dump(reranked_results, f, indent=4)
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Re-ranked results saved to {results_file}")
    print(f"Evaluation metrics saved to {metrics_file}")
    print("Evaluation Metrics:")
    for metric, score in metrics.items():
        print(f"{metric}: {score}")

if __name__ == '__main__':
    main()

    """
    python -m script.zamba_test \
    --dataset fiqa \
    --model Zyphra/Zamba-7B-v1 \
    --score_type binary \
    --num_test_samples 100 \
    --top_k_retrieval 100 \
    --top_k_rerank 100
    """