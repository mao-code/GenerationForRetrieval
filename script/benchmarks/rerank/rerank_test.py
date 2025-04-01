import os
import argparse
import logging
import torch
from tqdm import tqdm
import time
from tabulate import tabulate  # For pretty-printing the comparison table
import csv

# Import BEIR and BM25 utilities
from script.utils import load_dataset, beir_evaluate, beir_evaluate_custom

# Import Pyserini for retrieval.
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher

# Import the tokenizer and the reranker model for GFR.
from transformers import LlamaTokenizer
from GFR.modeling_GFR import GFRForSequenceScoring

# Import CrossEncoder for standard models.
from sentence_transformers import CrossEncoder

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Test script for reranking using BM25 + various reranker models")
    parser.add_argument("--dataset", type=str, default="msmarco", help="Dataset to use for testing (e.g., msmarco)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (e.g., test)")
    parser.add_argument("--models", type=str, nargs='+', required=True, 
                        help="List of models to test, in the form 'type:checkpoint' (e.g., 'gfr:/path/to/gfr', 'standard:cross-encoder/ms-marco-MiniLM-L-12-v2')")
    parser.add_argument("--log_file", type=str, default="test_results.log", help="File to log the evaluation results")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for reranking")
    parser.add_argument("--top_k", type=int, default=100, help="Number of top BM25 results to retrieve per query")
    parser.add_argument("--k_values", type=int, nargs='+', default=[1, 3, 5, 10],
                        help="List of k values for computing evaluation metrics (e.g., NDCG, MAP)")
    parser.add_argument("--retrieval_type", type=str, default="sparse", choices=["sparse", "dense"], 
                        help="Type of retrieval to use")
    parser.add_argument("--index_name", type=str, default=None, 
                        help="Specific index name to use; if None, use default based on retrieval_type")
    args = parser.parse_args()

    # Set up logging to both console and file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.log_file, mode="w")
        ],
        force=True
    )
    logger = logging.getLogger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load the dataset
    logger.info(f"Loading dataset '{args.dataset}' with split '{args.split}'...")
    corpus, queries, qrels = load_dataset(args.dataset, split=args.split)

    # Initialize the searcher based on retrieval type
    if args.index_name is not None:
        index_name = args.index_name
    elif args.retrieval_type == "sparse":
        index_name = 'msmarco-v1-passage'  # Default sparse index for MS MARCO
    elif args.retrieval_type == "dense":
        index_name = 'msmarco-v1-passage.tct_colbert-v2-hnp'  # Default dense index for MS MARCO
    else:
        raise ValueError("Invalid retrieval_type")
    
    if args.retrieval_type == "sparse":
        searcher = LuceneSearcher.from_prebuilt_index(index_name)
    elif args.retrieval_type == "dense":
        searcher = FaissSearcher.from_prebuilt_index(index_name)
    else:
        raise ValueError("Invalid retrieval_type")

    # List to store evaluation results for all models
    all_model_results = []

    # Process each model specified in --models
    for model_spec in args.models:
        model_type, model_checkpoint = model_spec.split(":", 1)
        model_id = f"{model_type}_{model_checkpoint.replace('/', '_')}"

        logger.info(f"Evaluating model: {model_id}")

        # Load the appropriate model based on type
        if model_type == "gfr":
            logger.info("Loading tokenizer for GFR...")
            tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if tokenizer.sep_token is None:
                tokenizer.add_special_tokens({"sep_token": "[SEP]"})
            if "[SCORE]" not in tokenizer.get_vocab():
                tokenizer.add_special_tokens({"additional_special_tokens": ["[SCORE]"]})

            logger.info("Loading GFR reranker model...")
            model = GFRForSequenceScoring.from_pretrained(model_checkpoint)
            model.to(device)
            model.eval()
        elif model_type == "standard":
            logger.info(f"Loading standard CrossEncoder model: {model_checkpoint}")
            model = CrossEncoder(
                model_checkpoint, 
                device=device, 
                automodel_args={
                    "torch_dtype": "auto",
                    "attn_implementation": "eager"  # Explicitly disable Flash Attention
                }, 
                trust_remote_code=True
            )
        else:
            raise ValueError(f"Invalid model type: {model_type}. Must be 'gfr' or 'standard'.")

        # Dictionary to hold reranking results
        reranked_results = {}

        # Timing variables
        total_inference_time = 0.0
        total_docs_processed = 0

        # Hit rate for different retrieval types
        total_hits_rate = 0.0

        logger.info(f"Using {args.retrieval_type} to retrieve top documents and reranking...")
        for qid, query_text in tqdm(queries.items(), desc="Processing queries"):
            # Retrieve top documents
            hits = searcher.search(query_text, k=args.top_k)
            candidate_doc_ids = [hit.docid for hit in hits]
            candidate_docs = [corpus[doc_id]['text'] for doc_id in candidate_doc_ids]

            relevant_doc_ids = list(qrels.get(qid, {}).keys())
            common_docs = set(relevant_doc_ids).intersection(set(candidate_doc_ids))
            hits_rate = len(common_docs) / len(relevant_doc_ids) if len(relevant_doc_ids) > 0 else 0
            total_hits_rate += hits_rate

            if model_type == "gfr":
                # GFR reranking with batch processing
                scores = []
                for i in range(0, len(candidate_docs), args.batch_size):
                    batch_docs = candidate_docs[i:i + args.batch_size]
                    batch_queries = [query_text] * len(batch_docs)
                    input_ids, token_type_ids, attention_mask = model.prepare_input(batch_docs, batch_queries, tokenizer)
                    input_ids = input_ids.to(device)
                    token_type_ids = token_type_ids.to(device)
                    attention_mask = attention_mask.to(device)

                    start_time = time.time()
                    with torch.no_grad():
                        output = model(
                            input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            return_dict=True
                        )
                    elapsed = time.time() - start_time
                    total_inference_time += elapsed
                    total_docs_processed += len(batch_docs)

                    batch_scores = output["logits"].squeeze(-1).tolist()
                    if isinstance(batch_scores, float):
                        batch_scores = [batch_scores]
                    scores.extend(batch_scores)
            elif model_type == "standard":
                # Standard CrossEncoder reranking
                pairs = [(query_text, doc) for doc in candidate_docs]
                start_time = time.time()
                scores = model.predict(pairs, batch_size=args.batch_size)
                elapsed = time.time() - start_time
                total_inference_time += elapsed
                total_docs_processed += len(candidate_docs)
                if not isinstance(scores, list):
                    scores = scores.tolist()

            # Map document IDs to scores
            reranked_results[qid] = {doc_id: score for doc_id, score in zip(candidate_doc_ids, scores)}

        # Evaluate reranked results
        logger.info("Evaluating reranked results...")
        ndcg, _map, recall, precision = beir_evaluate(qrels, reranked_results, args.k_values, ignore_identical_ids=True)
        mrr = beir_evaluate_custom(qrels, reranked_results, args.k_values, metric="mrr")
        top_k_accuracy = beir_evaluate_custom(qrels, reranked_results, args.k_values, metric="top_k_accuracy")

        # Calculate performance metrics
        avg_inference_time_ms = (total_inference_time / total_docs_processed) * 1000 if total_docs_processed > 0 else 0
        throughput_docs_per_sec = total_docs_processed / total_inference_time if total_inference_time > 0 else 0

        avg_hits_rate = total_hits_rate / len(queries)

        # Store results
        model_results = {
            "model_id": model_id,
            "ndcg": ndcg,
            "map": _map,
            "recall": recall,
            "precision": precision,
            "mrr": mrr,
            "top_k_accuracy": top_k_accuracy,
            "avg_inference_time_ms": avg_inference_time_ms,
            "throughput_docs_per_sec": throughput_docs_per_sec
        }
        all_model_results.append(model_results)

        # Log individual model results
        logger.info(f"Evaluation Metrics for {model_id}:")
        logging.info(f"Average Hits Rate for {args.retrieval_type} retriever: {avg_hits_rate}")
        logger.info(f"NDCG: {ndcg}")
        logger.info(f"MAP: {_map}")
        logger.info(f"Recall: {recall}")
        logger.info(f"Precision: {precision}")
        logger.info(f"MRR(dummy now): {mrr}")
        logger.info(f"Top_K_Accuracy(dummy now): {top_k_accuracy}")
        logger.info(f"Avg Inference Time (ms): {avg_inference_time_ms:.2f}")
        logger.info(f"Throughput (docs/sec): {throughput_docs_per_sec:.2f}")

    # Log comparison table for all models
    logger.info("Comparison of all models:")
    comparison_table = []
    for result in all_model_results:
        # We only use k=10 for comparison
        row = [
            result["model_id"],
            result["ndcg"].get("NDCG@10", "-"),
            result["map"].get("MAP@10", "-"),
            result["recall"].get("Recall@10", "-"),
            result["precision"].get("P@10", "-"),
            result["mrr"].get("MRR@10", "-"),
            result["top_k_accuracy"].get("Top_K_Accuracy@10", "-"),
            result["avg_inference_time_ms"],
            result["throughput_docs_per_sec"]
        ]
        comparison_table.append(row)

    headers = ["Model", "NDCG@10", "MAP@10", "Recall@10", "Precision@10", "MRR@10", "Top_K_Accuracy@10", "Avg Inference Time (ms)", "Throughput (docs/sec)"]
    logger.info("\n" + tabulate(comparison_table, headers=headers, tablefmt="grid"))

    # --- Save the comparison table to a CSV file ---
    csv_file = "rerank_comparison_table_gfr.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(comparison_table)
    logger.info(f"Comparison table saved to {csv_file}")

if __name__ == "__main__":
    main()

    """
    Example usage:
    python -m script.benchmarks.rerank.rerank_test \
    --dataset msmarco \
    --split test \
    --models gfr:./gfr_finetune_final_200m_msmarco standard:cross-encoder/ms-marco-MiniLM-L-12-v2 standard:mixedbread-ai/mxbai-rerank-large-v1 standard:jinaai/jina-reranker-v2-base-multilingual standard:BAAI/bge-reranker-v2-m3 \
    --log_file rerank_results.log \
    --batch_size 16 \
    --top_k 100 \
    --k_values 10 \
    --retrieval_type sparse
    """