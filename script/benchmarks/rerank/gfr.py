import os
import argparse
import logging
import torch
from tqdm import tqdm
import time

# Import BEIR and BM25 utilities.
from script.utils import load_dataset, beir_evaluate, beir_evaluate_custom # build_bm25_index, search_bm25

# Import Pyserini for retrieval.
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher

# Import the tokenizer and the reranker model.
from transformers import LlamaTokenizer
from GFR.modeling_GFR import GFRForSequenceScoring

def main():
    parser = argparse.ArgumentParser(description="Test script for reranking using BM25 + GFRForSequenceScoring")
    parser.add_argument("--dataset", type=str, default="msmarco", help="Dataset to use for testing (e.g., msmarco)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (e.g., test)")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the trained reranker model checkpoint")
    parser.add_argument("--log_file", type=str, default="test_results.log", help="File to log the evaluation results")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for reranking")
    parser.add_argument("--top_k", type=int, default=100, help="Number of top BM25 results to retrieve per query")
    parser.add_argument("--k_values", type=int, nargs='+', default=[1, 3, 5, 10],
                        help="List of k values for computing evaluation metrics (e.g., NDCG, MAP)")
    parser.add_argument("--retrieval_type", type=str, default="sparse", choices=["sparse", "dense"], help="Type of retrieval to use")
    parser.add_argument("--index_name", type=str, default=None, help="Specific index name to use; if None, use default based on retrieval_type")
    args = parser.parse_args()

    # Set up logging to write both to console and to a file.
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

    # Load the testing dataset.
    logger.info(f"Loading dataset '{args.dataset}' with split '{args.split}'...")
    corpus, queries, qrels = load_dataset(args.dataset, split=args.split)

    # 2. Initialize the searcher based on retrieval type.
    # For index names: https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md
    if args.index_name is not None:
        index_name = args.index_name
    elif args.retrieval_type == "sparse":
        index_name = 'msmarco-v1-passage'  # Default sparse index for MS MARCO passages
    elif args.retrieval_type == "dense":
        index_name = 'msmarco-v1-passage.tct_colbert-v2-hnp'  # Default dense index for MS MARCO passages
    else:
        raise ValueError("Invalid retrieval_type")
    
    if args.retrieval_type == "sparse":
        searcher = LuceneSearcher.from_prebuilt_index(index_name)
    elif args.retrieval_type == "dense":
        searcher = FaissSearcher.from_prebuilt_index(index_name)
    else:
        raise ValueError("Invalid retrieval_type")
    
    # Load the tokenizer.
    logger.info("Loading tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({"sep_token": "[SEP]"})
    if "[SCORE]" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["[SCORE]"]})

    # Load the reranker model.
    logger.info("Loading reranker model...")
    model = GFRForSequenceScoring.from_pretrained(args.model_checkpoint)
    model.to(device)
    model.eval()

    # Dictionary to hold the reranking results.
    # Each query id will map to a dict of {doc_id: reranker_score}.
    reranked_results = {}

    # Variables for measuring inference time.
    total_inference_time = 0.0
    total_docs_processed = 0

    logger.info("Retrieving BM25 top documents and reranking...")
    # For each query, retrieve BM25 candidates and then rerank using the reranker model.
    for qid, query_text in tqdm(queries.items(), desc="Processing queries"):
        # Retrieve top documents using Pyserini.
        hits = searcher.search(query_text, k=args.top_k)
        candidate_doc_ids = [hit.docid for hit in hits]

        # Debug: Print relevant document IDs from qrels for this query.
        relevant_doc_ids = list(qrels.get(qid, {}).keys())
        # logger.info("Query ID: %s", qid)
        # logger.info("Relevant document IDs from qrels: %s", relevant_doc_ids)
        # logger.info("Retrieved document IDs: %s", candidate_doc_ids)
        common_docs = set(relevant_doc_ids).intersection(set(candidate_doc_ids))
        # logger.info("Relevant docs present in retrieved candidates: %s", list(common_docs))
        logger.info("Percentage of relevant docs in retrieved candidates: %.2f%%", len(common_docs) / len(relevant_doc_ids) * 100)

        # Get the texts of the candidate documents from the corpus.
        candidate_docs = [corpus[doc_id]['text'] for doc_id in candidate_doc_ids]

        # Reranker scoring: process in batches.
        scores = []
        for i in tqdm(range(0, len(candidate_docs), args.batch_size), desc="Reranking documents", leave=False):
            batch_docs = candidate_docs[i: i + args.batch_size]
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

            # Get scores from the model's logits.
            batch_scores = output["logits"].squeeze(-1).tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend(batch_scores)

        # Map candidate document IDs to their reranker scores.
        doc_score_mapping = {doc_id: score for doc_id, score in zip(candidate_doc_ids, scores)}

        # Debug: Print ranked list from model (document ID: score) sorted by score descending.
        ranked_list = sorted(doc_score_mapping.items(), key=lambda x: x[1], reverse=True)
        logger.info("Ranked list from model (document ID: score):")
        for doc_id, score in ranked_list:
            logger.info("    %s: %f", doc_id, score)

        reranked_results[qid] = doc_score_mapping

    # 3. Evaluate the reranked results.
    logger.info("Evaluating reranked results...")
    ndcg, _map, recall, precision = beir_evaluate(qrels, reranked_results, args.k_values, ignore_identical_ids=True)
    mrr = beir_evaluate_custom(qrels, reranked_results, args.k_values, metric="mrr")
    top_k_accuracy = beir_evaluate_custom(qrels, reranked_results, args.k_values, metric="top_k_accuracy")

    # Calculate average inference time and throughput.
    avg_inference_time_ms = (total_inference_time / total_docs_processed) * 1000
    throughput_docs_per_sec = total_docs_processed / total_inference_time

    # 4. Log the results.
    logger.info("Final Evaluation Metrics:")
    logger.info(f"NDCG: {ndcg}")
    logger.info(f"MAP: {_map}")
    logger.info(f"Recall: {recall}")
    logger.info(f"Precision: {precision}")
    logger.info(f"MRR: {mrr}")
    logger.info(f"Top_K_Accuracy: {top_k_accuracy}")
    logger.info(f"Avg Inference Time (ms): {avg_inference_time_ms}")
    logger.info(f"Throughput (docs/sec): {throughput_docs_per_sec}")

if __name__ == "__main__":
    main()

    """
    Example usage:
    python -m script.benchmarks.rerank.gfr \
    --dataset msmarco \
    --split test \
    --model_checkpoint ./gfr_finetune_final_200m_msmarco \
    --log_file gfr_200m_msmarco_test_results.log \
    --retrieval_type sparse \
    --index_name msmarco-v1-passage
    """