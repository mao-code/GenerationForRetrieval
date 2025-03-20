import os
import argparse
import logging
import torch
from tqdm import tqdm

# Import BEIR and BM25 utilities.
from script.utils import load_dataset, build_bm25_index, search_bm25, beir_evaluate

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
    args = parser.parse_args()

    # Set up logging to write both to console and to a file.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.log_file, mode="w")
        ]
    )
    logger = logging.getLogger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load the testing dataset.
    logger.info(f"Loading dataset '{args.dataset}' with split '{args.split}'...")
    corpus, queries, qrels = load_dataset(args.dataset, split=args.split)

    # 2. Build BM25 index on the test corpus.
    logger.info("Building BM25 index on test corpus...")
    retriever, doc_ids = build_bm25_index(corpus)

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

    logger.info("Retrieving BM25 top documents and reranking...")
    # For each query, retrieve BM25 candidates and then rerank using the reranker model.
    for qid, query_text in tqdm(queries.items(), desc="Processing queries"):
        # Retrieve top BM25 results (list of tuples: (doc_id, bm25_score)).
        docs, scores = search_bm25(query_text, retriever, doc_ids, top_k=args.top_k)
        candidate_doc_ids = [doc_id for doc_id in docs]

        # Debug: Print relevant document IDs from qrels for this query.
        relevant_doc_ids = list(qrels.get(qid, {}).keys())
        logger.info("Query ID: %s", qid)
        logger.info("Relevant document IDs from qrels: %s", relevant_doc_ids)
        logger.info("BM25 candidate document IDs: %s", candidate_doc_ids)
        common_docs = set(relevant_doc_ids).intersection(set(candidate_doc_ids))
        logger.info("Relevant docs present in BM25 candidates: %s", list(common_docs))

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
            with torch.no_grad():
                output = model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )

                print("Logits sample:", output["logits"][:3])
            # Get scores from the model's logits.
            batch_scores = output["logits"].squeeze(-1).tolist()
            # If batch size is 1, ensure batch_scores is a list.
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

    # 4. Log the results.
    logger.info("Final Evaluation Metrics:")
    logger.info(f"NDCG: {ndcg}")
    logger.info(f"MAP: {_map}")
    logger.info(f"Recall: {recall}")
    logger.info(f"Precision: {precision}")

if __name__ == "__main__":
    main()

    """
    python -m script.benchmarks.rerank.gfr \
    --dataset msmarco \
    --split test \
    --model_checkpoint ./gfr_finetune_final_200m_msmarco \
    --log_file gfr_200m_msmarco_test_results.log
    """