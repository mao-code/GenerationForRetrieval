import time
import torch
from tqdm import tqdm
import pytrec_eval

def beir_evaluate(qrels: dict, results: dict, k_values: list, ignore_identical_ids: bool = True):
    """Evaluates ranking results using BEIR's pytrec_eval."""
    if ignore_identical_ids:
        # For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this."
        for qid, rels in results.items():
            for pid in list(rels.keys()):
                if qid == pid:
                    results[qid].pop(pid)
    
    ndcg = {f"NDCG@{k}": 0.0 for k in k_values}
    _map = {f"MAP@{k}": 0.0 for k in k_values}
    recall = {f"Recall@{k}": 0.0 for k in k_values}
    precision = {f"P@{k}": 0.0 for k in k_values}
    
    map_string = "map_cut." + ",".join(str(k) for k in k_values)
    ndcg_string = "ndcg_cut." + ",".join(str(k) for k in k_values)
    recall_string = "recall." + ",".join(str(k) for k in k_values)
    precision_string = "P." + ",".join(str(k) for k in k_values)
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
    scores = evaluator.evaluate(results)
    
    for query_id, query_scores in scores.items():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += query_scores[f"ndcg_cut_{k}"]
            _map[f"MAP@{k}"] += query_scores[f"map_cut_{k}"]
            recall[f"Recall@{k}"] += query_scores[f"recall_{k}"]
            precision[f"P@{k}"] += query_scores[f"P_{k}"]
    num_queries = len(scores)
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / num_queries, 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / num_queries, 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / num_queries, 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / num_queries, 5)
    return ndcg, _map, recall, precision

def beir_evaluate_custom(qrels: dict, results: dict, k_values: list, metric: str):
    """
    Computes custom evaluation metrics.
    
    For MRR@K:
      - For each query, it finds the first relevant document (with a relevance score > 0)
        within the top K retrieved documents and computes the reciprocal rank (1/rank).
      - The mean reciprocal rank over all queries is returned for each K.
    
    For Top-K Accuracy:
      - For each query, it checks whether at least one relevant document is present
        in the top K retrieved documents.
      - The fraction of queries with at least one relevant document in the top K is returned.
    """
    metric = metric.lower()
    scores = {}
    
    if metric in ["mrr", "mrr@k", "mrr_cut"]:
        # Initialize scores for each k value.
        for k in k_values:
            scores[f"MRR@{k}"] = 0.0
        num_queries = 0
        for qid in qrels:
            if qid not in results:
                continue
            num_queries += 1
            # Sort retrieved documents by score (descending)
            ranked_docs = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
            for k in k_values:
                mrr = 0.0
                # Look only at the top k retrieved documents.
                for rank, (doc_id, score) in enumerate(ranked_docs[:k], start=1):
                    if doc_id in qrels[qid] and qrels[qid][doc_id] > 0:
                        mrr = 1.0 / rank
                        break
                scores[f"MRR@{k}"] += mrr
        # Average the reciprocal ranks over all queries.
        for k in k_values:
            scores[f"MRR@{k}"] = round(scores[f"MRR@{k}"] / num_queries, 5) if num_queries > 0 else 0.0
        return scores

    elif metric in ["top_k_accuracy", "acc", "accuracy"]:
        for k in k_values:
            scores[f"Top_K_Accuracy@{k}"] = 0.0
        num_queries = 0
        for qid in qrels:
            if qid not in results:
                continue
            num_queries += 1
            ranked_docs = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
            for k in k_values:
                hit = 0
                # Check if any document in the top k is relevant.
                for doc_id, score in ranked_docs[:k]:
                    if doc_id in qrels[qid] and qrels[qid][doc_id] > 0:
                        hit = 1
                        break
                scores[f"Top_K_Accuracy@{k}"] += hit
        for k in k_values:
            scores[f"Top_K_Accuracy@{k}"] = round(scores[f"Top_K_Accuracy@{k}"] / num_queries, 5) if num_queries > 0 else 0.0
        return scores
    return {}

def evaluate_full_retrieval(model, corpus: dict, queries: dict, qrels: dict,
                            tokenizer, device, batch_size=8, k_values=[1, 5, 10],
                            use_cache=False, candidate_cache=None):
    """
    For each query, scores all documents in the corpus using the loaded model.
    If use_cache is True, candidate documents are either taken from the provided
    candidate_cache or, if not provided, are cached on the fly (the cache building time
    is not added to the ranking time).
    
    Returns evaluation metrics (NDCG, MAP, Recall, Precision, MRR, Top_K_Accuracy,
    Avg_Inference_Time_ms, and Throughput_docs_per_sec).
    """
    model.eval()
    results = {}
    total_inference_time = 0.0
    total_docs_processed = 0

    for query_id, query in tqdm(queries.items(), desc="Evaluating queries"):
        results[query_id] = {}
        # Full re-rank: use all documents in the corpus.
        doc_ids = list(corpus.keys())
        for i in tqdm(range(0, len(doc_ids), batch_size),
                      desc=f"Scoring docs for query {query_id}", leave=False):
            batch_doc_ids = doc_ids[i : i + batch_size]
            batch_docs = [corpus[doc_id]['text'] for doc_id in batch_doc_ids]

            if use_cache:
                print("Using cache for scoring.")
                # # Use provided candidate cache if available
                # if candidate_cache is not None:
                #     candidate_kv_cache = {doc_id: candidate_cache[doc_id] for doc_id in batch_doc_ids}
                # else:
                #     # Build cache on the fly (cache building time not measured)
                #     candidate_doc_dict = {doc_id: corpus[doc_id]['text'] for doc_id in batch_doc_ids}
                #     candidate_kv_cache = get_documents_cache(model, candidate_doc_dict, tokenizer, device, batch_size=len(candidate_doc_dict))
                # # Measure only the scoring time with cached representations.
                # start_time = time.time()
                # with torch.no_grad():
                #     batch_scores = score_with_cache(
                #         model,
                #         candidate_kv_cache,
                #         query,
                #         tokenizer,
                #         device
                #     )
                # elapsed = time.time() - start_time
            else:
                # Non-cache version: prepare input tensors and score directly.
                batch_input_ids, batch_token_type_ids, batch_attention_mask = model.prepare_input(
                    batch_docs, [query] * len(batch_docs), tokenizer
                )
                batch_input_ids = batch_input_ids.to(device)
                batch_token_type_ids = batch_token_type_ids.to(device)
                batch_attention_mask = batch_attention_mask.to(device)
                start_time = time.time()
                with torch.no_grad():
                    output = model(
                        input_ids=batch_input_ids,
                        token_type_ids=batch_token_type_ids,
                        attention_mask=batch_attention_mask,
                        return_dict=True
                    )
                elapsed = time.time() - start_time
                batch_scores = output["logits"].squeeze(-1).tolist()
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
            
            total_inference_time += elapsed
            total_docs_processed += len(batch_doc_ids)
            for j, doc_id in enumerate(batch_doc_ids):
                results[query_id][doc_id] = batch_scores[j]

    avg_inference_time_ms = (total_inference_time / total_docs_processed) * 1000  
    throughput_docs_per_sec = total_docs_processed / total_inference_time

    ndcg, _map, recall, precision = beir_evaluate(qrels, results, k_values, ignore_identical_ids=True)
    mrr = beir_evaluate_custom(qrels, results, k_values, metric="mrr")
    top_k_accuracy = beir_evaluate_custom(qrels, results, k_values, metric="top_k_accuracy")

    metrics = {
        "NDCG": ndcg,
        "MAP": _map,
        "Recall": recall,
        "Precision": precision,
        "MRR": mrr,
        "Top_K_Accuracy": top_k_accuracy,
        "Avg_Inference_Time_ms": avg_inference_time_ms,
        "Throughput_docs_per_sec": throughput_docs_per_sec,
    }
    return metrics