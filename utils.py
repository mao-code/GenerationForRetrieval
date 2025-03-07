import os
import random
import pathlib
import time
import logging

import numpy as np
import torch

# BEIR API imports.
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.custom_metrics import hole, mrr, recall_cap, top_k_accuracy

# For progress bar
from tqdm import tqdm

import wandb

import pytrec_eval

# Setup logging format.
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

def load_dataset(dataset: str, split: str):
    """
    Download and load a dataset (msmarco or hotpotqa) using BEIR's GenericDataLoader.
    """

    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = os.path.join(out_dir, dataset)
    if not os.path.exists(data_path):
        print(f"Dataset '{dataset}' not found locally. Downloading...")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        data_path = util.download_and_unzip(url, out_dir)
    else:
        print(f"Dataset '{dataset}' found locally. Skipping download.")

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    
    return corpus, queries, qrels

def prepare_training_samples(corpus: dict, queries: dict, qrels: dict):
    """
    For each query, create a training sample consisting of:
      - The query text.
      - A positive document (with a non-zero relevance score in qrels).
      - A negative document (sampled randomly from corpus that is not relevant).
    """
    training_samples = []
    all_doc_ids = list(corpus.keys())
    for qid, rel_docs in qrels.items():
        if qid not in queries:
            continue
        query_text = queries[qid]
        pos_doc_ids = [doc_id for doc_id, score in rel_docs.items() if score > 0]
        if not pos_doc_ids:
            continue
        for pos_doc_id in pos_doc_ids:
            pos_doc_text = corpus[pos_doc_id]['text']
            # Randomly sample a negative document (ensure it is not in the relevant docs)
            neg_doc_id = random.choice(all_doc_ids)
            while neg_doc_id in rel_docs:
                neg_doc_id = random.choice(all_doc_ids)
            neg_doc_text = corpus[neg_doc_id]['text']
            training_samples.append((query_text, pos_doc_text, neg_doc_text))
    return training_samples

def train_one_epoch(model, optimizer, loss_fn, training_samples, tokenizer, device, batch_size, epoch):
    """
    Fully vectorized training loop over the training samples.
    """
    base_model = model.module if hasattr(model, "module") else model

    model.train()
    total_loss = 0.0
    random.shuffle(training_samples)
    num_batches = len(training_samples) // batch_size + int(len(training_samples) % batch_size > 0)
    
    # Set up tqdm progress bar.
    pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}", unit="batch")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        batch = training_samples[start_idx : start_idx + batch_size]
        optimizer.zero_grad()
        
        # Create batched lists for queries, positive and negative documents.
        batch_queries = [query for query, pos_doc, neg_doc in batch]
        batch_pos_docs = [pos_doc for query, pos_doc, neg_doc in batch]
        batch_neg_docs = [neg_doc for query, pos_doc, neg_doc in batch]
        
        # Prepare batched inputs.
        input_ids_pos, token_type_ids_pos = base_model.prepare_input(batch_pos_docs, batch_queries, tokenizer)
        input_ids_neg, token_type_ids_neg = base_model.prepare_input(batch_neg_docs, batch_queries, tokenizer)
        
        # Transfer batched inputs to device.
        input_ids_pos = input_ids_pos.to(device)
        token_type_ids_pos = token_type_ids_pos.to(device)
        input_ids_neg = input_ids_neg.to(device)
        token_type_ids_neg = token_type_ids_neg.to(device)
        
        # Forward pass in batch.
        score_pos, _ = model(input_ids=input_ids_pos, token_type_ids=token_type_ids_pos, return_dict=True)
        score_neg, _ = model(input_ids=input_ids_neg, token_type_ids=token_type_ids_neg, return_dict=True)
        
        # Create target tensor for MarginRankingLoss.
        target = torch.ones(score_pos.size(), device=device)
        loss = loss_fn(score_pos.view(-1), score_neg.view(-1), target.view(-1))
        
        # Backward pass and optimizer step.
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Log to WandB.
        wandb.log({
            "iteration_loss": loss.item(),
            "epoch": epoch+1,
            "batch": batch_idx+1,
            "percent_complete": 100.0 * (batch_idx+1) / num_batches
        })
        
        # Update progress bar with current loss.
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        pbar.update(1)
        
        # # Also log on-screen every 1000 batches.
        # if (batch_idx + 1) % 1000 == 0 or batch_idx + 1 == num_batches:
        #     logging.info(
        #         f"Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.4f}, "
        #         f"Complete: {100.0 * (batch_idx+1) / num_batches:.2f}%"
        #     )
    
    pbar.close()
    avg_loss = total_loss / num_batches
    return avg_loss


def beir_evaluate(qrels: dict, results: dict, k_values: list, ignore_identical_ids: bool = True):
    """
    BEIR-style evaluation using pytrec_eval.
    The qrels and results should follow BEIR format:
      - qrels: { query_id: { doc_id: relevance, ... }, ... }
      - results: { query_id: { doc_id: score, ... }, ... }
    """
    if ignore_identical_ids:
        # Remove cases where query id equals document id.
        for qid, rels in results.items():
            for pid in list(rels.keys()):
                if qid == pid:
                    results[qid].pop(pid)

    # Initialize metric dictionaries.
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

    # Average metrics over all queries.
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

def beir_evaluate_custom(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int],
        metric: str,
    ) -> tuple[dict[str, float]]:
        if metric.lower() in ["mrr", "mrr@k", "mrr_cut"]:
            return mrr(qrels, results, k_values)

        elif metric.lower() in ["recall_cap", "r_cap", "r_cap@k"]:
            return recall_cap(qrels, results, k_values)

        elif metric.lower() in ["hole", "hole@k"]:
            return hole(qrels, results, k_values)

        elif metric.lower() in [
            "acc",
            "top_k_acc",
            "accuracy",
            "accuracy@k",
            "top_k_accuracy",
        ]:
            return top_k_accuracy(qrels, results, k_values)

def evaluate_full_retrieval(model, corpus: dict, queries: dict, qrels: dict, tokenizer, device, batch_size=32, k_values=[1, 3, 5, 10, 100]):
    """
    Full retrieval evaluation similar to BEIR's implementation using batched processing.
    
    For each query:
      - Iterate over the entire corpus in batches.
      - For each batch, use model.prepare_input with a list of document texts and a repeated query.
      - Compute scores with the model in a single forward pass.
      - Build a results dict in the format: { query_id: { doc_id: score, ... }, ... }
    
    Also computes:
      - Avg_Inference_Time_ms: Average time (in ms) per document.
      - Throughput_docs_per_sec: Number of documents processed per second.
    """
    base_model = model.module if hasattr(model, "module") else model

    model.eval()
    results = {}  # To store scores: { query_id: { doc_id: score, ... } }
    total_inference_time = 0.0
    total_docs_processed = 0

    # Process each query.
    for query_id, query in tqdm(queries.items(), desc="Evaluating queries"):
        results[query_id] = {}
        doc_ids = list(corpus.keys())
        # Process corpus in batches.
        for i in range(0, len(doc_ids), batch_size):
            batch_doc_ids = doc_ids[i:i+batch_size]
            batch_docs = [corpus[doc_id]['text'] for doc_id in batch_doc_ids]

            # Prepare batched input tensors for the entire batch of documents paired with the query.
            # The vectorized prepare_input now accepts a list of documents and a list of queries.
            batch_input_ids, batch_token_type_ids = base_model.prepare_input(batch_docs, [query] * len(batch_docs), tokenizer)
            batch_input_ids = batch_input_ids.to(device)
            batch_token_type_ids = batch_token_type_ids.to(device)

            start_time = time.time()
            with torch.no_grad():
                scores, _ = model(
                    input_ids=batch_input_ids,
                    token_type_ids=batch_token_type_ids,
                    return_dict=True
                )
            elapsed = time.time() - start_time
            total_inference_time += elapsed
            total_docs_processed += len(batch_doc_ids)

            # The model returns scores of shape (batch, 1); squeeze them to a list.
            batch_scores = scores.squeeze(-1).tolist()
            for j, doc_id in enumerate(batch_doc_ids):
                results[query_id][doc_id] = batch_scores[j]

    # Compute inference time metrics.
    avg_inference_time_ms = (total_inference_time / total_docs_processed) * 1000  
    throughput_docs_per_sec = total_docs_processed / total_inference_time

    # Evaluate retrieval performance using pytrec_eval.
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