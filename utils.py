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

# For progress bar
from tqdm import tqdm

import wandb

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
    Training loop over the training samples.
    Uses a progress bar to display batch loss and percentage complete.
    Logs detailed progress on-screen and to WandB.
    """
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
        losses = []
        
        for query, pos_doc, neg_doc in batch:
            # Prepare inputs for positive and negative pairs.
            input_ids_pos, token_type_ids_pos = model.prepare_input(pos_doc, query, tokenizer)
            input_ids_neg, token_type_ids_neg = model.prepare_input(neg_doc, query, tokenizer)
            input_ids_pos = input_ids_pos.to(device)
            token_type_ids_pos = token_type_ids_pos.to(device)
            input_ids_neg = input_ids_neg.to(device)
            token_type_ids_neg = token_type_ids_neg.to(device)
            
            # Forward pass: get relevance scores.
            score_pos, _ = model(input_ids=input_ids_pos, token_type_ids=token_type_ids_pos, return_dict=True)
            score_neg, _ = model(input_ids=input_ids_neg, token_type_ids=token_type_ids_neg, return_dict=True)
            
            # MarginRankingLoss: loss = max(0, margin - (s_pos - s_neg)).
            target = torch.ones(score_pos.size()).to(device)
            loss = loss_fn(score_pos.view(-1), score_neg.view(-1), target.view(-1))
            losses.append(loss)
        
        if losses:
            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            
            # Log to WandB.
            wandb.log({
                "iteration_loss": batch_loss.item(),
                "epoch": epoch+1,
                "batch": batch_idx+1,
                "percent_complete": 100.0 * (batch_idx+1) / num_batches
            })
            
            # Update progress bar with current loss.
            pbar.set_postfix(loss=f"{batch_loss.item():.4f}")
            pbar.update(1)
            
            # Also log on-screen every 100 batches.
            if (batch_idx + 1) % 100 == 0 or batch_idx + 1 == num_batches:
                logging.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches}, Loss: {batch_loss.item():.4f}, Complete: {100.0 * (batch_idx+1) / num_batches:.2f}%")
    
    pbar.close()
    avg_loss = total_loss / num_batches
    return avg_loss

def evaluate(model, samples, tokenizer, device, eval_batch_size):
    """
    Evaluate ranking quality on a small set of evaluation samples.
    Uses a progress bar to display evaluation progress.
    Logs detailed evaluation progress to WandB and the console.
    """
    model.eval()
    ndcg_list = []
    mrr_list = []
    precision_list = []
    recall_list = []
    inference_times = []
    
    pbar = tqdm(total=eval_batch_size, desc="Evaluating", unit="sample")
    
    for idx, (query, pos_doc, neg_doc) in enumerate(samples):
        if idx >= eval_batch_size:
            break
        start_time = time.time()
        input_ids_pos, token_type_ids_pos = model.prepare_input(pos_doc, query, tokenizer)
        input_ids_neg, token_type_ids_neg = model.prepare_input(neg_doc, query, tokenizer)
        input_ids_pos = input_ids_pos.to(device)
        token_type_ids_pos = token_type_ids_pos.to(device)
        input_ids_neg = input_ids_neg.to(device)
        token_type_ids_neg = token_type_ids_neg.to(device)
        
        with torch.no_grad():
            score_pos, _ = model(input_ids=input_ids_pos, token_type_ids=token_type_ids_pos, return_dict=True)
            score_neg, _ = model(input_ids=input_ids_neg, token_type_ids=token_type_ids_neg, return_dict=True)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        scores = [score_pos.item(), score_neg.item()]
        sorted_idx = np.argsort(scores)[::-1]  # descending order
        
        # Assume the positive document should rank first.
        ndcg = 1.0 if sorted_idx[0] == 0 else 0.0
        mrr = 1.0 if sorted_idx[0] == 0 else 0.0
        precision = 1.0 if sorted_idx[0] == 0 else 0.0
        recall = 1.0 if sorted_idx[0] == 0 else 0.0
        
        ndcg_list.append(ndcg)
        mrr_list.append(mrr)
        precision_list.append(precision)
        recall_list.append(recall)
        
        wandb.log({
            "eval_sample": idx+1,
            "ndcg": ndcg,
            "mrr": mrr,
            "precision": precision,
            "recall": recall,
            "inference_time": inference_time
        })
        
        pbar.update(1)
        if (idx + 1) % 5 == 0 or idx + 1 == eval_batch_size:
            logging.info(f"Evaluated {idx+1}/{eval_batch_size} samples")
    
    pbar.close()
    avg_ndcg = np.mean(ndcg_list)
    avg_mrr = np.mean(mrr_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_inference_time = np.mean(inference_times)
    throughput = (2 * eval_batch_size) / sum(inference_times)  # approximate queries per second

    metrics = {
        "NDCG": avg_ndcg,
        "MRR": avg_mrr,
        "Precision": avg_precision,
        "Recall": avg_recall,
        "Avg_Inference_Time": avg_inference_time,
        "Throughput": throughput,
    }
    return metrics