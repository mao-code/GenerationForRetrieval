import os
import argparse
import logging
import time
import random
from datetime import datetime
import pathlib
import bm25s

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from transformers import LlamaTokenizer
from GFR.configuration_GFR import GFRConfig
from GFR.modeling_GFR import GFRForCausalLM, GFRForSequenceScoring
import wandb

# BEIR imports.
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import pytrec_eval

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

retriever = bm25s.BM25()

#############################################
# Helper Functions
#############################################

def load_dataset(dataset: str, split: str):
    out_dir = "datasets"
    data_path = os.path.join(out_dir, dataset)
    if not os.path.exists(data_path):
        print(f"Dataset '{dataset}' not found locally. Downloading...")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        data_path = util.download_and_unzip(url, out_dir)
    else:
        print(f"Dataset '{dataset}' found locally. Skipping download.")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    
    # Prefix each doc_id in corpus, and update queries and qrels accordingly.
    new_corpus = {f"{dataset}_{doc_id}": content for doc_id, content in corpus.items()}
    new_queries = {f"{dataset}_{qid}": query for qid, query in queries.items()}
    new_qrels = {f"{dataset}_{qid}": {f"{dataset}_{doc_id}": score for doc_id, score in rels.items()}
                 for qid, rels in qrels.items()}
    
    return new_corpus, new_queries, new_qrels

def build_bm25_index(corpus: dict):
    """
    Builds a BM25S index using bm25s.
    Returns the BM25S object and a list of document IDs.
    """
    # Extract doc_ids and raw document texts.
    doc_ids = list(corpus.keys())
    raw_docs = [corpus[doc_id]['text'] for doc_id in doc_ids]
    
    logging.info("Tokenizing documents using BM25S tokenizer...")
    # bm25s.tokenize() expects a list of strings.
    tokenized_docs = bm25s.tokenize(raw_docs, stopwords="en")
    
    logging.info("Building BM25S index...")
    # Build the BM25S index using raw document texts.
    bm25_index = retriever.index(tokenized_docs)
    
    return bm25_index, doc_ids

def prepare_training_samples(corpus: dict, queries: dict, qrels: dict, hard_negative: bool = False, bm25_index=None, bm25_doc_ids=None):
    """
    Creates training sample tuples: (query_text, positive_doc_text, negative_doc_text).
    If hard_negative is True, uses the BM25S index to precompute hard negatives.
    """
    training_samples = []
    all_doc_ids = list(corpus.keys())
    
    # Precompute hard negatives using BM25S if enabled.
    hard_negatives = {}
    if hard_negative and bm25_index is not None and bm25_doc_ids is not None:
        for qid in tqdm(qrels, desc="Precomputing hard negatives"):
            query_text = queries[qid]
            tokenized_query = bm25s.tokenize(query_text)  # BM25S tokenizer
            # Retrieve top-5 candidate indices (BM25S.retrieve returns indices by default)
            # results: (doc ids, scores)
            results, _ = bm25_index.retrieve(tokenized_query, k=5)
            candidate_negatives = [id for id in results[0] if id not in qrels[qid]]
            if candidate_negatives:
                hard_negatives[qid] = candidate_negatives[0]  # Choose the top-ranked negative.
            else:
                # Fallback to a random negative if none of the top candidates are valid.
                neg_doc_id = random.choice(all_doc_ids)
                while neg_doc_id in qrels[qid]:
                    neg_doc_id = random.choice(all_doc_ids)
                hard_negatives[qid] = neg_doc_id
    
    # Process queries and create training samples.
    for qid, rel_docs in tqdm(qrels.items(), total=len(qrels), desc="Processing queries"):
        if qid not in queries:
            continue
        query_text = queries[qid]
        pos_doc_ids = [doc_id for doc_id, score in rel_docs.items() if score > 0]
        if not pos_doc_ids:
            continue
        for pos_doc_id in pos_doc_ids:
            pos_doc_text = corpus[pos_doc_id]['text']
            if hard_negative and qid in hard_negatives:
                neg_doc_id = hard_negatives[qid]  # Use the precomputed hard negative.
            else:
                # Fallback to a random negative.
                neg_doc_id = random.choice(all_doc_ids)
                while neg_doc_id in rel_docs:
                    neg_doc_id = random.choice(all_doc_ids)
            neg_doc_text = corpus[neg_doc_id]['text']
            training_samples.append((query_text, pos_doc_text, neg_doc_text))
            
    return training_samples

def train_and_validate(
    model, optimizer, loss_fn, training_samples, tokenizer, device, batch_size, epoch,
    grad_accum_steps=1, total_epochs=1, validate_every=1000, 
    eval_fn=None, eval_data=None, validation_samples=None,
    patience=3, best_metric=-float("inf"), best_model_dir="./best_model"
):
    """
    Trains for one epoch with gradient accumulation and performs validation (with early stopping)
    every `validate_every` training steps. Validation computes both a validation loss (on
    provided validation_samples) and full retrieval metrics (if eval_fn and eval_data are provided).
    
    If the primary metric (NDCG@10) improves, best_metric is updated, wait is reset, and the best model
    is saved. Otherwise, the wait counter increases, and if it exceeds patience, the training stops early.
    
    Returns a tuple:
       (avg_training_loss, steps_in_epoch, updated_best_metric, wait, early_stop_flag)
    """
    model.train()
    total_loss = 0.0
    random.shuffle(training_samples)
    num_batches = len(training_samples) // batch_size + int(len(training_samples) % batch_size > 0)
    local_steps = 0
    trained_samples_num = 0
    wait = 0
    early_stop = False

    pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}", unit="batch")
    optimizer.zero_grad()
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        batch = training_samples[start_idx : start_idx + batch_size]
        
        # Create batched lists.
        batch_queries = [query for query, pos_doc, neg_doc in batch]
        batch_pos_docs = [pos_doc for query, pos_doc, neg_doc in batch]
        batch_neg_docs = [neg_doc for query, pos_doc, neg_doc in batch]
        
        input_ids_pos, token_type_ids_pos, attention_mask_pos = model.prepare_input(batch_pos_docs, batch_queries, tokenizer)
        input_ids_neg, token_type_ids_neg, attention_mask_neg = model.prepare_input(batch_neg_docs, batch_queries, tokenizer)
        
        input_ids_pos = input_ids_pos.to(device)
        token_type_ids_pos = token_type_ids_pos.to(device)
        attention_mask_pos = attention_mask_pos.to(device)
        input_ids_neg = input_ids_neg.to(device)
        token_type_ids_neg = token_type_ids_neg.to(device)
        attention_mask_neg = attention_mask_neg.to(device)
        
        # Forward passes.
        output_pos = model(input_ids=input_ids_pos, token_type_ids=token_type_ids_pos, attention_mask=attention_mask_pos, return_dict=True)
        output_neg = model(input_ids=input_ids_neg, token_type_ids=token_type_ids_neg, attention_mask=attention_mask_neg, return_dict=True)
        score_pos = output_pos["logits"]  # shape: (batch, 1)
        score_neg = output_neg["logits"]  # shape: (batch, 1)

        # Create target tensor for MarginRankingLoss.
        target = torch.ones(score_pos.size(), device=device)
        loss = loss_fn(score_pos.view(-1), score_neg.view(-1), target.view(-1))
        loss = loss / grad_accum_steps
        loss.backward()
        
        total_loss += loss.item() * grad_accum_steps

        grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        
        # Perform an optimizer step if accumulated enough mini-batches.
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
            optimizer.step()
            optimizer.zero_grad()
            local_steps += 1
            trained_samples_num += len(batch)
            
            # Log training metrics.
            wandb.log({
                "train/loss": loss.item() * grad_accum_steps,
                "train/gradient_norm": grad_norm,
                "train/epoch": epoch + 1,
                "train/batch": batch_idx,
                "train/training_samples_completeness": trained_samples_num / len(training_samples)
            }, step=local_steps)
            
            # Run validation every `validate_every` training steps.
            if local_steps % validate_every == 0:
                model.eval()
                # Compute validation loss over the provided validation_samples.
                if validation_samples is not None and len(validation_samples) > 0:
                    val_loss_total = 0.0
                    val_batches = len(validation_samples) // batch_size + int(len(validation_samples) % batch_size > 0)
                    for v_batch_idx in range(val_batches):
                        v_start_idx = v_batch_idx * batch_size
                        v_batch = validation_samples[v_start_idx: v_start_idx + batch_size]
                        if not v_batch:
                            break
                        batch_queries_val = [query for query, pos_doc, neg_doc in v_batch]
                        batch_pos_docs_val = [pos_doc for query, pos_doc, neg_doc in v_batch]
                        batch_neg_docs_val = [neg_doc for query, pos_doc, neg_doc in v_batch]
                        
                        with torch.no_grad():
                            input_ids_pos_val, token_type_ids_pos_val, attention_mask_pos_val = model.prepare_input(batch_pos_docs_val, batch_queries_val, tokenizer)
                            input_ids_neg_val, token_type_ids_neg_val, attention_mask_neg_val = model.prepare_input(batch_neg_docs_val, batch_queries_val, tokenizer)
                            input_ids_pos_val = input_ids_pos_val.to(device)
                            token_type_ids_pos_val = token_type_ids_pos_val.to(device)
                            attention_mask_pos_val = attention_mask_pos_val.to(device)
                            input_ids_neg_val = input_ids_neg_val.to(device)
                            token_type_ids_neg_val = token_type_ids_neg_val.to(device)
                            attention_mask_neg_val = attention_mask_neg_val.to(device)
                            
                            output_pos_val = model(input_ids=input_ids_pos_val, token_type_ids=token_type_ids_pos_val, attention_mask=attention_mask_pos_val, return_dict=True)
                            output_neg_val = model(input_ids=input_ids_neg_val, token_type_ids=token_type_ids_neg_val, attention_mask=attention_mask_neg_val, return_dict=True)
                            score_pos_val = output_pos_val["logits"]
                            score_neg_val = output_neg_val["logits"]
                            target_val = torch.ones(score_pos_val.size(), device=device)
                            loss_val = loss_fn(score_pos_val.view(-1), score_neg_val.view(-1), target_val.view(-1))
                        val_loss_total += loss_val.item()
                    avg_val_loss = val_loss_total / val_batches if val_batches > 0 else 0.0
                    wandb.log({"val/loss": avg_val_loss, "val/step": local_steps}, step=local_steps)
                    logging.info(f"Validation loss at global step {local_steps}: {avg_val_loss:.4f}")
                
                # Run full retrieval evaluation if eval_fn and eval_data are provided.
                primary_metric = None
                if eval_fn is not None and eval_data is not None:
                    corpus_dev, queries_dev, qrels_dev, eval_batch_size, eval_accum_steps = eval_data
                    dev_metrics = eval_fn(
                        model, corpus_dev, queries_dev, qrels_dev, tokenizer, device,
                        batch_size=eval_batch_size, eval_accumulation_steps=eval_accum_steps
                    )
                    primary_metric = dev_metrics["NDCG"].get("NDCG@10", 0)
                    wandb.log({
                        "val/step": local_steps,
                        "val/NDCG@10": primary_metric,
                        "val/MAP@10": dev_metrics["MAP"].get("MAP@10", 0),
                    }, step=local_steps)
                    logging.info(f"Full evaluation at global step {local_steps}: {dev_metrics}")
                
                # Early stopping check using the primary metric from retrieval evaluation.
                if primary_metric is not None:
                    if primary_metric > best_metric:
                        best_metric = primary_metric
                        wait = 0  # reset early stopping counter
                        model.save_pretrained(best_model_dir)
                        logging.info(f"New best model saved at step {local_steps} with NDCG@10: {best_metric}")
                    else:
                        wait += 1
                        logging.info(f"No improvement in NDCG@10. Early stopping wait counter: {wait}/{patience}")
                    if wait >= patience:
                        logging.info(f"Early stopping triggered at global step {local_steps}.")
                        early_stop = True
                        model.train()
                        break  # break out of training batches
                model.train()
        
        pbar.set_postfix(loss=f"{loss.item() * grad_accum_steps:.4f}")
        pbar.update(1)
        if early_stop:
            break  # break out of epoch loop if early stopping triggered
    pbar.close()
    avg_loss = total_loss / num_batches
    return avg_loss, local_steps, best_metric, wait, early_stop

def beir_evaluate(qrels: dict, results: dict, k_values: list, ignore_identical_ids: bool = True):
    """
    Evaluates BEIR metrics using pytrec_eval.
    """
    if ignore_identical_ids:
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
    Dummy implementation for custom BEIR metrics.
    Replace with your actual metric computations as needed.
    """
    if metric.lower() in ["mrr", "mrr@k", "mrr_cut"]:
        avg_mrr = 0.5
        return {"MRR": avg_mrr}
    elif metric.lower() in ["top_k_accuracy", "acc", "accuracy"]:
        avg_acc = 0.5
        return {"Top_K_Accuracy": avg_acc}
    return {}

def evaluate_full_retrieval(model, corpus: dict, queries: dict, qrels: dict, tokenizer, device, batch_size=2, eval_accumulation_steps=1, k_values=[1, 3, 5, 10, 100]):
    """
    Full retrieval evaluation: for each query, iterate over the corpus in batches,
    compute scores with the model, and then calculate BEIR metrics.
    """
    model.eval()
    results = {}
    total_inference_time = 0.0
    total_docs_processed = 0
    
    for query_id, query in tqdm(queries.items(), desc="Evaluating queries"):
        results[query_id] = {}
        doc_ids = list(corpus.keys())
        for i in range(0, len(doc_ids), batch_size):
            batch_doc_ids = doc_ids[i : i + batch_size]
            batch_docs = [corpus[doc_id]['text'] for doc_id in batch_doc_ids]
            batch_input_ids, batch_token_type_ids, batch_attention_mask = model.prepare_input(batch_docs, [query] * len(batch_docs), tokenizer)
            batch_input_ids = batch_input_ids.to(device)
            batch_token_type_ids = batch_token_type_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            
            start_time = time.time()
            with torch.no_grad():
                output = model(input_ids=batch_input_ids, token_type_ids=batch_token_type_ids, attention_mask=batch_attention_mask, return_dict=True)
            elapsed = time.time() - start_time
            total_inference_time += elapsed
            total_docs_processed += len(batch_doc_ids)
            batch_scores = output["logits"].squeeze(-1).tolist()
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
    wandb.log({"eval_metrics": metrics})
    return metrics

def subsample_dev_set(queries_dev: dict, qrels_dev: dict, sample_percentage: float = 0.05):
    """
    Subsamples the development set by randomly selecting a percentage of queries.

    Args:
        queries_dev (dict): Dictionary of dev queries {query_id: query_text}.
        qrels_dev (dict): Dictionary of dev qrels {query_id: {doc_id: relevance, ...}}.
        sample_percentage (float, optional): Fraction of queries to sample. Default is 0.05 (5%).

    Returns:
        tuple: A tuple (sampled_queries, sampled_qrels) where:
            sampled_queries (dict): Subsampled dev queries.
            sampled_qrels (dict): Corresponding qrels for the subsampled queries.
    """
    dev_query_ids = list(queries_dev.keys())
    num_sample = max(1, int(len(dev_query_ids) * sample_percentage))
    sampled_ids = random.sample(dev_query_ids, num_sample)
    
    sampled_queries = {qid: queries_dev[qid] for qid in sampled_ids}
    sampled_qrels = {qid: qrels_dev[qid] for qid in sampled_ids if qid in qrels_dev}
    
    return sampled_queries, sampled_qrels

#############################################
# Main Finetuning Script
#############################################

def main():
    parser = argparse.ArgumentParser()
    # Training settings.
    parser.add_argument("--datasets", type=str, nargs='+', default=["msmarco", "hotpotqa", "nq-train"],
                        help="List of datasets to use (e.g., msmarco hotpotqa nq-train nq).")
    parser.add_argument("--pretrained_checkpoint", type=str, required=True,
                        help="Path to the pretrained GFRForCausalLM checkpoint.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--margin", type=float, default=1.0, help="Margin for MarginRankingLoss")
    
    # Evaluation settings.
    parser.add_argument("--sample_dev_percentage", type=float, default=0.05, help="Percentage of dev queries to sample for evaluation")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Per-device evaluation batch size")
    parser.add_argument("--eval_accumulation_steps", type=int, default=1, help="Evaluation accumulation steps")
    parser.add_argument("--patience", type=int, default=3, help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--validate_every_n_steps", type=int, default=1000, help="Perform validation every n training steps")
    
    # Logging and checkpointing.
    parser.add_argument("--output_dir", type=str, default="./gfr_finetune_ckpts", help="Output directory for model checkpoints")
    parser.add_argument("--save_model_path", type=str, default="gfr_finetune_final", help="Directory to save the final best model")
    parser.add_argument("--run_name", type=str, default="", help="Run name for logging")
    parser.add_argument("--wandb_project", type=str, default="gfr_finetuning_document_ranking", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="your_group_name", help="Wandb entity name")
    parser.add_argument("--wandb_api_key", type=str, default="your_wandb_api_key", help="Wandb API key for logging")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    now_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.run_name + "_" + now_datetime

    # Initialize Wandb.
    wandb.login(key=args.wandb_api_key)
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config={
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
    })

    # Load tokenizer.
    tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # if tokenizer.cls_token is None:
    #     tokenizer.add_special_tokens({"cls_token": "[CLS]"})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({"sep_token": "[SEP]"})
    if "[SCORE]" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["[SCORE]"]})

    # Load pretrained GFRForCausalLM and extract its backbone.
    pretrained_causal_model = GFRForCausalLM.from_pretrained(args.pretrained_checkpoint)
    backbone_model = pretrained_causal_model.get_decoder()
    config = pretrained_causal_model.config

    # Initialize GFRForSequenceScoring (which uses its own GFRModelWithTokenTypes backbone).
    model = GFRForSequenceScoring(config)
    # Transfer weights from the pretrained backbone (non-strict to allow extra embeddings).
    model.gfr.load_state_dict(backbone_model.state_dict(), strict=False)

    model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    # Set up loss and optimizer.
    loss_fn = nn.MarginRankingLoss(margin=args.margin)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Load datasets.
    training_samples = []
    dev_data = {}   # { dataset: (corpus, queries, qrels) }
    test_data = {}  # { dataset: (corpus, queries, qrels) }
    for dataset in args.datasets:
        logging.info(f"Loading dataset: {dataset} (train split)")
        corpus_train, queries_train, qrels_train = load_dataset(dataset, split="train")
        bm25_index, bm25_doc_ids = build_bm25_index(corpus_train)
        # Prepare training samples with hard negatives enabled:
        training_samples.extend(
            prepare_training_samples(
                corpus_train, queries_train, qrels_train,
                hard_negative=True,
                bm25_index=bm25_index,
                bm25_doc_ids=bm25_doc_ids
            )
        )
                
        logging.info(f"Loading dataset: {dataset} (dev split)")
        corpus_dev, queries_dev, qrels_dev = load_dataset(dataset, split="dev")
        dev_data[dataset] = (corpus_dev, queries_dev, qrels_dev)
        
        logging.info(f"Loading dataset: {dataset} (test split)")
        corpus_test, queries_test, qrels_test = load_dataset(dataset, split="test")
        test_data[dataset] = (corpus_test, queries_test, qrels_test)

    logging.info(f"Total training samples: {len(training_samples)}")

    # Combine dev data from all datasets.
    combined_corpus_dev = {}
    combined_queries_dev = {}
    combined_qrels_dev = {}
    for dataset, (corpus_d, queries_d, qrels_d) in dev_data.items():
        combined_corpus_dev.update(corpus_d)
        combined_queries_dev.update(queries_d)
        combined_qrels_dev.update(qrels_d)

    # Shuffle and subsample the combined dev set.
    sampled_queries_dev, sampled_qrels_dev = subsample_dev_set(combined_queries_dev, combined_qrels_dev, sample_percentage=args.sample_dev_percentage)

    # Build BM25 index on the combined dev corpus.
    val_bm25_index, val_bm25_doc_ids = build_bm25_index(combined_corpus_dev)
    # Prepare validation samples using hard negatives.
    validation_samples = prepare_training_samples(combined_corpus_dev, sampled_queries_dev, sampled_qrels_dev, hard_negative=True, bm25_index=val_bm25_index, bm25_doc_ids=val_bm25_doc_ids)

    for dataset in args.datasets:
        logging.info(f"Dataset {dataset} dev queries: {len(dev_data[dataset][1])}")
        logging.info(f"Sampled dev queries: {len(sampled_queries_dev)}")
        logging.info(f"Dataset {dataset} test queries: {len(test_data[dataset][1])}")

    checkpoint_dir = args.output_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_dir = os.path.join(checkpoint_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)

    global_step = 0
    checkpoint_interval = 10000  # save a checkpoint every 10k training steps

    # Training loop with integrated validation and early stopping.
    current_best_metric = -float("inf")
    wait = 0
    early_stop_global = False

    for epoch in range(args.num_train_epochs):
        logging.info(f"Starting epoch {epoch+1}")
        epoch_loss, steps_in_epoch, current_best_metric, wait, early_stop = train_and_validate(
            model, optimizer, loss_fn, training_samples, tokenizer, device,
            args.per_device_train_batch_size, epoch, grad_accum_steps=args.gradient_accumulation_steps,
            total_epochs=args.num_train_epochs,
            validate_every=args.validate_every_n_steps,
            eval_fn=evaluate_full_retrieval,
            eval_data=(combined_corpus_dev, sampled_queries_dev, sampled_qrels_dev, args.per_device_eval_batch_size, args.eval_accumulation_steps),
            validation_samples=validation_samples,
            patience=args.patience,
            best_metric=current_best_metric,
            best_model_dir=best_model_dir
        )
        global_step += steps_in_epoch
        logging.info(f"Epoch {epoch+1} training loss: {epoch_loss:.4f}")
        wandb.log({"epoch": epoch+1, "training_loss": epoch_loss, "global_step": global_step})
        
        # Save checkpoint if required.
        if global_step % checkpoint_interval < steps_in_epoch:
            checkpoint_filename = f"step_{global_step:06d}.pth"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved at {checkpoint_path}")

        # Also save an epoch checkpoint.
        epoch_ckpt_filename = f"epoch_{epoch+1:02d}.pth"
        epoch_ckpt_path = os.path.join(checkpoint_dir, epoch_ckpt_filename)
        torch.save(model.state_dict(), epoch_ckpt_path)
        logging.info(f"Epoch checkpoint saved at {epoch_ckpt_path}")
        
        if early_stop:
            early_stop_global = True
            break  # exit training loop early if triggered

    # Final evaluation on test sets.
    for dataset, (corpus_test, queries_test, qrels_test) in test_data.items():
        test_metrics = evaluate_full_retrieval(
            model, corpus_test, queries_test, qrels_test, tokenizer, device,
            batch_size=args.per_device_eval_batch_size
        )
        logging.info(f"Test metrics for {dataset}: {test_metrics}")
        wandb.log({f"{dataset}_test_metrics": test_metrics})

    # Option 1: Save using the HuggingFace standard (config + model weights)
    model.save_pretrained(args.save_model_path)
    logging.info("Training completed and best model saved.")
    wandb.finish()

if __name__ == "__main__":
    main()

    """
    python -m script.gfr_finetune \
    --datasets msmarco \
    --pretrained_checkpoint ./gfr_pretrain_causal_lm_final_finewebedu_v2_200m \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr 1e-4 \
    --margin 5.0 \
    --sample_dev_percentage 0.05 \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 1 \
    --patience 3 \
    --validate_every_n_steps 1000 \
    --output_dir ./gfr_finetune_ckpts_200m_msmarco \
    --save_model_path ./gfr_finetune_final_200m_msmarco \
    --run_name 200M_msmarco \
    --wandb_project gfr_finetuning_document_ranking \
    --wandb_entity nlp-maocode \
    --wandb_api_key your_wandb_api_key
    """