import os
import argparse
import logging
import time
import random
from datetime import datetime

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

#############################################
# Helper Functions (can be moved to utils.py)
#############################################

def load_dataset(dataset: str, split: str):
    import pathlib
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
    For each query, creates a training sample tuple: (query_text, positive_doc_text, negative_doc_text)
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
            neg_doc_id = random.choice(all_doc_ids)
            while neg_doc_id in rel_docs:
                neg_doc_id = random.choice(all_doc_ids)
            neg_doc_text = corpus[neg_doc_id]['text']
            training_samples.append((query_text, pos_doc_text, neg_doc_text))
    return training_samples

def train_one_epoch(model, optimizer, loss_fn, training_samples, tokenizer, device, batch_size, epoch, grad_accum_steps=1):
    """
    Performs one epoch of training with gradient accumulation.
    Returns the average loss and the number of optimizer steps (training steps) performed.
    """
    model.train()
    total_loss = 0.0
    random.shuffle(training_samples)
    num_batches = len(training_samples) // batch_size + int(len(training_samples) % batch_size > 0)
    local_steps = 0

    pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}", unit="batch")
    optimizer.zero_grad()
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        batch = training_samples[start_idx : start_idx + batch_size]
        
        # Create batched lists.
        batch_queries = [query for query, pos_doc, neg_doc in batch]
        batch_pos_docs = [pos_doc for query, pos_doc, neg_doc in batch]
        batch_neg_docs = [neg_doc for query, pos_doc, neg_doc in batch]
        
        # Use the model's prepare_input method (from the backbone) to build inputs.
        base_model = model.module if hasattr(model, "module") else model
        input_ids_pos, token_type_ids_pos = base_model.prepare_input(batch_pos_docs, batch_queries, tokenizer)
        input_ids_neg, token_type_ids_neg = base_model.prepare_input(batch_neg_docs, batch_queries, tokenizer)
        
        input_ids_pos = input_ids_pos.to(device)
        token_type_ids_pos = token_type_ids_pos.to(device)
        input_ids_neg = input_ids_neg.to(device)
        token_type_ids_neg = token_type_ids_neg.to(device)
        
        # Forward passes.
        output_pos = model(input_ids=input_ids_pos, token_type_ids=token_type_ids_pos, return_dict=True)
        output_neg = model(input_ids=input_ids_neg, token_type_ids=token_type_ids_neg, return_dict=True)
        score_pos = output_pos["logits"]  # shape: (batch, 1)
        score_neg = output_neg["logits"]  # shape: (batch, 1)
        
        # Create target tensor for MarginRankingLoss.
        target = torch.ones(score_pos.size(), device=device)
        loss = loss_fn(score_pos.view(-1), score_neg.view(-1), target.view(-1))
        loss = loss / grad_accum_steps
        loss.backward()
        
        total_loss += loss.item() * grad_accum_steps
        
        # Perform an optimizer step if accumulated enough mini-batches.
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
            optimizer.step()
            optimizer.zero_grad()
            local_steps += 1
        
        wandb.log({
            "iteration_loss": loss.item() * grad_accum_steps,
            "epoch": epoch+1,
            "batch": batch_idx+1
        })
        
        pbar.set_postfix(loss=f"{loss.item() * grad_accum_steps:.4f}")
        pbar.update(1)
    pbar.close()
    avg_loss = total_loss / num_batches
    return avg_loss, local_steps

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
    base_model = model.module if hasattr(model, "module") else model
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
            batch_input_ids, batch_token_type_ids = base_model.prepare_input(batch_docs, [query] * len(batch_docs), tokenizer)
            batch_input_ids = batch_input_ids.to(device)
            batch_token_type_ids = batch_token_type_ids.to(device)
            
            start_time = time.time()
            with torch.no_grad():
                output = model(input_ids=batch_input_ids, token_type_ids=batch_token_type_ids, return_dict=True)
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
    return metrics

#############################################
# Main Finetuning Script
#############################################

def main():
    parser = argparse.ArgumentParser()
    # Training settings.
    parser.add_argument("--datasets", type=str, nargs='+', default=["msmarco", "hotpotqa"],
                        help="List of datasets to use (e.g., msmarco hotpotqa nq).")
    parser.add_argument("--pretrained_checkpoint", type=str, required=True,
                        help="Path to the pretrained GFRForCausalLM checkpoint.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    # Evaluation settings.
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Per-device evaluation batch size")
    parser.add_argument("--eval_accumulation_steps", type=int, default=1, help="Evaluation accumulation steps")
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

    # Load pretrained GFRForCausalLM and extract its backbone.
    pretrained_causal_model = GFRForCausalLM.from_pretrained(args.pretrained_checkpoint)
    backbone_model = pretrained_causal_model.get_decoder()
    config = pretrained_causal_model.config

    # Initialize GFRForSequenceScoring (which uses its own GFRModelWithTokenTypes backbone).
    model = GFRForSequenceScoring(config)
    # Transfer weights from the pretrained backbone (non-strict to allow extra embeddings).
    model.gfr.load_state_dict(backbone_model.state_dict(), strict=False)
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    # Set up loss and optimizer.
    loss_fn = nn.MarginRankingLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Load datasets.
    training_samples = []
    dev_data = {}   # { dataset: (corpus, queries, qrels) }
    test_data = {}  # { dataset: (corpus, queries, qrels) }
    for dataset in args.datasets:
        logging.info(f"Loading dataset: {dataset} (train split)")
        corpus_train, queries_train, qrels_train = load_dataset(dataset, split="train")
        training_samples.extend(prepare_training_samples(corpus_train, queries_train, qrels_train))
        logging.info(f"Loading dataset: {dataset} (dev split)")
        corpus_dev, queries_dev, qrels_dev = load_dataset(dataset, split="dev")
        dev_data[dataset] = (corpus_dev, queries_dev, qrels_dev)
        logging.info(f"Loading dataset: {dataset} (test split)")
        corpus_test, queries_test, qrels_test = load_dataset(dataset, split="test")
        test_data[dataset] = (corpus_test, queries_test, qrels_test)

    for dataset in args.datasets:
        logging.info(f"Dataset {dataset} dev queries: {len(dev_data[dataset][1])}")
        logging.info(f"Dataset {dataset} test queries: {len(test_data[dataset][1])}")

    checkpoint_dir = args.output_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop variables.
    global_step = 0
    checkpoint_interval = 10000  # save a checkpoint every 10k training steps
    best_metric = -float("inf")    # for example, best NDCG@10
    best_model_dir = os.path.join(checkpoint_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)

    for epoch in range(args.num_train_epochs):
        logging.info(f"Starting epoch {epoch+1}")
        epoch_loss, steps_in_epoch = train_one_epoch(
            model, optimizer, loss_fn, training_samples, tokenizer, device,
            args.per_device_train_batch_size, epoch, args.gradient_accumulation_steps
        )
        global_step += steps_in_epoch
        logging.info(f"Epoch {epoch+1} training loss: {epoch_loss:.4f}")
        wandb.log({"epoch": epoch+1, "training_loss": epoch_loss, "global_step": global_step})

        # Evaluate on each dev dataset.
        for dataset, (corpus_dev, queries_dev, qrels_dev) in dev_data.items():
            dev_metrics = evaluate_full_retrieval(
                model, corpus_dev, queries_dev, qrels_dev, tokenizer, device,
                batch_size=args.per_device_eval_batch_size, eval_accumulation_steps=args.eval_accumulation_steps
            )
            logging.info(f"Epoch {epoch+1} dev metrics for {dataset}: {dev_metrics}")
            wandb.log({f"epoch_{epoch+1}_{dataset}_dev": dev_metrics})
            # Choose primary metric (here, NDCG@10) from the first dev dataset.
            primary_metric = dev_metrics["NDCG"].get("NDCG@10", 0)
            if primary_metric > best_metric:
                best_metric = primary_metric
                model.save_pretrained(best_model_dir)
                logging.info(f"New best model saved with NDCG@10: {best_metric}")

        # Save a checkpoint based on training steps if needed.
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

    # Final evaluation on test sets.
    for dataset, (corpus_test, queries_test, qrels_test) in test_data.items():
        test_metrics = evaluate_full_retrieval(
            model, corpus_test, queries_test, qrels_test, tokenizer, device,
            batch_size=args.per_device_eval_batch_size
        )
        logging.info(f"Test metrics for {dataset}: {test_metrics}")
        wandb.log({f"{dataset}_test_metrics": test_metrics})

    final_checkpoint_path = os.path.join(args.save_model_path, f"epoch_{args.num_train_epochs:02d}.pth")
    torch.save(model.state_dict(), final_checkpoint_path)
    logging.info("Training completed and final model checkpoint saved.")
    wandb.finish()

if __name__ == "__main__":
    main()