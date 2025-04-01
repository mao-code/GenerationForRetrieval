import os
import argparse
import logging
import time
import random
from datetime import datetime
import pathlib
import bm25s
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import LlamaTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from GFR.configuration_GFR import GFRConfig
from GFR.modeling_GFR import GFRForCausalLM, GFRForSequenceScoring
import wandb

# BEIR imports.
from beir import util
from beir.datasets.data_loader import GenericDataLoader

from utils import load_dataset
from script.utils import evaluate_full_retrieval, build_bm25_index, search_bm25
from dataset import DocumentRankingDataset

#############################################
# Helper Functions
#############################################

def prepare_training_samples_bce(corpus: dict, queries: dict, qrels: dict, hard_negative: bool = False, bm25_index=None, bm25_doc_ids=None):
    """
    Creates training sample pairs: (query_text, doc_text, label) where label is 1.0 for relevant docs and 0.0 for negatives.
    For each positive, a negative is also added so that their numbers match.
    """
    training_samples = []
    all_doc_ids = list(corpus.keys())
    
    # Precompute hard negatives if enabled.
    hard_negatives = {}
    if hard_negative and bm25_index is not None and bm25_doc_ids is not None:
        for qid in tqdm(qrels, desc="Precomputing hard negatives"):
            query_text = queries[qid]
            doc_ids, scores = search_bm25(query_text, bm25_index, bm25_doc_ids, top_k=10)
            candidate_negatives = [doc_id for doc_id in doc_ids if doc_id not in qrels[qid]]
            if candidate_negatives:
                hard_negatives[qid] = candidate_negatives  # store list of negatives
            else:
                neg_doc_id = random.choice(all_doc_ids)
                while neg_doc_id in qrels[qid]:
                    neg_doc_id = random.choice(all_doc_ids)
                hard_negatives[qid] = [neg_doc_id]

    # For each query, add positive examples and for each positive add a corresponding negative.
    for qid, rel_docs in tqdm(qrels.items(), total=len(qrels), desc="Processing queries"):
        if qid not in queries:
            continue
        query_text = queries[qid]
        pos_doc_ids = [doc_id for doc_id, score in rel_docs.items() if score > 0]
        if not pos_doc_ids:
            continue
        for pos_doc_id in pos_doc_ids:
            pos_doc_text = corpus[pos_doc_id]['text']
            training_samples.append((query_text, pos_doc_text, 1.0))
            # For each positive, add a corresponding negative sample.
            if hard_negative and qid in hard_negatives:
                candidate_negatives = hard_negatives[qid]
                neg_doc_id = random.choice(candidate_negatives)
            else:
                neg_doc_id = random.choice(all_doc_ids)
                while neg_doc_id in rel_docs:
                    neg_doc_id = random.choice(all_doc_ids)
            neg_doc_text = corpus[neg_doc_id]['text']
            training_samples.append((query_text, neg_doc_text, 0.0))
    
    return training_samples

def subsample_dev_set(queries_dev: dict, qrels_dev: dict, sample_percentage: float = 0.05):
    dev_query_ids = list(queries_dev.keys())
    num_sample = max(1, int(len(dev_query_ids) * sample_percentage))
    sampled_ids = random.sample(dev_query_ids, num_sample)
    
    sampled_queries = {qid: queries_dev[qid] for qid in sampled_ids}
    sampled_qrels = {qid: qrels_dev[qid] for qid in sampled_ids if qid in qrels_dev}
    
    return sampled_queries, sampled_qrels
    
#############################################
# Custom Trainer subclass to override compute_loss
#############################################

class DocumentRankingTrainer(Trainer):
    def __init__(self, loss_fn, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, return_dict=True)
        logits = outputs["logits"].view(-1)
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

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
    parser.add_argument("--margin", type=float, default=1.0, help="Margin for MarginRankingLoss (unused in BCE)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer.")
    
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
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({"sep_token": "[SEP]"})
    if "[SCORE]" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["[SCORE]"]})

    # Load pretrained GFRForCausalLM and extract its backbone.
    pretrained_causal_model = GFRForCausalLM.from_pretrained(args.pretrained_checkpoint)
    backbone_model = pretrained_causal_model.get_decoder()
    config = pretrained_causal_model.config

    # Initialize GFRForSequenceScoring.
    model = GFRForSequenceScoring(config)
    model.gfr.load_state_dict(backbone_model.state_dict(), strict=False)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    # Set up loss.
    loss_fn = nn.BCEWithLogitsLoss()

    # Load datasets.
    training_samples = []
    dev_data = {}   # { dataset: (corpus, queries, qrels) }
    test_data = {}  # { dataset: (corpus, queries, qrels) }
    for dataset in args.datasets:
        logging.info(f"Loading dataset: {dataset} (train split)")
        corpus_train, queries_train, qrels_train = load_dataset(dataset, split="train")
        bm25_index, bm25_doc_ids = build_bm25_index(corpus_train)
        training_samples.extend(
            prepare_training_samples_bce(
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

    # Subsample the combined dev set.
    sampled_queries_dev, sampled_qrels_dev = subsample_dev_set(combined_queries_dev, combined_qrels_dev, sample_percentage=args.sample_dev_percentage)
    # Build BM25 index on the combined dev corpus.
    val_bm25_index, val_bm25_doc_ids = build_bm25_index(combined_corpus_dev)
    validation_samples = prepare_training_samples_bce(combined_corpus_dev, sampled_queries_dev, sampled_qrels_dev, hard_negative=True, bm25_index=val_bm25_index, bm25_doc_ids=val_bm25_doc_ids)

    for dataset in args.datasets:
        logging.info(f"Dataset {dataset} dev queries: {len(dev_data[dataset][1])}")
        logging.info(f"Sampled dev queries: {len(sampled_queries_dev)}")
        logging.info(f"Dataset {dataset} test queries: {len(test_data[dataset][1])}")

    # Create PyTorch Datasets.
    train_dataset = DocumentRankingDataset(training_samples, tokenizer, model)
    val_dataset = DocumentRankingDataset(validation_samples, tokenizer, model)

    total_training_steps = math.ceil(len(train_dataset) / (args.per_device_train_batch_size * args.gradient_accumulation_steps)) * args.num_train_epochs
    warmup_steps = int(0.1 * total_training_steps)

    # Set up TrainingArguments.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=True,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,

        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_accumulation_steps=args.eval_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=args.validate_every_n_steps,

        logging_dir="./logs_finetune",
        logging_steps=50,
        logging_first_step=True,
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=run_name,

        remove_unused_columns=False
    )

    # Initialize our custom Trainer.
    trainer = DocumentRankingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss_fn=loss_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    # Train the model.
    trainer.train()

    # Final evaluation on test sets. (full-rerank is really slow, so we skip it here)
    # for dataset, (corpus_test, queries_test, qrels_test) in test_data.items():
    #     test_metrics = evaluate_full_retrieval(
    #         model, corpus_test, queries_test, qrels_test, tokenizer, device,
    #         batch_size=args.per_device_eval_batch_size
    #     )
    #     logging.info(f"Test metrics for {dataset}: {test_metrics}")
    #     wandb.log({f"{dataset}_test_metrics": test_metrics})

    # Save the final model.
    trainer.save_model(args.save_model_path)
    logging.info("Training completed and best model saved.")
    wandb.finish()

if __name__ == "__main__":
    main()

    """
    python -m script.gfr_finetune \
    --datasets msmarco \
    --pretrained_checkpoint gfr_pretrain_causal_lm_final_finewebedu_v2_200m \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --lr 1e-5 \
    --weight_decay 0.01 \
    --sample_dev_percentage 0.05 \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 1 \
    --patience 3 \
    --validate_every_n_steps 100 \
    --output_dir ./gfr_finetune_ckpts_200m_msmarco \
    --save_model_path ./gfr_finetune_final_200m_msmarco \
    --run_name 200M_msmarco \
    --wandb_project gfr_finetuning_document_ranking \
    --wandb_entity nlp-maocode \
    --wandb_api_key your_wandb_api_key
    """