import argparse
import logging
from datetime import datetime
import math

import torch
from transformers import TrainingArguments, EarlyStoppingCallback, DataCollatorWithPadding
import wandb
import random

from utils import *
from train.utils import *
from train.trainer import DocumentRankingTrainer
from dataset.document_ranking import DocumentRankingDataset

from GFR.modeling_GFR import GFRForCausalLM, GFRForSequenceScoring
from GFR.tokenizer_utils import get_tokenizer

def main():
    """
    For wandb logging, set the following environment variables:

    export WANDB_API_KEY="your_wandb_api_key"
    export WANDB_PROJECT="gfr_finetuning_document_ranking"
    export WANDB_ENTITY="nlp-maocode"
    """
    
    # =========== Setup Arguments ==============
    parser = argparse.ArgumentParser(description="Fine-tune a scoring model with token type embeddings and a score head.")    
    
    # Training settings.
    parser.add_argument("--deepspeed_config", type=str, default="deepspeed_config.json",
                    help="Path to the DeepSpeed configuration file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    # Specify multiple datasets.
    parser.add_argument("--pretrained_checkpoint", type=str, 
                        default="gfr_pretrain_causal_lm_final_finewebedu_v2_200m",
                        help="Path to the pretrained checkpoint directory.")
    parser.add_argument("--datasets", type=str, default="msmarco,nq-train,hotpotqa,fiqa",
                        help="Comma-separated list of dataset names to use for training (e.g., ms_marco,nq,hotpotqa,fiqa).")
    parser.add_argument("--samples_per_dataset", type=str, default="0,0,0,0",
                        help="Comma-separated list of number of training samples to use per dataset in the same order as --datasets. Use 0 to use all available samples.")
    # Accept a comma-separated list of index names corresponding to each dataset.
    parser.add_argument("--index_names", type=str,
                        default="msmarco-passage,beir-v1.0.0-nq.flat,beir-v1.0.0-hotpotqa.flat,beir-v1.0.0-fiqa.flat",
                        help="Comma-separated list of index names for each dataset, in the same order as --datasets.")
    parser.add_argument("--index_type", type=str, default="dense",
                        help="Type of index to use (dense or sparse).")
    parser.add_argument("--quey_encoder", type=str, default="BAAI/bge-base-en-v1.5", help="Query encoder model name for dense vectors.")
    parser.add_argument("--n_per_query", type=int, default=1,
                        help="Number of positive and negative samples to select per query.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing for memory efficiency.")
    
    # Evaluation settings.
    parser.add_argument("--eval_dataset_file", type=str, default="validation_samples.jsonl", help="Path to the evaluation dataset file.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Per-device evaluation batch size")
    parser.add_argument("--eval_accumulation_steps", type=int, default=1, help="Evaluation accumulation steps")
    parser.add_argument("--patience", type=int, default=3, help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--validate_every_n_steps", type=int, default=1000, help="Perform validation every n training steps")
    
    # Logging and checkpointing.
    parser.add_argument("--output_dir", type=str, default="./gfr_finetune_ckpts", help="Output directory for model checkpoints")
    parser.add_argument("--save_model_path", type=str, default="gfr_finetune_final", help="Directory to save the final best model")
    parser.add_argument("--run_name", type=str, default="", help="Run name for logging")

    parser.add_argument("--use_prepared_data", action="store_true", 
                        help="If set, load pre-organized data from prepared files rather than computing hard negatives.")
    parser.add_argument("--prepared_data_files", type=str,
                        default="datasets/bge_data/split_1/msmarco_hn_train.jsonl,datasets/bge_data/split_1/nq.jsonl,datasets/bge_data/split/fever.json,datasets/bge_data/split/hotpotqa_pairs.json,datasets/bge_data/split/mr-tydi_english.jsonl",
                        help="Comma-separated list of file paths for the prepared dataset in the desired order.")
    parser.add_argument("--prepared_data_sample_counts", type=str,
                        default="0,0,0,0,0",
                        help="Comma-separated list of sample counts for each prepared dataset file in the same order. Use 0 to use all available samples.")

    args = parser.parse_args()

    # =========== Basic Setup ==============
    # Check if the batch size is divisible by the group size. (in InfoNCE loss scenario)
    group_size = 1 + args.n_per_query
    if args.per_device_train_batch_size % group_size != 0:
        raise ValueError(
            f"per_device_train_batch_size ({args.per_device_train_batch_size}) "
            f"must be a multiple of group_size ({group_size})"
        )
    if args.per_device_eval_batch_size % group_size != 0:
        raise ValueError(
            f"per_device_eval_batch_size ({args.per_device_eval_batch_size}) "
            f"must be a multiple of group_size ({group_size})"
        )
    
    # Set up logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            # logging.FileHandler(args.log_file, mode="w")
        ],
        force=True
    )
    logger = logging.getLogger()
    logger.addFilter(MainProcessFilter(args.local_rank))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    now_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.run_name + "_" + now_datetime

    # Parse comma-separated lists.
    datasets_list = [d.strip() for d in args.datasets.split(",")]
    samples_list = [int(s.strip()) for s in args.samples_per_dataset.split(",")]
    index_names_list = [d.strip() for d in args.index_names.split(",")]
    if not (len(datasets_list) == len(samples_list) == len(index_names_list)):
        raise ValueError("The number of datasets, samples_per_dataset, and index_names must match.")

    # =========== Setup the Model ==============
    # Load tokenizer.
    tokenizer = get_tokenizer()

    # Load pretrained GFRForCausalLM and extract its backbone.
    pretrained_causal_model = GFRForCausalLM.from_pretrained(args.pretrained_checkpoint, weights_only=True)
    backbone_model = pretrained_causal_model.get_decoder()
    config = pretrained_causal_model.config

    # Initialize GFRForSequenceScoring.
    model = GFRForSequenceScoring(config)
    model.gfr.load_state_dict(backbone_model.state_dict(), strict=False)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    if args.gradient_checkpointing:
        # Enable gradient checkpointing for memory efficiency.
        # This is especially useful for large models.
        logger.info("Enabling gradient checkpointing...")
        model.gfr.gradient_checkpointing_enable()


    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {num_params}")

    # =========== Setup Training Dataset ==============
    if args.use_prepared_data:
        # Parse file paths and sample counts.
        prepared_files = [f.strip() for f in args.prepared_data_files.split(",")]
        sample_counts = [int(s.strip()) for s in args.prepared_data_sample_counts.split(",")]
        if len(prepared_files) != len(sample_counts):
            raise ValueError("The number of prepared files and sample counts must match.")
        logger.info("Loading prepared training samples from files with specified sample counts...")
        all_training_samples = load_prepared_samples(prepared_files, sample_counts, args.n_per_query, logger)
        logger.info(f"Total mixed training samples (prepared): {len(all_training_samples)}")
        if len(all_training_samples) > 0:
            logger.info(f"First prepared training sample group: {all_training_samples[:1 + args.n_per_query]}")
    else:
        # Original: load and mix training samples from multiple datasets.
        datasets_list = [d.strip() for d in args.datasets.split(",")]
        samples_list = [int(s.strip()) for s in args.samples_per_dataset.split(",")]
        index_names_list = [d.strip() for d in args.index_names.split(",")]
        if not (len(datasets_list) == len(samples_list) == len(index_names_list)):
            raise ValueError("The number of datasets, samples_per_dataset, and index_names must match.")

        all_training_samples = []
        for dataset_name, sample_count, index_name in zip(datasets_list, samples_list, index_names_list):
            logger.info(f"Loading dataset: {dataset_name} (train split)")
            corpus_train, queries_train, qrels_train = load_dataset(logger, dataset_name, split="train")
            logger.info(f"Using index '{index_name}' for dataset: {dataset_name}")
            logger.info(f"Preparing training samples for dataset: {dataset_name}")

            if sample_count > 0 and sample_count < len(qrels_train):
                sampled_qids = random.sample(list(qrels_train.keys()), sample_count)
                qrels_train_sampled = {qid: qrels_train[qid] for qid in sampled_qids}
                queries_train_sampled = {qid: queries_train[qid] for qid in sampled_qids if qid in queries_train}
            else:
                qrels_train_sampled = qrels_train
                queries_train_sampled = queries_train
            logger.info(f"Number of queries in the sampled training set: {len(queries_train_sampled)}")

            samples = prepare_training_samples_infonce(
                corpus_train,
                queries_train_sampled,
                qrels_train_sampled,
                n_per_query=args.n_per_query,
                hard_negative=True,
                index_name=index_name,
                index_type=args.index_type,
                query_encoder=args.quey_encoder
            )
            logger.info(f"Total samples generated for {dataset_name}: {len(samples)}")
            all_training_samples.extend(samples)
        logger.info(f"Total mixed training samples: {len(all_training_samples)}")
        logger.info(f"First Training samples: {all_training_samples[:1 + args.n_per_query]}")
    
    # Create PyTorch Dataset for training.
    train_dataset = DocumentRankingDataset(all_training_samples, tokenizer, model)
    logger.info(f"Training dataset size: {len(train_dataset)}")

    # =========== Setup Validation Dataset ==============
    validation_samples = load_json_file(args.eval_dataset_file)
    logger.info(f"Total samples generated for dev set: {len(validation_samples)}")
    logger.info(f"First Validation samples: {validation_samples[0]}")
    val_dataset = DocumentRankingDataset(validation_samples, tokenizer, model)
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # =========== Setup Trainer ==============
    total_training_steps = math.ceil(
        len(train_dataset) / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    ) * args.num_train_epochs
    warmup_steps = int(0.01 * total_training_steps)

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
        max_grad_norm=1.0,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_accumulation_steps=args.eval_accumulation_steps,
        eval_strategy="steps",
        eval_steps=args.validate_every_n_steps,
        logging_dir="./logs_finetune",
        logging_steps=50,
        logging_first_step=True,
        save_steps=5000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=run_name,
        remove_unused_columns=False,
        deepspeed=args.deepspeed_config,
    )
    log_training_config(training_args, logger)

    # Initialize our custom Trainer.
    data_collator = DataCollatorWithPadding(tokenizer)
    trainer = DocumentRankingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=val_dataset,
        n_per_query=args.n_per_query,
        tokenizer=tokenizer
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    # =========== Start Training ==============
    trainer.train()

    # Save the final model.
    if is_main_process():
        trainer.save_model(args.save_model_path)
        logger.info("Training completed and best model saved.")
        wandb.finish()

if __name__ == "__main__":
    main()

    """
    Ensure per_device_train_batch_size is a multiple of (1 + n_per_query)

    Dense Index names: (FAISS)
    - msmarco-v1-passage.bge-base-en-v1.5 (MS MARCO by BGE)
    - beir-v1.0.0-nq.bge-base-en-v1.5 (NQ by BGE)
    - beir-v1.0.0-hotpotqa.bge-base-en-v1.5 (HotPotQA by BGE)
    - beir-v1.0.0-fever.bge-base-en-v1.5 (FEVER by BGE)
    # - beir-v1.0.0-quora.bge-base-en-v1.5 (Quora by BGE, only dev and test)

    # - beir-v1.0.0-fiqa.bge-base-en-v1.5 (FiQA by BGE)
    # - wikipedia-dpr-100w.dkrr-tqa (TriviaQA)
    
    Sparse Index names: (Lucene Standard Inverted Indexes)
    - msmarco-v1-passage
    - beir-v1.0.0-nq.flat
    - beir-v1.0.0-hotpotqa.flat
    - beir-v1.0.0-fiqa.flat

    If you want to load the dataset by yourself, add the following arguments:
    --datasets "msmarco,nq-train,fever,hotpotqa" \
    --samples_per_dataset "650000,100000,150000,50000" \
    --index_names "msmarco-v1-passage.bge-base-en-v1.5,beir-v1.0.0-nq.bge-base-en-v1.5,beir-v1.0.0-fever.bge-base-en-v1.5,beir-v1.0.0-hotpotqa.bge-base-en-v1.5" \
    --index_type "dense" \
    --quey_encoder "BAAI/bge-base-en-v1.5" \
    
    Example usage:

    deepspeed --module train.gfr_finetune \
    --deepspeed_config deepspeed_config.json \
    --pretrained_checkpoint "gfr_pretrain_causal_lm_final_finewebedu_v2_200m" \
    --use_prepared_data \
    --prepared_data_files "datasets/bge_data/split_1/msmarco_hn_train.jsonl,datasets/bge_data/split_1/nq.jsonl,datasets/bge_data/split/fever.json,datasets/bge_data/split/hotpotqa_pairs.json,datasets/bge_data/split/mr-tydi_english.jsonl,datasets/bge_data/split/nli_simcse.json" \
    --prepared_data_sample_counts "0,0,0,0,0,0" \
    --n_per_query 15 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 16 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --eval_dataset_file "datasets/msmarco_val.jsonl" \
    --per_device_eval_batch_size 16 \
    --eval_accumulation_steps 1 \
    --patience 10 \
    --validate_every_n_steps 100 \
    --output_dir "./gfr_finetune_ckpts_210m_bgedata" \
    --save_model_path "gfr_finetune_ckpts_210m_bgedata_final" \
    --run_name "210m_bgedata" \
    --gradient_checkpointing

    
    export HF_HOME="~/../../work/asiis2025/huggingface_cache"    
    """