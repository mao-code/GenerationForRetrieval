import argparse
import os
import json
import logging

from utils import load_dataset, save_samples_to_file
from finetune.utils import prepare_training_samples_infonce, subsample_dev_set

def main():
    parser = argparse.ArgumentParser(description="Prepare validation set and save to JSON/JSONL file.")
    parser.add_argument("--dataset", type=str, default="msmarco", help="Name of the BEIR dataset to use")
    parser.add_argument("--split", type=str, default="dev", help="Dataset split to use (dev/test/train)")
    parser.add_argument("--index", type=str, default="msmarco-v1-passage.bge-base-en-v1.5", help="Prebuilt index name")
    parser.add_argument("--index_type", type=str, default="dense", choices=["dense", "sparse"], help="Type of index to use")
    parser.add_argument("--query_encoder", type=str, default="BAAI/bge-base-en-v1.5", help="Query encoder model")
    parser.add_argument("--sample_dev_percentage", type=float, default=0.1, help="Percentage of dev queries to sample")
    parser.add_argument("--n_per_query", type=int, default=16, help="Number of negatives per query")
    parser.add_argument("--hard_negative", action="store_true", help="Whether to compute hard negatives")
    parser.add_argument("--hard_negatives_file", type=str, default=None, help="Path to cached hard negatives file")
    parser.add_argument("--output_file", type=str, default="validation_samples.jsonl", help="Output file to save validation samples (json or jsonl)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set up logger.
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting validation set preparation...")

    # Load dataset.
    logger.info(f"Loading dataset: {args.dataset} (split: {args.split})")
    corpus, queries, qrels = load_dataset(logger, args.dataset, args.split)

    # Subsample dev set.
    logger.info("Subsampling dev set...")
    sampled_queries, sampled_qrels = subsample_dev_set(queries, qrels, sample_percentage=args.sample_dev_percentage, seed=args.seed)

    # Prepare validation samples.
    logger.info("Preparing validation samples...")
    validation_samples = prepare_training_samples_infonce(
        corpus,
        sampled_queries,
        sampled_qrels,
        n_per_query=args.n_per_query,
        hard_negative=args.hard_negative,
        index_name=args.index,
        index_type=args.index_type,
        query_encoder=args.query_encoder,
        hard_negatives_file=args.hard_negatives_file
    )
    logger.info(f"Total samples generated for validation set: {len(validation_samples)}")

    # Check if total sample count is devisible by n_per_query.
    if len(validation_samples) % args.n_per_query != 0:
        logger.warning(f"Total samples ({len(validation_samples)}) is not divisible by n_per_query ({args.n_per_query}).")

    # Print how many group of samples are generated.
    num_groups = len(validation_samples) // args.n_per_query
    logger.info(f"Number of groups of samples generated: {num_groups}")

    # Save samples to file.
    logger.info(f"Saving validation samples to {args.output_file}...")
    save_samples_to_file(validation_samples, args.output_file)
    logger.info("Validation set preparation completed.")

if __name__ == "__main__":
    """
    Example usage:
    python -m script.prepare_val_set \
    --dataset msmarco \
    --split dev \
    --index msmarco-v1-passage.bge-base-en-v1.5 \
    --index_type dense \
    --query_encoder BAAI/bge-base-en-v1.5 \
    --sample_dev_percentage 0.1 \
    --n_per_query 15 \
    --hard_negative \
    --output_file datasets/msmarco_val.jsonl
    """
    main()

