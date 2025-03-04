import os
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from transformers import BertTokenizer

# Import your model configuration and model code.
from GFR.configuration_GFR import GFRConfig
from GFR.modeling_GFR import GFRModel

# For WandB logging.
import wandb

# BEIR API imports.
from beir import util
from beir.datasets.data_loader import GenericDataLoader

# For training and evaluation functions
from utils import load_dataset, prepare_training_samples, train_one_epoch, evaluate
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training.")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Number of samples for evaluation metrics.")
    # New argument: list of datasets to use for training
    parser.add_argument("--datasets", type=str, nargs='+', default=["msmarco", "hotpotqa"],
                        help="List of training datasets to use (e.g., msmarco hotpotqa nq).")
    args = parser.parse_args()

    # Set up device and distributed training if enabled.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = 0
    if args.distributed:
        dist.init_process_group(backend="nccl")
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        logging.info(f"Distributed training enabled. Local rank: {local_rank}")

    # Initialize WandB for logging.
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.login(key="your_api_key_here")
    wandb.init(project="GFR_Document_Ranking", entity="your_account_name", name=run_name, config=vars(args))

    # Load the BERT tokenizer.
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Initialize GFR model.
    config = GFRConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,         # smaller hidden size for demo purposes
        num_hidden_block=3,
        num_layers_per_block=8,
        max_position_embeddings=512,
    )
    model = GFRModel(config)
    model.to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Set up the loss function.
    loss_fn = nn.MarginRankingLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Load and combine datasets for training, validation, and testing.
    training_samples = []
    validation_samples = []
    test_samples = []
    for dataset in args.datasets:
        logging.info(f"Loading dataset: {dataset} (train split)")
        corpus_train, queries_train, qrels_train = load_dataset(dataset, split="train")
        training_samples.extend(prepare_training_samples(corpus_train, queries_train, qrels_train))

        logging.info(f"Loading dataset: {dataset} (dev split)")
        corpus_dev, queries_dev, qrels_dev = load_dataset(dataset, split="dev")
        validation_samples.extend(prepare_training_samples(corpus_dev, queries_dev, qrels_dev))

        logging.info(f"Loading dataset: {dataset} (test split)")
        corpus_test, queries_test, qrels_test = load_dataset(dataset, split="test")
        test_samples.extend(prepare_training_samples(corpus_test, queries_test, qrels_test))

    logging.info(f"Total training samples: {len(training_samples)}")
    logging.info(f"Total validation samples: {len(validation_samples)}")
    logging.info(f"Total test samples: {len(test_samples)}")

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop.
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch+1}")
        epoch_loss = train_one_epoch(model, optimizer, loss_fn, training_samples, tokenizer, device, args.batch_size, epoch)
        logging.info(f"Epoch {epoch+1} training loss: {epoch_loss:.4f}")
        wandb.log({"epoch": epoch+1, "training_loss": epoch_loss})

        # Run evaluation on validation data.
        val_metrics = evaluate(model, validation_samples, tokenizer, device, args.eval_batch_size)
        logging.info(f"Epoch {epoch+1} validation metrics: {val_metrics}")
        wandb.log({f"epoch_{epoch+1}_validation": val_metrics})

        # Save checkpoint at the end of each epoch (only on main process if distributed)
        if not args.distributed or local_rank == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }
            
            checkpoint_filename = f"GFRModel_{run_name}_epoch_{epoch+1:02d}.pth"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
            torch.save(checkpoint, checkpoint_path)
            wandb.save(checkpoint_path)
            logging.info(f"Checkpoint saved at {checkpoint_path}")

    # Evaluate on test data after training.
    test_metrics = evaluate(model, test_samples, tokenizer, device, args.eval_batch_size)
    logging.info(f"Test metrics: {test_metrics}")
    wandb.log({"test_metrics": test_metrics})

    # Save the final model checkpoint.
    checkpoint_path = os.path.join(checkpoint_dir, f"GFRModel_{run_name}_epoch_{args.epochs:02d}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    wandb.save(checkpoint_path)
    logging.info("Training completed and model checkpoint saved.")

if __name__ == "__main__":
    main()

    """
    python -m script.gfr_train_eval \
        --epochs 3 \
        --batch_size 8 \
        --lr 1e-4 \
        --eval_batch_size 16 \
        --datasets msmarco hotpotqa
    """