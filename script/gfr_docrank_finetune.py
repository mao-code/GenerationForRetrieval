import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from transformers import BertTokenizer
from GFR.configuration_GFR import GFRConfig
from GFR.modeling_GFR import GFRModel
import wandb
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from utils import load_dataset, prepare_training_samples, train_one_epoch, evaluate_full_retrieval
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training.")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for full retrieval evaluation.")
    parser.add_argument("--datasets", type=str, nargs='+', default=["msmarco", "hotpotqa"],
                        help="List of datasets to use (e.g., msmarco hotpotqa nq).")
    args = parser.parse_args()

    # Set up device and distributed training if enabled.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = 0
    if args.distributed:
        dist.init_process_group(backend="nccl")
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
        logging.info(f"Distributed training enabled. Local rank: {local_rank}")

    # Initialize WandB.
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.login(key="your_api_key_here")
    wandb.init(project="GFR_Document_Ranking", entity="your_account_name", name=run_name, config=vars(args))

    # Load the BERT tokenizer.
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Initialize GFR model.
    config = GFRConfig(
        vocab_size=len(tokenizer),
        hidden_size=1024,        # BGE-reranker: 4096
        intermediate_size=4096,  # BGE-reranker: 11008

        num_attention_heads=12,
        num_key_value_heads=12,
        n_mamba_heads=2,

        num_hidden_block=3,
        num_layers_per_block=8,
        max_position_embeddings=512,
    )
    model = GFRModel(config)
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Set up loss and optimizer.
    loss_fn = nn.MarginRankingLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Load datasets for training, dev, and test splits.
    training_samples = []
    dev_data = {}   # dictionary: dataset -> (corpus, queries, qrels)
    test_data = {}  # dictionary: dataset -> (corpus, queries, qrels)
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

    logging.info(f"Total training samples: {len(training_samples)}")
    for dataset in args.datasets:
        logging.info(f"Dataset {dataset} dev queries: {len(dev_data[dataset][1])}")
        logging.info(f"Dataset {dataset} test queries: {len(test_data[dataset][1])}")

    checkpoint_dir = "checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop.
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch+1}")
        epoch_loss = train_one_epoch(model, optimizer, loss_fn, training_samples, tokenizer, device, args.batch_size, epoch, args.grad_accum_steps)
        logging.info(f"Epoch {epoch+1} training loss: {epoch_loss:.4f}")
        wandb.log({"epoch": epoch+1, "training_loss": epoch_loss})

        # Evaluate each dev dataset individually.
        for dataset, (corpus_dev, queries_dev, qrels_dev) in dev_data.items():
            dev_metrics = evaluate_full_retrieval(model, corpus_dev, queries_dev, qrels_dev, tokenizer, device, batch_size=args.eval_batch_size)
            logging.info(f"Epoch {epoch+1} dev metrics for {dataset}: {dev_metrics}")
            wandb.log({f"epoch_{epoch+1}_{dataset}_dev": dev_metrics})

        # Save checkpoint (only on main process if distributed).
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
            # wandb.save(checkpoint_path)
            logging.info(f"Checkpoint saved at {checkpoint_path}")

    # Evaluate on test data for each dataset after training.
    for dataset, (corpus_test, queries_test, qrels_test) in test_data.items():
        test_metrics = evaluate_full_retrieval(model, corpus_test, queries_test, qrels_test, tokenizer, device, batch_size=args.eval_batch_size)
        logging.info(f"Test metrics for {dataset}: {test_metrics}")
        wandb.log({f"{dataset}_test_metrics": test_metrics})

    # Save the final model checkpoint.
    checkpoint_path = os.path.join(checkpoint_dir, f"GFRModel_{run_name}_epoch_{args.epochs:02d}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    # wandb.save(checkpoint_path)
    logging.info("Training completed and model checkpoint saved.")
    wandb.finish()

if __name__ == "__main__":
    main()

"""
Example usage:

Distributed:
python -m torch.distributed.run --nproc_per_node=<NUM_GPUS> -m script.gfr_train_eval --distributed --epochs 3 --batch_size 4 --lr 1e-4 --eval_batch_size 4 --datasets msmarco hotpotqa

python -m script.gfr_docrank_finetune \
    --epochs 3 \
    --batch_size 2 \
    --grad_accum_steps 4 \
    --lr 1e-4 \
    --eval_batch_size 2 \
    --datasets msmarco hotpotqa
"""