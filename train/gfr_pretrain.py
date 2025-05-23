import os
import math
import torch
import wandb
import argparse
from datetime import datetime
import logging
import numpy as np
from itertools import islice

from transformers import (
    Trainer,
    TrainingArguments,
    LlamaTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback
)
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

from GFR.modeling_GFR import GFRForCausalLM 
from GFR.configuration_GFR import GFRConfig
from GFR.tokenizer_utils import get_tokenizer

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_tokens = 0
        self.processing_class = getattr(self.data_collator, "tokenizer", None)
        if self.processing_class is None:
            raise ValueError("No tokenizer found on the data_collator. Please provide a valid tokenizer.")
        
    def training_step(self, model, inputs, *args, **kwargs):
        # Count tokens in the batch (ignoring pad tokens)
        token_tensor = inputs["input_ids"]
        tokens_in_batch = (token_tensor != self.processing_class.pad_token_id).sum().item()
        self.total_tokens += tokens_in_batch

        # Run the usual training step
        return super().training_step(model, inputs, *args, **kwargs)

    def train(self, *args, **kwargs):
        output = super().train(*args, **kwargs)
        logging.info(f"Total tokens processed during training: {self.total_tokens}")
        self.log({"total_tokens_processed": self.total_tokens})
        return output

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
        labels = torch.from_numpy(labels)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = math.exp(loss.item()) if loss.item() < 20 else float("inf")
    return {"eval_loss": loss.item(), "perplexity": perplexity}

class QualitativeGenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, prompts=None, max_length=50):
        # If no custom prompts are provided, use these 10 examples:
        if prompts is None:
            prompts = [
                "The meaning of life is",
                "Once upon a time in a distant land",
                "In the future, artificial intelligence will",
                "The secret to success is",
                "In a shocking discovery, scientists found",
                "Deep in the forest, an ancient secret was hidden",
                "The history of the universe began when",
                "The best way to solve a problem is",
                "A mysterious event occurred when",
                "In the year 3000, humans will"
            ]
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        # Unwrap the model if it is wrapped (e.g., DDP or Accelerate wrapper)
        if hasattr(model, "module"):
            model = model.module

        # Force the model to full precision (fp32) to ensure consistency.
        model = model.float()

        # Tokenize the prompts and move them to the model's device.
        inputs = self.tokenizer(self.prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Disable autocast to ensure consistent dtypes during generation
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=False):
                outputs = model.generate(**inputs, max_length=self.max_length, use_cache=True)

        # Decode the outputs to human-readable text.
        decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        # Also print the outputs for immediate feedback.
        logging.info("Qualitative Generation Outputs:")
        for prompt, generated in zip(self.prompts, decoded_outputs):
            logging.info(f"Prompt: {prompt}\nGenerated: {generated}\n")

def main():
    # Training arguments
    parser = argparse.ArgumentParser(description="Pretrain GFR Model with Custom Arguments")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--grad_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--target_tokens", type=int, default=10e9, help="Target number of tokens to train on (recommended 20x model parameters according to scaling laws)")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")

    # Evaluation arguments
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Per-device evaluation batch size")
    parser.add_argument("--eval_accumulation_steps", type=int, default=1, help="Evaluation accumulation steps")
    parser.add_argument("--eval_size", type=int, default=100, help="Number of examples to use for evaluation")

    # Model arguments
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length for the LM")
    parser.add_argument("--num_blocks", type=int, default=3, help="Number of hidden blocks in the model")

    # Logging and output arguments
    parser.add_argument("--output_dir", type=str, default="./gfr_pretrain_finewebedu", help="Output directory for model checkpoints")
    parser.add_argument("--save_model_path", type=str, default="gfr_causal_lm_final_finewebedu_v2", help="Name of the final model to save")
    parser.add_argument("--run_name", type=str, default="", help="Run name for logging")

    args = parser.parse_args()

    """
    For wandb logging, set the following environment variables:

    export WANDB_API_KEY="your_wandb_api_key"
    export WANDB_PROJECT="gfr2_pretrain"
    export WANDB_ENTITY="nlp-maocode"
    """

    # ========= Hyperparameters and Settings ========= #
    learning_rate = 1e-4
    per_device_train_batch_size = args.batch_size
    gradient_accumulation_steps = args.grad_accumulation_steps
    per_device_eval_batch_size = args.per_device_eval_batch_size
    eval_accumulation_steps = args.eval_accumulation_steps
    num_train_epochs = args.num_train_epochs
    max_seq_length = args.max_seq_length

    # Wandb settings
    now_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.run_name + "_" + now_datetime

    # ========= Load Llama Tokenizer ========= #
    tokenizer = get_tokenizer()

    # ========= Load Model ========= #
    logging.info("Initializing GFR model for pretraining...")
    config = GFRConfig(
        vocab_size=len(tokenizer), # vocab size of the tokenizer
        hidden_size=1024,
        intermediate_size=1024 * 4,
        num_attention_heads=16,
        num_key_value_heads=16,
        n_mamba_heads=2,
        num_hidden_blocks=args.num_blocks,  # use value from argument parser
        num_layers_per_block=8,
        max_position_embeddings=1024, # no real effect
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GFRForCausalLM(config)
    model.resize_token_embeddings(len(tokenizer))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Number of parameters: {num_params}")

    # ========= Load and Preprocess the FineWeb-Edu Dataset ========= #
    logging.info("Loading FineWeb-Edu dataset...")

    # Copied from https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
    # New version to avoid wasting tokens (according to the recommendation from HuggingFace)
    def tokenize_and_chunk_function(batch):
        """
        Tokenizes a batch of texts by concatenating them with an eos token in between,
        then chunks the concatenated token list into segments of fixed length (max_seq_length).
        """
        # List to hold all token IDs
        all_tokens = []
        
        # Process each example in the batch:
        # 1. Tokenize without truncation or padding to get the full sequence.
        # 2. Append the tokens and an eos token id at the end.
        for text in batch["text"]:
            tokens = tokenizer(text, truncation=False, padding=False)['input_ids']
            all_tokens.extend(tokens + [tokenizer.eos_token_id])
        
        # Compute how many tokens we can use to form full chunks of size `max_seq_length`
        total_length = (len(all_tokens) // max_seq_length) * max_seq_length
        
        # Split the tokens into chunks of `max_seq_length`
        input_ids = [all_tokens[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        
        # Create an attention mask (all ones, since there is no padding in these full chunks)
        attention_masks = [[1] * max_seq_length for _ in input_ids]
        
        return {"input_ids": input_ids, "attention_mask": attention_masks}
    
    eval_size = args.eval_size
    raw_eval_stream = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    eval_examples = list(islice(raw_eval_stream, eval_size))
    logging.info(f"Loaded {len(eval_examples)} evaluation examples.")

    eval_dataset = Dataset.from_list(eval_examples)
    eval_dataset = eval_dataset.map(tokenize_and_chunk_function, batched=True, remove_columns=raw_eval_stream.column_names)

    val_test_splits = eval_dataset.train_test_split(test_size=0.5, seed=42)
    validation_dataset = val_test_splits["train"]
    test_dataset = val_test_splits["test"]

    # Fineweb-Edu dataset shows its high quality and effectiveness
    raw_train_stream = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    raw_train_stream = raw_train_stream.skip(eval_size)

    train_dataset = raw_train_stream.map(tokenize_and_chunk_function, batched=True, remove_columns=raw_train_stream.column_names)

    # ========= Data Collator for Causal LM ========= #
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ========= Set Total Training Steps ========= #
    target_tokens = args.target_tokens
    max_training_examples = int(target_tokens / max_seq_length)
    logging.info(f"Setting max_training_examples to {max_training_examples} documents to target {target_tokens} tokens.")

    steps_per_epoch = math.ceil(max_training_examples / (per_device_train_batch_size * gradient_accumulation_steps))
    total_training_steps = steps_per_epoch * num_train_epochs
    logging.info(f"Total training steps (approx): {total_training_steps}")

    warmup_steps = int(0.1 * total_training_steps)

    # ========= Define Training Arguments ========= #
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        max_steps=total_training_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=True,
        eval_strategy="steps",
        eval_steps=1000,
        eval_accumulation_steps=eval_accumulation_steps,
        logging_steps=50,
        logging_first_step=True,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        save_steps=10000,
        report_to="wandb",
        run_name=run_name,
        logging_dir="./logs_pretrain",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # ========= Initialize Trainer ========= #
    qualitative_callback = QualitativeGenerationCallback(tokenizer=tokenizer, max_length=50)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), qualitative_callback]
    )

    # ========= Train and Evaluate ========= #
    logging.info("Starting training...")
    trainer.train()
    results = trainer.evaluate(eval_dataset=test_dataset)
    logging.info("Final Evaluation Results: %s", results)

    # ========= Save the Final Model ========= #
    trainer.save_model(args.save_model_path)
    wandb.finish()

if __name__ == "__main__":
    main()

    """
    Example usage:

    python -m train.gfr_pretrain \
        --batch_size 4 \
        --grad_accumulation_steps 32 \
        --target_tokens 10000000000 \
        --num_train_epochs 1 \
        --per_device_eval_batch_size 2 \
        --eval_accumulation_steps 1 \
        --eval_size 100 \
        --max_seq_length 1024 \
        --num_blocks 3 \
        --output_dir ./gfr_pretrain_finewebedu_500m \
        --save_model_path gfr_pretrain_causal_lm_final_finewebedu_v2_500m \
        --run_name "fineweb10B_model500M"
    """