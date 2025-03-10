import os
import math
import torch
import wandb
from transformers import (
    Trainer,
    TrainingArguments,
    LlamaTokenizer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, Dataset
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
import numpy as np
from itertools import islice

from GFR.modeling_GFR import GFRForCausalLM, GFRConfig

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
        # Count tokens that are not padding tokens
        tokens_in_batch = (token_tensor != self.processing_class.pad_token_id).sum().item()
        self.total_tokens += tokens_in_batch

        # Run the usual training step
        return super().training_step(model, inputs, *args, **kwargs)

    def train(self, *args, **kwargs):
        # Run the standard training loop
        output = super().train(*args, **kwargs)
        # Log the total tokens processed at the end of training
        logging.info(f"Total tokens processed during training: {self.total_tokens}")
        # If you use wandb or another logger, you can log here as well:
        self.log({"total_tokens_processed": self.total_tokens})
        return output

# ========= Hyperparameters and Settings ========= #
# Standard training settings
learning_rate = 1e-4
per_device_train_batch_size = 2
gradient_accumulation_steps = 16
per_device_eval_batch_size = 1
eval_accumulation_steps=2
num_train_epochs = 1
max_seq_length = 1024  # maximum sequence length for the LM

# Wandb settings
run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
wandb.login(key="your_api_key_here")
wandb_project = "gfr-pretrain"
wandb_entity = "your_group_name"

# ========= Initialize Wandb ========= #
wandb.init(project=wandb_project, entity=wandb_entity, name=run_name, config={
    "learning_rate": learning_rate,
    "per_device_train_batch_size": per_device_train_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "num_train_epochs": num_train_epochs,
    "max_seq_length": max_seq_length
})

# ========= Load Llama Tokenizer ========= #
# Use Llama's tokenizer from Hugging Face (adjust pretrained model identifier as needed)
tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# ========= Load Model ========= #
# Define your model configuration. Adjust the parameters as needed.
logging.info("Initializing GFR model for pretraining...")
config = GFRConfig(
    vocab_size=len(tokenizer),
    hidden_size=max_seq_length,        
    intermediate_size=max_seq_length*4,  

    num_attention_heads=16,
    num_key_value_heads=16,
    n_mamba_heads=2,

    num_hidden_block=3,
    num_layers_per_block=8,
    max_position_embeddings=512,

    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = GFRForCausalLM(config)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
num_params = sum(p.numel() for p in model.parameters())
logging.info(f"Number of parameters: {num_params}")

# ========= Load and Preprocess the RedPajama Dataset ========= #
logging.info("Loading RedPajama dataset...")

# Define how many examples to use for evaluation.
eval_size = 100

# For evaluation, load the dataset in streaming mode and materialize the first eval_size examples.
raw_eval_stream = load_dataset(
    "togethercomputer/RedPajama-Data-V2", 
    name="default",
    languages=["en"],
    split="train",
    streaming=True,
    trust_remote_code=True
)
eval_examples = list(islice(raw_eval_stream, eval_size))
logging.info(f"Loaded {len(eval_examples)} evaluation examples.")

# For training, reload the streaming dataset and skip the first eval_size examples.
raw_train_stream = load_dataset(
    "togethercomputer/RedPajama-Data-V2", 
    name="default",
    languages=["en"],
    split="train",
    streaming=True,
    trust_remote_code=True
)
# Skip evaluation examples.
raw_train_stream = raw_train_stream.skip(eval_size)

def tokenize_function(example):
    # Redpajama dataset has a "raw_content" field.
    return tokenizer(example["raw_content"], truncation=True, max_length=max_seq_length)

# Materialize and tokenize the evaluation set.
eval_dataset = Dataset.from_list(eval_examples)
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["raw_content"])

val_test_splits = eval_dataset.train_test_split(test_size=0.5, seed=42)
validation_dataset = val_test_splits["train"]
test_dataset = val_test_splits["test"]

# Apply tokenization in a batched fashion.
train_dataset = raw_train_stream.map(tokenize_function, batched=True)

# ========= Data Collator for Causal LM ========= #
# For causal language modeling we do not use masked LM.
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ========= Set Total Training Steps ========= #
# Since the training dataset is streaming (no fixed length), define the desired number of training examples.
max_training_examples = 1000000  # adjust as needed
steps_per_epoch = math.ceil(max_training_examples / (per_device_train_batch_size * gradient_accumulation_steps))
total_training_steps = steps_per_epoch * num_train_epochs
print(f"Total training steps (approx): {total_training_steps}")

# Optionally, define warmup steps as a fraction of total steps.
warmup_steps = int(0.1 * total_training_steps)

# ========= Define Training Arguments ========= #
training_args = TrainingArguments(
    output_dir="./gfr_pretrain_redpajama_v2",
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    max_steps=total_training_steps,  # explicitly stop training after these many steps
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    eval_strategy="steps",
    eval_steps=1000,
    eval_accumulation_steps=eval_accumulation_steps,
    logging_steps=50,
    logging_first_step=True,
    learning_rate=learning_rate,
    warmup_steps=int(0.1 * total_training_steps),
    save_steps=10000,
    report_to="wandb",  # log training metrics to wandb
    logging_dir="./logs",
)

# ========= Define a Compute Metrics Function ========= #
# For evaluation, compute loss and perplexity.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert from numpy arrays to torch tensors if necessary.
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
        labels = torch.from_numpy(labels)
    # Shift logits and labels for next-token prediction.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # Set ignore_index to -100 to match the labels produced by the data collator.
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = math.exp(loss.item()) if loss.item() < 20 else float("inf")
    return {"eval_loss": loss.item(), "perplexity": perplexity}

# ========= Initialize Trainer ========= #
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # streaming training dataset (an iterator)
    eval_dataset=eval_dataset,    # materialized evaluation dataset
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ========= Train and Evaluate ========= #
logging.info("Starting training...")
trainer.train()
results = trainer.evaluate(eval_dataset=test_dataset)
logging.info("Final Evaluation Results:", results)

# ========= Save the Final Model ========= #
trainer.save_model("./gfr_causal_lm_final_redpajama_v2")
wandb.finish()

"""
python -m script.gfr_pretrain
"""