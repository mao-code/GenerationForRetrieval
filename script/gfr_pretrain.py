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
from datasets import load_dataset
import datetime

from model import GFRForCausalLM, GFRConfig

# ========= Hyperparameters and Settings ========= #
# Standard training settings
learning_rate = 1e-4
batch_size = 2
gradient_accumulation_steps = 16
num_epochs = 3
max_seq_length = 1024  # maximum sequence length for the LM
warmup_steps = 1000    # number of warmup steps for the LR scheduler

# Total training steps: if you know the total number of training examples, you can compute this.
# For demonstration, we set it to a fixed value.
total_training_steps = 100000  

# Evaluation every N global steps
eval_steps = 1000

# Wandb settings
run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
wandb.login(key="your_api_key_here")
wandb_project = "gfr-pretrain"
wandb_entity = "your_group_name"

# ========= Initialize Wandb ========= #
wandb.init(project=wandb_project, entity=wandb_entity, name=run_name, config={
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "num_epochs": num_epochs,
    "max_seq_length": max_seq_length
})

# ========= Load Llama Tokenizer ========= #
# Use Llama's tokenizer from Hugging Face (adjust pretrained model identifier as needed)
tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# ========= Load Model ========= #
# Define your model configuration. Adjust the parameters as needed.
config = GFRConfig(
    vocab_size=len(tokenizer),
    hidden_size=1024,        
    intermediate_size=4096,  

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
# If you added tokens to the tokenizer, you might need to resize the model's embeddings.
model.gfr.embed_tokens.resize_token_embeddings(len(tokenizer))
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ========= Load and Preprocess the RedPajama Dataset ========= #
# Here we use a subset for demonstration purposes.
# For full pretraining, remove the slicing.
train_dataset = load_dataset("togethercomputer/redpajama_open", split="train[:1%]")
eval_dataset = load_dataset("togethercomputer/redpajama_open", split="train[:0.1%]")

def tokenize_function(example):
    # Assumes each example has a "text" field.
    return tokenizer(example["text"], truncation=True, max_length=max_seq_length)

# Tokenize both training and evaluation splits.
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ========= Data Collator for Causal LM ========= #
# For causal language modeling we do not use masked LM.
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# ========= Compute Total Training Steps ========= #
# Total number of training examples from the dataset length.
total_training_examples = len(train_dataset)
per_device_train_batch_size = 2
gradient_accumulation_steps = 16
num_train_epochs = 3

steps_per_epoch = math.ceil(total_training_examples / (per_device_train_batch_size * gradient_accumulation_steps))
total_training_steps = steps_per_epoch * num_train_epochs
print(f"Total training examples: {total_training_examples}")
print(f"Total training steps: {total_training_steps}")

# Optionally, define warmup steps as a fraction of total steps.
warmup_steps = int(0.1 * total_training_steps)

# ========= Define Training Arguments ========= #
training_args = TrainingArguments(
    output_dir="./gfr_pretrain_redpajama",
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    evaluation_strategy="steps",
    eval_steps=1000,
    logging_steps=100,
    learning_rate=1e-4,
    warmup_steps=warmup_steps,
    save_steps=1000,
    report_to="wandb",  # Log training to wandb.
    logging_dir="./logs",
)

# ========= Define a Compute Metrics Function ========= #
# For evaluation, compute loss and perplexity.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Shift logits and labels for next-token prediction.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = math.exp(loss.item()) if loss.item() < 20 else float("inf")
    return {"eval_loss": loss.item(), "perplexity": perplexity}

# ========= Initialize Trainer ========= #
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ========= Train and Evaluate ========= #
trainer.train()
results = trainer.evaluate()
print("Final Evaluation Results:", results)

# ========= Save the Final Model ========= #
trainer.save_model("./gfr_causal_lm_final_redpajama")
wandb.finish()