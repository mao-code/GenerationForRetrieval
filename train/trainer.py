from transformers import Trainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class DocumentRankingTrainer(Trainer):
    def __init__(self, n_per_query, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.n_per_query = n_per_query
        self.tokenizer = tokenizer

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            drop_last=True, # To ensure all samples is divisible by group_size
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def get_eval_dataloader(self, eval_dataset=None, **kwargs):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            drop_last=True, # To ensure all samples is divisible by group_size
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
        return dataloader

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")  # Not used in this loss
        outputs = model(**inputs, return_dict=True)
        logits = outputs["logits"].view(-1)
        
        group_size = 1 + self.n_per_query

        if len(logits) % group_size != 0:
            raise ValueError(f"Batch size {len(logits)} must be a multiple of {group_size}")
        
        N_groups = len(logits) // group_size
        
        logits = logits.view(N_groups, group_size)

        targets = torch.zeros(N_groups, dtype=torch.long, device=logits.device) # Target loss to 0 at the first position.
        
        # Temperature parameter, can be tuned.
        # BGE: 0.01
        # Try range: 0.01 ~ 0.1
        tau = 0.01
        logits = logits / tau
        
        loss = nn.CrossEntropyLoss()(logits, targets)
        return (loss, outputs) if return_outputs else loss

class DocumentRankingBCETrainer(Trainer):
    def __init__(self, loss_fn, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, return_dict=True)
        logits = outputs["logits"].view(-1)
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss
