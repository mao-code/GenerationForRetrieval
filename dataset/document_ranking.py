from torch.utils.data import Dataset
import torch

class DocumentRankingDataset(Dataset):
    def __init__(self, samples, tokenizer, model):
        # samples is a list of tuples: (query_text, doc_text, label)
        self.samples = samples
        self.tokenizer = tokenizer
        self.model = model

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Extract query, doc, and label from the sample
        sample = self.samples[idx]
        # query, doc, label = self.samples[idx]
        query, doc, label = sample['query_text'], sample['doc_text'], sample['label']

        # Tokenize and prepare input tensors
        # Assuming model.prepare_input takes lists of docs and queries
        input_ids, token_type_ids, attention_mask = self.model.prepare_input(
            [doc], [query], self.tokenizer
        )

        # Return a dictionary with preprocessed tensors and label
        return {
            "input_ids": input_ids.squeeze(0),  # Remove batch dimension
            "token_type_ids": token_type_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float32)  # Use "labels" for Trainer
        }