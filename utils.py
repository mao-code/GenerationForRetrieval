import os
import logging
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import json

def load_dataset(logger, dataset: str, split: str):
    """Loads a BEIR dataset and prefixes ids with the dataset name."""
    out_dir = "datasets"
    data_path = os.path.join(out_dir, dataset)
    if not os.path.exists(data_path):
        logger.info(f"Dataset '{dataset}' not found locally. Downloading...")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        data_path = util.download_and_unzip(url, out_dir)
    else:
        logger.info(f"Dataset '{dataset}' found locally. Skipping download.")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    
    # Prefix each doc_id in corpus, and update queries and qrels accordingly.
    # new_corpus = {f"{dataset}_{doc_id}": content for doc_id, content in corpus.items()}
    # new_queries = {f"{dataset}_{qid}": query for qid, query in queries.items()}
    # new_qrels = {f"{dataset}_{qid}": {f"{dataset}_{doc_id}": score for doc_id, score in rels.items()}
    #              for qid, rels in qrels.items()}
    
    return corpus, queries, qrels

def load_json_file(file_path):
    """
    Load a JSON or JSONL file.
    """
    ext = os.path.splitext(file_path)[1]
    if ext == '.jsonl':
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    elif ext == '.json':
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            try:
                # Try to parse as a single JSON structure
                return json.loads(content)
            except json.decoder.JSONDecodeError:
                # Fallback: treat as JSON Lines if multiple JSON objects are found
                data = []
                for line in content.splitlines():
                    if line.strip():
                        data.append(json.loads(line))
                return data
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
def save_samples_to_file(samples, output_file):
    """Save the samples to a JSON or JSONL file based on the extension."""
    ext = os.path.splitext(output_file)[1]
    if ext == ".jsonl":
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
    elif ext == ".json":
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)
    else:
        raise ValueError("Unsupported file extension. Use .json or .jsonl")

class MainProcessFilter(logging.Filter):
    def __init__(self, local_rank):
        super().__init__()
        self.local_rank = local_rank

    def filter(self, record):
        return self.local_rank == 0