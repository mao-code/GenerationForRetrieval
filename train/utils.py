import random
from tqdm import tqdm
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
import torch

# Import Pyserini for retrieval.
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher

from utils import load_json_file

def process_prepared_sample(sample, n_per_query):
    """
    Convert a single prepared sample (with keys "query", "pos", "neg") into a group of training samples.
    The positive sample is taken from the first element of "pos".
    For negatives, if there are not enough, duplicate randomly.
    Returns a list of dictionaries with keys "query_text", "doc_text", and "label".
    """
    # Skip if no positive or no negatives.
    if not sample.get("pos") or not sample.get("neg"):
        return []
    
    query_text = sample["query"]
    pos_text = sample["pos"][0]
    negatives = sample["neg"].copy()  # copy to avoid modifying the original list

    if len(negatives) < n_per_query:
        # Duplicate negatives if fewer than required.
        while len(negatives) < n_per_query:
            negatives.append(random.choice(negatives))
    else:
        negatives = negatives[:n_per_query]
    
    group_samples = []
    # Positive sample first.
    group_samples.append({
        "query_text": query_text,
        "doc_text": pos_text,
        "label": 1.0
    })
    for neg in negatives:
        group_samples.append({
            "query_text": query_text,
            "doc_text": neg,
            "label": 0.0
        })
    return group_samples

def load_prepared_samples(file_paths, sample_counts, n_per_query, logger):
    """
    Given a list of file paths and corresponding sample counts (in the desired order),
    load and process each prepared sample.
    Returns a list where each query group is represented as consecutive samples
    (first positive then negatives).
    """
    all_samples = []
    for file_path, sample_count in zip(file_paths, sample_counts):
        data = load_json_file(file_path)

        if sample_count <=0:
            sample_count = len(data)

        if len(data) < sample_count:
            logger.warning(f"File {file_path} has fewer samples than requested. Using all available samples.")
            sample_count = len(data)

        logger.info(f"Loading prepared samples from {file_path} with sample count {sample_count}")
        # Use the first sample_count samples (preserving order)
        if len(data) > sample_count:
            data = data[:sample_count]
        for sample in tqdm(data, desc=f"Processing {file_path}"):
            group = process_prepared_sample(sample, n_per_query)
            if group:
                all_samples.extend(group)
    return all_samples

def compute_hard_negatives_for_query(qid, query, qrels, searcher, all_doc_ids):
    """
    Given a query id and query text, compute the hard negatives using the searcher.
    """
    hits = searcher.search(query, k=100)
    doc_ids = [hit.docid for hit in hits]
    candidate_negatives = [doc_id for doc_id in doc_ids if doc_id not in qrels[qid]]
    if candidate_negatives:
        return qid, candidate_negatives
    else:
        # Fallback: choose a random negative not in the qrels
        neg_doc_id = random.choice(all_doc_ids)
        while neg_doc_id in qrels[qid]:
            neg_doc_id = random.choice(all_doc_ids)
        return qid, [neg_doc_id]

def prepare_training_samples_infonce(
    corpus: dict,
    queries: dict,
    qrels: dict,
    n_per_query: int = 7,
    hard_negative: bool = False,
    index_name: str = "msmarco-v1-passage.bge-base-en-v1.5",
    index_type: str = "dense",
    query_encoder: str = "BAAI/bge-base-en-v1.5",
    hard_negatives_file: str = None
):
    training_samples = []
    all_doc_ids = list(corpus.keys())

    # Set default file name for caching if not provided.
    if hard_negatives_file is None:
        hard_negatives_file = f"hard_negatives_{index_name}.pkl"

    # Load cached hard negatives if available.
    if os.path.exists(hard_negatives_file):
        print(f"Loading hard negatives from {hard_negatives_file} (locally)")
        with open(hard_negatives_file, "rb") as f:
            hard_negatives = pickle.load(f)
    else:
        print(f"Computing hard negatives using index {index_name} (not found locally)")

        if index_type == "dense":
            searcher = FaissSearcher.from_prebuilt_index(index_name, query_encoder)
        elif index_type == "sparse":
            searcher = LuceneSearcher.from_prebuilt_index(index_name)
        else:  
            raise ValueError(f"Unsupported index type: {index_type}. Use 'dense' or 'sparse'.")
    
        hard_negatives = {}
        # Process queries in parallel using ThreadPoolExecutor.
        qids = [qid for qid in qrels if qid in queries]
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    compute_hard_negatives_for_query,
                    qid,
                    queries[qid]['text'] if isinstance(queries[qid], dict) else queries[qid],
                    qrels,
                    searcher,
                    all_doc_ids,
                ): qid for qid in qids
            }
            for future in tqdm(futures, total=len(futures), desc=f"Precomputing hard negatives using index {index_name}"):
                qid, negatives = future.result()
                hard_negatives[qid] = negatives
        
        # Cache the computed hard negatives.
        with open(hard_negatives_file, "wb") as f:
            pickle.dump(hard_negatives, f)

    # Process queries to prepare training samples.
    for qid, rel_docs in tqdm(qrels.items(), total=len(qrels), desc="Processing queries"):
        if qid not in queries:
            continue
        
        query_text = queries[qid]['text'] if isinstance(queries[qid], dict) else queries[qid]
        pos_doc_ids = [doc_id for doc_id, score in rel_docs.items() if score > 0]
        if not pos_doc_ids:
            continue
        
        for pos_doc_id in pos_doc_ids:
            # Select negatives.
            if hard_negative and qid in hard_negatives:
                candidate_negatives = hard_negatives[qid]
                if len(candidate_negatives) >= n_per_query:
                    neg_doc_ids = random.sample(candidate_negatives, n_per_query)
                else:
                    neg_doc_ids = candidate_negatives.copy()
                    while len(neg_doc_ids) < n_per_query:
                        neg_doc_ids.append(random.choice(candidate_negatives))
            else:
                neg_doc_ids = []
                while len(neg_doc_ids) < n_per_query:
                    neg_doc_id = random.choice(all_doc_ids)
                    if neg_doc_id not in rel_docs and neg_doc_id not in neg_doc_ids:
                        neg_doc_ids.append(neg_doc_id)
            
            # Append positive sample.
            training_samples.append({
                'query_text': query_text,
                'doc_text': corpus[pos_doc_id]['text'],
                'label': 1.0
            })

            # Append negative samples.
            for neg_doc_id in neg_doc_ids:
                training_samples.append({
                    'query_text': query_text,
                    'doc_text': corpus[neg_doc_id]['text'],
                    'label': 0.0
                })
                
    return training_samples

def subsample_dev_set(queries_dev: dict, qrels_dev: dict, sample_percentage: float = 0.1, seed=42):
    random.seed(seed)

    dev_query_ids = list(queries_dev.keys())
    num_sample = max(1, int(len(dev_query_ids) * sample_percentage))
    sampled_ids = random.sample(dev_query_ids, num_sample)
    
    sampled_queries = {qid: queries_dev[qid] for qid in sampled_ids}
    sampled_qrels = {qid: qrels_dev[qid] for qid in sampled_ids if qid in qrels_dev}
    
    return sampled_queries, sampled_qrels

def log_training_config(training_args, logger=None):
    # Determine the number of GPUs being used.
    if torch.distributed.is_initialized():
        num_gpus = torch.distributed.get_world_size()
    else:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Calculate the effective (global) batch size.
    per_device_bs = training_args.per_device_train_batch_size
    grad_accum_steps = training_args.gradient_accumulation_steps
    effective_batch_size = per_device_bs * grad_accum_steps * num_gpus

    # Log the configuration details.
    logger.info("==============================================")
    logger.info("Training Configuration:")
    logger.info(f"  Number of GPUs utilized         : {num_gpus}")
    logger.info(f"  Per-device batch size           : {per_device_bs}")
    logger.info(f"  Gradient accumulation steps     : {grad_accum_steps}")
    logger.info(f"  Effective (global) batch size   : {effective_batch_size}")
    logger.info("==============================================")

def prepare_training_samples_bce(
    corpus: dict,
    queries: dict,
    qrels: dict,
    n_per_query: int = 5,
    hard_negative: bool = False,
    index_name: str = "msmarco-v1-passage"
):
    """
    Creates training sample tuples: (query_text, doc_text, label)
    where label is 1.0 for relevant docs and 0.0 for negatives.

    For each query, this function:
      - Selects up to n_per_query positive documents (shuffled if more than n available).
      - Selects up to n_per_query negative documents. If hard_negative is enabled, it uses the provided index
        (via LuceneSearcher) to retrieve candidate negatives; otherwise, negatives are sampled randomly.
    """
    training_samples = []
    all_doc_ids = list(corpus.keys())
    searcher = LuceneSearcher.from_prebuilt_index(index_name)

    # Precompute hard negatives for each query.
    hard_negatives = {}
    for qid in tqdm(qrels, desc=f"Precomputing hard negatives using index {index_name}"):
        if qid not in queries:
            continue
        
        query = queries[qid]
        # Handle different query formats
        if isinstance(query, str):
            query_text = query
        elif isinstance(query, dict) and 'text' in query:
            query_text = query['text']  # Extract the query text
        else:
            print(f"Skipping qid {qid}: Invalid query format {query}")
            continue

        hits = searcher.search(query_text, k=100)
        doc_ids = [hit.docid for hit in hits]
        candidate_negatives = [doc_id for doc_id in doc_ids if doc_id not in qrels[qid]]
        if candidate_negatives:
            hard_negatives[qid] = candidate_negatives
        else:
            # Fallback: sample one negative document randomly if no candidate is found.
            neg_doc_id = random.choice(all_doc_ids)
            while neg_doc_id in qrels[qid]:
                neg_doc_id = random.choice(all_doc_ids)
            hard_negatives[qid] = [neg_doc_id]

    # Process each query.
    for qid, rel_docs in tqdm(qrels.items(), total=len(qrels), desc="Processing queries"):
        if qid not in queries:
            continue
        query_text = queries[qid]
        # Get all positive doc IDs for this query.
        pos_doc_ids = [doc_id for doc_id, score in rel_docs.items() if score > 0]
        if not pos_doc_ids:
            continue
        # random.shuffle(pos_doc_ids)

        pos_n_per_query = min(n_per_query, len(pos_doc_ids))
        pos_samples = pos_doc_ids[:pos_n_per_query]
        for pos_doc_id in pos_samples:
            training_samples.append((query_text, corpus[pos_doc_id]['text'], 1.0))
        
        # Sample negative docs.
        if hard_negative and qid in hard_negatives:
            candidate_negatives = hard_negatives[qid]
            if len(candidate_negatives) >= n_per_query:
                neg_samples = random.sample(candidate_negatives, n_per_query)
            else:
                neg_samples = candidate_negatives.copy()
                while len(neg_samples) < n_per_query:
                    neg_samples.append(random.choice(candidate_negatives))
        else:
            neg_samples = []
            while len(neg_samples) < n_per_query:
                neg_doc_id = random.choice(all_doc_ids)
                if neg_doc_id not in rel_docs and neg_doc_id not in neg_samples:
                    neg_samples.append(neg_doc_id)
        for neg_doc_id in neg_samples:
            training_samples.append((query_text, corpus[neg_doc_id]['text'], 0.0))
    
    return training_samples