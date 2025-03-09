from beir import util
from beir.datasets.data_loader import GenericDataLoader
import logging
from utils import load_dataset
import itertools

def main():
    dataset = "msmarco"
    logging.info(f"Loading dataset: {dataset} (train split)")
    corpus_train, queries_train, qrels_train = load_dataset(dataset, split="train")
    logging.info(f"Loading dataset: {dataset} (train split)")

    # Print the first 10 query samples
    if queries_train:
        print("\nFirst 10 Queries:")
        for query_id in itertools.islice(queries_train, 10):
            print(f"ID: {query_id}, Text: {queries_train[query_id]}")

    # Print the first 10 corpus samples
    # if corpus_train:
    #     print("\nFirst 10 Corpus Documents:")
    #     for doc_id in itertools.islice(corpus_train, 10):
    #         print(f"ID: {doc_id}, Document: {corpus_train[doc_id]}")

    # Print the first 10 qrels samples
    if qrels_train:
        print("\nFirst 10 Qrels:")
        for qrel_id in itertools.islice(qrels_train, 10):
            print(f"ID: {qrel_id}, Entries: {qrels_train[qrel_id]}")


if __name__ == "__main__":
    main()