from huggingface_hub import snapshot_download
import argparse
import os

if __name__ == "__main__":
    """
    Example usage:
    python -m script.bge_dataset_download --dataset_name Shitao/bge-reranker-data --output_dir datasets
    """

    parser = argparse.ArgumentParser(description="Download datasets from Hugging Face Hub.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="beir/trec-covid",
        help="Name of the dataset to download from Hugging Face Hub.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets",
        help="Directory to save the downloaded dataset.",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    output_dir = args.output_dir

    # Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)

    # Download the dataset from Hugging Face Hub and save it to the output directory.
    # Note: The `snapshot_download` function downloads the dataset to the specified directory.
    snapshot_download(repo_id=dataset_name, repo_type="dataset", local_dir=output_dir)

    print(f"Downloaded {dataset_name} dataset to {output_dir}.")