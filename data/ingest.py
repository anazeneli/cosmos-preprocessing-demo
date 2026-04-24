"""Download a dataset from HuggingFace."""

import argparse
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download dataset from HuggingFace")
    parser.add_argument("dataset", help="Dataset name from config.yaml (e.g. cosmos-nemo-assets)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    datasets = cfg["datasets"]

    if args.dataset not in datasets:
        available = ", ".join(datasets.keys())
        parser.error(f"Unknown dataset '{args.dataset}'. Available: {available}")

    ds = datasets[args.dataset]
    repo_id = ds["repo_id"]
    raw_dir = Path(ds["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} -> {raw_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(raw_dir),
        local_dir_use_symlinks=False,
    )

    videos = list(raw_dir.rglob("*.mp4"))
    print(f"Done. {len(videos)} videos downloaded to {raw_dir}")


if __name__ == "__main__":
    main()