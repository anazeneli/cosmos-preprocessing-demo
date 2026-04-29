"""cosmos-preprocessing-demo: data processing pipeline for Cosmos-Predict2.5.

Usage:
    python data/ingest.py cosmos-nemo-assets
    python data/preprocess.py cosmos-nemo-assets
    python data/validate.py cosmos-nemo-assets
    python tests/test_cosmos_compat.py
"""

import subprocess
import sys


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "cosmos-nemo-assets"
    steps = [
        ["python", "data/ingest.py", dataset],
        ["python", "data/preprocess.py", dataset],
        ["python", "data/validate.py", dataset],
        ["python", "tests/test_cosmos_compat.py", dataset],
    ]
    for cmd in steps:
        print(f"\n{'='*60}")
        print(f"Running: {' '.join(cmd)}")
        print(f"{'='*60}\n")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nFailed at: {' '.join(cmd)}")
            sys.exit(result.returncode)

    print(f"\nAll done. Dataset '{dataset}' is ready for training.")


if __name__ == "__main__":
    main()
