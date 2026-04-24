#!/bin/bash
# Studio 1: Data Processing — environment setup (CPU only)
set -e

echo "Setting up data processing environment..."

pip install --quiet pyyaml huggingface_hub

# ffmpeg for video re-encoding
if ! command -v ffmpeg &>/dev/null; then
    echo "Installing ffmpeg..."
    sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg
fi

echo "Done. Usage:"
echo "  python data/ingest.py cosmos-nemo-assets"
echo "  python data/preprocess.py cosmos-nemo-assets"
echo "  python data/validate.py cosmos-nemo-assets"
echo "Set HF_TOKEN in your environment: export HF_TOKEN=your_token"
