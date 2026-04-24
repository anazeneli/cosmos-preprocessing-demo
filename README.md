# cosmos-preprocessing-demo

Data processing studio for the Cosmos-Predict2.5 LoRA fine-tuning pipeline. Downloads, preprocesses, and validates robotics video datasets into the format expected by Cosmos-Predict2.5's `VideoDataset`.

Part of a multi-studio Lightning AI demo:

| Studio | Purpose |
|--------|---------|
| **This one** | Data ingestion, preprocessing, validation (CPU) |
| Training | Cosmos LoRA fine-tuning via torchrun (GPU) |
| Inference | LitServe endpoint serving checkpoints (GPU) |
| Pipelines | Orchestration and automation (CPU) |
| Web App | Streamlit demo (GPU) |


## Prerequisites

1. Accept the [Cosmos-NeMo-Assets dataset license](https://huggingface.co/datasets/nvidia/Cosmos-NeMo-Assets) on HuggingFace
2. Authenticate with HuggingFace:

```bash
hf auth login
```

Or set the token in your environment:

```bash
export HF_TOKEN="your_token_here"
```

## Quick Start

```bash
bash setup_env.sh

python data/ingest.py cosmos-nemo-assets
python data/preprocess.py cosmos-nemo-assets
python data/validate.py cosmos-nemo-assets
```

## Output Format

Cosmos-Predict2.5 `VideoDataset` expects:

```
<dataset_dir>/
├── videos/*.mp4    720p MP4, >=93 frames, 16fps
└── metas/*.txt     one caption per video (matching filename)
```

No pre-computed T5 embeddings needed — Cosmos-Predict2.5 uses Qwen2.5-VL-7B as its text encoder and computes embeddings online during training.

## Adding Datasets

Add a block to `config.yaml`:

```yaml
datasets:
  my-new-dataset:
    repo_id: "org/dataset-name"
    raw_dir: "data/raw/my-new-dataset"
    processed_dir: "data/processed/my-new-dataset"
    prompt: "A video of a robot doing X."
```

Then run the same three commands with the new name.

## What's Here

```
├── config.yaml          # dataset registry + Cosmos video requirements
├── setup_env.sh         # install pyyaml, huggingface_hub, ffmpeg
└── data/
    ├── ingest.py        # download from HuggingFace
    ├── preprocess.py    # reencode to 720p/16fps + create captions
    └── validate.py      # check format against Cosmos requirements
```
