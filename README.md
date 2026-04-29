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


## Datasets

| Name | Repo | Role | Videos |
|------|------|------|--------|
| `cosmos-nemo-assets` | [`nvidia/Cosmos-NeMo-Assets`](https://huggingface.co/datasets/nvidia/Cosmos-NeMo-Assets) | training (sks teal robot) | 4 |
| `gr1-100` | [`nvidia/PhysicalAI-Robotics-GR00T-GR1`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-GR1) | validation (Fourier GR1-T2 humanoid) | 92 |

## Prerequisites

1. Accept the [Cosmos-NeMo-Assets dataset license](https://huggingface.co/datasets/nvidia/Cosmos-NeMo-Assets) on HuggingFace (GR1-100 is CC-BY-4.0, no acceptance needed)
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

# Training set
python data/ingest.py cosmos-nemo-assets
python data/preprocess.py cosmos-nemo-assets
python data/validate.py cosmos-nemo-assets

# Validation set
python data/ingest.py gr1-100
python data/preprocess.py gr1-100
python data/validate.py gr1-100
```

Or run the full pipeline (ingest → preprocess → validate → cosmos-compat test) for one dataset:

```bash
python main.py cosmos-nemo-assets
python main.py gr1-100
```

## Output Format

Cosmos-Predict2.5 `VideoDataset` expects:

```
<dataset_dir>/
├── videos/*.mp4    720p MP4, >=93 frames, 16fps
└── metas/*.txt     one caption per video (matching filename)
```

No pre-computed T5 embeddings needed — Cosmos-Predict2.5 uses Qwen2.5-VL-7B as its text encoder and computes embeddings online during training.

## Shared Teamspace Storage

Processed data is written to `/teamspace/lightning_storage/datasets/`, a writable shared mount available to every Studio in the Teamspace. The Training and Inference Studios read from this path directly — no copying needed when switching to GPU. Raw downloads stay local (`data/raw/`) since they don't need to be shared.

## Parallel Preprocessing

Uses [LitData](https://github.com/Lightning-AI/litdata) for distributed data
preprocessing. With 4 videos this runs in seconds. With 10,000 videos from
50 customer sites, LitData distributes across workers (and nodes, when run on
Lightning AI) automatically — no manual pool management.

## Adding Datasets

Add a block to `config.yaml` — no code changes needed:

```yaml
datasets:
  my-new-dataset:
    repo_id: "org/dataset-name"
    raw_dir: "data/raw/my-new-dataset"
    processed_dir: "/teamspace/lightning_storage/datasets/my-new-dataset"
    prompt: "A video of a robot doing X."
```

Then run the same commands with the new name. `preprocess.py` recursively
finds `*.mp4` under `raw_dir`, so subdirectory layouts (e.g. `gr1/*.mp4`)
work without changes.

## What's Here

```
├── config.yaml          # dataset registry + Cosmos video requirements
├── setup_env.sh         # install pyyaml, huggingface_hub, litdata, ffmpeg
└── data/
    ├── ingest.py        # download from HuggingFace
    ├── preprocess.py    # reencode to 720p/16fps + create captions (via LitData)
    └── validate.py      # check format against Cosmos requirements
```
