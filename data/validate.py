"""Validate a dataset against Cosmos-Predict2.5 VideoDataset requirements.

Checks:
  - videos/*.mp4 exist and are 720p, exactly `min_frames` frames
  - metas/*.txt exist and match each video by stem name
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def get_video_info(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "csv=p=0", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        parts = out.split(",")
        w, h = int(parts[0]), int(parts[1])
        num, denom = map(int, parts[2].split("/"))
        fps = num / denom
        nb = int(parts[3]) if parts[3] != "N/A" else 0
        return {"width": w, "height": h, "fps": fps, "frames": nb}
    except Exception:
        return {"width": 0, "height": 0, "fps": 0, "frames": 0}


def main():
    parser = argparse.ArgumentParser(description="Validate dataset for Cosmos-Predict2.5")
    parser.add_argument("dataset", help="Dataset name from config.yaml (e.g. cosmos-nemo-assets)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    datasets = cfg["datasets"]

    if args.dataset not in datasets:
        available = ", ".join(datasets.keys())
        parser.error(f"Unknown dataset '{args.dataset}'. Available: {available}")

    ds = datasets[args.dataset]
    dataset_dir = Path(ds["processed_dir"])
    expected_height = cfg["video"]["height"]
    min_frames = cfg["video"]["min_frames"]

    videos_dir = dataset_dir / "videos"
    metas_dir = dataset_dir / "metas"
    errors = []

    if not videos_dir.is_dir():
        print(f"FAIL: {videos_dir} not found")
        sys.exit(1)
    if not metas_dir.is_dir():
        print(f"FAIL: {metas_dir} not found")
        sys.exit(1)

    videos = sorted(videos_dir.glob("*.mp4"))
    metas = sorted(metas_dir.glob("*.txt"))

    if not videos:
        print(f"FAIL: no .mp4 files in {videos_dir}")
        sys.exit(1)

    print(f"[{args.dataset}] Validating {len(videos)} videos in {dataset_dir}\n")

    video_stems = set()
    for v in videos:
        video_stems.add(v.stem)
        info = get_video_info(v)

        issues = []
        if info["height"] != expected_height:
            issues.append(f"height={info['height']} (want {expected_height})")
        if info["frames"] != min_frames:
            issues.append(f"frames={info['frames']} (need =={min_frames})")
        if info["width"] == 0:
            issues.append("ffprobe failed")

        status = "FAIL" if issues else "OK"
        detail = f" -- {', '.join(issues)}" if issues else f" -- {info['width']}x{info['height']}, {info['frames']}f"
        print(f"  [{status}] {v.name}{detail}")
        if issues:
            errors.append(v.name)

    meta_stems = {m.stem for m in metas}
    missing = video_stems - meta_stems
    orphans = meta_stems - video_stems

    print()
    if missing:
        errors.extend(missing)
        print(f"Missing captions: {sorted(missing)}")
    if orphans:
        print(f"Orphan captions (no matching video): {sorted(orphans)}")

    print()
    if errors:
        print(f"VALIDATION FAILED -- {len(errors)} issue(s)")
        sys.exit(1)
    else:
        print(f"PASSED -- {len(videos)} videos, {len(metas)} captions")
        print("Ready for Cosmos-Predict2.5 VideoDataset")


if __name__ == "__main__":
    main()