"""Preprocess videos into Cosmos-Predict2.5 training format.

Cosmos VideoDataset expects:
    <dataset_dir>/
    ├── videos/*.mp4    (720p, >=93 frames)
    └── metas/*.txt     (one text caption per video, matching filename)

Parallelism is handled by LitData's `map` — distributes work across workers
(and nodes when run on Lightning AI) without manual pool management.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import litdata as ld
import yaml


def get_video_info(path: Path) -> dict:
    """Get width, height, fps, and frame count via ffprobe."""
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
    except Exception as e:
        print(f"  WARNING: ffprobe failed on {path.name}: {e}")
        return {"width": 0, "height": 0, "fps": 0, "frames": 0}


def reencode(src: Path, dst: Path, height: int, fps: int) -> bool:
    """Re-encode video to target resolution and fps."""
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"scale=-2:{height},fps={fps}",
        "-c:v", "libx264", "-preset", "fast", "-an",
        str(dst),
    ]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0


def extract_chunk(src: Path, dst: Path, start: int, end: int) -> bool:
    """Extract frames [start, end] inclusive into dst."""
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"select='between(n\\,{start}\\,{end})',setpts=N/FR/TB",
        "-frames:v", str(end - start + 1),
        "-c:v", "libx264", "-preset", "fast", "-an",
        str(dst),
    ]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0


def process_video(item: dict, output_dir: str) -> None:
    """Re-encode then split into consecutive `chunk_frames`-frame clips.

    Each chunk inherits the source video's caption. Tail frames below
    `chunk_frames` are discarded.
    """
    src = Path(item["input_path"])
    height = item["height"]
    fps = item["fps"]
    chunk_frames = item["min_frames"]
    prompt = item["prompt"]

    out_root = Path(output_dir)
    videos_dir = out_root / "videos"
    metas_dir = out_root / "metas"
    videos_dir.mkdir(parents=True, exist_ok=True)
    metas_dir.mkdir(parents=True, exist_ok=True)

    caption_src = src.with_suffix(".txt")
    caption_text = caption_src.read_text() if caption_src.exists() else prompt

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        tmp = Path(tf.name)
    try:
        if not reencode(src, tmp, height, fps):
            print(f"  FAIL: {src.name} (reencode error)")
            return

        info = get_video_info(tmp)
        n_chunks = info["frames"] // chunk_frames
        if n_chunks == 0:
            print(f"  SKIP: {src.name} -- {info['frames']} frames (need {chunk_frames})")
            return

        for i in range(n_chunks):
            start = i * chunk_frames
            end = start + chunk_frames - 1
            stem = f"{src.stem}_chunk{i:03d}"
            chunk_path = videos_dir / f"{stem}.mp4"
            if not extract_chunk(tmp, chunk_path, start, end):
                print(f"  FAIL: {stem} (chunk extract error)")
                continue
            (metas_dir / f"{stem}.txt").write_text(caption_text)

        print(f"  OK: {src.name} -> {n_chunks} chunks ({info['frames']} frames, {info['width']}x{info['height']} @ {info['fps']:.0f}fps)")
    finally:
        if tmp.exists():
            tmp.unlink()


def main():
    parser = argparse.ArgumentParser(description="Preprocess videos for Cosmos-Predict2.5")
    parser.add_argument("dataset", help="Dataset name from config.yaml (e.g. cosmos-nemo-assets)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    datasets = cfg["datasets"]

    if args.dataset not in datasets:
        available = ", ".join(datasets.keys())
        parser.error(f"Unknown dataset '{args.dataset}'. Available: {available}")

    ds = datasets[args.dataset]
    raw_dir = Path(ds["raw_dir"])
    out_dir = Path(ds["processed_dir"])
    height = cfg["video"]["height"]
    fps = cfg["video"]["fps"]
    min_frames = cfg["video"]["min_frames"]
    prompt = ds.get("prompt", "A video.")
    num_workers = cfg["preprocess"]["num_workers"]

    raw_videos = sorted(raw_dir.rglob("*.mp4"))
    if not raw_videos:
        print(f"No .mp4 files found in {raw_dir}")
        return

    print(f"[{args.dataset}] Processing {len(raw_videos)} videos ({height}p @ {fps}fps, min {min_frames} frames)")
    print(f"Workers: {num_workers}\n")

    inputs = [
        {
            "input_path": str(p),
            "height": height,
            "fps": fps,
            "min_frames": min_frames,
            "prompt": prompt,
        }
        for p in raw_videos
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    ld.map(
        fn=process_video,
        inputs=inputs,
        output_dir=str(out_dir),
        input_dir=str(raw_dir),
        num_workers=num_workers,
    )

    n_videos = len(list((out_dir / "videos").glob("*.mp4")))
    n_metas = len(list((out_dir / "metas").glob("*.txt")))
    print(f"\nDone: {n_videos} videos, {n_metas} captions in {out_dir}")


if __name__ == "__main__":
    main()