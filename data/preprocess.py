"""Preprocess videos into Cosmos-Predict2.5 training format.

Cosmos VideoDataset expects:
    <dataset_dir>/
    ├── videos/*.mp4    (720p, >=93 frames)
    └── metas/*.txt     (one text caption per video, matching filename)
"""

import argparse
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def process_one(src: Path, videos_dir: Path, metas_dir: Path,
                height: int, fps: int, min_frames: int, prompt: str) -> str:
    """Process a single video. Returns status string."""
    dst = videos_dir / src.name
    info = get_video_info(src)

    needs_reencode = info["height"] != height or abs(info["fps"] - fps) > 0.5

    if needs_reencode:
        if not reencode(src, dst, height, fps):
            return f"FAIL: {src.name} (reencode error)"
    else:
        shutil.copy2(src, dst)

    final = get_video_info(dst)
    if final["frames"] < min_frames:
        dst.unlink()
        return f"SKIP: {src.name} -- {final['frames']} frames (need {min_frames})"

    caption_src = src.with_suffix(".txt")
    caption_dst = metas_dir / f"{src.stem}.txt"
    if caption_src.exists():
        shutil.copy2(caption_src, caption_dst)
    else:
        caption_dst.write_text(prompt)

    return f"OK: {src.name} -> {final['width']}x{final['height']} @ {final['fps']:.0f}fps, {final['frames']}f"


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

    videos_dir = out_dir / "videos"
    metas_dir = out_dir / "metas"
    videos_dir.mkdir(parents=True, exist_ok=True)
    metas_dir.mkdir(parents=True, exist_ok=True)

    raw_videos = sorted(raw_dir.rglob("*.mp4"))
    if not raw_videos:
        print(f"No .mp4 files found in {raw_dir}")
        return

    print(f"[{args.dataset}] Processing {len(raw_videos)} videos ({height}p @ {fps}fps, min {min_frames} frames)")
    print(f"Workers: {num_workers}\n")

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(process_one, src, videos_dir, metas_dir, height, fps, min_frames, prompt): src
            for src in raw_videos
        }
        for f in as_completed(futures):
            print(f"  {f.result()}")

    n_videos = len(list(videos_dir.glob("*.mp4")))
    n_metas = len(list(metas_dir.glob("*.txt")))
    print(f"\nDone: {n_videos} videos, {n_metas} captions in {out_dir}")


if __name__ == "__main__":
    main()