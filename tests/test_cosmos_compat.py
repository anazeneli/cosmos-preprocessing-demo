#!/usr/bin/env python3
"""Test that processed data is compatible with Cosmos-Predict2.5 VideoDataset.
Run BEFORE switching to GPU to avoid wasting credits on bad data.
"""
import sys
import subprocess
from pathlib import Path

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name} — {detail}")
        failed += 1

DATASET_DIR = Path("/teamspace/lightning_storage/datasets/cosmos-nemo-assets")
MIN_FRAMES = 94
REQUIRED_HEIGHT = 704
REQUIRED_FPS = 16

print("\n=== Cosmos Training Compatibility Test ===\n")

# 1. Directory structure
print("[Structure]")
check("videos/ exists", (DATASET_DIR / "videos").is_dir())
check("metas/ exists", (DATASET_DIR / "metas").is_dir())

# 2. File counts match
videos = sorted((DATASET_DIR / "videos").glob("*.mp4"))
metas = sorted((DATASET_DIR / "metas").glob("*.txt"))
check(f"Videos found: {len(videos)}", len(videos) > 0, "No .mp4 files")
check(f"Captions found: {len(metas)}", len(metas) > 0, "No .txt files")
check("Video/caption count match", len(videos) == len(metas),
      f"{len(videos)} videos vs {len(metas)} captions")

# 3. Every video has a matching caption
print("\n[Filename Matching]")
video_stems = {v.stem for v in videos}
meta_stems = {m.stem for m in metas}
unmatched_videos = video_stems - meta_stems
unmatched_metas = meta_stems - video_stems
check("All videos have captions", len(unmatched_videos) == 0,
      f"Missing captions for: {unmatched_videos}")
check("All captions have videos", len(unmatched_metas) == 0,
      f"Orphan captions: {unmatched_metas}")

# 4. Every video meets the minimum frame count at REQUIRED_HEIGHT and REQUIRED_FPS
print(f"\n[Video Format — expecting >={MIN_FRAMES} frames, {REQUIRED_HEIGHT}p, {REQUIRED_FPS}fps]")
for video in videos:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
        "-of", "csv=p=0", str(video)
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        parts = out.split(",")
        w, h = int(parts[0]), int(parts[1])
        num, denom = map(int, parts[2].split("/"))
        fps = round(num / denom)
        frames = int(parts[3])

        ok = frames >= MIN_FRAMES and h == REQUIRED_HEIGHT and fps == REQUIRED_FPS
        detail = f"{w}x{h}, {fps}fps, {frames}f"
        if not ok:
            detail += f" — EXPECTED >={MIN_FRAMES}f, {REQUIRED_HEIGHT}p, {REQUIRED_FPS}fps"
        check(f"{video.name}: {detail}", ok)
    except Exception as e:
        check(f"{video.name}", False, str(e))

# 5. Captions are non-empty
print("\n[Captions]")
empty_captions = []
for meta in metas:
    text = meta.read_text().strip()
    if not text:
        empty_captions.append(meta.name)
check("All captions non-empty", len(empty_captions) == 0,
      f"Empty captions: {empty_captions}")

# Summary
print(f"\n=== {passed} passed, {failed} failed ===")
sys.exit(0 if failed == 0 else 1)
