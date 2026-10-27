#!/usr/bin/env python3
"""Extract still frames from a dog behaviour clip.

Usage:
    python3 extract_frames.py --video clips/milo_sitting_01.mp4 \
        --out dog-data/milo/sitting --fps 1

Requires ffmpeg on PATH. Frames are written as <stem>_0001.jpg, ... into
--out. After extraction, do the human review pass described in
dataset-collection-guide.md and rename kept frames to
<dog>_<behaviour>_<nnn>.jpg before uploading to Cloud Storage.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract frames from a video clip.")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--out", required=True, help="Output directory for frames")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frames per second to extract (default: 1)")
    args = parser.parse_args()

    video = Path(args.video)
    out = Path(args.out)
    if not video.is_file():
        print(f"error: video not found: {video}", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    pattern = str(out / f"{video.stem}_%04d.jpg")
    cmd = ["ffmpeg", "-y", "-i", str(video), "-vf", f"fps={args.fps}", pattern]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-2000:], file=sys.stderr)
        return result.returncode

    count = len(list(out.glob(f"{video.stem}_*.jpg")))
    print(f"extracted {count} frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
