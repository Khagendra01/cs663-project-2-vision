#!/usr/bin/env python3
"""Cut the real-video test sequences for the Project 2 video layer.

Sources are the 6 verified Wikimedia Commons videos from dataset v1
(see ../gemini-finetune/real-dataset.md for licences/attribution) - in
each, the dog holds ONE behaviour for the whole clip, which v1 already
verified frame by frame. Each sequence here is therefore an honest
"sustained hold" case: run it with a command whose target is that
behaviour and the correct verdict is SUCCESS - except the wagging clip,
which is shorter than its target duration and must come back UNCERTAIN
(incomplete). Posture *changes* (early break) cannot be faked from
single-behaviour clips; those cases are covered by the synthetic
state-sequence fixtures in scenarios.json + the unit tests, and in real
use by the live camera.

Output (video-sequences/):
  <id>.mp4            re-encoded clip (640w, no audio)
  <id>_frames/*.jpg   frames sampled at SAMPLE_FPS, 640w
  <id>_labels.json    per-frame ground truth [{t, label, confidence}]
                      (confidence 1.0: human-verified in dataset v1)
  manifest.json       all sequence metadata + expected verdicts
"""
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIDEOS = HERE.parent / "gemini-finetune" / "real-videos"
OUT = HERE / "video-sequences"
SAMPLE_FPS = 2

# id, source file, start_s, end_s, behaviour, command, target, target_s,
# expected verdict when judged with those parameters
SEQUENCES = [
    ("seq01_stand_africanis", "Africanis Dogs Resting In Grass.ogv",
     0, 10, "standing", "stand", "standing", 8, "SUCCESS",
     "CC0", "Wikimedia Commons"),
    ("seq02_down_batumi", "Batumi2025dog-laying-relax.webm",
     2, 12, "lying_down", "down", "lying_down", 8, "SUCCESS",
     "CC BY 4.0", "Vitaly Zdanevich"),
    ("seq03_walk_mission", "Dog walking in the Mission in San Francisco.ogv",
     0, 12, "walking", "walk", "walking", 10, "SUCCESS",
     "CC BY-SA 3.0", "Sage Ross (Ragesoss)"),
    ("seq04_down_snoring", "A domestic dog snoring.ogv",
     0, 10, "lying_down", "down", "lying_down", 8, "SUCCESS",
     "CC BY-SA 4.0", "Serial Number 54129"),
    ("seq05_down_panting", "Irish Wolfhound mix, panting.ogv",
     5, 15, "lying_down", "down", "lying_down", 8, "SUCCESS",
     "CC BY-SA 3.0", "Sage Ross (Ragesoss)"),
    ("seq06_stand_wagging_short", "Dog wagging tail.webm",
     0, 3.6, "standing", "stand", "standing", 8, "UNCERTAIN",
     "CC BY-SA 4.0", "Liannadavis"),
]
LABEL_TEXT = {"sitting": "sitting", "standing": "standing",
              "lying_down": "lying down", "walking": "walking"}


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr[-400:]}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for (sid, src, start, end, behaviour, command, target, target_s,
         expected, licence, artist) in SEQUENCES:
        src_path = VIDEOS / src
        dur = end - start
        clip = OUT / f"{sid}.mp4"
        run(["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-i",
             str(src_path), "-t", f"{dur}", "-vf", "scale=640:-2",
             "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(clip)])
        fdir = OUT / f"{sid}_frames"
        fdir.mkdir(exist_ok=True)
        for old in fdir.glob("*.jpg"):
            old.unlink()
        run(["ffmpeg", "-y", "-v", "error", "-i", str(clip),
             "-vf", f"fps={SAMPLE_FPS}", str(fdir / "frame_%03d.jpg")])
        frames = sorted(fdir.glob("*.jpg"))
        labels = [{"t": round(i / SAMPLE_FPS, 2),
                   "label": LABEL_TEXT[behaviour], "confidence": 1.0}
                  for i in range(len(frames))]
        (OUT / f"{sid}_labels.json").write_text(json.dumps(labels, indent=1))
        manifest.append({
            "id": sid, "clip": f"{sid}.mp4",
            "frames_dir": f"{sid}_frames", "labels": f"{sid}_labels.json",
            "source_video": src, "source_start_s": start,
            "duration_s": dur, "behaviour_throughout": behaviour,
            "command": command, "target_label": LABEL_TEXT[behaviour],
            "target_duration_s": target_s,
            "expected_verdict": expected,
            "expected_note": (
                "sustained hold reaches the target duration"
                if expected == "SUCCESS" else
                "clip ends at 3.6s, before the 8s target - incomplete, "
                "so UNCERTAIN; not a break (the dog never leaves standing)"),
            "sample_fps": SAMPLE_FPS, "n_frames": len(frames),
            "license": licence, "artist": artist,
            "ground_truth_basis": "behaviour verified frame-by-frame in "
                                  "dataset v1 (real-dataset.md)",
        })
        print(f"{sid}: {len(frames)} frames, expected {expected}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"manifest: {len(manifest)} sequences")


if __name__ == "__main__":
    main()
