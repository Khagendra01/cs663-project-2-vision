#!/usr/bin/env python3
"""End-to-end video evaluation, Model A (local).

Loads files/model-a/model_a.keras (MobileNetV3Small, 4 classes) and
classifies every stored frame of all 6 real-video sequences, then feeds
each sequence's PREDICTED label stream into the temporal state machine
(target/duration/command from manifest.json).

Judging rules (--rules, see sequence_evaluate.py):
  app    (default) the Android app's rules (state_machine.py), t = real
         clip time = frame_index / 2 (the frames were stored at 2 fps).
         Writes results/video_predictions_model_a_app_rules.json.
  nov11  the Nov 11 run exactly: state_machine_nov11.py, smooth_window=3,
         t = frame index (1 fps effective, the app's sampler at the
         time). Writes results/video_predictions_model_a.json (the file
         the Nov 11 section of results/video_evaluation.md quotes).
The stored predictions can also be re-judged without TensorFlow by
rerun_nov25.py.

Preprocessing (exactly as in model-a/train_model_a.py): PIL RGB,
resize 224x224, float32 raw 0..255 pixels, NO extra Rescaling - the
backbone already contains its own Rescaling(1/127.5, -1) layer as its
first layer. Class order: ["sitting", "standing", "lying_down",
"walking"] (project order, argmax index).

Softmax confidence is passed to the state machine, so frames below its
min_confidence (0.65 app rules, 0.6 Nov 11 rules) count as uncertain
frames, exactly as the machine is designed to treat them.

Usage: ~/tfvenv/bin/python evaluate_model_a_video.py [--rules app|nov11]
                                                     [--out FILE]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from sequence_evaluate import RULES, judge, rules_description
from state_machine import FrameObservation, canonical_label

CLASS_NAMES = ["sitting", "standing", "lying_down", "walking"]
IMG_SIZE = 224
MODEL_PATH = ROOT / "model-a" / "model_a.keras"


def load_frames(frames_dir):
    return sorted(Path(frames_dir).glob("*.jpg"))


def predict(model, frames):
    """Batch-predict; returns [(label, confidence)] in frame order."""
    imgs = []
    for f in frames:
        arr = np.asarray(
            Image.open(f).convert("RGB").resize((IMG_SIZE, IMG_SIZE)),
            dtype=np.float32)
        imgs.append(arr)
    probs = model.predict(np.stack(imgs), verbose=0)
    out = []
    for p in probs:
        i = int(p.argmax())
        out.append((CLASS_NAMES[i], float(p[i])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", choices=RULES, default="app")
    ap.add_argument("--out")
    args = ap.parse_args()
    out = Path(args.out) if args.out else HERE / "results" / (
        "video_predictions_model_a.json" if args.rules == "nov11"
        else "video_predictions_model_a_app_rules.json")
    from tensorflow import keras
    model = keras.models.load_model(MODEL_PATH)
    manifest = json.load(open(HERE / "video-sequences" / "manifest.json"))

    results = {
        "model": "Model A - MobileNetV3Small (ImageNet), student fine-tuned "
                 "on dataset v4",
        "model_file": "model-a/model_a.keras",
        "preprocessing": "PIL RGB, resize 224x224, float32 raw 0..255, "
                         "no extra rescaling (backbone contains Rescaling)",
        "class_order": CLASS_NAMES,
        "time_base": ("t = frame index (1 fps effective, matches the app "
                      "sampler); labels-mode verification used t = i/2 "
                      "(stored 2 fps)" if args.rules == "nov11" else
                      "t = frame_index / 2 (real clip time, frames stored "
                      "at 2 fps)"),
        "state_machine": rules_description(args.rules),
        "sequences": [],
    }

    total_frames = 0
    correct_frames = 0
    correct_verdicts = 0
    print(f"{'sequence':28s} {'expected':10s} {'verdict':10s} "
          f"{'held':>5s} {'frames':>6s} {'match':>5s}")
    for m in manifest:
        frames = load_frames(HERE / "video-sequences" / m["frames_dir"])
        gt = json.load(open(HERE / "video-sequences" / m["labels"]))
        assert len(gt) == len(frames), (m["id"], len(gt), len(frames))
        preds = predict(model, frames)
        obs = []
        frame_rows = []
        for i, (f, (label, conf)) in enumerate(zip(frames, preds)):
            truth = canonical_label(gt[i]["label"])
            ok = label == truth
            total_frames += 1
            correct_frames += ok
            t = float(i) if args.rules == "nov11" else i / m["sample_fps"]
            obs.append(FrameObservation(t=t, label=label,
                                        confidence=conf))
            frame_rows.append({
                "frame": f.name, "t": t if args.rules != "nov11" else i,
                "predicted_label": label,
                "confidence": round(conf, 4),
                "ground_truth_label": truth,
                "correct": ok,
            })
        v = judge(obs, m["target_label"], m["target_duration_s"],
                  command=m["command"], rules=args.rules)
        match = v.verdict == m["expected_verdict"]
        correct_verdicts += match
        print(f"{m['id']:28s} {m['expected_verdict']:10s} {v.verdict:10s} "
              f"{v.held_s:5.1f} {len(frames):6d} {str(match):>5s}  "
              f"{v.reason}")
        results["sequences"].append({
            "id": m["id"],
            "command": m["command"],
            "target_label": m["target_label"],
            "target_duration_s": m["target_duration_s"],
            "expected_verdict": m["expected_verdict"],
            "verdict": v.verdict,
            "reason": v.reason,
            "target_reached_at_s": v.target_reached_at_s,
            "held_s": round(v.held_s, 2),
            "break_at_s": v.break_at_s,
            "n_frames": v.n_frames,
            "uncertain_ratio": round(v.uncertain_ratio, 3),
            "smoothed_sequence": [[t, x] for t, x in v.smoothed_sequence],
            "frame_accuracy": round(
                sum(r["correct"] for r in frame_rows) / len(frame_rows), 3),
            "match": match,
            "frames": frame_rows,
        })

    results["summary"] = {
        "sequences": len(manifest),
        "attempt_level_accuracy": round(correct_verdicts / len(manifest), 3),
        "correct_verdicts": f"{correct_verdicts}/{len(manifest)}",
        "frame_level_accuracy": round(correct_frames / total_frames, 3),
        "correct_frames": f"{correct_frames}/{total_frames}",
    }
    out.write_text(json.dumps(results, indent=1))
    print(f"\nframe-level accuracy: {correct_frames}/{total_frames} = "
          f"{correct_frames / total_frames:.3f}")
    print(f"attempt-level accuracy: {correct_verdicts}/{len(manifest)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
