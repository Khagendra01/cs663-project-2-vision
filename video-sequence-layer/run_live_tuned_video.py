#!/usr/bin/env python3
"""End-to-end video evaluation, tuned Gemini (live backend).

Runs every stored frame of all 6 real-video sequences through the
deployed tuned endpoint using sequence_evaluate.py's backend_label
(same prompt, same retries, same {prompt, imageBase64, mimeType} call
the Android app makes), then feeds each sequence's predicted label
stream into the temporal state machine with the same rules as the
Model A video run (target/duration/command from manifest.json).

Judging rules (--rules, see sequence_evaluate.py):
  app    (default) the Android app's rules (state_machine.py), t = real
         clip time = frame_index / 2 (frames stored at 2 fps). Writes
         results/video_predictions_tuned_app_rules.json and
         results/live_<seq>_tuned_app_rules.json.
  nov11  the Nov 11 run: state_machine_nov11.py, smooth_window=3,
         t = frame index (1 fps effective). Writes
         results/video_predictions_tuned.json and live_<seq>_tuned.json.
This makes one live backend call per frame (111 calls). To re-judge the
stored Nov 11 replies with no network calls, use rerun_nov25.py.

This driver exists because sequence_evaluate.py --out only records the
verdict, not the per-frame raw replies; it imports backend_label and
verdict_dict from sequence_evaluate.py rather than reimplementing the
backend client. A frame whose call fails after all retries is recorded
as a failed call with label None (an uncertain frame); a reply that
does not parse to one of the 4 labels is recorded with its raw text.

Usage: python3 run_live_tuned_video.py [--rules app|nov11]
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sequence_evaluate import (RULES, backend_label, judge,
                               rules_description, verdict_dict)
from state_machine import FrameObservation, canonical_label

BACKEND_URL = "https://cs663-gemini-service-917815040923.us-central1.run.app"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", choices=RULES, default="app")
    args = ap.parse_args()
    suffix = "" if args.rules == "nov11" else "_app_rules"
    manifest = json.load(open(HERE / "video-sequences" / "manifest.json"))
    results = {
        "model": "Model B - fine-tuned Gemini (tuned endpoint, live)",
        "backend_url": BACKEND_URL,
        "endpoint": "/api/fine-tuned-gemini",
        "prompt": "Look at the image of the dog. What is the dog doing? "
                  "Answer with exactly one of: sitting, standing, "
                  "lying down, walking.",
        "time_base": ("t = frame index (1 fps effective, matches the app "
                      "sampler); the earlier live_seq06_tuned.json proof "
                      "used the harness default fps=2 (t = i/2)"
                      if args.rules == "nov11" else
                      "t = frame_index / 2 (real clip time, frames stored "
                      "at 2 fps)"),
        "state_machine": rules_description(args.rules),
        "sequences": [],
    }
    total_frames = 0
    correct_frames = 0
    correct_verdicts = 0
    failed_calls = 0
    unparseable = 0
    t0 = time.time()

    print(f"{'sequence':28s} {'expected':10s} {'verdict':10s} "
          f"{'held':>5s} {'frames':>6s} {'match':>5s}", flush=True)
    for m in manifest:
        frames = sorted(
            (HERE / "video-sequences" / m["frames_dir"]).glob("*.jpg"))
        gt = json.load(open(HERE / "video-sequences" / m["labels"]))
        assert len(gt) == len(frames), (m["id"], len(gt), len(frames))
        obs = []
        frame_rows = []
        for i, f in enumerate(frames):
            raw = backend_label(BACKEND_URL, "tuned", f)
            if raw is None:
                failed_calls += 1
                parsed = None
            else:
                parsed = canonical_label(raw)
                if parsed == "uncertain_frame":
                    unparseable += 1
            truth = canonical_label(gt[i]["label"])
            ok = parsed == truth
            total_frames += 1
            correct_frames += ok
            t = float(i) if args.rules == "nov11" else i / m["sample_fps"]
            obs.append(FrameObservation(t=t, label=raw))
            frame_rows.append({
                "frame": f.name, "t": t if args.rules != "nov11" else i,
                "predicted_label": parsed,
                "raw_text": raw,
                "call_failed": raw is None,
                "unparseable": raw is not None
                               and parsed == "uncertain_frame",
                "ground_truth_label": truth,
                "correct": ok,
            })
            print(f"  {m['id']} frame {i + 1}/{len(frames)}: "
                  f"{raw!r} -> {parsed}", flush=True)
        v = judge(obs, m["target_label"], m["target_duration_s"],
                  command=m["command"], rules=args.rules)
        vd = verdict_dict(v, "backend tuned (live)")
        match = v.verdict == m["expected_verdict"]
        correct_verdicts += match
        print(f"{m['id']:28s} {m['expected_verdict']:10s} {v.verdict:10s} "
              f"{v.held_s:5.1f} {len(frames):6d} {str(match):>5s}  "
              f"{v.reason}", flush=True)
        (HERE / "results" / f"live_{m['id']}_tuned{suffix}.json").write_text(
            json.dumps(vd, indent=1))
        results["sequences"].append({
            "id": m["id"],
            "command": m["command"],
            "target_label": m["target_label"],
            "target_duration_s": m["target_duration_s"],
            "expected_verdict": m["expected_verdict"],
            **{k: vd[k] for k in ("verdict", "reason",
                                  "target_reached_at_s", "held_s",
                                  "break_at_s", "n_frames",
                                  "uncertain_ratio",
                                  "smoothed_sequence")},
            "frame_accuracy": round(
                sum(r["correct"] for r in frame_rows) / len(frame_rows), 3),
            "match": match,
            "frames": frame_rows,
        })

    elapsed = round(time.time() - t0, 1)
    results["summary"] = {
        "sequences": len(manifest),
        "attempt_level_accuracy": round(correct_verdicts / len(manifest), 3),
        "correct_verdicts": f"{correct_verdicts}/{len(manifest)}",
        "frame_level_accuracy": round(correct_frames / total_frames, 3),
        "correct_frames": f"{correct_frames}/{total_frames}",
        "failed_calls_after_retries": failed_calls,
        "unparseable_replies": unparseable,
        "total_calls": total_frames,
        "elapsed_s": elapsed,
    }
    out = HERE / "results" / f"video_predictions_tuned{suffix}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nframe-level accuracy: {correct_frames}/{total_frames} = "
          f"{correct_frames / total_frames:.3f}")
    print(f"attempt-level accuracy: {correct_verdicts}/{len(manifest)}")
    print(f"failed calls (after retries): {failed_calls}/{total_frames}")
    print(f"unparseable replies: {unparseable}/{total_frames}")
    print(f"elapsed: {elapsed}s")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
