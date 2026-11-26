#!/usr/bin/env python3
"""Sequence evaluation harness for the Project 2 video layer.

Takes ONE command attempt - a video clip, a folder of frames, or a
precomputed label stream - classifies each sampled frame, feeds the
labels through the temporal state machine, and prints/writes the
verdict (SUCCESS / EARLY_BREAK / UNCERTAIN) with its evidence.

Frame classification modes:
  --labels FILE.json   precomputed labels: [{t, label, confidence}] or a
                       plain list of label strings at --fps. Used with
                       the verified ground-truth labels of
                       video-sequences/ and with scenarios.json streams.
                       This validates the judging pipeline, NOT classifier
                       accuracy - the report says so.
  --backend URL        live mode: POST every sampled frame to the Cloud
                       Run backend ({prompt, imageBase64, mimeType} ->
                       {text}) at --endpoint base|tuned, parse the reply
                       to one of the 4 labels. This is the same call the
                       Android app makes, once per sampled frame.
  (frames with no classifier mode) error - pick one.

Examples:
  python3 sequence_evaluate.py --labels video-sequences/seq03_walk_mission_labels.json \
      --target walking --duration 10
  python3 sequence_evaluate.py --video video-sequences/seq03_walk_mission.mp4 \
      --backend https://cs663-gemini-service-....run.app --endpoint tuned \
      --target walking --duration 10 --out result.json
  python3 sequence_evaluate.py --all-sequences     # labels mode, manifest check
  python3 sequence_evaluate.py --scenarios         # synthetic scenarios check

Judging rules (--rules):
  app    (default) state_machine.py - the same rules and numbers as the
         Android app: smooth window 5, min confidence 0.65, settle grace
         2 s, break confirm 0.6 s, lost 1.5 s; Sit / Down judged as
         reach-then-hold (3 s response window, 0.6 s hold), other
         commands as a hold of --duration (override with --shape).
  nov11  state_machine_nov11.py with the Nov 11 numbers (smooth window
         3, min confidence 0.6, break confirm 1 s, lost 3 s, every
         command a hold) - reproduces results/verification_run.txt and
         the Nov 10-11 scenario checks exactly.
  --smooth-window / --min-confidence / --break-confirm / --lost override
  the chosen rule set's numbers.
"""
import argparse
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import state_machine_nov11
from state_machine import (PRESETS, FrameObservation, canonical_label,
                           config_for_command, run_sequence)

RULES = ("app", "nov11")

PROMPT = ("Look at the image of the dog. What is the dog doing? "
          "Answer with exactly one of: sitting, standing, lying down, walking.")


def labels_from_file(path, fps):
    data = json.load(open(path))
    if isinstance(data, dict) and "labels" in data:   # scenario object
        data = data["labels"]
    obs = []
    for i, item in enumerate(data):
        if isinstance(item, dict):
            obs.append(FrameObservation(
                t=float(item.get("t", i / fps)), label=item.get("label"),
                confidence=item.get("confidence"),
                dog_present=item.get("label") is not None))
        else:  # plain string / null list at fps
            obs.append(FrameObservation(t=i / fps, label=item))
    return obs


def extract_frames(video, fps, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video),
                        "-vf", f"fps={fps},scale=640:-2",
                        str(outdir / "frame_%03d.jpg")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-300:])
    return sorted(outdir.glob("*.jpg"))


def backend_label(url, endpoint, frame_path, timeout=60, tries=3):
    body = json.dumps({
        "prompt": PROMPT,
        "imageBase64": base64.b64encode(Path(frame_path).read_bytes()).decode(),
        "mimeType": "image/jpeg",
    }).encode()
    last = None
    for i in range(tries):   # the egress proxy / cold starts drop some calls
        try:
            req = urllib.request.Request(
                url.rstrip("/") + ("/api/fine-tuned-gemini"
                                   if endpoint == "tuned" else "/api/gemini"),
                data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                text = json.load(resp).get("text", "")
            label = canonical_label(text)
            return (None if label == "uncertain_frame" else text)
        except Exception as e:  # noqa: BLE001 - retried, then reported as no label
            last = e
            time.sleep(2 * (i + 1))
    print(f"WARN: backend call failed for {frame_path} after {tries} tries: "
          f"{last!r} - frame counted as no-label", file=sys.stderr)
    return None


def verdict_dict(v, mode):
    return {
        "verdict": v.verdict, "reason": v.reason,
        "target_label": v.target_label,
        "target_duration_s": v.target_duration_s,
        "target_reached_at_s": v.target_reached_at_s,
        "held_s": round(v.held_s, 2), "break_at_s": v.break_at_s,
        "n_frames": v.n_frames,
        "uncertain_ratio": round(v.uncertain_ratio, 3),
        "classification_mode": mode,
        "smoothed_sequence": [[t, x] for t, x in v.smoothed_sequence],
    }


def session_config(rules, command, duration, shape="auto", **overrides):
    """SessionConfig for one attempt under the chosen rule set.

    rules="app": config_for_command() (Sit / Down -> reach-then-hold
    unless shape says otherwise). rules="nov11": the frozen Nov 11
    module's SessionConfig with the Nov 11 numbers (hold shape only).
    Overrides that are None are ignored.
    """
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if rules == "nov11":
        if shape == "reach":
            raise ValueError("the Nov 11 rules have no reach-then-hold shape")
        kw = dict(PRESETS["nov11"])
        kw.update(overrides)
        return state_machine_nov11.SessionConfig(
            target_duration_s=float(duration), **kw)
    if rules != "app":
        raise ValueError(f"unknown rules {rules!r}; one of {RULES}")
    return config_for_command(command, duration, preset="app", shape=shape,
                              **overrides)


def judge(obs, target, duration, command=None, rules="app", shape="auto",
          **overrides):
    """Run one label stream through the chosen rule set -> Verdict."""
    cfg = session_config(rules, command, duration, shape, **overrides)
    runner = (state_machine_nov11.run_sequence if rules == "nov11"
              else run_sequence)
    return runner(obs, target, cfg, command=command)


def rules_description(rules):
    """Machine-readable record of the judging rules, for result files."""
    if rules == "nov11":
        return {"rules": "nov11 (state_machine_nov11.py, frozen Nov 11)",
                **PRESETS["nov11"],
                "command_shape": "every command judged as a hold of the "
                                 "target duration",
                "break_at_s": "first raw observation of the wrong posture"}
    return {"rules": "app (state_machine.py, same rules as the Android "
                     "app's SequenceSession)",
            **PRESETS["app"],
            "command_shape": "Sit / Down: reach within 3 s, then hold "
                             "0.6 s; other commands: hold the target "
                             "duration (settle grace 2 s)",
            "response_window_s": 3.0, "hold_after_reach_s": 0.6,
            "confirm_grace_s": 1.0,
            "reach_time": "first raw target frame of the confirmed run",
            "break_at_s": "first frame after the last raw target frame"}


def _overrides(args):
    return dict(smooth_window=getattr(args, "smooth_window", None),
                min_confidence=getattr(args, "min_confidence", None),
                break_confirm_s=getattr(args, "break_confirm", None),
                uncertain_timeout_s=getattr(args, "lost", None))


def evaluate(args, obs, mode):
    v = judge(obs, args.target, args.duration, command=args.command,
              rules=getattr(args, "rules", "app"),
              shape=getattr(args, "shape", "auto"), **_overrides(args))
    return verdict_dict(v, mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--frames")
    ap.add_argument("--labels")
    ap.add_argument("--backend")
    ap.add_argument("--endpoint", choices=["base", "tuned"], default="tuned")
    ap.add_argument("--target")
    ap.add_argument("--command")
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--rules", choices=RULES, default="app",
                    help="app (default): the Android app's rules; nov11: "
                         "the frozen Nov 11 rules and numbers")
    ap.add_argument("--shape", choices=["auto", "hold", "reach"],
                    default="auto",
                    help="command shape (app rules): auto = Sit/Down "
                         "reach-then-hold, others hold")
    ap.add_argument("--smooth-window", type=int, default=None,
                    help="majority-vote window in frames (default: 5 with "
                         "--rules app, 3 with --rules nov11)")
    ap.add_argument("--min-confidence", type=float, default=None,
                    help="default 0.65 (app) / 0.6 (nov11)")
    ap.add_argument("--break-confirm", type=float, default=None,
                    help="seconds; default 0.6 (app) / 1.0 (nov11)")
    ap.add_argument("--lost", type=float, default=None,
                    help="seconds unclear -> UNCERTAIN; default 1.5 (app) "
                         "/ 3.0 (nov11)")
    ap.add_argument("--out")
    ap.add_argument("--all-sequences", action="store_true")
    ap.add_argument("--scenarios", action="store_true")
    args = ap.parse_args()

    if args.all_sequences:
        manifest = json.load(open(HERE / "video-sequences" / "manifest.json"))
        ok = True
        for m in manifest:
            a = argparse.Namespace(**{**vars(args),
                                      "target": m["target_label"],
                                      "duration": m["target_duration_s"],
                                      "command": m["command"]})
            obs = labels_from_file(
                HERE / "video-sequences" / m["labels"], m["sample_fps"])
            got = evaluate(a, obs, "labels (verified ground truth)")
            mark = "PASS" if got["verdict"] == m["expected_verdict"] else "FAIL"
            ok &= mark == "PASS"
            print(f"[{mark}] {m['id']}: expected {m['expected_verdict']}, "
                  f"got {got['verdict']} - {got['reason']}")
        print("NOTE: labels mode validates the judging pipeline against "
              "human-verified frame labels; it is not a classifier score.")
        print(f"rules: {args.rules} (expected verdicts were set for the "
              "Nov 10 rules; see results/video_evaluation.md)")
        sys.exit(0 if ok else 1)

    if args.scenarios:
        data = json.load(open(HERE / "scenarios.json"))
        cfgd = data["config"]
        ok = True
        for sc in data["scenarios"]:
            a = argparse.Namespace(**{**vars(args),
                                      "target": sc["target_label"],
                                      "duration": cfgd["target_duration_s"],
                                      "command": sc["command"]})
            obs = labels_from_file_from_list(sc["labels"])
            got = evaluate(a, obs, "labels (synthetic scenario)")
            mark = "PASS" if got["verdict"] == sc["expected_verdict"] else "FAIL"
            ok &= mark == "PASS"
            print(f"[{mark}] {sc['id']}: expected {sc['expected_verdict']}, "
                  f"got {got['verdict']} - {got['reason']}")
        print(f"rules: {args.rules} (expected verdicts were set for the "
              "Nov 10 rules; see results/video_evaluation.md)")
        sys.exit(0 if ok else 1)

    if not args.target:
        ap.error("--target is required (or use --all-sequences/--scenarios)")

    if args.labels:
        obs = labels_from_file(args.labels, args.fps)
        mode = "labels (precomputed)"
    elif args.backend:
        if args.video:
            with tempfile.TemporaryDirectory() as td:
                frames = extract_frames(args.video, args.fps, td)
                if not frames:
                    ap.error(f"no frames extracted from {args.video}")
                obs = [FrameObservation(t=i / args.fps,
                                        label=backend_label(args.backend,
                                                            args.endpoint, f))
                       for i, f in enumerate(frames)]
        elif args.frames:
            frames = sorted(Path(args.frames).glob("*.jpg"))
            if not frames:
                ap.error(f"no .jpg frames in {args.frames}")
            obs = [FrameObservation(t=i / args.fps,
                                    label=backend_label(args.backend,
                                                        args.endpoint, f))
                   for i, f in enumerate(frames)]
        else:
            ap.error("--backend needs --video or --frames")
        mode = f"backend {args.endpoint} (live)"
    else:
        ap.error("choose a classification mode: --labels or --backend")

    result = evaluate(args, obs, mode)
    print(json.dumps({k: v for k, v in result.items()
                      if k != "smoothed_sequence"}, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
        print(f"wrote {args.out}")


def labels_from_file_from_list(labels, fps=2.0):
    return [FrameObservation(t=i / fps, label=x) for i, x in enumerate(labels)]


if __name__ == "__main__":
    main()
