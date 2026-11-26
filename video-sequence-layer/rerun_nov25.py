#!/usr/bin/env python3
"""Nov 25 re-run: re-judge the stored evidence with the app's rules.

No classifier runs and NO network calls: this script only re-judges
label streams that are already on disk.

  1. The 6 real-video sequences (video-sequences/manifest.json), with
     three label sources each:
       ground_truth  the human-verified per-frame labels (labels mode)
       model_a       Model A's stored per-frame predictions + softmax
                     confidence (results/video_predictions_model_a.json)
       tuned         the tuned Gemini endpoint's stored per-frame replies
                     (results/video_predictions_tuned.json; a failed call
                     would be a no-label frame - there were none)
     The frames were extracted at 2 fps, so t = frame_index / 2 = real
     clip time (the Nov 11 run used t = frame index).
  2. The 6 synthetic scenarios in scenarios.json (2 fps, t = i / 2).
  3. The 180 Model B scenarios in
     ../gemini-finetune/model-b-scenarios/scenarios.jsonl (1 step = 1 s,
     as written in their text). Scenario files are NOT modified; the
     script only reports where today's verdict differs from the stored
     one.

Every stream is judged at several steps, so each change can be traced
to one cause:
  nov11_frozen        state_machine_nov11.py + Nov 11 numbers (must
                      reproduce the stored verdicts)
  new_rules_nov11_numbers_hold
                      state_machine.py rules, Nov 11 numbers, every
                      command a hold
  app_numbers_hold    state_machine.py rules, app numbers, every command
                      a hold of the stated duration
  app                 the app exactly: as above, but Sit / Down judged
                      reach-then-hold (3 s window, 0.6 s hold)
For the real sequences there is one more step before these, the Nov 11
stored verdict at t = frame index, and every step above runs at clip time.

Usage:  python3 rerun_nov25.py
Writes: results/video_evaluation_nov25.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sequence_evaluate import judge, rules_description  # noqa: E402
from state_machine import FrameObservation, canonical_label  # noqa: E402

RESULTS = HERE / "results"
MODEL_B = HERE.parent / "gemini-finetune" / "model-b-scenarios" / \
    "scenarios.jsonl"
OUT = RESULTS / "video_evaluation_nov25.json"
STORED_FPS = 2.0

# (step name, rules, shape, preset numbers for the new rules)
STEPS = [
    ("nov11_frozen", "nov11", "auto", None),
    ("new_rules_nov11_numbers_hold", "app", "hold", "nov11"),
    ("app_numbers_hold", "app", "hold", None),
    ("app", "app", "auto", None),
]
NOV11_NUMBERS = dict(smooth_window=3, min_confidence=0.6,
                     break_confirm_s=1.0, uncertain_timeout_s=3.0)


def judge_step(obs, target, duration, command, step):
    _, rules, shape, numbers = step
    over = NOV11_NUMBERS if numbers == "nov11" else {}
    return judge(obs, target, duration, command=command, rules=rules,
                 shape=shape, **over)


def vd(v):
    return {"verdict": v.verdict, "reason": v.reason,
            "target_reached_at_s": v.target_reached_at_s,
            "held_s": round(v.held_s, 2), "break_at_s": v.break_at_s,
            "n_frames": v.n_frames,
            "uncertain_ratio": round(v.uncertain_ratio, 3)}


def first_change(stored, ladder):
    """Name of the first step whose verdict differs from the stored one."""
    for name, v in ladder:
        if v["verdict"] != stored:
            return name
    return None


# ------------------------------------------------------------ sequences
def sequence_streams(m, pred_a, pred_t):
    gt = json.load(open(HERE / "video-sequences" / m["labels"]))
    a = pred_a[m["id"]]["frames"]
    tu = pred_t[m["id"]]["frames"]
    assert len(gt) == len(a) == len(tu) == m["n_frames"], m["id"]
    # each stream as (label, confidence, dog_present) in frame order
    return {
        "ground_truth": [(g["label"], g.get("confidence"), True) for g in gt],
        "model_a": [(f["predicted_label"], f["confidence"], True) for f in a],
        "tuned": [(f["raw_text"], None, f["raw_text"] is not None)
                  for f in tu],
    }


def to_obs(stream, dt):
    return [FrameObservation(t=i * dt, label=lab, confidence=conf,
                             dog_present=dog)
            for i, (lab, conf, dog) in enumerate(stream)]


def run_sequences():
    manifest = json.load(open(HERE / "video-sequences" / "manifest.json"))
    pa = json.load(open(RESULTS / "video_predictions_model_a.json"))
    pt = json.load(open(RESULTS / "video_predictions_tuned.json"))
    pred_a = {s["id"]: s for s in pa["sequences"]}
    pred_t = {s["id"]: s for s in pt["sequences"]}
    rows = []
    for m in manifest:
        streams = sequence_streams(m, pred_a, pred_t)
        row = {"id": m["id"], "command": m["command"],
               "target_label": canonical_label(m["target_label"]),
               "target_duration_s": m["target_duration_s"],
               "expected_verdict": m["expected_verdict"],
               "n_frames": m["n_frames"],
               "clip_length_s": m["duration_s"], "sources": {}}
        for src, stream in streams.items():
            ladder = []
            # Nov 11 rules at the old time base (t = frame index): for the
            # classifiers this is the stored Nov 11 verdict; for ground
            # truth the Nov 10 labels-mode check ran at clip time instead.
            if src != "ground_truth":
                old = (pred_a if src == "model_a" else pred_t)[m["id"]]
                v = judge(to_obs(stream, 1.0), m["target_label"],
                          m["target_duration_s"], command=m["command"],
                          rules="nov11")
                assert v.verdict == old["verdict"], (src, m["id"])
                ladder.append(("nov11_frozen_frame_index_time", vd(v)))
                stored = old["verdict"]
            for step in STEPS:
                v = judge_step(to_obs(stream, 1.0 / STORED_FPS),
                               m["target_label"], m["target_duration_s"],
                               m["command"], step)
                ladder.append((step[0] + "_clip_time", vd(v)))
            if src == "ground_truth":
                stored = ladder[0][1]["verdict"]   # Nov 10 labels mode
            app = ladder[-1][1]
            row["sources"][src] = {
                "stored_nov11_verdict": stored,
                "app_verdict": app["verdict"],
                "app": app,
                "matches_expected_nov11": stored == m["expected_verdict"],
                "matches_expected_app": app["verdict"] == m["expected_verdict"],
                "first_step_that_changed_the_verdict":
                    first_change(stored, ladder),
                "steps": {name: v for name, v in ladder},
            }
        rows.append(row)
    summary = {}
    for src in ("ground_truth", "model_a", "tuned"):
        summary[src] = {
            "nov11_matches_expected": sum(
                r["sources"][src]["matches_expected_nov11"] for r in rows),
            "app_matches_expected": sum(
                r["sources"][src]["matches_expected_app"] for r in rows),
            "app_hold_shape_matches_expected": sum(
                r["sources"][src]["steps"]["app_numbers_hold_clip_time"]
                ["verdict"] == r["expected_verdict"] for r in rows),
            "verdicts_changed_vs_nov11": sum(
                r["sources"][src]["app_verdict"]
                != r["sources"][src]["stored_nov11_verdict"] for r in rows),
            "sequences": len(rows),
        }
    return rows, summary


# ------------------------------------------------------------ scenarios
def scenarios_json():
    data = json.load(open(HERE / "scenarios.json"))
    dur = data["config"]["target_duration_s"]
    out = []
    for sc in data["scenarios"]:
        obs = [FrameObservation(t=i / 2.0, label=x)
               for i, x in enumerate(sc["labels"])]
        ladder = [(s[0], vd(judge_step(obs, sc["target_label"], dur,
                                       sc["command"], s))) for s in STEPS]
        assert ladder[0][1]["verdict"] == sc["expected_verdict"], sc["id"]
        out.append({"id": sc["id"], "command": sc["command"],
                    "target_label": canonical_label(sc["target_label"]),
                    "stored_expected_verdict": sc["expected_verdict"],
                    "app_verdict": ladder[-1][1]["verdict"],
                    "app_reason": ladder[-1][1]["reason"],
                    "first_step_that_changed_the_verdict":
                        first_change(sc["expected_verdict"], ladder),
                    "steps": dict(ladder)})
    return out


DISPLAY_TO_LABEL = {"Sit": "sitting", "Stand": "standing",
                    "Down": "lying_down", "Walk": "walking"}
WORD_TO_LABEL = {"sitting": "sitting", "standing": "standing",
                 "lying down": "lying_down", "walking": "walking"}


def parse_model_b(user):
    """Rebuild (command, target, duration, frames, stated) from the text."""
    stated = user.rsplit("Verdict: ", 1)[1].strip()
    cmd_text = user.split("Command: ", 1)[1].split(". Target:", 1)[0]
    dur = float(user.split("Target: ", 1)[1].split(" seconds", 1)[0])
    if cmd_text.startswith("Stay"):
        command = "stay"
        target = WORD_TO_LABEL[cmd_text[cmd_text.index("(") + 1:-1]]
    else:
        command = cmd_text.lower()
        target = DISPLAY_TO_LABEL[cmd_text]
    seq = user.split("Sequence: [", 1)[1].rsplit("] Verdict:", 1)[0]
    obs = []
    for i, part in enumerate(seq.split(", ")):
        ts, lab = part.split(": ", 1)
        assert float(ts.rstrip("s")) == i
        if lab == "No dog":
            obs.append(FrameObservation(t=float(i), label=None,
                                        dog_present=False))
        elif lab == "Unclear":     # low-confidence frame, label not kept
            obs.append(FrameObservation(t=float(i), label=None))
        else:
            obs.append(FrameObservation(t=float(i),
                                        label=DISPLAY_TO_LABEL[lab],
                                        confidence=0.95))
    return command, cmd_text, target, dur, obs, stated


def model_b_scenarios():
    rows = []
    for n, line in enumerate(open(MODEL_B), start=1):
        if not line.strip():
            continue
        user = json.loads(line)["messages"][1]["content"]
        command, cmd_text, target, dur, obs, stated = parse_model_b(user)
        ladder = [(s[0], vd(judge_step(obs, target, dur, command, s)))
                  for s in STEPS]
        assert ladder[0][1]["verdict"] == stated, (n, user)
        rows.append({"line": n, "command": cmd_text, "target_label": target,
                     "target_duration_s": dur, "n_frames": len(obs),
                     "stored_verdict": stated,
                     "app_verdict": ladder[-1][1]["verdict"],
                     "app_reason": ladder[-1][1]["reason"],
                     "app_hold_shape_verdict":
                         ladder[2][1]["verdict"],
                     "first_step_that_changed_the_verdict":
                         first_change(stated, ladder),
                     "steps": dict(ladder)})
    return rows


def reason_key(reason):
    """Reason text with its numbers removed, to group like cases."""
    out, skip = [], False
    for ch in reason:
        if ch.isdigit() or (ch == "." and skip):
            if not skip:
                out.append("#")
            skip = True
        else:
            skip = False
            out.append(ch)
    return "".join(out)


def summarise(rows, stored_key):
    diff = [r for r in rows if r["app_verdict"] != r[stored_key]]
    hold_diff = [r for r in rows
                 if r["steps"]["app_numbers_hold"]["verdict"] != r[stored_key]]
    groups = Counter()
    for r in diff:
        groups[(r[stored_key], r["app_verdict"],
                r["first_step_that_changed_the_verdict"],
                r["command"].split(" (")[0],
                reason_key(r["app_reason"]))] += 1
    return {
        "total": len(rows),
        "frozen_nov11_rules_reproduce_stored": sum(
            r["steps"]["nov11_frozen"]["verdict"] == r[stored_key]
            for r in rows),
        "app_rules_agree_with_stored": len(rows) - len(diff),
        "app_rules_differ_from_stored": len(diff),
        "app_rules_hold_shape_agree_with_stored": len(rows) - len(hold_diff),
        "app_rules_hold_shape_differ_from_stored": len(hold_diff),
        "stored_verdict_counts": dict(Counter(r[stored_key] for r in rows)),
        "app_verdict_counts": dict(Counter(r["app_verdict"] for r in rows)),
        "transitions": {f"{a} -> {b}": c for (a, b), c in Counter(
            (r[stored_key], r["app_verdict"]) for r in diff).items()},
        "differences_by_cause": [
            {"stored": k[0], "app": k[1], "first_changed_at": k[2],
             "command": k[3], "app_reason_pattern": k[4], "count": c}
            for k, c in sorted(groups.items(), key=lambda kv: -kv[1])],
        "differing_ids": [r.get("id", r.get("line")) for r in diff],
    }


def main():
    seq_rows, seq_summary = run_sequences()
    scen = scenarios_json()
    mb = model_b_scenarios()
    result = {
        "run_date": "2026-11-25",
        "what": "Offline re-judge of stored label streams with the Android "
                "app's state-machine rules. No classifier was run and no "
                "network call was made; the per-frame predictions are the "
                "ones recorded on Nov 11.",
        "time_base": {
            "real_sequences": "t = frame_index / 2 (frames were extracted "
                              "at 2 fps, so this is real clip time); the "
                              "Nov 11 run used t = frame index",
            "scenarios_json": "t = i / 2 (2 fps), as in the Nov 10 check",
            "model_b_scenarios": "t = i (1 step = 1 s, as written in the "
                                 "scenario text)",
        },
        "rules": rules_description("app"),
        "steps": {
            "nov11_frozen": "state_machine_nov11.py, Nov 11 numbers",
            "new_rules_nov11_numbers_hold": "state_machine.py rules, Nov 11 "
                                            "numbers, every command a hold",
            "app_numbers_hold": "state_machine.py rules and app numbers, "
                                "every command a hold of the stated "
                                "duration",
            "app": "the app: Sit / Down reach-then-hold, others hold",
        },
        "sequences": seq_rows,
        "sequences_summary": seq_summary,
        "scenarios_json": {"summary": summarise(scen,
                                                "stored_expected_verdict"),
                           "scenarios": scen},
        "model_b_scenarios": {"file": "gemini-finetune/model-b-scenarios/"
                                      "scenarios.jsonl",
                              "summary": summarise(mb, "stored_verdict"),
                              "scenarios": mb},
    }
    OUT.write_text(json.dumps(result, indent=1))

    print(f"{'sequence':28s} {'expected':10s} "
          f"{'A nov11':10s} {'A app':12s} {'tuned nov11':12s} {'tuned app':12s}"
          f" {'GT app':10s}")
    for r in seq_rows:
        s = r["sources"]
        print(f"{r['id']:28s} {r['expected_verdict']:10s} "
              f"{s['model_a']['stored_nov11_verdict']:10s} "
              f"{s['model_a']['app_verdict']:12s} "
              f"{s['tuned']['stored_nov11_verdict']:12s} "
              f"{s['tuned']['app_verdict']:12s} "
              f"{s['ground_truth']['app_verdict']:10s}")
        for src in ("model_a", "tuned", "ground_truth"):
            print(f"    {src:12s} {s[src]['app']['reason']}")
    print(json.dumps(seq_summary, indent=1))
    for name, block in (("scenarios.json", result["scenarios_json"]),
                        ("model-b scenarios.jsonl",
                         result["model_b_scenarios"])):
        sm = block["summary"]
        print(f"\n{name}: frozen Nov 11 rules reproduce "
              f"{sm['frozen_nov11_rules_reproduce_stored']}/{sm['total']}; "
              f"app rules agree {sm['app_rules_agree_with_stored']}/"
              f"{sm['total']} (hold shape only: "
              f"{sm['app_rules_hold_shape_agree_with_stored']}/{sm['total']})")
        print("  transitions:", sm["transitions"])
        for g in sm["differences_by_cause"]:
            print(f"  {g['count']:3d}  {g['stored']} -> {g['app']}  "
                  f"[{g['first_changed_at']}] {g['command']}: "
                  f"{g['app_reason_pattern']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
