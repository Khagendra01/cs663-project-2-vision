#!/usr/bin/env python3
"""Generate the Model B scenario dataset (proposal section 8).

Proposal section 8 promised ~150-200 synthetic training scenarios that
map posture sequences to expert trainer feedback, to fine-tune the
Gemini reasoning model (Model B). This builder generates 180 of them
PROGRAMMATICALLY: command x target duration x behaviour pattern are
varied systematically (seeded RNG for reproducibility), the trainer
advice is produced by fixed rules (below), and - critically - the
verdict stated in every example is not asserted by the template: the
label stream is fed through the real temporal state machine
(../video-sequence-layer/state_machine_nov11.py - the Nov 11 rules,
imported) and the builder asserts the machine's verdict equals the pattern's intended verdict.
If a generated pattern does not judge the way it was designed to, the
build fails instead of emitting an inconsistent example.

Sequences are label streams at 1 step = 1 second. The state machine is
run with the video layer's demo config (SessionConfig with
smooth_window=3; settle grace 2s, break confirm 1s, uncertain timeout
3s) - the same config as video-sequence-layer/scenarios.json.

Advice rules (the same rules the app's runtime feedback follows):
  SUCCESS      praise, state the held time, reward now, and progress:
               next target = target + step (bigger duration, or add
               distance/distraction once duration is long).
  EARLY_BREAK  state the held time from the machine verdict, do NOT
               reward the break, next target set BELOW the held time,
               reduce distance/distraction, re-cue and end on a success.
  UNCERTAIN    no judgement of the dog at all: say the attempt could
               not be judged and give the concrete re-run fix for the
               specific reason (framing, lighting, keep the dog in
               frame, record the full attempt).

Output: scenarios.jsonl, one JSON object per line in the proposal's
fine-tuning shape {"messages":[system, user, model]}. Identical user
strings are deduped. Final validation re-parses every line, checks the
roles/system prompt, re-runs every sequence through the state machine
and re-asserts every stated verdict.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = HERE.parents[1]                       # .../files
sys.path.insert(0, str(FILES / "video-sequence-layer"))
# Pinned to the frozen Nov 11 rules this dataset was generated and
# validated with (commit 947da56), so a rebuild reproduces
# scenarios.jsonl. The current, app-matching rules are in
# state_machine.py; video-sequence-layer/rerun_nov25.py re-judges this
# file with them and reports where the verdicts differ.
from state_machine_nov11 import (FrameObservation,  # noqa: E402
                                 SessionConfig, run_sequence)

SYSTEM = ("You are an expert dog training assistant analyzing posture "
          "sequences.")
OUT = HERE / "scenarios.jsonl"
SEED = 20261110
TARGET_TOTAL = {"SUCCESS": 60, "EARLY_BREAK": 80, "UNCERTAIN": 40}

# config the verdicts are computed with (matches scenarios.json demo cfg)
CFG_BASE = dict(smooth_window=3, settle_grace_s=2.0, break_confirm_s=1.0,
                uncertain_timeout_s=3.0)

DISPLAY = {"sitting": "Sit", "standing": "Stand",
           "lying_down": "Down", "walking": "Walk"}
POSTURE_WORD = {"sitting": "sitting", "standing": "standing",
                "lying_down": "lying down", "walking": "walking"}
# command -> (display name, target label)
COMMANDS = [
    ("Sit", "sitting"),
    ("Down", "lying_down"),
    ("Stand", "standing"),
    ("Walk", "walking"),
    ("Stay", None),          # stay pairs with a posture, filled per scenario
]
DURATIONS = [3, 5, 5, 8, 8, 10, 10, 15, 15, 20, 30]

rng = random.Random(SEED)


def machine_config(target_s):
    return SessionConfig(target_duration_s=float(target_s), **CFG_BASE)


def judge(frames, target_label, target_s):
    """frames: list of (label|None, confidence|None, dog_present)."""
    obs = [FrameObservation(t=float(i), label=lab, confidence=conf,
                            dog_present=dog)
           for i, (lab, conf, dog) in enumerate(frames)]
    return run_sequence(obs, target_label, machine_config(target_s))


def T(label):
    return (label, 0.95, True)


def GLITCH(label):
    return (label, 0.95, True)      # wrong-label frame, confident


def LOWCONF(label):
    return (label, 0.45, True)      # real label, too unsure to count


def NODOG():
    return (None, None, False)


OTHERS = {
    "sitting": ["standing", "lying_down", "walking"],
    "standing": ["sitting", "walking", "lying_down"],
    "lying_down": ["sitting", "standing", "walking"],
    "walking": ["standing", "sitting", "lying_down"],
}


# ---------------------------------------------------------------- patterns
# Each returns (frames, intended_verdict, pattern_name). "held" for
# EARLY_BREAK patterns is the intended raw hold length in seconds.

def p_clean_hold(target, dur):
    n = dur + rng.choice([1, 2, 3])
    return [T(target)] * n, "SUCCESS", "clean hold"


def p_spare_hold(target, dur):
    # holds well past the target - success with time to spare
    n = dur + rng.choice([4, 6, 8])
    return [T(target)] * n, "SUCCESS", "held past target"


def p_settle_then_hold(target, dur):
    # one frame of another posture while the dog gets into position
    other = rng.choice(OTHERS[target])
    return [T(other)] + [T(target)] * (dur + 1), "SUCCESS", "settle then hold"


def p_glitchy_hold(target, dur):
    frames = [T(target)] * (dur + 2)
    spots = rng.sample(range(2, len(frames) - 1), k=min(3, dur // 3 + 1))
    for i in sorted(spots):
        if i - 1 in spots or i + 1 in spots:
            continue                      # never two glitches in a row
        kind = rng.random()
        if kind < 0.5:
            frames[i] = GLITCH(rng.choice(OTHERS[target]))
        else:
            frames[i] = LOWCONF(target)
    return frames, "SUCCESS", "noisy glitches"


def p_dip_recover(target, dur):
    # one-second dip out of the posture, recovers, still succeeds
    frames = [T(target)] * (dur + 3)
    dip_at = rng.randint(2, dur - 1)
    frames[dip_at] = GLITCH(rng.choice(OTHERS[target]))
    return frames, "SUCCESS", "dip and recover"


def p_early_break(target, dur):
    # n target frames, then a sustained move away. The smoother flips
    # 2 frames after the raw break, so n <= dur-2 keeps the machine's
    # held time below the target (a break at the boundary would smooth
    # into a success - the machine is the referee, patterns obey it).
    n = rng.randint(2, max(2, dur - 2))
    away = rng.choice(OTHERS[target])
    frames = [T(target)] * n
    frames += [T(away)] * rng.choice([4, 5])
    if rng.random() < 0.4:                # often wanders to a third posture
        third = rng.choice([o for o in OTHERS[target] if o != away])
        frames += [T(third)] * 2
    return frames, "EARLY_BREAK", "early break"


def p_break_near_target(target, dur):
    # the longest hold that still breaks: machine held time = dur-1
    away = rng.choice(OTHERS[target])
    frames = [T(target)] * (dur - 2) + [T(away)] * 4
    return frames, "EARLY_BREAK", "break near target"


def p_recover_then_break(target, dur):
    frames = [T(target)] * rng.choice([3, 4])
    frames.append(GLITCH(rng.choice(OTHERS[target])))   # the dip (1 frame)
    frames += [T(target)] * rng.choice([2, 3])          # recovers
    away = rng.choice(OTHERS[target])
    frames += [T(away)] * 4                              # then a real break
    # machine held time for this shape is (#target frames)+2; it must
    # stay below the target duration or the attempt judges as a success
    if frames.count(T(target)) > dur - 3:
        return None
    return frames, "EARLY_BREAK", "recover then break"


def p_never_settles(target, dur):
    pool = OTHERS[target]
    frames = [T(rng.choice(pool)) for _ in range(rng.choice([5, 6, 8]))]
    # guarantee the smoothed label is never the target in the grace window
    return frames, "UNCERTAIN", "never settles"


def p_leaves_frame(target, dur):
    frames = [T(target)] * rng.choice([2, 3, 4])
    frames += [NODOG()] * rng.choice([4, 5])
    if rng.random() < 0.5:                 # sometimes wanders back, too late
        frames += [T(target)] * 2
    return frames, "UNCERTAIN", "dog leaves frame"


def p_low_confidence(target, dur):
    # alternate good frames with unreadable ones; record ends mid-attempt.
    # Needs a long target: the stream must end before the duration while
    # >40% of frames are unreadable (the machine's UNCERTAIN ratio rule).
    if dur < 8:
        return None
    n = dur - 2
    frames = []
    for i in range(n):
        frames.append(T(target) if i % 2 == 0 else LOWCONF(target))
    return frames, "UNCERTAIN", "low confidence heavy"


def p_incomplete(target, dur):
    n = dur - rng.choice([1, 2])          # clean hold, recording stops early
    if n < 2:
        n = 2
    return [T(target)] * n, "UNCERTAIN", "incomplete recording"


PATTERNS = {
    "SUCCESS": [p_clean_hold, p_clean_hold, p_spare_hold, p_settle_then_hold,
                p_glitchy_hold, p_glitchy_hold, p_dip_recover],
    "EARLY_BREAK": [p_early_break, p_early_break, p_early_break,
                    p_break_near_target, p_recover_then_break],
    "UNCERTAIN": [p_never_settles, p_never_settles, p_leaves_frame,
                  p_leaves_frame, p_low_confidence, p_incomplete],
}


# ------------------------------------------------------------------- text
def fmt_seconds(x):
    x = float(x)
    return f"{x:.0f}" if x == int(x) else f"{x:.1f}"


def sequence_text(frames):
    parts = []
    for i, (lab, conf, dog) in enumerate(frames):
        if not dog:
            parts.append(f"{i}s: No dog")
        elif conf is not None and conf < 0.6:
            parts.append(f"{i}s: Unclear")
        else:
            parts.append(f"{i}s: {DISPLAY[lab]}")
    return "[" + ", ".join(parts) + "]"


def next_up(target_s):
    return target_s + (2 if target_s <= 5 else 5 if target_s <= 15 else 10)


def next_down(held_s):
    return max(2, int(held_s) - 1)


SUCCESS_TEMPLATES = [
    "Well done - your dog held {posture} for the full {dur} seconds. Mark and reward right away while the posture is still correct. Next session, build slowly: try {next} seconds at the same distance before adding any distractions.",
    "Success. The dog settled into {posture} and held it for {dur} seconds without breaking. Reward now - timing matters, so deliver the treat within a second or two. For the next attempt, increase the target to {next} seconds, or keep {dur} seconds and take one step further away.",
    "That is a clean hold: {dur} seconds of {posture}, exactly what you asked for. Reward and praise, then make the next repetition slightly harder - {next} seconds, or the same duration with a mild distraction. Only raise one difficulty at a time.",
    "Your dog nailed it, holding {posture} for the full {dur} seconds. Reward generously this time. To progress, set the next target to {next} seconds; if the dog breaks at the longer duration, drop back to {dur} seconds and rebuild.",
]
SUCCESS_SPARE_EXTRA = [
    " The dog actually kept holding past the target, which means this duration is solid - the increase is well earned.",
    " Holding beyond the target like that is a good sign the behaviour is reliable at this level.",
]
SUCCESS_GLITCH_EXTRA = [
    " There was a brief wobble mid-hold, but the dog recovered on its own - that still counts as a success, and recovering is exactly the skill you want.",
    " Ignore the momentary glitch in the middle; the dog corrected itself without help, which is good training progress.",
]
EARLY_TEMPLATES = [
    "The dog broke the command after holding {posture} for {held} seconds of the {dur}-second target. Do not reward that repetition - rewarding the break teaches the wrong lesson. Re-cue calmly, set the next target to {next} seconds (below what the dog just managed), and reward as soon as the dog reaches it so the session ends on a success.",
    "Early break: your dog held {posture} for {held} seconds, then left the position before the {dur}-second target. No treat for this one. Shorten the next attempt to {next} seconds and reduce the challenge - move closer to the dog and work somewhere quieter. Build back up in small steps.",
    "The hold lasted {held} seconds out of {dur} before the dog broke {posture}. That tells you the current step is a bit too hard. Do not reward the break; reset without frustration, ask again with a {next}-second target, and reward the moment the dog completes it. Two or three easy wins beat one failed stretch.",
    "Your dog held {posture} for {held} seconds and then broke before the {dur}-second goal. Withhold the reward for the break itself, re-cue, and make the next repetition easier: {next} seconds, less distance, fewer distractions. End the session after a successful short hold.",
]
EARLY_RECOVER_NOTE = [
    " The dog did wobble and recover once earlier in the attempt, so it was trying - the shorter target will let that effort pay off.",
    "",
]
UNCERTAIN_TEMPLATES = {
    "never settles": "I could not judge this attempt: the dog never settled into {posture} during the opening seconds, so there was no hold to measure. This is not scored as a failure. Re-run it with the dog already calm and close to the starting position, keep the camera steady and the whole dog in frame, and start the sequence as you give the command.",
    "dog leaves frame": "I could not judge this attempt: the dog left the frame partway through, so the hold could not be tracked to the end. No judgement on the dog - this is a filming problem, not a training result. Re-run with the camera further back or angled so the dog stays in view for the whole {dur} seconds, ideally on a tripod.",
    "low confidence heavy": "I could not judge this attempt: too many frames were unclear, which usually means poor light, motion blur, or the dog being too far from the camera. Do not read anything into this about the dog's training. Re-run in better light, move the camera closer, and wipe the lens - then the posture can be tracked reliably.",
    "incomplete recording": "I could not judge this attempt: the recording stopped after {seen} seconds, before the {dur}-second target could be confirmed. The dog may well have been on track - there is just not enough evidence. Re-run and let the recording continue until the verdict appears.",
}


def advice_for(verdict_word, pattern, cmd_display, target_label, dur,
               verdict, frames):
    posture = POSTURE_WORD[target_label]
    if verdict_word == "SUCCESS":
        text = rng.choice(SUCCESS_TEMPLATES).format(
            posture=posture, dur=fmt_seconds(dur), next=next_up(dur))
        if pattern == "held past target":
            text += rng.choice(SUCCESS_SPARE_EXTRA)
        if pattern in ("noisy glitches", "dip and recover"):
            text += rng.choice(SUCCESS_GLITCH_EXTRA)
        return text
    if verdict_word == "EARLY_BREAK":
        text = rng.choice(EARLY_TEMPLATES).format(
            posture=posture, held=fmt_seconds(verdict.held_s),
            dur=fmt_seconds(dur), next=next_down(verdict.held_s))
        if pattern == "recover then break":
            text += rng.choice(EARLY_RECOVER_NOTE)
        return text
    # UNCERTAIN: pattern-keyed, no judgement of the dog
    return UNCERTAIN_TEMPLATES[pattern].format(
        posture=posture, dur=fmt_seconds(dur), seen=len(frames) - 1)


# -------------------------------------------------------------- generation
def build():
    counts = Counter()
    per_command = Counter()
    per_pattern = Counter()
    seen_users = set()
    rows = []
    attempts = 0
    while sum(counts.values()) < sum(TARGET_TOTAL.values()):
        attempts += 1
        if attempts > 20000:
            raise RuntimeError(f"could not fill quotas: {dict(counts)}")
        want = rng.choice([v for v, cap in TARGET_TOTAL.items()
                           if counts[v] < cap])
        pattern_fn = rng.choice(PATTERNS[want])
        cmd_display, target_label = rng.choice(COMMANDS)
        if target_label is None:                       # Stay + posture
            target_label = rng.choice(list(DISPLAY))
            cmd_text = f"Stay ({POSTURE_WORD[target_label]})"
        else:
            cmd_text = cmd_display
        dur = rng.choice(DURATIONS)
        if want == "EARLY_BREAK" and dur < 5:
            dur = rng.choice([5, 8, 10])
        built = pattern_fn(target_label, dur)
        if built is None:
            continue
        frames, intended, pattern = built
        if intended != want:
            continue
        verdict = judge(frames, target_label, dur)
        assert verdict.verdict == intended, (
            f"pattern {pattern} intended {intended} but machine said "
            f"{verdict.verdict} ({verdict.reason}); frames={frames}")
        user = (f"Command: {cmd_text}. Target: {dur} seconds. "
                f"Sequence: {sequence_text(frames)} "
                f"Verdict: {verdict.verdict}")
        if user in seen_users:
            continue
        seen_users.add(user)
        model_text = advice_for(verdict.verdict, pattern, cmd_text,
                                target_label, dur, verdict, frames)
        rows.append({"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "model", "content": model_text}]})
        counts[verdict.verdict] += 1
        per_command[cmd_text.split(" (")[0]] += 1
        per_pattern[pattern] += 1
    return rows, counts, per_command, per_pattern


def validate(rows):
    """Re-parse the written file shape and re-assert every verdict."""
    users = set()
    counts = Counter()
    for row in rows:
        msgs = row["messages"]
        assert [m["role"] for m in msgs] == ["system", "user", "model"]
        assert msgs[0]["content"] == SYSTEM
        user = msgs[1]["content"]
        assert user not in users
        users.add(user)
        stated = user.rsplit("Verdict: ", 1)[1]
        # re-derive the frames from the sequence text and re-judge
        seq = user.split("Sequence: [", 1)[1].rsplit("] Verdict:", 1)[0]
        frames = []
        inv = {v: k for k, v in DISPLAY.items()}
        for part in seq.split(", "):
            lab = part.split(": ", 1)[1]
            if lab == "No dog":
                frames.append((None, None, False))
            elif lab == "Unclear":
                frames.append(("sitting", 0.45, True))
            else:
                frames.append((inv[lab], 0.95, True))
        cmd_text = user.split("Command: ", 1)[1].split(". Target:", 1)[0]
        dur = int(user.split("Target: ", 1)[1].split(" seconds", 1)[0])
        if cmd_text.startswith("Stay"):
            posture = cmd_text[cmd_text.index("(") + 1:-1]
            target_label = {v: k for k, v in POSTURE_WORD.items()}[posture]
        else:
            target_label = dict(COMMANDS)[cmd_text] if cmd_text != "Stay" \
                else None
            if target_label is None:
                target_label = dict(COMMANDS)[cmd_text]
        # 'Unclear' frames lose their true label in text form; they are
        # uncertain frames either way, which is all the machine uses.
        v = judge(frames, target_label, dur)
        assert v.verdict == stated, (user, v)
        counts[stated] += 1
    return counts


def main():
    rows, counts, per_command, per_pattern = build()
    rng.shuffle(rows)
    with open(OUT, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    # validate the actual written file, line by line
    parsed = [json.loads(line) for line in open(OUT) if line.strip()]
    vcounts = validate(parsed)
    assert vcounts == counts, (vcounts, counts)
    print(f"wrote {OUT}: {len(parsed)} scenarios")
    print(f"per verdict: {dict(counts)}")
    print(f"per command: {dict(per_command)}")
    print(f"per pattern: {dict(per_pattern)}")
    print(f"attempts used: validation re-judged every sequence - all "
          f"stated verdicts match the state machine")


if __name__ == "__main__":
    main()
