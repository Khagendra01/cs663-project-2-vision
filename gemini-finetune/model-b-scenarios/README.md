# Model B Scenario Dataset — CS663 Project 2, Dog Training Assistant

Proposal §6/§8: Model B is the fine-tuned Gemini "expert trainer". It
receives the command, the target duration, and the sequence of postures
the frame classifier saw over the attempt, and returns an evaluation
plus actionable advice. §8 promised **~150–200 synthetic training
scenarios** mapping posture sequences to expert trainer feedback.
This folder is that dataset: **180 scenarios**, generated
programmatically — not hand-written.

## Files

| File | What it is |
|---|---|
| `build_scenarios.py` | The generator. Run it to rebuild `scenarios.jsonl` (seeded, reproducible: seed 20261110). |
| `scenarios.jsonl` | 180 examples, one JSON object per line, in the proposal's fine-tuning shape (below). |

## Format

```json
{"messages": [
  {"role": "system", "content": "You are an expert dog training assistant analyzing posture sequences."},
  {"role": "user", "content": "Command: Stay. Target: 5 seconds. Sequence: [0s: Sit, 1s: Sit, 2s: Sit, 3s: Stand, 4s: Walk] Verdict: EARLY_BREAK"},
  {"role": "model", "content": "The dog broke the command after holding sitting for ... Do not reward ..."}
]}
```

Sequences are label streams at 1 step = 1 second. `No dog` marks frames
with no dog detected; `Unclear` marks frames too low-confidence to use.

## How the verdicts are guaranteed consistent

The verdict in each example is **not** written by a template. Every
generated label stream is fed through the project's real temporal
state machine (`../../video-sequence-layer/state_machine.py`,
`CommandSession` with smooth_window 3, settle grace 2s, break confirm
1s, uncertain timeout 3s), and the builder **asserts** the machine's
verdict equals the pattern's intended verdict — a pattern that does not
judge as designed fails the build. After writing, a validation pass
re-parses every line, reconstructs the frames from the text, re-judges
every sequence, and re-asserts every stated verdict, role order, system
prompt, and user-string uniqueness.

That referee relationship cuts both ways: two generator patterns had
to be redesigned around the machine's smoothing (a break starting
within 2 frames of the target smooths into a success, and its held
time is whatever the machine measures) — the machine's numbers are the
ones quoted in the advice text.

## Counts (this build)

- **Per verdict: SUCCESS 60 · EARLY_BREAK 80 · UNCERTAIN 40** (180 total;
  early break is deliberately the largest class — it is the
  teaching-rich case)
- Per command: Sit 40 · Down 49 · Stand 28 · Walk 31 · Stay (in a named
  posture) 32
- Behaviour patterns: clean hold 18, held past target 9, settle then
  hold 11, noisy glitches 16, dip and recover 6, early break 46,
  break near target 18, recover then break 16, never settles 15,
  dog leaves frame 12, low confidence heavy 8, incomplete recording 5
- Target durations range 3–30 seconds

## The advice rules

The model text is generated from fixed trainer rules — the same rules
the app's runtime feedback path follows:

- **SUCCESS** — praise, state the held time, reward *now* (timing
  matters), then progress one step: next target = target + 2s (≤5s),
  +5s (≤15s), or +10s, or add distance/distraction — never both at once.
- **EARLY_BREAK** — state the machine-measured held time, **do not
  reward the break**, set the next target *below* the held time
  (held − 1s, minimum 2s), reduce distance/distraction, re-cue and end
  the session on an easy success.
- **UNCERTAIN** — **no judgement of the dog, ever.** Say the attempt
  could not be judged and give the concrete fix for the specific
  reason: settle/framing for never-settles, camera distance/angle for
  leaves-frame, light/distance/lens for low-confidence, and record the
  full attempt for incomplete recordings.

## Disclosure

These scenarios are **synthetic, as planned in the proposal** — the
sequences are generated label streams, not recordings of real dogs, and
the advice is rule-generated template text (with varied phrasing), not
hand-written by a trainer. They are the fine-tuning corpus for Model B:
their job is to teach the *format, verdict discipline, and advice
rules*, consistently with the state machine that will actually judge
live attempts. In the app today, the runtime feedback path calls base
Gemini with the same rules in its prompt; fine-tuning on this set is
what turns that prompted behaviour into Model B. Real recorded
sessions should be added (and would outrank synthetic rows) as the
group collects them.
