# End-to-end video evaluation — Dog Training Assistant (CS663 Project 2)

Run date: 2026-11-11. This is the full-pipeline video check the project was
missing: until now the sequence layer was judged in **labels mode**
(human-verified labels fed to the state machine), which validates the
judging logic but says nothing about the classifiers on real video. Here
the actual video frames went through the actual classifiers, and the
resulting label streams were judged by the state machine.

## Method

- **Sequences:** the same 6 real-video sequences as the labels-mode check
  (`video-sequences/`, manifest.json): seq01 standing/Africanis,
  seq02 lying/Batumi, seq03 walking/Mission, seq04 lying/Snoring,
  seq05 lying/Panting, seq06 standing/Wagging (short). 111 frames total.
- **Classifiers:**
  - **Model A (local):** `model-a/model_a.keras` (MobileNetV3Small, 4
    classes), run with `~/tfvenv/bin/python`. Preprocessing exactly as in
    `model-a/train_model_a.py`: PIL RGB, resize 224x224, float32 raw
    0..255 pixels, no extra rescaling (the backbone contains its own
    Rescaling layer). Softmax confidence is passed to the state machine,
    so frames below its 0.6 threshold count as uncertain.
  - **Tuned Gemini (live):** the deployed tuned endpoint
    (`POST /api/fine-tuned-gemini`, fixed dataset prompt), called via
    `sequence_evaluate.py`'s `backend_label` (same prompt, same retries,
    same call the Android app makes). Raw reply text is recorded per
    frame; a call failing after all retries counts as a no-label frame.
- **State machine:** `state_machine.py` as it stood at run time, with the
  break-time rule from the updated proposal (5.3): `break_at_s` is the
  first raw observation of the wrong posture, not the confirmation
  frame. Config identical to the labels-mode verification:
  `smooth_window=3`, settle grace 2s, break confirm 1s, uncertain
  timeout 3s, max uncertain ratio 0.4; target and duration from the
  manifest. No EARLY_BREAK occurred in either run, so the new break-time
  rule was not exercised here.
- **Sampling rate and time base — read this before quoting the numbers:**
  the stored frames were extracted from the source clips at **2 fps**
  (20 frames per 10 s clip). This run feeds them at **t = frame index**,
  i.e. one stored frame per machine-second — the 1 fps effective rate of
  the Android app's sampler, which is what the task specified. That is
  NOT the same time base as the labels-mode verification (t = i/2 at the
  stored 2 fps). Consequences, stated plainly: (a) the machine sees a
  10 s clip as a 20 s interval, so an 8 s hold needs 8 of 20 frames
  instead of 16 of 20 — SUCCESS is easier to reach here than in the
  labels-mode check; (b) a real app attempt of the same clip would only
  produce ~10 frames, so this run gives each classifier twice as many
  frames as a live attempt would. These are app-equivalent verdicts on
  stored frames, not a real-time run. The verdicts are still comparable
  between the two classifiers here, because both were fed the identical
  frame streams at the identical time base.

## Per-sequence results

| Sequence | Target (dur) | Expected | Model A verdict | Tuned verdict | Match A / tuned |
|---|---|---|---|---|---|
| seq01 standing, Africanis (20f) | standing (8s) | SUCCESS | **UNCERTAIN** — target never established within 2s settle grace | SUCCESS — held 8.0s | N / Y |
| seq02 lying, Batumi (20f) | lying down (8s) | SUCCESS | SUCCESS — held 8.0s | SUCCESS — held 8.0s | Y / Y |
| seq03 walking, Mission (24f) | walking (10s) | SUCCESS | SUCCESS — held 10.0s | SUCCESS — held 10.0s | Y / Y |
| seq04 lying, Snoring (20f) | lying down (8s) | SUCCESS | SUCCESS — held 8.0s | SUCCESS — held 8.0s | Y / Y |
| seq05 lying, Panting (20f) | lying down (8s) | SUCCESS | SUCCESS — held 8.0s | SUCCESS — held 8.0s | Y / Y |
| seq06 standing, Wagging, short (7f) | standing (8s) | UNCERTAIN (incomplete) | UNCERTAIN — target never established within 2s settle grace | UNCERTAIN — held 6.0s of 8s, clip ended | Y* / Y |

\* seq06 Model A: the verdict matches the expected UNCERTAIN, but for the
wrong reason. The dog was standing throughout and the clip is too short
to succeed, so the expected reason is "incomplete hold". Model A never
recognised standing at all, so the machine gave up at the 2 s settle
grace instead. Verdict-right, reason-wrong — it does not count as the
system working.

## Attempt-level and frame-level accuracy

| Classifier | Attempt-level (verdicts) | Frame-level (labels) |
|---|---|---|
| Model A (local, MobileNetV3Small) | **5/6 = 83.3%** (4 clean + 1 right-verdict-wrong-reason) | **82/111 = 73.9%** |
| Tuned Gemini (live endpoint) | **6/6 = 100%** | **110/111 = 99.1%** |
| Labels mode (human labels, judging-logic check only) | 6/6 | n/a |

Live-call reliability (tuned run): **0 failed calls out of 111** after
the harness's retries, **0 unparseable replies** — every reply was a
clean single label (`standing`, `lying down`, …). Total elapsed
384.5 s for 111 calls, i.e. **~3.5 s per frame**. That is a real
constraint on the app: at a 1 fps sampling target the per-frame call
takes ~3.5 s, so the sampler cannot keep up in wall-clock time and the
app's fixed-delay sampler skips ticks — a 10 s target takes roughly
three times longer in real time than the target suggests. Verdicts are
unaffected (the machine counts frames, not wall clock), but the demo
should not be expected to run in real time.

Note: the earlier live proof `live_seq06_tuned.json` (held 3.0s) was
recorded at the harness default fps=2. This run re-ran seq06 at fps=1
and overwrote that file (held 6.0s now); the verdict is UNCERTAIN
either way. The old value is still quoted in `verification_run.txt`.

## Analysis — where the video verdicts went wrong

**Model A cannot see standing in these clips.** This is the whole story
of its 5/6, and it matches the test-set confusion matrix (standing
recall 0.633, 11 standing→sitting) rather than being a video-only
fluke:
- seq01: all 20 frames called **sitting** (confidence 0.77–0.95) for a
  dog that is standing in grass. The dog's front legs are folded under
  it in a resting stand and the camera is low; the model reads the
  compact body as a sit, confidently, on every frame. Smoothing cannot
  save a systematic error — there is no correct vote in the window.
- seq06: all 7 frames called **walking** (confidence 0.83–0.99) for a
  dog standing and wagging its tail. The tail motion reads as gait.
  Again systematic, again every frame.
- seq03: the only partial failure — 2 of 24 frames called standing
  (t=13, t=15) mid-stride. Here smoothing did its job: the two wrong
  votes never formed a majority and the SUCCESS survived.
- Lying down is the reliable class on video exactly as on the test set:
  seq02, seq04, seq05 are 20/20 each.

Model A's video frame accuracy (73.9%) is essentially its test-set
accuracy (74.9%) — the video run did not expose a new failure mode, it
exposed the same one (standing) on harder, real footage.

**The tuned endpoint is nearly perfect on these clips** (110/111). Its
single miss is seq03 frame_019 (t=18): a walking dog called standing.
One frame, smoothed away, SUCCESS unaffected. The endpoint also
correctly reads both standing clips that defeat Model A, including the
tail-wagging dog. That is the measured gap between the on-device student
model and the cloud fine-tuned model on real video, and it is why the
deployed per-frame classifier is the tuned endpoint rather than Model A.

**What the state machine did and did not do:** smoothing saved the two
single-run glitches (Model A seq03, tuned seq03) and correctly refused
to judge seq01 for Model A (never established → UNCERTAIN rather than a
false SUCCESS or a false break). It did not — and cannot — rescue a
classifier that is wrong on every frame. No EARLY_BREAK was produced by
either classifier on any sequence, because none of these clips contains
a genuine break; the break path remains proven only by the synthetic
scenarios and unit tests, as already recorded in TESTING.md.

## Reproducibility

```bash
# Model A (local) — from video-sequence-layer/
~/tfvenv/bin/python evaluate_model_a_video.py
# -> results/video_predictions_model_a.json

# Tuned Gemini (live) — from video-sequence-layer/
python3 run_live_tuned_video.py
# -> results/video_predictions_tuned.json
# -> results/live_<seq>_tuned.json per sequence
#    (uses sequence_evaluate.py:backend_label; ~6-7 min for 111 calls)
```

Per-frame predictions (frame, predicted label, confidence / raw text,
ground-truth label, correct) are in the two JSON files above. The
labels-mode check that these numbers build on is
`results/verification_run.txt` (`sequence_evaluate.py --all-sequences`).

## Files

- `video-sequence-layer/results/video_evaluation.md` (this file)
- `video-sequence-layer/results/video_predictions_model_a.json`
- `video-sequence-layer/results/video_predictions_tuned.json`
- `video-sequence-layer/results/live_seq01_stand_africanis_tuned.json`
  … `live_seq05_down_panting_tuned.json`, `live_seq06_stand_wagging_short_tuned.json`
  (per-sequence tuned verdicts; seq06 overwrites the earlier fps=2 proof)
- `video-sequence-layer/evaluate_model_a_video.py` (Model A driver)
- `video-sequence-layer/run_live_tuned_video.py` (live tuned driver)
- `video-sequence-layer/results/live_run_tuned.log` (full live-run log)
