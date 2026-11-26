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

---

## Nov 25 re-run with the app's rules (clip time, 2 fps)

Run date: 2026-11-25. The Nov 11 sections above are unchanged and still
describe the Nov 11 run. This section re-judges the **same stored
per-frame predictions** with the state-machine rules the Android app now
uses. No classifier was run and **no network call was made**: Model A's
predictions and the tuned endpoint's replies are the ones recorded on
Nov 11 in `video_predictions_model_a.json` and `video_predictions_tuned.json`.
Machine-readable results: `video_evaluation_nov25.json`.

### What changed since Nov 11

- **Rules.** `state_machine.py` now has the same rules, defaults and
  reason strings as the app's `SequenceSession.java`: smoothing window 5
  confident frames, min confidence 0.65, settle grace 2 s, break confirmed
  after 0.6 s, lost / unclear for 1.5 s -> UNCERTAIN (in any phase, with
  the app's EARLY_BREAK exception when >= 2 consecutive wrong raw frames
  or a confirmed break come before losing the dog), max uncertain ratio
  0.4; reach time and break time measured on raw frames; SUCCESS only
  when the deciding raw frame shows the target. **Sit and Down** are now
  judged the way the app judges them: reach the posture within a 3 s
  response window, then hold it 0.6 s (the target duration is not used
  for them). Stand, Walk and Stay keep the hold-for-the-duration rule.
  The 21 unit tests of the app's state machine are ported to
  `test_state_machine.py`; with the original 17 and 4 new config tests,
  42/42 pass. The Nov 11 rules are kept unchanged in
  `state_machine_nov11.py`, and `--rules nov11` on `sequence_evaluate.py`
  / `evaluate_model_a_video.py` / `run_live_tuned_video.py` reproduces
  the Nov 10-11 results exactly (checked: same verdicts, reasons and
  smoothed sequences as the stored files).
- **Time base.** The frames were extracted at 2 fps, so this run feeds
  them at **t = frame_index / 2 = real clip time**. The Nov 11 run used
  t = frame index (one stored frame per machine-second), which made a
  10 s clip look like a 20 s interval; that caveat (a) above no longer
  applies. An 8 s hold now needs 17 frames (t = 0 to 8.0 s) instead of 9.
- **Command shape for the real clips.** The manifest's commands are
  `stand` (seq01, seq06), `walk` (seq03) and `down` (seq02, seq04,
  seq05). Under the app's rules the three `down` clips are judged as
  Down (3 s window + 0.6 s hold), not as an 8 s hold. Because that is a
  much weaker test of a clip in which the dog lies down throughout, the
  table also gives the verdict with every command judged as a hold of
  the manifest duration ("hold shape", i.e. the clip treated as a Stay
  in that posture).

### Per-sequence results (old = Nov 11, frame-index time; new = Nov 25, app rules, clip time)

| Sequence | Command, app shape | Expected | Model A old -> new | Tuned old -> new | Labels mode (human labels) new | New, hold shape (A / tuned / labels) |
|---|---|---|---|---|---|---|
| seq01 standing, Africanis (20f) | Stand, hold 8 s | SUCCESS | UNCERTAIN -> **UNCERTAIN** (never established within 2 s settle grace) | SUCCESS -> **SUCCESS** (held 8.0 s) | SUCCESS | UNCERTAIN / SUCCESS / SUCCESS |
| seq02 lying, Batumi (20f) | Down, reach 3 s + hold 0.6 s | SUCCESS | SUCCESS -> **SUCCESS** (reached 0.0 s, decided at 1.0 s) | SUCCESS -> **SUCCESS** (same) | SUCCESS | SUCCESS / SUCCESS / SUCCESS (held 8.0 s) |
| seq03 walking, Mission (24f) | Walk, hold 10 s | SUCCESS | SUCCESS -> **SUCCESS** (held 10.0 s) | SUCCESS -> **SUCCESS** (held 10.0 s) | SUCCESS | SUCCESS / SUCCESS / SUCCESS |
| seq04 lying, Snoring (20f) | Down, reach 3 s + hold 0.6 s | SUCCESS | SUCCESS -> **SUCCESS** (decided at 1.0 s) | SUCCESS -> **SUCCESS** (same) | SUCCESS | SUCCESS / SUCCESS / SUCCESS (held 8.0 s) |
| seq05 lying, Panting (20f) | Down, reach 3 s + hold 0.6 s | SUCCESS | SUCCESS -> **SUCCESS** (decided at 1.0 s) | SUCCESS -> **SUCCESS** (same) | SUCCESS | SUCCESS / SUCCESS / SUCCESS (held 8.0 s) |
| seq06 standing, Wagging, short (7f) | Stand, hold 8 s | UNCERTAIN (incomplete) | UNCERTAIN -> **UNCERTAIN** (never established within 2 s settle grace) | UNCERTAIN -> **UNCERTAIN** (incomplete, held 3.0 s; was 6.0 s) | UNCERTAIN (held 3.0 s) | UNCERTAIN / UNCERTAIN / UNCERTAIN |

| Classifier | Attempt-level, Nov 11 | Attempt-level, Nov 25 app rules | Nov 25, hold shape | Frame-level (unchanged, same predictions) |
|---|---|---|---|---|
| Model A (local, MobileNetV3Small) | 5/6 | **5/6** | 5/6 | 82/111 = 73.9% |
| Tuned Gemini (stored replies) | 6/6 | **6/6** | 6/6 | 110/111 = 99.1% |
| Labels mode (human labels) | 6/6 | **6/6** | 6/6 | n/a |

**No verdict changed.** Model A's seq06 is still verdict-right,
reason-wrong (never recognised standing), so the honest Model A count
remains 4 clean + 1 right-verdict-wrong-reason, as on Nov 11.

What did change, and why:
- **seq06, tuned: held 6.0 s -> 3.0 s.** The 7 frames now span the real
  clip (t = 0 to 3.0 s) instead of 0 to 6 machine-seconds. This is the
  same 3.0 s the labels-mode check reported; the verdict is UNCERTAIN
  (incomplete) either way.
- **Down clips (seq02, seq04, seq05) are decided after 3 frames.** The
  dog is lying from the first frame, so the reach is at 0.0 s and the
  0.6 s hold is confirmed on the first frame at or after 0.6 s, which at
  2 fps is t = 1.0 s. The rest of each clip is not judged. These clips
  also cannot test the 3 s response window (the dog never has to get
  into the posture). Judged as an 8 s hold they still give SUCCESS for
  all three label sources.
- **Model A, seq03: uncertain ratio 0 -> 3/21 = 0.143.** Three frames
  have confidence below the threshold (0.549, 0.582 and 0.507 at clip
  time 5.5, 8.0 and 9.0 s). At frame-index time the Nov 11 run reached
  its 10 s after 11 frames, before any of them; at clip time the 10 s
  hold takes 21 frames, so they are now inside the judged interval and
  count as unclear frames. 0.143 is under the 0.4
  limit and the gaps are single frames, so SUCCESS stands. Raising the
  threshold from 0.6 to 0.65 moved no frame: none of Model A's 111
  confidences lies between 0.6 and 0.65. The tuned endpoint returns text
  without a confidence, so the threshold does not apply to it.
- **No EARLY_BREAK on any real clip**, as on Nov 11 - none of these
  clips contains a real break, so the new break rules are exercised only
  by the unit tests and the synthetic scenarios below.

### Synthetic scenarios re-judged (files unchanged)

The scenario files were **not** edited. Their stored verdicts were set by
the Nov 10 rules; the frozen Nov 11 module still reproduces every one of
them (6/6 and 180/180). Today's numbers:

| Scenario set | Time base | App rules agree with stored | App rules, hold shape | Differ |
|---|---|---|---|---|
| `scenarios.json` (6) | 2 fps, t = i / 2 | **4/6** | 6/6 | 2 |
| `model-b-scenarios/scenarios.jsonl` (180) | 1 step = 1 s | **119/180** | 180/180 | 61 |

**Every difference comes from one rule: Sit and Down are now
reach-then-hold commands.** Judged with every command as a hold of the
stated duration, the new rules and numbers agree with all 186 stored
verdicts. The new rules did change some details without changing a
verdict: none of the 80 EARLY_BREAK rows in the Model B set changed its
held time; 12 UNCERTAIN rows report a shorter held time because the
lost rule now fires after 1.5 s instead of 3 s; one SUCCESS (line 126,
Stay walking, a single Stand frame at the 8 s mark) is decided one frame
later because SUCCESS now needs the deciding raw frame to show the
target.

`scenarios.json`, the 2 that differ:
- `scen04_stand_never_sits` (Sit, dog standing the whole time):
  UNCERTAIN -> **EARLY_BREAK**, "target posture 'sitting' not reached
  within the 3s response window". The app treats a dog clearly seen
  not sitting at the deadline as a failed command, not as "cannot judge".
- `scen06_down_dog_leaves_frame` (Down, lying 0-1.5 s, then no dog):
  UNCERTAIN -> **SUCCESS**, "reached 'lying_down' within the 3s response
  window and held it for the 0.6s confirmation hold". The 0.6 s hold was
  complete at t = 1.0 s, before the dog left. As a hold it is still
  UNCERTAIN (now "dog lost / unclear for 1.5s in a row").

`model-b-scenarios/scenarios.jsonl`, the 61 that differ (all Sit or
Down; 89 of the 180 rows are Sit or Down, and the other 28 are SUCCESS
under both rule sets). Lines are 1-based line numbers in the file:

| Stored -> app | Count | Why | Lines |
|---|---|---|---|
| EARLY_BREAK -> SUCCESS | 21 Down | dog reached the posture and held it at least 0.6 s, then broke before the row's "Target" duration; the app does not use a duration for Down | 10, 11, 14, 23, 25, 38, 47, 57, 71, 77, 86, 87, 92, 96, 124, 129, 138, 158, 162, 167, 180 |
| EARLY_BREAK -> SUCCESS | 21 Sit | same, for Sit | 17, 27, 29, 33, 41, 42, 46, 48, 55, 74, 80, 100, 113, 116, 128, 146, 149, 150, 165, 166, 168 |
| UNCERTAIN -> SUCCESS | 4 Down | dog lying from the start, then left the frame; the 0.6 s hold was done first | 2, 31, 88, 115 |
| UNCERTAIN -> SUCCESS | 3 Down, 1 Sit | recording ended before the "Target" duration; the 0.6 s hold was done | 34, 50, 101 (Down); 174 (Sit) |
| UNCERTAIN -> SUCCESS | 2 Down, 1 Sit | too many unclear frames overall (ratio rule at the end), but the first two clear frames already gave the 0.6 s hold | 26, 83 (Down); 153 (Sit) |
| UNCERTAIN -> EARLY_BREAK | 4 Down, 4 Sit | dog clearly seen in other postures through the 3 s window ("never settles"); the app calls that a failed command | 3, 28, 60, 73 (Down); 18, 43, 139, 159 (Sit) |

Verdict counts in the Model B set: stored SUCCESS 60 / EARLY_BREAK 80 /
UNCERTAIN 40; with the app's rules SUCCESS 113 / EARLY_BREAK 46 /
UNCERTAIN 21.

What this means: the stored verdicts are correct for the rules they were
built with, and the advice text matches them. But 61 Model B training
rows teach a duration rule for Sit and Down ("held 4 of 8 seconds,
early break") that the app no longer applies. The files were left as
they are; whether to regenerate the Sit / Down rows with the
reach-then-hold rule (or present them as Stay-in-posture holds, which
agree 180/180) is a decision for the group. The Model B rows are also
on a 1 s step, so the app's 5-frame smoother spans 5 s of scenario
time there; they test the verdict logic, not real timing.

### Reproducibility (Nov 25)

```bash
# from video-sequence-layer/ - stdlib only, no network
python3 -m unittest test_state_machine          # 42 tests
python3 rerun_nov25.py                          # -> results/video_evaluation_nov25.json
python3 sequence_evaluate.py --all-sequences    # app rules, labels mode: 6/6
python3 sequence_evaluate.py --scenarios        # app rules: 4/6 (exit code 1, see above)
python3 sequence_evaluate.py --all-sequences --rules nov11   # = verification_run.txt
python3 sequence_evaluate.py --scenarios --rules nov11       # = verification_run.txt
```

`evaluate_model_a_video.py` (needs TensorFlow) and
`run_live_tuned_video.py` (live backend calls) now default to the app's
rules at clip time and write `*_app_rules.json` files, so the Nov 11
result files are not overwritten; `--rules nov11` reproduces the Nov 11
run. Neither was re-run for this section.

Files added or changed for this section:
- `video-sequence-layer/state_machine.py` (app rules), `state_machine_nov11.py` (frozen Nov 11 rules)
- `video-sequence-layer/test_state_machine.py` (42 tests)
- `video-sequence-layer/rerun_nov25.py` (this re-run)
- `video-sequence-layer/results/video_evaluation_nov25.json`
- `video-sequence-layer/sequence_evaluate.py`, `evaluate_model_a_video.py`, `run_live_tuned_video.py` (`--rules app|nov11`)
- `gemini-finetune/model-b-scenarios/build_scenarios.py` (pinned to `state_machine_nov11.py` so a rebuild reproduces the stored file)
