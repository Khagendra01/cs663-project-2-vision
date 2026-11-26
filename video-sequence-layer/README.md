# Video Sequence Layer — CS663 Project 2, Dog Training Assistant

Project 2 is a **video** project at the system level and an **image**
project at the model level. This folder is the video half.

- Proposal §3: the app's input is "a continuous live video feed from the
  mobile device's camera"; §2: it judges a command **over time** (hold
  the posture for a target duration = success, break early = failure).
- The classifiers (Model A / the Gemini endpoints) only ever see single
  frames. A still image cannot show a hold, a break, or walking — a
  **sequence** can.
- Grewe's proposal feedback (Oct 8) asked for exactly this layer:
  clarify video vs still-image, define the command interval (explicit
  **Start Sequence** + target duration), and add a **temporal state
  machine** (frame postures → success / early-break / uncertain) with
  **temporal smoothing**.

## How it fits the pipeline

```
live camera -> sample frames -> per-frame posture label
  (Model A on device, or POST frame to Cloud Run /api/gemini |
   /api/fine-tuned-gemini - selected frames, so Gemini stays multimodal)
  -> TemporalSmoother (majority vote, window 5 frames)
  -> CommandSession state machine -> SUCCESS / EARLY_BREAK / UNCERTAIN
  -> feedback to the trainer (Model B can phrase the verdict + evidence)
```

## Files

| File | What it is |
|---|---|
| `state_machine.py` | The judging logic. `CommandSession` + `TemporalSmoother`, pure stdlib. Since Nov 25 the same rules, defaults and reason strings as the Android app's `SequenceSession.java`; `config_for_command()` picks the command shape the way the app does (Sit / Down reach-then-hold, others hold). |
| `state_machine_nov11.py` | The Nov 11 rules, frozen, so the Nov 10-11 results stay exactly reproducible (`--rules nov11`). |
| `test_state_machine.py` | 42 unit tests: the original 17 (one expectation updated for the raw-time reach rule, commented), the app's 21 JUnit tests ported one to one, and 4 config / preset tests. **All 42 pass.** |
| `sequence_evaluate.py` | Harness: `--video` / `--frames` / `--labels` in, verdict JSON out. Two classifier modes: precomputed labels, or live backend calls to the deployed Cloud Run service. `--all-sequences` and `--scenarios` run the built-in checks. `--rules app` (default) or `--rules nov11`; `--smooth-window`, `--min-confidence`, `--break-confirm`, `--lost`, `--shape` override single values. |
| `rerun_nov25.py` | Offline re-judge of the stored Nov 11 predictions and of both scenario sets with the app's rules, at clip time; no network calls. Writes `results/video_evaluation_nov25.json`. |
| `build_sequences.py` | Cuts the real-video sequences below from the 6 verified Commons source videos in `../gemini-finetune/real-videos/`. |
| `video-sequences/` | 6 real clips (mp4, 640w) + frames at 2 fps + per-frame ground-truth labels + `manifest.json` with expected verdicts. |
| `scenarios.json` | 6 **synthetic** label-stream scenarios (success, early break, noisy success, never-settles, walking, dog leaves frame). Synthetic because the real clips are single-behaviour holds — a posture change cannot be honestly manufactured from them. |
| `results/` | Recorded verification run (`verification_run.txt`) and one live-backend result (`live_seq06_tuned.json`). |

## Nov 25 re-run with the app's rules

`results/video_evaluation.md` (last section) and
`results/video_evaluation_nov25.json`: the 6 real sequences re-judged
from the stored predictions at real clip time (t = frame_index / 2).
**No verdict changed**: Model A 5/6, tuned endpoint 6/6, labels mode
6/6. Of the stored scenario verdicts, the app's rules agree with 4/6
(`scenarios.json`) and 119/180 (Model B set); every difference is a Sit
or Down row and comes from the reach-then-hold rule (with every command
judged as a hold: 6/6 and 180/180). The scenario files were left
unchanged.

## Verified results (Nov 10, 2026)

Judged with the Nov 11 rules (now `--rules nov11`).

- Unit tests: **17/17 pass**
- Real sequences, labels mode: **6/6 match expected** — 5 sustained holds
  → SUCCESS; the 3.6s wagging clip judged against an 8s target →
  UNCERTAIN (incomplete: held 3.0s, clip ended, not a break)
- Synthetic scenarios: **6/6 match expected**, incl. EARLY_BREAK on the
  proposal's own stay example
- **Live end-to-end**: seq06's frames sent to the deployed tuned Gemini
  endpoint — all 7 classified "standing" — state machine verdict
  UNCERTAIN/incomplete, matching expected. This is the same per-frame
  call the Android app makes.
- Sequences seq01 (standing), seq03 (walking), seq05 (lying down) were
  additionally spot-checked frame-by-frame on contact sheets when built.

Labels mode validates the *judging pipeline* against human-verified
frame labels. It is **not** a classifier accuracy number — classifier
accuracy is the v3/v4 image evaluation (88.4% tuned on the v3 test set).

## Rules the machine enforces (and why)

- **Start Sequence starts the clock.** Before it, nothing is judged.
- **Settle grace (2s):** Stand / Walk / Stay must reach the posture
  within 2 s. The hold clock starts at the first *raw* target frame of
  the run the smoother confirmed (Nov 11: at the smoothed flip).
- **Sit / Down (since Nov 25, as in the app):** reach the posture within
  a 3 s response window, then hold it 0.6 s. Clearly seen in another
  posture at the deadline -> EARLY_BREAK (failed command); unclear at
  the deadline -> UNCERTAIN.
- **A break must be sustained (0.6s of smoothed non-target labels; 1s
  on Nov 11)** and is timestamped at the first frame after the last raw
  target frame. One bad frame never fails a dog.
- **Unclear is not failure.** Low-confidence (< 0.65) / no-dog frames
  are "uncertain frames": short gaps are tolerated, a 1.5s loss of the
  dog (3s on Nov 11) or >40% unclear frames makes the attempt UNCERTAIN,
  never a fake success or a fake break - unless the dog was seen leaving
  the posture (>= 2 consecutive wrong frames) right before it vanished,
  which is an EARLY_BREAK.
- **Ending early is incomplete,** not success and not a break.
- **"Roll over"-type behaviours are sequences, not postures**: they
  would be judged as a target *pattern* over time (a future target type),
  never as a single-frame label. The current targets are the 4 postures
  the frame classifier knows: sitting, standing, lying down, walking.

## Limitations

- Real-video sequences are single-behaviour stock clips, not the
  group's own dogs and not real training sessions; early-break evidence
  is synthetic label streams + unit tests until the group records real
  sessions (the Dec 1 live demo is the place for those).
- Live backend mode costs one API call per sampled frame (2 fps);
  on-device Model A is the production path, the backend mode is for
  evaluation and the lab app.
- seq01 has a second dog's head visible in the background grass; the
  tracked subject stands throughout.
