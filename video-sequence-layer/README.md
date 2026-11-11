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
live camera -> sample frames (2 fps) -> per-frame posture label
  (Model A on device, or POST frame to Cloud Run /api/gemini |
   /api/fine-tuned-gemini - selected frames, so Gemini stays multimodal)
  -> TemporalSmoother (majority vote, window 3-5 frames)
  -> CommandSession state machine -> SUCCESS / EARLY_BREAK / UNCERTAIN
  -> feedback to the trainer (Model B can phrase the verdict + evidence)
```

## Files

| File | What it is |
|---|---|
| `state_machine.py` | The judging logic. `CommandSession` + `TemporalSmoother`, pure stdlib. Tunables in `SessionConfig`: target duration, settle grace (2s to first reach the posture), break confirmation (1s sustained), uncertain timeout (3s), min confidence, max uncertain ratio. |
| `test_state_machine.py` | 17 unit tests: clean success, the proposal §6 early-break example, single-frame noise, low confidence, no-dog, lost-dog, incomplete interval, settle grace, walking/down holds. **All 17 pass.** |
| `sequence_evaluate.py` | Harness: `--video` / `--frames` / `--labels` in, verdict JSON out. Two classifier modes: precomputed labels, or live backend calls to the deployed Cloud Run service. `--all-sequences` and `--scenarios` run the built-in checks. |
| `build_sequences.py` | Cuts the real-video sequences below from the 6 verified Commons source videos in `../gemini-finetune/real-videos/`. |
| `video-sequences/` | 6 real clips (mp4, 640w) + frames at 2 fps + per-frame ground-truth labels + `manifest.json` with expected verdicts. |
| `scenarios.json` | 6 **synthetic** label-stream scenarios (success, early break, noisy success, never-settles, walking, dog leaves frame). Synthetic because the real clips are single-behaviour holds — a posture change cannot be honestly manufactured from them. |
| `results/` | Recorded verification run (`verification_run.txt`) and one live-backend result (`live_seq06_tuned.json`). |

## Verified results (Nov 10, 2026)

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
- **Settle grace (2s):** the dog gets a moment to get into the posture;
  the hold clock starts at the first *smoothed* target frame (smoothing
  adds ~1 frame of lag by design).
- **A break must be sustained (1s of smoothed non-target labels).** One
  bad frame never fails a dog.
- **Unclear is not failure.** Low-confidence / no-dog frames are
  "uncertain frames": short gaps are tolerated, a 3s loss of the dog or
  >40% unclear frames makes the attempt UNCERTAIN, never a fake success
  or a fake break.
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
