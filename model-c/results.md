# Model C / C+ Results — YOLOv8 Dog Detection (CS663 Project 2)

Proposal §4/§7: Model C is a **pretrained** YOLOv8 (COCO) that finds
the dog in each raw frame and crops it before Model A classifies the
posture. No training: stock weights, exactly as proposed. This run
demonstrates that stage for real on the project's own imagery.

## Setup

- ultralytics 8.4.210, torch 2.15.0, CPU inference, `yolov8n.pt`
  (COCO weights, downloaded from the ultralytics GitHub release)
- Dog = COCO class 16; confidence threshold 0.25
- Script: `detect_dogs.py` (this folder) — writes annotated images
  (PIL-drawn box + `dog <conf>`), one crop per detection, and
  `outputs/results.json` with every box and confidence
- Reproduce: `venv/bin/python detect_dogs.py --set frames111=../video-sequence-layer/video-sequences --set v4test60=sample-v4-test60 --out outputs --weights yolov8n.pt`
- Install note, for honesty: the first `pip install ultralytics`
  failed with "No space left on device" (this VM's /tmp is a 512 MB
  tmpfs and the torch wheel alone is 539 MB) and the PyTorch CPU index
  timed out through the proxy; the plain PyPI install then succeeded
  with TMPDIR redirected into the workspace. The venv is ~5.9 GB
  because PyPI torch bundles CUDA libraries; inference here ran on CPU.

## Headline numbers

| Set | Images | With ≥1 dog | Detection rate | Dogs found | Mean conf |
|---|---|---|---|---|---|
| Video-sequence frames | 111 | 65 | **58.6%** | 73 | 0.667 |
| v4 test sample (seed 42) | 60 | 49 | **81.7%** | 53 | 0.759 |

The still photos (v4) detect far better than the video frames — the
frame sets include the hard cases a live camera actually produces
(close-ups, top-down views, curled sleeping dogs), which is the honest
picture and exactly why the pipeline does not trust single frames.

## Video frames, per sequence

| Sequence | Behaviour | Detected | Notes |
|---|---|---|---|
| seq01 Africanis | standing | 12/20 | misses cluster in frames 013–020 (see below) |
| seq02 Batumi | lying down | **3/20** | worst set — see failure notes |
| seq03 Mission | walking | 16/24 | misses when the dog is small/far in frame |
| seq04 Snoring | lying down | 9/20 | frames 001–009 mostly found (some as 2 boxes), 010–020 all missed |
| seq05 Panting | lying down | 18/20 | close-up lying dog, detects well |
| seq06 Wagging | standing | 7/7 | clean front view, conf 0.88–0.92 |

## v4 test sample, per class (60 images, seed 42, from the 235-image test split)

| Class | Detected | Rate |
|---|---|---|
| sitting | 16/17 | 94% |
| standing | 10/11 | 91% |
| walking | 12/16 | 75% |
| lying down | 11/16 | **69%** |

Canonical side/front views (sitting, standing) are reliable. Lying
down is the weak class for the detector too — a dog flat on the ground,
curled, or viewed from above stops looking like COCO's mostly-upright
dogs.

## Where it failed (57 images, no dog found)

All failure filenames are in `outputs/results.json`; the complete
lists:

- **frames111 (46):** seq01 frames 013–020 (8); seq02 frames 001–004,
  008–020 (17); seq03 frames 009–011, 019–021, 023, 024 (8); seq04
  frames 010–020 (11); seq05 frames 018, 019 (2)
- **v4test60 (11):** ph_aleksey_churushkin_lying_down_001,
  ph_capi16_lying_down_001, ph_jatinpandey310_lying_down_001,
  ph_muntaqibah_lying_down_001, ph_tm_lying_down_001,
  ph_fotothekbot_walking_001, ph_ln9267_walking_001,
  ph_underwaterbuffalo_walking_004, vid_mission_walking_005,
  ph_famartin_standing_001, ph_geagea_sitting_001

Failure modes, from visually inspecting the missed frames (not
guessed):

1. **Top-down / rotated views.** seq02 (Batumi) is a portrait phone
   video: the dog lies flat across dark pavement, small and sideways
   in frame — 17 of 20 frames missed.
2. **Curled / rear views.** seq04's second half: a brindle dog curled
   on a bed with its back and rump to the camera, head turned away —
   large in frame, still missed on all 11 frames.
3. **Extreme close-up + heavy coat.** v4 ph_tm_lying_down_001: a very
   fluffy dog asleep, filling the frame — missed.
4. **Edge-cropped subjects.** v4 ph_geagea_sitting_001: a wet puppy
   sitting, cropped by the frame edges — missed (one of only 2
   sitting/standing misses out of 28).
5. **Small/distant dogs.** seq03's misses are the frames where the
   walking dog is furthest from the camera.
6. **Plain flicker.** seq01 frame_007 detects the standing dog at
   0.94 while the near-identical frame_016 misses it completely.
   Frame-to-frame inconsistency on the same dog in the same shot is
   real, and it is the strongest argument in this whole report for the
   temporal layer: the state machine's smoothing and its
   lost-dog/UNCERTAIN rules exist precisely because Model C output
   flickers like this.

## Multi-dog note

12 images produced 2 dog boxes (8 frame images, 4 v4 photos) — e.g.
seq01, where a second dog's head pokes out of the grass. `detect_dogs.py`
crops **every** detection, as a Model A feed would want for analysis,
but the app must pick one (largest/central box, then track it) or the
posture stream will mix two animals. Flagged here as a design
requirement, not solved by this demo.

## Examples (`examples/`)

- `ex1_walking_seq03_frame004.jpg` — walking, conf 0.93
- `ex2_standing_seq01_frame007.jpg` — standing, conf 0.94
- `ex3_lying_v4_yukikoko.jpg` — lying down, conf 0.96 (best of the run)
- `ex4_standing_v4_africanis.jpg` — standing, conf 0.94
- `ex5_twodogs_seq01_frame009.jpg` — two boxes: main dog 0.83, background dog 0.42
- `ex6_FAILURE_batumi_frame010_no_detection.jpg` — a miss, kept on
  purpose: the dog is plainly there (top-left, lying on the pavement)

## What this means for the project

Model C works as the proposal assumed **for well-framed, upright
dogs** (82% on held-out photos, 94% on sitting) and degrades exactly
where the final report should say it does: lying/curled postures,
top-down phone angles, and distant dogs. The mitigations already in
the design are the right ones — temporal smoothing, uncertain-frame
tolerance in the state machine, and (future) picking a larger model
(yolov8s/m) or a lower threshold for the lying-down case. Nothing in
these results was cherry-picked: every frame of both sets was run and
counted.
