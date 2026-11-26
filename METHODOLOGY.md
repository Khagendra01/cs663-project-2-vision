# Methodology — Dog Training Assistant (CS663 Project 2, Group 5)

Every claim in this document comes from the code and results in this
repository. File and line references are to the repository as it stands
on 2026-11-11. Where the implementation differs from the updated
proposal (Oct 26, 2026), the difference is stated in Section 9, not
smoothed over.

**Update, 2026-11-25 (on-device inference).** The live app has moved
from the v1 design (1 fps, every frame a cloud call to tuned Gemini)
to on-device inference: Model C and Model A now run on the phone and
Gemini is off the per-frame path. That app lives in GitHub B
(`cs663-project-2-app`), branch `app-three-model-integration`, which
is pushed but **not yet merged to main**; its `README.md` and `eval/`
hold the new results quoted here. The `android-app/` folder in this
repository is an older copy of the v1 app. Passages below that
describe v1 or the Nov 11 runs are kept and labelled as such. All
on-device latency is from the **Android emulator, not a physical
phone**; real-phone fps and latency come with the demo (issue #2).

## 1. System overview

The system judges a commanded posture **over time**, not from one
photo. The pipeline is:

```
phone camera (CameraX live preview, frames made upright)
  -> analysis loop capped at ~5 fps (180 ms minimum frame interval)
  -> Model C: dog detector, EfficientDet-Lite0 (on the phone)
  -> Model A: posture classifier, MobileNetV3 TFLite, whole frame
     (on the phone; counted only on frames where Model C found a dog)
  -> temporal state machine (on the phone, Java, time-based)
  -> verdict: SUCCESS / EARLY_BREAK / UNCERTAIN
  -> Model B (tuned Gemini, cloud) re-checks 2-3 decisive frames:
     VERIFIED / DISPUTED / PARTLY CHECKED / NOT CHECKED
  -> coaching tip (base Gemini, cloud, after the verdict)
```

This replaces the v1 pipeline used for the Nov 11 evaluation:
Camera2 preview -> one frame per second (1 fps) -> per-frame posture
classification by the tuned Gemini endpoint (cloud) -> the same kind
of on-phone state machine -> verdict -> base Gemini feedback.

The trainer picks a command (Sit, Down, Stand, Stay, Walk) and a target
duration, and presses **START** (gated by READY, Section 7.1; v1:
**Start Sequence**) — that tap is t = 0. During the
attempt the app shows the live smoothed posture and the held-time
progress. When a verdict is decided (or the trainer taps Stop), the
verdict panel shows the verdict, held seconds, break time if any, the
reason, and the full smoothed label sequence, and trainer feedback is
fetched.

Where the decision lives, stated exactly:

- The **verdict logic runs on the phone** (`SequenceSession.java`,
  originally a Java port of `video-sequence-layer/state_machine.py`;
  the Java version has since moved ahead of the Python one, Section 7).
  No cloud call can change a verdict; Model B's check and the coaching
  step are advisory (Section 5).
- The **per-frame labels now also come from the phone**: Model C finds
  the dog and Model A classifies the posture, both on-device, so the
  live loop no longer waits on the network. Measured on the emulator,
  the on-device time per frame (Model C + Model A + crop) had a median
  of 94–149 ms across four full attempts (Section 8, Live latency).
  The state machine is time-based (seconds, any fps), so its rules use
  the proposal's times directly.
- **v1, for the record (Nov 11):** every sampled frame was POSTed to
  the deployed tuned endpoint (`POST /api/fine-tuned-gemini`), measured
  at ~3.5 s per frame in the video evaluation. That is why the v1
  sampler ran at 1 fps and a 10 s target took roughly three times
  longer in wall-clock time than the target suggested. The on-device
  loop above replaces that path. The updated proposal §4 (first four
  stages on the phone, Gemini only after the attempt) is now what the
  app does, with one addition: Model B may also be asked before START
  as a READY tie-break when Model A stays unsure (Section 5; Section 9,
  row 4).

The app is native Android (owner decision, Nov 11: outcome over
methodology, no Flutter rebuild). The single-photo mode (base vs tuned
Gemini on one image) is the Week 11 lab flow; in the current app
("Model lab") it also runs that photo through Model C + Model A on the
phone.

## 2. Data

### What the data is

- **Dataset v3 — 171 images.** The original tuning set for the Gemini
  fine-tune (Model B). Classes: sitting 35, standing 42, walking 37,
  lying down 57. Split by source: train 102 / validation 26 / test 43.
  Built Oct 26–27, 2026 from Wikimedia Commons videos (v1: 37 frames)
  plus Commons photos (v2: 110 examples; v3: 171).
- **Dataset v4 — 1,000 images, exactly 250 per class.** The expanded
  set built Nov 10, 2026 to train Model A and to give evaluations a
  larger held-out test set. Split by source: **train 626 / validation
  139 / test 235**. Recorded in
  `gemini-finetune/real-data-v4/provenance.json` (title, Commons URL,
  licence, artist, uploader for every image); 483 distinct sources.

### Where the images came from

All imagery is **licensed stock from Wikimedia Commons** — openly
licensed photos and video frames, not the group's own dogs. The v4
collection looped Commons categories and search queries
(`collect_v4.py`, plus two extra walking rounds
`collect_v4_walk2.py` / `collect_v4_walk3.py`), deduplicated by title,
and pooled 520 raw candidates per class (2,880 total).

### How they were verified

Every candidate was **viewed on a PIL contact sheet before being
kept** (`make_sheets_v4.py`, 30 images per sheet, filename on each
cell). An image was kept only if it showed one clear dog in the
labelled posture; ambiguous, distant, person-dominant, or
multi-dog images were dropped. Verified keeps: sitting 284, standing
424, lying down 219, walking 235. Assembly (`assemble_v4.py`) took a
diversity-first subset (round-robin across uploaders) to reach exactly
250 per class. The same contact-sheet verification was used for v1–v3.
Rejected candidates are listed in `gemini-finetune/real-dataset.md`
(e.g. the Tokyo pro-dog-walker pack, blurry 1901 Boxer footage).

### The split, and its honest limit

Splits are made **by source** — the video id or photo uploader a group
of images came from — never by random shuffle. No source appears in
two splits (verified from `provenance.json`: 0 source leaks). This is
**not** the same as splitting by verified dog identity: for
open-source imagery the dog's identity cannot be verified, so the test
set contains **unseen sources, not proven unseen dogs**. The updated
proposal §8 asks for a by-dog split with at least two completely unseen
dogs and says open-data images with unverifiable identity should stay
in training only; we do not meet that standard, and the test numbers
in this report must be read as "unseen-source" numbers, not
"unseen-dog" numbers. One known v1 weakness carried into the record:
the walking test frames in v1 were a same-video holdout (last 5 s of
the Mission video also used for walking training).

## 3. Model A — student-trained MobileNetV3

Model A is the model we trained ourselves, exactly as specified in the
proposal (Model A = MobileNetV3, lightweight CNN). The script that
produced every number below is `model-a/train_model_a.py`; it was run
once, on CPU, with no GPU.

| Item | Exact value |
|---|---|
| Architecture | Keras `MobileNetV3Small`, ImageNet weights, `include_top=False`; head: GlobalAveragePooling → Dropout 0.3 → Dense 4 (softmax) |
| Input | 224×224 RGB, raw 0–255 pixels, batch 32 |
| Preprocessing | None added. The Keras backbone already contains a `Rescaling(1/127.5, −1)` first layer and `mobilenet_v3.preprocess_input` is an identity passthrough (both verified directly). Feeding rescaled pixels as well collapses training to chance — this was hit once during development, diagnosed, and the run discarded. |
| Seed | 42 (Python, NumPy, TensorFlow) |
| Loss / optimiser | Sparse categorical cross-entropy / Adam |
| Phase 1 | Backbone frozen, head trained. Adam lr 1e-3, up to 20 epochs, early stopping on val accuracy (patience 4, restore best). **Ran 15 epochs.** |
| Phase 2 | Top 30 backbone layers unfrozen (BatchNorm kept frozen — small data), Adam lr 1e-4, up to 8 epochs, early stopping (patience 3). **Ran 8 epochs.** Total 23 epochs. |
| Augmentation (training only, in-model) | RandomFlip horizontal, RandomRotation 0.08, RandomZoom 0.10, RandomBrightness 0.15 |
| Data | Dataset v4, split exactly as recorded in `provenance.json` (train 626 / val 139 / test 235). Nothing reshuffled. Test set evaluated exactly once, at the end. |

### Results (held-out v4 test set, 235 images)

- **Test accuracy 0.7489 (176/235); validation accuracy 0.7770.**
- Per class (test): lying_down P 0.867 / R 0.839 / **F1 0.853** (support
  62); sitting P 0.643 / R 0.871 / **F1 0.740** (62); walking P 0.812 /
  R 0.629 / **F1 0.709** (62); standing P 0.721 / R 0.633 / **F1 0.674**
  (49). **Macro-F1 0.744** — below the proposal's ≥0.80 target
  (Section 9, row 9).
- Confusion matrix (rows true, cols predicted; order sitting,
  standing, lying_down, walking):
  `[[54, 2, 4, 2], [11, 31, 2, 5], [7, 1, 52, 2], [12, 9, 2, 39]]`.
  The pattern is the finding: standing is called sitting (11) and
  walking (5); walking is called sitting (12) and standing (9). Single
  frames of these three postures genuinely overlap.
- Full per-epoch history: `model-a/history.json`; curves:
  `model-a/curves.png`; confusion heatmap: `model-a/confusion.png`;
  all 235 predictions: `model-a/test_predictions.json`; 10 sample
  predictions (7 correct, 3 misses kept): `model-a/sample_predictions.md`.

### Export

`model_a.keras` (4.2 MB) and `model_a.tflite` (1.95 MB, float16,
`tf.lite.Optimize.DEFAULT`, `supported_types=[float16]`). The TFLite
export was verified with the TF Lite interpreter on a held-out test
image (`model-a/tflite_check.json`): input [1, 224, 224, 3] float32,
output probabilities sum to 1.0, prediction lying_down @ 0.999 agrees
with the Keras model. **In the v1 app (Nov 11) the TFLite model was not
wired in**; the per-frame classifier was the tuned Gemini endpoint,
which measured higher (Section 4). **It is now the app's per-frame
posture classifier** (GitHub B, Nov 25): the same file, byte-identical
(shipped as `model_a_posture.tflite`), run with LiteRT on the whole
frame at 224×224 RGB 0–255, and counted only on frames where Model C
found a dog. Why the whole frame and not the detector crop is measured
in Section 8.

## 4. Model B — fine-tuned Gemini

### The tuned model (as actually deployed)

- **Base model: gemini-3.5-flash.** Tuning job A
  (1708100736908263424, Google Cloud project `cs663-dog-training`,
  region `us-central1`): created Oct 29, 2026, 11:56 PM PT; finished
  Oct 30, 4:53 PM PT (about 17 h). Tuned model resource:
  `projects/917815040923/locations/us/models/83175855617998848@1`.
- **Serving detail (measured, not assumed):** the tuned model does
  **not** answer when called by its model resource name (404 in
  us-central1, global, and us). It serves only through its deployed
  endpoint `projects/917815040923/locations/us/endpoints/514008491845746688`
  with client location `us`. The backend (`cloud-run-backend/index.js`)
  calls it that way.
- **Training data:** dataset v3 (171 images), single-image posture
  classification with the fixed prompt below. This is **not** the
  multimodal attempt format the updated proposal §7 specifies
  (command + timeline + 3–5 attempt crops + JSON contract); the tuning
  taught image-posture classification, and the sequence reasoning in
  the app comes from the state machine plus the runtime feedback step
  (Section 5).
- **Prompt (identical for training, evaluation, and the app):**
  "Look at the image of the dog. What is the dog doing? Answer with
  exactly one of: sitting, standing, lying down, walking."

### Evaluation (v3 held-out test set, 43 images, Oct 30, 2026)

Strict exact-match scoring (answer must be exactly one label;
sentence answers count as invalid):

| Model | Accuracy | Per class |
|---|---|---|
| Base gemini-3.5-flash | 37/43 = **86.0%** (3 invalid, 0 API errors) | sitting 8/9, standing 8/11, walking 8/9, lying down 13/14 |
| **Fine-tuned (Model B)** | 38/43 = **88.4%** (3 invalid, 0 API errors) | sitting 8/9, standing 8/11, walking 8/9, **lying down 14/14** |

The net gain is one image, and it is in lying down (13/14 → 14/14).
Lenient scoring (correct label inside a sentence counted) gives 40/43
for both. The 5 tuned misses: 2 wrong labels (standing called
walking; walking called standing) and 3 verbose answers, 2 of which
contained the correct label. Full per-image results:
`gemini-finetune/results/tuned-results.md` and `base-results.md`.
(For reference only: base `gemini-3-flash-preview`, the model the
backend uses for feedback, scored 40/43 = 93.0% on the same 43 images
— it is a different, newer base model, not the tuning base, so it is
not the comparison the proposal asks for.)

### The 180-scenario corpus

`gemini-finetune/model-b-scenarios/scenarios.jsonl` — 180 synthetic
training scenarios for the feedback side of Model B (proposal §8
promised ~150–200). Generated programmatically
(`build_scenarios.py`, seed 20261110), not hand-written: 12 behaviour
patterns × 5 commands × target durations 3–30 s. Counts:
**SUCCESS 60 / EARLY_BREAK 80 / UNCERTAIN 40**; per command Sit 40,
Down 49, Stand 28, Walk 31, Stay 32.

Every scenario's verdict was **machine-checked**: each generated label
stream (1 step = 1 s) was fed through the real state machine
(`video-sequence-layer/state_machine.py`, smooth_window 3, settle
grace 2 s, break confirm 1 s, uncertain timeout 3 s) and the builder
asserted the machine's verdict equals the pattern's intended verdict;
a post-write validation re-parsed all 180 lines, rebuilt the frames
from the text, re-judged, and re-asserted verdicts, roles, system
prompt, and uniqueness. Two patterns had to be redesigned around the
machine's smoothing (a break starting within ~2 frames of the target
smooths into a success) — the machine's numbers, not the template's,
are the ones quoted in the advice.

Advice follows fixed trainer rules (the same rules the app's runtime
feedback follows): SUCCESS → praise, state the held time, reward now,
next target up (+2/+5/+10 s); EARLY_BREAK → state the machine-measured
held time, do not reward the break, next target = held − 1 s, reduce
distance/distraction; UNCERTAIN → no judgement of the dog, concrete
re-run fix for the specific reason.

**Disclosure (proposal §7):** these scenarios are synthetic label
streams with rule-generated advice, not recordings of real dogs, and
are presented as such. They teach format, verdict discipline, and
advice rules. They are the fine-tuning corpus for Model B; in the v1
app, runtime feedback was base Gemini prompted with the same rules,
and in the current app runtime coaching is base Gemini given the app's
verdict (Section 5). Real recorded sessions should be added and would outrank
synthetic rows.

## 5. Model B at runtime — the JSON feedback contract

### Current app (GitHub B, Nov 25)

Gemini is off the per-frame path. After the on-device verdict, two
cloud steps run; neither can change the verdict, and if the backend
is unreachable the verdict still shows, marked NOT CHECKED.

- **Model B (tuned Gemini, `POST /api/fine-tuned-gemini`) checks the
  verdict.** It re-reads the 2–3 decisive frames (target reached /
  first wrong posture / last frame) as JPEG crops with the dataset
  prompt, and its posture word is compared with what the verdict
  claims about each frame: VERIFIED / DISPUTED / PARTLY CHECKED / NOT
  CHECKED. A disagreement is shown on the card, not hidden.
- **READY tie-break.** When Model A stays unsure for 2 s before a
  start, Model B is asked to confirm the starting posture on a crop.
- **Base Gemini writes the coaching** (`POST /api/gemini`). It is
  given the verdict, held time, break time, the posture timeline and
  Model B's check, is told **not** to re-judge, and must reply with a
  JSON object: `evidence_summary`, `coaching_tip`, `caution`. If the
  reply is not valid JSON or the call fails, a built-in template tip
  is shown instead (`SequenceActivity.java`, `coachingPrompt`,
  `templateTip`).
- **Measured on the emulator (Nov 25):** Model B calls took 2.7–4.8 s
  each; coaching calls 3.1–8.4 s; coaching returned valid JSON every
  time in the four full attempts. Four attempts are too few to call
  the proposal's ≥95% valid-JSON target met (Section 9, row 12).

### v1 (Nov 11): the feedback contract this replaced

The rest of this section describes the v1 sequence mode (GitHub B
commit `bb17868`), where Gemini was deliberately **not** told the
verdict. It is kept as the record of that design.

After a verdict is decided, the app asks the **base** model
(`gemini-3-flash-preview`, location `global`, via `POST /api/gemini`)
for feedback. The implementation is in
`SequenceActivity.java` (`fetchFeedback`, `resolveFeedback`,
`templateFeedback`).

- **Input sent:** command, target duration, and the timestamped
  smoothed posture timeline (one label per second). It deliberately
  does **not** include the engine's verdict and does **not** include
  the break time, so Model B judges the evidence independently instead
  of copying the local decision (updated proposal §7). One frame — the
  last captured frame — rides along only because the deployed backend
  requires `imageBase64` on every call (there is no text-only
  endpoint); the prompt does not ask the model to re-classify it.
- **Reply contract (JSON object, exactly these fields):**
  `status` (success / early_break / uncertain), `break_time_s`
  (number or null), `evidence_summary` (one sentence), `coaching_tip`
  (one short sentence), `caution` (optional, one short sentence).
  The parser tolerates a ```json code fence.
- **Authority rule (proposal §7):** the on-device temporal engine is
  the authority for status and timing. `resolveFeedback` compares the
  model's `status` with the local verdict: if they agree and both a
  summary and a tip are present, the model's summary and tip are shown;
  if they disagree, or the reply cannot be parsed, or the call fails,
  the **local verdict is shown with a deterministic template tip**
  (SUCCESS: praise + longer target next; EARLY_BREAK: held time, do
  not reward, shorter target next — or, for Sit/Down, "did not reach
  the target within the 3s response window"; UNCERTAIN: reposition the
  camera and retry, not a failure). The model's verdict is never
  displayed when it disagrees with the local one.
- **Never measured in v1:** this contract's agreement rate and
  JSON-validity rate were never run — the v1 app never ran on a
  device, so no live v1 feedback call was ever made. The v1 contract
  was replaced before it was measured; the current coaching step's
  JSON results are above.

## 6. Model C+ — YOLOv8 dog detection (offline evaluation only)

Model C+ is **pretrained YOLOv8n (COCO weights `yolov8n.pt`)**, used
off the shelf exactly as the proposal specifies — no custom
fine-tuning. Setup: ultralytics 8.4.210, torch 2.15.0, CPU inference,
dog = COCO class 16, confidence threshold 0.25. Script:
`model-c/detect_dogs.py`; every box and confidence is recorded in
`model-c/outputs/results.json`; annotated images and one crop per
detection are in `model-c/outputs/`.

| Set | Images | Dog found | Rate | Mean conf |
|---|---|---|---|---|
| Held-out v4 photos (seed-42 sample of the 235-image test split) | 60 | 49 | **81.7%** | 0.759 |
| Video-sequence frames (all 111) | 111 | 65 | **58.6%** | 0.667 |

Per class on the 60 photos: sitting 16/17 (94%), standing 10/11
(91%), walking 12/16 (75%), lying down 11/16 (69%). Per video
sequence: seq06 standing 7/7, seq05 lying 18/20, seq03 walking 16/24,
seq01 standing 12/20, seq04 lying 9/20, **seq02 lying (Batumi) 3/20**.

Failure modes (all 57 missed images listed by filename in
`model-c/results.md`): (1) top-down/rotated views — Batumi is a
portrait phone video, dog flat on dark pavement, 17/20 missed;
(2) curled/rear views — seq04 frames 010–020, dog curled with its
back to the camera, all missed despite being large in frame;
(3) extreme close-up + heavy coat (fluffy sleeping dog filling the
frame); (4) edge-cropped subjects; (5) small/distant dogs; (6) plain
flicker — seq01 frame_007 detects at 0.94 while the near-identical
frame_016 misses entirely. 12 images produced two dog boxes (e.g. a
second dog's head in the grass in seq01), so any consumer must pick
one box to track.

**Why Model C+ was not in the v1 live pipeline (Nov 11):** at 58.6%
detection on video frames, gating the pipeline on the detector would
have made the live app worse — a missed detection would delete frames
the posture classifier reads correctly (the tuned endpoint classified
the same Batumi frames correctly in the video evaluation). The v1 app
therefore classified full frames, and the temporal layer handled
missing/unclear frames explicitly. Model C+ was an evaluated component
of the design, not a live gate.

**Model C in the current app (GitHub B, Nov 25) is a different
detector: EfficientDet-Lite0** (COCO, int8) via MediaPipe Tasks, on
the phone, every frame. YOLOv8n above stays the offline evaluation;
EfficientDet-Lite0 is used on the phone because MediaPipe ships it
ready for Android. Both are pretrained COCO detectors, not fine-tuned,
so the 81.7% / 58.6% YOLOv8n numbers do **not** describe the phone's
detector. The app now gates Model A on the detector because Model A
has no "no dog" class: without a gate, an empty room would still get
a posture and could open READY or "hold" a Stay. The gate is usable
because of a widened accept rule — COCO "dog" ≥ 0.25, or
cat/horse/sheep/cow/bear ≥ 0.30, since EfficientDet often labels real
dogs as other four-legged animals — which finds the dog in 96% of the
111 clip frames (Section 8, design choices). The cost: a real cat or
other animal in view could also be taken as "the dog".

## 7. Temporal engine — the state machine

The reference implementation is
`video-sequence-layer/state_machine.py` (pure stdlib; 17/17 unit
tests, `test_state_machine.py`). The app runs a Java port
(`SequenceSession.java`, 21 JUnit tests in GitHub B). **The Python
reference is now behind the Java one.** It still has the Nov 11 rules
below and produced the Nov 11 video-evaluation numbers, which stay
valid for what they measured; the current app's rules are the Java
ones. The two implementations differ in these places, stated here
rather than hidden:

1. The Python reference has **no READY gate and no response-window
   phases** — it uses one settle grace (2 s) for every command. The
   READY gate and the Sit/Down response window exist only in the Java
   app (Section 7.1).
2. The Python reference's lost/unclear rule is a **timeout**
   (`uncertain_timeout_s = 3.0` s of unclear frames during a hold).
   The current Java app uses **1.5 s** with no dog or no confident
   label (the v1 Java app used 2 consecutive no-label frames, the
   1 fps analogue of the proposal's >1.0 s).
3. The Python dataclass default smoothing window is 5; every harness
   run in this project passes `smooth_window=3`. The v1 Java app
   hard-coded 3; the current Java app uses **5**, the proposal's
   value.
4. **Time base.** The current Java machine is time-based: t is
   elapsed seconds since START (clip time in replay), at any fps. The
   Python reference also takes t, but the Nov 11 video evaluation fed
   it t = frame index (1 fps); the labels-mode and scenario checks fed
   t = i/2 (stored 2 fps), Section 8.
5. **Break confirm and confidence.** Java: break confirm 0.6 s and a
   0.65 Model A softmax gate. Python: break confirm 1 s and a 0.6
   threshold that only fires when a confidence is supplied.
6. **Raw-frame timing.** Java takes both the reach time and the break
   time from raw (unsmoothed) frames and needs both the smoothed and
   the raw label to be the target on the frame that decides SUCCESS.
   Python takes the break time from raw frames only.

### Rules (as implemented)

- **Labels:** canonicalised — sit → sitting, down/lie → lying_down,
  stand → standing, walk → walking; anything else (transition,
  undefined, none, no dog, empty, unparseable reply, failed network
  call) → `uncertain_frame`. Uncertain frames pass through the
  smoother and do **not** enter the voting window.
- **Smoothing:** majority vote over the last usable labels — 3 in the
  Python harness runs (and v1 Java), 5 confident frames in the current
  Java app; ties go to the most recent label. One wrong frame can
  never flip the judged posture.
- **Settle grace (Python) / READY + response window (Java):** see
  Section 7.1.
- **Break rule:** a smoothed non-target posture must persist for
  `break_confirm_s` to confirm a break — 1 s in Python (1 frame at
  1 fps), 0.6 s in the current Java app.
  **Break timestamp = the first raw (unsmoothed) observation of the
  wrong posture**, not the confirmation frame (updated proposal §5.3).
  A single wrong raw frame that the smoother votes down is forgotten
  only once a raw frame shows the dog back in the target posture —
  resetting on a smoothed target alone would erase the break's true
  start (this bug was caught by the scen02 check: the break first
  reported 3.0 s instead of the raw 2.5 s).
- **Lost / unclear:** Python — unclear frames for 3.0 s during a hold
  → UNCERTAIN. Current Java — no dog or no confident label for 1.5 s
  → UNCERTAIN, unless the wrong posture was already seen after the
  target was reached, which is EARLY_BREAK at that first wrong frame
  (v1 Java: 2 consecutive no-label frames in any phase → UNCERTAIN).
- **Unclear share:** if more than 40% of frames were unclear when the
  interval ends (`max_uncertain_ratio = 0.4`), the verdict is UNCERTAIN.
- **Incomplete:** if the interval ends (STOP, or the clip ends) with
  the target reached but the duration unmet, the verdict is UNCERTAIN
  ("incomplete, not a success and not a break").
- **Confidence:** the Python code has a `min_confidence = 0.6`
  threshold, but it only fires when a confidence value is supplied.
  Gemini replies are text only, so it never fired for the v1 live
  pipeline, where the proposal's "<0.65 confidence → Undefined" rule
  could not be applied. The Model A video run did pass softmax
  confidences, so the 0.6 threshold applied there. **The current Java
  app applies the proposal's rule:** a Model A softmax below 0.65 is
  an unclear frame (Section 9, row 6).

### 7.1 App-only rules (Java, updated proposal §5.2)

- **READY gate.** The START button shows "START (waiting)" and
  refuses a plain tap until the gate opens (a long press overrides it,
  and the log records that). While the
  sequence screen is open, every analysed frame (Model C + Model A)
  goes through a 5-frame smoother, and the gate opens only when the
  required starting posture has been the smoothed label for **0.6 s**
  (with at least 3 frames where Model C actually found the dog): Sit
  starts from **standing**, Down starts from **sitting**, Stay starts
  from the **selected posture**, Stand starts from standing, Walk has
  no requirement (it is outside the proposal's MVP table). When
  Model A stays unsure for 2 s, Model B is asked to break the tie
  (Section 5; at most once every 6 s). (v1, 1 fps: one frame per second, and the 0.6 s
  rule mapped to 1 confirmed smoothed frame.) Changing the command or
  Stay posture restarts the check. The gate is a UI check, not a
  verdict: the session still judges from t = 0, so a single noisy
  frame opening the gate cannot fake a result.
- **Sit/Down — reach then hold.** After t = 0 the dog must **reach**
  the target within a **3.0 s response window**, then **hold it for
  0.6 s** (v1: 3 frames and 1 frame at 1 fps). The reach time is the
  first raw target frame; the smoother may confirm an in-window reach
  up to 1.0 s after the deadline. If the window expires unreached,
  the verdict is EARLY_BREAK with held 0 s — a failed command, not
  "ran out of time" — unless the dog was unclear or not visible at
  the deadline, which is UNCERTAIN. If the target is reached but left
  during the 0.6 s hold, it is EARLY_BREAK with the held time
  measured. The duration spinner does not apply to Sit/Down.
- **Stay / Stand / Walk — hold.** Reach the target within the 2 s
  settle grace (v1: 2 frames), then hold it for the full target
  duration (default 5 s; options 5 / 10 / 20 s).

### Proposal times in the current app (time-based, ~5 fps cap)

The current Java state machine runs on seconds, so the proposal's
times are used as values, not mapped to frame counts
(`SequenceSession.java`, `SequenceActivity.java` in GitHub B).

| Proposal time | Current app value | Note |
|---|---|---|
| Analyse ~5 frames/s | ~5 fps cap (180 ms minimum frame interval) | On-device inference; real-phone fps not yet measured (emulator live camera: ~1.1–1.6 fps, Section 8) |
| 0.6 s hold (Sit/Down) | 0.6 s | As proposed |
| Starting posture stable ≥0.6 s (READY) | 0.6 s, with ≥3 detected frames | Model B tie-break if Model A is unsure for 2 s |
| Break confirmed after 3 successive smoothed observations (~0.6 s at 5 fps) | 0.6 s | Break timestamp is the first raw wrong frame |
| Lost/occluded >1.0 s → uncertain | 1.5 s | Leaves room for a few low-confidence frames at 5 fps |
| Smoothing: majority of last 5 | Majority of last 5 confident frames | As proposed |
| Settle grace 2 s | 2 s | As proposed |
| Response window 3 s (Sit/Down) | 3 s (+1.0 s smoother confirm grace) | Reach time from raw frames |
| Confidence <0.65 → Undefined | 0.65 on Model A softmax → unclear frame | Possible now that Model A returns a probability |

### v1 mapping table (1 fps, Nov 11 app; superseded)

This is the mapping the v1 app used and the time base of the Nov 11
evaluation. It no longer describes the app.

| Proposal time | App value (1 fps) | Why |
|---|---|---|
| Analyse ~5 frames/s | 1 frame/s | Each frame is a cloud call (~3.5 s measured); 5 fps is impossible with cloud classification |
| 0.6 s hold (Sit/Down) | 1 frame | 1 frame ≈ the 0.6 s confirmation hold |
| Starting posture stable ≥0.6 s (READY) | 1 confirmed smoothed frame | Same mapping as the hold |
| Break confirmed after 3 successive smoothed observations (~0.6 s at 5 fps) | 1 further smoothed non-target frame (break_confirm 1 s) | At 1 fps the 1 s rule is 1 frame; the break timestamp is still the first raw wrong observation |
| Lost/occluded >1.0 s → uncertain | 2 consecutive no-label frames | 2 frames at 1 fps ≈ >1 s of lost/unclear |
| Smoothing: majority of last 5 | Majority of last 3 | At 1 fps a window of 5 would lag half of a 5 s stay |
| Settle grace 2 s | 2 frames | Direct frame mapping |
| Confidence <0.65 → Undefined | Not applied | No confidence score exists (text-only endpoint) |

## 8. Evaluation protocol

### What was run, on what, at which time base

| Check | Input | Time base | Config | Result |
|---|---|---|---|---|
| Unit tests | `test_state_machine.py` | — | — | **17/17 pass** (verified 2026-11-11 by re-running) |
| Labels mode | 6 real-video sequences, human-verified labels | t = i/2 (stored 2 fps) | smooth_window 3, settle 2 s, break 1 s, uncertain timeout 3 s | **6/6 expected verdicts** |
| Synthetic scenarios | 6 scenarios (`scenarios.json`) | t = i/2 (2 fps) | same | **6/6 expected verdicts** |
| Video evaluation, Model A | All 111 stored frames through `model_a.keras` | **t = frame index (1 fps)** | same machine config | Attempt-level **5/6**, frame-level **82/111 (73.9%)** |
| Video evaluation, tuned Gemini | All 111 stored frames through the live tuned endpoint | **t = frame index (1 fps)** | same machine config | Attempt-level **6/6**, frame-level **110/111 (99.1%)** |
| Model A image test | v4 held-out test split | — | — | **74.9% (176/235)** |
| Model B image test | v3 held-out test split | — | — | **Tuned 88.4% vs base 86.0% (43)** |
| Model C+ | 60 v4 photos + 111 video frames | — | conf ≥0.25 | **81.7% / 58.6%** |
| Java unit tests (Nov 25, GitHub B) | `SequenceSessionTest` | seconds | current Java rules | **21 tests** |
| Model C accept rule (Nov 25, GitHub B `eval/model_c_recall.py`) | 111 clip frames, EfficientDet-Lite0 | — | dog/cat only vs widened rule | **61% → 96%** of frames |
| Model A input (Nov 25, GitHub B `eval/model_a_crop_vs_full.py`) | v4 test split (235) + 111 clip frames | — | whole frame vs padded crop | **75.3% vs 68.5%**; clips **73.9% vs 62.2%** |
| Full system on emulator (Nov 25, GitHub B) | 4 project clips + 1 constructed splice, replay mode | clip time, stored 2 fps | all three models, current Java rules | **3 SUCCESS + 1 EARLY_BREAK (splice), all VERIFIED by Model B; 1 Stand attempt never started** (below) |

### The two time bases — read this before quoting any verdict

The stored sequence frames were extracted from the source clips at
**2 fps** (20 frames per 10 s clip). The labels-mode check feeds them
at t = i/2, their real time base. The Nov 11 video evaluation feeds
them at t = frame index — **1 fps, matching the v1 app's sampler**.
(The Nov 25 emulator runs below use replay mode on clip time, the
stored 2 fps, so they are on the real time base, like labels mode.)
Consequences for the Nov 11 video evaluation, stated plainly:

- The machine sees a 10 s clip as a 20 s interval, so an 8 s hold
  needs 8 of 20 frames instead of 16 of 20 — **SUCCESS is easier to
  reach in the video evaluation than in the labels-mode check**.
- A real v1 app attempt of the same clip would only have produced
  ~10 frames, so the video evaluation gives each classifier **twice
  as many frames** as a live v1 attempt would.
- Both classifiers were fed identical frame streams at the identical
  time base, so the **comparison between them stands**; the absolute
  verdicts are v1-app-equivalent, not real-time.

The same time-base dependence shows in the seq06 live proof, which
exists in two versions: `live_seq06_tuned.json` (fps = 2 time base,
**held 3.0 s** of 8 s) and
`live_seq06_stand_wagging_short_tuned.json` (fps = 1 time base, **held
6.0 s** of 8 s). Same 7 frames, same verdict (UNCERTAIN — clip ended),
different held seconds, because the clock is frames. The current
report cites the fps = 1 file (6.0 s), matching the v1 app.

### Attempt-level vs frame-level

Frame-level accuracy is the classifier's per-image score. Attempt-level
accuracy is the verdict the state machine produced for a whole
sequence. They differ on purpose: smoothing can rescue isolated frame
errors (Model A seq03: 2 wrong frames of 24, SUCCESS survived; tuned
seq03 frame_019, smoothed away) and cannot rescue a classifier wrong
on every frame (Model A seq01: all 20 frames called sitting for a
standing dog; seq06: all 7 called walking for a tail-wagging standing
dog). The video evaluation's headline numbers are therefore attempt
level — 5/6 for Model A, 6/6 for tuned — with frame level alongside.

### Live latency

**On-device loop (current app, Nov 25) — emulator only.** Measured on
the Android emulator on an Apple-silicon Mac (Pixel 10 Pro system
image, arm64, Android 37, 16 KB pages), against the live Cloud Run
backend, in debug replay mode feeding the project clips at their
stored 2 fps. Per-frame on-device time (Model C + Model A + crop,
from the app's log) over 4 full attempts: **median 94, 102, 114 and
149 ms; p90 128–335 ms**. On the emulator, that is under the
proposal's 300 ms median target, against ~3.5 s per frame for the v1
cloud path below. It is **not** a phone measurement: the emulator's
live camera reached only ~1.1–1.6 analysed fps (emulator CPU load,
System UI ANRs), which says nothing about a phone. Real-phone fps and
latency come with the demo (issue #2). The cloud steps now run after
the verdict, not per frame: Model B 2.7–4.8 s per call, coaching
3.1–8.4 s.

**Per-frame cloud path (v1, Nov 11 video evaluation).** The tuned
live run made 111 calls in 384.5 s: **~3.5 s per frame**, 0 failed
calls after retries, 0 unparseable replies. At a 1 fps sampling
target the per-frame call took ~3.5 s, so the v1 sampler could not
keep up in wall-clock time and its fixed-delay sampler skipped ticks —
a 10 s target took roughly three times longer in real time than the
target suggested. Verdicts counted frames, not wall clock.

### Two design choices measured offline (Nov 25, GitHub B `eval/`)

Both scripts use the same model files, accept rule and crop math as
the app.

- **Model C accept rule** (`eval/model_c_recall.py`,
  `eval/model_c_recall_results.md`). On the 6 project clips (111
  frames, every one contains a dog), EfficientDet-Lite0 accepting only
  COCO "dog"/"cat" found the dog in **61%** of frames; the widened
  rule (dog ≥ 0.25, or cat/horse/sheep/cow/bear ≥ 0.30) found it in
  **96%**. The largest change is seq01 (africanis): 15% → 95%. The
  app uses the widened rule.
- **Model A on the whole frame vs the Model C crop**
  (`eval/model_a_crop_vs_full.py`, `eval/model_a_crop_vs_full.md`).
  On Model A's held-out v4 test split (235 photos): whole image
  **75.3%** vs padded crop 68.5% (walking 43/62 vs 17/62). On the 111
  clip frames: 73.9% vs 62.2%. Model A was trained on whole photos, so
  the app runs it on the whole frame and uses Model C as the gate and
  for the crop Model B checks. The 75.3% is the same model as the
  stored 74.9% (Section 3): this re-run resizes with PIL bilinear
  rather than the training pipeline, so both are reported — 75.3% in
  this re-run, 74.9% stored.

### Full-system run on the emulator (Nov 25, GitHub B)

All three models, from camera frame (replayed clip) to coaching tip,
one attempt per row, same emulator setup as above:

| Clip | Command | Verdict | Timing | Model B check | On-device ms/frame (median / p90) |
|---|---|---|---|---|---|
| seq02 (lying, holds) | Stay: lying down, 5 s | **SUCCESS** | reached 0.5 s, held 5.0 s | VERIFIED 2/2 | 94 / 166 |
| seq05 (lying, panting) | Stay: lying down, 5 s | **SUCCESS** | reached 0.5 s, held 5.0 s | VERIFIED 2/2 | 114 / 128 |
| seq03 (walking) | Walk, 5 s | **SUCCESS** | held 6.5 s | VERIFIED 2/2 | 102 / 138 |
| splice: seq05 lying → seq01 standing | Stay: lying down, 5 s | **EARLY_BREAK** at 1.5 s | held 1.0 s | VERIFIED 3/3 | 149 / 335 |
| seq01 (standing) | Stand | never started | READY never opened | — | — |

- The splice is a **constructed clip** (frames of two real clips
  joined), not a real break by a real dog.
- The failure row is real: Model A reads the standing africanis dog
  as sitting at 91–99% confidence, so a Stand attempt never gets
  READY. The state machine cannot fix a confident wrong label.

### What is still only proven synthetically

No EARLY_BREAK occurred in either video run, because none of the
licensed real clips contains a genuine break. The break path is proven
by: the 6 synthetic scenarios (scen02: dog leaves sitting after
holding **2.5 s** of a 5 s target — raw break at t = 2.5 s, confirmed
at t = 4.0 s — → EARLY_BREAK), 6 of the 17 unit tests, and the 80
EARLY_BREAK rows of the scenario corpus. No posture change was edited
into a real clip. The Nov 25 emulator run adds one EARLY_BREAK on a
**constructed splice** (seq05 lying frames followed by seq01 standing
frames); it exercises the full app path on real frames, but it is
still not a genuine break by a real dog.

## 9. Alignment with the updated proposal (Oct 26, 2026)

One row per deviation. "Proposal says" is the updated proposal;
"we did" is the implementation in this repository.

| # | Proposal says | We did | Why |
|---|---|---|---|
| 1 | Flutter mobile app (§4, §9) | Native Android app (Java; Camera2 in v1, CameraX now) | **Owner decision, Nov 11:** outcome over methodology — "we just want project done". The lab app was Android; a Flutter rebuild was not feasible here and adds no capability the rubric grades. |
| 2 | Model A: 5 classes — Sit, Stand, Down, Jump, Undefined/Transition (§6) | 4 classes: sitting, standing, lying down, walking | No Jump images exist in the data (no Commons jump collection was built). Undefined is not a class; in the current app a Model A softmax below 0.65 becomes an unclear frame (row 6), which plays that role in the state machine. Walking is the 4th class from the original proposal and the data we have. |
| 3 | Analyse ~5 frames/s (§3, §5.3) | Current app: ~5 fps cap (180 ms minimum frame interval), on-device. v1 (Nov 11): 1 frame/s | v1: each frame was a cloud call (~3.5 s measured), so 5 fps was impossible; the v1 1 fps mapping is kept in Section 7. On-device inference removed that limit. Real-phone fps is not yet measured (emulator live camera ~1.1–1.6 fps; demo, issue #2). |
| 4 | First four stages run on the phone; Gemini called only after the attempt ends (§4) | Current app: detection, posture, smoothing and verdict all on the phone; Gemini only after the verdict, plus a READY tie-break before START when Model A is unsure for 2 s. v1: per-frame classification was a cloud call during the attempt | GitHub B branch `app-three-model-integration`, not yet merged to main. In v1 the tuned model lived in the cloud and Model A's TFLite export was not wired in. |
| 5 | Smoothing: majority of last 5 confident predictions (§5.3) | Current Java app: majority of last 5 confident frames. v1 app and Python harness runs: majority of last 3 | v1 reason: at 1 fps a window of 5 would lag half of a 5 s stay. The time-based ~5 fps app uses the proposal's 5. |
| 6 | Ignore classifications under 0.65 confidence as Undefined (§5.3) | Current app: applied — Model A softmax < 0.65 → unclear frame. v1: not applied | v1: the tuned endpoint returned text only — no confidence score to threshold. The Python code has a 0.6 threshold that never fires for text labels; it applied only in the Model A video run (softmax confidences). |
| 7 | Dataset ~2,000 crops, half group-collected, split by dog identity, ≥2 fully unseen dogs (§8) | 1,000 images, all open-data, split by source | No group-collected dogs exist. Open-data dog identity is unverifiable, so the test set is unseen sources, not proven unseen dogs (Section 2). |
| 8 | Model B: supervised-tuned Gemini 3.5 Flash on multimodal attempt examples — timeline + 3–5 crops + JSON contract (§7) | Model ID matches (gemini-3.5-flash fine-tune, Job A) but tuning data was single-image posture classification on v3; the JSON contract exists only in the runtime feedback step | The tuning exercise predates the updated proposal. The 180-scenario corpus is text-only synthetic, disclosed as such; the proposal says not to present synthetic text as image-based tuning data, and we do not. |
| 9 | Model A macro-F1 ≥ 0.80 (§10) | Macro-F1 **0.744** | Missed. Reported honestly, as §10 itself requires ("report honest results if targets are missed"). Weakest class is standing (F1 0.674). |
| 10 | Temporal attempt-level macro-F1 ≥ 0.85, false-break rate <5% (§10) | Not computed as F1 | Attempt-level verdicts are reported instead (labels mode 6/6; video 5/6 Model A, 6/6 tuned). No genuine break exists in the real clips, so a false-break rate cannot be measured on real video. |
| 11 | Median live loop latency <300 ms at 5 fps (§10) | Current app, **emulator only**: on-device median 94–149 ms per frame over 4 attempts (p90 128–335 ms). v1: ~3.5 s per frame (cloud) | Under the target on the emulator; not measured on a physical phone (demo, issue #2). In v1, cloud classification dominated and the target was unreachable. |
| 12 | Gemini valid JSON ≥ 95% (§10) | Valid JSON on all 4 full emulator attempts (Nov 25) | Too few attempts to call the target met; no physical-phone run yet. (v1: never measured.) |
| 13 | EARLY_BREAK demonstrated on real attempts | Proven synthetically, plus one constructed splice of two real clips on the emulator (Nov 25) | No licensed real clip contains a genuine break; none was faked, and the splice is labelled as constructed. |
| 14 | Tuned model trained on the current dataset | Tuned Gemini is still v3-trained (171 images) while v4 (1,000) exists | v4 was collected after the tuning run. Model A is trained on v4; retraining Gemini on v4 was not done. |

## 10. Reproducibility

Every result in this report can be re-run from the repository.
Working directory is `files/` unless stated.

```bash
# Model A training (CPU; ~7 min) — model-a/
~/tfvenv/bin/python train_model_a.py
# -> history.json, curves.png, confusion.png, metrics.json/.md,
#    sample_predictions.md, model_a.keras, model_a.tflite

# Model C+ detection — model-c/ (venv already installed; ~5.9 GB)
venv/bin/python detect_dogs.py --set frames111=../video-sequence-layer/video-sequences \
    --set v4test60=sample-v4-test60 --out outputs --weights yolov8n.pt
# -> outputs/results.json, annotated images, crops

# Temporal layer unit tests — video-sequence-layer/
python3 -m unittest test_state_machine        # 17 tests
python3 sequence_evaluate.py --all-sequences  # labels mode, 6 sequences
python3 sequence_evaluate.py --scenarios      # synthetic scenarios, 6/6

# End-to-end video evaluation — video-sequence-layer/
~/tfvenv/bin/python evaluate_model_a_video.py  # Model A over 111 frames
python3 run_live_tuned_video.py               # tuned endpoint, ~6-7 min, 111 calls
# -> results/video_evaluation.md, video_predictions_*.json

# Model B scenario corpus — gemini-finetune/model-b-scenarios/
python3 build_scenarios.py                    # rebuilds scenarios.jsonl (seed 20261110)

# v1 Android app (offline Gradle) — android-app/ (older copy, Nov 11)
./rebuild-apk.sh                              # -> app/build/outputs/apk/debug/app-debug.apk

# Current app + offline checks — GitHub B (cs663-project-2-app),
# branch app-three-model-integration (not yet merged to main); run from its root
(cd android-app && ./gradlew assembleDebug testDebugUnitTest) # APK + 21 SequenceSession tests
python -m venv venv && venv/bin/pip install mediapipe ai-edge-litert pillow numpy
venv/bin/python eval/model_c_recall.py \
    android-app/app/src/main/assets/efficientdet_lite0.tflite \
    ../cs663-project-2-vision/video-sequence-layer/video-sequences
venv/bin/python eval/model_a_crop_vs_full.py \
    android-app/app/src/main/assets/efficientdet_lite0.tflite \
    android-app/app/src/main/assets/model_a_posture.tflite \
    ../cs663-project-2-vision

# This report — report/
~/tfvenv/bin/python build_report.py
```

## 11. What was not verified

- The app has never run on a physical phone. The current app (GitHub
  B, not yet merged to main) has run end to end on the Android
  emulator in replay mode on the project clips, with all three models
  and the live backend (Section 8); the emulator's live camera was
  checked only against its virtual scene, with no dog. Real fps, real
  lighting and a real dog are untested until the recorded demo
  (issue #2).
- The coaching JSON validity rests on four emulator attempts
  (Section 5); the v1 feedback contract was never measured.
- No latency, battery, or thermal measurement exists for a physical
  phone (proposal §10 asks for it). Measured so far: on-device
  per-frame time on the emulator (median 94–149 ms), and the v1
  per-frame cloud-call latency (~3.5 s) from the harness.
- The Python reference state machine has not been updated to the
  current Java rules (Section 7).
- Dog identity in the test sets is unverifiable (Section 2); the
  "unseen dog" claim the proposal asks for cannot be made.
