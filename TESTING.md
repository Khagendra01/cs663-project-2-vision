# Testing Documentation — Dog Training Assistant (CS663 Project 2, Group 5)

Every number below was produced by running the code in this repository or the
deployed backend, on held-out data the models were not trained on. Splits are
by source (dog/session), never by random shuffle: a dog in the test set never
appears in training.

## Results summary

| Component | Test | Result |
|---|---|---|
| Model A (MobileNetV3Small, student fine-tuned, dataset v4) | 235 held-out images | **74.9% accuracy** (176/235); val 77.7% |
| Model B (Gemini, fine-tuned, dataset v3) | 43 held-out images | **tuned 88.4%** vs base 86.0% |
| Model C+ (YOLOv8n, COCO pretrained) | 60 held-out v4 photos | dog detected in **81.7%** (49/60) |
| Model C+ (YOLOv8n) | 111 video-sequence frames | dog detected in **58.6%** (65/111) |
| Temporal layer (state machine) | 17 unit tests | **17/17 pass** |
| Temporal layer | 6 real-video sequences (labels mode) | **6/6 expected verdicts** |
| Temporal layer | 6 synthetic scenarios | **6/6 expected verdicts** |
| Temporal layer, live | seq06 frames via deployed tuned endpoint | verdict **UNCERTAIN**, matches expected |

## Model A — curves and metrics

Training/validation accuracy and loss curves (two training phases marked):
![Model A curves](model-a/curves.png)

Confusion matrix on the 235-image test set:
![Model A confusion](model-a/confusion.png)

Per-class (test): lying_down F1 0.853 (P 0.867 / R 0.839) · sitting F1 0.740 ·
walking F1 0.709 · standing F1 0.674. Full metrics: `model-a/metrics.md`,
raw history: `model-a/history.json`, all 235 predictions:
`model-a/test_predictions.json`. The deployable export is `model-a/model_a.tflite`
(float16, verified to load and agree with the Keras model).

## Model B — fine-tuned Gemini

On the v3 test set (43 images, same prompt and splits for both models):
base Gemini 37/43 = **86.0%**, fine-tuned Gemini 38/43 = **88.4%**; the gain is
in lying down (tuned 14/14). Full results in `gemini-finetune/` (results and
per-image outputs from the original exercise). The tuned endpoint is live in
the deployed backend and is what the app calls per frame.
Dataset v4 (1,000 images) is collected and verified for the next tuning round
(`gemini-finetune/real-data-v4/`); the deployed tuned model is still the v3 one.
The Model B sequence-advice corpus (180 synthetic scenarios, every verdict
machine-checked against the state machine) is in
`gemini-finetune/model-b-scenarios/`.

## Model C+ — YOLOv8 detection

Full report with every failure filename: `model-c/results.md`.
Detection is reliable on upright, well-framed dogs (sitting 94% on held-out
photos) and weak on lying/curled/top-down views — the worst case is the Batumi
lying sequence at 3/20 frames (dark pavement, portrait orientation). 12 images
contain a second dog with its own box, so a consumer must pick one box to
track. This is why the live pipeline judges posture from full frames with the
Gemini models and why the temporal layer treats missing/unclear frames
explicitly instead of guessing.

## Temporal layer

- Unit tests: `video-sequence-layer/test_state_machine.py` — 17 tests covering
  settle grace, smoothing, the sustained-break rule, lost-dog and low-quality
  UNCERTAIN paths, progress reporting. All pass.
- Real sequences: 6 clips cut from real dog videos with human-verified labels
  (5 sustained holds → SUCCESS, 1 short clip → UNCERTAIN incomplete). All 6
  judged as expected. Raw run: `video-sequence-layer/results/verification_run.txt`.
- Live proof: seq06's 7 frames sent to the deployed tuned endpoint; all
  classified standing; the state machine returned UNCERTAIN (held 3.0s of the
  8s target, clip ended) — `video-sequence-layer/results/live_seq06_tuned.json`.

## 10 unique test cases

| # | Case | Setup | Expected | Actual | Evidence |
|---|---|---|---|---|---|
| TC-01 | Model A, sitting, clean photo | Held-out test image, unseen source | sitting | sitting, conf 0.999 | `model-a/sample_predictions.md` #1 |
| TC-02 | Model A, lying down | Held-out test image | lying_down | lying_down, conf 1.000 | `model-a/sample_predictions.md` #6 |
| TC-03 | Model A, walking mid-stride | Held-out test image | walking | walking, conf 1.000 | `model-a/sample_predictions.md` #8 |
| TC-04 | Model A, hard case — sitting called walking | Held-out test image where posture is ambiguous in one frame | sitting | **walking, conf 0.991 (miss)** | `model-a/sample_predictions.md` #3 |
| TC-05 | Model A, hard case — standing called lying down | Held-out test image | standing | **lying_down, conf 0.968 (miss)** | `model-a/sample_predictions.md` #5 |
| TC-06 | Fine-tuned vs base Gemini, same test set | v3 43-image test, identical prompt | tuned ≥ base | tuned 88.4% vs base 86.0% | `gemini-finetune/` results |
| TC-07 | YOLO on upright held-out photos | 60-image v4 test sample | dog found in most | 49/60 = 81.7% (sitting 94%) | `model-c/results.md` |
| TC-08 | YOLO failure mode — lying dog, top-down, dark ground | Batumi sequence, 20 frames | dog found | **3/20** — documented failure mode | `model-c/results.md`, `model-c/examples/ex6_*` |
| TC-09 | Full sequence judged SUCCESS | Real clip, dog lying 10s, target lying down 8s | SUCCESS | SUCCESS — held the full 8s | `video-sequence-layer/results/verification_run.txt` (seq02) |
| TC-10 | Live end-to-end: real frames through deployed tuned model | seq06 wagging clip (3.6s) vs 8s target, live backend mode | UNCERTAIN (too short to succeed, not a break) | UNCERTAIN — held 3.0s of 8s, clip ended | `video-sequence-layer/results/live_seq06_tuned.json` |

The EARLY_BREAK path is covered by the synthetic scenario suite (scen02:
dog leaves sitting after 3.0s of a 5s target → EARLY_BREAK with the held time
reported; 6/6 scenarios pass) and by 6 of the 17 unit tests. It is synthetic
because none of the licensed real clips contain a genuine break; no posture
change was edited into a real clip.

## Known limitations (also in GitHub B, Section 4)

- All imagery is licensed stock, not the group's own dogs; breed and viewpoint
  coverage is uneven (standing is show-photo heavy; two gait sequences are
  Muybridge-style).
- Single-frame posture is ambiguous between standing/sitting/walking — the
  measured confusion pattern — which is the reason the system judges sequences,
  not photos.
- YOLOv8n under-detects lying and curled dogs; it is an assistive component,
  not a gate, in the live pipeline.
- The app's Sequence mode is verified by build plus the Python reference
  tests; its first live on-device run is the recorded demo (GitHub B, Section 5).
