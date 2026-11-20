# Project 2 — Journey and Requirements Audit

**Dog Training Assistant · Group 5 (Arnav Verma, Khagendra Khatri, Kokila Ganesan) · CS663 Fall 2026**

This file is the audit trail for "Project 2 - Results" (due Tue Dec 1, 2026, 11:00 AM PT, 200 pts). It records what was built, when, how each requirement was satisfied, and where it was not. Nothing here is claimed that the repository does not show. Status key: **Done** (built and verified), **Partial** (built with a documented gap), **Open** (not done yet).

Sources: the Canvas "Project 2 - Results" rubric (checked Oct 20, 2026), the course spec (Project2New), and the updated proposal of Oct 26, 2026 (`docs/CS663_Project2_Proposal_Group5_Dog_Training_Assistant.pdf`). The full methodology is in `METHODOLOGY.md`; testing detail is in `TESTING.md`.

## 1. The journey

| When | What happened |
|---|---|
| Oct 8, 2026 | Proposal graded 28/30. Grewe feedback: Model A and C+ YES, Model B PARTIALLY. |
| Oct 20, 2026 | Results plan written from the Canvas rubric (200 pts). Biggest risk identified: live camera + 3 models on Android. |
| Oct 26, 2026 | Updated proposal (12 pp): Flutter app, 5 posture classes, ~5 fps, genuine multimodal Gemini tuning, ~2,000-image dataset half group-collected. |
| Oct 27, 2026 | Cloud Run backend built; dataset v2 built (110 examples). |
| Oct 29–30, 2026 | Gemini tuning job A: base gemini-3.5-flash, created Oct 29 11:56 PM PT, finished Oct 30 4:53 PM PT (~17 h). Base vs tuned evaluated on 43 test images. |
| Oct 31, 2026 | Gemini fine-tuning exercise Attempt 2 submitted (tuned 88.4% vs base 86.0%). |
| Nov 6, 2026 | Week 11 lab due (Android Backend Integration). Submitted late, Nov 10, with the deployed backend. |
| Nov 10, 2026 | Backend deployed to Cloud Run (base + tuned endpoints verified live). Android app built. Both GitHub repos created. Model A trained on v4 (74.9%). Model C evaluated. 180 advice scenarios built. TESTING.md, 9-page report, Canvas draft written. |
| Nov 11, 2026 | App rules updated to the proposal (READY gate, 3-frame response window, break at first wrong frame, JSON feedback). End-to-end video evaluation: Model A 5/6 attempts, tuned Gemini 6/6. METHODOLOGY.md written; report rebuilt to 12 pages. **Owner decision: NO Flutter** — outcome over methodology, Android stays. **Correction recorded: Model B IS a gemini-3.5-flash fine-tune** — the deviation is its tuning data, not the model. |
| Nov 16, 2026 | Demo narration script written (~3 min, plain student voice). |
| Nov 19, 2026 | Both repos made public. Issues #1–#7 opened (epic + tickets with handoff notes). Arnav Verma invited as collaborator (push) to both repos. This audit written. |

## 2. Rubric audit — the 200 points

| Pts | Rubric item | Status | Evidence |
|---|---|---|---|
| 160 | The application itself: all portions implemented and running, all deliverables turned in | **Partial** | App implemented and builds (`DogBackendLab-Sequence.apk`, 38 KB, aapt-verified); backend live at https://cs663-gemini-service-917815040923.us-central1.run.app (both endpoints verified); offline video pipeline runs end to end. Not yet proven: the app has never run on a physical phone. The on-device run is the demo (issue #2). |
| 20 | Testing docs: accuracy/loss curves + 10 unique test cases (GitHub A); app-start/results (GitHub B) | **Done** | `model-a/curves.png`, `confusion.png`, `history.json`, `metrics.json`. `TESTING.md`: 10 unique test cases TC-01–TC-10, all executed, misses included. GitHub B README Section 3: live-verified `/health` + both endpoints returning "lying down". |
| 5 | GitHub B docs Section 1: app execution instructions | **Done** | README Section 1 in `Khagendra01/cs663-project-2-app`. |
| 5 | GitHub B docs Section 2: code description | **Done** | README Section 2 in the app repo. |
| 5 | GitHub B docs Section 4: what is incomplete / errors | **Done** | README Section 4: 7 items stated plainly (Sequence mode not run on a phone, tuned Gemini still v3-trained, YOLO not wired into the app, no real-video EARLY_BREAK, and 3 more). |
| 5 | GitHub B docs Section 5: YouTube live demo of the app discussing accuracy | **Open** | No video recorded yet. Issue #2. |
| — | Spec: mobile app with live camera input | **Partial** | Camera2 live mode implemented (1 fps sampling); not yet run on a device. |
| — | Spec: 3 models (A student-trained/fine-tuned vision, B fine-tuned Gemini multimodal, C pretrained/foundation) | **Partial** | A: done (fine-tuned MobileNetV3 on v4). B: partial (fine-tuned Gemini exists, but single-image v3 tuning, not multimodal attempt tuning — see §4). C: done (pretrained YOLOv8n, COCO, no custom fine-tuning, as specified). |
| — | Spec: 2 GitHub repos | **Done** | `Khagendra01/cs663-project-2-vision` and `Khagendra01/cs663-project-2-app`, both public since Nov 19, 2026. |
| — | Spec: Documentation Report PDF on Canvas | **Done** | `docs/CS663_Project2_Results_Report_Group5.pdf`, 12 pages, 12 images. Submission itself is issue #4. |
| — | Spec: narrated YouTube demo of the actual app via camera (no code/Colab in the video) | **Open** | Issue #2. Script ready. |
| — | Proposal | **Done** | 28/30. |

Bottom line: the work behind roughly 195 of the 200 points is in place and documented. What remains is the demo video (5 pts) and the on-device proof that sits inside the 160-point app item.

## 3. The three models — what each one is

### Model A — student-trained posture classifier (Done)

Fine-tuned MobileNetV3Small on dataset v4. Phase 1: 15 epochs frozen; phase 2: 8 epochs with the top 30 layers unfrozen; 224 px input. Held-out v4 test (235 images): **74.9% (176/235)**, validation 77.7%, macro-F1 **0.744**. Per-class F1: lying down 0.853, sitting 0.740, walking 0.709, standing 0.674. Exported as `model_a.tflite` (float16, load-verified) in `model-a/`, with curves, confusion matrix, history, metrics, and 10 sample predictions (7 correct, 3 misses kept). Training is reproducible: `model-a/train_model_a.py` (~7 min on CPU).

### Model B — fine-tuned Gemini (Partial)

A real gemini-3.5-flash supervised fine-tune (tuning job 1708100736908263424, base gemini-3.5-flash, us-central1, ~17 h). Held-out v3 test (43 images, Oct 30, 2026): tuned **88.4% (38/43)** vs base gemini-3.5-flash **86.0% (37/43)**. It serves only through its deployed endpoint with client location `us`: `projects/917815040923/locations/us/endpoints/514008491845746688` (the model resource name returns 404). Full record: `gemini-finetune/results/tuned-results.md`.

The gap is the tuning data, not the model: the tune used single-image posture classification on v3 (171 images). The proposal (§7) asks for multimodal attempt examples. That gap is audited in §4, row 8, and tracked in issue #6. The tuned model is also still v3-trained while v4 exists (issue #5).

### Model C+ — YOLOv8 dog detection (Done)

Pretrained YOLOv8n on COCO, no custom fine-tuning, exactly as the proposal specifies. Measured, not assumed: **81.7% (49/60)** on held-out v4 test photos, **58.6% (65/111)** on video frames. Weak on lying/curled/top-down views (Batumi 3/20); all 57 failure filenames are listed in `model-c/results.md`. Used offline for evaluation; not wired into the phone app (stated in GitHub B Section 4).

## 4. Proposal audit — the 14 deviations

Condensed from `METHODOLOGY.md` §9. "Proposal says" is the updated Oct 26 proposal; "we did" is what is in these repositories.

| # | Proposal says | We did | Why |
|---|---|---|---|
| 1 | Flutter mobile app | Native Android app (Java, Camera2) | Owner decision, Nov 11: outcome over methodology. The lab app was Android; a Flutter rebuild adds no capability the rubric grades. |
| 2 | Model A: 5 classes — Sit, Stand, Down, Jump, Undefined/Transition | 4 classes: sitting, standing, lying down, walking | No Jump images exist in the data; Undefined needs a confidence signal the endpoint does not provide. |
| 3 | Analyse ~5 frames/s | 1 frame/s | Each frame is a cloud call (~3.5 s measured). 5 fps is impossible with cloud classification. |
| 4 | First four stages run on the phone; Gemini called only after the attempt ends | Per-frame classification is a cloud call during the attempt; only the verdict logic runs on the phone | The tuned model lives in the cloud. Model A has a TFLite export for on-device use, but it is not wired into the app and it measured lower (74.9% vs 88.4%). |
| 5 | Smoothing: majority of last 5 confident predictions | Majority of last 3 | At 1 fps a window of 5 would lag half of a 5 s stay. |
| 6 | Ignore classifications under 0.65 confidence as Undefined | Not applied | The tuned endpoint returns text only, no confidence score to threshold. |
| 7 | Dataset ~2,000 crops, half group-collected, split by dog identity, ≥2 fully unseen dogs | 1,000 images, all open-data, split by source | No group-collected dogs exist. Open-data dog identity is unverifiable, so the test set is unseen sources, not proven unseen dogs. |
| 8 | Model B: supervised-tuned Gemini 3.5 Flash on multimodal attempt examples — timeline + 3–5 crops + JSON contract | Model ID matches (gemini-3.5-flash fine-tune, Job A) but tuning data was single-image posture classification on v3; the JSON contract exists only in the runtime feedback step | The tuning exercise predates the updated proposal. The 180-scenario corpus is text-only synthetic, disclosed as such; the proposal says not to present synthetic text as image-based tuning data, and we do not. |
| 9 | Model A macro-F1 ≥ 0.80 | Macro-F1 **0.744** | Missed. Reported honestly, as the proposal itself requires. Weakest class is standing (F1 0.674). |
| 10 | Temporal attempt-level macro-F1 ≥ 0.85, false-break rate <5% | Not computed as F1 | Attempt-level verdicts are reported instead (tuned 6/6, Model A 5/6). No genuine break exists in the real clips, so a false-break rate cannot be measured on real video. |
| 11 | Median live loop latency <300 ms at 5 fps | ~3.5 s per frame measured | Cloud classification dominates; the target is unreachable with the current architecture. Verdicts count frames, not wall clock. |
| 12 | Gemini valid JSON ≥ 95% | Untested | The JSON feedback contract is implemented in the app, but the app has never run on a device, so no live feedback call has ever been made. |
| 13 | EARLY_BREAK demonstrated on real attempts | Proven only synthetically | No licensed real clip contains a genuine break; none was faked. |
| 14 | Tuned model trained on the current dataset | Tuned Gemini is still v3-trained (171 images) while v4 (1,000) exists | v4 was collected after the tuning run. Model A is trained on v4; retraining Gemini on v4 was not done (issue #5). |

What the proposal asked for that WAS delivered as specified: the MVP command set (Sit from Stand, Down from Sit, Stay from Sit/Down, 3.0 s response window, 0.6 s hold, 5.0 s default Stay), the READY gate (Start disabled until the required starting posture is stable), the rule-based temporal state machine (17/17 unit tests), the JSON feedback contract (status, break_time_s, evidence_summary, coaching_tip, caution — local verdict wins on disagreement), and a dataset split with zero leakage (split by source, 626/139/235).

## 5. Video input and output — and what can be inferenced later

The system is video in, verdict out, and that path already runs two ways:

1. **Offline (Python, proven):** `video-sequence-layer/` — video file → frames → per-frame classification → state machine → verdict + JSON feedback. Run on 6 real clips / 111 frames on Nov 11, 2026 (tuned Gemini 6/6, Model A 5/6, ~3.5 s/frame). Entry points: `sequence_evaluate.py`, `evaluate_model_a_video.py`, `run_live_tuned_video.py`.
2. **Live (phone app, built, not yet run on device):** Camera2 samples at 1 fps, frames go to the cloud tuned endpoint, the state machine runs locally in Java (`SequenceSession.java`).

For later inference: any new video can be fed through the offline path with no new training, and `model_a.tflite` exists load-verified for on-device use. The one thing not wired in yet is on-device inference inside the app — the app still classifies through the cloud, and YOLO is not wired in either. Wiring TFLite (and YOLO) into the app is the work that unlocks 5 fps sampling and sub-300 ms latency; it is tracked in issue #7.

## 6. Decisions that shaped the result

- **NO Flutter (Nov 11, owner decision):** the updated proposal says Flutter; the working stack is Android. Outcome over methodology — the rubric grades the app and its results, not the framework.
- **Model B correction (Nov 11):** Model B IS a gemini-3.5-flash fine-tune. The real deviation is its tuning data (single-image classification on v3), not the model ID. Fixed in METHODOLOGY.md and the report.
- **Cloud-first sequencing:** the live path classifies in the cloud because the tuned model lives there and measured higher (88.4% vs 74.9%). On-device inference is future work, not a pre-submission rebuild.
- **Honest limits, stated not hidden:** the report, METHODOLOGY.md, and GitHub B Section 4 all carry the same limits (no on-device run yet, v3-trained Gemini, 4 classes, 1 fps, open-data dogs). No synthetic result is presented as a real one.

## 7. What is left

Tracked in GitHub issues on `Khagendra01/cs663-project-2-vision` (epic #1, milestone "Project 2 Results — Dec 1"):

- #2 — Record narrated YouTube demo on a real dog (P0, Khagendra)
- #3 — Approve Canvas submission draft (P0, Khagendra)
- #4 — Submit "Project 2 - Results" on Canvas (P0, Khagendra, due Dec 1, 11:00 AM PT)
- #5 — Decide: retrain Model B (tuned Gemini) on v4 (P1, decision)
- #6 — Model B is not the multimodal fine-tune proposal Section 7 describes (P1, decision)
- #7 — Post-Dec 1 improvements backlog (P2, future)

## 8. Key links

- Repos: https://github.com/Khagendra01/cs663-project-2-vision · https://github.com/Khagendra01/cs663-project-2-app
- Issues: https://github.com/Khagendra01/cs663-project-2-vision/issues (epic #1)
- Backend (live): https://cs663-gemini-service-917815040923.us-central1.run.app
- Report: `docs/CS663_Project2_Results_Report_Group5.pdf` (12 pp)
- Methodology: `METHODOLOGY.md` · Testing: `TESTING.md` · Proposal: `docs/CS663_Project2_Proposal_Group5_Dog_Training_Assistant.pdf`
