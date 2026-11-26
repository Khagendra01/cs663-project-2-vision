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
| Nov 25, 2026 | On-device inference built in GitHub B, branch `app-three-model-integration` (pushed, not yet merged to main). Models C (EfficientDet-Lite0) and A (`model_a.tflite`) now run on the phone every frame, ~5 fps cap; Gemini is off the per-frame path. Full system run on the Android emulator (not a physical phone) against the live backend: on-device median 94–149 ms/frame. |

## 2. Rubric audit — the 200 points

| Pts | Rubric item | Status | Evidence |
|---|---|---|---|
| 160 | The application itself: all portions implemented and running, all deliverables turned in | **Partial** | App implemented and builds (v1: `DogBackendLab-Sequence.apk`, 38 KB, aapt-verified; current: `app-debug.apk` from GitHub B branch `app-three-model-integration`, not yet merged to main, with Models C and A on the phone and 21 state-machine unit tests); backend live at https://cs663-gemini-service-917815040923.us-central1.run.app (both endpoints verified); offline video pipeline runs end to end. Nov 25: all three models run end to end on the Android emulator in replay mode (4 of 5 attempts reached a verdict; the Stand attempt never opened READY because Model A reads the standing dog as sitting). Not yet proven: the app has never run on a physical phone. The physical-phone run is the demo (issue #2). |
| 20 | Testing docs: accuracy/loss curves + 10 unique test cases (GitHub A); app-start/results (GitHub B) | **Done** | `model-a/curves.png`, `confusion.png`, `history.json`, `metrics.json`. `TESTING.md`: 16 unique test cases TC-01–TC-16 (TC-11–TC-15 on the Android emulator, Nov 25), all executed, misses included. GitHub B README Section 3: live-verified `/health` + both endpoints returning "lying down". |
| 5 | GitHub B docs Section 1: app execution instructions | **Done** | README Section 1 in `Khagendra01/cs663-project-2-app`. |
| 5 | GitHub B docs Section 2: code description | **Done** | README Section 2 in the app repo. |
| 5 | GitHub B docs Section 4: what is incomplete / errors | **Done** | README Section 4: 7 items stated plainly. On main (v1): Sequence mode not run on a phone, tuned Gemini still v3-trained, YOLO not wired into the app, no real-video EARLY_BREAK, and 3 more. On branch `app-three-model-integration` (not yet merged), rewritten for the on-device app: not yet run on a physical phone (emulator only), Model A misreads some standing dogs as sitting, Model C is a pretrained COCO detector with a widened accept rule, tuned Gemini still v3-trained, cloud parts need internet, debug build only, dataset bias. |
| 5 | GitHub B docs Section 5: YouTube live demo of the app discussing accuracy | **Open** | No video recorded yet. Issue #2. |
| — | Spec: mobile app with live camera input | **Partial** | Live camera mode implemented. v1 was Camera2 at 1 fps with each frame a cloud call; the Nov 25 build (branch `app-three-model-integration`, not yet merged) is CameraX with Models C and A on the phone, capped at ~5 fps. Not yet run on a physical phone: on the emulator the live camera reached only ~1.1–1.6 analysed fps (emulator CPU load), which says nothing about a phone. |
| — | Spec: 3 models (A student-trained/fine-tuned vision, B fine-tuned Gemini multimodal, C pretrained/foundation) | **Partial** | A: done (fine-tuned MobileNetV3 on v4). B: partial (fine-tuned Gemini exists, but single-image v3 tuning, not multimodal attempt tuning — see §4). C: done (pretrained YOLOv8n, COCO, no custom fine-tuning, as specified, evaluated offline; the app runs a different pretrained COCO detector, EfficientDet-Lite0, on the phone). |
| — | Spec: 2 GitHub repos | **Done** | `Khagendra01/cs663-project-2-vision` and `Khagendra01/cs663-project-2-app`, both public since Nov 19, 2026. |
| — | Spec: Documentation Report PDF on Canvas | **Done** | `docs/CS663_Project2_Results_Report_Group5.pdf`, 12 pages, 12 images. Submission itself is issue #4. |
| — | Spec: narrated YouTube demo of the actual app via camera (no code/Colab in the video) | **Open** | Issue #2. Script ready. |
| — | Proposal | **Done** | 28/30. |

Bottom line: the work behind roughly 195 of the 200 points is in place and documented. What remains is the demo video (5 pts) and the physical-phone proof that sits inside the 160-point app item.

## 3. The three models — what each one is

### Model A — student-trained posture classifier (Done)

Fine-tuned MobileNetV3Small on dataset v4. Phase 1: 15 epochs frozen; phase 2: 8 epochs with the top 30 layers unfrozen; 224 px input. Held-out v4 test (235 images): **74.9% (176/235)**, validation 77.7%, macro-F1 **0.744**. Per-class F1: lying down 0.853, sitting 0.740, walking 0.709, standing 0.674. Exported as `model_a.tflite` (float16, load-verified) in `model-a/`, with curves, confusion matrix, history, metrics, and 10 sample predictions (7 correct, 3 misses kept). Training is reproducible: `model-a/train_model_a.py` (~7 min on CPU).

### Model B — fine-tuned Gemini (Partial)

A real gemini-3.5-flash supervised fine-tune (tuning job 1708100736908263424, base gemini-3.5-flash, us-central1, ~17 h). Held-out v3 test (43 images, Oct 30, 2026): tuned **88.4% (38/43)** vs base gemini-3.5-flash **86.0% (37/43)**. It serves only through its deployed endpoint with client location `us`: `projects/917815040923/locations/us/endpoints/514008491845746688` (the model resource name returns 404). Full record: `gemini-finetune/results/tuned-results.md`.

The gap is the tuning data, not the model: the tune used single-image posture classification on v3 (171 images). The proposal (§7) asks for multimodal attempt examples. That gap is audited in §4, row 8, and tracked in issue #6. The tuned model is also still v3-trained while v4 exists (issue #5).

### Model C+ — YOLOv8 dog detection (Done)

Pretrained YOLOv8n on COCO, no custom fine-tuning, exactly as the proposal specifies. Measured, not assumed: **81.7% (49/60)** on held-out v4 test photos, **58.6% (65/111)** on video frames. Weak on lying/curled/top-down views (Batumi 3/20); all 57 failure filenames are listed in `model-c/results.md`. YOLOv8n is used offline for evaluation and is not in the phone app. Since Nov 25 the app (GitHub B branch `app-three-model-integration`, not yet merged) runs a different pretrained COCO detector on the phone: EfficientDet-Lite0 (int8, MediaPipe Tasks, which ships it ready for Android), also not fine-tuned. It accepts COCO "dog" ≥ 0.25 or cat/horse/sheep/cow/bear ≥ 0.30, because EfficientDet often labels real dogs as other four-legged animals; on the 6 project clips (111 frames, all with a dog) that rule finds the dog in 96% of frames vs 61% for dog/cat only (GitHub B `eval/model_c_recall_results.md`).

## 4. Proposal audit — the 14 deviations

Condensed from `METHODOLOGY.md` §9. "Proposal says" is the updated Oct 26 proposal; "we did" is what is in these repositories.

| # | Proposal says | We did | Why |
|---|---|---|---|
| 1 | Flutter mobile app | Native Android app (Java; Camera2 in v1, CameraX in the Nov 25 build) | Owner decision, Nov 11: outcome over methodology. The lab app was Android; a Flutter rebuild adds no capability the rubric grades. |
| 2 | Model A: 5 classes — Sit, Stand, Down, Jump, Undefined/Transition | 4 classes: sitting, standing, lying down, walking | No Jump images exist in the data. Undefined is not a class; in the Nov 25 build a Model A softmax below 0.65 becomes an unclear frame (row 6), which plays that role. (v1: the text-only endpoint gave no confidence.) |
| 3 | Analyse ~5 frames/s | Nov 25 build: ~5 fps cap (180 ms minimum frame interval), on-device. v1 and the Nov 11 run: 1 frame/s | v1 made each frame a cloud call (~3.5 s measured), so 5 fps was impossible. On-device inference removed that bound. The cap is not yet confirmed on a physical phone; the emulator's live camera reached only ~1.1–1.6 analysed fps (issue #2). |
| 4 | First four stages run on the phone; Gemini called only after the attempt ends | Nov 25 build: Model C (EfficientDet-Lite0) and Model A (`model_a.tflite`, whole frame, counted only when C finds a dog) run on the phone every frame, plus the state machine. Model B re-checks 2–3 decisive frames after the verdict and breaks READY ties when Model A is unsure for 2 s; base Gemini writes the coaching. v1: per-frame classification was a cloud call | Matches the proposal except the READY tie-break, which calls Model B before the attempt. Model A judges the whole frame because it measured higher than on Model C's crop (75.3% vs 68.5% on its held-out test split in this re-run; 74.9% stored). Tuned Gemini is too slow per call (2.7–4.8 s) to judge timing frame by frame, so it checks the verdict instead of producing it (image tests: Model A 74.9% on v4/235, tuned Gemini 88.4% on v3/43, different test sets). Branch not yet merged to main. |
| 5 | Smoothing: majority of last 5 confident predictions | Nov 25 build: majority of last 5 confident frames, time-based rules (seconds, any fps). v1 and the Nov 11 run: majority of last 3 | At 1 fps a window of 5 would lag half of a 5 s stay, so v1 used 3. At ~5 fps the proposal's 5 fits. The Python reference (`video-sequence-layer/state_machine.py`) still has the Nov 11 rules (smoothing 3 as every harness run passes it, break confirm 1 s, 3 s timeout) and is now behind the Java `SequenceSession`. |
| 6 | Ignore classifications under 0.65 confidence as Undefined | Nov 25 build: applied to Model A's softmax on the phone (frames below 0.65 are treated as unclear). v1: not applied | The tuned endpoint returns text only, no confidence score to threshold; on-device Model A gives one. |
| 7 | Dataset ~2,000 crops, half group-collected, split by dog identity, ≥2 fully unseen dogs | 1,000 images, all open-data, split by source | No group-collected dogs exist. Open-data dog identity is unverifiable, so the test set is unseen sources, not proven unseen dogs. |
| 8 | Model B: supervised-tuned Gemini 3.5 Flash on multimodal attempt examples — timeline + 3–5 crops + JSON contract | Model ID matches (gemini-3.5-flash fine-tune, Job A) but tuning data was single-image posture classification on v3; the JSON contract exists only in the runtime feedback step | The tuning exercise predates the updated proposal. The 180-scenario corpus is text-only synthetic, disclosed as such; the proposal says not to present synthetic text as image-based tuning data, and we do not. |
| 9 | Model A macro-F1 ≥ 0.80 | Macro-F1 **0.744** | Missed. Reported honestly, as the proposal itself requires. Weakest class is standing (F1 0.674). |
| 10 | Temporal attempt-level macro-F1 ≥ 0.85, false-break rate <5% | Not computed as F1 | Attempt-level verdicts are reported instead (tuned 6/6, Model A 5/6). No genuine break exists in the real clips, so a false-break rate cannot be measured on real video. |
| 11 | Median live loop latency <300 ms at 5 fps | Nov 25 build, emulator only: on-device time per frame (Model C + Model A + crop) median 94, 102, 114, 149 ms over 4 full attempts, p90 128–335 ms. v1: ~3.5 s per frame | Under the 300 ms median target on the emulator (Android emulator on an Apple-silicon Mac, Pixel 10 Pro image, replay of the project clips). Not measured on a physical phone yet; real-phone fps and latency come with the demo (issue #2). Cloud calls are now off the per-frame path (Model B 2.7–4.8 s per call, coaching 3.1–8.4 s, both after the verdict). |
| 12 | Gemini valid JSON ≥ 95% | Not measured as a rate. In the Nov 25 emulator runs, base Gemini coaching returned valid JSON every time | The JSON feedback contract is implemented in the app. The emulator runs are a handful of attempts, too few to state a rate, and none was on a physical phone. |
| 13 | EARLY_BREAK demonstrated on real attempts | Proven synthetically, plus one constructed splice of two real clips (seq05 lying → seq01 standing) on the emulator, Nov 25 | No licensed real clip contains a genuine break; none was faked, and the splice is labelled as constructed, not a real break. |
| 14 | Tuned model trained on the current dataset | Tuned Gemini is still v3-trained (171 images) while v4 (1,000) exists | v4 was collected after the tuning run. Model A is trained on v4; retraining Gemini on v4 was not done (issue #5). |

What the proposal asked for that WAS delivered as specified: the MVP command set (Sit from Stand, Down from Sit, Stay from Sit/Down, 3.0 s response window, 0.6 s hold, 5.0 s default Stay), the READY gate (v1: Start disabled until the required starting posture is stable; Nov 25 build: START shows "START (waiting)" and refuses a plain tap until then, a long press overrides it and is logged), the rule-based temporal state machine (Python reference 17/17 unit tests; the app's Java `SequenceSession` 21 unit tests), the JSON coaching contract (v1: status, break_time_s, evidence_summary, coaching_tip, caution, local verdict wins on disagreement; Nov 25 build: base Gemini is given the verdict and returns evidence_summary, coaching_tip, caution, with a template tip as fallback), and a dataset split with zero leakage (split by source, 626/139/235).

## 5. Video input and output — and what can be inferenced later

The system is video in, verdict out, and that path already runs two ways:

1. **Offline (Python, proven):** `video-sequence-layer/` — video file → frames → per-frame classification → state machine → verdict + JSON feedback. Run on 6 real clips / 111 frames on Nov 11, 2026 (tuned Gemini 6/6, Model A 5/6, ~3.5 s/frame). Entry points: `sequence_evaluate.py`, `evaluate_model_a_video.py`, `run_live_tuned_video.py`.
2. **Live (phone app, run on the emulator, not yet on a physical phone):** in v1, Camera2 sampled at 1 fps and each frame went to the cloud tuned endpoint. Since Nov 25 (GitHub B branch `app-three-model-integration`, not yet merged to main), CameraX frames are analysed on the phone at up to ~5 fps: Model C (EfficientDet-Lite0) finds the dog, Model A (`model_a.tflite`) reads the posture of the whole frame, and the time-based state machine in Java (`SequenceSession.java`) gives the verdict. Model B then re-checks 2–3 decisive frames in the cloud. Replay mode feeds the project clips through the same path.

For later inference: any new video can be fed through the offline path with no new training. On-device inference, which issue #7 listed as the work that unlocks 5 fps sampling and sub-300 ms latency, is now in the app. Measured on the Android emulator only: on-device median 94–149 ms per frame over 4 full attempts (p90 128–335 ms), vs ~3.5 s per frame before. Full-system emulator results (Nov 25, replay, all three models): Stay lying 5 s on seq02 and seq05 SUCCESS, Walk 5 s on seq03 SUCCESS, a constructed splice (seq05 lying joined to seq01 standing, not a real break) EARLY_BREAK at 1.5 s, all VERIFIED by Model B; a Stand attempt on seq01 never started because Model A reads that standing dog as sitting at 91–99%. Real-phone fps and latency come with the demo (issue #2). The Nov 11 numbers above were produced with the Python state machine at 1 fps and stay valid for that run; the Python reference is now behind the Java one.

## 6. Decisions that shaped the result

- **NO Flutter (Nov 11, owner decision):** the updated proposal says Flutter; the working stack is Android. Outcome over methodology — the rubric grades the app and its results, not the framework.
- **Model B correction (Nov 11):** Model B IS a gemini-3.5-flash fine-tune. The real deviation is its tuning data (single-image classification on v3), not the model ID. Fixed in METHODOLOGY.md and the report.
- **Cloud-first sequencing (Nov 11, since replaced):** v1's live path classified in the cloud because the tuned model lives there and measured higher (88.4% vs 74.9%), and on-device inference was planned as future work. Replaced Nov 25 (GitHub B branch, not yet merged): Models C and A judge every frame on the phone so timing can follow the proposal (~5 fps, under 300 ms on the emulator), and tuned Gemini, the more accurate model, checks the verdict's decisive frames instead of producing it.
- **Honest limits, stated not hidden:** the report, METHODOLOGY.md, and GitHub B Section 4 all carried the same limits on Nov 11 (no on-device run yet, v3-trained Gemini, 4 classes, 1 fps, open-data dogs); the report PDF and GitHub B main's Section 4 still carry that v1 list. As of Nov 25, per METHODOLOGY.md and GitHub B Section 4 on branch `app-three-model-integration`: on-device inference runs, but only on the emulator, not yet on a physical phone; v3-trained Gemini, 4 classes and open-data dogs are unchanged. No synthetic result is presented as a real one.

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
