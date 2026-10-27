# Gemini Fine-Tuning Exercise — Plan (for review before any training)

**Owner of this plan:** Muse (leading end-to-end) — Khagendra reviews everything below and approves changes before any training job is launched. Nothing trains until he says go.

**Status (Oct 26, evening): plan approved, execution started.** Prep kit built (see section 11). Pipeline completed end-to-end through a demo smoke test (see section 12). No training launched, no cloud resources created — those need his Google Cloud account and real data, both still pending.

**Exercise:** "Exercise - Fine-tuning Gemini", CS663 Computer Vision (Prof. Lynne Grewe)
**Due:** Fri Oct 30, 2026, 11:00 AM PT — group submission, 30 points, counts for the whole CS663 group
**Canvas instruction (verbatim essence):** modify the corresponding lab Colab so it fine-tunes Gemini on YOUR custom dataset for Project 2, using Google Colab credits
**Model B link:** this exercise builds Project 2's Model B — the fine-tuned multimodal Gemini — later wired into the app via the Cloud Run backend lab (due Nov 6).

## 1. The thesis (one paragraph)

Fine-tune Gemini 3.5 Flash so that, given a single dog image (a video frame, not a text description), it returns the dog's posture/behaviour as one constrained label. The tuned model must be genuinely multimodal — it reads the image itself — and it must be evaluated on dogs it never saw in training, against the untuned base model on identical inputs. If the tuned model beats base on held-out dogs, Model B is justified; if not, the report says so honestly.

## 2. Decisions needed from Khagendra (at Phase 0, Tue Oct 27)

These are proposals, not settled facts. The group has not locked scope yet.

1. **Behaviour label set (proposed):** sitting, standing, lying down, walking/coming, rolling over. Start with 4–5 classes max; every added class multiplies the data needed. "Roll over" is video, not a single pose — if kept, it needs short frame sequences or a clearly defined mid-roll frame, otherwise drop it for this exercise.
2. **Who owns what:** proposed — Khagendra leads the fine-tune pipeline (dataset spec, JSONL, tuning job, evaluation); one teammate owns data collection (recording dogs); one owns the writeup + Canvas submission. Adjust at Phase 0.
3. **Dataset source:** group members' own dogs, recorded on phones, frames extracted from video. Own recordings avoid licensing/terms-of-use problems entirely. No random internet dog photos unless terms allow it and the source is documented.
4. **Tuning route:** Option 1 — Google Cloud Agent Platform Console (the current, documented route). Option 2 (Colab) still needs updating from the Vertex AI API to the Agent Platform API, so it is the fallback only.

## 3. Dataset specification

- **Format:** one JSON object per line (JSONL, UTF-8, no array wrapper, no comments). Each record:
  - user message: image reference (`fileData`, MIME type image/jpeg or image/png, `fileUri` = gs:// URI of the uploaded image) + text prompt
  - model message: the target label text, exactly one class name
- **Prompt (fixed for every record — consistency is what the model learns):**
  "Look at the image of the dog. What is the dog doing? Answer with exactly one word: sitting, standing, lying down, walking, or rolling over."
  (Final word list = the locked label set from decision 1.)
- **Splits — split by dog/session, never by random shuffle:**
  - train: ~100–200 examples, multiple dogs, multiple angles/lighting per class
  - validation: ~20–30 examples (optional for the tuning job, used to sanity-check settings)
  - test: ~30–50 examples, dogs that appear NOWHERE in train/validation (this is the honest evaluation set, and it is also what the proposal feedback demands)
- **Quality rules:** every image actually shows the labelled behaviour; blurry, occluded, or ambiguous frames are dropped, not kept; no people faces or identifying info in frame filenames; each record's image URI must match the uploaded object exactly.

### Example record (shape only — bucket name is a placeholder)

```json
{"contents":[{"role":"user","parts":[{"fileData":{"mimeType":"image/jpeg","fileUri":"gs://YOUR_BUCKET/images/dog03_sit_014.jpg"}},{"text":"Look at the image of the dog. What is the dog doing? Answer with exactly one word: sitting, standing, lying down, walking, or rolling over."}]},{"role":"model","parts":[{"text":"sitting"}]}]}
```

## 4. Google Cloud setup

1. Console: console.cloud.google.com → select the project → confirm billing is enabled and the Vertex AI API is on.
2. Check current Vertex AI tuning pricing and quotas BEFORE launching — tuning and inference can incur charges.
3. Create a Cloud Storage bucket (globally unique name), upload images first into `images/`, then `train.jsonl` and `validation.jsonl` into `datasets/`. Verify every gs:// URI.

## 5. Tuning job specification (pilot first)

- Path: Agent Platform > Models > Tuning → Managed tuning (not Custom tuning)
- Region: us-central1 (shown in the guide; confirm the chosen base model supports tuning there)
- Method: Supervised fine-tuning
- Base model: gemini-3.5-flash (Preview) — confirm it is still offered for tuning at launch time; the guide notes 3.6 Flash is not offered for tuning
- Display name: e.g. `dog-training-gemini-v1`
- Pilot settings: **1 epoch, learning rate multiplier 1.0, adapter size 4** (the guide's starter values — cheap pipeline validation)
- If the pilot succeeds and validation supports it: rerun at 2–3 epochs
- Record: project ID, region, dataset URIs, model name, settings, job outcome

## 6. Evaluation plan (what the report must contain)

- **Primary metric:** accuracy on the held-out test set = exact label match. Constrained one-word outputs make this unambiguous.
- **Per-class accuracy + confusion matrix:** shows which behaviours get confused (e.g. standing vs walking).
- **Tuned vs base comparison:** run the SAME test inputs through the untuned gemini-3.5-flash and report both accuracies side by side. This is the key result.
- **Qualitative spot-check:** 5–10 test examples shown with model output, including failures — failures reported honestly, not hidden.
- **What "good" looks like:** tuned accuracy clearly above base on unseen dogs. If it is not, the report states that and names the likely cause (too little data, ambiguous frames, class imbalance) instead of claiming success.

## 7. Testing method (Colab)

Adapt the guide's testing Colab: authenticate to the same project, load test.jsonl, send ONLY the user parts (image + question) to the tuned model ID, collect outputs, compare to reference labels, save predictions + results separately. Never send the expected answer as part of the input. Repeat with the base model for the comparison.

## 8. Timeline to the Oct 30, 11:00 AM deadline

- **Tue Oct 27 (Phase 0):** lock label set, roles, dataset source; Khagendra approves this plan (with edits)
- **Wed Oct 28:** record dogs, extract frames, label, build train/validation/test JSONL
- **Thu Oct 29:** upload to Cloud Storage, launch 1-epoch pilot, test, evaluate; rerun at 2–3 epochs only if justified
- **Fri Oct 30 (morning):** finish writeup + evaluation report, submit on Canvas before 11:00 AM

## 9. Risks and open questions

- Tuning job queue time is unknown — the pilot must launch Oct 29 at the latest, not Oct 30 morning.
- Billing/quota: if the project lacks billing or quota, the job cannot start; check this Oct 27, not Oct 29.
- Dataset collection is the real bottleneck: if the group cannot record enough dogs by Oct 28, shrink the label set (3 classes) rather than faking data.
- Base-model tuning availability can change (Preview status); verify at launch.
- The Canvas instruction mentions Colab credits; the console route's charges are separate — confirm which account/credits the group is actually using before launch.

## 10. Review gates (nothing proceeds past a gate without Khagendra's approval)

1. ✅ This plan (now)
2. Dataset spec + 10 sample labelled records (after Phase 0)
3. Final JSONL files + tuning settings (before Start tuning is clicked)
4. Evaluation report + Canvas submission text (before submitting)

## Submission checklist (Canvas, group submission)

- [ ] Tuned model ID + base model + region
- [ ] Dataset description: classes, counts per split, split-by-dog statement
- [ ] Tuning settings (epochs, learning rate multiplier, adapter size)
- [ ] Evaluation table: tuned vs base accuracy, per-class accuracy, confusion matrix
- [ ] Spot-check examples including failures
- [ ] Link or attachment for the testing Colab / results file

## 11. Prep kit (built Oct 26, all in `files/gemini-finetune/`)

- `dataset-collection-guide.md` — recording rules, folder structure, target counts, human review pass, JSONL build steps
- `train-template.jsonl` — 5 example records (one per proposed class), placeholder bucket URIs; shape reference only, not training data
- `extract_frames.py` — ffmpeg-based frame extraction from clips (1 fps default) into dog/behaviour folders
- `test_tuned_model.py` — Colab script: runs tuned + base models over test.jsonl, writes predictions.jsonl
- `evaluate.py` — scores predictions: tuned vs base accuracy, per-class accuracy, confusion matrix

All scripts syntax-checked; the template parses as valid JSONL. Real data collection starts after Phase 0 locks the label set and roles.

## 12. End-to-end pipeline status (built Oct 26 night)

Everything that can be built without real data and without the Google Cloud account is built and smoke-tested:

- `build_jsonl.py` — frames → dog-based train/validation/test JSONL, with validation (labels, JSON, no dog overlap). Smoke-tested on 4 AI-generated demo dog photos (2 dogs × 2 behaviours, clearly marked DEMO, never for submission): bella → train (lying down, walking), milo → test (sitting, standing), zero dog overlap, all records valid.
- `upload_to_gcs.sh` — images → gs://BUCKET/images/, JSONL → gs://BUCKET/datasets/ (syntax-checked; needs `gcloud auth login` + project with billing to actually run).
- `test_tuned_model.py` + `evaluate.py` — Colab run of tuned vs base over test.jsonl, scoring script (accuracy, per-class, confusion matrix).
- `submission-draft.md` — Canvas writeup template with fill-ins for the real run's results.
- Demo data lives in `files/gemini-finetune/demo-data/` and `demo-jsonl/` — DELETE or ignore these before the real run; they must never be uploaded as training data.

Still genuinely blocked (only Khagendra / the group can unblock):
1. Real dog videos/frames — the group's dogs, after Phase 0 locks the label set and roles.
2. Google Cloud project with billing + sign-in — bucket creation, tuning job, and inference all run under his account.
3. Review gate — per his own rule, he reviews the real dataset spec + sample records before Start tuning is clicked, and reviews the evaluation report before Canvas submission.

When those three clear, the remaining sequence is mechanical: record → extract → review frames → build_jsonl → upload → pilot (1 epoch) → test → evaluate → fill submission draft → submit before Oct 30, 11:00 AM.

## 13. Real dataset — expanded v3 (built Oct 27, 2026, supersedes v2)

At Khagendra's "Get more", a third Commons round added 61 verified photos to v2's 110,
re-split fresh. The dataset is real openly-licensed Wikimedia Commons photos + video frames,
**4 classes: sitting, standing, lying down, walking** (rolling over stays out, video-only).

- **171 examples, all visually verified** (640px JPEG): train 102, validation 26, test 43.
  Per class: sitting 35 (21/5/9), standing 42 (25/6/11), walking 37 (22/6/9), lying down 57 (34/9/14).
  Split BY SOURCE (video id or photo uploader) — no source appears in two splits. See
  `files/gemini-finetune/real-dataset.md` (v3 section) and
  `files/gemini-finetune/real-data-v3/provenance.json` for every item's source and licence.
- **JSONL ready:** `files/gemini-finetune/real-jsonl-v3/` (train/validation/test), prompt lists
  exactly the 4 classes ("Answer with exactly one of: sitting, standing, lying down, walking.").
- **Verification honesty (round 3):** 27 of 88 candidates dropped (wrong behaviour, distant dogs,
  running mistaken for walking, person-dominant shots).
- **Disclosed caveats (must be stated in the Canvas report):** licensed stock footage, not the
  group's own dogs (swap-in path unchanged); v3's test set is all photos (fresh split moved
  `vid_wagging` to train) — still fully held-out sources, but state the modality mix.
- **Rejected and excluded:** Tokyo dog-walker pack, Boxer 1901 footage, Buster and the Dude, Hondenwandeltocht, St. Bernardshonden, and assorted irrelevant search hits.

Remaining to run: upload to a bucket (`upload_to_gcs.sh`), launch the 1-epoch pilot tuning job, test, evaluate, fill `submission-draft.md`, submit before Oct 30, 11:00 AM. All of that still waits on his Google Cloud account + his review of these real records (review gate 2).
