# CS663 Project 2 — Dog Training Assistant

Computer Vision (CS663), California State University, East Bay — Fall 2026
Group 5: Arnav Verma, Khagendra Khatri, Kokila Ganesan

An intelligent mobile vision system that watches a dog through the phone camera and tells the trainer, in real time, whether the dog is holding a commanded behaviour — sitting, standing, lying down, or walking.

Proposal graded 28/30. Final deliverable (Project 2 - Results) due Dec 1, 2026.

## What's in this repo

| Folder | Contents |
|---|---|
| `gemini-finetune/` | The custom dataset (v4: 1,000 images, 250/class), fine-tuning pipeline, and evaluation for the Gemini model (Model B) |
| `video-sequence-layer/` | The video half of the project: temporal state machine (frame postures → SUCCESS / EARLY_BREAK / UNCERTAIN, with smoothing), 17 unit tests, sequence evaluation harness, and 6 real-video test sequences. See its own README. |
| `cloud-run-backend/` | Cloud Run backend (Node / Express, `@google/genai`) that serves the base and fine-tuned Gemini models to the Android app |
| `android-app/` | Older v1 Android app (package `com.cs663.dogbackend`) for the Cloud Run lab: camera / photo picker, prompt field, calls both backend endpoints and shows base vs tuned results. Built APK: `android-app/app-debug.apk`. This is not the current app: the three-model app (Models C and A on the phone, Gemini checking the verdict) is in GitHub B (https://github.com/Khagendra01/cs663-project-2-app), branch `app-three-model-integration`, not yet merged to main |
| `docs/` | Proposal, accomplishment report, lab deliverables, and project plans |
| `internal-notes/` | Working check notes and project history |

## Dataset

171 visually verified dog images, all licensed stock (Wikimedia Commons, CC0 / CC BY / CC BY-SA — see `gemini-finetune/real-dataset-licenses.txt`). It is not the group's own dogs.

- Split by source, so the test set is unseen dogs/sources: train 102 / validation 26 / test 43
- Lying down 57, standing 42, walking 37, sitting 35
- Versions kept in full: v1 (`real-data/`, 37), v2 (`real-data-v2/`, 110), v3 (`real-data-v3/`, 171, final)
- JSONL for Vertex AI tuning: `gemini-finetune/real-jsonl-v3/` (train / validation / test)
- Source videos the frames were extracted from: `gemini-finetune/real-videos/`
- Exact prompt used in every training record and evaluation: "Look at the image of the dog. What is the dog doing? Answer with exactly one of: sitting, standing, lying down, walking."

## Results (Gemini fine-tuning exercise)

Fine-tuned on Vertex AI in Google Cloud project `cs663-dog-training`, evaluated on the 43 held-out test images, strict exact-match scoring:

- Tuned (gemini-3.5-flash, 1 epoch): 38/43 = 88.4%
- Base gemini-3.5-flash: 37/43 = 86.0%
- Per class (tuned): lying down 14/14, sitting 8/9, walking 8/9, standing 8/11

Full numbers: `gemini-finetune/results/tuned-results.md` and `gemini-finetune/results/base-results.md`.

The tuned model serves through its deployed endpoint (client location `us`): see `cloud-run-backend/.env.example`.

## Pipeline scripts

In `gemini-finetune/`:

- `download_candidates.py`, `download_round3.py`, `download_sitting2.py` — collect candidate stock photos/videos
- `extract_frames.py` — extract frames from source videos
- `build_jsonl.py`, `assemble_v2.py`, `assemble_v3.py` — build the train / validation / test JSONL, split by source
- `upload_to_gcs.sh` — upload images and JSONL to the Cloud Storage bucket
- `test_tuned_model.py`, `evaluate.py` — evaluate tuned vs base (accuracy, per-class, confusion matrix)

## Backend

```bash
cd cloud-run-backend
npm install
npm start   # node index.js, PORT 8080
```

Copy `.env.example` to `.env` for local testing. Never commit a real `.env` or API key.

## Maintainer

khagendra01 — khagendrakhatri365@gmail.com
