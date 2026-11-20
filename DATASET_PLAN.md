# DogTrain-Attempts — dataset build plan (working title)

**Dog Training Assistant · Group 5 (Khagendra Khatri, Arnav Verma) · CS663 Fall 2026**

The publishable dataset behind the Project 2 system: an attempt-level video dataset of real dogs performing Sit, Down, and Stay attempts, with per-frame posture labels, timelines, and trainer verdicts, split by dog. This plan starts after the Dec 1, 2026 Results submission. Nothing here is due before that.

## Why this dataset

No public dataset has training attempts with timelines. The sensor datasets have the behaviors but no video; the image work has stills but no attempts. This dataset is also the missing piece for the genuine multimodal Gemini fine-tune (proposal Section 7) and for the paper (system + dataset). The three baselines already exist (Model A, YOLOv8n, tuned Gemini) and the state machine already runs, so the dataset gets a benchmark the day it is labeled.

## What gets recorded

- Commands: Sit, Down, Stay (the MVP set).
- Target: **100 attempts across 6 to 8 dogs**. Mix to aim for: about 60 SUCCESS, 25 EARLY_BREAK, 15 UNCERTAIN. Breaks are rare in real training, so plan deliberate break captures or accept a SUCCESS-heavy mix and say so.
- Per command: about 30 attempts (Sit ~35, Down ~35, Stay ~30).
- Per attempt: one video, 5 to 15 seconds, phone on tripod, fixed side view, whole dog in frame, 30 fps recorded, labeled at 1 fps (the system's time base). About 5 to 15 labeled frames per attempt, 800 to 1,200 labeled frames total.
- Per dog: 10 to 20 attempts, mixed commands.

## Recording protocol

- Phone on tripod, side view, dog fully in frame, plain background, good light.
- Handler gives the command on camera. Record from the start posture through the verdict (hold or break). One attempt per clip, no cuts.
- File name: `dogID_command_attemptNN.mp4` (e.g. `dog03_stay_07.mp4`).
- Owner consent in writing (a message or a short form) before recording: permission to record, label, and publish the video under CC BY 4.0. Positive reinforcement only, no stressful training.

## Annotation

- Per frame (1 fps): posture label (sitting / standing / lying down / walking).
- Per attempt: start posture, target posture, time target reached (seconds), break time (seconds) or none, verdict (SUCCESS / EARLY_BREAK / UNCERTAIN), coaching note (one line).
- Two sets of eyes: one person labels, the other checks a 20% sample. Report agreement (Cohen's kappa) on the checked subset.
- Format: one JSONL record per attempt, plus a CSV summary. Videos in MP4 H.264. The annotation schema lives in the repo.

## Dog split (the benchmark)

- Train: 4 to 5 dogs. Validation: 1 dog. Test: 2 dogs, never seen in training. Dog identity is the split key, not random frames.
- Report per-dog results so a reader can see which dogs are hard.

## Baselines (already built, to re-run on this data)

- Model A (MobileNetV3Small, v4-trained): frame accuracy, per-class F1.
- YOLOv8n: detection plus posture proxy.
- Tuned Gemini (v3): frame accuracy.
- State machine: attempt verdict accuracy, break-time error (MAE), false-break rate.
- Then: the genuine multimodal fine-tune on this dataset's timelines (the proposal Section 7 experiment), evaluated the same way.

## Release

- License: CC BY 4.0 (own recordings, clean ownership).
- Host: GitHub repo (data + code) plus a Zenodo DOI for the dataset archive. NeurIPS Evaluations & Datasets requires a stable host and a datasheet; write the datasheet from the NeurIPS template.
- Include: videos, JSONL annotations, CSV summary, datasheet, baseline results, and the v4 image set as supplementary material.

## Timeline (starts after Dec 1)

- Dec 2–7: recruit dogs and owners, get consent, test the recording setup on one dog.
- Dec 8–21: record attempts (2 to 3 weekends, 15 to 20 attempts per session).
- Dec 22–Jan 4: label and check (1 to 2 weeks, evenings).
- Jan 5–25: baselines, multimodal fine-tune, evaluation.
- Jan 26–Feb 15: paper draft (4 pages for CV4Animals, longer for ACI / NeurIPS).
- Targets: CV4Animals 2027, about March 2027 (4-page, non-archival); ACI 2027 Emerging Work, about May 2027; NeurIPS Evaluations & Datasets, about May 2027 (stretch goal). Confirm every date on the official call for papers when it opens.

## Who does what (2 people)

- Khagendra: recording, labeling, baselines, fine-tune, paper draft.
- Arnav: recording, labeling check, paper.
- Kokila Ganesan dropped from Group 5 (Nov 19, 2026) and is not on this work.

## Risks (honest)

1. **Dogs are the bottleneck.** Neither of us owns a dog. Every dog comes from a classmate, friend, or family member, and each needs owner consent. If we cannot get 6 dogs, the dog-separated test split collapses and the benchmark claim weakens. Mitigation: start recruiting the week of Dec 2. A 3-dog pilot (train 2, test 1) is the minimum viable version.
2. **Real breaks are rare.** Most real attempts succeed, so EARLY_BREAK will be thin. Mitigation: deliberately record attempts where the dog is likely to break (longer stays, distractions), and report the mix honestly.
3. **Labeling time.** About 1,000 frames at roughly 30 seconds each is about 8 hours. Mitigation: label at 1 fps only, and check a sample rather than double-labeling everything.
4. **Bandwidth.** Two people, Spring 2027 load (CSUEB plus GT OMSCS). Mitigation: CV4Animals 4-page paper first; NeurIPS Evaluations & Datasets only if the data is strong.

## Cost

$0. Phone cameras, free tools, free hosting (GitHub + Zenodo). The cost is time: roughly 40 to 60 hours across the group.
