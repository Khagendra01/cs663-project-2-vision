# CS663 Project 2 — Results: working plan
**Dog Training Assistant · Group 5 · due Tue Dec 1, 2026, 11:00 AM (Canvas "Project 2 - Results")**

Source: Canvas "Project 2 - Results" rubric, checked 2026-10-20 (side chat with Canvas).
The public course site carries no due dates — Canvas is the source of truth.

## What the professor actually grades (200 pts)
| Pts | Item |
|---|---|
| 160 | The application itself — all portions implemented and running, all deliverables turned in |
| 20 | Testing docs: accuracy/loss curves + **10 unique test cases** for GitHub A; app-start/results for GitHub B |
| 5 | GitHub B docs Section 1: app execution instructions |
| 5 | GitHub B docs Section 2: code description |
| 5 | GitHub B docs Section 4: what's incomplete / errors |
| 5 | GitHub B docs Section 5: YouTube live demo of the app discussing accuracy |

Plus per the project spec (Project2New.html): mobile app with **live camera input**, 3 models (A: student-trained/fine-tuned vision model, B: fine-tuned Gemini multimodal, C: additional pretrained/foundation model), **2 GitHub repos**, Documentation Report PDF on Canvas, narrated YouTube demo of the **actual app via camera** (explicitly no code/Colab in the video). Proposal is done (28/30); professor's feedback warned the group may hit implementation issues — that warning is now the plan's main risk.

## What already exists
- **Model A candidate:** your ActiveRec APK — MobileNetV2 + LSTM classifier trained on UCF101-style clips, install-ready at `~/workspace/your_files/cs663-lstm/ActiveRec.apk` (verified working, val accuracy 1.0000 on 5 classes). **The gap:** it picks clips, it doesn't take live camera input. Closing that gap is the single biggest build item.
- **Model B:** nothing yet — fine-tuned Gemini multimodal needs a training set + fine-tuning access decided.
- **Model C:** nothing yet — pick a pretrained/foundation model that fits the dog-training use case.
- **Group:** only Arnav Verma (posted the proposal) is visible so far; other Group 5 members unknown.

## Phase plan (41 days)
### Phase 0 — Group alignment (by Mon Oct 27)
- [ ] Identify the other Group 5 members; agree on roles: app/Android dev, Model A owner, Model B owner, docs+testing lead.
- [ ] Lock the app scope: which dog-training behaviors the app recognizes (e.g., sit/stay/come posture detection vs. breed ID vs. trick timing) — scope decides Models B and C.
- [ ] Decide Model C (suggestion: a mobile-friendly pretrained detector/pose model that runs on-device).
- [ ] Create the 2 GitHub repos (A and B) with READMEs.

### Phase 1 — Model A + live camera (Oct 28 – Nov 10)
- [ ] Adapt ActiveRec to live camera input (CameraX or equivalent) — the core build.
- [ ] Record accuracy/loss curves for the Model A training run (reuse your TensorBoard logs).
- [ ] Design the **10 unique test cases** for GitHub A (different dogs, lighting, angles, distances, backgrounds); start capturing them.

### Phase 2 — Model B integration (Nov 11 – Nov 20)
- [ ] Fine-tune Gemini multimodal on the dog-behavior set; integrate behind a clean interface.
- [ ] Capture app-start and results screenshots/video for GitHub B.
- [ ] Write GitHub B Section 1 (execution instructions) and Section 2 (code description).

### Phase 3 — Testing docs + demo video (Nov 21 – Nov 27)
- [ ] Finalize all testing documentation (curves + 10 test cases + app start/results).
- [ ] Write Section 4 (incomplete/errors) — be honest; the rubric rewards it.
- [ ] Record the narrated YouTube live demo: **full screen**, actual app via camera, discussing accuracy — no code/Colab in the video. Per your standing rule: capture **every** visual, not just the required subset, and compile into a single PDF alongside.

### Phase 4 — Submission buffer (Nov 28 – Dec 1, 11:00 AM)
- [ ] Documentation Report PDF on Canvas; Text/Web URL/Media/Upload all accepted, unlimited attempts.
- [ ] Verify all 2 repos, video link, and PDF render correctly before the 11:00 AM cutoff.

## Open risks
1. Live camera + 3 models on Android is the hard part (professor flagged this exact risk at 28/30).
2. Gemini fine-tuning access/cost not yet scoped — decide early in Phase 0.
3. Group coordination is invisible so far — only Arnav has surfaced; nail down owners in Phase 0.
