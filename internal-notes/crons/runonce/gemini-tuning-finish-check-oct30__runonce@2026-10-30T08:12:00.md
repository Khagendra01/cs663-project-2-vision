---
id: gemini-tuning-finish-check-oct30
title: Gemini tuning finish check + evaluation — Oct 30 morning
enabled: true
owner: goal:cs663-project-2-dog-training-assistant
mode: task
schedule:
  kind: runonce
  timezone: America/Los_Angeles
  at: 2026-10-30T08:12:00
timeout_secs: 3600
delivery:
  - chat_id: 8ca9864e-fb49-451b-9712-5a6fb4cf988b
metadata:
  tags: [cron:flexible-time]
  originating_chat_context_json: '{"chat_id":"8ca9864e-fb49-451b-9712-5a6fb4cf988b","origin_provider":"main","chat_kind":"direct","provider":"main","event_kind":"message","require_mention":false,"device_id":"ed6d136a-9144-4ddc-aa3e-276bb74d356f"}'
  presentation_locale: en-US
---
Check the CS663 Gemini tuning jobs and complete the evaluation so the exercise is ready for Khagendra's review before the Fri Oct 30, 2026 11:00 AM PT deadline.

CRITICAL UPDATE (Oct 30 ~08:10 AM): direct publisher-model inference is currently 404ing for BOTH models in this project/region via the vertexai.generative_models SDK path used in Cloud Shell: all 43 base calls to gemini-3.5-flash AND all 43 to gemini-2.5-flash returned 404 "Publisher model ... was not found or your project does not have access to it" (us-central1). The resulting ~/base_eval_results.json and ~/base25_eval_results.json files contain ONLY errors — they are NOT accuracy results; never quote them as 0% accuracy. A diagnostic is being run in the main chat to find a working inference pattern (candidates: versioned model ID e.g. gemini-2.5-flash-001, location "global", or the google-genai SDK Client(vertexai=True), or Model Garden enablement). Before running ANY evaluation, first confirm a single test call actually returns text; if the main chat has already found and reported a working pattern by the time you run, use it. If no pattern works for base publisher models, a SUCCEEDED tuning job's tuned model may still serve through its endpoint resource name — test that first with one image.

Jobs, Google Cloud project cs663-dog-training (account khagendrakhatri365@gmail.com, signed in in the browser; project number 917815040923, us-central1):
- Job B (primary): projects/917815040923/locations/us-central1/tuningJobs/6332118877752786944 — base gemini-2.5-flash, display name dog-training-gemini-25-v1, submitted Oct 30 07:40 AM PT. Khagendra confirmed 2.5 is fully acceptable; on-time submission is the priority.
- Job A (stuck): projects/917815040923/locations/us-central1/tuningJobs/1708100736908263424 — base gemini-3.5-flash, dog-training-gemini-v1, stuck in "Preparing for tuning" since Oct 29 23:56 PT (the 404 pattern is the likely cause).
Both: 1 epoch / LR 1.0 / adapter 4, train gs://cs663-dog-training-dog-data/datasets/train.jsonl, validation .../validation.jsonl, test .../test.jsonl (43 held-out; supports: sitting 9, standing 11, walking 9, lying down 14). EXACT prompt from training records: "Look at the image of the dog. What is the dog doing? Answer with exactly one of: sitting, standing, lying down, walking."

Steps:
1. Check BOTH jobs' states.
2. If Job B SUCCEEDED: get its tuned model resource name / endpoint. Verify ONE tuned-model call on one test image returns text (via its endpoint resource name). Then predict all 43 test images with the tuned model; compute tuned accuracy, per-class accuracy, confusion matrix. For the base comparison, use a VERIFIED working base-model call pattern (see critical update) for gemini-2.5-flash on the same 43 images; if no base pattern works, say so explicitly in the draft rather than inventing numbers. Save results under ~/workspace/goals/cs663-project-2-dog-training-assistant/files/gemini-finetune/results/ and fill the [RESULT] placeholders in submission-draft.md in that folder, including 5–10 spot-checks with failures and an honest conclusion. Report to Khagendra: Job B succeeded, tuned accuracy, base accuracy (or why the base number is unavailable), draft ready for review. Do NOT submit on Canvas — his review is the gate. Do NOT cancel Job A; report its state.
3. If Job B still RUNNING but Job A SUCCEEDED: same evaluation with Job A's tuned model; base comparison via any verified working pattern, disclosed in the draft.
4. If BOTH still RUNNING: report both states plainly and urgently with runtimes (deadline 11:00 AM PT today).
5. If a job FAILED: report its exact error verbatim; do not relaunch anything.
