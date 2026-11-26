# Cloud Shell export — Gemini evaluation files, 2026-11-26

Exported from the Cloud Shell home of khagendrakhatri365@gmail.com
(project cs663-dog-training) on 2026-11-26. Archive
`gemini-eval-export-2026-11-26.tar.gz` (10,608 bytes, sha256
c05aef4d7b0051efeafdd9c21a702cb0ef16cbbc0e79344c80770f15df976fb2)
contains exactly these 14 files. No secrets in any of them.

## Per-image evaluation results (43 held-out test images each)

- `tuned_eval_winner.json` — tuned gemini-3.5-flash via endpoint
  .../endpoints/514008491845746688, location us. 38/43 = 88.4% strict,
  3 invalid, 0 errors.
- `base_eval_gemini-3.5-flash_global.json` — base gemini-3.5-flash,
  location global. 37/43 = 86.0%, 3 invalid, 0 errors.
- `base_eval_gemini-3-flash-preview_global.json` — base
  gemini-3-flash-preview, location global. 40/43 = 93.0%, 1 invalid,
  0 errors.

## Do NOT quote as accuracy

- `base_eval_results.json` and `base25_eval_results.json` — error-only
  files from Oct 30: all 43 predictions are 404 errors (publisher model
  not found in us-central1). They record a failed run, not 0% accuracy.

## Tuning job A (1708100736908263424, dog-training-gemini-v1)

- `tuning_job_1708100736908263424.json` — `gcloud ai tuning-jobs
  describe` output. Base gemini-3.5-flash, us-central1, 1 epoch,
  adapter size 4, learning-rate multiplier 1.0. Created
  2026-10-30T06:56:49Z, ended 2026-10-30T23:53:57Z, SUCCEEDED.
- `tuning_curves_1708100736908263424.txt` — the training/validation
  loss and accuracy values read off the Vertex console Monitor page.
  There is no loss file: the describe JSON has no loss fields and no
  tuning/loss objects exist in GCS for this job. Step 1: train loss
  0.17, val loss 0.349, train acc 0.9692, val acc 0.9663. Step 2
  (checkpoint 1, epoch 1): train loss 0.263, val loss 0.347, train acc
  0.949, val acc 0.966.

## Eval scripts and logs

`base_eval.py`, `base25_eval.py`, `run_eval.py`, `tuned_eval_uc.py`,
`tuned_eval_us.py`, `base_eval.log`, `base25_eval.log` — the scripts
and logs that produced the evaluations above, Oct 30–31.
