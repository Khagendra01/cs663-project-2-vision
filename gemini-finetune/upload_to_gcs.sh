#!/bin/bash
# Upload labelled frames + JSONL files to Cloud Storage.
# Prerequisites (run once, on the machine with the Google Cloud account):
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT_ID
# Usage:
#   bash upload_to_gcs.sh YOUR_BUCKET dog-data jsonl
# Uploads:
#   dog-data/*/*/*.jpg  ->  gs://YOUR_BUCKET/images/
#   jsonl/*.jsonl       ->  gs://YOUR_BUCKET/datasets/
# After upload, verify every gs:// URI in the JSONL files matches an
# uploaded object before launching the tuning job.
set -euo pipefail
BUCKET="${1:?usage: upload_to_gcs.sh BUCKET DATA_DIR JSONL_DIR}"
DATA_DIR="${2:?usage: upload_to_gcs.sh BUCKET DATA_DIR JSONL_DIR}"
JSONL_DIR="${3:?usage: upload_to_gcs.sh BUCKET DATA_DIR JSONL_DIR}"

echo "creating bucket gs://${BUCKET} (skipped if it already exists)..."
gsutil mb -l us-central1 "gs://${BUCKET}" 2>/dev/null || echo "bucket exists or creation skipped"

echo "uploading images..."
gsutil -m cp "${DATA_DIR}"/*/*/*.jpg "gs://${BUCKET}/images/" 2>/dev/null \
  || gsutil -m cp "${DATA_DIR}"/*/*/*.png "gs://${BUCKET}/images/" \
  || echo "no images found under ${DATA_DIR}"

echo "uploading JSONL..."
gsutil cp "${JSONL_DIR}"/*.jsonl "gs://${BUCKET}/datasets/"

echo "uploaded objects:"
gsutil ls "gs://${BUCKET}/images/" | head -20
gsutil ls "gs://${BUCKET}/datasets/"
