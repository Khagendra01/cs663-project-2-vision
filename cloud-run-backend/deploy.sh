#!/bin/sh
# Deploy the CS663 Cloud Run backend.
# Defaults are the group's real project and tuned endpoint from the Oct 30 exercise.
# Base Gemini uses Vertex AI + ADC (no AI Studio key / secret needed).
# Usage: ./deploy.sh   (override with PROJECT_ID=... TUNED_ENDPOINT=... if needed)
set -eu
PROJECT_ID="${PROJECT_ID:-cs663-dog-training}"
REGION="${REGION:-us-central1}"
TUNED_ENDPOINT="${TUNED_ENDPOINT:-projects/917815040923/locations/us/endpoints/514008491845746688}"
TUNED_VERTEX_LOCATION="${TUNED_VERTEX_LOCATION:-us}"
BASE_VERTEX_LOCATION="${BASE_VERTEX_LOCATION:-global}"
BASE_MODEL="${BASE_MODEL:-gemini-3-flash-preview}"

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com aiplatform.googleapis.com
gcloud iam service-accounts create cs663-gemini-runtime 2>/dev/null || true
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:cs663-gemini-runtime@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
gcloud run deploy cs663-gemini-service \
  --source . --region "$REGION" \
  --allow-unauthenticated \
  --service-account "cs663-gemini-runtime@${PROJECT_ID}.iam.gserviceaccount.com" \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID},BASE_VERTEX_LOCATION=${BASE_VERTEX_LOCATION},TUNED_VERTEX_LOCATION=${TUNED_VERTEX_LOCATION},BASE_MODEL=${BASE_MODEL},TUNED_ENDPOINT=${TUNED_ENDPOINT}"
echo "Deployed. The service URL printed above is lab deliverable 1."
echo "Note: --allow-unauthenticated is the group's lab choice so the browser test page and Android app can call it without Google sign-in. The Gemini credentials themselves stay server side in the service account. Project credits are limited, which bounds exposure."
