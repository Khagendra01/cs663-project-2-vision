# CS663 Week 11 Lab — Cloud Run Backend

**Lab:** Lab Android Backend Integration to Tuned & Base Models
**Canvas:** https://csueb.instructure.com/courses/57037/assignments/675845 — due Fri Nov 6, 2026, 11:00pm, 30 points
**Lab guide:** `CS663_Lab_Android_Multi_Model_Cloud_Run_With_Deliverables.pdf` (11 pages, in the goal files folder)
**Status (Oct 27, 2026): backend code is built and locally tested in mock mode. It is NOT deployed yet — see “What is still missing” below.**

## What this is

A Node.js + Express service, following lab guide §3, that the Android app calls over HTTPS:

| Endpoint | Model |
|---|---|
| `POST /api/gemini` (alias: `/api/base-gemini`) | Base Gemini — Gemini Developer API key, or Vertex AI |
| `POST /api/fine-tuned-gemini` | Fine-tuned Gemini — Vertex AI / Agent Platform endpoint via Cloud Run ADC |
| `GET /health` | Status: mock mode, whether base/tuned are configured |
| `GET /` | Browser test page (pick an image + prompt, call both endpoints, compare results) |

Request body (both endpoints): `{ "prompt": "...", "imageBase64": "...", "mimeType": "image/jpeg" }`
Response: `{ "text": "..." }`
`mimeType` must be `image/jpeg`, `image/png`, or `image/webp`. A `data:` URI prefix on the image is stripped automatically. The Android client may send `text` instead of `prompt`.

## Run locally

```sh
cd cloud-run-backend
npm install
cp .env.example .env   # edit .env — never commit a real .env
MOCK_MODE=true npm start
# in another terminal:
curl http://localhost:8080/health
```

Mock mode returns a clearly labelled stub (`mock: true`) so the Android client and the browser test page can be exercised before any real credentials exist. **Mock output is not a lab deliverable and must not be used in the demo video.**

With real credentials (`.env` set, `MOCK_MODE=false`), the same server calls live Gemini. The tuned endpoint will return HTTP 503 until `GOOGLE_CLOUD_PROJECT` and `TUNED_ENDPOINT` are set — that is expected, not a bug.

## Deploy to Cloud Run (template — not run yet)

```sh
PROJECT_ID=your-project TUNED_ENDPOINT=projects/.../endpoints/... ./deploy.sh
```

`deploy.sh` enables Cloud Run / Cloud Build / Artifact Registry / Vertex AI, creates a dedicated runtime service account (`cs663-gemini-runtime`, `roles/aiplatform.user`), and deploys from source with the AI Studio key as a Secret Manager secret. It refuses to run without `PROJECT_ID` and `TUNED_ENDPOINT`, because inventing those would deploy to the wrong project or a nonexistent model.

## What is still missing for a real deployment

1. **Your group Google Cloud project** — project ID, billing/educational credits enabled, and who owns it. No project ID is on record yet.
2. **A Gemini Developer API key** (Google AI Studio) for the base model, to be stored in Secret Manager — not in chat, not in source. Alternatively, base Gemini can go through Vertex AI + ADC instead, which removes this key.
3. **The tuned-model endpoint** — the full `projects/.../locations/.../endpoints/...` ID from the Oct 30 Gemini fine-tuning exercise. No training has launched yet (blocked on your review of the real v3 records and on your Google Cloud account), so this endpoint does not exist yet.
4. **Access-control decision for a public URL** — the lab guide warns against deploying this as an unrestricted public paid-model proxy. Options it names: Firebase Authentication ID tokens validated by the backend (+ Firebase App Check), or a private Cloud Run service with an authorized caller. This is a group decision; the code deliberately does not assume one.
5. **The submission itself** — the deployed service URL (deliverable 1) and the YouTube demo video showing the browser test page plus the Android app calling both endpoints (deliverable 2, next step after this backend).

## Files

- `index.js` — the service (adapted from lab guide §3.3, with the `/api/gemini` rename from §8, the alias, data-URI handling, and mock mode)
- `public/index.html` — browser test page
- `package.json`, `.env.example`, `.gitignore`, `deploy.sh`
