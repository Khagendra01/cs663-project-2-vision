#!/usr/bin/env python3
"""Run the tuned and base Gemini models over test.jsonl and save predictions.

Run this in Google Colab (adapted from the course fine-tuning guide):
  1. %pip -q install -U google-genai
  2. Authenticate to the same Google Cloud project as the tuned model and
     make sure the notebook identity has Vertex AI prediction permission.
  3. Fill in PROJECT_ID, LOCATION, TUNED_MODEL, BASE_MODEL below.

Input : test.jsonl — one JSON object per line, same shape as train.jsonl
        (user message with image fileData + text, model message = reference).
Output: predictions.jsonl — one JSON object per line:
        {"expected": <label>, "tuned": <label>, "base": <label>}
        Only the user parts are sent to the models; the expected answer is
        never part of the model input.
Then score with evaluate.py.
"""

import json

from google.colab import auth
from google import genai
from google.genai import types

PROJECT_ID = "YOUR_PROJECT_ID"
LOCATION = "us-central1"  # must match the tuned model's region
TUNED_MODEL = "YOUR_TUNED_MODEL_ID_OR_RESOURCE_NAME"
BASE_MODEL = "gemini-3.5-flash"

TEST_FILE = "test.jsonl"
OUT_FILE = "predictions.jsonl"


def ask(client, model, image_uri, mime_type, question):
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_uri(file_uri=image_uri, mime_type=mime_type),
            types.Part.from_text(text=question),
        ],
    )
    return (response.text or "").strip()


def main():
    auth.authenticate_user()
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    with open(TEST_FILE, encoding="utf-8") as f_in, \
            open(OUT_FILE, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            user_parts = record["contents"][0]["parts"]
            image = next(p["fileData"] for p in user_parts if "fileData" in p)
            question = next(p["text"] for p in user_parts if "text" in p)
            expected = record["contents"][1]["parts"][0]["text"].strip()
            tuned = ask(client, TUNED_MODEL, image["fileUri"],
                        image["mimeType"], question)
            base = ask(client, BASE_MODEL, image["fileUri"],
                       image["mimeType"], question)
            f_out.write(json.dumps(
                {"expected": expected, "tuned": tuned, "base": base}
            ) + "\n")
    print(f"wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
