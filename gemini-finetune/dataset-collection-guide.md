# Dataset Collection Guide — Gemini Fine-Tuning (Project 2, Model B)

## What we are collecting

Phone videos of dogs performing the behaviours in the locked label set, turned into labelled still frames. The tuned model sees ONE image + one question and must answer with ONE label word.

Proposed label set (confirm or change at Phase 0, Tue Oct 27):
sitting, standing, lying down, walking, rolling over

Note on "rolling over": a roll is motion, not a single pose. If it stays in the set, use a clearly mid-roll frame and label it rolling over, or drop the class for this exercise. Do not mix mid-roll frames with lying-down frames — that ambiguity will hurt accuracy more than dropping the class.

## Folder structure (on whoever's machine collects the data)

```
dog-data/
  <dog-name>/
    sitting/
    standing/
    lying_down/
    walking/
    rolling_over/
```

One folder per dog per behaviour. Dog names are the split key: a dog's folders go ENTIRELY into train, validation, or test — never split across them.

## Recording rules

1. Film each behaviour in short clips (5–15 seconds), several clips per dog per behaviour, different angles, distances, and lighting. Variety beats volume.
2. One dog at a time in frame. If another dog or person dominates the frame, discard it.
3. Keep the dog fully or mostly in frame. A head-only crop of a sitting dog is ambiguous — discard it.
4. Avoid identifying info: no house numbers, faces, or names on collars visible in kept frames.
5. Target counts (across the whole group, not per person):
   - train: 100–200 frames total, at least 3 dogs, every class covered by at least 2 dogs
   - validation: 20–30 frames
   - test: 30–50 frames, from dogs that appear NOWHERE in train or validation
6. If the group cannot reach these counts by Oct 28, shrink the label set (drop to 3 classes) rather than padding with duplicates or internet photos of unknown licence.

## From video to labelled frames

1. Run `extract_frames.py` on each clip (see that file): it pulls stills at 1 frame per second into the matching `dog-data/<dog>/<behaviour>/` folder.
2. Human review pass: open each folder and DELETE any frame that is blurry, occluded, ambiguous, or mislabelled. This pass matters more than collecting extra clips.
3. Rename kept frames clearly: `<dog>_<behaviour>_<nnn>.jpg` (e.g. `milo_sitting_014.jpg`).

## From frames to JSONL

1. Upload images to the Cloud Storage bucket first, into `images/`.
2. Build one JSON object per line (see `train-template.jsonl` for the exact shape):
   - user parts: image reference (`fileData`, MIME type, gs:// URI) + the fixed prompt text
   - model parts: exactly one label word
3. The prompt text is IDENTICAL on every record (only the label set words change if Phase 0 changes the classes).
4. Validate before upload: every line parses as JSON, every gs:// URI matches an uploaded object, every label is one of the locked classes, and no test dog appears in train/validation.
