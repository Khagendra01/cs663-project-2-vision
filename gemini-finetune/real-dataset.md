# Real Dataset — Narrow Case (3 classes, Wikimedia Commons footage)

Built Oct 26–27, 2026. Every frame was visually verified before labelling.
All source videos are openly licensed on Wikimedia Commons; attribution below.

## Classes and splits (split BY VIDEO — no video appears in two splits)

| Split | Video (source) | Behaviour | Frames |
|---|---|---|---|
| train | Africanis Dogs Resting In Grass | standing | 8 |
| train | Batumi2025dog-laying-relax (from t=2s, empty opening frames dropped) | lying down | 6 |
| train | Dog walking in the Mission in San Francisco (first 12s) | walking | 6 |
| validation | Irish Wolfhound mix, panting | lying down | 5 |
| test | Dog wagging tail | standing | 4 |
| test | A domestic dog snoring (first 30s, single-dog portion) | lying down | 6 |
| test | Dog walking in the Mission in San Francisco (last 5s) | walking | 2 |

Totals: **train 20, validation 5, test 12 = 37 frames.**
Per-class totals: standing 12, lying down 17, walking 8.

## Files

- Frames: `real-data/<video>/<behaviour>/<video>_<behaviour>_NNN.jpg` (640px wide, JPEG)
- JSONL: `real-jsonl/train.jsonl`, `real-jsonl/validation.jsonl`, `real-jsonl/test.jsonl`
  (prompt: "Look at the image of the dog. What is the dog doing? Answer with exactly one word: standing, lying down, walking.")
- Build command: `python3 build_jsonl.py --data real-data --bucket YOUR_BUCKET --out real-jsonl --labels standing "lying down" walking --test-dogs wagging snoring mission_test --val-dogs panting`

## Sources and licences (attribution for the report)

- Africanis Dogs Resting In Grass — CC0 — https://commons.wikimedia.org/wiki/File:Africanis_Dogs_Resting_In_Grass.ogv
- Batumi2025dog-laying-relax — CC BY 4.0, Vitaly Zdanevich — https://commons.wikimedia.org/wiki/File:Batumi2025dog-laying-relax.webm
- Dog walking in the Mission in San Francisco — CC BY-SA 3.0, Sage Ross (Ragesoss) — https://commons.wikimedia.org/wiki/File:Dog_walking_in_the_Mission_in_San_Francisco.ogv
- Dog wagging tail — CC BY-SA 4.0, Liannadavis — https://commons.wikimedia.org/wiki/File:Dog_wagging_tail.webm
- A domestic dog snoring — CC BY-SA 4.0, Serial Number 54129 — https://commons.wikimedia.org/wiki/File:A_domestic_dog_snoring.ogv
- Irish Wolfhound mix, panting — CC BY-SA 3.0, Sage Ross (Ragesoss) — https://commons.wikimedia.org/wiki/File:Irish_Wolfhound_mix,_panting.ogv

## Disclosed limitations (state these in the report, do not hide them)

1. **Walking test is a same-video holdout.** Only one clean walking video was found, so the 2 walking test frames come from the last 5 seconds of the same Mission video used for walking training. Standing and lying-down test frames are fully held-out videos. Before claiming generalisation, add a second held-out walking video.
2. **This is openly-licensed stock footage, not the group's own dogs.** The exercise asks for the group's custom Project 2 dataset; this dataset was built at Khagendra's direction as the real-data narrow case. The group can swap in their own dogs later using the same pipeline (`extract_frames.py` + `build_jsonl.py`) without changing anything else.
3. **Small pilot.** 37 frames is a pilot, not a production dataset. Near-duplicate consecutive frames were reduced by sampling every 2s (train) / 5–10s (test/validation), but frames from one video still resemble each other.
4. **Rejected during verification (do not reuse without re-checking):** Tokyo pro-dog-walker (pack of dogs), Boxer 1901 footage (blurry, dog small), Buster and the Dude (dog tiny in frame), Hondenwandeltocht and St. Bernardshonden (crowds, dogs distant), plus several long home movies and irrelevant search hits.

---

# Expanded Dataset v2 — 4 classes, 110 examples (built Oct 27, 2026; SUPERSEDES v1)

At Khagendra's direction ("is this dataset enough… if not, hunt for as many as required NOW"),
v1's 37 frames were expanded the same morning. **Sitting is back as a 4th class** — Commons
photos (not video) supplied clean sitting examples, so the class set is now
**sitting, standing, lying down, walking**.

## Classes and splits (split BY SOURCE — video id or photo uploader; no source in two splits)

| Split | Composition | sitting | standing | walking | lying down | Total |
|---|---|---|---|---|---|---|
| train | 37 photos + 33 video frames | 15 | 16 | 16 | 23 | **70** |
| validation | 14 photos | 3 | 3 | 4 | 4 | **14** |
| test | 22 photos + 4 video frames (wagging) | 6 | 6 | 6 | 8 | **26** |
| **Total** | 73 photos + 37 video frames | **24** | **25** | **26** | **35** | **110** |

Per-class totals: sitting 24, standing 25, walking 26, lying down 35.
Photo sources: 58 distinct Wikimedia Commons uploaders; video sources: the same 7 v1
videos, with `mission_test` merged into `vid_mission` (train) — see fix below.

## Files

- Frames/photos: `real-data-v2/<source>/<behaviour>/<source>_<behaviour>_NNN.jpg` (640px JPEG)
- JSONL: `real-jsonl-v2/train.jsonl`, `real-jsonl-v2/validation.jsonl`, `real-jsonl-v2/test.jsonl`
  (prompt fixed to: "Look at the image of the dog. What is the dog doing? Answer with exactly one of: sitting, standing, lying down, walking.")
- Provenance (every item's source, title, URL, licence, artist, uploader, split): `real-data-v2/provenance.json`
- Build: `python3 assemble_v2.py` (verifies photos, assigns splits greedily per class at
  ~60/15/25, then runs `build_jsonl.py` with explicit `--test-dogs` / `--val-dogs` source lists)
- v1 files (`real-data/`, `real-jsonl/`) are preserved untouched for comparison; v2 is the
  dataset to upload and train on.

## Verification honesty (read before claiming quality)

- Every photo was viewed on a contact sheet before keeping: **39 of 98 round-1 photos and
  8 of 17 round-2 sitting photos were dropped** (wrong behaviour, statues/paintings, dogs too
  distant, ambiguous posture). Search-result titles lie; the keeps are the verified subset.
- One lying candidate ("Dog at rest, leashed") was actually a sitting dog and was reassigned
  to sitting, not dropped.
- 4 photos that ffmpeg could not resize (CMYK JPEGs) were converted with PIL instead and
  verified separately — all 5 retries kept (4 sitting, 1 walking).

## What v2 fixes vs v1, and what it does not

- **FIXED: the same-video walking test.** Walking test is now 6 held-out photos from
  uploaders with nothing in train. The 2 old Mission test frames moved to train.
- **FIXED (partly): size.** 110 examples (70 train) vs 37 (20 train) — a defensible course
  fine-tune, not a pilot. The guide asks for "multiple training examples per scenario, with
  accurate outputs and varied images"; v2 meets that bar.
- **STILL TRUE: stock footage, not the group's own dogs.** Same disclosure as v1 caveat 2;
  the group can swap in their own dogs with the same scripts.
- **STILL THINNEST: sitting (24 total, 15 train).** If the group wants a stronger sitting
  class, the next source is the Pexels golden-retriever sitting clip #7515916 (needs a browser
  download) or the group's own sitting photos.
- **Test is photo-heavy for lying/walking:** the only video frames in test are the 4 wagging
  (standing) frames; lying/walking test examples are all photos. Test sources are still fully
  held out, so the accuracy claim stands — but note the modality mix in the report.

---

# Expanded Dataset v3 — 171 examples (built Oct 27, 2026; SUPERSEDES v2)

At Khagendra's "Get more", a third Commons round (fresh queries + the Sitting/Sleeping/
Standing/Walking category lists) added 88 candidates; after contact-sheet verification
**61 kept** (27 dropped: wrong behaviour, distant dogs, running mistaken for walking,
person-dominant shots). Merged with v2's 110 and re-split fresh.

| Split | sitting | standing | walking | lying down | Total |
|---|---|---|---|---|---|
| train | 21 | 25 | 22 | 34 | **102** |
| validation | 5 | 6 | 6 | 9 | **26** |
| test | 9 | 11 | 9 | 14 | **43** |
| **Total** | **35** | **42** | **37** | **57** | **171** |

Files: `real-data-v3/<source>/<behaviour>/`, `real-jsonl-v3/*.jsonl`,
`real-data-v3/provenance.json`. Build: `python3 assemble_v3.py`. v2 files preserved untouched.
Note: the fresh greedy split moved `vid_wagging` to train, so v3's test set is all photos —
still fully held-out sources, but state the modality mix in the report. Previous caveats
(stock footage, sitting thinnest) still apply; sitting is now 35, no longer the outlier it was.
