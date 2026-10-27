#!/usr/bin/env python3
"""Assemble the v2 expanded dataset (photos + video frames) into real-data-v2/.

- Prunes round-2 sitting drops first (sheet verify-sitting2-1.jpg tiles 2,3,4,6,8,9,10).
- Sources: video ids (vid_*) and photo uploaders (ph_*). A source goes to ONE
  split globally, so test sources are fully held out.
- Split assignment is greedy per class (60% train / 15% val / 25% test targets),
  processing classes thinnest-first (sitting, standing, walking, lying down);
  a source locked by an earlier class keeps its split for later classes.
- Copies files to real-data-v2/<source>/<behaviour>/ with unique names
  <source>_<behaviour>_<nnn>.jpg, then runs build_jsonl.py with explicit
  --test-dogs / --val-dogs lists. Writes real-data-v2/provenance.json.
"""
import json
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGING = HERE / "staging"
OUT = HERE / "real-data-v2"
BEHAVIOUR_DIR = {"sitting": "sitting", "standing": "standing",
                 "lying_down": "lying_down", "walking": "walking"}

# --- 1. prune round-2 sitting drops (tiles on verify-sitting2-1.jpg) ---
s2 = json.load(open(STAGING / "sheets-sitting2.json"))["verify-sitting2-1.jpg"]
s2_manifest = json.load(open(STAGING / "manifest-sitting2.json"))
s2_by_file = {r["file"]: r for r in s2_manifest}
DROPS_S2 = {2, 3, 4, 6, 8, 9, 10, 16}
s2_kept = []
for i, f in enumerate(s2, 1):
    if i in DROPS_S2:
        (STAGING / f).unlink(missing_ok=True)
    else:
        s2_kept.append(s2_by_file[f])
print(f"sitting2 kept: {len(s2_kept)} (dropped {len(DROPS_S2)})")

# --- 2. gather photo items ---
photos = []
photos += json.load(open(STAGING / "manifest-kept.json"))
photos += s2_kept
photos += json.load(open(STAGING / "manifest-retry.json"))
print(f"photos total: {len(photos)}")


def slug(s):
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s or "unknown"


photo_items = []
for r in photos:
    photo_items.append({
        "src": STAGING / r["file"],
        "source": "ph_" + slug(r["uploader"]),
        "behaviour": BEHAVIOUR_DIR[r["class"]],
        "label": r["class"].replace("_", " "),
        "title": r["title"], "url": r["url"],
        "license": r["license"], "artist": r["artist"],
        "uploader": r["uploader"], "kind": "photo",
    })

# --- 3. gather video items (mission_test merges into mission) ---
VIDEO_MAP = {
    "africanis": ("vid_africanis", {"standing": "standing"}),
    "batumi": ("vid_batumi", {"lying_down": "lying_down"}),
    "mission_train": ("vid_mission", {"walking": "walking"}),
    "mission_test": ("vid_mission", {"walking": "walking"}),
    "panting": ("vid_panting", {"lying_down": "lying_down"}),
    "snoring": ("vid_snoring", {"lying_down": "lying_down"}),
    "wagging": ("vid_wagging", {"standing": "standing"}),
}
video_items = []
for folder, (source, behaviours) in VIDEO_MAP.items():
    d = HERE / "real-data" / folder
    for beh_dir in sorted(d.iterdir()):
        if not beh_dir.is_dir():
            continue
        for img in sorted(beh_dir.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            video_items.append({
                "src": img, "source": source,
                "behaviour": beh_dir.name,
                "label": beh_dir.name.replace("_", " "),
                "title": f"Wikimedia Commons video ({folder})", "url": "",
                "license": "see real-dataset.md v1", "artist": "",
                "uploader": folder, "kind": "video",
            })
print(f"video frames total: {len(video_items)}")

items = photo_items + video_items
print(f"TOTAL POOL: {len(items)}")

# --- 4. split assignment: greedy per class, thinnest class first ---
by_source = defaultdict(list)
for it in items:
    by_source[it["source"]].append(it)
by_class_sources = defaultdict(set)
for it in items:
    by_class_sources[it["label"]].add(it["source"])
class_totals = {c: sum(len(by_source[s]) for s in ss)
                for c, ss in by_class_sources.items()}
print("class totals:", dict(class_totals))

split_of = {}
TARGET = {"train": 0.60, "validation": 0.15, "test": 0.25}
for cls in sorted(class_totals, key=lambda c: class_totals[c]):
    counts = Counter()
    for s in by_class_sources[cls]:
        if s in split_of:
            counts[split_of[s]] += len([i for i in by_source[s] if i["label"] == cls])
    unlocked = sorted((s for s in by_class_sources[cls] if s not in split_of),
                      key=lambda s: -len([i for i in by_source[s] if i["label"] == cls]))
    for s in unlocked:
        n = len([i for i in by_source[s] if i["label"] == cls])
        # pick split with the largest deficit vs its target share of this class
        def deficit(sp):
            return TARGET[sp] * class_totals[cls] - counts[sp]
        best = max(("train", "validation", "test"), key=deficit)
        split_of[s] = best
        counts[best] += n
    print(f"{cls}: split counts {dict(counts)}")

# --- 5. copy files into real-data-v2 ---
if OUT.exists():
    shutil.rmtree(OUT)
counters = Counter()
for it in sorted(items, key=lambda x: (x["source"], x["behaviour"], str(x["src"]))):
    counters[(it["source"], it["behaviour"])] += 1
    n = counters[(it["source"], it["behaviour"])]
    dest_dir = OUT / it["source"] / it["behaviour"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{it['source']}_{it['behaviour']}_{n:03d}.jpg"
    shutil.copyfile(it["src"], dest)
    it["dest"] = str(dest.relative_to(HERE))
    it["split"] = split_of[it["source"]]

prov = [{k: it[k] for k in ("dest", "source", "behaviour", "label", "split",
                            "kind", "title", "url", "license", "artist", "uploader")}
        for it in items]
json.dump(prov, open(OUT / "provenance.json", "w"), indent=1)
test_sources = sorted(s for s, sp in split_of.items() if sp == "test")
val_sources = sorted(s for s, sp in split_of.items() if sp == "validation")
print("test sources:", test_sources)
print("val sources:", val_sources)

# --- 6. build JSONL ---
r = subprocess.run([sys.executable, str(HERE / "build_jsonl.py"),
                    "--data", str(OUT), "--bucket", "YOUR_BUCKET",
                    "--out", str(HERE / "real-jsonl-v2"),
                    "--test-dogs", *test_sources,
                    "--val-dogs", *val_sources,
                    "--labels", "sitting", "standing", "lying down", "walking"],
                   capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print(r.stderr, file=sys.stderr)
    raise SystemExit(r.returncode)
for split in ("train", "validation", "test"):
    lines = open(HERE / "real-jsonl-v2" / f"{split}.jsonl").read().strip().split("\n")
    per_class = Counter(json.loads(ln)["contents"][1]["parts"][0]["text"] for ln in lines)
    print(f"{split}: {len(lines)} {dict(per_class)}")
