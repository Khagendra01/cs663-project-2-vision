#!/usr/bin/env python3
"""Assemble v3: v2 items (from real-data-v2 provenance) + round-3 kept photos.

Fresh greedy split per class (~60/15/25 targets), thinnest class first,
one split per source globally. Outputs real-data-v3/, real-jsonl-v3/,
real-data-v3/provenance.json. v2 files are left untouched.
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
OUT = HERE / "real-data-v3"
BEHAVIOUR_DIR = {"sitting": "sitting", "standing": "standing",
                 "lying_down": "lying_down", "walking": "walking"}


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower() or "unknown"


items = []
# v2 items: copy from their v2 dest paths
for r in json.load(open(HERE / "real-data-v2" / "provenance.json")):
    items.append({
        "src": HERE / r["dest"], "source": r["source"],
        "behaviour": r["behaviour"], "label": r["label"],
        "title": r["title"], "url": r["url"], "license": r["license"],
        "artist": r["artist"], "uploader": r["uploader"], "kind": r["kind"],
    })
# round-3 keeps
for r in json.load(open(STAGING / "manifest-round3-kept.json")):
    items.append({
        "src": STAGING / r["file"], "source": "ph_" + slug(r["uploader"]),
        "behaviour": BEHAVIOUR_DIR[r["class"]],
        "label": r["class"].replace("_", " "),
        "title": r["title"], "url": r["url"], "license": r["license"],
        "artist": r["artist"], "uploader": r["uploader"], "kind": "photo",
    })
print(f"TOTAL POOL: {len(items)}")

by_source = defaultdict(list)
for it in items:
    by_source[it["source"]].append(it)
by_class_sources = defaultdict(set)
for it in items:
    by_class_sources[it["label"]].add(it["source"])
class_totals = {c: sum(1 for it in items if it["label"] == c)
                for c in by_class_sources}
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
        best = max(("train", "validation", "test"),
                   key=lambda sp: TARGET[sp] * class_totals[cls] - counts[sp])
        split_of[s] = best
        counts[best] += n
    print(f"{cls}: split counts {dict(counts)}")

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

r = subprocess.run([sys.executable, str(HERE / "build_jsonl.py"),
                    "--data", str(OUT), "--bucket", "YOUR_BUCKET",
                    "--out", str(HERE / "real-jsonl-v3"),
                    "--test-dogs", *test_sources,
                    "--val-dogs", *val_sources,
                    "--labels", "sitting", "standing", "lying down", "walking"],
                   capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print(r.stderr, file=sys.stderr)
    raise SystemExit(r.returncode)
for split in ("train", "validation", "test"):
    lines = open(HERE / "real-jsonl-v3" / f"{split}.jsonl").read().strip().split("\n")
    per_class = Counter(json.loads(ln)["contents"][1]["parts"][0]["text"] for ln in lines)
    print(f"{split}: {len(lines)} {dict(per_class)}")
