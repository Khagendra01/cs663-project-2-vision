#!/usr/bin/env python3
"""Assemble v4: v3 items (171, from real-data-v3 provenance) + visually
verified v4 keeps (staging-v4/manifest-kept.json).

Target: 250 per class / 1000 total. If a class has more verified keeps
than needed, a diversity-first subset is taken (round-robin across
uploaders). If it has fewer, ALL keeps are used and the shortfall is
printed - never padded with unverified images.

Fresh greedy split per class (~60/15/25 targets), thinnest class first,
one split per source globally (same method as assemble_v3.py).
Outputs real-data-v4/, real-jsonl-v4/, real-data-v4/provenance.json.
v3 files are left untouched.
"""
import json
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGING = HERE / "staging-v4"
OUT = HERE / "real-data-v4"
TARGET_PER_CLASS = 250
BEHAVIOUR_DIR = {"sitting": "sitting", "standing": "standing",
                 "lying_down": "lying_down", "walking": "walking"}


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower() or "unknown"


items = []
for r in json.load(open(HERE / "real-data-v3" / "provenance.json")):
    items.append({
        "src": HERE / r["dest"], "source": r["source"],
        "behaviour": r["behaviour"], "label": r["label"],
        "title": r["title"], "url": r["url"], "license": r["license"],
        "artist": r["artist"], "uploader": r["uploader"], "kind": r["kind"],
    })
v3_counts = Counter(it["label"] for it in items)
print("v3 pool:", dict(v3_counts))

keeps = json.load(open(STAGING / "manifest-kept.json"))
keep_counts = Counter(r["class"].replace("_", " ") for r in keeps)
print("verified v4 keeps:", dict(keep_counts))

# diversity-first selection per class up to the needed count
by_class_keeps = defaultdict(list)
for r in keeps:
    by_class_keeps[r["class"]].append(r)
selected = []
for cls, rows in by_class_keeps.items():
    label = cls.replace("_", " ")
    need = TARGET_PER_CLASS - v3_counts.get(label, 0)
    if len(rows) <= need:
        chosen = rows
        if len(rows) < need:
            print(f"SHORTFALL {label}: need {need} new, only {len(rows)} verified keeps")
    else:
        by_uploader = defaultdict(list)
        for r in sorted(rows, key=lambda r: (r["uploader"], r["file"])):
            by_uploader[r["uploader"]].append(r)
        chosen = []
        uploaders = sorted(by_uploader)
        while len(chosen) < need:
            progressed = False
            for u in uploaders:
                if by_uploader[u] and len(chosen) < need:
                    chosen.append(by_uploader[u].pop(0))
                    progressed = True
            if not progressed:
                break
        print(f"{label}: selected {len(chosen)} of {len(rows)} verified keeps")
    selected.extend(chosen)

for r in selected:
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
                    "--out", str(HERE / "real-jsonl-v4"),
                    "--test-dogs", *test_sources,
                    "--val-dogs", *val_sources,
                    "--labels", "sitting", "standing", "lying down", "walking"],
                   capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print(r.stderr, file=sys.stderr)
    raise SystemExit(r.returncode)
for split in ("train", "validation", "test"):
    lines = open(HERE / "real-jsonl-v4" / f"{split}.jsonl").read().strip().split("\n")
    per_class = Counter(json.loads(ln)["contents"][1]["parts"][0]["text"] for ln in lines)
    print(f"{split}: {len(lines)} {dict(per_class)}")
