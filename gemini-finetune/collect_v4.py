#!/usr/bin/env python3
"""v4 collection loop: hunt Wikimedia Commons until the raw candidate pool
per class is big enough that visual verification can keep 250/class
(1000 total incl. the 171 already-verified v3 items).

Loops category listings + search pages with continuation offsets, in
rounds, until RAW_TARGET per class is reached or every source is
exhausted. Downloads 640px thumbs, dedupes by title (vs v3 provenance
and within pool) and by perceptual average-hash (vs v3 images and
within pool). Output: staging-v4/<class>/cand_NNN.jpg + manifest.json
with title/url/license/artist/uploader for every candidate.

Labels at this stage are CANDIDATE labels only (category/search source).
Nothing here is training data until the contact-sheet verification pass.
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "CourseProjectDataset/1.0 (CS663; contact khagendrakhatri365@gmail.com)"}
HERE = Path(__file__).resolve().parent
OUT = HERE / "staging-v4"
RAW_TARGET = 520          # per class, pre-verification pool
MAX_PER_SOURCE_PAGE = 50

SOURCES = {
    "sitting": {
        "categories": ["Dogs sitting", "Sitting dogs"],
        "searches": ["dog sitting", "puppy sitting", "dog sitting grass",
                     "dog sitting indoors", "dog sits waiting"],
    },
    "standing": {
        "categories": ["Dogs standing", "Standing dogs"],
        "searches": ["dog standing", "puppy standing", "dog standing grass",
                     "dog standing profile", "dogs standing alert"],
    },
    "lying_down": {
        "categories": ["Dogs lying down", "Sleeping dogs", "Dogs sleeping"],
        "searches": ["dog lying down", "dog sleeping", "dog resting",
                     "sleeping dog floor", "dog lying grass"],
    },
    "walking": {
        "categories": ["Dogs walking", "Walking dogs"],
        "searches": ["dog walking", "walking the dog", "dog on leash walking",
                     "dog walking park", "person walking dog"],
    },
}

BLACKLIST = ["illustration", "drawing", "painting", "statue", "sculpture",
             "logo", "diagram", "cartoon", "toy dog", "stuffed", "poster",
             "sign", "cake", "paw print", "silhouette", "vector", "clipart",
             "engraving", "etching", "woodcut", "tapestry", "mosaic"]
CLASS_BLACKLIST = {
    "walking": ["running", "jumping", "swimming", "agility"],
    "sitting": [],
    "standing": [],
    "lying_down": [],
}
FREE_LICENSE = ("CC", "Public domain", "PD")


def api(params, tries=3):
    url = API + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.load(r)
        except Exception as e:
            print(f"  api retry {i+1}: {e}", flush=True)
            time.sleep(2 * (i + 1))
    return {}


def ahash(img):
    g = img.convert("L").resize((8, 8))
    px = list(g.getdata())
    avg = sum(px) / 64
    bits = 0
    for i, p in enumerate(px):
        if p > avg:
            bits |= 1 << i
    return bits


def hamming(a, b):
    return bin(a ^ b).count("1")


def load_v3():
    prov = json.load(open(HERE / "real-data-v3" / "provenance.json"))
    titles = {r["title"].lower() for r in prov}
    hashes = set()
    for p in (HERE / "real-data-v3").rglob("*.jpg"):
        try:
            hashes.add(ahash(Image.open(p)))
        except Exception:
            pass
    return titles, hashes


def fetch_imageinfo(titles):
    out = {}
    for i in range(0, len(titles), 40):
        batch = titles[i:i + 40]
        d = api({"action": "query", "format": "json",
                 "titles": "|".join("File:" + t for t in batch),
                 "prop": "imageinfo",
                 "iiprop": "url|size|mime|user|extmetadata",
                 "iiurlwidth": 640,
                 "iiextmetadatafilter": "LicenseShortName|Artist"})
        for p in d.get("query", {}).get("pages", {}).values():
            if "missing" in p or not p.get("imageinfo"):
                continue
            out[p["title"][5:]] = p["imageinfo"][0]
    return out


def category_titles(cat, cont):
    d = api({"action": "query", "format": "json", "list": "categorymembers",
             "cmtitle": "Category:" + cat, "cmtype": "file",
             "cmlimit": 500, **({"cmcontinue": cont} if cont else {})})
    titles = [m["title"][5:] for m in d.get("query", {}).get("categorymembers", [])
              if m["title"].startswith("File:")]
    nxt = d.get("continue", {}).get("cmcontinue")
    return titles, nxt


def search_titles(query, offset):
    d = api({"action": "query", "format": "json", "generator": "search",
             "gsrsearch": query + " filetype:bitmap",
             "gsrnamespace": 6, "gsrlimit": MAX_PER_SOURCE_PAGE,
             "gsroffset": offset, "prop": "imageinfo",
             "iiprop": "url|size|mime|user|extmetadata", "iiurlwidth": 640,
             "iiextmetadatafilter": "LicenseShortName|Artist"})
    pages = d.get("query", {}).get("pages", {})
    items = {}
    for p in pages.values():
        if p.get("imageinfo"):
            items[p["title"][5:]] = p["imageinfo"][0]
    nxt = d.get("continue", {}).get("gsroffset")
    return items, nxt


def ok_candidate(cls, title, ii):
    t = title.lower()
    if any(b in t for b in BLACKLIST + CLASS_BLACKLIST.get(cls, [])):
        return False
    if ii.get("mime") != "image/jpeg":
        return False
    if ii.get("width", 0) < 500 or ii.get("height", 0) < 320:
        return False
    lic = ii.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "")
    if not lic.startswith(FREE_LICENSE):
        return False
    return bool(ii.get("thumburl"))


def download(cls, n, title, ii, seen_hashes):
    req = urllib.request.Request(ii["thumburl"], headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            data = r.read()
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        print(f"  DL FAIL {title}: {e}", flush=True)
        return None
    if img.width < 400:
        return None
    h = ahash(img)
    for old in seen_hashes:
        if hamming(h, old) <= 2:
            return None
    if img.width != 640:
        img = img.resize((640, round(img.height * 640 / img.width)))
    outdir = OUT / cls
    outdir.mkdir(parents=True, exist_ok=True)
    name = f"cand_{n:04d}.jpg"
    img.save(outdir / name, quality=87)
    em = ii.get("extmetadata", {})
    seen_hashes.add(h)
    return {"file": f"{cls}/{name}", "class": cls, "title": title,
            "url": ii.get("url", ""), "license": em.get("LicenseShortName", {}).get("value", "?"),
            "artist": re.sub("<[^>]+>", "", em.get("Artist", {}).get("value", "?"))[:80],
            "uploader": ii.get("user", "?")}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v3_titles, v3_hashes = load_v3()
    manifest_path = OUT / "manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else []
    seen_titles = {m["title"].lower() for m in manifest} | v3_titles
    seen_hashes = set(v3_hashes)
    for m in manifest:  # rehash existing downloads so resumes dedupe too
        try:
            seen_hashes.add(ahash(Image.open(OUT / m["file"])))
        except Exception:
            pass
    counts = {c: sum(1 for m in manifest if m["class"] == c) for c in SOURCES}
    print(f"start counts: {counts}; v3 titles excluded: {len(v3_titles)}", flush=True)

    # per-source cursors
    cursors = {}
    for cls, src in SOURCES.items():
        for cat in src["categories"]:
            cursors[(cls, "cat", cat)] = None
        for q in src["searches"]:
            cursors[(cls, "search", q)] = 0
    exhausted = set()

    rnd = 0
    while any(counts[c] < RAW_TARGET for c in SOURCES) and len(exhausted) < len(cursors):
        rnd += 1
        print(f"--- round {rnd}: {counts} ---", flush=True)
        for (cls, kind, key), cur in list(cursors.items()):
            if counts[cls] >= RAW_TARGET or (cls, kind, key) in exhausted:
                continue
            if kind == "cat":
                titles, nxt = category_titles(key, cur)
                if not titles:
                    exhausted.add((cls, kind, key)); continue
                info = fetch_imageinfo(titles)
                pairs = [(t, info[t]) for t in titles if t in info]
                cursors[(cls, kind, key)] = nxt
                if nxt is None:
                    exhausted.add((cls, kind, key))
            else:
                info, nxt = search_titles(key, cur)
                pairs = list(info.items())
                cursors[(cls, kind, key)] = nxt
                if nxt is None:
                    exhausted.add((cls, kind, key))
            for title, ii in pairs:
                if counts[cls] >= RAW_TARGET:
                    break
                if title.lower() in seen_titles or not ok_candidate(cls, title, ii):
                    continue
                seen_titles.add(title.lower())
                entry = download(cls, counts[cls] + 1, title, ii, seen_hashes)
                if entry:
                    manifest.append(entry)
                    counts[cls] += 1
            manifest_path.write_text(json.dumps(manifest, indent=1))
            print(f"  {cls}/{kind}:{key} -> pool {counts[cls]}", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"DONE after {rnd} rounds. Final raw pool: {counts} total={sum(counts.values())}", flush=True)


if __name__ == "__main__":
    main()
