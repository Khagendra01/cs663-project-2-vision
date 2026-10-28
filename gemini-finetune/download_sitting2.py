#!/usr/bin/env python3
"""Round-2 sitting candidates: download, resize, manifest (same scheme as round 1)."""
import json
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "CourseProjectDataset/1.0 (CS663)"}
ROOT = Path(__file__).resolve().parent / "staging"
OUT = ROOT / "sitting2"
OUT.mkdir(parents=True, exist_ok=True)

TITLES = [
    "A dog sitting in a field (AM 75047-1).jpg",
    "Dog with pipe in mouth.jpg",
    "A dog sits on the island of Saint Martin, Bangladesh.jpg",
    "A dog sits on the island of Saint Martin, Cox's Bazar, Bangladesh 2.jpg",
    "Australian Cattle Dog Lab mix at Old Fisherman's Wharf.jpg",
    "Come and sit by me at the dog park.jpg",
    "Cute dog sit in the car on the front seat (50012106247).jpg",
    "Dog Bad Sit Slipping With Smiley Ball (43588614260).jpg",
    "Dog Sit.jpg",
    "Labrador Retriever chocolate Hershey sit (cropped).jpg",
    "Phil teaching Bentley to sit (50696963516).jpg",
    "A Beagle sitting in a car looking outside the window.jpg",
    "A Beagle sitting on the sofa looking up.jpg",
    "Beagle sitting on a balcony.jpg",
    "11.10.2015 Samoyed (cropped).jpg",
    "Dog smiling at the camera - Reta, Buenos Aires, Argentina (04-01-2024).jpg",
    "Karjalankarhukoira, Karelian Bear Dog looking at camera.jpg",
    "StJohnWaterDogPuppy.jpg",
    "Girl and dog sit on bench near bridge (38808349365).jpg",
    "Dog in front seat.jpg",
    "Dog on the back seat of HiPhi X 2021022002.jpg",
]


def api(params):
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def main():
    info = {}
    for i in range(0, len(TITLES), 40):
        d = api({
            "action": "query", "format": "json",
            "titles": "|".join("File:" + t for t in TITLES[i:i + 40]),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|user|extmetadata",
            "iiextmetadatafilter": "LicenseShortName|Artist",
        })
        for p in d["query"]["pages"].values():
            if "missing" in p:
                print(f"MISSING: {p.get('title')}")
                continue
            ii = (p.get("imageinfo") or [{}])[0]
            info[p["title"][5:]] = ii
    manifest = []
    for n, title in enumerate(TITLES, 1):
        ii = info.get(title)
        if not ii:
            continue
        mime, size = ii.get("mime", ""), ii.get("size", 0)
        if not mime.startswith("image/") or mime.endswith("svg+xml") or mime.endswith("tiff") or size > 15_000_000:
            print(f"SKIP: {title}")
            continue
        out = OUT / f"sitting2_{n:03d}.jpg"
        if out.exists() and out.stat().st_size > 0:
            print(f"SKIP (done): {out.name}")
        else:
            raw = ROOT / f"raw_s2_{n:03d}"
            req = urllib.request.Request(ii["url"], headers=UA)
            try:
                with urllib.request.urlopen(req, timeout=120) as resp, open(raw, "wb") as f:
                    f.write(resp.read())
            except Exception as e:
                print(f"DOWNLOAD FAIL {title}: {e}")
                continue
            r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw),
                                "-vf", "scale=640:-1", "-q:v", "4", str(out)],
                               capture_output=True, text=True)
            raw.unlink(missing_ok=True)
            if r.returncode != 0 or not out.exists():
                print(f"RESIZE FAIL {title}")
                continue
        em = ii.get("extmetadata", {})
        manifest.append({
            "file": f"sitting2/{out.name}", "class": "sitting", "title": title,
            "url": ii.get("url", ""),
            "license": em.get("LicenseShortName", {}).get("value", "?"),
            "artist": re.sub("<[^>]+>", "", em.get("Artist", {}).get("value", "?"))[:80],
            "uploader": ii.get("user", "?"),
        })
        print(f"OK {out.name} <- {title}")
    json.dump(manifest, open(ROOT / "manifest-sitting2.json", "w"), indent=1)
    print("total:", len(manifest))


if __name__ == "__main__":
    main()
