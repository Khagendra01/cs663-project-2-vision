#!/usr/bin/env python3
"""Download Commons photo candidates, resize to 640w JPEG, write manifest.

Usage: python3 download_candidates.py
Reads CANDIDATES below, queries Commons imageinfo+extmetadata in batches,
downloads originals, resizes with ffmpeg, writes staging/<class>/NNN.jpg
and staging/manifest.json (file, class, title, url, license, artist, uploader).
Skips SVG/TIFF and files over 15 MB.
"""
import json
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "CourseProjectDataset/1.0 (CS663)"}
ROOT = Path(__file__).resolve().parent / "staging"

CANDIDATES = {
    "sitting": [
        "3125-boxer-sitting-dog (20313840449).jpg",
        "A Beagle sitting next to dog food.jpg",
        "A dog sitting on a chair, smiling.jpg",
        "A dog sitting outside near a lake.jpg",
        "A long hair dog sitting at floor.jpg",
        "A lovely dog sitting on floor.jpg",
        "Black dog sitting outdoors in autumn.jpg",
        "Cute dog sitting in forest.jpg",
        "Dog Sitting in a car.jpg",
        "Dog in sitting position.jpg",
        "Dog, Sitting, Nafplio.jpg",
        "German Shepherd Dog sitting leash.jpg",
        "Indian Pariah Dog named Scratchy sitting in a pile of tires.jpg",
        "Indian Pariah Dog sitting on a ground.jpg",
        "New Guinea Singing Dog, sitting on the ground.jpg",
        "Old dog sitting outdoors in autumn.jpg",
        "Sitting dog 003.jpg",
        "Street Dog 27 sitting on a car.jpg",
        "Street Dog 28 sitting on a car.jpg",
        "Sitting With The Dog - geograph.org.uk - 3399944.jpg",
        "Sitting Bulakenya with tethered dog in Barangay San Gabriel Malolos 01.jpg",
        "Sitting Bulakenya with tethered dog in Barangay San Gabriel Malolos 03.jpg",
        "A beagle sitting below a chair on the grass-2.jpg",
        "A beagle sitting below a chair on the grass.jpg",
        "Bavaria Dog 2011 001.JPG",
        "Dog sitting on grass (7257609304).jpg",
        "Dog sitting on grass - panoramio (1228).jpg",
        "Red and white dog puppy sitting in grass.jpg",
        "Small brown dog sitting on grass.jpg",
        "Dog in the grass.jpg",
        "Dog Friend (2595637875).jpg",
    ],
    "standing": [
        "A beagle standing in a dog park.jpg",
        "A Dog standing with his paws on a table.jpg",
        "Indian Pariah Dog named Scratchy standing.jpg",
        "Karjalankarhukoira, Karelian Bear Dog standing profile.jpg",
        "Photo of a standing dog.jpg",
        "Standing Bernese Mountain Dog Female (cropped).jpg",
        "Standing dog-MA 6095-IMG 9564-black.jpg",
        "Standing dog-MA 6095-IMG 9566-black.jpg",
        "Beagle puppy standing on a tree stump.jpg",
        "Golden Retriever puppy standing.jpg",
        "Puppy José standing.jpg",
        "White miniature schnauzer puppy standing from front.jpg",
        "Corgi puppy standing funny.jpg",
        "Contre-jour photograph of a standing puppy at sunset with colorful sky in Don Det Laos.jpg",
        "All puppy 44.jpg",
        "Black and white mixed-breed puppy.JPG",
        "Puppy and the Tansies.jpg",
        "Standing Neo.jpg",
        "Dog standing outside on a lawn, probably a terrier, ca. 1905-1910. (15408070774).jpg",
        "Cutting dog hair (1).jpg",
        "Cutting dog hair (2).jpg",
        "Aussie puppy match show.jpg",
        "Puppy in Lipa valley.jpg",
        "Four puppies drinking their mother's milk, standing on a dirt road in Don Khon Laos.jpg",
        "Two puppies playing in the countryside one standing over the other at golden hour in Don Det Laos.jpg",
    ],
    "lying_down": [
        "Dog lying on the Dhaka University Campus 01.jpg",
        "Dog lying on the Dhaka University Campus 02.jpg",
        "Lying dog (8951650545).jpg",
        "Nura lying in grass (DSCF6712).jpg",
        "Playful dog lying on his back (51160180621).jpg",
        "White dog sleeping on the floor with puppies after nursing in Don Det Laos.jpg",
        "Yellow dog lying on bamboo mat.jpg",
        "2014 Dog relaxing on a hot summer day.jpg",
        "43 Old dog - senior labrador dog with collar resting on sidewalk.jpg",
        "Dog Resting, Chalcot Square.jpg",
        "Dog at rest, leashed.jpg",
        "Dog resting along Loughmacrory Road - geograph.org.uk - 7205824.jpg",
        "Dog resting along the roadside, Binnafreaghan - geograph.org.uk - 7765301.jpg",
        "Dog asleep on bed.jpg",
        "Dog asleep on the Moscow Metro.jpg",
        "Stray dog - asleep.jpg",
        "A dog fell asleep in front of Lugu Lake.jpg",
        "Korky, The Pub Dog, Asleep In The Rising Sun, Isleworth - London.jpg",
        "Dog sleeping in a dog bed.JPG",
        "Greyhound roaching 2.jpg",
        "Small dog in doggy bed.jpg",
    ],
    "walking": [
        "Couple walking dog at Buckingham Fountain.jpg",
        "Dog Walking Party in Highgate Wood - geograph.org.uk - 1317936.jpg",
        "Dog Walking at Tringford Reservoir - geograph.org.uk - 1419270.jpg",
        "Dog Walking in Debdale Park - geograph.org.uk - 1159175.jpg",
        "Dog Walking in Parking Area (51203911374).jpg",
        "Dog Walking in Snowy Woolhampton - geograph.org.uk - 333349.jpg",
        "Dog on a Leash, Wassenaarse Slag, Netherlands-DSC02431.jpg",
        "Dog on leash, walking in the Red Rocks Park, Morrison, Colorado 16.jpg",
        "Dog walking at the beach along the Bayview Trail at Mason Neck (14732270891).jpg",
        "Dog walking in Kingsford Forest Park - geograph.org.uk - 886496.jpg",
        "Dog walking in West Park - geograph.org.uk - 998126.jpg",
        "Dog walking on Walden Recreation Ground - geograph.org.uk - 1655814.jpg",
        "Dog walking trail (4479928464).jpg",
        "Dog walking woman.jpg",
        "Dog with owner walking.jpg",
        "Girl Walking Dog in New Albany, Ohio.jpg",
        "Girl walking dog 001.jpg",
        "Girl walking dog 002.jpg",
        "Girl walking dog at Lake Johnson.jpg",
        "Visitor walking dog leash rialto beach hikers c bubar march 06-2015 (17184427098).jpg",
        "Visitor walking happy dog leash hoh campground camping pets d archuleta march 05 2015 (16751947833).jpg",
        "Dog walking in Florence.jpg",
        "Dog-Walker at Twilight Cox Bay Vancouver Island 7189.jpg",
        "Man walking a dog, Jardim da Praça de Londres, Lisbon, Portugal julesvernex2.jpg",
        "Man walking his dog during the COVID-19 lockdown, Rossio, Lisbon, Portugal julesvernex2.jpg",
        "Walking the dog (1945527533).jpg",
        "Walking the dog in winter in Tuntorp, Brastad.jpg",
        "The dog walking by the lake.jpg",
        "Dog Walking on Cache Valley Road (51201775235).jpg",
        "Woman and dog walk on beach at Cape Lookout State Park (Oregon).jpg",
    ],
}


def api(params):
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for cls, titles in CANDIDATES.items():
        outdir = ROOT / cls
        outdir.mkdir(exist_ok=True)
        info = {}
        for i in range(0, len(titles), 40):
            batch = titles[i:i + 40]
            d = api({
                "action": "query", "format": "json",
                "titles": "|".join("File:" + t for t in batch),
                "prop": "imageinfo",
                "iiprop": "url|size|mime|user|extmetadata",
                "iiextmetadatafilter": "LicenseShortName|Artist",
            })
            for p in d["query"]["pages"].values():
                if "missing" in p:
                    print(f"MISSING: {p.get('title')}")
                    continue
                ii = (p.get("imageinfo") or [{}])[0]
                title = p["title"][5:]  # strip "File:"
                info[title] = ii
        for n, title in enumerate(titles, 1):
            ii = info.get(title)
            if not ii:
                continue
            mime = ii.get("mime", "")
            size = ii.get("size", 0)
            if not mime.startswith("image/") or mime.endswith("svg+xml") or mime.endswith("tiff"):
                print(f"SKIP (mime): {title}")
                continue
            if size > 15_000_000:
                print(f"SKIP (size {size}): {title}")
                continue
            raw = ROOT / f"raw_{cls}_{n:03d}"
            out = outdir / f"{cls}_{n:03d}.jpg"
            em = ii.get("extmetadata", {})
            entry = {
                "file": f"{cls}/{out.name}", "class": cls, "title": title,
                "url": ii.get("url", ""),
                "license": em.get("LicenseShortName", {}).get("value", "?"),
                "artist": re.sub("<[^>]+>", "", em.get("Artist", {}).get("value", "?"))[:80],
                "uploader": ii.get("user", "?"),
            }
            if out.exists() and out.stat().st_size > 0:
                manifest.append(entry)
                print(f"SKIP (already done) {cls}/{out.name}")
                continue
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
                print(f"RESIZE FAIL {title}: {r.stderr[-300:]}")
                continue
            manifest.append(entry)
            print(f"OK {cls}/{out.name} <- {title}")
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nTotal downloaded+resized: {len(manifest)}")


if __name__ == "__main__":
    main()
