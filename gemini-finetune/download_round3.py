#!/usr/bin/env python3
"""Round-3 photo candidates: download via Commons API, convert with PIL, manifest."""
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from PIL import Image

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "CourseProjectDataset/1.0 (CS663)"}
ROOT = Path(__file__).resolve().parent / "staging"
OUT = ROOT / "round3"
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATES = {
    "sitting": [
        "Beagle-Shepard Mixed-breed dog - Sitting.png",
        "My Dog, Towie 02.jpg",
        "My Dog, Towie 05.jpg",
        "Photo Op. of Rex The Dog Sitting on a Couch in The White House - DPLA - 2c64eca2ad2d768cd38c0d3816808cb1.jpg",
        "A beagle sitting on the bed.jpg",
        "Beagle sitting on a bed.jpg",
        "A dog sitting at a river.jpeg",
        "Papillon Dog Sitting.jpeg",
        "Ruby-GSD.jpg",
        "A Beagle sitting in a car.jpg",
        "A Beagle sitting in the car 01.jpg",
        "A dog named Peanut, sitting in a car.jpg",
        "Dog Sitting In A Car (50011837761).jpg",
        "Dog in car sitting alone (50011308463).jpg",
        "Yellow labrador retriever sitting in the back of a car.jpg",
        "A Beagle sitting on a sofa 3.jpg",
        "A beagle sitting on a sofa.jpg",
        "Beagle Puppy sitting on the sofa.jpg",
        "Beagle sitting on the sofa with toy.jpg",
        "Beagle sitting on the sofa.jpg",
        "Dog Sitting in a car.jpg",
        "Catahoula Leopard Dog Bones.jpg",
        "Motel-6-dog-on-bed.jpg",
        "Woman sitting on the arm of a couch with dog (AM 79405-1).jpg",
    ],
    "standing": [
        "A beagle wearing a harness standing in a park.jpg",
        "Beagle standing alone 2.jpg",
        "Beagle standing alone.jpg",
        "A beagle standing in a dog park.jpg",
        "Beagle puppy standing on a tree stump.jpg",
        "Dog standing on the fountain in parc Tenreuken (Auderghem, Belgium, DSCF2948).jpg",
        "Portrait of a dog standing on the grass (AM 76498-1).jpg",
        "Portrait of a dog standing on the grass (AM 76499-1).jpg",
        "Portrait of a dog standing on the grass (AM 76500-1).jpg",
        "Portrait of a dog standing on the grass (AM 85292-1).jpg",
        "Kissing Booth Dog Day 2007 NOLA.jpg",
        "Policeman standing in a park holding a dog by the leash LCCN2016829032.jpg",
        "Red dog standing in the sagebrush (53482547271).jpg",
        "1-BerneseMountainDogInGrass.jpg",
        "2-BerneseMountainDogInGrass.jpg",
        "1Shiba Inu Kazumi.jpg",
        "(01) P1120987 PitBull.jpg",
        "000 American Pit Bull Terrier.jpg",
        "001 American Pit Bull Terrier.jpg",
        "007 American Pit Bull Terrier.jpg",
        "'A Jack Russell Terrier in the snow'. Squaddie in a field near Belvoir Castle - Dec 2005.JPG",
    ],
    "lying_down": [
        "Brown-and-white dog relaxing on the sofa.jpg",
        "Chiweenie on a couch.jpg",
        "Dog sleeping Rome.JPG",
        "Joker, a Dog that lives in Herronton, Alberta.jpg",
        "Lazy Dog..jpg",
        "Lying dog.jpg",
        "Rex Cavalier King Charles Spaniel.jpg",
        "2021-03-30 22 56 07 A Golden Doodle lying on a dog bed in the Franklin Farm section of Oak Hill, Fairfax County, Virginia.jpg",
        "A Beagle lying on a bed in the afternoon sun.jpg",
        "Beagle lying on the bed.jpg",
        "Dog lying at his bed in a house (51784464611).jpg",
        "Big white dog lying on the floor.jpg",
        "Black dog lying on the floor.jpg",
        "Jack Russell Terrier lying on the floor (2008).jpg",
        "Mixed-breed dog black lying.jpg",
        "Search Dog Lies on Floor (9a8d9504-9bce-483e-92cf-0bf73a57ffd2).jpg",
        "Search Dog on Floor. (89c40985-c957-47eb-8e34-542e4fde00c8).jpg",
        "Shes just stella, stella from the floor (304102666).jpg",
        "Tobias sleeping in the floor 2.JPG",
        "2017-365-37 King of the Naps (32635784151).jpg",
        "Alter Zwergschnauzer.jpg",
        "Siberianhuskywhite1.jpg",
        "Siberianhuskywhite2.jpg",
    ],
    "walking": [
        "A greyhound walking through a forest.jpg",
        "Dog Walking in Trent Park - geograph.org.uk - 3752506.jpg",
        "Dog park (Parque La Carolina) Quito, Ecuador. a dog walking, hanging out. looking to the left.jpg",
        "Dog park (Parque La Carolina) Quito, Ecuador. a dog walking, hanging out.jpg",
        "Dog park (Parque La Carolina) Quito, Ecuador. two dogs walking, hanging out.jpg",
        "Dog walking in the park - geograph.org.uk - 1168430.jpg",
        "FLSP Walking your dog at First Landing State Park beach (24889805989).jpg",
        "Dog is walking along with snow to reach it's destination.jpg",
        "Dog walking in snow (52577707175).jpg",
        "Dog walking in snow-covered Sacher Park, 2015.jpg",
        "Dog walking in the snow (5824007034).jpg",
        "Shih tzu walking in the snow.jpg",
        "Walking dog in snow.jpg",
        "Dog owner walking his dogs on the beach..JPG",
        "Walking the dog trail at Beverly Beach State Park.jpg",
        "Hiker and dog in Goat Rocks Wilderness.jpg",
        "Dog walker - Buenos Aires.jpg",
        "The Man with Dog Walking on East Shore of Keelung Port 20131227.jpg",
        "Visitors hiking beach dog leash rialto beach c bubar march 06-2015 (22762306859).jpg",
        "City Park Avenue New Orleans March 2018 Allard Walking Dog.jpg",
        "Person walking a dog on an icy sidewalk in Quebec City.jpg",
        "Walking the dog (2155400232).jpg",
    ],
}


def api(params):
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def main():
    manifest = []
    for cls, titles in CANDIDATES.items():
        info = {}
        for i in range(0, len(titles), 40):
            d = api({
                "action": "query", "format": "json",
                "titles": "|".join("File:" + t for t in titles[i:i + 40]),
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
        for n, title in enumerate(titles, 1):
            ii = info.get(title)
            if not ii:
                continue
            mime, size = ii.get("mime", ""), ii.get("size", 0)
            if not mime.startswith("image/") or mime.endswith("svg+xml") or mime.endswith("tiff") or size > 15_000_000:
                print(f"SKIP: {title}")
                continue
            out = OUT / f"r3_{cls}_{n:03d}.jpg"
            if out.exists() and out.stat().st_size > 0:
                print(f"SKIP (done): {out.name}")
            else:
                raw = ROOT / f"raw_r3_{cls}_{n:03d}"
                req = urllib.request.Request(ii["url"], headers=UA)
                try:
                    with urllib.request.urlopen(req, timeout=120) as resp, open(raw, "wb") as f:
                        f.write(resp.read())
                    im = Image.open(raw).convert("RGB")
                    im.thumbnail((640, 640))
                    im.save(out, quality=90)
                except Exception as e:
                    print(f"FAIL {title}: {e}")
                    continue
                finally:
                    raw.unlink(missing_ok=True)
            em = ii.get("extmetadata", {})
            manifest.append({
                "file": f"round3/{out.name}", "class": cls, "title": title,
                "url": ii.get("url", ""),
                "license": em.get("LicenseShortName", {}).get("value", "?"),
                "artist": re.sub("<[^>]+>", "", em.get("Artist", {}).get("value", "?"))[:80],
                "uploader": ii.get("user", "?"),
            })
            print(f"OK {out.name} <- {title}")
    json.dump(manifest, open(ROOT / "manifest-round3.json", "w"), indent=1)
    from collections import Counter
    print(Counter(r["class"] for r in manifest), "total", len(manifest))


if __name__ == "__main__":
    main()
