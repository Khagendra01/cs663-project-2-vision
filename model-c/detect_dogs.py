#!/usr/bin/env python3
"""Model C / C+ demo - YOLOv8 (COCO pretrained) dog detection.

Proposal section 4/7: Model C is a pretrained YOLOv8 that locates the
dog in each raw camera frame and crops it before Model A classifies the
posture. This script demonstrates that stage for real: it runs
ultralytics YOLOv8 (yolov8n, COCO weights - 'dog' is COCO class 16) over
image sets, keeps only dog detections, and writes:

  <out>/<set>/annotated/<image>   box + "dog <conf>" drawn with PIL
  <out>/<set>/crops/<image>_dogN  one crop per detected dog (Model A input)
  <out>/results.json              per-image boxes/confidences + per-set
                                  summary (detection rate, mean confidence)

Usage:
  python3 detect_dogs.py --set NAME=DIR [--set NAME=DIR ...] \
      --out OUTDIR [--weights yolov8n.pt] [--conf 0.25]

No training or fine-tuning: the weights are the stock COCO release,
exactly as the proposal specifies for Model C+.
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def draw_boxes(img: Image.Image, dets):
    d = ImageDraw.Draw(img)
    w = max(2, img.width // 320)
    for (x1, y1, x2, y2), conf in dets:
        d.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=w)
        label = f"dog {conf:.2f}"
        ty = max(0, y1 - 14)
        d.rectangle([x1, ty, x1 + 8 * len(label) + 4, ty + 13],
                    fill=(255, 40, 40))
        d.text((x1 + 2, ty + 2), label, fill=(255, 255, 255))
    return img


def run_set(model, dog_id, name, folder, out_root, conf):
    folder = Path(folder)
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)
    ann_dir = out_root / name / "annotated"
    crop_dir = out_root / name / "crops"
    ann_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    per_image = []
    n_with = 0
    total_dogs = 0
    conf_sum = 0.0
    # batch predict for speed; stream=True keeps memory flat
    results = model.predict([str(p) for p in images], conf=conf,
                            verbose=False, stream=True)
    for path, res in zip(images, results):
        dets = []
        if res.boxes is not None:
            for box in res.boxes:
                if int(box.cls[0]) != dog_id:
                    continue
                xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
                c = float(box.conf[0])
                dets.append((xyxy, c))
        img = Image.open(path).convert("RGB")
        rel = path.relative_to(folder)
        out_name = rel.as_posix().replace("/", "__")
        annotated = draw_boxes(img.copy(), dets)
        annotated.save(ann_dir / out_name, quality=90)
        for i, ((x1, y1, x2, y2), c) in enumerate(dets):
            crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
            crop.save(crop_dir / f"{Path(out_name).stem}_dog{i}.jpg",
                      quality=90)
        if dets:
            n_with += 1
        total_dogs += len(dets)
        conf_sum += sum(c for _, c in dets)
        per_image.append({
            "set": name, "file": out_name, "source": str(path),
            "n_dogs": len(dets),
            "boxes": [b for b, _ in dets],
            "confidences": [round(c, 3) for _, c in dets],
        })
        print(f"[{name}] {out_name}: {len(dets)} dog(s)", flush=True)

    summary = {
        "n_images": len(images),
        "n_with_dog": n_with,
        "detection_rate": round(n_with / len(images), 4) if images else 0.0,
        "total_dogs": total_dogs,
        "mean_confidence": round(conf_sum / total_dogs, 4) if total_dogs else None,
    }
    print(f"[{name}] SUMMARY {summary}", flush=True)
    return summary, per_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True,
                    help="NAME=DIR (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model = YOLO(args.weights)
    dog_id = next(i for i, n in model.names.items() if n == "dog")
    print(f"model: {args.weights} | dog class id: {dog_id} | conf>={args.conf}")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    report = {"weights": args.weights, "conf_threshold": args.conf,
              "sets": {}, "images": []}
    for spec in args.set:
        name, folder = spec.split("=", 1)
        summary, per_image = run_set(model, dog_id, name, folder,
                                     out_root, args.conf)
        report["sets"][name] = summary
        report["images"].extend(per_image)
    (out_root / "results.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {out_root / 'results.json'}")


if __name__ == "__main__":
    main()
