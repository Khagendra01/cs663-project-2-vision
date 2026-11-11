#!/usr/bin/env python3
"""Build labelled contact sheets from staging-v4 candidates for the
mandatory visual verification pass. 30 candidates per sheet (5 cols x 6
rows), each cell labelled with its candidate file name (cand_NNNN).
Output: staging-v4/sheets/<class>_sheet_NN.jpg
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
STAGING = HERE / "staging-v4"
OUT = STAGING / "sheets"
PER = 30
COLS = 5
CELL_W, CELL_H, LABEL_H = 320, 240, 26

classes = sys.argv[1:] or ["sitting", "standing", "lying_down", "walking"]
OUT.mkdir(parents=True, exist_ok=True)
for cls in classes:
    files = sorted((STAGING / cls).glob("cand_*.jpg"))
    for si in range(0, len(files), PER):
        batch = files[si:si + PER]
        rows = (len(batch) + COLS - 1) // COLS
        sheet = Image.new("RGB", (COLS * CELL_W, rows * (CELL_H + LABEL_H)), "white")
        d = ImageDraw.Draw(sheet)
        for i, f in enumerate(batch):
            x, y = (i % COLS) * CELL_W, (i // COLS) * (CELL_H + LABEL_H)
            try:
                img = Image.open(f).convert("RGB")
            except Exception:
                continue
            img.thumbnail((CELL_W, CELL_H))
            sheet.paste(img, (x + (CELL_W - img.width) // 2, y))
            d.rectangle([x, y + CELL_H, x + CELL_W, y + CELL_H + LABEL_H], fill="black")
            d.text((x + 6, y + CELL_H + 6), f"{f.stem}  [{i + si + 1}/{len(files)}]", fill="white")
        out = OUT / f"{cls}_sheet_{si // PER + 1:02d}.jpg"
        sheet.save(out, quality=88)
        print(f"{out.name}: {len(batch)} imgs")
print("done")
