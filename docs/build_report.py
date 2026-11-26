#!/usr/bin/env python3
"""Build the Documentation Report PDF for CS663 Project 2 (Dog Training Assistant, Group 5).

Output: docs/CS663_Project2_Results_Report_Group5.pdf (overwritten on every run).

How to run (from the root of the cs663-project-2-vision repository):

    python -m venv .venv-report
    .venv-report/bin/pip install reportlab pillow
    .venv-report/bin/python docs/build_report.py

The app repository (cs663-project-2-app) is expected next to this one
(../cs663-project-2-app); set APP_REPO=/path/to/cs663-project-2-app to point
elsewhere. It is only used for an optional consistency check; the report
builds without it.

Where the numbers come from. Numbers that live in machine-readable files in
this repository are read from those files at build time (Model A metrics and
history, dataset provenance and splits, the Nov 11 and Nov 25 video
evaluation JSON). Numbers that exist only in Markdown result files are typed
in below, each next to a comment naming the file it comes from. Nothing in
the report is estimated. On-device timing numbers come from the Android
emulator only; the report says so wherever they appear.

Fonts: Bitstream Vera (ships inside ReportLab), plus the standard Symbol font
for arrows, so the build needs no system fonts.
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import reportlab
from PIL import Image as PILImage
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether,
                                NextPageTemplate, PageBreak, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle, XPreformatted)
from reportlab.platypus.tableofcontents import TableOfContents

# ----------------------------------------------------------------------------- paths
DOCS = Path(__file__).resolve().parent
VISION = DOCS.parent
APP = Path(os.environ.get("APP_REPO", VISION.parent / "cs663-project-2-app"))
OUT = DOCS / "CS663_Project2_Results_Report_Group5.pdf"

GF = VISION / "gemini-finetune"
MA = VISION / "model-a"
MC = VISION / "model-c"
VSL = VISION / "video-sequence-layer"
FIG_APP = DOCS / "figures" / "app"

VISION_URL = "https://github.com/Khagendra01/cs663-project-2-vision"
APP_URL = "https://github.com/Khagendra01/cs663-project-2-app"
BACKEND_URL = "https://cs663-gemini-service-917815040923.us-central1.run.app"


def load(p):
    with open(p) as f:
        return json.load(f)


# ----------------------------------------------------------------------------- data read from files
METRICS = load(MA / "metrics.json")
HIST = load(MA / "history.json")
TEST_PRED = load(MA / "test_predictions.json")
PROV3 = load(GF / "real-data-v3" / "provenance.json")
PROV4 = load(GF / "real-data-v4" / "provenance.json")
VID_A = load(VSL / "results" / "video_predictions_model_a.json")
VID_T = load(VSL / "results" / "video_predictions_tuned.json")
NOV25 = load(VSL / "results" / "video_evaluation_nov25.json")
MANIFEST = load(VSL / "video-sequences" / "manifest.json")

N_TEST_CORRECT = sum(1 for r in TEST_PRED if r["true"] == r["pred"])
assert len(TEST_PRED) == METRICS["counts"]["test"] == 235
assert abs(N_TEST_CORRECT / len(TEST_PRED) - METRICS["test_accuracy"]) < 1e-3
PC = METRICS["per_class"]
MACRO_F1 = sum(PC[c]["f1"] for c in METRICS["class_order"]) / 4.0
CM = METRICS["confusion_matrix"]["rows_true_cols_pred"]


def split_stats(prov):
    by = Counter((r["split"], r["behaviour"]) for r in prov)
    src = defaultdict(set)
    for r in prov:
        src[r["source"]].add(r["split"])
    leaks = sum(1 for s in src.values() if len(s) > 1)
    src_per_split = Counter(next(iter(v)) for v in src.values() if len(v) == 1)
    kinds = Counter((r["kind"], r["split"]) for r in prov)
    return by, src, leaks, src_per_split, kinds


BY3, SRC3, LEAK3, SPS3, KIND3 = split_stats(PROV3)
BY4, SRC4, LEAK4, SPS4, KIND4 = split_stats(PROV4)
LIC4 = Counter(r["license"] for r in PROV4)
assert len(PROV4) == 1000 and len(PROV3) == 171 and LEAK3 == 0 and LEAK4 == 0
PROV3_BY_FILE = {Path(r["dest"]).name: r for r in PROV3}
PROV4_BY_FILE = {Path(r["dest"]).name: r for r in PROV4}

CLIP_VIDEO = {  # sequence id -> source id in provenance (same six Commons videos)
    "seq01_stand_africanis": "vid_africanis", "seq02_down_batumi": "vid_batumi",
    "seq03_walk_mission": "vid_mission", "seq04_down_snoring": "vid_snoring",
    "seq05_down_panting": "vid_panting", "seq06_stand_wagging_short": "vid_wagging",
}

CLASSES = ["sitting", "standing", "lying_down", "walking"]
NICE = {"sitting": "sitting", "standing": "standing", "lying_down": "lying down", "walking": "walking"}

# ----------------------------------------------------------------------------- fonts and styles
FD = Path(reportlab.__file__).parent / "fonts"
pdfmetrics.registerFont(TTFont("Vera", str(FD / "Vera.ttf")))
pdfmetrics.registerFont(TTFont("VeraBd", str(FD / "VeraBd.ttf")))
pdfmetrics.registerFont(TTFont("VeraIt", str(FD / "VeraIt.ttf")))
pdfmetrics.registerFont(TTFont("VeraBI", str(FD / "VeraBI.ttf")))
pdfmetrics.registerFontFamily("Vera", normal="Vera", bold="VeraBd", italic="VeraIt", boldItalic="VeraBI")

INK = colors.HexColor("#1d2733")
MUTED = colors.HexColor("#5a6573")
ACCENT = colors.HexColor("#1f4e79")
ACCENT_L = colors.HexColor("#e8eef5")
RULE = colors.HexColor("#c9d2dc")
ZEBRA = colors.HexColor("#f5f7fa")
NOTE_BG = colors.HexColor("#fff7e6")
NOTE_BD = colors.HexColor("#e0b252")
PHONE_C = colors.HexColor("#e3eefa")
PHONE_B = colors.HexColor("#2f6db3")
CLOUD_C = colors.HexColor("#eef7ea")
CLOUD_B = colors.HexColor("#4b8b3b")

BODY = ParagraphStyle("body", fontName="Vera", fontSize=8.8, leading=12.2, textColor=INK,
                      spaceAfter=5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.6, leading=10, spaceAfter=3)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=11, bulletIndent=2, spaceAfter=2.5)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=7.6, leading=9.8, textColor=MUTED,
                     spaceBefore=3, spaceAfter=9)
H1 = ParagraphStyle("h1", fontName="VeraBd", fontSize=14, leading=18, textColor=ACCENT,
                    spaceBefore=4, spaceAfter=7, keepWithNext=1)
H2 = ParagraphStyle("h2", fontName="VeraBd", fontSize=10.4, leading=13.5, textColor=INK,
                    spaceBefore=8, spaceAfter=4, keepWithNext=1)
TC = ParagraphStyle("tc", fontName="Vera", fontSize=7.3, leading=9.1, textColor=INK)
TH = ParagraphStyle("th", parent=TC, fontName="VeraBd", textColor=colors.white)
CODE = ParagraphStyle("code", fontName="Courier", fontSize=7.2, leading=9, textColor=INK)

TEXT_W = letter[0] - 2 * 0.85 * inch


def sym(s):
    """Render arrows with the Symbol font (Vera has no arrow glyphs)."""
    return s.replace("→", '<font name="Symbol">→</font>').replace("←", '<font name="Symbol">←</font>')


def P(text, style=BODY):
    return Paragraph(sym(text), style)


def B(text):
    return Paragraph(sym(text), BUL, bulletText="•")


def bullets(items):
    return [B(t) for t in items]


FIG_N = [0]
TAB_N = [0]
REFS = {}        # label -> number, filled while the story is built
REFS_PREV = {}   # labels from the previous build of the story (for forward references)


def ref(label):
    return str(REFS_PREV.get(label, REFS.get(label, "?")))


def fig_caption(text, label=None):
    FIG_N[0] += 1
    if label:
        REFS[label] = FIG_N[0]
    return P(f"<b>Figure {FIG_N[0]}.</b> {text}", CAP)


def tab_caption(text, label=None):
    TAB_N[0] += 1
    if label:
        REFS[label] = TAB_N[0]
    return P(f"<b>Table {TAB_N[0]}.</b> {text}", CAP)


def table(rows, widths, caption=None, header=True, font=None, zebra=True, raw=False, total_w=None,
          label=None):
    """rows: list of lists of str (str may contain markup). widths in inches (scaled to fit)."""
    tot = sum(widths)
    w = [x / tot * (total_w or TEXT_W) for x in widths]
    st_c = TC if font is None else ParagraphStyle("tcx", parent=TC, fontSize=font, leading=font * 1.25)
    st_h = TH if font is None else ParagraphStyle("thx", parent=TH, fontSize=font, leading=font * 1.25)
    data = []
    for i, r in enumerate(rows):
        data.append([Paragraph(sym(str(c)), st_h if (header and i == 0) else st_c) for c in r])
    def styled(part, first):
        # part: rows data[first:first+len(part)]; styling uses the row's index in the full table
        tt = Table(part, colWidths=w, repeatRows=1 if (header and first == 0) else 0)
        cmds = [("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
                ("BOX", (0, 0), (-1, -1), 0.6, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
                ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6)]
        if header and first == 0:
            cmds += [("BACKGROUND", (0, 0), (-1, 0), ACCENT)]
        if zebra:
            for k in range(len(part)):
                i = first + k
                if (header and i == 0):
                    continue
                if (i % 2 == 0) == header:
                    cmds.append(("BACKGROUND", (0, k), (-1, k), ZEBRA))
        tt.setStyle(TableStyle(cmds))
        return tt

    if raw:
        return styled(data, 0)
    cap = tab_caption(caption, label) if caption else Spacer(1, 6)
    if len(rows) > 9:
        # Long tables may split across pages (the header row repeats). The last two rows go in a
        # separate table kept together with the caption, so a caption never starts a page alone.
        cut = len(data) - 2
        return [styled(data[:cut], 0), KeepTogether([styled(data[cut:], cut), cap])]
    return KeepTogether([styled(data, 0), cap])


def note(text):
    t = Table([[P(text, ParagraphStyle("n", parent=BODY, fontSize=8.2, leading=11, spaceAfter=0))]],
              colWidths=[TEXT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NOTE_BG),
                           ("LINEBEFORE", (0, 0), (0, -1), 2.5, NOTE_BD),
                           ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                           ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return [t, Spacer(1, 7)]


def code(text):
    t = Table([[XPreformatted(text, CODE)]], colWidths=[TEXT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f3f4f6")),
                           ("BOX", (0, 0), (-1, -1), 0.4, RULE),
                           ("LEFTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 4),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return [t, Spacer(1, 6)]


def img(path, width=None, height=None, crop=None):
    """Image scaled to width (inches) or height (inches); optional crop box (l, t, r, b) in pixels."""
    path = Path(path)
    if crop:
        tmp = DOCS / "figures" / ".build_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        cp = tmp / (path.stem + "_crop_%d_%d_%d_%d.jpg" % crop)
        if not cp.exists():
            PILImage.open(path).convert("RGB").crop(crop).save(cp, quality=90)
        path = cp
    w, h = PILImage.open(path).size
    if width is not None:
        W = width * inch
        H = W * h / w
    else:
        H = height * inch
        W = H * w / h
    return Image(str(path), width=W, height=H)


def stacked(path, bands, total_h=None, gap=10):
    """Full-width horizontal bands (top, bottom) of a screenshot stacked into one image, with a
    thin rule between bands; white padding at the bottom up to total_h pixels. Returns the path."""
    path = Path(path)
    tmp = DOCS / "figures" / ".build_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    key = "_".join("%d-%d" % b for b in bands) + ("_h%d" % total_h if total_h else "")
    out = tmp / f"{path.stem}_stack_{key}.jpg"
    if not out.exists():
        src = PILImage.open(path).convert("RGB")
        parts = [src.crop((0, t, src.width, b)) for t, b in bands]
        h = sum(p.height for p in parts) + gap * (len(parts) - 1)
        H = max(h, total_h or 0)
        im = PILImage.new("RGB", (src.width, H), (255, 255, 255))
        y = 0
        for i, p in enumerate(parts):
            if i:
                im.paste((200, 205, 212), (0, y + gap // 2 - 1, src.width, y + gap // 2 + 1))
                y += gap
            im.paste(p, (0, y))
            y += p.height
        im.save(out, quality=90)
    return out


def fig(path, width, caption, crop=None, label=None):
    return KeepTogether([img(path, width=width, crop=crop), fig_caption(caption, label)])


def image_row(cells, caption, col_w=None, sub_style=None, label=None, valign="BOTTOM"):
    """cells: list of (flowable_image, subcaption). One row, subcaptions under each image."""
    n = len(cells)
    cw = col_w or [TEXT_W / n] * n
    ss = sub_style or ParagraphStyle("sub", parent=CAP, spaceBefore=1, spaceAfter=0, alignment=TA_CENTER)
    t = Table([[c[0] for c in cells], [Paragraph(sym(c[1]), ss) for c in cells]], colWidths=cw)
    t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, 0), valign),
                           ("VALIGN", (0, 1), (-1, 1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    return KeepTogether([t, fig_caption(caption, label)])


def credit(fname, version=4):
    r = (PROV4_BY_FILE if version == 4 else PROV3_BY_FILE)[fname]
    a = " ".join(r.get("artist", "").split())
    m = re.search(r"flickr\.com/people/([^/]+)/?", a)
    if m:
        a = m.group(1) + " (Flickr)"
    a = re.sub(r"^Original photo:\s*", "", a)
    a = a.split(" Modified by")[0].strip()
    a = re.sub(r",?\s*\d{4}-\d{4},?\s*photographer", "", a)
    if len(a) > 48:
        a = a[:46].rstrip() + "..."
    return f"{a or r.get('uploader', '')}, {r['license']}, Wikimedia Commons"


# ----------------------------------------------------------------------------- doc template
class ReportDoc(BaseDocTemplate):
    def __init__(self, fn, **kw):
        super().__init__(fn, pagesize=letter, leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                         topMargin=0.85 * inch, bottomMargin=0.8 * inch,
                         title="CS663 Project 2 - Dog Training Assistant - Documentation Report (Group 5)",
                         author="Group 5: Arnav Verma, Khagendra Khatri", **kw)
        fr = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f",
                   leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate("title", [fr], onPage=lambda c, d: None),
                               PageTemplate("body", [fr], onPage=self._decor)])
        self._bm = 0

    def beforeDocument(self):
        self._bm = 0

    def _decor(self, c, d):
        c.saveState()
        c.setFont("Vera", 7)
        c.setFillColor(MUTED)
        y = letter[1] - 0.55 * inch
        c.drawString(self.leftMargin, y, "CS663 Project 2 · Dog Training Assistant · Group 5")
        c.drawRightString(letter[0] - self.rightMargin, y, "Documentation Report · November 2026")
        c.setStrokeColor(RULE)
        c.setLineWidth(0.5)
        c.line(self.leftMargin, y - 4, letter[0] - self.rightMargin, y - 4)
        c.restoreState()

    def afterFlowable(self, f):
        lvl = getattr(f, "_toc_level", None)
        if lvl is None:
            return
        text = f._toc_text
        key = "bm%d" % self._bm
        self._bm += 1
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=lvl, closed=False)
        self.notify("TOCEntry", (lvl, text, self.page, key))


class NumberedCanvas(rl_canvas.Canvas):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        n = len(self._saved)
        for st in self._saved:
            self.__dict__.update(st)
            if self._pageNumber > 1:
                self.setFont("Vera", 7.5)
                self.setFillColor(MUTED)
                self.drawCentredString(letter[0] / 2, 0.45 * inch, "Page %d of %d" % (self._pageNumber, n))
            super().showPage()
        super().save()


def H(num, text, level=0, cond=None):
    full = f"{num}  {text}" if num else text
    p = Paragraph(full, H1 if level == 0 else H2)
    p._toc_level = level
    p._toc_text = f"{num}  {text}" if num else text
    if cond is None:
        cond = 1.6 if level == 0 else 1.1
    return [CondPageBreak(cond * inch), p]


# ----------------------------------------------------------------------------- architecture diagram
def arch_diagram():
    W, Hh = TEXT_W, 300
    d = Drawing(W, Hh)
    sx = W / 489.6

    def X(v):
        return v * sx

    def box(x, y, w, h, fill, stroke, title, lines, tsize=7.6, lsize=6.4):
        d.add(Rect(X(x), y, X(w), h, rx=4, ry=4, fillColor=fill, strokeColor=stroke, strokeWidth=0.9))
        cy = y + h - 12
        d.add(String(X(x + w / 2), cy, title, fontName="VeraBd", fontSize=tsize, textAnchor="middle",
                     fillColor=INK))
        for i, ln in enumerate(lines):
            d.add(String(X(x + w / 2), cy - 11 - i * 8.6, ln, fontName="Vera", fontSize=lsize,
                         textAnchor="middle", fillColor=INK))

    def arrow(pts, dashed=False, color=INK):
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            ln = Line(X(x1), y1, X(x2), y2, strokeColor=color, strokeWidth=0.9)
            if dashed:
                ln.strokeDashArray = [2.5, 2]
            d.add(ln)
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        import math
        a = math.atan2(y2 - y1, X(x2) - X(x1))
        L, Wd = 5.5, 2.6
        tip = (X(x2), y2)
        b1 = (tip[0] - L * math.cos(a) + Wd * math.sin(a), tip[1] - L * math.sin(a) - Wd * math.cos(a))
        b2 = (tip[0] - L * math.cos(a) - Wd * math.sin(a), tip[1] - L * math.sin(a) + Wd * math.cos(a))
        d.add(Polygon([tip[0], tip[1], b1[0], b1[1], b2[0], b2[1]], fillColor=color, strokeColor=color,
                      strokeWidth=0.3))

    def label(x, y, text, anchor="start", size=6.1, color=MUTED, bold=False):
        d.add(String(X(x), y, text, fontName="VeraBd" if bold else "Vera", fontSize=size,
                     textAnchor=anchor, fillColor=color))

    # lanes
    d.add(Rect(X(4), 150, X(395), 146, rx=6, ry=6, fillColor=colors.HexColor("#f6f9fd"),
               strokeColor=PHONE_B, strokeWidth=0.7, strokeDashArray=[3, 2]))
    label(10, 286, "PHONE  (on device, works offline)", bold=True, color=PHONE_B, size=6.8)
    d.add(Rect(X(4), 6, X(395), 128, rx=6, ry=6, fillColor=colors.HexColor("#f7fbf5"),
               strokeColor=CLOUD_B, strokeWidth=0.7, strokeDashArray=[3, 2]))
    label(10, 124, "CLOUD", bold=True, color=CLOUD_B, size=6.8)
    for i, ln in enumerate(["Cloud Run service", "cs663-gemini-service", "(Node, us-central1);", "models on Vertex AI"]):
        label(10, 113 - i * 7.6, ln, size=5.6, color=CLOUD_B)

    # phone boxes
    bw, by, bh = 86, 170, 104
    xs = [9, 107, 205, 303]
    box(xs[0], by, bw, bh, PHONE_C, PHONE_B, "Camera / input",
        ["CameraX preview +", "ImageAnalysis", "frames made upright", "640 x 480", "capped at ~5 fps",
         "(debug: replay frames)"])
    box(xs[1], by, bw, bh, PHONE_C, PHONE_B, "Model C",
        ["EfficientDet-Lite0", "COCO, int8, MediaPipe", "is there a dog? where?", "box + score,",
         "tracked by IoU", "no dog: frame unclear"])
    box(xs[2], by, bw, bh, PHONE_C, PHONE_B, "Model A",
        ["MobileNetV3-Small", "fine-tuned by us, LiteRT", "whole frame 224 x 224", "posture + softmax",
         "counted only when", "Model C found a dog"])
    box(xs[3], by, bw, bh, PHONE_C, PHONE_B, "State machine",
        ["SequenceSession (Java)", "READY gate, majority", "of 5, 0.65 gate, timing", "break and lost rules",
         "SUCCESS / EARLY_BREAK", "/ UNCERTAIN"])
    for i in range(3):
        arrow([(xs[i] + bw, by + bh / 2), (xs[i + 1], by + bh / 2)])
    # result card (phone UI) spanning both lanes on the right
    box(406, 6, 80, 290, PHONE_C, PHONE_B, "Result card", [], tsize=7.6)
    for i, ln in enumerate(["(phone UI)", "", "verdict + time", "spoken cue +", "vibration", "",
                            "Model B status", "VERIFIED /", "DISPUTED /", "PARTLY /", "NOT CHECKED", "",
                            "coaching tip", "(or built-in tip)", "", "per-attempt log", "CSV + JSON"]):
        d.add(String(X(446), 270 - i * 9.2, ln, fontName="Vera", fontSize=6.4, textAnchor="middle",
                     fillColor=INK))
    arrow([(xs[3] + bw, by + bh / 2 + 18), (406, by + bh / 2 + 18)])

    # cloud boxes
    cy, ch = 22, 82
    box(150, cy, 118, ch, CLOUD_C, CLOUD_B, "Model B",
        ["fine-tuned gemini-3.5-flash", "Vertex endpoint (us)", "POST /api/fine-tuned-gemini",
         "JPEG crop + dataset prompt", "→ one posture word".replace("→", "returns")])
    box(278, cy, 118, ch, CLOUD_C, CLOUD_B, "Base Gemini",
        ["gemini-3-flash-preview", "Vertex (global)", "POST /api/gemini", "verdict facts + frame",
         "returns JSON coaching"])
    # state machine -> Model B (decisive frames)
    arrow([(318, by), (318, 146), (230, 146), (230, cy + ch)])
    label(234, 138, "2-3 decisive frames after the verdict", size=5.6)
    # state machine -> base Gemini
    arrow([(372, by), (372, cy + ch)])
    label(368, 112, "verdict, timeline", anchor="end", size=5.6)
    # Model B -> base Gemini (its check goes into the coaching prompt)
    arrow([(268, cy + ch / 2), (278, cy + ch / 2)])
    # Model B -> card (below cloud boxes)
    arrow([(190, cy), (190, 13), (406, 13)], color=CLOUD_B)
    # base Gemini -> card
    arrow([(396, cy + ch / 2 - 14), (406, cy + ch / 2 - 14)], color=CLOUD_B)
    # READY tie-break: Model C crop -> Model B (dashed)
    arrow([(122, by), (122, cy + 18), (150, cy + 18)], dashed=True)
    for i, ln in enumerate(["dashed: READY tie-break,", "one crop when Model A", "is unsure for 2 s"]):
        label(10, 76 - i * 7.6, ln, size=5.6)
    for i, ln in enumerate(["Cloud steps never", "change the verdict;", "offline: NOT CHECKED"]):
        label(10, 46 - i * 7.6, ln, size=5.6, color=INK)
    return d


# ----------------------------------------------------------------------------- content helpers
def pct(a, b):
    return f"{a}/{b} = {100.0 * a / b:.1f}%"


def seq_rows_nov11():
    a = {s["id"]: s for s in VID_A["sequences"]}
    t = {s["id"]: s for s in VID_T["sequences"]}
    rows = [["Sequence (frames)", "Expected", "Model A verdict (frames right)", "Tuned Gemini verdict (frames right)"]]
    for m in MANIFEST:
        sid = m["id"]
        sa, st = a[sid], t[sid]
        na = sum(1 for f in sa["frames"] if f["correct"])
        nt = sum(1 for f in st["frames"] if f["correct"])
        def v(s):
            r = s["verdict"]
            if r == "SUCCESS":
                return f"SUCCESS, held {s['held_s']:.1f} s"
            if "never established" in s["reason"]:
                return "<b>UNCERTAIN</b>, target never established"
            return f"UNCERTAIN, held {s['held_s']:.1f} s of {m['target_duration_s']} s"
        rows.append([f"{sid.replace('_', ' ', 1).split(' ')[0]} {m['behaviour_throughout'].replace('_', ' ')} ({m['n_frames']})",
                     m["expected_verdict"], f"{v(sa)} ({na}/{len(sa['frames'])})",
                     f"{v(st)} ({nt}/{len(st['frames'])})"])
    return rows


def seq_rows_nov25():
    rows = [["Clip", "App command shape", "Model A", "Tuned Gemini", "Human labels", "All as holds (A / tuned / labels)"]]
    for s in NOV25["sequences"]:
        src = s["sources"]
        shape = ("Down: reach in 3 s, hold 0.6 s" if s["command"] == "down"
                 else f"{s['command'].capitalize()}: hold {s['target_duration_s']} s")
        def v(k):
            app = src[k]["app"]
            r = app["verdict"]
            if r == "SUCCESS":
                return "SUCCESS" + ("" if s["command"] == "down" else f", held {app['held_s']:.1f} s")
            if "never established" in app["reason"]:
                return "UNCERTAIN (never established)"
            return f"UNCERTAIN (held {app['held_s']:.1f} s)"
        hold = " / ".join(src[k]["steps"]["app_numbers_hold_clip_time"]["verdict"].replace("UNCERTAIN", "UNC.")
                          .replace("SUCCESS", "SUCC.") for k in ("model_a", "tuned", "ground_truth"))
        rows.append([s["id"].split("_")[0], shape, v("model_a"), v("tuned"), v("ground_truth"), hold])
    return rows


# ----------------------------------------------------------------------------- story
def build():
    S = []

    # ---------------- title page
    S.append(Spacer(1, 1.3 * inch))
    S.append(P("CS663 Computer Vision · Project 2", ParagraphStyle("k", parent=BODY, fontSize=10,
                                                                  textColor=MUTED, alignment=TA_CENTER)))
    S.append(Spacer(1, 6))
    S.append(P("Dog Training Assistant", ParagraphStyle("t", fontName="VeraBd", fontSize=28, leading=34,
                                                        textColor=ACCENT, alignment=TA_CENTER)))
    S.append(P("Documentation Report", ParagraphStyle("t2", fontName="Vera", fontSize=15, leading=20,
                                                      textColor=INK, alignment=TA_CENTER, spaceAfter=4)))
    S.append(P("Data documentation · Model results · System documentation",
               ParagraphStyle("t3", parent=BODY, fontSize=9.5, textColor=MUTED, alignment=TA_CENTER)))
    S.append(Spacer(1, 0.45 * inch))
    tp = [["Group", "Group 5"],
          ["Members", "Arnav Verma, Khagendra Khatri"],
          ["Course", "CS663 Computer Vision, Fall 2026, California State University, East Bay (Prof. Lynne Grewe)"],
          ["Date", "November 2026"],
          ["Repository 1: mobile app + backend", f'<link href="{APP_URL}" color="#1f4e79">{APP_URL}</link>'],
          ["Repository 2: models, data, evaluation", f'<link href="{VISION_URL}" color="#1f4e79">{VISION_URL}</link>'],
          ["Live backend (Cloud Run)", BACKEND_URL],
          ["Demonstration video", "to be added (not recorded yet, see Section 13)"]]
    t = Table([[P(f"<b>{a}</b>", TC), P(b, TC)] for a, b in tp], colWidths=[1.95 * inch, TEXT_W - 1.95 * inch])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                           ("BACKGROUND", (0, 0), (0, -1), ACCENT_L)]))
    S.append(t)
    S.append(Spacer(1, 0.35 * inch))
    S.append(P("The two repositories call themselves GitHub B (app) and GitHub A (vision) in their own "
               "README files; this report uses the course's numbering above. Kokila Ganesan, a co-author of the "
               "proposal, was a member of Group 5 until Nov 19, 2026.",
               ParagraphStyle("tn", parent=SMALL, textColor=MUTED, alignment=TA_CENTER)))
    S.append(Spacer(1, 0.25 * inch))
    S.append(P("Where results were measured: model accuracy on held-out image test splits; the integrated "
               "app on the <b>Android emulator only</b> (Apple-silicon Mac, Pixel 10 Pro system image), replaying "
               "the project's video clips. The app has not yet been run on a physical phone, so every on-device "
               "timing or frame-rate number in this report is an emulator number.",
               ParagraphStyle("tn2", parent=SMALL, alignment=TA_CENTER)))
    S.append(NextPageTemplate("body"))
    S.append(PageBreak())

    # ---------------- TOC
    S.append(P("Contents", H1))
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("toc0", fontName="VeraBd", fontSize=8.8, leading=11.2, leftIndent=0,
                                      firstLineIndent=0, spaceBefore=2.2),
                       ParagraphStyle("toc1", fontName="Vera", fontSize=7.8, leading=9.6, leftIndent=16,
                                      firstLineIndent=0, textColor=MUTED)]
    toc.dotsMinLevel = 0
    toc.tableStyle = TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 0),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 0), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                 ("RIGHTPADDING", (0, 0), (-1, -1), 0)])
    S.append(P("Where the course's required items are: <b>Data documentation</b>: Section 4. "
               "<b>Model results</b> (training, validation and test results, metrics, graphs, successful and "
               "failure examples): Sections 5-7 and 10. <b>Base vs fine-tuned Gemini</b>: Section 6.3. "
               "<b>System documentation</b>: Sections 1-13, in the order the course lists them.", SMALL))
    S.append(Spacer(1, 6))
    S.append(toc)
    S.append(PageBreak())

    # ================================================================ 1 overview
    S += H("1", "Project Overview")
    S.append(P("<b>Problem.</b> Amateur dog trainers and pet owners struggle to time rewards and to notice the "
               "moment a dog breaks a sustained command such as Sit, Down or Stay. A reward that comes late, or "
               "after the dog has already moved, sends the dog a confusing signal. "
               "<b>Users:</b> dog owners and amateur trainers working with one dog at home or outdoors. "
               "<b>Why computer vision:</b> whether a dog held a posture is a visual question asked over time; the "
               "camera lets the trainer keep both hands and their attention on the dog."))
    S.append(P("<b>What the system does.</b> The owner props the phone up facing the dog, picks a command (Sit, "
               "Down, Stand, Stay or Walk; for Stay also the posture; for Stand, Stay and Walk a duration of 5, 10 "
               "or 20 s) and taps START once the app "
               "says READY. The app watches the attempt and answers <b>SUCCESS</b>, <b>EARLY_BREAK</b> (with the "
               "second the dog broke) or <b>UNCERTAIN</b> (it could not see well enough to judge, which is never "
               "reported as a success or a failure). It speaks and vibrates at the verdict, has the fine-tuned "
               "Gemini model re-check the frames the verdict rests on, and shows one coaching tip."))
    S.append(P("<b>The three models</b>, each with a job the others cannot do:"))
    S += bullets([
        "<b>Model C</b> (pretrained EfficientDet-Lite0, on the phone) answers <i>is there a dog, and where?</i> "
        "Model A has no \"no dog\" class and would name a posture even for an empty room, so no frame is judged "
        "unless Model C found a dog.",
        "<b>Model A</b> (MobileNetV3-Small fine-tuned by us, on the phone) gives the posture of every frame fast "
        "enough to time a 0.6 s hold.",
        "<b>Model B</b> (gemini-3.5-flash fine-tuned by us on Vertex AI, in the cloud) scored higher on its own "
        "held-out image test (88.4% on 43 v3 photos vs Model A's 74.9% on 235 v4 photos; different test sets) "
        "but takes seconds per call, so it checks the verdict's decisive frames after the attempt "
        "instead of producing the verdict. Base Gemini (gemini-3-flash-preview) writes the coaching text.",
    ])
    S.append(P("A temporal state machine on the phone turns the per-frame labels into one verdict. It is the part "
               "that makes this more than recognition: it smooths labels, gates on confidence, applies the response "
               "window and hold timers, and decides between a real break and a dog that was simply not visible."))
    glance = [["Component", "Test", "Result", "Where measured"],
              ["Model A (MobileNetV3-Small, dataset v4)", "235 held-out photos",
               f"{pct(N_TEST_CORRECT, 235)}, macro-F1 {MACRO_F1:.3f}; val {METRICS['val_accuracy']:.1%}",
               "offline, CPU"],
              ["Model B vs base gemini-3.5-flash (dataset v3)", "43 held-out photos",
               "tuned 38/43 = 88.4% vs base 37/43 = 86.0%", "Vertex AI, Oct 30"],
              ["Model C offline (YOLOv8n)", "60 held-out photos / 111 clip frames",
               "dog found 81.7% / 58.6%", "offline, CPU"],
              ["Model C in the app (EfficientDet-Lite0)", "111 clip frames (rule tuned on them; not held out, "
               "Section 4.3)", "dog found 96% (dog/cat only: 61%)", "offline, app's model file"],
              ["State machine", "21 Java + 42 Python unit tests", "all pass", "JVM / Python"],
              ["Offline video pipeline", "6 real clips (not held out, Section 4.3)",
               "Model A 5/6 verdicts; tuned Gemini 6/6", "Python, Nov 11 and Nov 25"],
              ["Full system (all three models)", "5 replayed attempts",
               "4/5 expected verdicts; Model B VERIFIED on all 4 verdicts", "<b>Android emulator only</b>"],
              ["On-device time per frame (C + A + crop)", "4 full attempts",
               "median 94-149 ms, p90 128-335 ms", "<b>Android emulator only</b>"]]
    S.append(table(glance, [1.9, 1.45, 2.25, 1.2], "Results at a glance. Sources are given in the sections below."))
    S += note("<b>Status, stated plainly.</b> The app is built (repository 1, main branch) and runs end to end on the "
              "Android emulator with all three models and the live backend. It has <b>not yet run on a physical "
              "phone or with a live dog</b>; that run is the demonstration video, which is not recorded yet. All "
              "imagery is openly licensed stock, not the group's own dogs.")

    # ================================================================ 2 architecture
    S += H("2", "System Architecture")
    S.append(P("Figure 1 shows the integrated system. Everything up to the verdict runs on the phone. The cloud calls "
               "(Model B on the 2-3 decisive frames, then one coaching call) happen after the verdict, plus optional "
               "READY tie-break calls (at most one every 6 s) while Model A is unsure; they can only annotate the "
               "verdict. The verdict therefore never waits on the network."))
    S.append(KeepTogether([arch_diagram(), fig_caption(label="f_arch", text=
        "System architecture. Blue: on the phone; green: cloud. Solid arrows are the per-attempt data flow; the dashed "
        "arrow is the optional READY tie-break. Model B's check is also passed to base Gemini so the coaching text "
        "can mention it.")]))
    S += H("2.1", "Processing steps and information passed between components", 1)
    S += bullets([
        "<b>Preprocessing.</b> CameraX frames are rotated upright (also when the phone is held in landscape) and "
        "analysed at most every 180 ms (a cap of about 5 fps). Model A gets the whole frame resized to 224×224; "
        "Model B gets a crop around Model C's box padded by 15% and encoded as JPEG (quality 85, longest side 640).",
        "<b>Tracking.</b> The tracked dog is the box that overlaps the previous one (IoU ≥ 0.2) or, failing that, "
        "the one whose centre is close. If the detector misses, the box is held for up to 1 s for display only; "
        "those frames count as unclear, never as judged.",
        "<b>Why Model A reads the whole frame and not Model C's crop.</b> Measured with the app's own model files: "
        "on Model A's 235 held-out test photos the whole image scored 75.3% and the padded crop 68.5% (walking "
        "43/62 vs 17/62); on the 111 clip frames 73.9% vs 62.2% (Table " + ref("t_crop") + "). Model A was trained on whole photos.",
        "<b>After the verdict</b>, the phone sends Model B the decisive frames (target reached, first wrong posture, "
        "last frame) and sends base Gemini the verdict, held time, break time, a thinned posture timeline, Model B's "
        "status and the decisive frame.",
    ])
    S += H("2.2", "Per-model specification", 1)
    spec = [["Model", "Purpose", "Input", "Output", "Where it runs", "Customization", "Next step"],
            ["<b>C</b>: EfficientDet-Lite0 (COCO, int8), MediaPipe Tasks",
             "Is a dog in view, and where? Without a dog no frame is judged, so an empty room cannot open READY or "
             "\"hold\" a Stay", "One upright camera frame (640×480)", "Boxes + score + COCO label",
             "Phone, every analysed frame", "Pretrained, not fine-tuned. Accept rule chosen on our clips: \"dog\" "
             "≥ 0.25, or cat/horse/sheep/cow/bear ≥ 0.30",
             "Dog found → Model A's reading counts; box → overlay, tracking, and the crop Model B checks"],
            ["<b>A</b>: MobileNetV3-<br/>Small (ImageNet backbone), LiteRT",
             "Posture of the dog on every frame, fast enough for timing", "Whole frame resized to 224×224 RGB, "
             "0-255", "sitting / standing / lying_down / walking + softmax confidence", "Phone, every analysed frame",
             "Fine-tuned by us on dataset v4 (626 training images)", "State machine (confidence < 0.65 → unclear frame)"],
            ["<b>B</b>: gemini-3.5-<br/>flash, supervised fine-tune (Vertex AI)",
             "(1) Re-check the 2-3 decisive frames of the verdict; (2) READY tie-break when Model A is unsure 2 s",
             "One JPEG crop + the dataset prompt", "One posture word", "Cloud: Vertex endpoint via Cloud Run",
             "Fine-tuned by us on dataset v3 (102 training images)",
             "VERIFIED / DISPUTED / PARTLY CHECKED / NOT CHECKED on the card; also given to base Gemini"],
            ["(base) gemini-3-<br/>flash-preview", "Coaching text only; told the verdict and told not to re-judge it",
             "Verdict facts, timeline, Model B status, decisive frame", "JSON: evidence_summary, coaching_tip, caution",
             "Cloud: Vertex (global) via Cloud Run", "None (prompted)", "Coaching card; built-in tip on error"]]
    S.append(table(spec, [0.95, 1.25, 0.95, 1.2, 0.8, 1.15, 1.1],
                   "Model specification table in the course's format (sources: repository 1 README and the Java sources "
                   "<i>DogDetector</i>, <i>PostureClassifier</i>, <i>VisionPipeline</i>, <i>SequenceActivity</i>, "
                   "<i>GeminiClient</i>).", font=6.9))
    S += H("2.3", "Application logic: the temporal state machine", 1)
    S.append(P("<i>SequenceSession.java</i> (repository 1) is the state machine the app runs; "
               "<i>video-sequence-layer/state_machine.py</i> (repository 2) was brought to the same rules on Nov 25 "
               "(branch report-and-python-port) and is used for offline evaluation. Times are in seconds, so the rules hold at any frame rate."))
    rules = [["Rule", "Value", "Effect"],
             ["READY gate", "starting posture stable 0.6 s on ≥ 3 detected frames",
              "START is refused until then (\"START (waiting)\"); a long press overrides it and is logged"],
             ["Starting posture", "Sit: standing · Down: sitting · Stand: standing · Stay: the chosen posture · Walk: none",
              "Matches the proposal's command set"],
             ["Command shape: Sit, Down", "reach target within 3.0 s, then hold 0.6 s",
              "Dog clearly in another posture at the deadline → EARLY_BREAK (failed command); unclear at the deadline → UNCERTAIN"],
             ["Command shape: Stand, Stay, Walk", "reach within 2.0 s settle grace, then hold the target duration",
              "Durations 5 / 10 / 20 s (picker)"],
             ["Smoothing", "majority of the last 5 confident labels", "One wrong frame cannot flip the judged posture"],
             ["Confidence gate", "Model A softmax < 0.65 → unclear frame", "Plays the proposal's \"Undefined\" role"],
             ["Break", "a smoothed non-target run must last 0.6 s",
              "Break time = first frame after the last raw target frame, not the lagged smoothed flip"],
             ["Lost / unclear", "no dog or no confident label for 1.5 s → UNCERTAIN",
              "Unless ≥ 2 consecutive wrong raw frames came first: then EARLY_BREAK at the first wrong frame"],
             ["Unclear share", "> 40% unclear frames → UNCERTAIN", "Checked when the attempt ends"],
             ["Reach time", "first raw target frame of the confirmed run", "Removes the smoother's lag from the timer"]]
    S.append(table(rules, [1.25, 2.4, 3.15], "State machine rules as implemented (SequenceSession.java; same values in "
                                            "state_machine.py and in video_evaluation_nov25.json)."))

    # ================================================================ 3 installation
    S += H("3", "Installation / Setup")
    S += H("3.1", "Using the app", 1)
    S += bullets([
        "Android 7.0 (API 24) or newer, 64-bit or 32-bit ARM CPU (arm64-v8a, armeabi-v7a; MediaPipe ships no x86 "
        "build), a camera, and internet for the two Gemini steps. Models C and A run offline.",
        "No accounts or keys: the backend is deployed and its URL is prefilled (editable) in the app.",
        "The APK is a debug build made from source (Section 3.2). The file <i>DogBackendLab.apk</i> in the root of "
        "repository 1 is the old v1 build and does not contain this version.",
        "Point the phone at the dog side-on with the whole dog in frame, pick the command, wait for READY, tap START, "
        "then give the command. <i>Share log</i> exports the attempt's per-frame CSV and JSON summary; "
        "<i>Model lab</i> runs one photo through C and A on the phone and compares base vs tuned Gemini on it.",
    ])
    S += H("3.2", "Building the app", 1)
    S += code("cd cs663-project-2-app/android-app\n"
              "export JAVA_HOME=\"/Applications/Android Studio.app/Contents/jbr/Contents/Home\"   # any JDK 17-21\n"
              "./gradlew assembleDebug testDebugUnitTest\n"
              "# -> app/build/outputs/apk/debug/app-debug.apk ; 21 SequenceSession unit tests")
    S.append(P("Gradle 8.7, Android Gradle Plugin 8.5.2, compileSdk 35, minSdk 24, targetSdk 34, Java 17. Both "
               "<i>.tflite</i> files ship uncompressed in <i>app/src/main/assets/</i> "
               "(<i>efficientdet_lite0.tflite</i>, 4.6 MB; <i>model_a_posture.tflite</i>, 1.95 MB, byte-identical to "
               "<i>model-a/model_a.tflite</i> in repository 2)."))
    S += H("3.3", "Backend", 1)
    S.append(P("Nothing to run: Cloud Run service <i>cs663-gemini-service</i> is live at " + BACKEND_URL +
               ". To run it locally or redeploy it (needs Vertex AI access to the Google Cloud project):"))
    S += code("cd cs663-project-2-app/cloud-run-backend\n"
              "npm install\n"
              "MOCK_MODE=true npm start        # local stub replies, flagged mock:true; port 8080\n"
              "./deploy.sh                     # gcloud run deploy --source . (project, region, endpoint as defaults)")
    S += H("3.4", "Python environments for training and evaluation (repository 2)", 1)
    env = [["Task", "Needs"],
           ["State machine tests, offline re-run, scenario checks", "Python 3 standard library only"],
           ["Model A training and the Model A video driver", "TensorFlow / Keras (CPU is enough; training took about 7 min)"],
           ["Model C offline evaluation", "ultralytics 8.4.210, torch 2.15.0, yolov8n.pt (COCO)"],
           ["App-side offline checks (repository 1, eval/)", "mediapipe, ai-edge-litert, pillow, numpy"],
           ["Gemini evaluation and the live video driver", "google-genai and access to the Google Cloud project; the live driver calls the backend"],
           ["This report", "reportlab, pillow"]]
    S.append(table(env, [3.0, 3.8], "Software needed per task."))

    # ================================================================ 4 data
    S += H("4", "Data")
    S.append(P("All imagery is openly licensed stock from Wikimedia Commons (CC0, CC BY, CC BY-SA, public domain), "
               "credited per image in <i>provenance.json</i>. None of it shows the group's own dogs. Every candidate "
               "was looked at on a contact sheet before it got a label; titles and search results were not trusted. "
               "There are four posture classes: sitting, standing, lying down, walking."))
    ver = [["Version", "Images", "Built", "Used for", "Location (repository 2)"],
           ["v1", "37 video frames, 3 classes", "Oct 26-27", "pilot only", "gemini-finetune/real-data/"],
           ["v2", "110 (73 photos, 37 video frames)", "Oct 27", "superseded", "gemini-finetune/real-data-v2/"],
           ["<b>v3</b>", "<b>171</b> (134 photos, 37 video frames)", "Oct 27", "<b>Model B fine-tune</b> and its base vs tuned test",
            "gemini-finetune/real-data-v3/, real-jsonl-v3/"],
           ["<b>v4</b>", "<b>1,000</b> (963 photos, 37 video frames), 250 per class", "Nov 10",
            "<b>Model A</b> training, validation, test; Model C photo sample", "gemini-finetune/real-data-v4/, real-jsonl-v4/"],
           ["Video clips", "6 clips, 111 frames at 2 fps", "Nov 10", "Sequence evaluation, emulator replay, Model C recall",
            "video-sequence-layer/video-sequences/"],
           ["Scenario corpus", "180 synthetic label streams + advice", "Nov 10", "Model B feedback corpus (not used by the deployed tune)",
            "gemini-finetune/model-b-scenarios/"]]
    S.append(table(ver, [0.75, 1.6, 0.68, 1.8, 2.17], "Datasets in the project (gemini-finetune/real-dataset.md, provenance files)."))

    S += H("4.1", "Collection, labelling and format", 1)
    S += bullets([
        "<b>Collection.</b> Scripts query Commons categories and searches (<i>download_candidates.py</i>, "
        "<i>download_round3.py</i>, <i>download_sitting2.py</i> for v1-v3; <i>collect_v4*.py</i> for v4). The v4 "
        "collector keeps only CC / public-domain JPEGs of at least 500×320 and drops duplicates by title and "
        "perceptual hash, also against v3. Video frames come from six Commons videos via <i>extract_frames.py</i>.",
        "<b>Verification.</b> v4 started from 2,880 raw candidates (sitting 520, standing 520, lying down 520, walking "
        "1,320). Each was viewed on a 30-image contact sheet (<i>make_sheets_v4.py</i>) and dropped unless it showed "
        "one clear dog plainly in the labelled posture (dropped: running, statues and paintings, other animals, "
        "packs, tiny dogs). Verified keeps: sitting 284, standing 424, lying down 219, walking 235. "
        "<i>assemble_v4.py</i> took 250 per class round-robin across uploaders; no unverified image pads a class.",
        "<b>Format.</b> JPEG, 640 px wide, stored as <i>real-data-vN/&lt;source&gt;/&lt;class&gt;/&lt;file&gt;.jpg</i>. "
        "Labels are the folder name and the provenance record (title, URL, licence, artist, uploader, split). For "
        "Gemini tuning, each image is one Vertex AI JSONL record: the image (Cloud Storage URI) and the fixed prompt "
        "<i>\"Look at the image of the dog. What is the dog doing? Answer with exactly one of: sitting, standing, "
        "lying down, walking.\"</i>, with the label as the model's answer.",
        "<b>Preprocessing</b> is model-specific: Model A resizes to 224×224 and augments in training only (horizontal "
        "flip, rotation 0.08, zoom 0.10, brightness 0.15); Gemini receives the stored JPEG.",
    ])
    S += H("4.2", "Splits and leak checks", 1)
    S.append(P("Splits are made <b>by source</b> (photo uploader or video), never by shuffling images, so no source "
               "appears in two splits. The counts below are recomputed from the provenance files when this report is "
               "built, including the leak check."))
    rows = [["Split", "sitting", "standing", "lying down", "walking", "Total", "Sources", "Photos / video frames"]]
    for sp in ("train", "validation", "test"):
        n = [BY4[(sp, c)] for c in CLASSES]
        rows.append([sp, *map(str, n), str(sum(n)), str(SPS4[sp]), f"{KIND4[('photo', sp)]} / {KIND4[('video', sp)]}"])
    rows.append(["<b>v4 total</b>", *[f"<b>{sum(BY4[(s, c)] for s in ('train', 'validation', 'test'))}</b>" for c in CLASSES],
                 f"<b>{len(PROV4)}</b>", f"<b>{len(SRC4)}</b>", f"{sum(v for (k, _), v in KIND4.items() if k == 'photo')} / "
                 f"{sum(v for (k, _), v in KIND4.items() if k == 'video')}"])
    S.append(table(rows, [1.0, 0.7, 0.7, 0.8, 0.7, 0.6, 0.7, 1.4],
                   f"Dataset v4 (Model A). Source leaks between splits: <b>{LEAK4}</b> of {len(SRC4)} sources."))
    rows = [["Split", "sitting", "standing", "lying down", "walking", "Total", "Sources"]]
    for sp in ("train", "validation", "test"):
        n = [BY3[(sp, c)] for c in CLASSES]
        rows.append([sp, *map(str, n), str(sum(n)), str(SPS3[sp])])
    rows.append(["<b>v3 total</b>", *[f"<b>{sum(BY3[(s, c)] for s in ('train', 'validation', 'test'))}</b>" for c in CLASSES],
                 f"<b>{len(PROV3)}</b>", f"<b>{len(SRC3)}</b>"])
    S.append(table(rows, [1.0, 0.75, 0.75, 0.85, 0.75, 0.7, 0.75],
                   f"Dataset v3 (Model B). Source leaks: <b>{LEAK3}</b>. All 37 video frames are in train; the 43 test "
                   "images are all photos."))
    lic_photo = [(k, v) for k, v in LIC4.most_common() if not k.startswith("see ")]
    n_vid_lic = sum(v for k, v in LIC4.items() if k.startswith("see "))
    top_lic = ", ".join(f"{k} {v}" for k, v in lic_photo[:6])
    rest = lic_photo[6:]
    rest_txt = ", ".join(f"{k} {v}" for k, v in rest[:3])
    assert sum(v for _, v in lic_photo) + n_vid_lic == len(PROV4)
    S.append(P(f"<b>Licences (v4).</b> {top_lic}; {sum(v for _, v in rest)} more images under {len(rest)} other CC / "
               f"public-domain variants ({rest_txt}, and {len(rest) - 3} with 1-3 images each); the {n_vid_lic} v1 video "
               "frames carry the licences of their source videos (listed in real-dataset.md). Full credit for every image is in <i>real-data-v4/provenance.json</i>; "
               "images reproduced in this report are credited in their captions."))

    S += H("4.3", "Video clips used for sequence testing", 1)
    vrows = [["Clip", "Source video (Commons), licence", "Frames", "Behaviour", "Command, target",
              "Source video's split"]]
    v4split = {s: next(iter(v)) for s, v in SRC4.items()}
    for m in MANIFEST:
        vid = CLIP_VIDEO[m["id"]]
        in3 = "v3 " + next(iter(SRC3[vid]))
        in4 = "v4 " + v4split[vid]
        vrows.append([m["id"].split("_")[0], f"{m['source_video'].rsplit('.', 1)[0]}, {m['license']}",
                      str(m["n_frames"]), m["behaviour_throughout"].replace("_", " "),
                      f"{m['command']}, {m['target_duration_s']} s", f"{in3}; {in4}"])
    S.append(table(vrows, [0.47, 3.1, 0.52, 0.76, 0.85, 1.1],
                   "The six clips (2 fps; 10-12 s, seq06 3.6 s) and the dataset split of each source video "
                   "(provenance files)."))
    S += note("<b>Overlap with training data.</b> The clips were cut from the same six Commons videos that supplied "
              "the 37 video frames of v1-v4. In v3 all six videos are in the <b>training</b> split, so Model B has "
              "seen frames of every clip's source video; in v4 the three lying-down videos (Batumi, Snoring, Panting) "
              "are in training and the other three are in the test split. Results on the clips are therefore not "
              "held-out results, for either model, and the accept rule of Model C was also chosen on these clips. "
              "The held-out numbers are the image test splits in Sections 5-7.")
    S += H("4.4", "Limitations and potential biases", 1)
    S += bullets([
        "Open stock imagery, not the group's dogs and not real training sessions; no clip contains a genuine break.",
        "Standing is show-photo heavy (stacked show dogs were the cleanest standing source); walking includes two "
        "historical gait-study frame sets (about 50 near-duplicate frames of two dogs), kept within one split.",
        "Split by source is not split by dog identity: open-data dog identity cannot be verified, so the test sets "
        "contain unseen sources, not proven unseen dogs.",
        "Four classes only: no Jump images were found, and \"Undefined\" is handled by the confidence gate.",
        "Breed, viewpoint and lighting were not balanced; dogs that are lying flat, curled up or seen from above "
        "are rarer in photos than in phone video, which shows in Model C's results (Section 7).",
    ])
    S += H("4.5", "Synthetic scenario corpus (Model B feedback)", 1)
    S.append(P("<i>model-b-scenarios/scenarios.jsonl</i>: 180 programmatic scenarios (<i>build_scenarios.py</i>, seed "
               "20261110; 12 behaviour patterns × 5 commands, targets 3-30 s), each a posture label stream with a verdict "
               "and rule-based trainer advice. Counts: SUCCESS 60 / EARLY_BREAK 80 / UNCERTAIN 40; Sit 40, Down 49, "
               "Stand 28, Walk 31, Stay 32. Every verdict was checked by running the stream through the Nov 11 state "
               "machine. These are synthetic text, not recordings, and are presented as such; the deployed Model B was "
               "<b>not</b> tuned on them. Section 10.3 shows that 61 of the Sit/Down rows encode a duration rule the "
               "app no longer uses."))

    # ================================================================ 5 Model A
    S += H("5", "Model A Specs: student fine-tuned MobileNetV3-Small")
    p1, p2 = METRICS["phase1"], METRICS["phase2"]
    spec = [["Item", "Value"],
            ["Architecture", "Keras MobileNetV3Small, ImageNet weights, include_top=False → global average pooling → "
                             f"dropout {METRICS['dropout']} → dense 4 (softmax). The backbone contains its own rescaling "
                             "layer, so the model takes raw 0-255 pixels."],
            ["Input / output", "224×224×3 float32, 0-255 / probabilities for sitting, standing, lying_down, walking"],
            ["Data", f"Dataset v4 as split in provenance.json: train {METRICS['counts']['train']}, validation "
                     f"{METRICS['counts']['validation']}, test {METRICS['counts']['test']} (test evaluated once, at the end)"],
            ["Phase 1", f"backbone frozen, Adam lr {p1['lr']}, up to {p1['max_epochs']} epochs, early stopping on val "
                        f"accuracy (patience 4, best weights restored); ran {p1['epochs_run']}"],
            ["Phase 2", f"top {p2['unfrozen_top_layers']} backbone layers unfrozen (BatchNorm frozen), Adam lr {p2['lr']}, "
                        f"up to {p2['max_epochs']} epochs (patience 3); ran {p2['epochs_run']}"],
            ["Other", f"batch {METRICS['batch_size']}, seed {METRICS['seed']}, sparse categorical cross-entropy, augmentation "
                      "flip / rotation 0.08 / zoom 0.10 / brightness 0.15, CPU only"],
            ["Export", "model_a.keras (4.3 MB) and model_a.tflite (float16, 1.95 MB); the TFLite file was checked to "
                       "load and to agree with Keras on a held-out image (tflite_check.json). The app runs this exact file."],
            ["Code", "model-a/train_model_a.py (produces every file in model-a/)"]]
    S.append(table(spec, [1.1, 5.7], "Model A specification (model-a/metrics.json, train_model_a.py)."))
    S += H("5.1", "Training, validation and test results", 1)
    a, va, l, vl = HIST["accuracy"], HIST["val_accuracy"], HIST["loss"], HIST["val_loss"]
    e1 = HIST["epochs_phase1"]
    res = [["Point", "Train accuracy", "Train loss", "Validation accuracy", "Validation loss", "Test accuracy"],
           [f"End of phase 1 (epoch {e1})", f"{a[e1-1]:.3f}", f"{l[e1-1]:.3f}", f"{va[e1-1]:.3f}", f"{vl[e1-1]:.3f}", "-"],
           [f"End of phase 2 (epoch {len(a)})", f"{a[-1]:.3f}", f"{l[-1]:.3f}", f"{va[-1]:.3f}", f"{vl[-1]:.3f}",
            f"<b>{METRICS['test_accuracy']:.3f}</b> ({N_TEST_CORRECT}/235)"]]
    S.append(table(res, [1.6, 0.9, 0.8, 1.1, 1.0, 1.4],
                   "Model A results (history.json, metrics.json). Training numbers are Keras running averages with "
                   "augmentation and dropout active."))
    S.append(fig(MA / "curves.png", 5.1, label="f_curves", caption="Model A accuracy (left) and loss (right), train vs validation, over all 23 epochs; "
                                          "the dashed line marks the start of fine-tuning (model-a/curves.png)."))
    S.append(P(f"The gap between train ({a[-1]:.3f}) and validation ({va[-1]:.3f}) accuracy at the end, with validation "
               "loss between 0.69 and 0.79 through phase 2, shows the model starting to overfit 626 images; early stopping "
               "kept the best validation weights."))
    S += H("5.2", "Metrics on the held-out test split", 1)
    pcr = [["Class", "Precision", "Recall", "F1", "Support"]]
    for c in CLASSES:
        pcr.append([NICE[c], f"{PC[c]['precision']:.3f}", f"{PC[c]['recall']:.3f}", f"{PC[c]['f1']:.3f}", str(PC[c]["support"])])
    pcr.append(["<b>macro average</b>", f"{sum(PC[c]['precision'] for c in CLASSES)/4:.3f}",
                f"{sum(PC[c]['recall'] for c in CLASSES)/4:.3f}", f"<b>{MACRO_F1:.3f}</b>", "235"])
    cmrows = [["true \\ predicted", *[NICE[c] for c in CLASSES]]]
    for i, c in enumerate(CLASSES):
        cmrows.append([NICE[c], *[(f"<b>{v}</b>" if i == j else str(v)) for j, v in enumerate(CM[i])]])
    t1 = Table([[table(pcr, [1.1, 0.7, 0.6, 0.6, 0.6], raw=True, total_w=TEXT_W * 0.53),
                 img(MA / "confusion.png", width=2.75)]], colWidths=[TEXT_W * 0.56, TEXT_W * 0.44])
    t1.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    REFS["t_perclass"] = TAB_N[0] + 1
    REFS["f_cm"] = FIG_N[0] + 1
    S.append(KeepTogether([t1, P(f"<b>Table {TAB_N[0]+1}</b> (left). Per-class metrics, test split. "
                                 f"<b>Figure {FIG_N[0]+1}</b> (right). Confusion matrix, rows true, columns predicted "
                                 "(model-a/confusion.png).", CAP)]))
    TAB_N[0] += 1
    FIG_N[0] += 1
    S.append(P(f"Lying down is the most reliable class (F1 {PC['lying_down']['f1']:.3f}). The errors are the postures "
               f"that look alike in one frame: standing called sitting ({CM[1][0]}) or walking ({CM[1][3]}); walking called "
               f"sitting ({CM[3][0]}) or standing ({CM[3][1]}). Sitting has high recall but low precision because other "
               f"postures fall into it. The proposal's target of macro-F1 ≥ 0.80 was <b>missed</b> ({MACRO_F1:.3f})."))
    S += H("5.3", "Successful examples", 1)
    succ = [("ph_presto808_sitting_001.jpg", "sitting", 0.999), ("ph_adventurous36_standing_002.jpg", "standing", 0.999),
            ("ph_nicole_qowens_lying_down_001.jpg", "lying_down", 1.000), ("ph_chyah_walking_002.jpg", "walking", 1.000)]
    cells = []
    for f, c, conf in succ:
        r = PROV4_BY_FILE[f]
        cells.append((img(GF / r["dest"], height=1.15),
                      f"true {NICE[c]} → <b>{NICE[c]}</b> {conf:.3f}<br/>{credit(f)}"))
    S.append(image_row(cells, label="f_asucc", caption="Model A successes on held-out v4 test photos (model-a/sample_predictions.md #1, #4, #7, #8)."))
    S += H("5.4", "Failure examples and why they happen", 1)
    fail = [("ph_jason_nlw_sitting_001.jpg", "sitting", "walking", 0.991),
            ("ph_basile_morin_standing_001.jpg", "standing", "lying_down", 0.968),
            ("ph_wolfmann_walking_001.jpg", "walking", "sitting", 0.983)]
    cells = []
    for f, c, pr, conf in fail:
        r = PROV4_BY_FILE[f]
        cells.append((img(GF / r["dest"], height=1.15),
                      f"true {NICE[c]} → <b>{NICE[pr]}</b> {conf:.3f}<br/>{credit(f)}"))
    S.append(image_row(cells, label="f_afail", caption="Model A failures on held-out v4 test photos, kept on purpose (sample_predictions.md #3, #5, "
                              "#10). All three are confident, which the 0.65 gate cannot catch."))
    S.append(P("Why: a single frame of a dog mid-stride, standing still, or sitting upright can share almost the same "
               "silhouette; the training set has only 150-176 images per class with uneven viewpoints, and standing is "
               "dominated by show-ring photos. The same weakness appears on video: all 20 frames of the standing "
               "Africanis dog (seq01) were called sitting with confidence 0.77-0.95, and all 7 frames of the wagging "
               "dog (seq06) walking (Section 10.2, and the app screenshot in Figure " + ref("f_guifail") + ")."))
    S += H("5.5", "Model A input on the phone: whole frame vs Model C crop", 1)
    crop = [["Model A input", "Test photos (235)", "Confident ≥ 0.65", "Acc. when confident", "walking (62)",
             "Clip frames (111)", "standing clip frames (27)"],
            ["<b>whole image</b>", "<b>75.3%</b>", "85%", "80.5%", "43/62", "<b>73.9%</b>", "0/27"],
            ["Model C crop, pad 0.15", "68.5%", "80%", "75.4%", "17/62", "62.2%", "4/27"],
            ["Model C crop, pad 0.35", "70.2%", "83%", "76.3%", "26/62", "65.8%", "1/27"],
            ["Model C crop, pad 0.6", "73.2%", "86%", "78.7%", "36/62", "71.2%", "0/27"],
            ["mean of whole + crop 0.15", "74.9%", "75%", "81.2%", "36/62", "71.2%", "0/27"]]
    S.append(table(crop, [1.6, 0.95, 0.9, 0.95, 0.8, 0.9, 1.1], label="t_crop", caption=
                   "Whole frame vs crop, with the app's detector, accept rule, crop maths and Model A file "
                   "(repository 1, eval/model_a_crop_vs_full.md). Model C found the dog in 223/235 photos and 107/111 "
                   "frames; images with no dog use the whole-frame answer in every row. 75.3% here vs the stored 74.9% "
                   "is the same model with a different resize library (PIL bilinear vs TensorFlow); both are as measured."))

    # ================================================================ 6 Model B
    S += H("6", "Model B Gemini Fine-Tuning Specs")
    # values typed from gemini-finetune/results/tuned-results.md, base-results.md, submission-attempt2-draft.md
    spec = [["Item", "Value"],
            ["Base model", "gemini-3.5-flash"],
            ["Method / platform", "Supervised fine-tuning on Vertex AI, Google Cloud project cs663-dog-training, region us-central1"],
            ["Tuning job", "Job A, 1708100736908263424: created Oct 29, 2026 11:56 PM PT, finished Oct 30 4:53 PM PT (about 17 h)"],
            ["Hyperparameters", "1 epoch, learning-rate multiplier 1.0, adapter size 4"],
            ["Training data", "Dataset v3 JSONL: train 102, validation 26 (single image + fixed prompt → one label); test 43 held out"],
            ["Tuned model", "projects/917815040923/locations/us/models/83175855617998848@1"],
            ["Serving", "Only through its deployed endpoint projects/917815040923/locations/us/endpoints/514008491845746688 "
                        "with client location us; the model resource name returns 404 (checked in us-central1, global, us)"],
            ["Called by", "POST /api/fine-tuned-gemini on the Cloud Run backend; body {prompt, imageBase64, mimeType} → {text}"],
            ["Role in the app", "Re-checks the 2-3 decisive frames of each verdict; READY tie-break when Model A is unsure for 2 s "
                                "(at most once every 6 s)"],
            ["Code", "gemini-finetune/: build_jsonl.py, assemble_v3.py, upload_to_gcs.sh, test_tuned_model.py, evaluate.py"]]
    S.append(table(spec, [1.2, 5.6], "Model B specification (gemini-finetune/results/tuned-results.md, "
                                     "submission-attempt2-draft.md, cloud-run-backend/index.js)."))
    S.append(P("<b>What the tuning taught.</b> The proposal planned to tune Gemini on whole training attempts (command, "
               "posture timeline, 3-5 crops and a JSON verdict). What was tuned is single-image posture classification on "
               "v3, because the tuning was done as the separate Gemini fine-tuning exercise (dataset v1-v3 built Oct "
               "26-27, job run Oct 29-30) in single-image form, and it was not redone in the attempt format of the "
               "Oct 26 updated proposal. The model ID matches the proposal; the "
               "tuning data does not. The sequence reasoning in the app comes from the state machine, and the JSON "
               "contract exists only in the runtime coaching step."))
    S += H("6.1", "Training and validation", 1)
    S.append(P("The tuning job ran one epoch over the 102 training records, with the 26 validation records as its "
               "validation set. The job's training and validation loss values are not stored in either repository, so "
               "no tuning curves are shown; the evaluation of Model B is the held-out test comparison below."))
    S += H("6.2", "Test results", 1)
    S.append(P("43 held-out v3 photos (sitting 9, standing 11, walking 9, lying down 14), the exact training prompt, strict "
               "exact-match scoring (the reply must be exactly one label; a sentence counts as invalid). Measured Oct 30, "
               "2026."))
    S += H("6.3", "Base vs fine-tuned Gemini", 1)
    bt = [["Model", "Strict accuracy", "Invalid replies", "Lenient*", "sitting (9)", "standing (11)", "walking (9)", "lying down (14)"],
          ["Base gemini-3.5-flash", "37/43 = 86.0%", "3", "40/43", "8/9", "8/11", "8/9", "13/14"],
          ["<b>Fine-tuned (Model B)</b>", "<b>38/43 = 88.4%</b>", "3", "40/43", "8/9", "8/11", "8/9", "<b>14/14</b>"],
          ["Reference: base gemini-3-flash-preview", "40/43 = 93.0%", "1", "41/43", "9/9", "10/11", "7/9", "14/14"]]
    S.append(table(bt, [1.75, 1.05, 0.7, 0.6, 0.65, 0.7, 0.65, 0.8],
                   "Base vs fine-tuned Gemini on the same 43 images (gemini-finetune/results/base-results.md, "
                   "tuned-results.md). *Lenient: the right label inside a sentence counts. gemini-3-flash-preview is a "
                   "different base model (reference only) and not the tuning base; it is the model the app uses for "
                   "coaching."))
    cm = [["true \\ predicted", "sitting", "standing", "walking", "lying down", "invalid"],
          ["sitting", "8 / 8", "1 / 0", "0 / 0", "0 / 0", "0 / 1"],
          ["standing", "0 / 0", "8 / 8", "1 / 1", "0 / 0", "2 / 2"],
          ["walking", "0 / 0", "1 / 1", "8 / 8", "0 / 0", "0 / 0"],
          ["lying down", "0 / 0", "0 / 0", "0 / 0", "13 / 14", "1 / 0"]]
    S.append(table(cm, [1.3, 0.9, 0.9, 0.9, 0.9, 0.9], "Confusion matrices, base / tuned (rows true)."))
    S.append(P("The gain is one image, in lying down. Both models miss the same standing/walking pair, and with lenient "
               "scoring they tie at 40/43, so most of the strict gap is output format. One epoch reduced but did not remove "
               "sentence answers (3 remain). The tuned model also outscores Model A on images (88.4% vs 74.9%), but on "
               "different test sets of different size, so that is not a like-for-like comparison."))
    S += H("6.4", "Successful and failure examples", 1)
    ex = [("ph_doggiebreath_lying_down_001.jpg", "true lying down → <b>\"lying down\"</b> (correct)"),
          ("ph_acabashi_standing_001.jpg", "true standing → <b>\"walking\"</b> (wrong; base also wrong)"),
          ("ph_sardaka_walking_002.jpg", "true walking → <b>\"standing\"</b> (wrong)"),
          ("ph_f_standing_002.jpg", "true standing → <b>\"Based on the image, the dog is **standing**.\"</b> (invalid format)")]
    cells = [(img(GF / PROV3_BY_FILE[f]["dest"], height=1.15), f"{t}<br/>{credit(f, 3)}") for f, t in ex]
    S.append(image_row(cells, label="f_bex", caption="Fine-tuned Gemini on held-out v3 test photos: one success and three failures "
                              "(tuned-results.md spot checks)."))
    S.append(P("Why: the two wrong labels are the standing/walking pair, where one frame cannot show motion; the format "
               "failures are the base model's habit of answering in a sentence, which one epoch of 102 examples did not "
               "fully overwrite. On the video clips the tuned endpoint read 110 of 111 frames correctly, including both "
               "standing clips that defeat Model A, but its training set contains frames of all six source videos "
               "(Section 4.3), so that number does not show generalisation."))
    S += H("6.5", "Model B at runtime", 1)
    S.append(P("In the four emulator attempts that reached a verdict (Section 10.4), Model B answered every key frame and "
               "supported every verdict: VERIFIED 2/2, 2/2, 2/2 and 3/3, at 2.7-4.8 s per call. In the separate replay "
               "session shown in Figure " + ref("f_guicards") + " (also the emulator), its five calls took 3.1-8.3 s "
               "and again agreed with both verdicts. In the constructed break "
               "it said <i>standing</i> on the two break frames where Model A said <i>sitting</i>: right where Model A was "
               "wrong. The verdict was still correct (the dog left lying down either way), so the check agreed with the "
               "verdict while exposing the wrong Model A label in the log. Model B never changes a verdict; a "
               "contradiction is shown as DISPUTED."))

    # ================================================================ 7 Model C
    S += H("7", "Additional Models Specs (Model C)")
    S.append(P("Model C is a pretrained COCO object detector, used without fine-tuning, as the course allows. Two detectors "
               "were used for two purposes: <b>YOLOv8n</b> was the proposal's choice and was evaluated offline in "
               "repository 2; <b>EfficientDet-Lite0</b> is the one in the app, because MediaPipe Tasks ships it ready for "
               "Android (int8, 4.6 MB)."))
    cs = [["Item", "YOLOv8n (offline evaluation)", "EfficientDet-Lite0 (in the app)"],
          ["Weights", "yolov8n.pt, COCO (ultralytics release)", "efficientdet_lite0.tflite, COCO, int8"],
          ["Runtime", "ultralytics 8.4.210, torch 2.15.0, CPU", "MediaPipe Tasks ObjectDetector 0.10.26.1, on the phone"],
          ["Dog rule", "COCO class 16 (dog), confidence ≥ 0.25",
           "\"dog\" ≥ 0.25, or cat/horse/sheep/cow/bear ≥ 0.30 (EfficientDet often labels real dogs as other four-legged animals)"],
          ["Output used", "boxes and crops (model-c/outputs/)", "gate (dog present?), tracked box for the overlay, crop for Model B"],
          ["Code", "model-c/detect_dogs.py, results.md", "DogDetector.java; eval/model_c_recall.py (repository 1)"]]
    S.append(table(cs, [1.0, 2.6, 3.2], "Model C specification."))
    S += H("7.1", "YOLOv8n offline results", 1)
    yo = [["Set", "Images", "With ≥ 1 dog", "Detection rate", "Mean confidence"],
          ["v4 test sample (seed 42, from the 235 test photos)", "60", "49", "<b>81.7%</b>", "0.759"],
          ["Video clip frames", "111", "65", "<b>58.6%</b>", "0.667"]]
    S.append(table(yo, [2.9, 0.8, 1.0, 1.0, 1.1], "YOLOv8n detection (model-c/results.md)."))
    per = [["Clip", "seq01 standing", "seq02 lying", "seq03 walking", "seq04 lying", "seq05 lying", "seq06 standing"],
           ["Detected", "12/20", "<b>3/20</b>", "16/24", "9/20", "18/20", "7/7"]]
    S.append(table(per, [0.9, 1, 1, 1, 1, 1, 1], "YOLOv8n per clip. Photos per class: sitting 16/17, standing 10/11, walking "
                                                 "12/16, lying down 11/16 (69%)."))
    S.append(image_row(
        [(img(MC / "examples" / "ex2_standing_seq01_frame007.jpg", height=1.2), "standing dog, 0.94 (seq01 frame 7)"),
         (img(MC / "examples" / "ex3_lying_v4_yukikoko.jpg", height=1.2), "lying dog, 0.96 (v4 test photo)"),
         (img(MC / "examples" / "ex6_FAILURE_batumi_frame010_no_detection.jpg", height=1.2),
          "<b>failure</b>: lying dog on dark pavement, no box (seq02 frame 10)")],
        "YOLOv8n outputs as produced (model-c/examples/): two successes and one failure. Frames from Commons videos "
        "\"Africanis Dogs Resting In Grass\" (CC0) and \"Batumi2025dog-laying-relax\" (Vitaly Zdanevich, CC BY 4.0); photo "
        "credit: " + credit("ph_yukikoko_lying_down_001.jpg") + ".", label="f_cex"))
    S.append(P("Failure modes, from looking at every missed frame: top-down or rotated views (seq02, a portrait video of a "
               "dog flat on dark pavement, 17/20 missed); curled dogs seen from behind (seq04 frames 10-20); extreme "
               "close-ups; dogs cut off by the frame edge; small distant dogs; and plain flicker (seq01 frame 7 found at "
               "0.94, the near-identical frame 16 missed). 12 images held two dogs, so a consumer must track one box, "
               "which the app's tracker does. Gating the app on a dog-only rule (YOLOv8n 58.6%, EfficientDet dog/cat "
               "61%) would miss many frames. The app uses EfficientDet-Lite0 because MediaPipe ships it for Android, "
               "and widens the accept rule (96% on the same clips, which were also used to choose it)."))
    S += H("7.2", "EfficientDet-Lite0 (the app's detector): accept-rule recall, offline", 1)
    rec = [["Clip", "Frames", "Dog / cat only", "Widened rule (four-legged)"],
           ["seq01 standing, Africanis", "20", "15%", "95%"], ["seq02 lying, Batumi", "20", "85%", "95%"],
           ["seq03 walking, Mission", "24", "67%", "96%"], ["seq04 lying, Snoring", "20", "60%", "95%"],
           ["seq05 lying, Panting", "20", "70%", "100%"], ["seq06 standing, Wagging", "7", "86%", "100%"],
           ["<b>All</b>", "<b>111</b>", "<b>61%</b>", "<b>96%</b>"]]
    S.append(table(rec, [2.4, 0.8, 1.3, 2.3], "Share of frames in which the app's detector found the dog; every frame contains "
                                              "a dog (repository 1, eval/model_c_recall_results.md). The rule was chosen on "
                                              "these same frames, so this is a tuning result, not a held-out one."))
    S.append(P("The cost of the widened rule: a real cat or other four-legged pet in view could be taken as \"the dog\"; the "
               "tracker keeps following the first animal it locked onto."))

    # ================================================================ 8 GUI
    S += H("8", "Mobile Application GUI + Specs")
    S.append(P("One training screen (<i>SequenceActivity</i>) carries the whole flow: live preview with the tracked box and "
               "posture label, command / posture / duration pickers, START and STOP, a status line and progress bar, and the "
               "result card. A second screen (<i>Model lab</i>) runs a single photo through C and A and compares base vs "
               "tuned Gemini. The course asks that the user can start the task, see that processing is happening, receive "
               "the result and recover when the system cannot decide; Figures " + ref("f_guiflow") + "-" + ref("f_guifail")
               + " show each of these."))
    # status text under each panel is transcribed from the same screenshot
    shots = [("01_waiting_for_ready.jpg", "1. START (waiting): needs the starting posture<br/><i>\"Waiting for the "
              "starting posture: lying down (now: lying down).\"</i>"),
             ("02_attempt_running.jpg", "2. Attempt running: held 0.5 / 5.0 s<br/><i>\"Holding lying down · "
              "t = 1.0 s · 2 frames judged on the phone\"</i>"),
             ("03_holding_3s.jpg", "3. Holding: 3.0 / 5.0 s, progress bar<br/><i>\"Holding lying down · t = 3.5 s · "
              "7 frames judged on the phone\"</i>"),
             ("04_success_banner.jpg", "4. SUCCESS banner at the verdict<br/><i>\"Attempt finished: SUCCESS. Get "
              "the dog back into the starting posture to go again.\"</i>")]
    # preview (rows 88-878) + status block below the buttons; pickers and buttons left out
    status_end = {"01_waiting_for_ready.jpg": 1244, "02_attempt_running.jpg": 1215,
                  "03_holding_3s.jpg": 1215, "04_success_banner.jpg": 1244}
    flow = [(img(stacked(FIG_APP / f, [(88, 878), (1100, status_end[f])], total_h=944), width=1.62), t)
            for f, t in shots]
    S.append(image_row(flow, label="f_guiflow", caption=
                       "Attempt flow, Stay: lying down for 5 s, on the seq02 (Batumi) clip. Each panel is the camera "
                       "preview and, below the rule, the status line under the START button (pickers and buttons cut "
                       "out); the label under each panel repeats its status text. Android emulator, debug replay mode feeding the stored project frames; the fps and "
                       "per-model ms shown on the preview are emulator numbers. Box colour: green = Model A confident "
                       "(orange = unsure, yellow = detector briefly lost the dog)."))
    S.append(image_row(
        [(img(stacked(FIG_APP / "05_success_card_modelB_coaching.jpg", [(936, 1528)], total_h=675), width=3.15),
          "SUCCESS card: timing, frames judged, Model B VERIFIED 2/2, coaching tip"),
         (img(stacked(FIG_APP / "07_early_break_card_modelB.jpg", [(878, 1553)], total_h=675), width=3.15),
          "EARLY_BREAK card: first wrong posture at 1.5 s, Model B VERIFIED 3/3")],
        "Result cards (lower part of the screen; emulator). The left card is the seq02 attempt of Figure "
        + ref("f_guiflow") + "; its closing coaching note runs under the phone's navigation bar and is left out. The right "
        "card is the constructed break clip (seq05 lying frames followed by seq01 standing frames, not a real break): "
        "Model B sees <i>standing</i> on the break frames where Model A says <i>sitting</i>. These two attempts come "
        "from a separate replay session on Nov 25, so their on-device latencies and Model B call times (3.1-8.3 s) "
        "differ from Table " + ref("t_fullsys") + ".", label="f_guicards", valign="TOP"))
    S.append(KeepTogether([Table([[img(FIG_APP / "06_break_model_a_misreads_standing.jpg", height=2.45),
                                   [P("<b>Recovering when the system is unsure or wrong.</b>", BODY)] + bullets([
                                       "Unclear frames (no dog, low confidence) are shown as such and lead to UNCERTAIN, "
                                       "with a re-run tip (\"move the phone so the whole dog is in frame...\"), never a "
                                       "fake success or failure.",
                                       "If the backend is unreachable the verdict still shows, marked NOT CHECKED, with a "
                                       "built-in coaching tip.",
                                       "If Model B contradicts the verdict on a decisive frame, the card says DISPUTED.",
                                       "A long press on START overrides the READY gate (logged), for a dog the gate cannot "
                                       "confirm.",
                                       "The screenshot shows the limit of this: Model A reads the standing Africanis dog as "
                                       "<i>sitting 96%</i>. A confident wrong label passes the 0.65 gate, so a Stand "
                                       "attempt with this dog never reaches READY (Section 10.4)."])]],
                                 colWidths=[1.7 * inch, TEXT_W - 1.7 * inch],
                                 style=[("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]),
                           fig_caption("Failure on screen (emulator, break clip): Model A labels a standing dog sitting.",
                                       "f_guifail")]))
    S += H("8.1", "App specifications and code structure", 1)
    app = [["Item", "Value"],
           ["Platform", "Native Android, Java 17; package com.cs663.dogbackend, version 2.0 (versionCode 2); debug build"],
           ["Android", "minSdk 24, targetSdk 34, compileSdk 35; ABIs arm64-v8a, armeabi-v7a; 16 KB page-size aligned"],
           ["Libraries", "CameraX 1.4.2, MediaPipe tasks-vision 0.10.26.1, LiteRT 1.4.0, AndroidX; JUnit 4.13.2"],
           ["Frame loop", "one analysis thread owns all model state; ≥ 180 ms between analysed frames (≤ ~5 fps); 640×480 upright frames"],
           ["Cues", "text-to-speech and vibration at READY, GO, target reached and verdict"],
           ["Logging", "per-attempt CSV (box, both Model A readings, state machine input, per-stage latency) + JSON summary"],
           ["Network", "HTTP to the backend, 10 s connect / 25 s read timeout; each cloud reply is tagged with the attempt's "
                       "generation number so a late reply cannot land on a newer attempt's card"]]
    S.append(table(app, [1.1, 5.7], "App specification (android-app/app/build.gradle and Java sources, repository 1)."))
    code_t = [["Class", "Responsibility"],
              ["SequenceActivity", "Training screen: camera, analysis thread, READY gate, attempt life cycle, cloud checks, result card, replay mode"],
              ["VisionPipeline", "Per frame: Model C → track one dog → padded crop for Model B → Model A on the whole frame"],
              ["DogDetector / PostureClassifier", "Model C (MediaPipe, accept rule) / Model A (LiteRT, 224×224 RGB 0-255)"],
              ["SequenceSession", "Time-based state machine (Section 2.3); 21 JUnit tests in SequenceSessionTest"],
              ["GeminiClient", "Both Gemini endpoints: timeouts, error classes, mock detection, label parsing"],
              ["OverlayView, Cues, FrameLog", "Box and label overlay; speech and vibration; CSV / JSON attempt logs"],
              ["MainActivity", "Model lab: one photo through C + A on the phone, base vs tuned Gemini side by side"]]
    S.append(table(code_t, [1.7, 5.1], "Main classes (android-app/app/src/main/java/com/cs663/dogbackend/)."))

    # ================================================================ 9 backend
    S += H("9", "Cloud / Backend Specs")
    be = [["Item", "Value"],
          ["Service", "Cloud Run service cs663-gemini-service, region us-central1, " + BACKEND_URL],
          ["Stack", "Node.js, Express 4, @google/genai; JSON body limit 12 MB; listens on $PORT (8080 by default)"],
          ["Authentication to Gemini", "Vertex AI with Application Default Credentials of a dedicated service account "
                                       "(cs663-gemini-runtime, role aiplatform.user); no API key in the app"],
          ["Base model", "gemini-3-flash-preview via Vertex location global (it answers only there in this project)"],
          ["Tuned model", "Vertex endpoint .../locations/us/endpoints/514008491845746688 via location us"],
          ["Access", "--allow-unauthenticated, the group's lab choice so the app and the browser test page need no sign-in; "
                     "the Gemini credentials stay server side"],
          ["Mock mode", "MOCK_MODE=true returns stubs flagged mock:true; the app detects them and says so instead of "
                        "treating them as Gemini answers"]]
    S.append(table(be, [1.4, 5.4], "Backend specification (cloud-run-backend/index.js and deploy.sh, repository 1)."))
    ep = [["Endpoint", "Request", "Response", "Used by"],
          ["GET /health", "-", "{status, mockMode, baseModel, baseVia, tunedConfigured, tunedLocation, ...}", "checks, app"],
          ["POST /api/fine-tuned-gemini", "{prompt, imageBase64, mimeType: jpeg/png/webp}", "{text}", "Model B check, READY tie-break, Model lab"],
          ["POST /api/gemini (alias /api/base-gemini)", "same", "{text}", "Coaching, Model lab"],
          ["GET /", "-", "browser test page", "manual comparison"]]
    S.append(table(ep, [1.8, 1.9, 1.9, 1.2], "Backend endpoints. Errors return HTTP 400 with {error}; a missing tuned endpoint returns 503."))
    S.append(P("Measured from the app on the emulator (Nov 25): Model B calls 2.7-4.8 s and coaching calls 3.1-8.4 s in "
               "the four attempts of Table " + ref("t_fullsys") + "; Model B calls 3.1-8.3 s in the separate session of "
               "Figure " + ref("f_guicards") + ". Coaching JSON was valid in every logged attempt. In the Nov 11 offline run, 111 "
               "tuned calls took 384.5 s (about 3.5 s each) with 0 failed calls after retries and 0 unparseable replies. "
               "These cloud times come from the emulator and a development machine, not a phone."))

    # ================================================================ 10 testing
    S += H("10", "Testing / Evaluation")
    S.append(P("Testing runs at four levels: single models on held-out image splits (Sections 5-7), the state machine on "
               "its own, the offline video pipeline (real frames → classifier → state machine), and the full app on the "
               "emulator. Table " + ref("t_testsum") + " summarises the evidence; the subsections give the system-level results."))
    tests = [["Level", "Test", "Result", "Evidence"],
             ["Model A", "235 held-out v4 photos", f"{METRICS['test_accuracy']:.1%}, macro-F1 {MACRO_F1:.3f}", "model-a/metrics.json"],
             ["Model B", "43 held-out v3 photos", "88.4% vs base 86.0%", "gemini-finetune/results/"],
             ["Model C", "60 photos / 111 frames (YOLOv8n); 111 frames (EfficientDet)", "81.7% / 58.6%; 96%", "model-c/results.md; eval/"],
             ["State machine (Java)", "21 JUnit tests", "21/21 pass", "SequenceSessionTest (repository 1)"],
             ["State machine (Python)", "42 unit tests: original 17, the 21 Java tests ported, 4 config tests", "42/42 pass",
              "video-sequence-layer/<br/>test_state_machine.py"],
             ["Judging logic", "6 real clips with human labels", "6/6 expected verdicts (Nov 11 and app rules)", "verification_run.txt, Nov 25 JSON"],
             ["Judging logic", "6 synthetic scenarios", "6/6 under Nov 11 rules; 4/6 agree under app rules", "scenarios.json"],
             ["Video pipeline", "6 clips, 111 frames, Model A / tuned Gemini", "5/6 / 6/6 verdicts", "results/video_evaluation.md"],
             ["Full system", "5 replayed attempts, emulator", "4/5 expected; seq01 Stand never started", "repository 1 README §3"],
             ["Latency", "4 full attempts, emulator", "on-device median 94-149 ms/frame", "repository 1 README §3"]]
    S.append(table(tests, [1.05, 2.2, 1.75, 1.8], "Testing summary.", label="t_testsum"))
    S += H("10.1", "State machine: unit tests and labels mode", 1)
    S.append(P("The Java <i>SequenceSession</i> has 21 JUnit tests (smoothing, confidence gate, settle grace, response window, "
               "hold, break timing on raw frames, lost-dog and EARLY_BREAK-before-loss paths). On Nov 25 the Python reference "
               "was brought to the same rules and the 21 tests were ported one to one; with the original 17 (one expectation "
               "updated: reach time is now the first raw frame, 1.5 s instead of 2.0 s) and 4 configuration tests, 42/42 "
               "pass. Fed the human labels of the six clips, the machine gives the expected verdict on 6/6 under both the "
               "Nov 11 rules and the app's rules. The Nov 11 rules are kept frozen in <i>state_machine_nov11.py</i>; with "
               "<i>--rules nov11</i> the harness reproduces the stored Nov 11 verdict lines exactly."))
    S += H("10.2", "Offline video pipeline (Nov 11 run, re-judged Nov 25)", 1)
    S.append(P("<b>Nov 11 run.</b> The 111 stored clip frames went through Model A (the Keras model, locally) and through the live tuned "
               "endpoint, and the label streams were judged by the Nov 11 state machine (smoothing 3, break confirm 1 s, "
               "lost after 3 s, confidence 0.6). This run fed frames at t = frame index, i.e. one stored 2 fps frame per "
               "machine-second, the 1 fps rate the v1 app used; a 10 s clip therefore looked like 20 s, which makes "
               "SUCCESS easier than in real time. Both classifiers saw identical streams, so they are comparable."))
    S.append(table(seq_rows_nov11(), [1.3, 0.75, 2.6, 2.15],
                   "Nov 11 verdicts per clip (read from video_predictions_model_a.json and video_predictions_tuned.json)."))
    acc = [["Classifier", "Attempt-level (Nov 11)", "Attempt-level (Nov 25, app rules)", "Frame-level"],
           ["Model A (Keras model, run locally on CPU)",f"{VID_A['summary']['correct_verdicts']}",
            f"{NOV25['sequences_summary']['model_a']['app_matches_expected']}/6", f"{VID_A['summary']['correct_frames']} = 73.9%"],
           ["Tuned Gemini (live endpoint)", f"{VID_T['summary']['correct_verdicts']}",
            f"{NOV25['sequences_summary']['tuned']['app_matches_expected']}/6", f"{VID_T['summary']['correct_frames']} = 99.1%"],
           ["Human labels (logic check)", "6/6", f"{NOV25['sequences_summary']['ground_truth']['app_matches_expected']}/6", "n/a"]]
    S.append(table(acc, [2.2, 1.4, 1.8, 1.4], "Attempt-level and frame-level accuracy on the six clips. Model A's seq06 verdict "
                                               "is right for the wrong reason (it never recognised standing), so its clean "
                                               "count is 4 + 1. These clips are not held out (Section 4.3)."))
    S.append(P("<b>Nov 25 re-run.</b> The stored Nov 11 predictions were re-judged offline with the app's rules at real clip "
               "time (t = frame index / 2); no classifier was run and no network call was made. <b>No verdict changed.</b> "
               "The Down clips are now judged as Down (reach within 3 s, hold 0.6 s) and are decided at 1.0 s; judged as an "
               "8 s hold instead, they are SUCCESS for all three label sources. seq06's tuned hold became 3.0 s (was 6.0 s "
               "at frame-index time). Raising the confidence gate from 0.6 to 0.65 affected no frame: none of Model A's 111 "
               "confidences lies between them. No EARLY_BREAK occurred on any real clip."))
    S.append(table(seq_rows_nov25(), [0.6, 1.3, 1.2, 1.32, 1.32, 1.18],
                   "Nov 25 verdicts with the app's rules (read from video_evaluation_nov25.json). Last column: every command "
                   "judged as a hold of the manifest duration."))
    S += H("10.3", "Synthetic scenarios under the app's rules", 1)
    mb = NOV25["model_b_scenarios"]["summary"]
    sj = NOV25["scenarios_json"]["summary"]
    S.append(P(f"The scenario files were not changed. The frozen Nov 11 module still reproduces all stored verdicts "
               f"({sj['frozen_nov11_rules_reproduce_stored']}/6 and {mb['frozen_nov11_rules_reproduce_stored']}/180). Under "
               f"the app's rules, {sj['app_rules_agree_with_stored']}/6 of <i>scenarios.json</i> and "
               f"{mb['app_rules_agree_with_stored']}/180 of the Model B corpus agree. Every difference is a Sit or Down row and "
               "comes from one rule, reach-then-hold: judged with every command as a hold, agreement is 6/6 and 180/180. "
               "In the corpus, 42 EARLY_BREAK rows become SUCCESS (the dog held at least 0.6 s and the app uses no duration "
               "for Sit/Down), 11 UNCERTAIN rows become SUCCESS (the hold was done before the dog left or the clip ended), "
               "and 8 UNCERTAIN rows become EARLY_BREAK (the dog never settled within 3 s). Verdict counts move from "
               "60/80/40 to 113/46/21 (SUCCESS/EARLY_BREAK/UNCERTAIN). In <i>scenarios.json</i>, the never-sits scenario "
               "becomes EARLY_BREAK and the down-then-leaves scenario SUCCESS. Whether to regenerate the 61 Sit/Down rows "
               "is open (Section 11.4)."))
    S += H("10.4", "Full system on the Android emulator", 1)
    S.append(P("All three models, the live backend and the real UI, on the Android emulator (Apple-silicon Mac, Pixel 10 Pro "
               "system image, arm64, Android 37, 16 KB pages), with the debug replay mode feeding the project frames at "
               "their stored 2 fps. Timing rules run on clip time, so emulator slowness does not change a verdict."))
    fs = [["Clip", "Command", "Verdict", "Timing", "Model B check", "On-device ms/frame, median / p90 (emulator)"],
          ["seq02 (lying, holds)", "Stay: lying down 5 s", "<b>SUCCESS</b>", "reached 0.5 s, held 5.0 s", "VERIFIED 2/2", "94 / 166"],
          ["seq05 (lying, panting)", "Stay: lying down 5 s", "<b>SUCCESS</b>", "reached 0.5 s, held 5.0 s", "VERIFIED 2/2", "114 / 128"],
          ["seq03 (walking)", "Walk 5 s", "<b>SUCCESS</b>", "held 6.5 s", "VERIFIED 2/2", "102 / 138"],
          ["splice: seq05 lying → seq01 standing (constructed)", "Stay: lying down 5 s", "<b>EARLY_BREAK</b> at 1.5 s",
           "held 1.0 s", "VERIFIED 3/3", "149 / 335"],
          ["seq01 (standing)", "Stand", "<b>never started</b>", "READY never opened", "-", "-"]]
    S.append(table(fs, [1.6, 1.0, 1.0, 1.1, 0.85, 1.25], label="t_fullsys", caption=
                   "Full-system replay, one attempt per row, Nov 25 (repository 1 README §3; TESTING.md TC-11 to TC-15). "
                   "The splice joins frames of two real clips and is not a real break."))
    S += H("10.5", "Response time", 1)
    lat = [["Measurement", "Value", "Where"],
           ["On-device time per frame (Model C + Model A + crop)", "median 94, 102, 114, 149 ms; p90 128-335 ms (4 attempts)",
            "<b>Android emulator only</b>"],
           ["Analysed frame rate, replay", "2 fps (the clips' stored rate)", "<b>Android emulator only</b>"],
           ["Analysed frame rate, live emulator camera", "about 1.1-1.6 fps (emulator CPU load)", "<b>Android emulator only</b>"],
           ["Model B check, per call", "2.7-4.8 s in the 4 attempts of Table " + ref("t_fullsys") + "; 3.1-8.3 s in "
            "the Figure " + ref("f_guicards") + " session (after the verdict)", "emulator → Cloud Run"],
           ["Coaching, per call", "3.1-8.4 s (after the verdict)", "emulator → Cloud Run"],
           ["v1 design for comparison: tuned Gemini per frame", "about 3.5 s per frame (111 calls in 384.5 s)",
            "development machine → Cloud Run, Nov 11"]]
    S.append(table(lat, [2.4, 2.8, 1.6], "Response times. Per-frame on-device model time (C + A + crop) was under 300 ms "
                                         "(median 94-149 ms) on the Android emulator in 2 fps replay. The proposal's 5 fps "
                                         "rate and the live-loop latency have not been measured; the emulator's live camera "
                                         "reached only 1.1-1.6 fps, and no physical phone has been measured."))
    S += H("10.6", "Test cases", 1)
    tc = [["#", "Case", "Expected", "Actual"],
          ["TC-01", "Model A, sitting photo", "sitting", "sitting 0.999"],
          ["TC-02", "Model A, lying down photo", "lying down", "lying down 1.000"],
          ["TC-03", "Model A, walking mid-stride", "walking", "walking 1.000"],
          ["TC-04", "Model A, ambiguous sitting photo", "sitting", "<b>walking 0.991 (miss)</b>"],
          ["TC-05", "Model A, standing photo", "standing", "<b>lying down 0.968 (miss)</b>"],
          ["TC-06", "Tuned vs base Gemini, same 43 photos", "tuned ≥ base", "88.4% vs 86.0%"],
          ["TC-07", "YOLOv8n on 60 test photos", "dog found in most", "49/60 = 81.7%"],
          ["TC-08", "YOLOv8n, lying dog seen from above (seq02)", "dog found", "<b>3/20 frames (miss)</b>"],
          ["TC-09", "Labels mode, seq02 lying 10 s vs 8 s target (Nov 11 rules)", "SUCCESS", "SUCCESS, held 8 s"],
          ["TC-10", "Live tuned endpoint, seq06 (3.6 s) vs 8 s target", "UNCERTAIN", "UNCERTAIN, clip ended"],
          ["TC-11", "Emulator, Stay lying 5 s, seq02", "SUCCESS", "SUCCESS, VERIFIED 2/2"],
          ["TC-12", "Emulator, Stay lying 5 s, seq05", "SUCCESS", "SUCCESS, VERIFIED 2/2"],
          ["TC-13", "Emulator, Walk 5 s, seq03", "SUCCESS", "SUCCESS, VERIFIED 2/2"],
          ["TC-14", "Emulator, constructed break (seq05 → seq01)", "EARLY_BREAK", "EARLY_BREAK at 1.5 s, VERIFIED 3/3"],
          ["TC-15", "Emulator, Stand, seq01", "READY opens", "<b>never opened (miss)</b>"],
          ["TC-16", "EfficientDet accept rule, seq01", "dog found in most", "15% → 95% of frames"]]
    S.append(table(tc, [0.55, 2.9, 1.25, 2.1], "Unique test cases (TESTING.md, repository 2). TC-11 to TC-15 ran on the Android "
                                               "emulator, not a phone."))

    # ================================================================ 11 discussion
    S += H("11", "Results Discussion: Outcomes, Failures and Limitations")
    S += H("11.1", "Outcomes", 1)
    S += bullets([
        "<b>The integrated system works end to end</b> on the emulator: camera frames (replayed) → Model C → Model A → "
        "state machine → verdict, then Model B's check and base Gemini's coaching, in 4 of 5 attempts with the expected "
        "verdict, and Model B supported all 4 verdicts.",
        "<b>Per-frame inference moved onto the phone.</b> v1 sent every frame to the cloud (about 3.5 s each, 1 fps); "
        "the current app runs Models C and A on the device at a median of 94-149 ms per frame <b>on the emulator</b>, "
        "which let the timing rules follow the proposal (0.6 s holds, 3 s response window).",
        "<b>The fine-tuned Gemini beat its base</b> on held-out photos (88.4% vs 86.0%), with the gain in lying down.",
        "<b>The state machine refuses to guess:</b> when Model A never sees the target, the result is UNCERTAIN, not a "
        "false success or a false break.",
    ])
    S += H("11.2", "Where errors come from and how they propagate", 1)
    S += bullets([
        "<b>Model A causes the most failures.</b> Its 29 wrong frames on the clips are all on standing (20 in seq01, 7 in "
        "seq06) or walking (2 in seq03). A confident wrong label passes the 0.65 gate and the smoother, because there is no "
        "correct vote in the window; the result is a READY gate that never opens (TC-15) or a verdict for the wrong reason.",
        "<b>Model C misses become unclear frames</b>, not wrong labels. Short gaps are absorbed; a gap over 1.5 s ends the "
        "attempt as UNCERTAIN. The widened accept rule cut these misses from 39% to 4% of clip frames.",
        "<b>Model B can expose but not fix an error.</b> It is advisory by design; when it contradicts a decisive frame the "
        "card says DISPUTED, but the verdict stays Model A's.",
        "<b>The cloud steps cannot break the verdict.</b> Without a network the card says NOT CHECKED and shows a built-in tip.",
    ])
    S += H("11.3", "Failures and limitations", 1)
    S += bullets([
        "<b>No physical phone yet.</b> Every on-device timing and frame-rate number is from the Android emulator; real fps, "
        "lighting, camera and a live dog are untested until the demonstration. The emulator's live camera was checked only "
        "against its virtual scene with no dog.",
        "<b>Model A reads some standing dogs as sitting</b> (91-99% on the Africanis clip in the app), so a Stand attempt with "
        f"that dog cannot start. Its macro-F1 ({MACRO_F1:.3f}) misses the proposal's 0.80.",
        "<b>Model B is single-image v3 tuning</b> (171 images, 1 epoch), not the proposal's multimodal attempt tuning, and "
        "it was not retrained on v4.",
        "<b>Open data only:</b> stock images and clips, not the group's dogs; split by source, not verified dog identity. "
        "The six clips share source videos with the training data, so clip results are not held out.",
        "<b>Four classes, no Jump;</b> \"Undefined\" is the confidence gate, not a class.",
        "<b>EARLY_BREAK has only been shown on a constructed splice</b> and on synthetic scenarios and unit tests: no licensed "
        "real clip contains a genuine break.",
        "<b>Model C may accept a cat or other pet</b> as the dog because of the widened rule.",
        "<b>Debug build only</b>, not signed for release; the cloud steps need internet.",
    ])
    S += H("11.4", "Deviations from the proposal", 1)
    dev = [["Proposal (updated Oct 26)", "Built", "Reason"],
           ["Flutter app", "Native Android (Java, CameraX)", "Group decision Nov 11; the course lab app was Android"],
           ["Model C: YOLOv8 on the phone, crops to Model A", "EfficientDet-Lite0 on the phone as a gate; YOLOv8n evaluated offline",
            "MediaPipe ships EfficientDet for Android; Model A scored higher on whole frames (75.3% vs 68.5%)"],
           ["Model A: 5 classes incl. Jump, Undefined", "4 classes; Undefined = softmax < 0.65", "No Jump images found"],
           ["Model B tuned on multimodal attempts", "Single-image posture tuning on v3",
            "Done as the separate fine-tuning exercise (data Oct 26-27, job Oct 29-30); not redone in attempt format"],
           ["Gemini only after the attempt", "Yes, plus an optional READY tie-break call", "Model A can be unsure at the start"],
           ["~2,000 images, half group-collected, split by dog", "1,000 open-data images, split by source", "No group-collected dogs"],
           ["Model A macro-F1 ≥ 0.80", f"{MACRO_F1:.3f}", "Missed; reported as measured"],
           ["Temporal macro-F1 ≥ 0.85, false-break rate < 5%", "Attempt-level verdict counts instead", "No genuine break in the real clips"],
           ["Median loop < 300 ms at 5 fps", "Per-frame model time (C + A + crop) 94-149 ms median, emulator, 2 fps "
            "replay", "5 fps and the live loop not measured (emulator live camera 1.1-1.6 fps); no phone yet"],
           ["Gemini valid JSON ≥ 95%", "valid in every emulator attempt logged (the 4 of Table " + ref("t_fullsys") +
            " and the 2 of Figure " + ref("f_guicards") + ")", "Too few attempts to state a rate"]]
    S.append(table(dev, [2.1, 2.4, 2.3], "Proposal vs built (JOURNEY.md and METHODOLOGY.md §9, repository 2)."))
    S.append(P("<b>Open decisions:</b> whether to regenerate the 61 Sit/Down rows of the scenario corpus with the reach-then-hold "
               "rule (or recast them as Stay holds, which agree 180/180); and whether to retrain Model B on v4. With the "
               "app's rules as default, <i>sequence_evaluate.py --scenarios</i> now exits with status 1 because two stored "
               "scenario verdicts differ (Section 10.3)."))

    # ================================================================ 12 reproduce
    S += H("12", "Instructions for Reproducing the System")
    S.append(P("Clone both repositories side by side (the app's evaluation scripts read the vision repository at "
               "<i>../cs663-project-2-vision</i>)."))
    S += code(f"git clone {APP_URL}\ngit clone {VISION_URL}")
    S += H("12.1", "Data and models (repository 2)", 1)
    S.append(P("The datasets are committed as built (<i>real-data-v3/</i>, <i>real-data-v4/</i> with "
               "<i>provenance.json</i>, and the JSONL splits). <i>assemble_v3.py</i> and <i>assemble_v4.py</i> are the "
               "scripts that built them; they re-split from the staging folders and overwrite their outputs, so they are "
               "not needed to reproduce the results."))
    S += code("cd cs663-project-2-vision/gemini-finetune\n"
              "bash upload_to_gcs.sh YOUR_BUCKET real-data-v3 real-jsonl-v3   # images + JSONL to Cloud Storage\n"
              "#   then create the supervised tuning job in Vertex AI (base gemini-3.5-flash, 1 epoch, LR x1.0, adapter 4)\n"
              "python3 test_tuned_model.py          # fill PROJECT_ID / TUNED_MODEL first; writes predictions.jsonl\n"
              "python3 evaluate.py --predictions predictions.jsonl   # tuned vs base, per class, confusion matrix\n"
              "\n"
              "cd ../model-a && python train_model_a.py       # TensorFlow; ~7 min CPU; writes every model-a/ file\n"
              "cd ../model-c && python detect_dogs.py --set frames111=../video-sequence-layer/video-sequences \\\n"
              "    --set v4test60=sample-v4-test60 --out outputs --weights yolov8n.pt")
    S += H("12.2", "State machine and video evaluation (repository 2, standard library only unless noted)", 1)
    S += code("cd cs663-project-2-vision/video-sequence-layer\n"
              "python3 -m unittest test_state_machine                       # 42 tests\n"
              "python3 sequence_evaluate.py --all-sequences                 # labels mode, app rules: 6/6\n"
              "python3 sequence_evaluate.py --all-sequences --rules nov11   # verdicts as in results/verification_run.txt\n"
              "python3 rerun_nov25.py                                       # -> results/video_evaluation_nov25.json\n"
              "python evaluate_model_a_video.py [--rules nov11]             # needs TensorFlow\n"
              "python3 run_live_tuned_video.py [--rules nov11]              # calls the live backend, 111 calls\n"
              "cd ../gemini-finetune/model-b-scenarios && python3 build_scenarios.py   # scenario corpus, seed 20261110")
    S += H("12.3", "App, backend and app-side checks (repository 1)", 1)
    S += code("cd cs663-project-2-app/android-app && ./gradlew assembleDebug testDebugUnitTest\n"
              "adb install -r app/build/outputs/apk/debug/app-debug.apk\n"
              "\n"
              "# replay a project clip without a dog (debug builds only)\n"
              "adb push ../../cs663-project-2-vision/video-sequence-layer/video-sequences/seq02_down_batumi_frames \\\n"
              "    /data/local/tmp/dogreplay/seq02_down_batumi\n"
              "adb shell run-as com.cs663.dogbackend sh -c \\\n"
              "    \"mkdir -p files/replay && cp -r /data/local/tmp/dogreplay/seq02_down_batumi files/replay/\"\n"
              "adb shell am start -S -n com.cs663.dogbackend/.SequenceActivity --es replay seq02_down_batumi \\\n"
              "    --ei cmd 3 --ei stay 1 --ef fps 2 --ef clipFps 2 --ez autostart true\n"
              "#   cmd: 0 Sit, 1 Down, 2 Stand, 3 Stay, 4 Walk; stay: 0 sitting, 1 lying down, 2 standing\n"
              "\n"
              "cd .. && python -m venv venv && venv/bin/pip install mediapipe ai-edge-litert pillow numpy\n"
              "venv/bin/python eval/model_c_recall.py android-app/app/src/main/assets/efficientdet_lite0.tflite \\\n"
              "    ../cs663-project-2-vision/video-sequence-layer/video-sequences\n"
              "venv/bin/python eval/model_a_crop_vs_full.py android-app/app/src/main/assets/efficientdet_lite0.tflite \\\n"
              "    android-app/app/src/main/assets/model_a_posture.tflite ../cs663-project-2-vision\n"
              "\n"
              "cd cloud-run-backend && npm install && ./deploy.sh            # redeploy the backend (needs GCP access)")
    S.append(P("This report is rebuilt with <i>python docs/build_report.py</i> in repository 2 (ReportLab and Pillow). It "
               "reads the JSON result files (Model A metrics and history, provenance, Nov 11 / Nov 25 video JSON); "
               "numbers kept only in Markdown are copied in with their source named."))

    # ================================================================ 13 demo
    S += H("13", "Demonstration Video")
    S += note("<b>YouTube link: to be added.</b> The narrated demonstration has not been recorded yet.")
    S.append(P("Planned content (repository 1 README, Section 5): the app on a physical phone with a real dog that the models "
               "have not seen, using the live camera: a Stay SUCCESS, a broken Stay (EARLY_BREAK with its time), the Model B "
               "check on the result card, the coaching tip, one failure case (dog partly out of frame → UNCERTAIN), and the "
               "Model lab comparing base and fine-tuned Gemini on one photo. It will also be the first physical-phone "
               "measurement of frame rate and latency, which this report could only give for the emulator."))
    return S


def main():
    build()                      # first pass only collects figure / table numbers
    REFS_PREV.update(REFS)
    REFS.clear()
    FIG_N[0] = TAB_N[0] = 0
    def flat(items):
        for it in items:
            if isinstance(it, (list, tuple)):
                yield from flat(it)
            else:
                yield it

    def glue_headings(items):
        # A heading is kept on the same page as the flowable after it. (ReportLab's keepWithNext
        # does not group a paragraph with a following KeepTogether, so it is done here.)
        out, i = [], 0
        while i < len(items):
            if getattr(items[i], "_toc_level", None) is not None:
                j = i
                while j < len(items) and getattr(items[j], "_toc_level", None) is not None:
                    j += 1
                group = []
                for f in items[i:j + 1]:
                    group += list(f._content) if isinstance(f, KeepTogether) else [f]
                out.append(KeepTogether(group))
                i = j + 1
            else:
                out.append(items[i])
                i += 1
        return out

    doc = ReportDoc(str(OUT))
    doc.multiBuild(glue_headings(list(flat(build()))), canvasmaker=NumberedCanvas)
    tmp = DOCS / "figures" / ".build_tmp"
    if tmp.exists():
        for f in tmp.iterdir():
            f.unlink()
        tmp.rmdir()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
