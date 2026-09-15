"""A complete book font from one generated glyph sheet, with no hand mapping.

The Streamlit app needs a person to segment, merge and label glyphs, and it
scales every glyph to the same height, so a lowercase "x" or an apostrophe
comes out as tall as a capital. This module is the headless version Quill
calls: it asks an image model for a sheet laid out in known rows, reads the
glyphs back in that order, and keeps each glyph's real size and height above
the baseline.

The layout does the work. Every row starts with a capital H, so each row
carries its own baseline and cap height, and every row's glyph count is
known, so ``"`` (two marks) or ``;`` (two parts) group correctly by
splitting only at the widest gaps.

The image model is injected (``generate_image(prompt) -> bytes``), so the
caller brings its own client and credentials; ``check_labels`` optionally
lets a vision model confirm each glyph is the character it is filed under.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from myfont.character_mapping.charset import (
    BOOK_REQUIRED,
    TYPOGRAPHIC_ALIASES,
    get_glyph_name,
)

ANCHOR = "H"

# Row contents after the anchor H. Together these are exactly BOOK_REQUIRED.
SHEET_ROWS: List[str] = [
    "ABCDEFGHIJKLM",
    "NOPQRSTUVWXYZ",
    "abcdefghijklm",
    "nopqrstuvwxyz",
    "0123456789",
    ".,'!?-:;&\"()",
]

UNITS_PER_EM = 1000
CAP_HEIGHT = 700
# Space either side of a glyph, as a share of the stroke width: heavy bubble
# letters need more air than a thin serif. Clamped to sane bounds in units.
SIDE_BEARING_PER_STROKE = 0.35
SIDE_BEARING_RANGE = (25, 60)

# Glyphs whose bottom sits on the baseline; used to fit a tilted baseline and
# to snap small bounces back onto it.
ON_BASELINE = set("ABCDEFGHIKLMNPRSTUVWXYZabcdefhiklmnorstuvwxz012345789.!?:&")

SPECIMEN_TEXT = ["Ellie's Big Adventure!", "The quick brown fox jumps", "over 12 lazy dogs; (really?)"]


class SheetError(ValueError):
    """The sheet can't be read as the requested layout (retry generation)."""


@dataclass
class Glyph:
    char: str
    mask: np.ndarray          # bool, True = ink, cropped to the glyph
    left: int                 # sheet px of the crop's left edge
    top: int                  # sheet px of the crop's top edge
    baseline: Callable[[float], float]  # sheet y of the baseline at sheet x
    cap_px: float             # anchor H height in this row


@dataclass
class BookFontResult:
    ttf: bytes
    family: str
    sheet_png: bytes
    specimen_png: bytes
    missing: List[str]
    warnings: List[str] = field(default_factory=list)
    attempts: int = 1


def sheet_prompt(style: str) -> str:
    rows = "\n".join(
        f"Row {i + 1}: {' '.join([ANCHOR] + list(r))}" for i, r in enumerate(SHEET_ROWS)
    )
    return (
        "A font specimen sheet for a children's book typeface. Draw exactly these "
        "characters, in exactly this order, as six rows of text on a plain pure "
        f"white background:\n{rows}\n\n"
        f"Style: {style}.\n"
        "Every row starts with a capital H drawn at the same size, and all rows "
        "share one letter size, so lowercase letters are smaller than capitals and "
        "punctuation is small, sitting on the baseline as in normal text. "
        "Solid dark ink only, no outlines, shadows, textures, colour, gridlines, "
        "labels or decorations. Wide even spacing: no character touches another, "
        "and plenty of space between rows."
    )


# ---------------------------------------------------------------- extraction


def _ink_mask(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary > 0


def _components(ink: np.ndarray):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8
    )
    H, W = ink.shape
    comps = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue  # border junk, never a glyph
        comps.append({"id": i, "x": x, "y": y, "w": w, "h": h, "area": area})
    if not comps:
        raise SheetError("no ink found on the sheet")
    tall_h = np.percentile([c["h"] for c in comps], 90)
    typical = np.median([c["area"] for c in comps if c["h"] >= 0.6 * tall_h])
    comps = [c for c in comps if c["area"] >= 0.01 * typical]
    return labels, comps


def _group_rows(comps, expected_rows: int):
    tall_h = np.percentile([c["h"] for c in comps], 90)
    tall = sorted(
        (c for c in comps if c["h"] >= 0.6 * tall_h), key=lambda c: c["y"] + c["h"] / 2
    )
    rows: List[list] = []
    for c in tall:
        centre = c["y"] + c["h"] / 2
        if rows and centre - np.mean([r["y"] + r["h"] / 2 for r in rows[-1]]) < 0.5 * tall_h:
            rows[-1].append(c)
        else:
            rows.append([c])
    if len(rows) != expected_rows:
        raise SheetError(f"found {len(rows)} rows of text, expected {expected_rows}")
    bands = [
        (np.median([c["y"] for c in r]), np.median([c["y"] + c["h"] for c in r]))
        for r in rows
    ]
    members: List[list] = [[] for _ in bands]
    for c in comps:
        top, bottom = c["y"], c["y"] + c["h"]
        centre = (top + bottom) / 2

        def distance(band):
            b_top, b_bottom = band
            overlap = min(bottom, b_bottom) - max(top, b_top)
            if overlap > 0:
                return -overlap
            return abs(centre - (b_top + b_bottom) / 2)

        members[int(np.argmin([distance(b) for b in bands]))].append(c)
    return members


def _cluster_row(comps, labels, expected: int, row_label: str, warnings: List[str]):
    """Group a row's components into ``expected`` glyphs.

    Components that mostly share columns (the dot of an i, the halves of a
    colon) join first; then the narrowest gaps merge ("  the two marks of a
    quote) until the count is right, or, if glyphs are joined (a serif
    curling into its neighbour), the widest glyph splits at its thinnest
    column.
    """
    comps = sorted(comps, key=lambda c: c["x"])
    clusters: List[dict] = []
    for c in comps:
        if clusters:
            last = clusters[-1]
            overlap = min(last["x1"], c["x"] + c["w"]) - max(last["x0"], c["x"])
            if overlap > 0.5 * min(c["w"], last["x1"] - last["x0"]):
                last["parts"].append(c)
                last["x1"] = max(last["x1"], c["x"] + c["w"])
                continue
        clusters.append({"parts": [c], "x0": c["x"], "x1": c["x"] + c["w"], "cut": None})

    def gap(i):
        return clusters[i + 1]["x0"] - clusters[i]["x1"]

    merged_gaps = []
    while len(clusters) > expected:
        i = int(np.argmin([gap(i) for i in range(len(clusters) - 1)]))
        merged_gaps.append(gap(i))
        right = clusters.pop(i + 1)
        clusters[i]["parts"] += right["parts"]
        clusters[i]["x1"] = max(clusters[i]["x1"], right["x1"])

    while len(clusters) < expected:
        widths = [k["x1"] - k["x0"] for k in clusters]
        i = int(np.argmax(widths))
        if widths[i] < 1.4 * np.median(widths):
            raise SheetError(
                f"row {row_label!r}: found {len(clusters)} characters, expected {expected} "
                "(a character is missing)"
            )
        k = clusters[i]
        ids = [p["id"] for p in k["parts"]]
        y0 = min(p["y"] for p in k["parts"]); y1 = max(p["y"] + p["h"] for p in k["parts"])
        column_ink = np.isin(labels[y0:y1, k["x0"]:k["x1"]], ids).sum(axis=0)
        lo, hi = int(len(column_ink) * 0.3), int(len(column_ink) * 0.7)
        cut = k["x0"] + lo + int(np.argmin(column_ink[lo:hi]))
        warnings.append(f"row {row_label!r}: split joined characters")
        left = {"parts": k["parts"], "x0": k["x0"], "x1": cut, "cut": None}
        right = {"parts": k["parts"], "x0": cut, "x1": k["x1"], "cut": None}
        clusters[i:i + 1] = [left, right]

    if merged_gaps and len(clusters) > 1:
        kept = min(gap(i) for i in range(len(clusters) - 1))
        if max(merged_gaps) > 0.6 * kept:
            warnings.append(f"row {row_label!r}: spacing is uneven, check the grouping")
    return clusters


def _fit_baseline(anchor_bottom: Tuple[float, float], points: List[Tuple[float, float]], cap_px):
    xs = [anchor_bottom[0]] + [p[0] for p in points]
    ys = [anchor_bottom[1]] + [p[1] for p in points]
    if len(xs) >= 4:
        slope, intercept = np.polyfit(xs, ys, 1)
        residual = np.median(np.abs(np.polyval([slope, intercept], xs) - ys))
        if abs(slope) < 0.05 and residual < 0.06 * cap_px:
            return lambda x, s=slope, b=intercept: s * x + b
    flat = float(np.median(ys))
    return lambda x, v=flat: v


def extract_glyphs(image: np.ndarray, warnings: Optional[List[str]] = None) -> Dict[str, Glyph]:
    """Read a sheet drawn from ``sheet_prompt`` into one Glyph per character.

    ``image`` is RGB (or grayscale). Raises SheetError when the layout doesn't
    match, which usually means the model skipped or joined characters.
    """
    warnings = warnings if warnings is not None else []
    ink = _ink_mask(image)
    labels, comps = _components(ink)
    rows = _group_rows(comps, len(SHEET_ROWS))

    anchors = []
    glyphs: Dict[str, Glyph] = {}
    for row_chars, row_comps in zip(SHEET_ROWS, rows):
        clusters = _cluster_row(row_comps, labels, len(row_chars) + 1, row_chars, warnings)
        boxes = []
        for cluster in clusters:
            x0, x1 = cluster["x0"], cluster["x1"]
            ids = [k["id"] for k in cluster["parts"]]
            ink_rows = np.where(np.isin(labels[:, x0:x1], ids).any(axis=1))[0]
            y0, y1 = int(ink_rows[0]), int(ink_rows[-1]) + 1
            mask = np.isin(labels[y0:y1, x0:x1], ids)
            boxes.append((x0, y0, x1, y1, mask))

        ax0, ay0, ax1, ay1, _ = boxes[0]
        cap_px = float(ay1 - ay0)
        anchors.append(cap_px)
        on_line = [
            ((b[0] + b[2]) / 2, b[3]) for ch, b in zip(row_chars, boxes[1:]) if ch in ON_BASELINE
        ]
        baseline = _fit_baseline(((ax0 + ax1) / 2, ay1), on_line, cap_px)

        for ch, (x0, y0, x1, y1, mask) in zip(row_chars, boxes[1:]):
            glyphs[ch] = Glyph(ch, mask, x0, y0, baseline, cap_px)

    spread = max(anchors) / min(anchors)
    if spread > 1.35:
        raise SheetError(f"rows are drawn at different sizes ({spread:.2f}x apart)")
    return glyphs


# ------------------------------------------------------------------- tracing


def _trace(mask: np.ndarray, upscale: int = 2):
    """Potrace the glyph. Returns contours of ("move"|"line"|"curve", pts) in
    crop pixel coordinates (y down)."""
    import potrace  # the pure-Python port (``potracer``); no binary needed

    if upscale > 1:
        big = cv2.resize(mask.astype(np.uint8) * 255, None, fx=upscale, fy=upscale,
                         interpolation=cv2.INTER_CUBIC)
        big = cv2.GaussianBlur(big, (0, 0), upscale * 0.6) > 127
    else:
        big = mask
    padded = np.pad(big, 2)
    bitmap = potrace.Bitmap(~padded)  # potracer inverts: pass ink as False
    try:
        path = bitmap.trace(turdsize=4 * upscale * upscale, alphamax=1.0, opttolerance=0.4)
    except ValueError:
        # potracer's curve optimiser can hit a sqrt of a negative on some
        # shapes; the unoptimised curves are only slightly larger.
        path = bitmap.trace(turdsize=4 * upscale * upscale, alphamax=1.0, opticurve=False)

    def pt(p):
        return ((p.x - 2) / upscale, (p.y - 2) / upscale)

    contours = []
    for curve in path:
        ops = [("move", [pt(curve.start_point)])]
        for seg in curve:
            if seg.is_corner:
                ops.append(("line", [pt(seg.c)]))
                ops.append(("line", [pt(seg.end_point)]))
            else:
                ops.append(("curve", [pt(seg.c1), pt(seg.c2), pt(seg.end_point)]))
        contours.append(ops)
    return contours


def _signed_area(points):
    return 0.5 * sum(
        x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])
    )


# ------------------------------------------------------------------ building


def stroke_width(glyphs: Dict[str, Glyph]) -> float:
    """Typical stroke thickness in font units (twice the distance from the
    edge to the middle of a stroke, over plain letters)."""
    widths = []
    for ch in "abcdehmnopsuHNOE":
        g = glyphs.get(ch)
        if g is None:
            continue
        dist = cv2.distanceTransform(np.pad(g.mask, 1).astype(np.uint8), cv2.DIST_L2, 3)
        inside = dist[dist > 0]
        if inside.size:
            widths.append(2 * np.percentile(inside, 90) * CAP_HEIGHT / g.cap_px)
    return float(np.median(widths)) if widths else 100.0


def _glyph_outline(glyph: Glyph, side_bearing: int):
    """Trace one glyph into font units. Returns (contours, advance, bounds)."""
    scale = CAP_HEIGHT / glyph.cap_px
    height, width = glyph.mask.shape
    centre_x = glyph.left + width / 2
    base = glyph.baseline(centre_x)
    bottom = glyph.top + height
    # Hand-drawn letters bounce; put ones that belong on the line back on it.
    if glyph.char in ON_BASELINE and abs(bottom - base) < 0.06 * glyph.cap_px:
        base = bottom

    def to_units(p):
        x, y = p
        return (x * scale + side_bearing, (base - (glyph.top + y)) * scale)

    contours = []
    for ops in _trace(glyph.mask):
        contours.append([(op, [to_units(p) for p in pts]) for op, pts in ops])
    advance = int(round(width * scale + 2 * side_bearing))
    y_top = (base - glyph.top) * scale
    y_bottom = (base - bottom) * scale
    return contours, advance, (y_bottom, y_top)


def _draw(contours, pen):
    # TrueType wants outer contours clockwise (negative area with y up).
    # Potrace winds holes opposite to their outer, so each contour is judged
    # by whether it is an outer (the largest-area direction) and flipped as a
    # set.
    largest = max(contours, key=lambda ops: abs(_signed_area([p[-1] for _, p in ops])))
    flip = _signed_area([pts[-1] for _, pts in largest]) > 0
    for ops in contours:
        seq = ops
        if flip:
            seq = _reverse(ops)
        pen.moveTo(seq[0][1][0])
        for op, pts in seq[1:]:
            if op == "line":
                pen.lineTo(pts[0])
            else:
                pen.curveTo(*pts)
        pen.closePath()


def _reverse(ops):
    """Reverse a closed contour of move/line/curve ops."""
    start = ops[0][1][0]
    segments = []  # (op, control points, end point) with explicit start
    current = start
    for op, pts in ops[1:]:
        segments.append((op, current, pts))
        current = pts[-1]
    out = [("move", [current])]
    for op, seg_start, pts in reversed(segments):
        if op == "line":
            out.append(("line", [seg_start]))
        else:
            c1, c2, _ = pts
            out.append(("curve", [c2, c1, seg_start]))
    return out


def build_font(glyphs: Dict[str, Glyph], family: str) -> TTFont:
    outlines = {}
    advances = {}
    y_min, y_max = 0.0, float(CAP_HEIGHT)
    low_sb, high_sb = SIDE_BEARING_RANGE
    side_bearing = int(min(high_sb, max(low_sb, SIDE_BEARING_PER_STROKE * stroke_width(glyphs))))
    for ch, glyph in glyphs.items():
        contours, advance, (low, high) = _glyph_outline(glyph, side_bearing)
        name = get_glyph_name(ch)
        outlines[name] = contours
        advances[name] = advance
        y_min, y_max = min(y_min, low), max(y_max, high)

    n_advance = advances.get(get_glyph_name("n"), 500)
    glyph_order = [".notdef", "space"] + list(outlines)
    cmap = {32: "space", 0xA0: "space"}
    cmap.update({ord(ch): get_glyph_name(ch) for ch in glyphs})
    for fancy, plain in TYPOGRAPHIC_ALIASES.items():
        if plain in glyphs:
            cmap[ord(fancy)] = get_glyph_name(plain)

    fb = FontBuilder(UNITS_PER_EM, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)

    tt_glyphs = {}
    for name in glyph_order:
        pen = TTGlyphPen(None)
        if name == ".notdef":
            for rect in [((50, 0), (450, 700)), ((100, 650), (400, 50))]:
                (x0, y0), (x1, y1) = rect
                pen.moveTo((x0, y0)); pen.lineTo((x0, y1)); pen.lineTo((x1, y1)); pen.lineTo((x1, y0)); pen.closePath()
        elif name in outlines and outlines[name]:
            _draw(outlines[name], Cu2QuPen(pen, max_err=1.0, reverse_direction=False))
        tt_glyphs[name] = pen.glyph()
    fb.setupGlyf(tt_glyphs)

    metrics = {".notdef": (500, 50), "space": (max(220, int(n_advance * 0.55)), 0)}
    for name in outlines:
        g = tt_glyphs[name]
        g.recalcBounds(fb.font["glyf"])
        metrics[name] = (advances[name], getattr(g, "xMin", 0))
    fb.setupHorizontalMetrics(metrics)

    ascent = int(math.ceil(max(y_max, 900)))
    descent = int(math.floor(min(y_min, -250)))
    fb.setupHorizontalHeader(ascent=ascent, descent=descent)
    x_glyph = glyphs.get("x")
    x_height = CAP_HEIGHT
    if x_glyph is not None:
        x_height = int(x_glyph.mask.shape[0] * CAP_HEIGHT / x_glyph.cap_px)
    fb.setupOS2(
        sTypoAscender=ascent,
        sTypoDescender=descent,
        sTypoLineGap=0,
        usWinAscent=ascent,
        usWinDescent=-descent,
        sxHeight=x_height,
        sCapHeight=CAP_HEIGHT,
        fsSelection=0x40 | 0x80,  # REGULAR | USE_TYPO_METRICS
        version=4,
    )
    ps_name = "".join(ch for ch in family if ch.isalnum()) or "QuillFont"
    fb.setupNameTable({
        "familyName": family,
        "styleName": "Regular",
        "uniqueFontIdentifier": f"{ps_name}-Regular",
        "fullName": family,
        "psName": f"{ps_name}-Regular",
    })
    fb.setupPost()
    fb.setupHead(unitsPerEm=UNITS_PER_EM)
    fb.setupMaxp()
    return fb.font


def missing_glyphs(font: TTFont) -> List[str]:
    best = font.getBestCmap() or {}
    return [ch for ch in BOOK_REQUIRED if ord(ch) not in best]


def font_bytes(font: TTFont) -> bytes:
    buf = io.BytesIO()
    font.save(buf)
    return buf.getvalue()


def render_specimen(ttf: bytes, lines: Sequence[str] = SPECIMEN_TEXT, size: int = 64) -> bytes:
    font = ImageFont.truetype(io.BytesIO(ttf), size)
    pad = size // 2
    width = int(max(font.getlength(line) for line in lines)) + 2 * pad
    line_h = int(size * 1.35)
    img = Image.new("RGB", (width, line_h * len(lines) + 2 * pad), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((pad, pad + i * line_h), line, font=font, fill=(30, 24, 40))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def label_sheet(glyphs: Dict[str, Glyph], cell: int = 96) -> bytes:
    """A grid of the extracted glyphs, numbered in sheet order and drawn at
    their real relative size on a shared baseline (so "o" and "O" still look
    different), for a vision model to read back."""
    order = [ch for row in SHEET_ROWS for ch in row if ch in glyphs]
    cols = 13
    label_h = 20
    baseline = label_h + int(cell * 0.62)
    img = Image.new("L", (cols * cell, math.ceil(len(order) / cols) * (cell + label_h)), 255)
    draw = ImageDraw.Draw(img)
    for i, ch in enumerate(order):
        g = glyphs[ch]
        k = cell * 0.5 / g.cap_px
        h, w = g.mask.shape
        crop = Image.fromarray(np.where(g.mask, 0, 255).astype(np.uint8)).resize(
            (max(1, int(w * k)), max(1, int(h * k))), Image.LANCZOS
        )
        x, y = (i % cols) * cell, (i // cols) * (cell + label_h)
        above = (g.baseline(g.left + w / 2) - g.top) * k
        img.paste(crop, (x + (cell - crop.width) // 2, int(y + baseline - above)))
        draw.text((x + 4, y + 2), str(i + 1), fill=120)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def expected_labels() -> List[str]:
    return [ch for row in SHEET_ROWS for ch in row]


# ------------------------------------------------------------------ pipeline


def font_from_sheet(sheet: bytes, family: str) -> Tuple[TTFont, Dict[str, Glyph], List[str]]:
    image = np.array(Image.open(io.BytesIO(sheet)).convert("RGB"))
    warnings: List[str] = []
    glyphs = extract_glyphs(image, warnings)
    return build_font(glyphs, family), glyphs, warnings


def make_book_font(
    style: str,
    family: str,
    generate_image: Callable[[str], bytes],
    check_labels: Optional[Callable[[bytes, List[str]], List[str]]] = None,
    attempts: int = 3,
    log: Callable[[str], None] = lambda message: None,
) -> BookFontResult:
    """Generate sheets until one reads cleanly, then build the font.

    ``check_labels(label_sheet_png, expected)`` returns the characters a vision
    model thinks are wrong; a sheet with any is retried, and the attempt with
    the fewest problems is kept if none is perfect.
    """
    best = None
    errors = []
    for attempt in range(1, attempts + 1):
        log(f"attempt {attempt}: generating sheet")
        sheet = generate_image(sheet_prompt(style))
        try:
            font, glyphs, warnings = font_from_sheet(sheet, family)
        except SheetError as e:
            errors.append(str(e))
            log(f"attempt {attempt}: unreadable sheet: {e}")
            continue
        wrong: List[str] = []
        if check_labels is not None:
            try:
                wrong = check_labels(label_sheet(glyphs), expected_labels())
            except Exception as e:  # a failed check shouldn't sink a good sheet
                warnings.append(f"label check failed: {e}")
        candidate = (len(wrong), attempt, font, sheet, warnings, wrong)
        if best is None or candidate[0] < best[0]:
            best = candidate
        if not wrong:
            break
        log(f"attempt {attempt}: glyphs look wrong: {' '.join(wrong)}")
    if best is None:
        raise SheetError("; ".join(errors) or "no sheet generated")

    _, attempt, font, sheet, warnings, wrong = best
    if wrong:
        warnings.append("may be drawn wrong: " + " ".join(wrong))
    ttf = font_bytes(font)
    return BookFontResult(
        ttf=ttf,
        family=family,
        sheet_png=sheet,
        specimen_png=render_specimen(ttf),
        missing=missing_glyphs(font),
        warnings=warnings,
        attempts=attempt,
    )
