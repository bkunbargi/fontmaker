"""book_font reads a sheet back without an image model: these draw the sheet
with Pillow's bundled typeface in the layout ``sheet_prompt`` asks for."""

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from myfont import book_font as bf


def draw_sheet(gap=0.45, size=110, squeeze=None):
    """Rows of "H" + characters. ``squeeze=(row, index)`` pushes that glyph
    into its left neighbour so the two touch."""
    font = ImageFont.load_default(size)
    line_h = int(size * 1.9)
    img = Image.new("L", (int(size * 16 * (1 + gap)), line_h * len(bf.SHEET_ROWS) + size), 255)
    draw = ImageDraw.Draw(img)
    for r, row in enumerate(bf.SHEET_ROWS):
        x = size * 0.5
        baseline = size + r * line_h
        for i, ch in enumerate(bf.ANCHOR + row):
            if squeeze == (r, i):
                x -= size * gap + font.getlength(ch) * 0.3
            draw.text((x, baseline), ch, font=font, fill=0, anchor="ls")
            x += font.getlength(ch) + size * gap
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def advance(font, ch):
    return font["hmtx"][font.getBestCmap()[ord(ch)]][0]


def bounds(font, ch):
    glyph = font["glyf"][font.getBestCmap()[ord(ch)]]
    glyph.recalcBounds(font["glyf"])
    return glyph.yMin, glyph.yMax


def test_full_book_set_with_real_proportions():
    font, glyphs, warnings = bf.font_from_sheet(draw_sheet(), "Test Sheet")
    assert bf.missing_glyphs(font) == []
    assert warnings == []
    # lowercase sits lower than capitals, descenders drop below the baseline,
    # the apostrophe floats high and the full stop is small
    assert bounds(font, "x")[1] < 0.85 * bounds(font, "H")[1]
    assert bounds(font, "g")[0] < -80
    assert bounds(font, "'")[0] > 300
    assert bounds(font, ".")[1] < 200
    assert abs(bounds(font, "H")[1] - bf.CAP_HEIGHT) < 30
    # curly quotes and dashes reuse the plain glyphs
    cmap = font.getBestCmap()
    assert cmap[0x2019] == cmap[ord("'")]
    assert cmap[0x2014] == cmap[ord("-")]
    # two-part glyphs stay whole
    assert advance(font, '"') > advance(font, "'")


def test_touching_glyphs_are_split():
    row = bf.SHEET_ROWS.index("NOPQRSTUVWXYZ")
    font, glyphs, warnings = bf.font_from_sheet(
        draw_sheet(squeeze=(row, 1 + "NOPQRSTUVWXYZ".index("S"))), "Touching"
    )
    assert bf.missing_glyphs(font) == []
    assert any("split" in w for w in warnings)
    # the cut lands between the two, not through the middle of either
    r, s_ = glyphs["R"].mask.shape[1], glyphs["S"].mask.shape[1]
    assert 0.6 < r / s_ < 1.6


def test_missing_character_is_refused():
    sheet = Image.open(io.BytesIO(draw_sheet())).convert("L")
    # blank out most of the last row: nothing to recover from
    arr = np.array(sheet)
    size, line_h = 110, int(110 * 1.9)
    baseline = size + (len(bf.SHEET_ROWS) - 1) * line_h
    arr[baseline - size: baseline + size // 2, int(arr.shape[1] * 0.2):] = 255
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    with pytest.raises(bf.SheetError):
        bf.font_from_sheet(buf.getvalue(), "Broken")


def test_make_book_font_retries_until_a_sheet_reads():
    def fake_model(prompt):
        assert "Row 6" in prompt
        return next(sheets)

    blank = io.BytesIO()
    Image.new("L", (400, 300), 255).save(blank, "PNG")
    sheets = iter([blank.getvalue(), draw_sheet()])
    result = bf.make_book_font("plain", "Retry", fake_model, attempts=2)
    assert result.attempts == 2
    assert result.missing == []
    assert result.specimen_png.startswith(b"\x89PNG")


def test_label_check_prefers_the_clean_attempt():
    calls = []

    def check(png, expected):
        calls.append(len(expected))
        return ["Q"] if len(calls) == 1 else []

    result = bf.make_book_font("plain", "Checked", lambda p: draw_sheet(), check_labels=check, attempts=3)
    assert result.attempts == 2
    assert not any("drawn wrong" in w for w in result.warnings)


def draw_pair(char, size=160):
    font = ImageFont.load_default(size)
    img = Image.new("L", (size * 7, size * 3), 255)
    draw = ImageDraw.Draw(img)
    draw.text((size * 0.5, size * 2), "H", font=font, fill=0, anchor="ls")
    draw.text((size * 2, size * 2), char, font=font, fill=0, anchor="ls")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_patch_replaces_one_glyph_at_the_right_size():
    plain, _, _ = bf.font_from_sheet(draw_sheet(), "Plain")
    patched, glyphs, _ = bf.font_from_sheet(
        draw_sheet(), "Patched", patches={"g": draw_pair("g"), "'": draw_pair("'")}
    )
    # the redraw was made at a different pixel size, yet lands at the same
    # font-unit size because it is measured against its own H
    for ch in ("g", "'"):
        lo_a, hi_a = bounds(plain, ch)
        lo_b, hi_b = bounds(patched, ch)
        assert abs(hi_a - hi_b) < 25 and abs(lo_a - lo_b) < 25
        assert abs(advance(plain, ch) - advance(patched, ch)) < 40
    assert bf.missing_glyphs(patched) == []
    assert "'" in bf.glyph_prompt("'")


def test_pair_with_extra_marks_is_refused():
    img = Image.open(io.BytesIO(draw_pair("R"))).convert("L")
    draw = ImageDraw.Draw(img)
    draw.text((640, 320), "XYZ", font=ImageFont.load_default(160), fill=0, anchor="ls")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    with pytest.raises(bf.SheetError):
        bf.extract_pair(np.array(img.convert("RGB")), "R")
