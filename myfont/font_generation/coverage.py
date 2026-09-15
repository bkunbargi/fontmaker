"""Is this font complete enough to set a book in?

A generated font only contains the glyphs that were drawn, segmented and
mapped. Anything else falls back to another font when rendered, so a title
like "Ellie's Adventure" silently comes out with a stray apostrophe in a
different typeface — or, in the print renderer, a missing glyph box. Two
fonts already in Quill were exported this way: Funners has no M, 0 or
apostrophe, and NewEidFont has no lowercase at all.

``add_typographic_aliases`` points curly quotes and dashes at the plain
glyphs, because people type those without meaning to. ``check_coverage``
lists what is still missing against the book set.
"""

from typing import Iterable, List

from fontTools.ttLib import TTFont

from myfont.character_mapping.charset import BOOK_REQUIRED, TYPOGRAPHIC_ALIASES


def _cmaps(font: TTFont):
    return [t for t in font["cmap"].tables if t.isUnicode()] if "cmap" in font else []


def add_typographic_aliases(font: TTFont) -> List[str]:
    """Map ’ ‘ “ ” – — onto the glyphs for ' " -, where those exist.

    Returns the characters that were added.
    """
    added = []
    tables = _cmaps(font)
    if not tables:
        return added
    best = font.getBestCmap() or {}
    for fancy, plain in TYPOGRAPHIC_ALIASES.items():
        target = best.get(ord(plain))
        if not target or ord(fancy) in best:
            continue
        for table in tables:
            table.cmap[ord(fancy)] = target
        added.append(fancy)
    return added


def check_coverage(font: TTFont, required: Iterable[str] = BOOK_REQUIRED) -> List[str]:
    """Characters from ``required`` the font has no glyph for, in order."""
    best = font.getBestCmap() or {}
    return [ch for ch in required if ord(ch) not in best]


def describe_missing(missing: List[str]) -> str:
    """ "M, 0, ' (apostrophe)" — readable, with the invisible ones named."""
    names = {"'": "' (apostrophe)", '"': '" (quote)', ",": ", (comma)",
             ".": ". (full stop)", "-": "- (hyphen)", ";": "; (semicolon)",
             ":": ": (colon)"}
    return ", ".join(names.get(ch, ch) for ch in missing)
