"""Font building using FontTools."""

from typing import Dict, List, Any, Optional
import numpy as np

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

from ..config import (
    UNITS_PER_EM,
    ASCENDER,
    DESCENDER,
    CAP_HEIGHT,
    X_HEIGHT,
    LINE_GAP,
    DEFAULT_FONT_FAMILY,
    DEFAULT_FONT_VERSION,
)
from ..character_mapping.mapper import CharacterMapper
from ..character_mapping.charset import get_glyph_name
from .glyph_builder import create_glyph, create_notdef_glyph, create_space_glyph, draw_glyph_to_pen
from ..vectorization.outline_converter import draw_outline_to_pen


def build_font(
    glyph_images: Dict[str, np.ndarray],
    font_family: str = DEFAULT_FONT_FAMILY,
    style_name: str = "Regular",
    version: str = DEFAULT_FONT_VERSION,
    units_per_em: int = UNITS_PER_EM,
    font_mode: str = "monochrome",
) -> TTFont:
    """Build a complete TrueType font from glyph images.

    Args:
        glyph_images: Dictionary mapping character -> glyph image.
        font_family: Font family name.
        style_name: Style name (e.g., "Regular", "Bold").
        version: Font version string.
        units_per_em: Units per em.
        font_mode: "monochrome" for standard TTF, "color" for OpenType-SVG.

    Returns:
        TTFont object.
    """
    # Branch based on font mode
    if font_mode == "color":
        from .cbdt_builder import build_cbdt_font
        from .coverage import add_typographic_aliases
        font = build_cbdt_font(
            glyph_images,
            font_family=font_family,
            style_name=style_name,
            version=version,
            units_per_em=units_per_em,
        )
        add_typographic_aliases(font)
        return font

    # Default: monochrome pipeline
    # Create FontBuilder
    fb = FontBuilder(units_per_em, isTTF=True)

    # Build glyph data
    glyph_order = ['.notdef', 'space']
    character_map = {}
    glyph_data = {}

    # Create .notdef glyph
    glyph_data['.notdef'] = create_notdef_glyph()

    # Create space glyph
    glyph_data['space'] = create_space_glyph(int(units_per_em * 0.25))
    character_map[32] = 'space'  # ASCII space

    # Build character glyphs
    for char, image in glyph_images.items():
        glyph_name = get_glyph_name(char)

        try:
            data = create_glyph(image, glyph_name, units_per_em)
            glyph_data[glyph_name] = data
            glyph_order.append(glyph_name)
            character_map[ord(char)] = glyph_name
        except Exception as e:
            print(f"Warning: Failed to create glyph for '{char}': {e}")
            continue

    # Set glyph order
    fb.setupGlyphOrder(glyph_order)

    # Setup character map
    fb.setupCharacterMap(character_map)

    # Create glyph table
    pen_class = TTGlyphPen

    def draw_glyph(name, pen):
        data = glyph_data.get(name)
        if data and data.get('commands'):
            draw_outline_to_pen(data['commands'], pen)

    # Build glyphs using pen
    glyph_table = {}
    advance_widths = {}

    for name in glyph_order:
        pen = pen_class(None)
        draw_glyph(name, pen)

        try:
            glyph = pen.glyph()
            glyph_table[name] = glyph
        except Exception:
            # Empty glyph
            from fontTools.ttLib.tables._g_l_y_f import Glyph
            glyph_table[name] = Glyph()

        # Get advance width
        data = glyph_data.get(name, {})
        advance_widths[name] = data.get('advance_width', int(units_per_em * 0.5))

    fb.setupGlyf(glyph_table)

    # Setup metrics
    fb.setupHorizontalMetrics({
        name: (advance_widths.get(name, 500), 0)
        for name in glyph_order
    })

    # Setup head table (FontBuilder handles timestamps automatically)
    fb.setupHead(unitsPerEm=units_per_em)

    # Setup horizontal header
    fb.setupHorizontalHeader(
        ascent=ASCENDER,
        descent=DESCENDER,
    )

    # Setup maxp (maximum profile)
    fb.setupMaxp()

    # Setup OS/2 table
    fb.setupOS2(
        sTypoAscender=ASCENDER,
        sTypoDescender=DESCENDER,
        sTypoLineGap=LINE_GAP,
        usWinAscent=ASCENDER,
        usWinDescent=abs(DESCENDER),
        sxHeight=X_HEIGHT,
        sCapHeight=CAP_HEIGHT,
    )

    # Setup post table
    fb.setupPost()

    # Setup name table
    name_strings = {
        'familyName': font_family,
        'styleName': style_name,
    }
    fb.setupNameTable(name_strings)

    from .coverage import add_typographic_aliases
    add_typographic_aliases(fb.font)
    return fb.font


def build_font_from_mapper(
    mapper: CharacterMapper,
    glyph_images: Dict[int, np.ndarray],
    font_family: str = DEFAULT_FONT_FAMILY,
    **kwargs
) -> TTFont:
    """Build font using CharacterMapper for glyph assignments.

    Args:
        mapper: CharacterMapper with glyph assignments.
        glyph_images: Dictionary mapping glyph_index -> image.
        font_family: Font family name.
        **kwargs: Additional arguments for build_font.

    Returns:
        TTFont object.
    """
    # Convert mapper assignments to character -> image dict
    char_images = {}

    for mapping in mapper.get_all_mappings():
        glyph_idx = mapping.glyph_index
        char = mapping.character

        if glyph_idx in glyph_images:
            char_images[char] = glyph_images[glyph_idx]

    return build_font(char_images, font_family=font_family, **kwargs)


class FontProject:
    """Manages a font building project."""

    def __init__(
        self,
        font_family: str = DEFAULT_FONT_FAMILY,
        units_per_em: int = UNITS_PER_EM
    ):
        self.font_family = font_family
        self.units_per_em = units_per_em
        self.mapper = CharacterMapper()
        self.glyph_images: Dict[int, np.ndarray] = {}
        self.font: Optional[TTFont] = None

    def add_glyph(self, index: int, image: np.ndarray, character: Optional[str] = None):
        """Add a glyph image to the project.

        Args:
            index: Glyph index.
            image: Glyph image.
            character: Optional character to assign.
        """
        self.glyph_images[index] = image
        if character:
            self.mapper.assign(index, character)

    def build(self) -> TTFont:
        """Build the font.

        Returns:
            TTFont object.
        """
        self.font = build_font_from_mapper(
            self.mapper,
            self.glyph_images,
            self.font_family,
            units_per_em=self.units_per_em
        )
        return self.font

    def save_ttf(self, path: str):
        """Save font as TTF file.

        Args:
            path: Output file path.
        """
        if self.font is None:
            self.build()
        self.font.save(path)
