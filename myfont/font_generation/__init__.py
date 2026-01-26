"""Font generation module for building and exporting fonts."""

from .glyph_builder import create_glyph, create_notdef_glyph, create_space_glyph
from .font_builder import build_font
from .metrics import calculate_advance_width, calculate_side_bearings
from .exporter import (
    export_ttf, export_woff, export_woff2, export_all,
    export_svg_ot, get_svg_ot_bytes, is_color_font,
)
from .svg_ot_builder import SVGOTBuilder, build_svg_font
from .color_glyph_builder import ColorGlyphBuilder

__all__ = [
    'create_glyph',
    'create_notdef_glyph',
    'create_space_glyph',
    'build_font',
    'calculate_advance_width',
    'calculate_side_bearings',
    'export_ttf',
    'export_woff',
    'export_woff2',
    'export_all',
    'export_svg_ot',
    'get_svg_ot_bytes',
    'is_color_font',
    'SVGOTBuilder',
    'build_svg_font',
    'ColorGlyphBuilder',
]
