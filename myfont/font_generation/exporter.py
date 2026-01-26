"""Font export utilities for TTF, WOFF, and WOFF2 formats."""

import io
from typing import Dict, Optional, Tuple
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.ttLib.woff2 import compress as woff2_compress


def export_ttf(font: TTFont, output_path: str) -> str:
    """Export font as TrueType Font (TTF).

    Args:
        font: TTFont object.
        output_path: Output file path.

    Returns:
        Path to exported file.
    """
    path = Path(output_path)
    if path.suffix.lower() != '.ttf':
        path = path.with_suffix('.ttf')

    font.save(str(path))
    return str(path)


def export_woff(font: TTFont, output_path: str) -> str:
    """Export font as Web Open Font Format (WOFF).

    Args:
        font: TTFont object.
        output_path: Output file path.

    Returns:
        Path to exported file.
    """
    path = Path(output_path)
    if path.suffix.lower() != '.woff':
        path = path.with_suffix('.woff')

    font.flavor = 'woff'
    font.save(str(path))
    font.flavor = None  # Reset

    return str(path)


def export_woff2(font: TTFont, output_path: str) -> str:
    """Export font as Web Open Font Format 2 (WOFF2).

    Args:
        font: TTFont object.
        output_path: Output file path.

    Returns:
        Path to exported file.
    """
    path = Path(output_path)
    if path.suffix.lower() != '.woff2':
        path = path.with_suffix('.woff2')

    font.flavor = 'woff2'
    font.save(str(path))
    font.flavor = None  # Reset

    return str(path)


def export_svg_ot(font: TTFont, output_path: str) -> str:
    """Export font as OpenType-SVG TTF (color font).

    Args:
        font: TTFont object with SVG table.
        output_path: Output file path.

    Returns:
        Path to exported file.
    """
    path = Path(output_path)
    if path.suffix.lower() != '.ttf':
        path = path.with_suffix('.ttf')

    # Ensure no web font flavor for SVG-OT
    font.flavor = None
    font.save(str(path))

    return str(path)


def export_all(
    font: TTFont,
    output_dir: str,
    base_name: str
) -> Dict[str, str]:
    """Export font in all supported formats.

    Args:
        font: TTFont object.
        output_dir: Output directory.
        base_name: Base filename (without extension).

    Returns:
        Dictionary mapping format -> file path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # TTF
    ttf_path = output_path / f"{base_name}.ttf"
    results['ttf'] = export_ttf(font, str(ttf_path))

    # WOFF
    woff_path = output_path / f"{base_name}.woff"
    results['woff'] = export_woff(font, str(woff_path))

    # WOFF2
    woff2_path = output_path / f"{base_name}.woff2"
    results['woff2'] = export_woff2(font, str(woff2_path))

    return results


def font_to_bytes(font: TTFont, format: str = 'ttf') -> bytes:
    """Convert font to bytes.

    Args:
        font: TTFont object.
        format: Output format ('ttf', 'woff', 'woff2').

    Returns:
        Font data as bytes.
    """
    buffer = io.BytesIO()

    if format == 'woff':
        font.flavor = 'woff'
    elif format == 'woff2':
        font.flavor = 'woff2'
    else:
        font.flavor = None

    font.save(buffer)
    font.flavor = None  # Reset

    return buffer.getvalue()


def get_all_formats_bytes(font: TTFont) -> Dict[str, bytes]:
    """Get font data in all formats as bytes.

    Args:
        font: TTFont object.

    Returns:
        Dictionary mapping format -> bytes.
    """
    return {
        'ttf': font_to_bytes(font, 'ttf'),
        'woff': font_to_bytes(font, 'woff'),
        'woff2': font_to_bytes(font, 'woff2'),
    }


def get_svg_ot_bytes(font: TTFont) -> bytes:
    """Get OpenType-SVG font as bytes.

    Args:
        font: TTFont object with SVG table.

    Returns:
        Font data as bytes.
    """
    buffer = io.BytesIO()
    font.flavor = None  # Ensure TTF format
    font.save(buffer)
    return buffer.getvalue()


def is_color_font(font: TTFont) -> bool:
    """Check if font is a color font (has CBDT, COLR, or SVG table).

    Args:
        font: TTFont object.

    Returns:
        True if font has a color table.
    """
    return 'CBDT' in font or 'COLR' in font or 'SVG ' in font


def is_cbdt_font(font: TTFont) -> bool:
    """Check if font is a CBDT color bitmap font.

    Args:
        font: TTFont object.

    Returns:
        True if font has CBDT table.
    """
    return 'CBDT' in font


def get_cbdt_bytes(font: TTFont) -> bytes:
    """Get CBDT color bitmap font as bytes.

    Args:
        font: TTFont object with CBDT table.

    Returns:
        Font data as bytes.
    """
    buffer = io.BytesIO()
    font.flavor = None  # Ensure TTF format
    font.save(buffer)
    return buffer.getvalue()


def get_font_info(font: TTFont) -> Dict:
    """Get font metadata and statistics.

    Args:
        font: TTFont object.

    Returns:
        Dictionary with font information.
    """
    info = {
        'glyph_count': len(font.getGlyphOrder()),
        'units_per_em': font['head'].unitsPerEm,
    }

    # Name table info
    if 'name' in font:
        for record in font['name'].names:
            if record.nameID == 1:  # Family name
                try:
                    info['family_name'] = record.toUnicode()
                except:
                    pass
            elif record.nameID == 2:  # Style name
                try:
                    info['style_name'] = record.toUnicode()
                except:
                    pass

    # OS/2 metrics
    if 'OS/2' in font:
        os2 = font['OS/2']
        info['ascender'] = os2.sTypoAscender
        info['descender'] = os2.sTypoDescender
        info['x_height'] = os2.sxHeight if hasattr(os2, 'sxHeight') else None
        info['cap_height'] = os2.sCapHeight if hasattr(os2, 'sCapHeight') else None

    # Character coverage
    if 'cmap' in font:
        cmap = font.getBestCmap()
        if cmap:
            info['character_count'] = len(cmap)
            info['characters'] = sorted(cmap.keys())

    return info


def validate_font(font: TTFont) -> Tuple[bool, list]:
    """Basic font validation.

    Args:
        font: TTFont object.

    Returns:
        Tuple of (is_valid, list of issues).
    """
    issues = []

    # Check if this is a color font
    is_svg_font = 'SVG ' in font
    is_cbdt_font_check = 'CBDT' in font

    # Check required tables (glyf/loca are required but may be empty for color fonts)
    required_tables = ['head', 'hhea', 'maxp', 'OS/2', 'hmtx', 'cmap', 'name', 'post', 'glyf', 'loca']
    for table in required_tables:
        if table not in font:
            issues.append(f"Missing required table: {table}")

    # For SVG fonts, check SVG table
    if is_svg_font:
        svg_table = font.get('SVG ')
        if svg_table is None or not svg_table.docList:
            issues.append("SVG table is empty")

    # For CBDT fonts, check CBDT and CBLC tables
    if is_cbdt_font_check:
        if 'CBLC' not in font:
            issues.append("CBDT font missing CBLC table")
        cbdt_table = font.get('CBDT')
        if cbdt_table is None:
            issues.append("CBDT table is empty")

    # Check .notdef glyph
    glyph_order = font.getGlyphOrder()
    if not glyph_order or glyph_order[0] != '.notdef':
        issues.append("First glyph must be .notdef")

    # Check character map
    if 'cmap' in font:
        cmap = font.getBestCmap()
        if not cmap:
            issues.append("No usable cmap subtable")
    else:
        issues.append("Missing cmap table")

    return len(issues) == 0, issues
