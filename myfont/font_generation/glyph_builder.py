"""Glyph creation utilities."""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.pens.ttGlyphPen import TTGlyphPen

from ..config import UNITS_PER_EM
from ..vectorization.potrace_wrapper import trace_bitmap
from ..vectorization.svg_parser import parse_svg_path, get_svg_dimensions, extract_paths_from_svg
from ..vectorization.outline_converter import (
    svg_to_glyph_outline,
    draw_outline_to_pen,
    convert_cubic_to_quadratic
)
from .metrics import estimate_glyph_metrics, scale_to_em


class GlyphBuilder:
    """Builder for creating font glyphs from images."""

    def __init__(self, units_per_em: int = UNITS_PER_EM):
        self.units_per_em = units_per_em
        self.target_height = int(units_per_em * 0.7)  # 70% of em for cap height

    def build_from_image(
        self,
        glyph_image: np.ndarray,
        glyph_name: str
    ) -> Dict[str, Any]:
        """Build a glyph from a binary image.

        Args:
            glyph_image: Binary image of the glyph.
            glyph_name: Name for the glyph.

        Returns:
            Dictionary with glyph data including outline and metrics.
        """
        # The segmented glyphs should be white (255) on black (0) background
        # Potrace traces BLACK areas, so we need to invert
        # Trace to SVG
        svg = trace_bitmap(glyph_image, invert=True)

        # Extract paths
        paths = extract_paths_from_svg(svg)
        if not paths:
            return self._create_empty_glyph(glyph_name)

        # Get SVG dimensions
        svg_width, svg_height = get_svg_dimensions(svg)

        # Combine all paths
        combined_path = ' '.join(paths)

        # Convert to glyph outline
        commands, advance_width = svg_to_glyph_outline(
            combined_path,
            glyph_height=svg_height,
            target_height=self.target_height,
            baseline_ratio=0.85
        )

        return {
            'name': glyph_name,
            'commands': commands,
            'advance_width': advance_width,
            'source_width': svg_width,
            'source_height': svg_height,
        }

    def _create_empty_glyph(self, glyph_name: str) -> Dict[str, Any]:
        """Create an empty glyph placeholder.

        Args:
            glyph_name: Name for the glyph.

        Returns:
            Dictionary with empty glyph data.
        """
        return {
            'name': glyph_name,
            'commands': [],
            'advance_width': int(self.units_per_em * 0.5),
            'source_width': 0,
            'source_height': 0,
        }


def create_glyph(
    glyph_image: np.ndarray,
    glyph_name: str,
    units_per_em: int = UNITS_PER_EM
) -> Dict[str, Any]:
    """Create a glyph from a binary image.

    Args:
        glyph_image: Binary image of the glyph.
        glyph_name: Name for the glyph.
        units_per_em: Font units per em.

    Returns:
        Dictionary with glyph data.
    """
    builder = GlyphBuilder(units_per_em)
    return builder.build_from_image(glyph_image, glyph_name)


def create_notdef_glyph(
    width: int = 500,
    height: int = 700,
    stroke_width: int = 50
) -> Dict[str, Any]:
    """Create a .notdef glyph (rectangular box).

    Args:
        width: Glyph width.
        height: Glyph height.
        stroke_width: Width of the box stroke.

    Returns:
        Dictionary with glyph data.
    """
    # Create a rectangular outline
    # Outer rectangle (clockwise)
    # Inner rectangle (counter-clockwise for hole)

    sw = stroke_width
    commands = [
        # Outer rectangle (clockwise)
        {'command': 'M', 'args': [0, 0], 'is_relative': False},
        {'command': 'L', 'args': [width, 0], 'is_relative': False},
        {'command': 'L', 'args': [width, height], 'is_relative': False},
        {'command': 'L', 'args': [0, height], 'is_relative': False},
        {'command': 'Z', 'args': [], 'is_relative': False},
        # Inner rectangle (counter-clockwise for hole)
        {'command': 'M', 'args': [sw, sw], 'is_relative': False},
        {'command': 'L', 'args': [sw, height - sw], 'is_relative': False},
        {'command': 'L', 'args': [width - sw, height - sw], 'is_relative': False},
        {'command': 'L', 'args': [width - sw, sw], 'is_relative': False},
        {'command': 'Z', 'args': [], 'is_relative': False},
    ]

    # Convert to PathCommand objects
    from ..vectorization.svg_parser import PathCommand
    path_commands = [
        PathCommand(c['command'], c['args'], c['is_relative'])
        for c in commands
    ]

    return {
        'name': '.notdef',
        'commands': path_commands,
        'advance_width': width + 100,
        'source_width': width,
        'source_height': height,
    }


def create_space_glyph(width: int = 250) -> Dict[str, Any]:
    """Create a space glyph (no outline, just advance width).

    Args:
        width: Space width.

    Returns:
        Dictionary with glyph data.
    """
    return {
        'name': 'space',
        'commands': [],
        'advance_width': width,
        'source_width': 0,
        'source_height': 0,
    }


def draw_glyph_to_pen(glyph_data: Dict[str, Any], pen) -> None:
    """Draw a glyph to a FontTools pen.

    Args:
        glyph_data: Glyph dictionary with 'commands' key.
        pen: FontTools pen object.
    """
    commands = glyph_data.get('commands', [])
    if commands:
        draw_outline_to_pen(commands, pen)
