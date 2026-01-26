"""Font metrics calculation utilities."""

from typing import Tuple, List, Optional
import numpy as np

from ..config import UNITS_PER_EM, ASCENDER, DESCENDER


def calculate_advance_width(
    glyph_width: float,
    left_bearing: float = 50,
    right_bearing: float = 50
) -> int:
    """Calculate advance width for a glyph.

    Args:
        glyph_width: Width of the glyph outline.
        left_bearing: Left side bearing.
        right_bearing: Right side bearing.

    Returns:
        Advance width in font units.
    """
    return int(left_bearing + glyph_width + right_bearing)


def calculate_side_bearings(
    glyph_bounds: Tuple[float, float, float, float],
    advance_width: float
) -> Tuple[int, int]:
    """Calculate left and right side bearings.

    Args:
        glyph_bounds: (xMin, yMin, xMax, yMax) of glyph.
        advance_width: Total advance width.

    Returns:
        Tuple of (left_bearing, right_bearing).
    """
    x_min, _, x_max, _ = glyph_bounds
    glyph_width = x_max - x_min

    left_bearing = int(x_min)
    right_bearing = int(advance_width - x_max)

    return left_bearing, right_bearing


def scale_to_em(
    pixel_height: int,
    target_height: float = UNITS_PER_EM * 0.7
) -> float:
    """Calculate scale factor to fit glyph in em square.

    Args:
        pixel_height: Original height in pixels.
        target_height: Target height in font units (default 70% of em).

    Returns:
        Scale factor.
    """
    if pixel_height <= 0:
        return 1.0
    return target_height / pixel_height


def calculate_baseline_offset(
    ascender: int = ASCENDER,
    cap_height_ratio: float = 0.875
) -> int:
    """Calculate baseline offset for glyph positioning.

    Args:
        ascender: Font ascender value.
        cap_height_ratio: Ratio of cap height to ascender.

    Returns:
        Baseline Y offset.
    """
    return int(ascender * cap_height_ratio)


def estimate_glyph_metrics(
    glyph_image: np.ndarray,
    scale: float = 1.0
) -> dict:
    """Estimate metrics from a glyph image.

    Args:
        glyph_image: Binary glyph image.
        scale: Scale factor to apply.

    Returns:
        Dictionary with estimated metrics.
    """
    height, width = glyph_image.shape[:2]

    # Find actual bounds (non-zero pixels)
    nonzero = np.where(glyph_image > 0)
    if len(nonzero[0]) == 0:
        return {
            'width': 0,
            'height': 0,
            'x_min': 0,
            'y_min': 0,
            'x_max': 0,
            'y_max': 0,
            'advance_width': int(200 * scale),
        }

    y_min, y_max = nonzero[0].min(), nonzero[0].max()
    x_min, x_max = nonzero[1].min(), nonzero[1].max()

    glyph_width = (x_max - x_min) * scale
    glyph_height = (y_max - y_min) * scale

    return {
        'width': int(glyph_width),
        'height': int(glyph_height),
        'x_min': int(x_min * scale),
        'y_min': int(y_min * scale),
        'x_max': int(x_max * scale),
        'y_max': int(y_max * scale),
        'advance_width': int(glyph_width + 100 * scale),  # Add side bearings
    }


def calculate_font_metrics(
    glyph_heights: List[int],
    units_per_em: int = UNITS_PER_EM
) -> dict:
    """Calculate overall font metrics from glyph data.

    Args:
        glyph_heights: List of glyph heights.
        units_per_em: Units per em.

    Returns:
        Dictionary with font metrics.
    """
    if not glyph_heights:
        return {
            'ascender': ASCENDER,
            'descender': DESCENDER,
            'line_gap': 0,
            'cap_height': int(ASCENDER * 0.875),
            'x_height': int(ASCENDER * 0.5),
        }

    max_height = max(glyph_heights)
    avg_height = sum(glyph_heights) / len(glyph_heights)

    # Estimate metrics based on glyph data
    ascender = int(max_height * 1.1)
    ascender = min(ascender, units_per_em - 100)

    return {
        'ascender': ascender,
        'descender': DESCENDER,
        'line_gap': 0,
        'cap_height': int(ascender * 0.9),
        'x_height': int(avg_height * 0.7),
    }


def calculate_monospace_width(
    glyph_widths: List[int],
    method: str = 'max'
) -> int:
    """Calculate width for monospace font.

    Args:
        glyph_widths: List of glyph widths.
        method: 'max', 'average', or 'median'.

    Returns:
        Uniform advance width.
    """
    if not glyph_widths:
        return 600

    if method == 'max':
        return max(glyph_widths)
    elif method == 'average':
        return int(sum(glyph_widths) / len(glyph_widths))
    elif method == 'median':
        sorted_widths = sorted(glyph_widths)
        mid = len(sorted_widths) // 2
        return sorted_widths[mid]
    else:
        return max(glyph_widths)
