"""Color glyph builder for extracting metrics from color images."""

from typing import Dict, Any
import numpy as np

from ..config import UNITS_PER_EM


class ColorGlyphBuilder:
    """Builder for extracting metrics from color glyph images."""

    def __init__(self, units_per_em: int = UNITS_PER_EM):
        self.units_per_em = units_per_em
        self.target_height = int(units_per_em * 0.7)  # 70% of em for cap height

    def build_from_color_image(
        self,
        image: np.ndarray,
        glyph_name: str
    ) -> Dict[str, Any]:
        """Extract metrics from color glyph image.

        Args:
            image: Color image as numpy array.
            glyph_name: Name for the glyph.

        Returns:
            Dictionary with glyph metrics (no outline conversion needed).
        """
        # Get image dimensions
        if len(image.shape) == 3:
            height, width = image.shape[:2]
        else:
            height, width = image.shape

        # Calculate scaling factor
        scale = self.target_height / height if height > 0 else 1.0

        # Calculate advance width from scaled image width
        scaled_width = int(width * scale)
        # Add side bearings (50 units on each side)
        advance_width = scaled_width + 100

        return {
            'name': glyph_name,
            'advance_width': advance_width,
            'source_width': width,
            'source_height': height,
            'scaled_width': scaled_width,
            'scaled_height': self.target_height,
            'scale': scale,
        }

    def get_bounds_from_alpha(
        self,
        image: np.ndarray
    ) -> tuple:
        """Get tight bounds from image alpha channel or content.

        Args:
            image: Image as numpy array.

        Returns:
            Tuple of (x_min, y_min, x_max, y_max) or None if empty.
        """
        if len(image.shape) == 3 and image.shape[2] == 4:
            # Use alpha channel
            alpha = image[:, :, 3]
            nonzero = np.where(alpha > 0)
        elif len(image.shape) == 3:
            # RGB - find non-white pixels
            gray = np.mean(image, axis=2)
            nonzero = np.where(gray < 250)
        else:
            # Grayscale
            nonzero = np.where(image < 250)

        if len(nonzero[0]) == 0:
            return None

        y_min, y_max = nonzero[0].min(), nonzero[0].max()
        x_min, x_max = nonzero[1].min(), nonzero[1].max()

        return (x_min, y_min, x_max, y_max)
