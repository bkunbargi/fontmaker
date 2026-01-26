"""OpenType-SVG font builder for color fonts."""

import base64
import io
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.S_V_G_ import table_S_V_G_

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
from ..character_mapping.charset import get_glyph_name
from .color_glyph_builder import ColorGlyphBuilder


class SVGOTBuilder:
    """Builder for OpenType-SVG color fonts."""

    def __init__(self, units_per_em: int = UNITS_PER_EM):
        self.units_per_em = units_per_em
        self.target_height = int(units_per_em * 0.7)  # 70% of em for cap height
        self.color_glyph_builder = ColorGlyphBuilder(units_per_em)

    def build_svg_font(
        self,
        glyph_images: Dict[str, np.ndarray],
        font_family: str = DEFAULT_FONT_FAMILY,
        style_name: str = "Regular",
        version: str = DEFAULT_FONT_VERSION,
    ) -> TTFont:
        """Build OpenType-SVG font from color glyph images.

        Args:
            glyph_images: Dictionary mapping character -> color image (numpy array).
            font_family: Font family name.
            style_name: Style name (e.g., "Regular", "Bold").
            version: Font version string.

        Returns:
            TTFont object with SVG table.
        """
        # Create FontBuilder
        fb = FontBuilder(self.units_per_em, isTTF=True)

        # Build glyph data
        glyph_order = ['.notdef', 'space']
        character_map = {}
        glyph_metrics = {}
        svg_docs = []

        # Create .notdef glyph metrics
        glyph_metrics['.notdef'] = {
            'advance_width': 500,
        }

        # Create space glyph metrics
        space_width = int(self.units_per_em * 0.25)
        glyph_metrics['space'] = {
            'advance_width': space_width,
        }
        character_map[32] = 'space'  # ASCII space

        # Build character glyphs
        glyph_index = 2  # Start after .notdef and space
        print(f"DEBUG SVG: Processing {len(glyph_images)} glyph images")
        for char, image in glyph_images.items():
            glyph_name = get_glyph_name(char)
            print(f"DEBUG SVG: Processing glyph '{char}' -> {glyph_name}, image shape: {image.shape}")

            try:
                # Get metrics for the glyph
                metrics = self.color_glyph_builder.build_from_color_image(
                    image, glyph_name
                )
                glyph_metrics[glyph_name] = metrics

                # Create SVG document for this glyph
                svg_doc = self._create_svg_document(
                    image,
                    metrics['advance_width'],
                    glyph_index
                )
                svg_docs.append((svg_doc, glyph_index, glyph_index))

                glyph_order.append(glyph_name)
                character_map[ord(char)] = glyph_name
                glyph_index += 1

            except Exception as e:
                print(f"Warning: Failed to create color glyph for '{char}': {e}")
                continue

        # Set glyph order
        fb.setupGlyphOrder(glyph_order)

        # Setup character map
        fb.setupCharacterMap(character_map)

        # Create empty glyph outlines (SVG table provides the actual rendering)
        glyph_table = {}
        for name in glyph_order:
            pen = TTGlyphPen(None)
            # Create empty glyph (SVG table will render instead)
            from fontTools.ttLib.tables._g_l_y_f import Glyph
            glyph_table[name] = Glyph()

        fb.setupGlyf(glyph_table)

        # Setup metrics
        fb.setupHorizontalMetrics({
            name: (glyph_metrics.get(name, {}).get('advance_width', 500), 0)
            for name in glyph_order
        })

        # Setup head table
        fb.setupHead(unitsPerEm=self.units_per_em)

        # Setup horizontal header
        fb.setupHorizontalHeader(
            ascent=ASCENDER,
            descent=DESCENDER,
        )

        # Setup maxp
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

        # Add SVG table
        font = fb.font
        print(f"DEBUG SVG: Created {len(svg_docs)} SVG documents")
        if svg_docs:
            svg_table = table_S_V_G_()
            svg_table.docList = svg_docs
            svg_table.colorPalettes = None
            font['SVG '] = svg_table
            print(f"DEBUG SVG: Added SVG table to font")

        print(f"DEBUG SVG: Final font tables: {list(font.keys())}")
        return font

    def _create_svg_document(
        self,
        image: np.ndarray,
        advance_width: int,
        glyph_id: int
    ) -> bytes:
        """Create SVG glyph document with embedded PNG.

        Args:
            image: Color image as numpy array.
            advance_width: Glyph advance width in font units.
            glyph_id: Glyph ID for the SVG element.

        Returns:
            SVG document as bytes.
        """
        # Encode image as base64 PNG
        base64_png = self._encode_image_base64(image)

        # Get image dimensions
        if len(image.shape) == 3:
            img_height, img_width = image.shape[:2]
        else:
            img_height, img_width = image.shape

        # Calculate scaling to fit in em square
        scale = self.target_height / img_height if img_height > 0 else 1.0
        scaled_width = int(img_width * scale)
        scaled_height = int(img_height * scale)

        # Position glyph: baseline is at y=0, ascender goes up (positive)
        # For SVG, we need to flip y-axis since SVG y increases downward
        # Place the image so its bottom aligns with y=0 (baseline)
        y_offset = -scaled_height  # Negative because SVG y increases downward

        # Create SVG document
        # Using transform to handle coordinate system differences
        svg = f'''<svg id="glyph{glyph_id}" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
  <image x="0" y="{y_offset}" width="{scaled_width}" height="{scaled_height}" xlink:href="data:image/png;base64,{base64_png}"/>
</svg>'''

        # SVG table expects bytes, not string
        return svg.encode('utf-8')

    def _encode_image_base64(self, image: np.ndarray) -> str:
        """Encode numpy array image as base64 PNG.

        Args:
            image: Image as numpy array (can be grayscale, BGR, or BGRA from OpenCV).

        Returns:
            Base64-encoded PNG string.
        """
        import cv2

        # Convert numpy array to PIL Image
        if len(image.shape) == 2:
            # Grayscale
            pil_image = Image.fromarray(image, mode='L')
        elif image.shape[2] == 3:
            # BGR (OpenCV format) -> RGB (PIL format)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image, mode='RGB')
        elif image.shape[2] == 4:
            # BGRA -> RGBA
            rgba_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            pil_image = Image.fromarray(rgba_image, mode='RGBA')
        else:
            # Try to handle as grayscale
            pil_image = Image.fromarray(image[:, :, 0], mode='L')

        # Save to bytes buffer as PNG
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)

        # Encode as base64
        return base64.b64encode(buffer.getvalue()).decode('ascii')


def build_svg_font(
    glyph_images: Dict[str, np.ndarray],
    font_family: str = DEFAULT_FONT_FAMILY,
    style_name: str = "Regular",
    version: str = DEFAULT_FONT_VERSION,
    units_per_em: int = UNITS_PER_EM,
) -> TTFont:
    """Build OpenType-SVG font from color glyph images.

    Convenience function that creates an SVGOTBuilder and builds the font.

    Args:
        glyph_images: Dictionary mapping character -> color image.
        font_family: Font family name.
        style_name: Style name.
        version: Font version string.
        units_per_em: Units per em.

    Returns:
        TTFont object with SVG table.
    """
    builder = SVGOTBuilder(units_per_em)
    return builder.build_svg_font(
        glyph_images,
        font_family=font_family,
        style_name=style_name,
        version=version,
    )
