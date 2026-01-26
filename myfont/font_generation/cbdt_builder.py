"""CBDT/CBLC color bitmap font builder.

This module creates color fonts using the CBDT (Color Bitmap Data Table)
and CBLC (Color Bitmap Location Table) format, which has excellent browser
support compared to OpenType-SVG with embedded images.
"""

import io
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image
import cv2

from fontTools.fontBuilder import FontBuilder
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._g_l_y_f import Glyph
from fontTools.ttLib.tables import C_B_L_C_, C_B_D_T_
from fontTools.ttLib.tables import E_B_L_C_  # Base classes for Strike, etc.
from fontTools.ttLib.tables.BitmapGlyphMetrics import SmallGlyphMetrics

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


class CBDTBuilder:
    """Builder for CBDT/CBLC color bitmap fonts."""

    def __init__(self, units_per_em: int = UNITS_PER_EM):
        self.units_per_em = units_per_em
        self.target_height = int(units_per_em * 0.7)  # 70% of em for cap height
        self.color_glyph_builder = ColorGlyphBuilder(units_per_em)
        # PPEM (pixels per em) - determines bitmap resolution
        # Using 128 PPEM gives good quality while keeping file size reasonable
        self.ppem = 128

    def build_cbdt_font(
        self,
        glyph_images: Dict[str, np.ndarray],
        font_family: str = DEFAULT_FONT_FAMILY,
        style_name: str = "Regular",
        version: str = DEFAULT_FONT_VERSION,
    ) -> TTFont:
        """Build CBDT/CBLC color bitmap font from color glyph images.

        Args:
            glyph_images: Dictionary mapping character -> color image (numpy array).
            font_family: Font family name.
            style_name: Style name (e.g., "Regular", "Bold").
            version: Font version string.

        Returns:
            TTFont object with CBDT and CBLC tables.
        """
        # Create FontBuilder
        fb = FontBuilder(self.units_per_em, isTTF=True)

        # Build glyph data
        glyph_order = ['.notdef', 'space']
        character_map = {}
        glyph_metrics = {}
        bitmap_data = {}  # glyph_name -> (png_bytes, metrics_dict)

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
        print(f"DEBUG CBDT: Processing {len(glyph_images)} glyph images")

        for char, image in glyph_images.items():
            glyph_name = get_glyph_name(char)
            print(f"DEBUG CBDT: Processing glyph '{char}' -> {glyph_name}, image shape: {image.shape}")

            try:
                # Get metrics for the glyph
                metrics = self.color_glyph_builder.build_from_color_image(
                    image, glyph_name
                )
                glyph_metrics[glyph_name] = metrics

                # Create PNG bitmap data for this glyph
                png_bytes, bmp_metrics = self._create_bitmap_data(image, metrics)
                bitmap_data[glyph_name] = (png_bytes, bmp_metrics)

                glyph_order.append(glyph_name)
                character_map[ord(char)] = glyph_name

            except Exception as e:
                print(f"Warning: Failed to create color glyph for '{char}': {e}")
                import traceback
                traceback.print_exc()
                continue

        # Set glyph order
        fb.setupGlyphOrder(glyph_order)

        # Setup character map
        fb.setupCharacterMap(character_map)

        # Create empty glyph outlines (CBDT table provides the actual rendering)
        glyph_table = {}
        for name in glyph_order:
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

        # Get the font and add CBDT/CBLC tables
        font = fb.font

        # Build and add CBDT/CBLC tables
        if bitmap_data:
            self._add_cbdt_cblc_tables(font, bitmap_data, glyph_order)
            print(f"DEBUG CBDT: Added CBLC and CBDT tables to font")

        print(f"DEBUG CBDT: Final font tables: {list(font.keys())}")
        return font

    def _create_bitmap_data(
        self,
        image: np.ndarray,
        metrics: Dict[str, Any]
    ) -> Tuple[bytes, Dict[str, int]]:
        """Create PNG bitmap data for a glyph.

        Args:
            image: Color image as numpy array.
            metrics: Glyph metrics from color_glyph_builder.

        Returns:
            Tuple of (PNG bytes, bitmap metrics dict).
        """
        # Convert numpy array to PIL Image
        if len(image.shape) == 2:
            # Grayscale
            pil_image = Image.fromarray(image, mode='L').convert('RGBA')
        elif image.shape[2] == 3:
            # BGR (OpenCV format) -> RGB -> RGBA
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image, mode='RGB').convert('RGBA')
        elif image.shape[2] == 4:
            # BGRA -> RGBA
            rgba_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            pil_image = Image.fromarray(rgba_image, mode='RGBA')
        else:
            pil_image = Image.fromarray(image[:, :, 0], mode='L').convert('RGBA')

        # Calculate bitmap dimensions
        # Scale to fit target height at our PPEM
        scale_to_ppem = self.ppem / self.units_per_em
        target_bitmap_height = int(self.target_height * scale_to_ppem)

        # Scale image proportionally
        orig_height, orig_width = image.shape[:2]
        scale = target_bitmap_height / orig_height if orig_height > 0 else 1.0
        bitmap_width = max(1, int(orig_width * scale))
        bitmap_height = max(1, target_bitmap_height)

        # Resize the image
        pil_image = pil_image.resize((bitmap_width, bitmap_height), Image.Resampling.LANCZOS)

        # Save to PNG bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG', optimize=True)
        png_bytes = buffer.getvalue()

        # Calculate bitmap metrics (in bitmap pixels, not font units)
        # bearingX: horizontal distance from origin to left edge of bitmap
        # bearingY: vertical distance from baseline to top edge of bitmap (positive = up)
        # advance: horizontal advance after drawing glyph
        bearing_x = 0
        bearing_y = bitmap_height  # Top of bitmap is bitmap_height above baseline

        # Advance width in bitmap pixels
        advance = int(bitmap_width * 1.1)  # Add small side bearing

        bitmap_metrics = {
            'height': bitmap_height,
            'width': bitmap_width,
            'BearingX': bearing_x,
            'BearingY': bearing_y,
            'Advance': advance,
        }

        return png_bytes, bitmap_metrics

    def _add_cbdt_cblc_tables(
        self,
        font: TTFont,
        bitmap_data: Dict[str, Tuple[bytes, Dict[str, int]]],
        glyph_order: List[str]
    ):
        """Add CBLC and CBDT tables to the font.

        Args:
            font: TTFont object to add tables to.
            bitmap_data: Dict mapping glyph_name -> (png_bytes, metrics).
            glyph_order: List of glyph names in order.
        """
        # Create CBLC table
        cblc = C_B_L_C_.table_C_B_L_C_()
        cblc.version = 3.0
        cblc.strikes = []

        # Create CBDT table
        cbdt = C_B_D_T_.table_C_B_D_T_()
        cbdt.version = 3.0
        cbdt.strikeData = []

        # Get glyph names that have bitmap data (excluding .notdef and space)
        bitmap_glyph_names = [name for name in glyph_order if name in bitmap_data]

        if not bitmap_glyph_names:
            font['CBLC'] = cblc
            font['CBDT'] = cbdt
            return

        # Get glyph ID range
        first_glyph_id = font.getGlyphID(bitmap_glyph_names[0])
        last_glyph_id = font.getGlyphID(bitmap_glyph_names[-1])

        # Create a strike (set of bitmaps at a specific size)
        strike = E_B_L_C_.Strike()

        # Configure bitmap size table
        bst = strike.bitmapSizeTable
        bst.ppemX = self.ppem
        bst.ppemY = self.ppem
        bst.bitDepth = 32  # 32-bit RGBA PNG
        bst.flags = 0x01  # Horizontal metrics
        bst.colorRef = 0
        bst.startGlyphIndex = first_glyph_id
        bst.endGlyphIndex = last_glyph_id

        # Calculate line metrics from PPEM
        ppem_scale = self.ppem / self.units_per_em

        # Configure horizontal line metrics
        bst.hori = E_B_L_C_.SbitLineMetrics()
        bst.hori.ascender = int(ASCENDER * ppem_scale)
        bst.hori.descender = int(DESCENDER * ppem_scale)
        bst.hori.widthMax = max(m[1]['width'] for m in bitmap_data.values())
        bst.hori.caretSlopeNumerator = 1
        bst.hori.caretSlopeDenominator = 0
        bst.hori.caretOffset = 0
        bst.hori.minOriginSB = 0
        bst.hori.minAdvanceSB = 0
        bst.hori.maxBeforeBL = bst.hori.ascender
        bst.hori.minAfterBL = bst.hori.descender
        bst.hori.pad1 = 0
        bst.hori.pad2 = 0

        # Configure vertical line metrics
        bst.vert = E_B_L_C_.SbitLineMetrics()
        bst.vert.ascender = bst.hori.ascender
        bst.vert.descender = bst.hori.descender
        bst.vert.widthMax = bst.hori.widthMax
        bst.vert.caretSlopeNumerator = 0
        bst.vert.caretSlopeDenominator = 1
        bst.vert.caretOffset = 0
        bst.vert.minOriginSB = 0
        bst.vert.minAdvanceSB = 0
        bst.vert.maxBeforeBL = 0
        bst.vert.minAfterBL = 0
        bst.vert.pad1 = 0
        bst.vert.pad2 = 0

        # Create index subtable (format 1 - variable metrics per glyph)
        # We need to create a mock index subtable structure
        indexSubTable = _IndexSubTable1Mock(
            firstGlyphIndex=first_glyph_id,
            lastGlyphIndex=last_glyph_id,
            imageFormat=17,  # Format 17: SmallGlyphMetrics + PNG
            names=bitmap_glyph_names
        )
        strike.indexSubTables = [indexSubTable]

        cblc.strikes.append(strike)

        # Create strike data for CBDT
        strike_data = {}
        for glyph_name in bitmap_glyph_names:
            png_bytes, bmp_metrics = bitmap_data[glyph_name]

            # Create bitmap glyph object
            glyph_bitmap = _BitmapGlyph17(png_bytes, bmp_metrics)
            strike_data[glyph_name] = glyph_bitmap

        cbdt.strikeData.append(strike_data)

        # Add tables to font
        font['CBLC'] = cblc
        font['CBDT'] = cbdt


class _IndexSubTable1Mock:
    """Mock index subtable for format 1 (variable metrics)."""

    def __init__(self, firstGlyphIndex, lastGlyphIndex, imageFormat, names):
        self.firstGlyphIndex = firstGlyphIndex
        self.lastGlyphIndex = lastGlyphIndex
        self.indexFormat = 1
        self.imageFormat = imageFormat
        self.imageDataOffset = 0  # Will be calculated during compile
        self.names = names
        self.locations = []  # Will be filled during compile

    def padBitmapData(self, data):
        """Pad bitmap data (no padding needed for format 17)."""
        return data

    def compile(self, ttFont):
        """Compile index subtable to binary data."""
        # This is called by EBLC compile
        # Format 1: array of offsets
        import struct
        from fontTools.misc.textTools import bytesjoin

        # Index format 1: header + array of glyph offsets
        # Each offset is relative to imageDataOffset
        data = []

        # Header: indexFormat (2), imageFormat (2), imageDataOffset (4)
        data.append(struct.pack(">HHL", self.indexFormat, self.imageFormat, self.imageDataOffset))

        # Offset array - one offset per glyph plus sentinel
        # Offsets are calculated based on locations set by EBDT compile
        for start, end in self.locations:
            data.append(struct.pack(">L", start - self.imageDataOffset))
        # Sentinel offset (points to end of last glyph)
        if self.locations:
            data.append(struct.pack(">L", self.locations[-1][1] - self.imageDataOffset))

        return bytesjoin(data)


class _BitmapGlyph17:
    """Bitmap glyph data for format 17 (SmallGlyphMetrics + PNG)."""

    def __init__(self, png_bytes: bytes, metrics: Dict[str, int]):
        self.imageData = png_bytes
        self.metrics = SmallGlyphMetrics()
        self.metrics.height = metrics['height']
        self.metrics.width = metrics['width']
        self.metrics.BearingX = metrics['BearingX']
        self.metrics.BearingY = metrics['BearingY']
        self.metrics.Advance = metrics['Advance']

    def compile(self, ttFont):
        """Compile bitmap glyph to binary data."""
        import struct
        from fontTools.misc import sstruct
        from fontTools.misc.textTools import bytesjoin
        from fontTools.ttLib.tables.BitmapGlyphMetrics import smallGlyphMetricsFormat

        data = []
        # SmallGlyphMetrics (5 bytes)
        data.append(sstruct.pack(smallGlyphMetricsFormat, self.metrics))
        # Data length (4 bytes, big endian)
        data.append(struct.pack(">L", len(self.imageData)))
        # PNG data
        data.append(self.imageData)
        return bytesjoin(data)


def build_cbdt_font(
    glyph_images: Dict[str, np.ndarray],
    font_family: str = DEFAULT_FONT_FAMILY,
    style_name: str = "Regular",
    version: str = DEFAULT_FONT_VERSION,
    units_per_em: int = UNITS_PER_EM,
) -> TTFont:
    """Build CBDT/CBLC color bitmap font from color glyph images.

    Convenience function that creates a CBDTBuilder and builds the font.

    Args:
        glyph_images: Dictionary mapping character -> color image.
        font_family: Font family name.
        style_name: Style name.
        version: Font version string.
        units_per_em: Units per em.

    Returns:
        TTFont object with CBDT and CBLC tables.
    """
    builder = CBDTBuilder(units_per_em)
    return builder.build_cbdt_font(
        glyph_images,
        font_family=font_family,
        style_name=style_name,
        version=version,
    )
