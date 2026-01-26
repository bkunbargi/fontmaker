"""Font preview rendering utilities."""

import io
import base64
import re
from typing import Dict, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os

from fontTools.ttLib import TTFont


def is_color_font(font: TTFont) -> bool:
    """Check if font is a color font (has CBDT, COLR, or SVG table)."""
    return 'CBDT' in font or 'COLR' in font or 'SVG ' in font


def is_cbdt_font(font: TTFont) -> bool:
    """Check if font is a CBDT color bitmap font."""
    return 'CBDT' in font


def render_text_preview(
    font: TTFont,
    text: str,
    font_size: int = 48,
    padding: int = 20,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Render sample text using the generated font.

    Args:
        font: TTFont object.
        text: Text to render.
        font_size: Font size in points.
        padding: Padding around text.
        bg_color: Background color (RGB).
        text_color: Text color (RGB).

    Returns:
        PIL Image with rendered text.
    """
    # Save font to temporary file for PIL
    with tempfile.NamedTemporaryFile(suffix='.ttf', delete=False) as tmp:
        font.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Load font with PIL
        pil_font = ImageFont.truetype(tmp_path, font_size)

        # Calculate text size
        dummy_img = Image.new('RGB', (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), text, font=pil_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Create image
        img_width = text_width + 2 * padding
        img_height = text_height + 2 * padding

        img = Image.new('RGB', (img_width, img_height), bg_color)
        draw = ImageDraw.Draw(img)

        # Draw text
        x = padding - bbox[0]
        y = padding - bbox[1]
        draw.text((x, y), text, font=pil_font, fill=text_color)

        return img

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass


def create_preview_image(
    font: TTFont,
    include_uppercase: bool = True,
    include_lowercase: bool = True,
    include_digits: bool = True,
    include_symbols: bool = False,
    sample_text: Optional[str] = None,
    font_size: int = 36,
    line_spacing: int = 1.5,
    width: int = 800,
) -> Image.Image:
    """Create a comprehensive font preview image.

    Args:
        font: TTFont object.
        include_uppercase: Include A-Z.
        include_lowercase: Include a-z.
        include_digits: Include 0-9.
        include_symbols: Include common symbols.
        sample_text: Optional custom sample text.
        font_size: Font size in points.
        line_spacing: Line spacing multiplier.
        width: Image width.

    Returns:
        PIL Image with font preview.
    """
    lines = []

    if include_uppercase:
        lines.append("ABCDEFGHIJKLM")
        lines.append("NOPQRSTUVWXYZ")

    if include_lowercase:
        lines.append("abcdefghijklm")
        lines.append("nopqrstuvwxyz")

    if include_digits:
        lines.append("0123456789")

    if include_symbols:
        lines.append("!@#$%^&*()_+-=")

    if sample_text:
        lines.append("")
        # Wrap long sample text
        words = sample_text.split()
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            if len(test_line) > 30:
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        if current_line:
            lines.append(current_line)

    # Save font temporarily
    with tempfile.NamedTemporaryFile(suffix='.ttf', delete=False) as tmp:
        font.save(tmp.name)
        tmp_path = tmp.name

    try:
        pil_font = ImageFont.truetype(tmp_path, font_size)

        # Calculate dimensions
        line_height = int(font_size * line_spacing)
        padding = 30
        height = len(lines) * line_height + 2 * padding

        # Create image
        img = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw lines
        y = padding
        for line in lines:
            draw.text((padding, y), line, font=pil_font, fill=(0, 0, 0))
            y += line_height

        return img

    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


def create_glyph_grid(
    font: TTFont,
    cell_size: int = 60,
    cols: int = 10,
    font_size: int = 36,
) -> Image.Image:
    """Create a grid showing all glyphs in the font.

    Args:
        font: TTFont object.
        cell_size: Size of each grid cell.
        cols: Number of columns.
        font_size: Font size for glyphs.

    Returns:
        PIL Image with glyph grid.
    """
    # Get character map
    cmap = font.getBestCmap()
    if not cmap:
        # Return empty image
        return Image.new('RGB', (200, 100), (255, 255, 255))

    chars = [chr(cp) for cp in sorted(cmap.keys()) if cp >= 32]

    if not chars:
        return Image.new('RGB', (200, 100), (255, 255, 255))

    # Calculate grid size
    rows = (len(chars) + cols - 1) // cols
    width = cols * cell_size
    height = rows * cell_size

    # Save font temporarily
    with tempfile.NamedTemporaryFile(suffix='.ttf', delete=False) as tmp:
        font.save(tmp.name)
        tmp_path = tmp.name

    try:
        pil_font = ImageFont.truetype(tmp_path, font_size)

        img = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw grid
        for i, char in enumerate(chars):
            row = i // cols
            col = i % cols

            x = col * cell_size
            y = row * cell_size

            # Draw cell border
            draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1], outline=(200, 200, 200))

            # Draw character centered
            bbox = draw.textbbox((0, 0), char, font=pil_font)
            char_width = bbox[2] - bbox[0]
            char_height = bbox[3] - bbox[1]

            char_x = x + (cell_size - char_width) // 2 - bbox[0]
            char_y = y + (cell_size - char_height) // 2 - bbox[1]

            draw.text((char_x, char_y), char, font=pil_font, fill=(0, 0, 0))

        return img

    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


def _extract_cbdt_images(font: TTFont) -> Dict[str, Image.Image]:
    """Extract glyph images from CBDT table.

    Args:
        font: TTFont object with CBDT table.

    Returns:
        Dictionary mapping glyph name to PIL Image.
    """
    glyph_images = {}

    if 'CBDT' not in font or 'CBLC' not in font:
        return glyph_images

    cbdt = font['CBDT']
    cblc = font['CBLC']

    # Iterate through strikes and extract bitmap data
    for strike_index, strike in enumerate(cblc.strikes):
        if strike_index >= len(cbdt.strikeData):
            continue

        strike_data = cbdt.strikeData[strike_index]

        for glyph_name, bitmap_data in strike_data.items():
            try:
                # Extract PNG data from bitmap
                if hasattr(bitmap_data, 'imageData') and bitmap_data.imageData:
                    img = Image.open(io.BytesIO(bitmap_data.imageData))
                    if img.mode not in ('RGB', 'RGBA'):
                        img = img.convert('RGBA')
                    glyph_images[glyph_name] = img
            except Exception:
                pass

    return glyph_images


def _extract_svg_images(font: TTFont) -> Dict[str, Image.Image]:
    """Extract glyph images from SVG table.

    Args:
        font: TTFont object with SVG table.

    Returns:
        Dictionary mapping glyph name to PIL Image.
    """
    glyph_images = {}
    glyph_order = font.getGlyphOrder()

    if 'SVG ' not in font:
        return glyph_images

    svg_table = font['SVG ']

    for svg_doc, start_gid, end_gid in svg_table.docList:
        # Decode SVG document
        if isinstance(svg_doc, bytes):
            svg_str = svg_doc.decode('utf-8')
        else:
            svg_str = svg_doc

        # Extract base64 PNG from SVG
        match = re.search(r'data:image/png;base64,([A-Za-z0-9+/=]+)', svg_str)
        if match:
            base64_data = match.group(1)
            try:
                img_data = base64.b64decode(base64_data)
                img = Image.open(io.BytesIO(img_data))
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGBA')
                for gid in range(start_gid, end_gid + 1):
                    if gid < len(glyph_order):
                        glyph_images[glyph_order[gid]] = img
            except Exception:
                pass

    return glyph_images


def render_color_font_preview(
    font: TTFont,
    text: str,
    font_size: int = 48,
    padding: int = 20,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Render preview for color fonts by extracting embedded images.

    Args:
        font: TTFont object with CBDT or SVG table.
        text: Text to render.
        font_size: Target font size in points.
        padding: Padding around text.
        bg_color: Background color (RGB).

    Returns:
        PIL Image with rendered text.
    """
    if not is_color_font(font):
        raise ValueError("Font does not have color tables (CBDT or SVG)")

    # Get character map
    cmap = font.getBestCmap()

    # Extract images from appropriate table
    if 'CBDT' in font:
        glyph_images = _extract_cbdt_images(font)
    else:
        glyph_images = _extract_svg_images(font)

    # Map characters to glyph names
    char_to_glyph = {chr(cp): glyph_name for cp, glyph_name in cmap.items()}

    # Calculate target height based on font_size (approximate)
    target_height = int(font_size * 1.2)

    # Collect images for the text
    images_to_render = []
    total_width = padding

    for char in text:
        if char == ' ':
            # Add space
            images_to_render.append(('space', None, int(font_size * 0.3)))
            total_width += int(font_size * 0.3)
        elif char in char_to_glyph:
            glyph_name = char_to_glyph[char]
            if glyph_name in glyph_images:
                img = glyph_images[glyph_name]
                # Scale image to target height
                scale = target_height / img.height
                new_width = int(img.width * scale)
                images_to_render.append((char, img, new_width))
                total_width += new_width + 2  # 2px spacing

    total_width += padding

    # Create output image
    img_height = target_height + 2 * padding
    result = Image.new('RGBA', (total_width, img_height), bg_color + (255,))

    # Render each character
    x = padding
    y = padding

    for char, img, width in images_to_render:
        if img is not None:
            # Resize image
            scale = target_height / img.height
            new_size = (int(img.width * scale), target_height)
            resized = img.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to RGBA if needed
            if resized.mode != 'RGBA':
                resized = resized.convert('RGBA')

            # Paste onto result
            result.paste(resized, (x, y), resized if resized.mode == 'RGBA' else None)

        x += width + 2

    # Convert to RGB for display
    if result.mode == 'RGBA':
        bg = Image.new('RGB', result.size, bg_color)
        bg.paste(result, mask=result.split()[3])
        return bg

    return result.convert('RGB')


def create_color_glyph_grid(
    font: TTFont,
    cell_size: int = 60,
    cols: int = 10,
) -> Image.Image:
    """Create a grid showing all glyphs in a color font.

    Args:
        font: TTFont object with CBDT or SVG table.
        cell_size: Size of each grid cell.
        cols: Number of columns.

    Returns:
        PIL Image with glyph grid.
    """
    if not is_color_font(font):
        raise ValueError("Font does not have color tables (CBDT or SVG)")

    # Get character map
    cmap = font.getBestCmap()

    if not cmap:
        return Image.new('RGB', (200, 100), (255, 255, 255))

    # Extract images from appropriate table
    if 'CBDT' in font:
        glyph_images = _extract_cbdt_images(font)
    else:
        glyph_images = _extract_svg_images(font)

    # Get characters with images
    chars = []
    for cp in sorted(cmap.keys()):
        if cp >= 32:
            glyph_name = cmap[cp]
            if glyph_name in glyph_images:
                chars.append((chr(cp), glyph_images[glyph_name]))

    if not chars:
        return Image.new('RGB', (200, 100), (255, 255, 255))

    # Calculate grid size
    rows = (len(chars) + cols - 1) // cols
    width = cols * cell_size
    height = rows * cell_size

    # Create image
    result = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(result)

    # Draw grid
    target_size = cell_size - 10  # Padding inside cell

    for i, (char, img) in enumerate(chars):
        row = i // cols
        col = i % cols

        x = col * cell_size
        y = row * cell_size

        # Draw cell border
        draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1], outline=(200, 200, 200))

        # Resize and center glyph image
        scale = min(target_size / img.width, target_size / img.height)
        new_size = (int(img.width * scale), int(img.height * scale))
        resized = img.resize(new_size, Image.Resampling.LANCZOS)

        # Center in cell
        paste_x = x + (cell_size - new_size[0]) // 2
        paste_y = y + (cell_size - new_size[1]) // 2

        # Handle transparency
        if resized.mode == 'RGBA':
            result.paste(resized, (paste_x, paste_y), resized)
        else:
            result.paste(resized, (paste_x, paste_y))

    return result


def get_css_fontface(font_name: str, woff2_base64: Optional[str] = None) -> str:
    """Generate CSS @font-face rule.

    Args:
        font_name: Font family name.
        woff2_base64: Optional base64-encoded WOFF2 data.

    Returns:
        CSS string.
    """
    if woff2_base64:
        return f"""@font-face {{
    font-family: '{font_name}';
    src: url(data:font/woff2;base64,{woff2_base64}) format('woff2');
    font-weight: normal;
    font-style: normal;
}}"""
    else:
        return f"""@font-face {{
    font-family: '{font_name}';
    src: url('{font_name}.woff2') format('woff2'),
         url('{font_name}.woff') format('woff'),
         url('{font_name}.ttf') format('truetype');
    font-weight: normal;
    font-style: normal;
}}"""


def get_html_preview(font_name: str, sample_text: str = "The quick brown fox") -> str:
    """Generate HTML preview page.

    Args:
        font_name: Font family name.
        sample_text: Sample text to display.

    Returns:
        HTML string.
    """
    css = get_css_fontface(font_name)

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{font_name} Preview</title>
    <style>
        {css}
        body {{
            font-family: sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
        }}
        .preview {{
            font-family: '{font_name}', serif;
            margin: 20px 0;
        }}
        .size-12 {{ font-size: 12px; }}
        .size-16 {{ font-size: 16px; }}
        .size-24 {{ font-size: 24px; }}
        .size-36 {{ font-size: 36px; }}
        .size-48 {{ font-size: 48px; }}
        .size-72 {{ font-size: 72px; }}
        h1 {{ color: #333; }}
        .label {{ color: #666; font-size: 12px; margin-bottom: 5px; }}
    </style>
</head>
<body>
    <h1>{font_name}</h1>

    <div class="label">72px</div>
    <div class="preview size-72">{sample_text}</div>

    <div class="label">48px</div>
    <div class="preview size-48">{sample_text}</div>

    <div class="label">36px</div>
    <div class="preview size-36">{sample_text}</div>

    <div class="label">24px</div>
    <div class="preview size-24">{sample_text}</div>

    <div class="label">16px</div>
    <div class="preview size-16">{sample_text}</div>

    <div class="label">12px</div>
    <div class="preview size-12">{sample_text}</div>

    <h2>Character Set</h2>
    <div class="preview size-24">ABCDEFGHIJKLMNOPQRSTUVWXYZ</div>
    <div class="preview size-24">abcdefghijklmnopqrstuvwxyz</div>
    <div class="preview size-24">0123456789</div>
</body>
</html>"""
