# MyFont - Local Font Generator

Convert glyph images into usable font files (TTF, WOFF, WOFF2).

## Features

- Upload a glyph sheet or individual character images
- Automatic glyph extraction via contour detection
- Interactive character-to-glyph mapping
- Vector conversion using Potrace
- Export to TTF, WOFF, and WOFF2 formats
- Live font preview

## Headless book fonts (used by Quill)

`myfont.book_font` builds a complete book font (A-Z, a-z, 0-9, punctuation)
from one AI-drawn glyph sheet with no manual segmenting or mapping. The sheet
is requested in fixed rows that each start with a capital H, which gives
every row its baseline and cap height, so lowercase, descenders and
punctuation keep their real size. Tracing uses `potracer` (pure Python), so
no Potrace binary is needed.

```python
from myfont.book_font import make_book_font

result = make_book_font(
    "chunky crayon letters", "Quill Crayon",
    generate_image=lambda prompt: my_image_model(prompt),   # -> PNG bytes
    check_labels=None,  # optional vision check, see the docstring
)
open("QuillCrayon.ttf", "wb").write(result.ttf)
```

Quill installs this package from GitHub (`pip install git+https://github.com/bkunbargi/fontmaker@<sha>`).
Tests: `pytest tests/`.

## Requirements

- Python 3.9+
- Potrace (for bitmap-to-vector conversion)

### Installing Potrace

**macOS (Homebrew):**
```bash
brew install potrace
```

**Ubuntu/Debian:**
```bash
sudo apt-get install potrace
```

**Windows:**
Download from [potrace.sourceforge.net](http://potrace.sourceforge.net/#downloading), extract, and add to PATH.

Verify installation:
```bash
potrace --version
```

## Installation

1. Clone or download this repository:
```bash
cd myfont
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Start the Streamlit application:
```bash
streamlit run app.py
```

This opens a web interface in your browser (typically at http://localhost:8501).

### Workflow

1. **Upload** - Upload a glyph sheet (all characters in one image) or individual glyph images
2. **Segment** - Review and adjust automatically extracted glyphs
3. **Map** - Assign characters to each glyph (A, B, C, etc.)
4. **Generate** - Create and download your font files

## Tips for Best Results

### Preparing Your Glyph Image

- Use **high contrast**: Black glyphs on white background (or vice versa)
- Keep glyphs **well-separated**: Leave clear space between characters
- Use **consistent sizing**: All characters at approximately the same height
- Higher resolution = better quality vectors

### Glyph Sheet Layout

Arrange glyphs in reading order (left-to-right, top-to-bottom):
```
A B C D E F G H I J K L M
N O P Q R S T U V W X Y Z
a b c d e f g h i j k l m
n o p q r s t u v w x y z
0 1 2 3 4 5 6 7 8 9
```

## Project Structure

```
myfont/
├── app.py                    # Main Streamlit entry point
├── requirements.txt
├── README.md
│
├── myfont/                   # Core library
│   ├── config.py             # Constants
│   ├── image_processing/     # Load, preprocess, segment images
│   ├── vectorization/        # Potrace wrapper, SVG parsing
│   ├── font_generation/      # Build fonts with FontTools
│   ├── character_mapping/    # Character set definitions
│   └── preview/              # Font preview rendering
│
└── ui/                       # Streamlit pages
    ├── state.py              # Session state management
    └── pages/                # Multi-page app pages
```

## Programmatic Usage

You can also use the library directly:

```python
from myfont.image_processing import load_image, preprocess_image, segment_glyphs
from myfont.font_generation import build_font
from myfont.font_generation.exporter import export_all

# Load and preprocess image
image = load_image("glyphs.png")
binary = preprocess_image(image)

# Extract glyphs
glyphs = segment_glyphs(binary)

# Map characters to glyph images
char_images = {}
characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
for i, (glyph_img, bounds) in enumerate(glyphs):
    if i < len(characters):
        char_images[characters[i]] = glyph_img

# Build and export font
font = build_font(char_images, font_family="MyCustomFont")
export_all(font, "output/", "MyCustomFont")
```

## Technical Notes

- **Units Per Em (UPM)**: 1000 (industry standard)
- **Coordinate system**: SVG Y-axis is flipped for font coordinates
- **Curve format**: TrueType uses quadratic Beziers; cubic curves from Potrace are converted
- **Required glyphs**: .notdef and space are automatically created

## Troubleshooting

### "Potrace not found"
Make sure Potrace is installed and in your PATH. Run `potrace --version` to verify.

### Glyphs not detected
- Increase contrast in your source image
- Adjust the "Minimum Glyph Area" parameter
- Try inverting the image

### Font looks different than expected
- Check that glyphs are mapped to correct characters
- Verify the source image has sufficient resolution
- Adjust Potrace parameters for smoother curves

## License

MIT License
