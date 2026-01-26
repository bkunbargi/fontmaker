"""Configuration constants for MyFont."""

# Font metrics
UNITS_PER_EM = 1000
ASCENDER = 800
DESCENDER = -200
CAP_HEIGHT = 700
X_HEIGHT = 500
LINE_GAP = 0

# Default font metadata
DEFAULT_FONT_FAMILY = "MyFont"
DEFAULT_FONT_VERSION = "1.0"

# Image processing
MIN_GLYPH_SIZE = 10  # Minimum pixel dimension for a valid glyph
MAX_GLYPH_SIZE = 2000  # Maximum pixel dimension
BINARY_THRESHOLD = 128  # Default threshold for binarization (0-255)

# Supported image formats
SUPPORTED_IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

# Export formats
EXPORT_FORMATS = ['ttf', 'woff', 'woff2']
