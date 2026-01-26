"""Image loading and validation utilities."""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
from PIL import Image
import io

from ..config import SUPPORTED_IMAGE_FORMATS, MIN_GLYPH_SIZE, MAX_GLYPH_SIZE


class ImageLoadError(Exception):
    """Raised when image loading fails."""
    pass


class ImageValidationError(Exception):
    """Raised when image validation fails."""
    pass


def load_image(source: Union[str, Path, bytes, io.BytesIO]) -> np.ndarray:
    """Load an image from various sources.

    Args:
        source: File path, bytes, or BytesIO object containing image data.

    Returns:
        Image as numpy array in BGR format (OpenCV standard).

    Raises:
        ImageLoadError: If loading fails.
    """
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise ImageLoadError(f"File not found: {path}")
            if path.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
                raise ImageLoadError(
                    f"Unsupported format: {path.suffix}. "
                    f"Supported: {', '.join(SUPPORTED_IMAGE_FORMATS)}"
                )
            image = cv2.imread(str(path))
            if image is None:
                raise ImageLoadError(f"Failed to decode image: {path}")
            return image

        elif isinstance(source, bytes):
            nparr = np.frombuffer(source, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ImageLoadError("Failed to decode image from bytes")
            return image

        elif isinstance(source, io.BytesIO):
            source.seek(0)
            return load_image(source.read())

        else:
            raise ImageLoadError(f"Unsupported source type: {type(source)}")

    except ImageLoadError:
        raise
    except Exception as e:
        raise ImageLoadError(f"Failed to load image: {e}")


def load_image_pil(source: Union[str, Path, bytes, io.BytesIO]) -> Image.Image:
    """Load an image using PIL.

    Args:
        source: File path, bytes, or BytesIO object containing image data.

    Returns:
        PIL Image object.

    Raises:
        ImageLoadError: If loading fails.
    """
    try:
        if isinstance(source, (str, Path)):
            return Image.open(source)
        elif isinstance(source, bytes):
            return Image.open(io.BytesIO(source))
        elif isinstance(source, io.BytesIO):
            source.seek(0)
            return Image.open(source)
        else:
            raise ImageLoadError(f"Unsupported source type: {type(source)}")
    except ImageLoadError:
        raise
    except Exception as e:
        raise ImageLoadError(f"Failed to load image with PIL: {e}")


def validate_image(
    image: np.ndarray,
    min_size: int = MIN_GLYPH_SIZE,
    max_size: int = MAX_GLYPH_SIZE
) -> Tuple[bool, Optional[str]]:
    """Validate an image for glyph processing.

    Args:
        image: Image as numpy array.
        min_size: Minimum dimension in pixels.
        max_size: Maximum dimension in pixels.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if image is None:
        return False, "Image is None"

    if len(image.shape) < 2:
        return False, "Invalid image dimensions"

    height, width = image.shape[:2]

    if height < min_size or width < min_size:
        return False, f"Image too small: {width}x{height} (min: {min_size})"

    if height > max_size or width > max_size:
        return False, f"Image too large: {width}x{height} (max: {max_size})"

    return True, None


def get_image_info(image: np.ndarray) -> dict:
    """Get information about an image.

    Args:
        image: Image as numpy array.

    Returns:
        Dictionary with image information.
    """
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1

    return {
        'width': width,
        'height': height,
        'channels': channels,
        'dtype': str(image.dtype),
        'size_bytes': image.nbytes,
    }


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale if needed.

    Args:
        image: Input image (BGR or grayscale).

    Returns:
        Grayscale image.
    """
    if len(image.shape) == 2:
        return image
    elif len(image.shape) == 3:
        if image.shape[2] == 1:
            return image[:, :, 0]
        elif image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ImageLoadError(f"Cannot convert image with shape {image.shape} to grayscale")
