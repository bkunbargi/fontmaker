"""Image preprocessing for glyph extraction."""

import cv2
import numpy as np
from typing import Optional, Tuple

from ..config import BINARY_THRESHOLD


def preprocess_image(
    image: np.ndarray,
    threshold: Optional[int] = None,
    invert: bool = False,
    denoise: bool = True
) -> np.ndarray:
    """Preprocess image for glyph extraction.

    Applies grayscale conversion, optional denoising, and binarization.

    Args:
        image: Input image (BGR or grayscale).
        threshold: Binarization threshold (None for Otsu's method).
        invert: If True, invert the binary image.
        denoise: If True, apply noise removal.

    Returns:
        Preprocessed binary image (0 = background, 255 = foreground).
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Denoise
    if denoise:
        gray = remove_noise(gray)

    # Binarize
    binary = binarize(gray, threshold)

    # Invert if needed (ensure glyphs are white on black background)
    if invert:
        binary = cv2.bitwise_not(binary)

    return binary


def binarize(
    image: np.ndarray,
    threshold: Optional[int] = None,
    method: str = 'otsu'
) -> np.ndarray:
    """Convert grayscale image to binary.

    Args:
        image: Grayscale image.
        threshold: Fixed threshold value. If None, use automatic method.
        method: Automatic thresholding method ('otsu' or 'adaptive').

    Returns:
        Binary image (0 and 255 values only).
    """
    if threshold is not None:
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    elif method == 'otsu':
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    else:
        _, binary = cv2.threshold(image, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

    return binary


def remove_noise(
    image: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:
    """Remove noise from image using morphological operations.

    Args:
        image: Input image (grayscale or binary).
        kernel_size: Size of the morphological kernel.
        iterations: Number of iterations.

    Returns:
        Denoised image.
    """
    # Apply Gaussian blur for initial noise reduction
    denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return denoised


def clean_binary(
    image: np.ndarray,
    min_area: int = 10,
    remove_holes: bool = True
) -> np.ndarray:
    """Clean up a binary image by removing small components and filling holes.

    Args:
        image: Binary image.
        min_area: Minimum area for components to keep.
        remove_holes: If True, fill small holes inside components.

    Returns:
        Cleaned binary image.
    """
    # Remove small components
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            cv2.drawContours(mask, [contour], -1, 255, -1)

    # Fill holes using morphological closing
    if remove_holes:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def auto_detect_invert(image: np.ndarray) -> bool:
    """Detect if image needs to be inverted.

    Assumes glyphs should be the minority of pixels.

    Args:
        image: BGR, grayscale, or binary image.

    Returns:
        True if image should be inverted.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Convert to binary if needed
    if gray.max() > 1:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = gray

    # Count white pixels
    white_ratio = np.sum(binary > 127) / binary.size

    # If more than 50% is white, assume background is white
    return white_ratio > 0.5


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Enhance image contrast using CLAHE.

    Args:
        image: Grayscale image.

    Returns:
        Contrast-enhanced image.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def resize_for_processing(
    image: np.ndarray,
    max_dimension: int = 4000
) -> Tuple[np.ndarray, float]:
    """Resize image if too large for efficient processing.

    Args:
        image: Input image.
        max_dimension: Maximum width or height.

    Returns:
        Tuple of (resized_image, scale_factor).
    """
    height, width = image.shape[:2]
    max_dim = max(height, width)

    if max_dim <= max_dimension:
        return image, 1.0

    scale = max_dimension / max_dim
    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale
