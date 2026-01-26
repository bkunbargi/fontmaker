"""Glyph segmentation via contour detection."""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..config import MIN_GLYPH_SIZE


@dataclass
class GlyphBounds:
    """Bounding box and metadata for an extracted glyph."""
    x: int
    y: int
    width: int
    height: int
    contour: np.ndarray
    area: float


def segment_glyphs(
    binary_image: np.ndarray,
    min_area: int = 50,
    min_size: int = MIN_GLYPH_SIZE,
    padding: int = 2,
    sort_by: str = 'position',
    source_image: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, GlyphBounds]]:
    """Extract individual glyphs from a binary image.

    Args:
        binary_image: Binary image with white glyphs on black background.
        min_area: Minimum contour area to consider.
        min_size: Minimum glyph dimension.
        padding: Padding around each glyph.
        sort_by: Sorting method ('position' for reading order, 'area', 'none').
        source_image: Optional source image to extract glyphs from (for color extraction).
                      If None, extracts from binary_image.

    Returns:
        List of (glyph_image, bounds) tuples.
    """
    # Use source_image for extraction if provided, otherwise use binary_image
    extract_from = source_image if source_image is not None else binary_image

    # Find contours from binary image
    contours, hierarchy = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    glyphs = []
    height, width = binary_image.shape[:2]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        if w < min_size or h < min_size:
            continue

        # Add padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)

        # Extract glyph image from source
        glyph_image = extract_from[y1:y2, x1:x2].copy()

        bounds = GlyphBounds(
            x=x1, y=y1,
            width=x2 - x1, height=y2 - y1,
            contour=contour,
            area=area
        )

        glyphs.append((glyph_image, bounds))

    # Sort glyphs
    if sort_by == 'position':
        glyphs = sort_glyphs_by_position(glyphs)
    elif sort_by == 'area':
        glyphs.sort(key=lambda g: g[1].area, reverse=True)

    return glyphs


def sort_glyphs_by_position(
    glyphs: List[Tuple[np.ndarray, GlyphBounds]],
    row_threshold: Optional[int] = None
) -> List[Tuple[np.ndarray, GlyphBounds]]:
    """Sort glyphs in reading order (top-to-bottom, left-to-right).

    Args:
        glyphs: List of (glyph_image, bounds) tuples.
        row_threshold: Vertical distance to consider same row. Auto-detected if None.

    Returns:
        Sorted list of glyphs.
    """
    if not glyphs:
        return glyphs

    # Auto-detect row threshold based on average glyph height
    if row_threshold is None:
        avg_height = np.mean([g[1].height for g in glyphs])
        row_threshold = int(avg_height * 0.5)

    # Sort by y coordinate first
    glyphs = sorted(glyphs, key=lambda g: g[1].y)

    # Group into rows
    rows = []
    current_row = [glyphs[0]]
    current_y = glyphs[0][1].y

    for glyph in glyphs[1:]:
        if abs(glyph[1].y - current_y) <= row_threshold:
            current_row.append(glyph)
        else:
            rows.append(current_row)
            current_row = [glyph]
            current_y = glyph[1].y

    rows.append(current_row)

    # Sort each row by x coordinate
    sorted_glyphs = []
    for row in rows:
        row.sort(key=lambda g: g[1].x)
        sorted_glyphs.extend(row)

    return sorted_glyphs


def extract_glyph_bounds(binary_image: np.ndarray) -> List[GlyphBounds]:
    """Extract just the bounding boxes without glyph images.

    Args:
        binary_image: Binary image.

    Returns:
        List of GlyphBounds objects.
    """
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    bounds = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        bounds.append(GlyphBounds(x=x, y=y, width=w, height=h, contour=contour, area=area))

    return bounds


def merge_nearby_contours(
    glyphs: List[Tuple[np.ndarray, GlyphBounds]],
    binary_image: np.ndarray,
    distance_threshold: int = 5,
    source_image: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, GlyphBounds]]:
    """Merge glyphs that are close together (e.g., 'i' dot and stem).

    Args:
        glyphs: List of (glyph_image, bounds) tuples.
        binary_image: Original binary image for re-extraction.
        distance_threshold: Maximum distance between components to merge.
        source_image: Optional source image to extract glyphs from (for color extraction).
                      If None, extracts from binary_image.

    Returns:
        List of merged glyphs.
    """
    if len(glyphs) <= 1:
        return glyphs

    # Use source_image for extraction if provided
    extract_from = source_image if source_image is not None else binary_image

    # Create list of bounds for merging
    bounds_list = [g[1] for g in glyphs]
    merged = [False] * len(bounds_list)
    result = []

    height, width = binary_image.shape[:2]

    for i, bounds1 in enumerate(bounds_list):
        if merged[i]:
            continue

        # Find all nearby bounds
        merge_group = [i]
        for j, bounds2 in enumerate(bounds_list[i + 1:], i + 1):
            if merged[j]:
                continue

            # Check horizontal overlap and vertical proximity
            h_overlap = not (bounds1.x + bounds1.width < bounds2.x - distance_threshold or
                            bounds2.x + bounds2.width < bounds1.x - distance_threshold)

            v_distance = min(
                abs(bounds1.y - (bounds2.y + bounds2.height)),
                abs(bounds2.y - (bounds1.y + bounds1.height))
            )

            if h_overlap and v_distance <= distance_threshold:
                merge_group.append(j)
                merged[j] = True

        # Create merged bounding box
        group_bounds = [bounds_list[idx] for idx in merge_group]
        min_x = min(b.x for b in group_bounds)
        min_y = min(b.y for b in group_bounds)
        max_x = max(b.x + b.width for b in group_bounds)
        max_y = max(b.y + b.height for b in group_bounds)

        # Extract merged glyph
        padding = 2
        x1 = max(0, min_x - padding)
        y1 = max(0, min_y - padding)
        x2 = min(width, max_x + padding)
        y2 = min(height, max_y + padding)

        glyph_image = extract_from[y1:y2, x1:x2].copy()

        # Combine contours
        combined_contour = np.vstack([bounds_list[idx].contour for idx in merge_group])

        new_bounds = GlyphBounds(
            x=x1, y=y1,
            width=x2 - x1, height=y2 - y1,
            contour=combined_contour,
            area=sum(bounds_list[idx].area for idx in merge_group)
        )

        result.append((glyph_image, new_bounds))

    return result


def split_connected_glyphs(
    glyph_image: np.ndarray,
    expected_count: int = 2
) -> List[np.ndarray]:
    """Attempt to split connected glyphs using vertical projection.

    Args:
        glyph_image: Binary image containing potentially connected glyphs.
        expected_count: Expected number of glyphs.

    Returns:
        List of split glyph images.
    """
    # Vertical projection
    projection = np.sum(glyph_image, axis=0)

    # Find valleys (potential split points)
    valleys = []
    threshold = projection.max() * 0.1

    in_valley = False
    valley_start = 0

    for i, val in enumerate(projection):
        if val <= threshold and not in_valley:
            in_valley = True
            valley_start = i
        elif val > threshold and in_valley:
            in_valley = False
            valley_center = (valley_start + i) // 2
            valleys.append(valley_center)

    if len(valleys) < expected_count - 1:
        return [glyph_image]

    # Split at valley points
    split_points = [0] + sorted(valleys[:expected_count - 1]) + [glyph_image.shape[1]]
    glyphs = []

    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]
        if end > start:
            glyphs.append(glyph_image[:, start:end])

    return glyphs
