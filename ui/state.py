"""Session state management for the Streamlit UI."""

import streamlit as st
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class GlyphData:
    """Represents an extracted glyph."""
    image: np.ndarray
    bounds: tuple  # (x, y, width, height)
    index: int
    assigned_char: Optional[str] = None


@dataclass
class AppState:
    """Application state container."""
    # Uploaded image data
    uploaded_image: Optional[np.ndarray] = None
    uploaded_filename: Optional[str] = None
    image_source: str = "upload"  # "upload" or "ai_generated"

    # Preprocessed image
    processed_image: Optional[np.ndarray] = None

    # Extracted glyphs
    glyphs: List[GlyphData] = field(default_factory=list)

    # Character mappings (glyph_index -> character)
    mappings: Dict[int, str] = field(default_factory=dict)

    # Font settings
    font_family: str = "MyFont"
    font_mode: str = "monochrome"  # "monochrome" or "color"

    # Generated font data
    font_bytes: Optional[bytes] = None


def init_state():
    """Initialize session state with default values."""
    if 'app_state' not in st.session_state:
        st.session_state.app_state = AppState()
    return st.session_state.app_state


def get_state() -> AppState:
    """Get the current application state."""
    return init_state()


def reset_state():
    """Reset all application state."""
    st.session_state.app_state = AppState()


def set_uploaded_image(image: np.ndarray, filename: str, source: str = "upload"):
    """Set the uploaded image in state.

    Args:
        image: The image as a numpy array (BGR format from OpenCV).
        filename: The filename to display.
        source: Image source - "upload" or "ai_generated".
    """
    state = get_state()
    state.uploaded_image = image
    state.uploaded_filename = filename
    state.image_source = source
    # Clear downstream state
    state.processed_image = None
    state.glyphs = []
    state.mappings = {}
    state.font_bytes = None


def set_processed_image(image: np.ndarray):
    """Set the preprocessed image in state."""
    state = get_state()
    state.processed_image = image


def set_glyphs(glyphs: List[GlyphData]):
    """Set the extracted glyphs in state."""
    state = get_state()
    state.glyphs = glyphs
    # Clear mappings when glyphs change
    state.mappings = {}
    state.font_bytes = None


def set_mapping(glyph_index: int, character: str):
    """Set a character mapping for a glyph."""
    state = get_state()
    state.mappings[glyph_index] = character


def remove_mapping(glyph_index: int):
    """Remove a character mapping."""
    state = get_state()
    if glyph_index in state.mappings:
        del state.mappings[glyph_index]


def set_font_bytes(font_bytes: bytes):
    """Set the generated font bytes."""
    state = get_state()
    state.font_bytes = font_bytes


def get_mapped_glyphs() -> Dict[str, GlyphData]:
    """Get glyphs with their assigned characters."""
    state = get_state()
    result = {}
    for glyph in state.glyphs:
        if glyph.index in state.mappings:
            char = state.mappings[glyph.index]
            result[char] = glyph
    return result


def merge_selected_glyphs(glyphs: List[GlyphData], indices: List[int], image: np.ndarray) -> List[GlyphData]:
    """Merge glyphs at specified indices into one.

    Args:
        glyphs: List of all glyphs.
        indices: Indices of glyphs to merge.
        image: The processed image to extract the merged region from.

    Returns:
        New list of glyphs with selected ones merged.
    """
    to_merge = [g for g in glyphs if g.index in indices]
    others = [g for g in glyphs if g.index not in indices]

    # Combined bounding box
    min_x = min(g.bounds[0] for g in to_merge)
    min_y = min(g.bounds[1] for g in to_merge)
    max_x = max(g.bounds[0] + g.bounds[2] for g in to_merge)
    max_y = max(g.bounds[1] + g.bounds[3] for g in to_merge)

    # Extract merged region from the image
    merged_image = image[min_y:max_y, min_x:max_x].copy()

    # Create new glyph with merged bounds
    merged = GlyphData(
        image=merged_image,
        bounds=(min_x, min_y, max_x - min_x, max_y - min_y),
        index=min(indices)
    )

    # Combine and reindex by position (top-to-bottom, left-to-right)
    result = others + [merged]
    for i, g in enumerate(sorted(result, key=lambda x: (x.bounds[1], x.bounds[0]))):
        g.index = i

    return result


