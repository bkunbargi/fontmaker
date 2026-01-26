"""Wrapper for Potrace bitmap tracing tool."""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import io


class PotraceError(Exception):
    """Raised when Potrace execution fails."""
    pass


def check_potrace_installed() -> Tuple[bool, str]:
    """Check if Potrace is installed and available.

    Returns:
        Tuple of (is_installed, version_or_error_message).
    """
    try:
        result = subprocess.run(
            ['potrace', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip().split('\n')[0]
            return True, version
        return False, result.stderr or "Unknown error"
    except FileNotFoundError:
        return False, "Potrace not found. Install with: brew install potrace (macOS) or apt-get install potrace (Ubuntu)"
    except subprocess.TimeoutExpired:
        return False, "Potrace command timed out"
    except Exception as e:
        return False, str(e)


def trace_bitmap(
    image: np.ndarray,
    turdsize: int = 2,
    alphamax: float = 1.0,
    opttolerance: float = 0.2,
    invert: bool = False
) -> str:
    """Trace a bitmap image to SVG using Potrace.

    Args:
        image: Binary image (numpy array, 0 and 255 values).
        turdsize: Suppress speckles of up to this size (default 2).
        alphamax: Corner threshold (0-1.334, default 1.0).
        opttolerance: Curve optimization tolerance (default 0.2).
        invert: Invert the image before tracing.

    Returns:
        SVG string.

    Raises:
        PotraceError: If tracing fails.
    """
    installed, msg = check_potrace_installed()
    if not installed:
        raise PotraceError(msg)

    # Ensure image is binary
    if image.max() > 1:
        binary = (image > 127).astype(np.uint8) * 255
    else:
        binary = (image * 255).astype(np.uint8)

    # Invert if needed (Potrace traces black areas)
    if invert:
        binary = 255 - binary

    # Convert to PIL Image
    pil_image = Image.fromarray(binary, mode='L')

    # Save to temporary BMP file (Potrace prefers BMP/PBM)
    with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as tmp_in:
        pil_image.save(tmp_in.name, format='BMP')
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path.replace('.bmp', '.svg')

    try:
        # Run Potrace
        cmd = [
            'potrace',
            '-s',  # SVG output
            '-t', str(turdsize),
            '-a', str(alphamax),
            '-O', str(opttolerance),
            '-o', tmp_out_path,
            tmp_in_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            raise PotraceError(f"Potrace failed: {result.stderr}")

        # Read SVG output
        with open(tmp_out_path, 'r') as f:
            svg_content = f.read()

        return svg_content

    except subprocess.TimeoutExpired:
        raise PotraceError("Potrace timed out")
    except FileNotFoundError:
        raise PotraceError("Potrace output file not found")
    finally:
        # Clean up temporary files
        for path in [tmp_in_path, tmp_out_path]:
            try:
                os.unlink(path)
            except:
                pass


def trace_bitmap_to_paths(
    image: np.ndarray,
    **kwargs
) -> str:
    """Trace bitmap and return just the path data.

    Args:
        image: Binary image.
        **kwargs: Arguments passed to trace_bitmap.

    Returns:
        SVG path data string (d attribute content).
    """
    from .svg_parser import extract_paths_from_svg

    svg = trace_bitmap(image, **kwargs)
    paths = extract_paths_from_svg(svg)

    # Combine all paths
    return ' '.join(paths)


def get_potrace_install_instructions() -> str:
    """Get installation instructions for Potrace.

    Returns:
        Installation instructions string.
    """
    return """
Potrace Installation Instructions:

macOS (Homebrew):
    brew install potrace

Ubuntu/Debian:
    sudo apt-get install potrace

Fedora:
    sudo dnf install potrace

Windows:
    Download from: http://potrace.sourceforge.net/#downloading
    Extract and add to PATH

After installation, verify with:
    potrace --version
"""
