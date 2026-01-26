"""SVG path parsing utilities."""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass
import xml.etree.ElementTree as ET


@dataclass
class PathCommand:
    """Represents a single SVG path command."""
    command: str  # M, L, C, Q, Z, etc.
    args: List[float]
    is_relative: bool


def extract_paths_from_svg(svg_content: str) -> List[str]:
    """Extract path data strings from SVG content.

    Args:
        svg_content: SVG file content as string.

    Returns:
        List of path data strings (d attribute values).
    """
    paths = []

    try:
        # Parse SVG XML
        root = ET.fromstring(svg_content)

        # Handle namespace
        ns = {'svg': 'http://www.w3.org/2000/svg'}

        # Find all path elements (with and without namespace)
        for path in root.iter():
            if path.tag.endswith('path') or path.tag == 'path':
                d = path.get('d')
                if d:
                    paths.append(d)

    except ET.ParseError:
        # Fallback: use regex if XML parsing fails
        pattern = r'd="([^"]+)"'
        matches = re.findall(pattern, svg_content)
        paths.extend(matches)

    return paths


def get_svg_dimensions(svg_content: str) -> Tuple[float, float]:
    """Extract dimensions from SVG.

    Args:
        svg_content: SVG file content.

    Returns:
        Tuple of (width, height).
    """
    try:
        root = ET.fromstring(svg_content)

        width = root.get('width', '100')
        height = root.get('height', '100')

        # Remove units if present
        width = float(re.sub(r'[^\d.]', '', width) or 100)
        height = float(re.sub(r'[^\d.]', '', height) or 100)

        return width, height

    except:
        return 100.0, 100.0


def parse_svg_path(path_data: str) -> List[PathCommand]:
    """Parse SVG path data into commands.

    Args:
        path_data: SVG path d attribute string.

    Returns:
        List of PathCommand objects.
    """
    commands = []

    # Regex to split path into commands
    # Matches command letter followed by numbers/coordinates
    pattern = r'([MmZzLlHhVvCcSsQqTtAa])([^MmZzLlHhVvCcSsQqTtAa]*)'

    for match in re.finditer(pattern, path_data):
        cmd = match.group(1)
        args_str = match.group(2).strip()

        # Parse numeric arguments
        args = []
        if args_str:
            # Handle negative numbers and decimals
            num_pattern = r'-?\d*\.?\d+(?:[eE][+-]?\d+)?'
            args = [float(n) for n in re.findall(num_pattern, args_str)]

        is_relative = cmd.islower()
        commands.append(PathCommand(
            command=cmd.upper(),
            args=args,
            is_relative=is_relative
        ))

    return commands


def path_commands_to_string(commands: List[PathCommand]) -> str:
    """Convert PathCommand list back to SVG path string.

    Args:
        commands: List of PathCommand objects.

    Returns:
        SVG path data string.
    """
    parts = []

    for cmd in commands:
        letter = cmd.command.lower() if cmd.is_relative else cmd.command
        if cmd.args:
            args_str = ' '.join(f'{a:.2f}' for a in cmd.args)
            parts.append(f'{letter} {args_str}')
        else:
            parts.append(letter)

    return ' '.join(parts)


def transform_path_commands(
    commands: List[PathCommand],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    translate_x: float = 0.0,
    translate_y: float = 0.0,
    flip_y: bool = False,
    height: float = 0.0
) -> List[PathCommand]:
    """Apply transformations to path commands.

    Args:
        commands: List of PathCommand objects.
        scale_x: Horizontal scale factor.
        scale_y: Vertical scale factor.
        translate_x: Horizontal translation.
        translate_y: Vertical translation.
        flip_y: Flip Y axis (for font coordinate conversion).
        height: Original height (needed for Y flip).

    Returns:
        Transformed PathCommand list.
    """
    result = []
    current_x, current_y = 0.0, 0.0

    for cmd in commands:
        new_cmd = PathCommand(
            command=cmd.command,
            args=cmd.args.copy(),
            is_relative=cmd.is_relative
        )

        args = new_cmd.args

        if cmd.command == 'M' or cmd.command == 'L':
            # Move/Line: x, y pairs
            for i in range(0, len(args), 2):
                if i + 1 < len(args):
                    x, y = args[i], args[i + 1]
                    if cmd.is_relative:
                        x += current_x
                        y += current_y
                    if flip_y:
                        y = height - y
                    x = x * scale_x + translate_x
                    y = y * scale_y + translate_y
                    args[i] = x
                    args[i + 1] = y
                    current_x, current_y = x, y

        elif cmd.command == 'H':
            # Horizontal line: x
            for i in range(len(args)):
                x = args[i]
                if cmd.is_relative:
                    x += current_x
                x = x * scale_x + translate_x
                args[i] = x
                current_x = x

        elif cmd.command == 'V':
            # Vertical line: y
            for i in range(len(args)):
                y = args[i]
                if cmd.is_relative:
                    y += current_y
                if flip_y:
                    y = height - y
                y = y * scale_y + translate_y
                args[i] = y
                current_y = y

        elif cmd.command == 'C':
            # Cubic bezier: x1, y1, x2, y2, x, y
            for i in range(0, len(args), 6):
                if i + 5 < len(args):
                    for j in range(3):
                        x_idx = i + j * 2
                        y_idx = i + j * 2 + 1
                        x, y = args[x_idx], args[y_idx]
                        if cmd.is_relative:
                            x += current_x
                            y += current_y
                        if flip_y:
                            y = height - y
                        x = x * scale_x + translate_x
                        y = y * scale_y + translate_y
                        args[x_idx] = x
                        args[y_idx] = y
                    current_x = args[i + 4]
                    current_y = args[i + 5]

        elif cmd.command == 'Q':
            # Quadratic bezier: x1, y1, x, y
            for i in range(0, len(args), 4):
                if i + 3 < len(args):
                    for j in range(2):
                        x_idx = i + j * 2
                        y_idx = i + j * 2 + 1
                        x, y = args[x_idx], args[y_idx]
                        if cmd.is_relative:
                            x += current_x
                            y += current_y
                        if flip_y:
                            y = height - y
                        x = x * scale_x + translate_x
                        y = y * scale_y + translate_y
                        args[x_idx] = x
                        args[y_idx] = y
                    current_x = args[i + 2]
                    current_y = args[i + 3]

        # Convert relative to absolute after transformation
        new_cmd.is_relative = False
        result.append(new_cmd)

    return result
