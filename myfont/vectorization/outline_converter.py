"""Convert SVG paths to FontTools glyph outlines."""

from typing import List, Tuple, Optional
import numpy as np

from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.cu2qu import curve_to_quadratic

from .svg_parser import parse_svg_path, PathCommand
from ..config import UNITS_PER_EM

# Maximum coordinate value for TrueType (signed 16-bit)
MAX_COORD = 16000
MIN_COORD = -16000


def clamp_coord(val: float) -> float:
    """Clamp coordinate to valid TrueType range."""
    return max(MIN_COORD, min(MAX_COORD, val))


def svg_to_glyph_outline(
    path_data: str,
    glyph_height: float,
    target_height: float = 700,  # Target cap height
    baseline_ratio: float = 0.0,  # Baseline at y=0
    use_quadratic: bool = True
) -> Tuple[List, int]:
    """Convert SVG path data to glyph outline.

    Args:
        path_data: SVG path d attribute string.
        glyph_height: Original glyph height in pixels.
        target_height: Target height in font units.
        baseline_ratio: Where baseline sits.
        use_quadratic: Convert cubic curves to quadratic (for TrueType).

    Returns:
        Tuple of (outline_commands, advance_width).
    """
    # Parse path
    commands = parse_svg_path(path_data)

    if not commands:
        return [], 0

    # First: convert all relative coordinates to absolute
    abs_commands = []
    cur_x, cur_y = 0.0, 0.0

    for cmd in commands:
        args = cmd.args
        new_args = []

        if cmd.command == 'Z':
            abs_commands.append(PathCommand('Z', [], False))
            continue

        if cmd.command in ('M', 'L'):
            for i in range(0, len(args) - 1, 2):
                x, y = args[i], args[i + 1]
                if cmd.is_relative:
                    x += cur_x
                    y += cur_y
                new_args.extend([x, y])
                cur_x, cur_y = x, y
            abs_commands.append(PathCommand(cmd.command, new_args, False))

        elif cmd.command == 'C':
            for i in range(0, len(args) - 5, 6):
                coords = []
                for j in range(0, 6, 2):
                    x, y = args[i + j], args[i + j + 1]
                    if cmd.is_relative:
                        x += cur_x
                        y += cur_y
                    coords.extend([x, y])
                new_args.extend(coords)
                cur_x, cur_y = coords[-2], coords[-1]
            abs_commands.append(PathCommand('C', new_args, False))

        elif cmd.command == 'Q':
            for i in range(0, len(args) - 3, 4):
                coords = []
                for j in range(0, 4, 2):
                    x, y = args[i + j], args[i + j + 1]
                    if cmd.is_relative:
                        x += cur_x
                        y += cur_y
                    coords.extend([x, y])
                new_args.extend(coords)
                cur_x, cur_y = coords[-2], coords[-1]
            abs_commands.append(PathCommand('Q', new_args, False))

        elif cmd.command == 'H':
            for x in args:
                if cmd.is_relative:
                    x += cur_x
                new_args.extend([x, cur_y])
                cur_x = x
            abs_commands.append(PathCommand('L', new_args, False))

        elif cmd.command == 'V':
            for y in args:
                if cmd.is_relative:
                    y += cur_y
                new_args.extend([cur_x, y])
                cur_y = y
            abs_commands.append(PathCommand('L', new_args, False))

    if not abs_commands:
        return [], 0

    # Find bounding box
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    for cmd in abs_commands:
        for i in range(0, len(cmd.args) - 1, 2):
            x, y = cmd.args[i], cmd.args[i + 1]
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)

    if min_x == float('inf'):
        return [], 0

    orig_width = max_x - min_x
    orig_height = max_y - min_y

    if orig_height <= 0 or orig_width <= 0:
        return [], 0

    # Calculate scale to fit target height
    scale = target_height / orig_height

    # Transform: normalize to origin and scale
    # Note: Potrace SVG already has Y=0 at top, and we want Y=0 at baseline
    # So we just normalize without flipping
    transformed = []
    for cmd in abs_commands:
        new_args = []
        for i in range(0, len(cmd.args) - 1, 2):
            x = (cmd.args[i] - min_x) * scale
            y = (cmd.args[i + 1] - min_y) * scale
            new_args.extend([clamp_coord(x), clamp_coord(y)])

        transformed.append(PathCommand(cmd.command, new_args, False))

    # Convert cubic to quadratic if needed
    if use_quadratic:
        transformed = convert_cubic_to_quadratic(transformed)

    # Calculate advance width
    advance_width = int(orig_width * scale + 100)

    return transformed, advance_width


def convert_cubic_to_quadratic(
    commands: List[PathCommand],
    max_err: float = 1.0
) -> List[PathCommand]:
    """Convert cubic bezier curves to quadratic (TrueType format).

    Args:
        commands: List of PathCommand objects.
        max_err: Maximum error tolerance.

    Returns:
        Commands with cubic curves converted to quadratic.
    """
    result = []
    current_x, current_y = 0.0, 0.0

    for cmd in commands:
        if cmd.command == 'C':
            # Convert each cubic segment
            args = cmd.args
            for i in range(0, len(args), 6):
                if i + 5 < len(args):
                    # Get control points
                    x1, y1 = args[i], args[i + 1]
                    x2, y2 = args[i + 2], args[i + 3]
                    x3, y3 = args[i + 4], args[i + 5]

                    # Simple approximation: use midpoint of cubic controls as quadratic control
                    qx = clamp_coord((x1 + x2) / 2)
                    qy = clamp_coord((y1 + y2) / 2)
                    result.append(PathCommand(
                        command='Q',
                        args=[qx, qy, clamp_coord(x3), clamp_coord(y3)],
                        is_relative=False
                    ))
                    current_x, current_y = x3, y3

        elif cmd.command in ('M', 'L'):
            # Clamp coordinates
            clamped_args = [clamp_coord(a) for a in cmd.args]
            result.append(PathCommand(
                command=cmd.command,
                args=clamped_args,
                is_relative=False
            ))
            if cmd.args:
                current_x = cmd.args[-2] if len(cmd.args) >= 2 else current_x
                current_y = cmd.args[-1] if len(cmd.args) >= 1 else current_y

        elif cmd.command == 'Q':
            clamped_args = [clamp_coord(a) for a in cmd.args]
            result.append(PathCommand(
                command='Q',
                args=clamped_args,
                is_relative=False
            ))
            if cmd.args and len(cmd.args) >= 4:
                current_x = cmd.args[-2]
                current_y = cmd.args[-1]

        else:
            result.append(cmd)

    return result


def calculate_advance_width(commands: List[PathCommand]) -> float:
    """Calculate advance width from path commands.

    Args:
        commands: List of PathCommand objects.

    Returns:
        Advance width (max x + small side bearing).
    """
    if not commands:
        return 0

    max_x = 0
    min_x = float('inf')

    for cmd in commands:
        args = cmd.args
        if cmd.command in ('M', 'L'):
            for i in range(0, len(args), 2):
                if i < len(args):
                    x = args[i]
                    max_x = max(max_x, x)
                    min_x = min(min_x, x)
        elif cmd.command == 'C':
            for i in range(0, len(args), 6):
                for j in [0, 2, 4]:
                    if i + j < len(args):
                        x = args[i + j]
                        max_x = max(max_x, x)
                        min_x = min(min_x, x)
        elif cmd.command == 'Q':
            for i in range(0, len(args), 4):
                for j in [0, 2]:
                    if i + j < len(args):
                        x = args[i + j]
                        max_x = max(max_x, x)
                        min_x = min(min_x, x)

    # Add small side bearings
    lsb = max(0, min_x) if min_x != float('inf') else 0
    width = max_x - min_x if min_x != float('inf') else max_x

    # Advance = left bearing + width + right bearing
    return width + abs(lsb) + 50  # 50 units right side bearing


def draw_outline_to_pen(commands: List[PathCommand], pen) -> None:
    """Draw path commands using a FontTools pen.

    Args:
        commands: List of PathCommand objects.
        pen: FontTools pen object.
    """
    if not commands:
        return

    has_move = False

    for cmd in commands:
        args = cmd.args

        if cmd.command == 'M':
            # MoveTo
            for i in range(0, len(args), 2):
                if i + 1 < len(args):
                    x, y = int(round(args[i])), int(round(args[i + 1]))
                    if i == 0:
                        pen.moveTo((x, y))
                        has_move = True
                    else:
                        pen.lineTo((x, y))

        elif cmd.command == 'L':
            # LineTo
            for i in range(0, len(args), 2):
                if i + 1 < len(args):
                    x, y = int(round(args[i])), int(round(args[i + 1]))
                    pen.lineTo((x, y))

        elif cmd.command == 'C':
            # Cubic bezier
            for i in range(0, len(args), 6):
                if i + 5 < len(args):
                    pen.curveTo(
                        (int(round(args[i])), int(round(args[i + 1]))),
                        (int(round(args[i + 2])), int(round(args[i + 3]))),
                        (int(round(args[i + 4])), int(round(args[i + 5])))
                    )

        elif cmd.command == 'Q':
            # Quadratic bezier
            for i in range(0, len(args), 4):
                if i + 3 < len(args):
                    pen.qCurveTo(
                        (int(round(args[i])), int(round(args[i + 1]))),
                        (int(round(args[i + 2])), int(round(args[i + 3])))
                    )

        elif cmd.command == 'Z':
            # Close path
            if has_move:
                pen.closePath()
                has_move = False

    # Close any unclosed path
    if has_move:
        pen.closePath()


def create_ttglyph_from_commands(
    commands: List[PathCommand],
    glyph_set
) -> 'TTGlyph':
    """Create a TrueType glyph from path commands.

    Args:
        commands: List of PathCommand objects.
        glyph_set: GlyphSet for the font.

    Returns:
        TTGlyph object.
    """
    pen = TTGlyphPen(glyph_set)
    draw_outline_to_pen(commands, pen)
    return pen.glyph()
