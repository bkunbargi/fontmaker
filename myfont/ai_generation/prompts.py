"""Prompt templates and style presets for AI glyph generation."""

from typing import Dict, List

# Style presets for different font aesthetics
STYLE_PRESETS: Dict[str, str] = {
    "handwritten": "casual handwritten style with natural pen strokes and slight irregularities",
    "serif": "elegant serif typeface with classic proportions and refined terminals",
    "sans-serif": "clean modern sans-serif with geometric precision and uniform stroke width",
    "bold": "heavy bold weight with thick strokes and strong presence",
    "script": "flowing cursive script with connected letters and elegant flourishes",
    "pixel": "retro pixel art style with blocky 8-bit aesthetic",
    "brush": "expressive brush lettering with dynamic thick-thin stroke variation",
    "graffiti": "urban graffiti style with bold outlines and street art influence",
    "vintage": "nostalgic vintage typography with distressed textures and retro charm",
    "minimalist": "ultra-minimal letterforms reduced to essential geometric shapes",
}

# Character sets for generation
CHARACTER_SETS: Dict[str, str] = {
    "uppercase": "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
    "lowercase": "a b c d e f g h i j k l m n o p q r s t u v w x y z",
    "digits": "0 1 2 3 4 5 6 7 8 9",
    "uppercase_digits": "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9",
    "lowercase_digits": "a b c d e f g h i j k l m n o p q r s t u v w x y z 0 1 2 3 4 5 6 7 8 9",
    "all_letters": "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z a b c d e f g h i j k l m n o p q r s t u v w x y z",
    "full": "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z a b c d e f g h i j k l m n o p q r s t u v w x y z 0 1 2 3 4 5 6 7 8 9",
}

# Human-readable labels for character sets
CHARACTER_SET_LABELS: Dict[str, str] = {
    "uppercase": "Uppercase (A-Z)",
    "lowercase": "Lowercase (a-z)",
    "digits": "Digits (0-9)",
    "uppercase_digits": "Uppercase + Digits (A-Z, 0-9)",
    "lowercase_digits": "Lowercase + Digits (a-z, 0-9)",
    "all_letters": "All Letters (A-Z, a-z)",
    "full": "Full Set (A-Z, a-z, 0-9)",
}


def build_prompt(style_description: str, character_set_key: str) -> str:
    """Build a prompt for glyph sheet generation."""
    characters = CHARACTER_SETS.get(character_set_key, CHARACTER_SETS["uppercase"])

    return f"""Generate an image of these letters for a font: {characters}

Style: {style_description}

White background. Letters not touching. Large and clear."""


def get_style_description(preset_key: str, custom_style: str = "") -> str:
    """Get the full style description from preset and/or custom input.

    Args:
        preset_key: Key from STYLE_PRESETS, or "custom" for custom only.
        custom_style: Additional custom style description.

    Returns:
        Combined style description string.
    """
    if preset_key == "custom":
        return custom_style if custom_style else "clean, readable letterforms"

    preset_desc = STYLE_PRESETS.get(preset_key, "")

    if custom_style:
        return f"{preset_desc}, {custom_style}"

    return preset_desc if preset_desc else "clean, readable letterforms"
