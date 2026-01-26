"""Character set definitions for font generation."""

import string

# Basic character sets
UPPERCASE = list(string.ascii_uppercase)  # A-Z
LOWERCASE = list(string.ascii_lowercase)  # a-z
DIGITS = list(string.digits)               # 0-9

# Common symbols
SYMBOLS = list('!@#$%^&*()_+-=[]{}|;:\'",.<>/?`~')

# Punctuation subset
PUNCTUATION = list('.,:;!?\'"-()[]{}')

# Common symbols for basic fonts
BASIC_SYMBOLS = list(' .,:;!?\'"-')

# Full ASCII printable characters (excluding space which is separate)
ASCII_PRINTABLE = [chr(i) for i in range(33, 127)]

# Combined character sets
ALL_LETTERS = UPPERCASE + LOWERCASE
ALL_ALPHANUMERIC = ALL_LETTERS + DIGITS
ALL_CHARACTERS = ALL_ALPHANUMERIC + SYMBOLS

# Character set presets
CHARSETS = {
    'uppercase': UPPERCASE,
    'lowercase': LOWERCASE,
    'letters': ALL_LETTERS,
    'digits': DIGITS,
    'alphanumeric': ALL_ALPHANUMERIC,
    'basic': UPPERCASE + DIGITS + BASIC_SYMBOLS,
    'full': ALL_CHARACTERS,
    'ascii': ASCII_PRINTABLE,
}


def get_charset(name: str) -> list:
    """Get a predefined character set by name.

    Args:
        name: Character set name ('uppercase', 'lowercase', 'letters',
              'digits', 'alphanumeric', 'basic', 'full', 'ascii').

    Returns:
        List of characters.

    Raises:
        ValueError: If charset name is unknown.
    """
    if name not in CHARSETS:
        available = ', '.join(CHARSETS.keys())
        raise ValueError(f"Unknown charset '{name}'. Available: {available}")
    return CHARSETS[name].copy()


def get_unicode_codepoint(char: str) -> int:
    """Get the Unicode codepoint for a character.

    Args:
        char: Single character.

    Returns:
        Unicode codepoint as integer.
    """
    return ord(char)


def get_glyph_name(char: str) -> str:
    """Get the standard glyph name for a character.

    Args:
        char: Single character.

    Returns:
        Standard glyph name (e.g., 'A', 'a', 'one', 'period').
    """
    # Standard mappings for common characters
    name_map = {
        ' ': 'space',
        '!': 'exclam',
        '"': 'quotedbl',
        '#': 'numbersign',
        '$': 'dollar',
        '%': 'percent',
        '&': 'ampersand',
        "'": 'quotesingle',
        '(': 'parenleft',
        ')': 'parenright',
        '*': 'asterisk',
        '+': 'plus',
        ',': 'comma',
        '-': 'hyphen',
        '.': 'period',
        '/': 'slash',
        '0': 'zero',
        '1': 'one',
        '2': 'two',
        '3': 'three',
        '4': 'four',
        '5': 'five',
        '6': 'six',
        '7': 'seven',
        '8': 'eight',
        '9': 'nine',
        ':': 'colon',
        ';': 'semicolon',
        '<': 'less',
        '=': 'equal',
        '>': 'greater',
        '?': 'question',
        '@': 'at',
        '[': 'bracketleft',
        '\\': 'backslash',
        ']': 'bracketright',
        '^': 'asciicircum',
        '_': 'underscore',
        '`': 'grave',
        '{': 'braceleft',
        '|': 'bar',
        '}': 'braceright',
        '~': 'asciitilde',
    }

    if char in name_map:
        return name_map[char]

    # Letters use their character as name
    if char.isalpha():
        return char

    # Fallback: use 'uniXXXX' format
    return f'uni{ord(char):04X}'


def validate_character(char: str) -> bool:
    """Check if a character is valid for font generation.

    Args:
        char: Character to validate.

    Returns:
        True if valid.
    """
    if len(char) != 1:
        return False

    codepoint = ord(char)
    # Accept printable ASCII and common Unicode ranges
    return 32 <= codepoint <= 126 or codepoint >= 160
