"""Character mapping module for managing glyph-to-character assignments."""

from .charset import UPPERCASE, LOWERCASE, DIGITS, SYMBOLS, ALL_CHARACTERS
from .mapper import CharacterMapper

__all__ = [
    'UPPERCASE',
    'LOWERCASE',
    'DIGITS',
    'SYMBOLS',
    'ALL_CHARACTERS',
    'CharacterMapper',
]
