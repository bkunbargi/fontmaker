"""Character to glyph mapping management."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json

from .charset import get_glyph_name, get_unicode_codepoint, validate_character


@dataclass
class GlyphMapping:
    """Represents a mapping between a glyph and a character."""
    glyph_index: int
    character: str
    glyph_name: str
    codepoint: int


class CharacterMapper:
    """Manages character to glyph assignments."""

    def __init__(self):
        self._mappings: Dict[int, GlyphMapping] = {}  # glyph_index -> mapping
        self._char_to_glyph: Dict[str, int] = {}      # character -> glyph_index

    def assign(self, glyph_index: int, character: str) -> GlyphMapping:
        """Assign a character to a glyph.

        Args:
            glyph_index: Index of the glyph.
            character: Character to assign.

        Returns:
            The created GlyphMapping.

        Raises:
            ValueError: If character is invalid or already assigned.
        """
        if not validate_character(character):
            raise ValueError(f"Invalid character: {repr(character)}")

        # Remove existing assignment for this glyph
        if glyph_index in self._mappings:
            old_char = self._mappings[glyph_index].character
            del self._char_to_glyph[old_char]

        # Remove existing glyph for this character
        if character in self._char_to_glyph:
            old_glyph = self._char_to_glyph[character]
            del self._mappings[old_glyph]

        # Create new mapping
        mapping = GlyphMapping(
            glyph_index=glyph_index,
            character=character,
            glyph_name=get_glyph_name(character),
            codepoint=get_unicode_codepoint(character)
        )

        self._mappings[glyph_index] = mapping
        self._char_to_glyph[character] = glyph_index

        return mapping

    def unassign(self, glyph_index: int) -> Optional[GlyphMapping]:
        """Remove assignment for a glyph.

        Args:
            glyph_index: Index of the glyph.

        Returns:
            The removed mapping, or None if not found.
        """
        if glyph_index not in self._mappings:
            return None

        mapping = self._mappings[glyph_index]
        del self._char_to_glyph[mapping.character]
        del self._mappings[glyph_index]

        return mapping

    def get_by_glyph(self, glyph_index: int) -> Optional[GlyphMapping]:
        """Get mapping for a glyph index.

        Args:
            glyph_index: Index of the glyph.

        Returns:
            GlyphMapping or None.
        """
        return self._mappings.get(glyph_index)

    def get_by_character(self, character: str) -> Optional[GlyphMapping]:
        """Get mapping for a character.

        Args:
            character: Character to look up.

        Returns:
            GlyphMapping or None.
        """
        glyph_index = self._char_to_glyph.get(character)
        if glyph_index is not None:
            return self._mappings.get(glyph_index)
        return None

    def get_all_mappings(self) -> List[GlyphMapping]:
        """Get all mappings sorted by glyph index.

        Returns:
            List of GlyphMappings.
        """
        return sorted(self._mappings.values(), key=lambda m: m.glyph_index)

    def get_cmap_entries(self) -> Dict[int, str]:
        """Get character map entries for font generation.

        Returns:
            Dictionary of codepoint -> glyph_name.
        """
        return {m.codepoint: m.glyph_name for m in self._mappings.values()}

    def is_assigned(self, glyph_index: int) -> bool:
        """Check if a glyph has an assigned character.

        Args:
            glyph_index: Index to check.

        Returns:
            True if assigned.
        """
        return glyph_index in self._mappings

    def is_character_used(self, character: str) -> bool:
        """Check if a character is already assigned.

        Args:
            character: Character to check.

        Returns:
            True if used.
        """
        return character in self._char_to_glyph

    def clear(self):
        """Remove all mappings."""
        self._mappings.clear()
        self._char_to_glyph.clear()

    def count(self) -> int:
        """Get number of mappings.

        Returns:
            Count of mappings.
        """
        return len(self._mappings)

    def auto_assign(self, glyph_indices: List[int], characters: List[str]) -> int:
        """Auto-assign characters to glyphs in order.

        Args:
            glyph_indices: List of glyph indices.
            characters: List of characters to assign.

        Returns:
            Number of assignments made.
        """
        count = 0
        for glyph_idx, char in zip(glyph_indices, characters):
            try:
                self.assign(glyph_idx, char)
                count += 1
            except ValueError:
                continue
        return count

    def to_dict(self) -> Dict[str, Any]:
        """Serialize mappings to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            'mappings': [
                {
                    'glyph_index': m.glyph_index,
                    'character': m.character,
                    'glyph_name': m.glyph_name,
                    'codepoint': m.codepoint
                }
                for m in self._mappings.values()
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CharacterMapper':
        """Deserialize mappings from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            CharacterMapper instance.
        """
        mapper = cls()
        for m in data.get('mappings', []):
            mapper.assign(m['glyph_index'], m['character'])
        return mapper

    def to_json(self) -> str:
        """Serialize mappings to JSON string.

        Returns:
            JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'CharacterMapper':
        """Deserialize mappings from JSON string.

        Args:
            json_str: JSON string.

        Returns:
            CharacterMapper instance.
        """
        return cls.from_dict(json.loads(json_str))
