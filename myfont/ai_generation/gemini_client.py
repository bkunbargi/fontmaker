"""Google Gemini API client for AI glyph generation."""

import io
import base64
from typing import Optional, Tuple, Dict, Any
import numpy as np

# Check if google-genai is available
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    types = None


class GeminiError(Exception):
    """Exception raised for Gemini API errors."""
    pass


class GeminiClient:
    """Client for generating glyph sheets using Google Gemini's image generation."""

    # Gemini 3 Pro Image for native image generation
    MODEL_ID = "gemini-3-pro-image-preview"

    def __init__(self, api_key: str):
        """Initialize the Gemini client.

        Args:
            api_key: Google AI API key.

        Raises:
            GeminiError: If google-genai package is not installed.
        """
        if not GEMINI_AVAILABLE:
            raise GeminiError(
                "google-genai package is not installed. "
                "Install it with: pip install google-genai"
            )

        self.api_key = api_key
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Get or create the Gemini client instance."""
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate_glyph_sheet(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate a glyph sheet image using Gemini 2.0 Flash.

        Args:
            prompt: The generation prompt describing the desired glyph sheet.
            aspect_ratio: Image aspect ratio (ignored for Gemini, kept for API compatibility).

        Returns:
            Tuple of (image as BGR numpy array, metadata dict).

        Raises:
            GeminiError: If generation fails.
        """
        try:
            client = self._get_client()

            # Use Gemini 2.0 Flash with image generation capability
            response = client.models.generate_content(
                model=self.MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["image", "text"],
                ),
            )

            # Extract image from response
            image_part = None
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None and part.inline_data.mime_type.startswith("image/"):
                    image_part = part
                    break

            if image_part is None:
                raise GeminiError("No image was generated. The model returned text only.")

            # Decode the image data
            image_data = image_part.inline_data.data

            # Convert to PIL Image
            from PIL import Image
            pil_image = Image.open(io.BytesIO(image_data))

            # Convert PIL to numpy array (RGB)
            rgb_array = np.array(pil_image.convert("RGB"))

            # Convert RGB to BGR for OpenCV compatibility
            bgr_array = rgb_array[:, :, ::-1]

            metadata = {
                "model": self.MODEL_ID,
                "aspect_ratio": aspect_ratio,
                "width": bgr_array.shape[1],
                "height": bgr_array.shape[0],
            }

            return bgr_array, metadata

        except Exception as e:
            if isinstance(e, GeminiError):
                raise
            raise GeminiError(f"Failed to generate image: {str(e)}") from e

    def validate_api_key(self) -> bool:
        """Validate that the API key is working.

        Returns:
            True if the API key is valid, False otherwise.
        """
        try:
            # Try to create client - this will fail with invalid key
            self._get_client()
            return True
        except Exception:
            return False
