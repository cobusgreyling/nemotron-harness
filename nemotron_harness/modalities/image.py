"""Image/screenshot encoding utilities for Nemotron Harness."""

import base64
import os

IMAGE_MIME_MAP = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
}


def encode_image_file(file_path: str) -> str:
    """Read a local image file and return a base64 data URL.

    Args:
        file_path: Path to the image file.

    Returns:
        Base64-encoded data URL string.
    """
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    mime = IMAGE_MIME_MAP.get(ext, "image/jpeg")
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_image_message(image_path: str, prompt: str) -> dict:
    """Build an OpenAI-format message with image input.

    Args:
        image_path: Path to the image file.
        prompt: Text prompt to accompany the image.

    Returns:
        Message dict with image_url and text content.
    """
    data_url = encode_image_file(image_path)
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt},
        ],
    }
