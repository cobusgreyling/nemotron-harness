"""Audio encoding utilities for Nemotron Harness."""

import base64
import os

AUDIO_MIME_MAP = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "m4a": "audio/mp4",
}


def encode_audio(file_path: str) -> tuple[str, str]:
    """Read an audio file and return (data_url, mime_type).

    Supported formats: wav, mp3, ogg, flac, m4a.
    """
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    mime = AUDIO_MIME_MAP.get(ext, "audio/wav")
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"
    return data_url, mime


def build_audio_message(file_path: str, prompt: str) -> dict:
    """Build an OpenAI-format message with audio input.

    Args:
        file_path: Path to the audio file.
        prompt: Text prompt to accompany the audio.

    Returns:
        Message dict with audio_url and text content.
    """
    data_url, _ = encode_audio(file_path)
    return {
        "role": "user",
        "content": [
            {"type": "audio_url", "audio_url": {"url": data_url}},
            {"type": "text", "text": prompt},
        ],
    }
