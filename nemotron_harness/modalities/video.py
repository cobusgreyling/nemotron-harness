"""Video encoding utilities for Nemotron Harness."""

import base64
import os

VIDEO_MIME_MAP = {
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "avi": "video/x-msvideo",
    "webm": "video/webm",
}


def encode_video(file_path: str) -> tuple[str, str]:
    """Read a video file and return (data_url, mime_type).

    Supported formats: mp4, mov, avi, webm.
    """
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    mime = VIDEO_MIME_MAP.get(ext, "video/mp4")
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"
    return data_url, mime


def build_video_message(
    file_path: str, prompt: str, *, include_audio: bool = True
) -> tuple[dict, dict]:
    """Build an OpenAI-format message with video input.

    Args:
        file_path: Path to the video file.
        prompt: Text prompt to accompany the video.
        include_audio: Whether to process the audio track in the video.

    Returns:
        Tuple of (message_dict, extra_body) — the extra_body includes
        mm_processor_kwargs for audio-in-video when requested.
    """
    data_url, _ = encode_video(file_path)
    message = {
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": data_url}},
            {"type": "text", "text": prompt},
        ],
    }
    extra_body = {}
    if include_audio:
        extra_body["mm_processor_kwargs"] = {"use_audio_in_video": True}
    return message, extra_body
