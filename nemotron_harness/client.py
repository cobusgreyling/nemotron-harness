"""
NVIDIA NIM client setup for Nemotron 3 Nano Omni.

Uses the OpenAI-compatible API endpoint provided by NVIDIA's
API catalog / NIM infrastructure.
"""

import os
from openai import OpenAI


BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "private/nvidia/nemotron-3-nano-omni-reasoning-30b-a3b"


def get_client(api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    """Return an OpenAI client configured for NVIDIA NIM.

    Args:
        api_key: NVIDIA API key. Falls back to NVIDIA_API_KEY env var.
        base_url: Override the default NIM endpoint.

    Returns:
        Configured OpenAI client.

    Raises:
        ValueError: If no API key is provided or found in the environment.
    """
    key = api_key or os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        raise ValueError(
            "Set NVIDIA_API_KEY env var or pass api_key to get_client()"
        )
    return OpenAI(
        base_url=base_url or BASE_URL,
        api_key=key,
        default_headers={"NVCF-POLL-SECONDS": "1800"},
    )
