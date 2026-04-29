"""
Nemotron Harness — Runtime orchestration for NVIDIA Nemotron 3 Nano Omni.

A harness framework that wraps the Nemotron model in a ReAct loop with
adaptive context compaction, doom-loop detection, system reminder injection,
and modality-aware inference configuration.
"""

from nemotron_harness.harness import NemotronHarness
from nemotron_harness.client import get_client
from nemotron_harness.tools import ToolRegistry
from nemotron_harness.stream import create_stream, consume_stream, StreamAccumulator

__version__ = "0.1.0"

__all__ = [
    "NemotronHarness",
    "get_client",
    "ToolRegistry",
    "create_stream",
    "consume_stream",
    "StreamAccumulator",
]
