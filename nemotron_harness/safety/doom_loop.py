"""
Doom-loop detection for the Nemotron Harness.

When the model calls the same tool with identical arguments multiple times
in a row, it's stuck in a reasoning loop. This module detects that pattern
and provides escalation signals to the harness.

Detection uses fingerprinting: each tool call is hashed by (name, arguments).
If the same fingerprint appears N times consecutively, the detector fires.

Two-tier escalation:
  1. Warning — inject a nudge asking the model to try a different approach.
  2. Halt — stop the loop and return control to the caller.
"""

import hashlib
import json


class DoomLoopDetector:
    """Detects repeated identical tool calls.

    Args:
        warn_threshold: Number of consecutive identical calls before warning.
        halt_threshold: Number of consecutive identical calls before halting.
    """

    def __init__(self, warn_threshold: int = 2, halt_threshold: int = 3):
        self.warn_threshold = warn_threshold
        self.halt_threshold = halt_threshold
        self._history: list[str] = []
        self._consecutive_count: int = 0
        self._last_fingerprint: str | None = None

    def _fingerprint(self, tool_calls: list[dict]) -> str:
        """Create a deterministic hash of a set of tool calls."""
        normalized = []
        for tc in sorted(tool_calls, key=lambda t: t.get("function", {}).get("name", "")):
            fn = tc.get("function", {})
            normalized.append(f"{fn.get('name', '')}:{fn.get('arguments', '')}")
        combined = "|".join(normalized)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def check(self, tool_calls: list[dict]) -> str:
        """Check a new set of tool calls for doom-loop patterns.

        Args:
            tool_calls: Tool calls from the model's response, in OpenAI format.

        Returns:
            "ok" — no loop detected.
            "warn" — consecutive repeat detected, inject a nudge.
            "halt" — too many repeats, stop the loop.
        """
        fp = self._fingerprint(tool_calls)
        self._history.append(fp)

        if fp == self._last_fingerprint:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 1
            self._last_fingerprint = fp

        if self._consecutive_count >= self.halt_threshold:
            return "halt"
        if self._consecutive_count >= self.warn_threshold:
            return "warn"
        return "ok"

    def reset(self):
        """Reset detection state."""
        self._history.clear()
        self._consecutive_count = 0
        self._last_fingerprint = None

    @property
    def warning_message(self) -> str:
        """Message to inject when a doom loop is detected at the warn level."""
        return (
            "You appear to be repeating the same tool call. "
            "Try a different approach or tool. If you believe the task "
            "is complete, respond with your final answer instead."
        )

    @property
    def halt_message(self) -> str:
        """Message returned when the loop is halted."""
        return (
            "Tool-calling loop halted after repeated identical calls. "
            "The harness stopped execution to prevent an infinite loop."
        )
