"""
Premature completion detection for the Nemotron Harness.

When the model signals it's done but there are outstanding items
(e.g. tools that haven't been called, questions not addressed),
this module generates nudges to keep the model working.
"""


class CompletionChecker:
    """Checks whether the model's output indicates premature completion.

    The checker looks for completion signals in the model's content
    and compares against expected outcomes to determine if the model
    is finishing too early.

    Args:
        completion_phrases: Phrases that signal the model thinks it's done.
        min_tool_calls: Minimum number of tool calls expected before completion.
    """

    COMPLETION_PHRASES = [
        "analysis complete",
        "task complete",
        "i'm done",
        "that's everything",
        "nothing more to",
        "all items captured",
        "no further",
        "that covers",
    ]

    def __init__(
        self,
        completion_phrases: list[str] | None = None,
        min_tool_calls: int = 0,
    ):
        self.phrases = [
            p.lower() for p in (completion_phrases or self.COMPLETION_PHRASES)
        ]
        self.min_tool_calls = min_tool_calls
        self._total_tool_calls = 0

    def record_tool_calls(self, count: int):
        """Record that tool calls were executed this round."""
        self._total_tool_calls += count

    def check(self, content: str, has_tool_calls: bool) -> bool:
        """Check if the model is completing prematurely.

        Args:
            content: The model's text content for this turn.
            has_tool_calls: Whether the model also made tool calls this turn.

        Returns:
            True if the model appears to be completing prematurely.
        """
        if has_tool_calls:
            return False

        if not content:
            return False

        content_lower = content.lower()
        signals_done = any(phrase in content_lower for phrase in self.phrases)

        if signals_done and self._total_tool_calls < self.min_tool_calls:
            return True

        return False

    @property
    def nudge_message(self) -> str:
        """Message to inject when premature completion is detected."""
        remaining = self.min_tool_calls - self._total_tool_calls
        return (
            f"You indicated completion, but only {self._total_tool_calls} "
            f"tool call(s) have been made (expected at least "
            f"{self.min_tool_calls}). Please continue processing — "
            f"there may be items you haven't captured yet."
        )

    def reset(self):
        """Reset the checker state."""
        self._total_tool_calls = 0
