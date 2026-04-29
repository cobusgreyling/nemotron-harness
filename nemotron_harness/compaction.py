"""
Adaptive context compaction for the Nemotron Harness.

As conversations grow, the context window fills with tool results,
reasoning traces, and intermediate outputs. This module monitors
context pressure and applies progressively aggressive compaction
to keep the window usable.

Five stages of compaction:
  1. Trim — Remove reasoning traces from older turns.
  2. Summarise — Replace older tool results with one-line summaries.
  3. Collapse — Merge consecutive assistant+tool rounds into summaries.
  4. Prune — Drop the oldest N turns entirely.
  5. Reset — Keep only the system prompt and a summary of prior work.
"""

from nemotron_harness.stream import create_stream, consume_stream, DIM, RESET


class ContextCompactor:
    """Monitors and manages context window pressure.

    Args:
        max_tokens: Estimated context window budget (in tokens).
        stage_thresholds: Token thresholds for each compaction stage,
            expressed as fractions of max_tokens. Defaults to
            [0.5, 0.65, 0.8, 0.9, 0.95].
    """

    DEFAULT_THRESHOLDS = [0.50, 0.65, 0.80, 0.90, 0.95]
    STAGE_NAMES = ["trim", "summarise", "collapse", "prune", "reset"]

    def __init__(
        self,
        max_tokens: int = 16384,
        stage_thresholds: list[float] | None = None,
    ):
        self.max_tokens = max_tokens
        self.thresholds = stage_thresholds or self.DEFAULT_THRESHOLDS
        self._current_stage = 0

    def estimate_tokens(self, messages: list[dict]) -> int:
        """Rough token estimate for a message list."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content) // 4
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text", "")
                        total += len(text) // 4
                        # Media content is harder to estimate; use a fixed budget
                        if part.get("type") in ("image_url", "audio_url", "video_url"):
                            total += 1000
            # Tool call arguments
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                total += len(fn.get("arguments", "")) // 4
        return total

    def pressure(self, messages: list[dict]) -> float:
        """Return context pressure as a fraction of max_tokens."""
        return self.estimate_tokens(messages) / self.max_tokens

    def needed_stage(self, messages: list[dict]) -> int:
        """Determine which compaction stage is needed (0 = none)."""
        p = self.pressure(messages)
        stage = 0
        for threshold in self.thresholds:
            if p >= threshold:
                stage += 1
            else:
                break
        return stage

    def compact(self, messages: list[dict], verbose: bool = True) -> list[dict]:
        """Apply the appropriate compaction stage to the message list.

        Modifies and returns the message list. Each stage is additive —
        higher stages include the effects of lower stages.

        Args:
            messages: The conversation message list.
            verbose: Print compaction info to the terminal.

        Returns:
            The compacted message list.
        """
        stage = self.needed_stage(messages)
        if stage == 0:
            return messages

        if verbose:
            print(
                f"\n{DIM}[Compaction] Stage {stage} "
                f"({self.STAGE_NAMES[stage - 1]}) — "
                f"pressure: {self.pressure(messages):.0%}{RESET}"
            )

        if stage >= 1:
            messages = self._trim_reasoning(messages)
        if stage >= 2:
            messages = self._summarise_tool_results(messages)
        if stage >= 3:
            messages = self._collapse_rounds(messages)
        if stage >= 4:
            messages = self._prune_oldest(messages)
        if stage >= 5:
            messages = self._reset_with_summary(messages)

        self._current_stage = stage
        return messages

    def _trim_reasoning(self, messages: list[dict]) -> list[dict]:
        """Stage 1: Strip reasoning traces from older assistant messages.

        Keeps reasoning only in the last 2 assistant messages.
        """
        assistant_indices = [
            i for i, m in enumerate(messages) if m.get("role") == "assistant"
        ]
        # Keep reasoning in the last 2 assistant messages
        cutoff = assistant_indices[-2] if len(assistant_indices) >= 2 else len(messages)

        for i, msg in enumerate(messages):
            if i < cutoff and msg.get("role") == "assistant":
                msg.pop("reasoning", None)
                msg.pop("reasoning_content", None)
        return messages

    def _summarise_tool_results(self, messages: list[dict]) -> list[dict]:
        """Stage 2: Replace older tool results with brief summaries.

        Keeps full results for the last 3 tool messages.
        """
        tool_indices = [
            i for i, m in enumerate(messages) if m.get("role") == "tool"
        ]
        if len(tool_indices) <= 3:
            return messages

        for idx in tool_indices[:-3]:
            content = messages[idx].get("content", "")
            if len(content) > 200:
                messages[idx]["content"] = content[:100] + "... [truncated]"
        return messages

    def _collapse_rounds(self, messages: list[dict]) -> list[dict]:
        """Stage 3: Merge older assistant+tool sequences into summaries."""
        if len(messages) <= 6:
            return messages

        # Keep system prompt + last 4 messages; collapse everything in between
        preserved_head = messages[:1]  # system prompt
        preserved_tail = messages[-4:]
        middle = messages[1:-4]

        if not middle:
            return messages

        tool_count = sum(1 for m in middle if m.get("role") == "tool")
        assistant_count = sum(1 for m in middle if m.get("role") == "assistant")
        summary = (
            f"[Prior context: {assistant_count} assistant turns, "
            f"{tool_count} tool calls — details compacted to save context]"
        )
        collapsed = {"role": "user", "content": summary}
        return preserved_head + [collapsed] + preserved_tail

    def _prune_oldest(self, messages: list[dict]) -> list[dict]:
        """Stage 4: Drop older turns, keeping system prompt and recent messages."""
        if len(messages) <= 4:
            return messages
        return [messages[0]] + messages[-3:]

    def _reset_with_summary(self, messages: list[dict]) -> list[dict]:
        """Stage 5: Keep only system prompt and a final summary."""
        system = messages[0] if messages and messages[0].get("role") == "system" else None
        result = []
        if system:
            result.append(system)

        result.append({
            "role": "user",
            "content": (
                "[Context was reset due to window pressure. "
                "The previous conversation has been discarded. "
                "Continue from where you left off based on the "
                "system prompt and any new inputs provided.]"
            ),
        })
        return result
