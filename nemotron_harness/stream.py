"""
Streaming helpers for the Nemotron 3 Nano Omni API.

Handles reasoning traces, content deltas, and tool-call streaming
with coloured terminal output.
"""

import json
from openai import OpenAI

from nemotron_harness.client import MODEL

# ---------------------------------------------------------------------------
# Terminal colours
# ---------------------------------------------------------------------------
GREEN = "\033[32m"
ORANGE = "\033[38;5;208m"
CYAN = "\033[36m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Stream creation
# ---------------------------------------------------------------------------

def create_stream(
    client: OpenAI,
    messages: list[dict],
    *,
    model: str | None = None,
    enable_thinking: bool = False,
    reasoning_budget: int | None = None,
    tools: list[dict] | None = None,
    tool_choice: str = "auto",
    temperature: float = 0.6,
    max_tokens: int = 16384,
    top_p: float = 0.95,
    extra_body: dict | None = None,
):
    """Create a streaming chat completion against the Nano Omni endpoint.

    Args:
        client: OpenAI client configured for NVIDIA NIM.
        messages: Conversation messages.
        model: Model ID override. Defaults to Nemotron 3 Nano Omni.
        enable_thinking: Whether to enable the reasoning trace.
        reasoning_budget: Optional token budget for reasoning.
        tools: Tool definitions in OpenAI function-calling format.
        tool_choice: Tool selection strategy ("auto", "required", "none").
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling threshold.
        extra_body: Additional request body parameters.

    Returns:
        Streaming response iterator.
    """
    body = dict(extra_body or {})
    body.setdefault("chat_template_kwargs", {})["enable_thinking"] = enable_thinking
    if reasoning_budget is not None:
        body["reasoning_budget"] = reasoning_budget

    kwargs = dict(
        model=model or MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
        extra_body=body,
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    return client.chat.completions.create(**kwargs)


# ---------------------------------------------------------------------------
# Stream accumulator
# ---------------------------------------------------------------------------

class StreamAccumulator:
    """Collects content, reasoning, and tool calls from a streaming response."""

    def __init__(self):
        self.content: str = ""
        self.reasoning: str = ""
        self.tool_calls: dict[int, dict] = {}  # index -> {id, name, arguments}
        self.finish_reason: str | None = None

    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def get_tool_calls(self) -> list[dict]:
        """Return tool calls in OpenAI message format."""
        calls = []
        for idx in sorted(self.tool_calls):
            tc = self.tool_calls[idx]
            calls.append({
                "id": tc.get("id", f"call_{idx}"),
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                },
            })
        return calls

    @property
    def token_estimate(self) -> int:
        """Rough token count estimate for the accumulated content."""
        total = len(self.content) + len(self.reasoning)
        for tc in self.tool_calls.values():
            total += len(tc.get("arguments", ""))
        return total // 4  # rough chars-to-tokens


# ---------------------------------------------------------------------------
# Stream consumer
# ---------------------------------------------------------------------------

def consume_stream(
    response,
    *,
    show_output: bool = True,
    show_tool_deltas: bool = False,
    show_reasoning: bool = True,
) -> StreamAccumulator:
    """Consume a streaming response, printing coloured output to the terminal.

    Args:
        response: Streaming response from create_stream.
        show_output: Whether to print content to the terminal.
        show_tool_deltas: Whether to print tool call arguments as they stream.
        show_reasoning: Whether to print reasoning tokens.

    Returns:
        StreamAccumulator with the full content, reasoning, and tool calls.
    """
    acc = StreamAccumulator()
    in_reasoning = False

    for chunk in response:
        choice = chunk.choices[0] if chunk.choices else None
        if choice is None:
            continue

        delta = choice.delta
        acc.finish_reason = choice.finish_reason or acc.finish_reason

        # --- Reasoning tokens ---
        reasoning_text = (
            getattr(delta, "reasoning", None)
            or getattr(delta, "reasoning_content", None)
        )
        if reasoning_text:
            acc.reasoning += reasoning_text
            if show_output and show_reasoning:
                if not in_reasoning:
                    print(f"\n{GREEN}{BOLD}[Reasoning]{RESET} {GREEN}", end="")
                    in_reasoning = True
                print(reasoning_text, end="", flush=True)

        # --- Content tokens ---
        if delta.content:
            if in_reasoning and show_output:
                print(f"{RESET}\n\n{BOLD}[Answer]{RESET} ", end="")
                in_reasoning = False
            acc.content += delta.content
            if show_output:
                print(delta.content, end="", flush=True)

        # --- Tool-call deltas ---
        if delta.tool_calls:
            if in_reasoning and show_output:
                print(RESET, end="")
                in_reasoning = False
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in acc.tool_calls:
                    acc.tool_calls[idx] = {
                        "id": getattr(tc_delta, "id", None) or f"call_{idx}",
                        "name": "",
                        "arguments": "",
                    }
                if tc_delta.function:
                    if tc_delta.function.name:
                        acc.tool_calls[idx]["name"] = tc_delta.function.name
                        if show_output and show_tool_deltas:
                            print(
                                f"\n{ORANGE}[Tool Call] "
                                f"{tc_delta.function.name}({RESET}",
                                end="",
                            )
                    if tc_delta.function.arguments:
                        acc.tool_calls[idx]["arguments"] += (
                            tc_delta.function.arguments
                        )
                        if show_output and show_tool_deltas:
                            print(
                                f"{ORANGE}{tc_delta.function.arguments}{RESET}",
                                end="",
                                flush=True,
                            )

    if show_output:
        if in_reasoning:
            print(RESET, end="")
        print()

    return acc
