"""
Nemotron Harness — Core runtime orchestration.

Wraps NVIDIA Nemotron 3 Nano Omni in a ReAct loop with:
  - Adaptive context compaction (5-stage)
  - Doom-loop detection with 2-tier escalation
  - System reminder injection to counter instruction fade-out
  - Modality-aware inference configuration
  - Premature completion detection

The model is a single function call inside this loop. Everything
that makes the agent actually work lives here.
"""

import json

from nemotron_harness.client import get_client
from nemotron_harness.stream import (
    create_stream,
    consume_stream,
    StreamAccumulator,
    BOLD,
    CYAN,
    DIM,
    RESET,
)
from nemotron_harness.tools import ToolRegistry
from nemotron_harness.compaction import ContextCompactor
from nemotron_harness.reminders import ReminderInjector
from nemotron_harness.safety.doom_loop import DoomLoopDetector
from nemotron_harness.safety.completion import CompletionChecker


# ---------------------------------------------------------------------------
# Modality inference constraints (from the Nano Omni spec)
# ---------------------------------------------------------------------------
# Modality     enable_thinking    temperature    Notes
# ---------------------------------------------------------------
# Text         true / false       0.6            Reasoning optional
# Image        true / false       0.6            Full reasoning
# Audio        false (required)   0 (required)   No reasoning on audio
# Video        false (required)   0 (required)   use_audio_in_video flag
# Tool call    true / false       0.6            Works with all modalities

MODALITY_CONFIG = {
    "text":  {"enable_thinking": True,  "temperature": 0.6},
    "image": {"enable_thinking": True,  "temperature": 0.6},
    "audio": {"enable_thinking": False, "temperature": 0.0},
    "video": {"enable_thinking": False, "temperature": 0.0},
}


class HarnessResult:
    """Result returned by a harness run.

    Attributes:
        content: The model's final text output.
        rounds: Number of tool-calling rounds executed.
        tool_calls_total: Total number of tool calls made.
        tool_results: List of (tool_name, arguments, result) tuples.
        halted: Whether the loop was halted by a safety check.
        halt_reason: Reason for halting, if applicable.
    """

    def __init__(self):
        self.content: str = ""
        self.rounds: int = 0
        self.tool_calls_total: int = 0
        self.tool_results: list[tuple[str, dict, str]] = []
        self.halted: bool = False
        self.halt_reason: str = ""


class NemotronHarness:
    """Runtime harness for NVIDIA Nemotron 3 Nano Omni.

    Orchestrates the model in a multi-turn ReAct loop with harness-level
    intelligence: compaction, doom-loop detection, reminders, and
    modality-aware inference.

    Args:
        tools: ToolRegistry with registered tool definitions and handlers.
        system_prompt: The system prompt for the agent.
        api_key: NVIDIA API key. Falls back to NVIDIA_API_KEY env var.
        max_rounds: Maximum tool-calling rounds before stopping.
        reminder_interval: Inject system reminders every N rounds.
        reminder_text: Custom reminder text (optional).
        doom_loop_warn: Consecutive identical calls before warning.
        doom_loop_halt: Consecutive identical calls before halting.
        min_tool_calls: Minimum expected tool calls before accepting completion.
        max_context_tokens: Context window budget for compaction.
        verbose: Print status info to the terminal.
    """

    def __init__(
        self,
        tools: ToolRegistry,
        system_prompt: str,
        *,
        api_key: str | None = None,
        max_rounds: int = 20,
        reminder_interval: int = 5,
        reminder_text: str | None = None,
        doom_loop_warn: int = 2,
        doom_loop_halt: int = 3,
        min_tool_calls: int = 0,
        max_context_tokens: int = 16384,
        verbose: bool = True,
    ):
        self.client = get_client(api_key)
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_rounds = max_rounds
        self.verbose = verbose

        # Harness components
        self._compactor = ContextCompactor(max_tokens=max_context_tokens)
        self._reminder = ReminderInjector(
            interval=reminder_interval,
            reminder_text=reminder_text,
        )
        self._doom_detector = DoomLoopDetector(
            warn_threshold=doom_loop_warn,
            halt_threshold=doom_loop_halt,
        )
        self._completion_checker = CompletionChecker(
            min_tool_calls=min_tool_calls,
        )

    def run(
        self,
        user_input: str | dict,
        *,
        modality: str = "text",
        extra_body: dict | None = None,
        tool_choice: str = "auto",
    ) -> HarnessResult:
        """Execute a full harness loop for a user input.

        Args:
            user_input: Either a text string or a pre-built message dict
                (e.g. from build_audio_message / build_image_message).
            modality: Input modality — "text", "image", "audio", or "video".
                Controls inference parameters per the Nano Omni spec.
            extra_body: Additional request body parameters.
            tool_choice: Tool selection strategy for the first round.

        Returns:
            HarnessResult with the final output and execution stats.
        """
        result = HarnessResult()

        # Reset safety state
        self._doom_detector.reset()
        self._completion_checker.reset()

        # Build the initial message list
        messages = [{"role": "system", "content": self.system_prompt}]
        if isinstance(user_input, str):
            messages.append({"role": "user", "content": user_input})
        else:
            messages.append(user_input)

        # Get modality-aware inference config
        mod_config = MODALITY_CONFIG.get(modality, MODALITY_CONFIG["text"])

        for round_num in range(self.max_rounds):
            # --- 1. Check context pressure → compact if needed ---
            messages = self._compactor.compact(messages, verbose=self.verbose)

            # --- 2. Inject system reminder if instructions are fading ---
            if self._reminder.should_inject(round_num):
                reminder = self._reminder.build_reminder(self.system_prompt)
                messages.append(reminder)
                if self.verbose:
                    print(
                        f"{DIM}[Reminder] Injected at round "
                        f"{round_num}{RESET}"
                    )

            # --- 3. Call the model with modality-aware params ---
            # Only use thinking on round 0; disable for subsequent rounds
            enable_thinking = mod_config["enable_thinking"] if round_num == 0 else False
            temperature = mod_config["temperature"]

            response = create_stream(
                self.client,
                messages,
                enable_thinking=enable_thinking,
                temperature=temperature,
                tools=self.tools.definitions if self.tools else None,
                tool_choice=tool_choice if round_num == 0 else "auto",
                extra_body=extra_body,
            )
            acc = consume_stream(
                response,
                show_output=self.verbose,
                show_tool_deltas=self.verbose,
            )

            result.rounds = round_num + 1

            # --- 4. No tool calls → check for premature completion ---
            if not acc.has_tool_calls():
                if self._completion_checker.check(acc.content, False):
                    if self.verbose:
                        print(
                            f"\n{DIM}[Completion Check] "
                            f"Premature — nudging{RESET}"
                        )
                    messages.append({
                        "role": "assistant",
                        "content": acc.content,
                    })
                    messages.append({
                        "role": "user",
                        "content": self._completion_checker.nudge_message,
                    })
                    continue

                result.content = acc.content
                break

            # --- 5. Check for doom loops ---
            doom_status = self._doom_detector.check(acc.get_tool_calls())
            if doom_status == "halt":
                if self.verbose:
                    print(
                        f"\n{BOLD}[Doom Loop] HALT — "
                        f"stopping execution{RESET}"
                    )
                result.content = self._doom_detector.halt_message
                result.halted = True
                result.halt_reason = "doom_loop"
                break

            if doom_status == "warn":
                if self.verbose:
                    print(
                        f"\n{DIM}[Doom Loop] Warning — "
                        f"injecting nudge{RESET}"
                    )

            # --- 6. Execute tools and feed results back ---
            assistant_msg = {
                "role": "assistant",
                "content": acc.content or None,
                "tool_calls": acc.get_tool_calls(),
            }
            messages.append(assistant_msg)

            if self.verbose:
                print(
                    f"\n  Round {round_num + 1}: "
                    f"{len(acc.get_tool_calls())} tool call(s)"
                )

            for tc in acc.get_tool_calls():
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    if self.verbose:
                        print(f"    Warning: Could not parse arguments for {name}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": "parse error",
                    })
                    continue

                tool_result = self.tools.execute(name, args)
                result.tool_results.append((name, args, tool_result))
                result.tool_calls_total += 1
                self._completion_checker.record_tool_calls(1)

                if self.verbose:
                    print(f"    {tool_result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })

            # Inject doom-loop warning if needed
            if doom_status == "warn":
                messages.append({
                    "role": "user",
                    "content": self._doom_detector.warning_message,
                })
            else:
                messages.append({
                    "role": "user",
                    "content": (
                        "Continue calling tools for any remaining items. "
                        "When done, provide your final response."
                    ),
                })

        return result

    def chat(
        self,
        messages: list[dict],
        *,
        modality: str = "text",
        extra_body: dict | None = None,
    ) -> StreamAccumulator:
        """Single-turn inference without the tool loop.

        Useful for summary generation or follow-up questions where
        you don't need the full harness loop.

        Args:
            messages: Full message list including system prompt.
            modality: Input modality for inference config.
            extra_body: Additional request body parameters.

        Returns:
            StreamAccumulator with the response.
        """
        mod_config = MODALITY_CONFIG.get(modality, MODALITY_CONFIG["text"])
        response = create_stream(
            self.client,
            messages,
            enable_thinking=mod_config["enable_thinking"],
            temperature=mod_config["temperature"],
            extra_body=extra_body,
        )
        return consume_stream(response, show_output=self.verbose)
