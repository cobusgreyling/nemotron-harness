"""
System reminder injection for the Nemotron Harness.

After many tool-calling rounds, the model's attention drifts away from
the system prompt. This module injects reminders as user-role messages
at configurable intervals to counter instruction fade-out.
"""


class ReminderInjector:
    """Injects system reminders into the conversation at regular intervals.

    The reminder is injected as a user-role message because user messages
    receive stronger attention than system messages at high token distances.

    Args:
        interval: Inject a reminder every N rounds.
        reminder_text: Custom reminder text. If None, uses a default
            that references the system prompt.
    """

    def __init__(
        self,
        interval: int = 5,
        reminder_text: str | None = None,
    ):
        self.interval = interval
        self.custom_text = reminder_text

    def should_inject(self, round_num: int) -> bool:
        """Check if a reminder should be injected at this round.

        Args:
            round_num: Current round number (0-based).

        Returns:
            True if a reminder should be injected.
        """
        if round_num == 0:
            return False
        return round_num % self.interval == 0

    def build_reminder(self, system_prompt: str | None = None) -> dict:
        """Build a reminder message.

        Args:
            system_prompt: The original system prompt, used to generate
                a contextual reminder if no custom text was provided.

        Returns:
            A user-role message dict containing the reminder.
        """
        if self.custom_text:
            text = self.custom_text
        elif system_prompt:
            text = (
                f"[System Reminder] Remember your core instructions: "
                f"{system_prompt[:300]}{'...' if len(system_prompt) > 300 else ''} "
                f"Continue following these guidelines carefully."
            )
        else:
            text = (
                "[System Reminder] Continue following your original "
                "instructions carefully. Be thorough and use the "
                "appropriate tools for each item you identify."
            )
        return {"role": "user", "content": text}
