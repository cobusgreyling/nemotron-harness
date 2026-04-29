"""
Voice Assistant — Example 03

An audio-only conversational assistant that processes voice recordings
through the Nemotron Harness. Demonstrates the audio modality constraints
(no reasoning, temperature=0) and tool-based response structuring.

Usage:
    export NVIDIA_API_KEY="nvapi-your-key"

    # Process an audio file
    python examples/03_voice_assistant/voice_assistant.py --audio recording.wav

    # Demo mode with text simulation
    python examples/03_voice_assistant/voice_assistant.py --demo
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nemotron_harness import NemotronHarness, ToolRegistry
from nemotron_harness.modalities import build_audio_message
from nemotron_harness.stream import BOLD, CYAN, GREEN, RESET


# ---------------------------------------------------------------------------
# Conversation record
# ---------------------------------------------------------------------------

class ConversationRecord:
    """Accumulates structured output from voice conversations."""

    def __init__(self):
        self.transcripts: list[dict] = []
        self.intents: list[dict] = []
        self.entities: list[dict] = []
        self.responses: list[dict] = []

    def to_markdown(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [f"# Voice Conversation Report", f"Generated: {now}\n"]

        if self.transcripts:
            lines.append("## Transcript\n")
            for t in self.transcripts:
                speaker = t.get("speaker", "User")
                text = t.get("text", "")
                lines.append(f"**{speaker}:** {text}\n")

        if self.intents:
            lines.append("## Detected Intents\n")
            for intent in self.intents:
                conf = intent.get("confidence", "?")
                lines.append(f"- **{intent.get('intent', '')}** (confidence: {conf})")
                if intent.get("slots"):
                    lines.append(f"  - Slots: {intent['slots']}")
            lines.append("")

        if self.entities:
            lines.append("## Extracted Entities\n")
            lines.append("| Entity | Type | Value |")
            lines.append("|--------|------|-------|")
            for e in self.entities:
                lines.append(
                    f"| {e.get('text', '')} | {e.get('entity_type', '')} "
                    f"| {e.get('normalized', '')} |"
                )
            lines.append("")

        if self.responses:
            lines.append("## Suggested Responses\n")
            for r in self.responses:
                lines.append(f"- {r.get('text', '')}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def build_voice_tools(record: ConversationRecord) -> ToolRegistry:
    registry = ToolRegistry()

    @registry.register(
        name="transcribe_speech",
        description="Transcribe a segment of speech from the audio.",
        parameters={
            "type": "object",
            "properties": {
                "speaker": {"type": "string", "description": "Speaker label"},
                "text": {"type": "string", "description": "Transcribed text"},
                "language": {"type": "string", "description": "Detected language"},
            },
            "required": ["speaker", "text"],
            "additionalProperties": False,
        },
    )
    def transcribe_speech(speaker: str, text: str, language: str = "en") -> str:
        record.transcripts.append({"speaker": speaker, "text": text, "language": language})
        return f"Transcribed: {speaker}: {text[:60]}"

    @registry.register(
        name="detect_intent",
        description="Detect the user's intent from their speech.",
        parameters={
            "type": "object",
            "properties": {
                "intent": {"type": "string", "description": "Detected intent name"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"], "description": "Confidence"},
                "slots": {"type": "string", "description": "Key-value slots extracted"},
            },
            "required": ["intent", "confidence"],
            "additionalProperties": False,
        },
    )
    def detect_intent(intent: str, confidence: str, slots: str = "") -> str:
        record.intents.append({"intent": intent, "confidence": confidence, "slots": slots})
        return f"Intent: {intent} ({confidence})"

    @registry.register(
        name="extract_entity",
        description="Extract a named entity from the conversation.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Entity as mentioned"},
                "entity_type": {"type": "string", "description": "Type: person, location, date, time, number, etc."},
                "normalized": {"type": "string", "description": "Normalized value"},
            },
            "required": ["text", "entity_type"],
            "additionalProperties": False,
        },
    )
    def extract_entity(text: str, entity_type: str, normalized: str = "") -> str:
        record.entities.append({"text": text, "entity_type": entity_type, "normalized": normalized})
        return f"Entity: {text} ({entity_type})"

    @registry.register(
        name="suggest_response",
        description="Suggest a response the assistant should give.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Suggested response text"},
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    )
    def suggest_response(text: str) -> str:
        record.responses.append({"text": text})
        return f"Response queued: {text[:60]}"

    return registry


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a voice assistant processing engine. Your job is to analyse speech \
input and extract structured information using the tools provided.

For every audio input:
1. Transcribe all speech using transcribe_speech (one call per speaker turn).
2. Detect the user's intent using detect_intent.
3. Extract named entities (dates, names, locations, numbers) using extract_entity.
4. Suggest an appropriate response using suggest_response.

Be precise with transcription. Identify all entities mentioned.\
"""

SAMPLE_VOICE_INPUT = """\
[Simulated audio transcript for demo purposes]

User: Hey, can you book me a flight from Cape Town to London on June 15th?
I need to arrive before 3 PM local time. Business class if possible,
and I'd prefer British Airways. My budget is around 25,000 rand.
Oh, and I'll need a hotel near Paddington Station for three nights.
"""


def main():
    parser = argparse.ArgumentParser(description="Voice Assistant — Nemotron Harness")
    parser.add_argument("--audio", type=str, help="Audio file to process")
    parser.add_argument("--demo", action="store_true", help="Run with sample text input")
    args = parser.parse_args()

    if not args.audio and not args.demo:
        parser.print_help()
        print("\nProvide --audio FILE or --demo")
        sys.exit(1)

    print(f"""
{'='*60}
  VOICE ASSISTANT
  Powered by Nemotron Harness + Nemotron 3 Nano Omni
{'='*60}
""")

    record = ConversationRecord()
    tools = build_voice_tools(record)

    harness = NemotronHarness(
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        max_rounds=10,
        min_tool_calls=2,
    )

    if args.audio:
        print(f"{CYAN}{BOLD}[Audio]{RESET} Processing: {args.audio}\n")
        message = build_audio_message(
            args.audio,
            prompt=(
                "Listen to this audio carefully. Transcribe all speech, "
                "detect the user's intent, extract entities, and suggest a response."
            ),
        )
        result = harness.run(message, modality="audio")
    else:
        print(f"{CYAN}{BOLD}[Demo]{RESET} Processing simulated voice input...\n")
        result = harness.run(
            f"Process this voice input:\n\n{SAMPLE_VOICE_INPUT}",
            modality="text",
            tool_choice="required",
        )

    print(f"\n{'='*60}")
    print(f"  CONVERSATION REPORT")
    print(f"{'='*60}\n")
    print(record.to_markdown())

    print(f"\n{BOLD}Stats:{RESET}")
    print(f"  Rounds executed  : {result.rounds}")
    print(f"  Total tool calls : {result.tool_calls_total}")
    print(f"  Transcripts      : {len(record.transcripts)}")
    print(f"  Intents          : {len(record.intents)}")
    print(f"  Entities         : {len(record.entities)}")
    print(f"  Responses        : {len(record.responses)}")


if __name__ == "__main__":
    main()
