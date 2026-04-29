"""
Meeting Intelligence Agent — Example 01

Processes meeting recordings (audio, video, screenshots, text) through
the Nemotron Harness and extracts structured intelligence: transcripts,
action items, slide summaries, decisions, and questions.

Usage:
    export NVIDIA_API_KEY="nvapi-your-key"
    python examples/01_meeting_agent/meeting_agent.py
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nemotron_harness import NemotronHarness, ToolRegistry
from nemotron_harness.modalities import build_audio_message, build_image_message
from nemotron_harness.stream import BOLD, CYAN, RESET


# ---------------------------------------------------------------------------
# Meeting record accumulator
# ---------------------------------------------------------------------------

class MeetingRecord:
    """Accumulates structured meeting intelligence from tool calls."""

    def __init__(self):
        self.transcripts: list[dict] = []
        self.action_items: list[dict] = []
        self.slides: list[dict] = []
        self.decisions: list[dict] = []
        self.questions: list[dict] = []

    def to_markdown(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [f"# Meeting Intelligence Report", f"Generated: {now}\n"]

        if self.transcripts:
            lines.append("## Transcript\n")
            for t in self.transcripts:
                ts = t.get("timestamp", "??:??")
                speaker = t.get("speaker", "Unknown")
                text = t.get("text", "")
                lines.append(f"**[{ts}] {speaker}:** {text}\n")

        if self.slides:
            lines.append("## Slides / Screen Shares\n")
            for s in self.slides:
                num = s.get("slide_number", "?")
                title = s.get("title", "Untitled")
                lines.append(f"### Slide {num}: {title}\n")
                lines.append(f"{s.get('content_summary', '')}\n")
                if s.get("visual_elements"):
                    lines.append(f"*Visuals:* {s['visual_elements']}\n")

        if self.action_items:
            lines.append("## Action Items\n")
            for a in self.action_items:
                assignee = a.get("assignee", "Unassigned")
                action = a.get("action", "")
                deadline = a.get("deadline", "unspecified")
                lines.append(f"- [ ] **{assignee}**: {action} *(due: {deadline})*")
                if a.get("context"):
                    lines.append(f"  - Context: {a['context']}")
            lines.append("")

        if self.decisions:
            lines.append("## Key Decisions\n")
            for d in self.decisions:
                lines.append(f"- **{d.get('decision', '')}**")
                if d.get("rationale"):
                    lines.append(f"  - Rationale: {d['rationale']}")
                if d.get("participants"):
                    lines.append(f"  - Participants: {d['participants']}")
            lines.append("")

        if self.questions:
            lines.append("## Questions\n")
            for q in self.questions:
                status = "Answered" if q.get("answered") else "UNANSWERED"
                lines.append(f"- [{status}] {q.get('question', '')}")
                if q.get("asked_by"):
                    lines.append(f"  - Asked by: {q['asked_by']}")
                if q.get("answer_summary"):
                    lines.append(f"  - Answer: {q['answer_summary']}")
            lines.append("")

        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "action_items": self.action_items,
                "decisions": self.decisions,
                "questions": self.questions,
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def build_meeting_tools(record: MeetingRecord) -> ToolRegistry:
    """Create a ToolRegistry with meeting intelligence tools."""
    registry = ToolRegistry()

    @registry.register(
        name="capture_transcript",
        description=(
            "Capture a timestamped segment of the meeting transcript. "
            "Call once per distinct speaker turn."
        ),
        parameters={
            "type": "object",
            "properties": {
                "timestamp": {"type": "string", "description": "Approximate timestamp in MM:SS format"},
                "speaker": {"type": "string", "description": "Speaker name or label"},
                "text": {"type": "string", "description": "Verbatim transcription"},
            },
            "required": ["timestamp", "speaker", "text"],
            "additionalProperties": False,
        },
    )
    def capture_transcript(timestamp: str, speaker: str, text: str) -> str:
        record.transcripts.append({"timestamp": timestamp, "speaker": speaker, "text": text})
        return f"Transcript captured: [{timestamp}] {speaker}"

    @registry.register(
        name="extract_action_item",
        description="Extract an action item with assignee and optional deadline.",
        parameters={
            "type": "object",
            "properties": {
                "assignee": {"type": "string", "description": "Person responsible"},
                "action": {"type": "string", "description": "What needs to be done"},
                "deadline": {"type": "string", "description": "Deadline if mentioned"},
                "context": {"type": "string", "description": "Brief context"},
            },
            "required": ["assignee", "action"],
            "additionalProperties": False,
        },
    )
    def extract_action_item(assignee: str, action: str, deadline: str = "unspecified", context: str = "") -> str:
        record.action_items.append({"assignee": assignee, "action": action, "deadline": deadline, "context": context})
        return f"Action item: {assignee} -> {action}"

    @registry.register(
        name="summarise_slide",
        description="Summarise a slide, chart, or shared screen.",
        parameters={
            "type": "object",
            "properties": {
                "slide_number": {"type": "integer", "description": "Slide sequence number"},
                "title": {"type": "string", "description": "Slide title"},
                "content_summary": {"type": "string", "description": "Structured summary"},
                "visual_elements": {"type": "string", "description": "Charts, diagrams, images"},
            },
            "required": ["slide_number", "content_summary"],
            "additionalProperties": False,
        },
    )
    def summarise_slide(slide_number: int, content_summary: str, title: str = "Untitled", visual_elements: str = "") -> str:
        record.slides.append({"slide_number": slide_number, "title": title, "content_summary": content_summary, "visual_elements": visual_elements})
        return f"Slide {slide_number} summarised"

    @registry.register(
        name="flag_decision",
        description="Flag a key decision made during the meeting.",
        parameters={
            "type": "object",
            "properties": {
                "decision": {"type": "string", "description": "The decision"},
                "rationale": {"type": "string", "description": "Why it was made"},
                "participants": {"type": "string", "description": "Who was involved"},
            },
            "required": ["decision"],
            "additionalProperties": False,
        },
    )
    def flag_decision(decision: str, rationale: str = "", participants: str = "") -> str:
        record.decisions.append({"decision": decision, "rationale": rationale, "participants": participants})
        return f"Decision flagged: {decision[:60]}"

    @registry.register(
        name="detect_question",
        description="Detect a question and whether it was answered.",
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question"},
                "asked_by": {"type": "string", "description": "Who asked"},
                "answered": {"type": "boolean", "description": "Whether answered"},
                "answer_summary": {"type": "string", "description": "Summary of answer"},
            },
            "required": ["question", "answered"],
            "additionalProperties": False,
        },
    )
    def detect_question(question: str, answered: bool, asked_by: str = "", answer_summary: str = "") -> str:
        record.questions.append({"question": question, "asked_by": asked_by, "answered": answered, "answer_summary": answer_summary})
        status = "answered" if answered else "unanswered"
        return f"Question ({status}): {question[:60]}"

    return registry


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a meeting intelligence agent. Your job is to analyse meeting content \
and extract structured information using the tools provided.

For every piece of meeting content you receive, you MUST call the appropriate \
tools to capture the information. Use multiple tool calls in a single response \
when you identify multiple items.

Guidelines:
- capture_transcript: Call once per speaker turn. Include timestamps if \
  inferable from context.
- extract_action_item: Call whenever someone commits to doing something or \
  is assigned a task.
- summarise_slide: Call for each distinct slide or screen share visible.
- flag_decision: Call when the group agrees on a course of action.
- detect_question: Call for every question asked, noting if it was answered.

Be thorough. It is better to capture too much than to miss something important.\
"""

# ---------------------------------------------------------------------------
# Sample meeting notes for demo
# ---------------------------------------------------------------------------

SAMPLE_MEETING_NOTES = """\
Meeting: Q2 Planning Review
Date: 2026-04-24
Attendees: Sarah (PM), James (Engineering Lead), Priya (Design), Marcus (Data)

Sarah: Let's start with the Q2 roadmap. James, where are we on the auth migration?

James: We're about 60% done. The new OAuth2 flow is working in staging. Main blocker
is the legacy session cleanup — we have about 2 million stale sessions that need to
be migrated or purged before we can flip the switch.

Sarah: What's the timeline?

James: If Marcus can get me the session analytics by end of next week, I can have
the migration script ready by May 15th. Full rollout by end of May.

Marcus: I can do that. I'll pull the session data from our analytics pipeline and
have a report ready by May 2nd.

Priya: Quick question — will the new auth flow change the login UI at all? I want
to make sure we don't need new designs.

James: No, the UI stays the same. It's all backend changes. The only visible
difference is faster login times — about 200ms improvement.

Sarah: Great. Let's make a decision on the feature flag approach. Do we do a
gradual rollout or a hard cutover?

James: I'd recommend gradual. 10% of users first, monitor for a week, then ramp.

Sarah: Agreed. Let's do gradual rollout. James owns the rollout plan.

Marcus: One more thing — the dashboard latency has been spiking. Are we going to
address that this quarter?

Sarah: Good catch. James, can you look into that?

James: I can investigate, but I'll need to deprioritise something else. Can we
discuss offline?

Sarah: Yes, let's sync on that tomorrow. Anything else? No? Let's wrap up.
"""


def main():
    print(f"""
{'='*60}
  MEETING INTELLIGENCE AGENT
  Powered by Nemotron Harness + Nemotron 3 Nano Omni
{'='*60}
""")

    # Build tools and harness
    record = MeetingRecord()
    tools = build_meeting_tools(record)

    harness = NemotronHarness(
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        max_rounds=20,
        reminder_interval=5,
        doom_loop_warn=2,
        doom_loop_halt=3,
        min_tool_calls=3,
    )

    # Run the harness on sample meeting notes
    print(f"{CYAN}{BOLD}[Text]{RESET} Processing sample meeting notes...\n")
    result = harness.run(
        f"Analyse these meeting notes:\n\n{SAMPLE_MEETING_NOTES}",
        modality="text",
        tool_choice="required",
    )

    # Print the report
    print(f"\n{'='*60}")
    print(f"  GENERATED REPORT")
    print(f"{'='*60}\n")
    print(record.to_markdown())

    print(f"\n{BOLD}Stats:{RESET}")
    print(f"  Rounds executed     : {result.rounds}")
    print(f"  Total tool calls    : {result.tool_calls_total}")
    print(f"  Transcript segments : {len(record.transcripts)}")
    print(f"  Action items        : {len(record.action_items)}")
    print(f"  Slides summarised   : {len(record.slides)}")
    print(f"  Decisions flagged   : {len(record.decisions)}")
    print(f"  Questions detected  : {len(record.questions)}")
    if result.halted:
        print(f"  Halted: {result.halt_reason}")


if __name__ == "__main__":
    main()
