"""
Document Analyst — Example 02

Analyses documents (images of pages, screenshots, diagrams) through
the Nemotron Harness and extracts structured information: key findings,
data points, tables, and follow-up questions.

Usage:
    export NVIDIA_API_KEY="nvapi-your-key"
    python examples/02_document_analyst/document_analyst.py
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nemotron_harness import NemotronHarness, ToolRegistry
from nemotron_harness.modalities import build_image_message
from nemotron_harness.stream import BOLD, CYAN, RESET


# ---------------------------------------------------------------------------
# Document record
# ---------------------------------------------------------------------------

class DocumentRecord:
    """Accumulates structured findings from document analysis."""

    def __init__(self):
        self.key_findings: list[dict] = []
        self.data_points: list[dict] = []
        self.tables: list[dict] = []
        self.follow_ups: list[dict] = []

    def to_markdown(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [f"# Document Analysis Report", f"Generated: {now}\n"]

        if self.key_findings:
            lines.append("## Key Findings\n")
            for i, f in enumerate(self.key_findings, 1):
                lines.append(f"{i}. **{f.get('title', 'Finding')}**")
                lines.append(f"   {f.get('description', '')}")
                if f.get("confidence"):
                    lines.append(f"   *Confidence: {f['confidence']}*")
            lines.append("")

        if self.data_points:
            lines.append("## Data Points\n")
            lines.append("| Metric | Value | Source |")
            lines.append("|--------|-------|--------|")
            for d in self.data_points:
                lines.append(
                    f"| {d.get('metric', '')} | {d.get('value', '')} "
                    f"| {d.get('source', '')} |"
                )
            lines.append("")

        if self.tables:
            lines.append("## Tables Extracted\n")
            for t in self.tables:
                lines.append(f"### {t.get('title', 'Table')}\n")
                lines.append(f"{t.get('content', '')}\n")

        if self.follow_ups:
            lines.append("## Follow-Up Questions\n")
            for q in self.follow_ups:
                priority = q.get("priority", "medium")
                lines.append(f"- [{priority.upper()}] {q.get('question', '')}")
            lines.append("")

        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "key_findings": self.key_findings,
                "data_points": self.data_points,
                "tables": self.tables,
                "follow_ups": self.follow_ups,
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def build_document_tools(record: DocumentRecord) -> ToolRegistry:
    registry = ToolRegistry()

    @registry.register(
        name="extract_finding",
        description="Extract a key finding from the document.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short title for the finding"},
                "description": {"type": "string", "description": "Detailed description"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"], "description": "Confidence level"},
            },
            "required": ["title", "description"],
            "additionalProperties": False,
        },
    )
    def extract_finding(title: str, description: str, confidence: str = "medium") -> str:
        record.key_findings.append({"title": title, "description": description, "confidence": confidence})
        return f"Finding extracted: {title}"

    @registry.register(
        name="extract_data_point",
        description="Extract a specific data point, metric, or statistic.",
        parameters={
            "type": "object",
            "properties": {
                "metric": {"type": "string", "description": "Name of the metric"},
                "value": {"type": "string", "description": "The value or statistic"},
                "source": {"type": "string", "description": "Where in the document this appears"},
            },
            "required": ["metric", "value"],
            "additionalProperties": False,
        },
    )
    def extract_data_point(metric: str, value: str, source: str = "") -> str:
        record.data_points.append({"metric": metric, "value": value, "source": source})
        return f"Data point: {metric} = {value}"

    @registry.register(
        name="extract_table",
        description="Extract a table or structured data from the document.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Table title or caption"},
                "content": {"type": "string", "description": "Table content in Markdown format"},
            },
            "required": ["title", "content"],
            "additionalProperties": False,
        },
    )
    def extract_table(title: str, content: str) -> str:
        record.tables.append({"title": title, "content": content})
        return f"Table extracted: {title}"

    @registry.register(
        name="flag_follow_up",
        description="Flag a question or gap that requires follow-up.",
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The follow-up question"},
                "priority": {"type": "string", "enum": ["high", "medium", "low"], "description": "Priority level"},
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    )
    def flag_follow_up(question: str, priority: str = "medium") -> str:
        record.follow_ups.append({"question": question, "priority": priority})
        return f"Follow-up flagged: {question[:60]}"

    return registry


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a document analyst. Your job is to examine documents and extract \
structured information using the tools provided.

For every document you receive, you MUST call the appropriate tools:
- extract_finding: For key insights, conclusions, or notable observations.
- extract_data_point: For specific numbers, metrics, percentages, or statistics.
- extract_table: For any tabular data visible in the document.
- flag_follow_up: For gaps, ambiguities, or questions the document raises.

Be thorough and precise. Cite where in the document each item appears.\
"""

SAMPLE_DOCUMENT = """\
Q2 2026 Revenue Report — Acme Corp (CONFIDENTIAL)

Executive Summary:
Revenue grew 15% YoY to $142M in Q2, driven primarily by the Enterprise
segment (+23% YoY). Consumer revenue was flat at $38M. Gross margin
improved 2 points to 67% due to infrastructure cost optimisation.

Key Metrics:
  Total Revenue:     $142M  (+15% YoY)
  Enterprise:        $104M  (+23% YoY)
  Consumer:           $38M  (+0% YoY)
  Gross Margin:       67%   (+2pp YoY)
  Operating Margin:   21%   (+3pp YoY)
  Net Income:         $18M  (+28% YoY)
  Customer Count:    4,200  (+12% YoY)
  NRR:               118%   (unchanged)

Risks:
  - Consumer segment stagnation: 3 consecutive flat quarters.
  - Key enterprise contract renewal (MegaCorp, $12M ARR) due Aug 2026.
  - Engineering headcount 15% below hiring plan — affecting roadmap.

Board Discussion Items:
  1. Should we sunset the consumer product line?
  2. MegaCorp retention strategy — discount vs. feature commitment?
  3. Engineering hiring: increase comp bands or accept slower delivery?
"""


def main():
    print(f"""
{'='*60}
  DOCUMENT ANALYST
  Powered by Nemotron Harness + Nemotron 3 Nano Omni
{'='*60}
""")

    record = DocumentRecord()
    tools = build_document_tools(record)

    harness = NemotronHarness(
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        max_rounds=15,
        min_tool_calls=3,
    )

    print(f"{CYAN}{BOLD}[Text]{RESET} Analysing sample revenue report...\n")
    result = harness.run(
        f"Analyse this document:\n\n{SAMPLE_DOCUMENT}",
        modality="text",
        tool_choice="required",
    )

    print(f"\n{'='*60}")
    print(f"  ANALYSIS REPORT")
    print(f"{'='*60}\n")
    print(record.to_markdown())

    print(f"\n{BOLD}Stats:{RESET}")
    print(f"  Rounds executed  : {result.rounds}")
    print(f"  Total tool calls : {result.tool_calls_total}")
    print(f"  Key findings     : {len(record.key_findings)}")
    print(f"  Data points      : {len(record.data_points)}")
    print(f"  Tables           : {len(record.tables)}")
    print(f"  Follow-ups       : {len(record.follow_ups)}")


if __name__ == "__main__":
    main()
