"""Extractor agent — Stage 1 (Dataset Construction).

Prompts a real LLM to extract the "real-world problems" conclusion from a
single paper paragraph, optionally referencing historical evaluation records
from previous rounds to refine the conclusion.

Input JSON::

    {
        "paragraph": "<one article paragraph>",
        "history": [                        # optional; empty on the first round
            {"score": 7.5, "reason": "..."},
            ...
        ]
    }

Output JSON (exactly one key)::

    {"conclusion": "..."}
"""
from __future__ import annotations

from typing import Any, Dict, List

from agents.base import BaseAgent


# System prompt — kept strictly as specified.
SYSTEM_PROMPT = (
    "Role: Paper Analyst\n"
    "Task: Identify and analyze the real-world problems addressed by the provided article paragraph.\n"
    "Requirements: If the input only contains the paragraph, then extract the conclusion based on the paragraph. "
    "If the input includes historical records from previous instances, it is necessary to reference both the "
    "paragraph and the previous scores and reasons to provide a better conclusion."
)

_OUTPUT_FORMAT_HINT = (
    'Output your answer as a single JSON object with exactly one key, like: {"conclusion": "..."}.'
)


class ExtractorAgent(BaseAgent):
    name = "extractor"

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        paragraph = input_data["paragraph"]
        history = input_data.get("history") or []

        user_prompt = self._build_user_prompt(paragraph, history)
        result = self.llm.chat_json(SYSTEM_PROMPT, user_prompt)

        conclusion = result.get("conclusion")
        if not isinstance(conclusion, str) or not conclusion.strip():
            raise ValueError(f"[extractor] LLM output missing 'conclusion': {result}")
        return {"conclusion": conclusion.strip()}

    @staticmethod
    def _build_user_prompt(paragraph: str, history: List[Dict[str, Any]]) -> str:
        lines = [f"Paragraph:\n{paragraph}"]
        if history:
            lines.append("\nHistorical evaluation records (previous scores and reasons):")
            for i, record in enumerate(history, start=1):
                lines.append(
                    f"Round {i} - score: {record.get('score')}, reason: {record.get('reason')}"
                )
            lines.append(
                "\nReference both the paragraph and the records above to provide a better conclusion."
            )
        lines.append(f"\n{_OUTPUT_FORMAT_HINT}")
        return "\n".join(lines)
