"""Checker agent — Stage 1 (Dataset Construction).

Receives the original paper paragraph and the Extractor's conclusion, and
scores how well the conclusion matches the paragraph.

Input JSON::

    {"paragraph": "...", "conclusion": "..."}

Output JSON::

    {"score": 8.5, "reason": "..."}    # score in [0, 10], step 0.1
"""
from __future__ import annotations

from typing import Any, Dict

from agents.base import BaseAgent


SYSTEM_PROMPT = (
    "Role: Paper Evaluator\n"
    "Task: Given the original paper paragraph and a conclusion extracted from it, "
    "evaluate how well the conclusion matches the original paragraph.\n"
    "Requirements: Score the match on a 0-10 scale with a step of 0.1 (one decimal place). "
    "Provide a concise reason explaining the score."
)

_OUTPUT_FORMAT_HINT = (
    'Output your answer as a single JSON object with exactly two keys, like: {"score": 8.5, "reason": "..."}.'
)


class CheckerAgent(BaseAgent):
    name = "checker"

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        paragraph = input_data["paragraph"]
        conclusion = input_data["conclusion"]

        user_prompt = (
            f"Original paragraph:\n{paragraph}\n\n"
            f"Extracted conclusion:\n{conclusion}\n\n"
            f"{_OUTPUT_FORMAT_HINT}"
        )
        result = self.llm.chat_json(SYSTEM_PROMPT, user_prompt)

        score = self._coerce_score(result.get("score"))
        reason = str(result.get("reason", "")).strip()
        return {"score": score, "reason": reason}

    @staticmethod
    def _coerce_score(value: Any) -> float:
        try:
            score = round(float(value), 1)  # enforce the 0.1 step
        except (TypeError, ValueError):
            raise ValueError(f"[checker] LLM returned an invalid score: {value!r}")
        return max(0.0, min(10.0, score))
