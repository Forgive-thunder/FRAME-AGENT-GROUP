"""Evaluator agent — Stage 2 (Training).

Evaluates a set of generated task instances across three dimensions:
instruction difficulty, diversity, and accuracy.

Input JSON::

    {"instances": [{"instruction", "input", "output"}, ...]}

Output JSON::

    {"score": 8.5, "reason": "...", "verdict": "pass"}   # verdict: "pass" | "fail"
"""
from __future__ import annotations

import json
from typing import Any, Dict

from agents.base import BaseAgent


SYSTEM_PROMPT = (
    "Role: Task Evaluator\n"
    "Task: Evaluate a set of generated task instances along three dimensions: "
    "instruction difficulty, diversity, and accuracy.\n"
    "Requirements: Score the set on a 0-10 scale (maximum 10). "
    "Set 'verdict' to exactly 'pass' or 'fail' and provide a concise reason."
)

_OUTPUT_FORMAT_HINT = (
    'Output your answer as a single JSON object like: {"score": 8.5, "reason": "...", "verdict": "pass"}.'
)


class EvaluatorAgent(BaseAgent):
    name = "evaluator"

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        instances = input_data.get("instances") or []

        user_prompt = (
            "Generated task instances:\n"
            f"{json.dumps(instances, ensure_ascii=False, indent=2)}\n\n"
            f"{_OUTPUT_FORMAT_HINT}"
        )
        result = self.llm.chat_json(SYSTEM_PROMPT, user_prompt)

        score = self._coerce_score(result.get("score"))
        verdict = str(result.get("verdict", "")).strip().lower()
        if verdict not in ("pass", "fail"):
            verdict = "pass" if score >= 8.0 else "fail"
        reason = str(result.get("reason", "")).strip()

        return {"score": score, "reason": reason, "verdict": verdict}

    @staticmethod
    def _coerce_score(value: Any) -> float:
        try:
            score = round(float(value), 1)
        except (TypeError, ValueError):
            raise ValueError(f"[evaluator] LLM returned an invalid score: {value!r}")
        return max(0.0, min(10.0, score))
