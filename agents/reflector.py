"""Reflector agent — Stage 2 (Training).

When the Evaluator returns ``fail`` or a score below the threshold, the
Reflector analyzes the failure and produces concrete improvement guidance for
the next generation.

Input JSON::

    {"instances": [{"instruction", "input", "output"}, ...], "reason": "..."}

Output JSON::

    {"feedback": "..."}
"""
from __future__ import annotations

import json
from typing import Any, Dict

from agents.base import BaseAgent


SYSTEM_PROMPT = (
    "Role: Task Reflector\n"
    "Task: Given a set of generated task instances and the evaluator's reason for a failing or low score, "
    "analyze the root causes of the failure and provide concrete, actionable guidance for improving the next generation.\n"
    "Requirements: Return only a 'feedback' string describing the problems and specific improvement suggestions."
)

_OUTPUT_FORMAT_HINT = (
    'Output your answer as a single JSON object like: {"feedback": "..."}.'
)


class ReflectorAgent(BaseAgent):
    name = "reflector"

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        instances = input_data.get("instances") or []
        reason = input_data.get("reason", "")

        user_prompt = (
            f"Evaluator reason:\n{reason}\n\n"
            "Generated task instances:\n"
            f"{json.dumps(instances, ensure_ascii=False, indent=2)}\n\n"
            f"{_OUTPUT_FORMAT_HINT}"
        )
        result = self.llm.chat_json(SYSTEM_PROMPT, user_prompt)

        feedback = result.get("feedback")
        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError(f"[reflector] LLM output missing 'feedback': {result}")
        return {"feedback": feedback.strip()}
