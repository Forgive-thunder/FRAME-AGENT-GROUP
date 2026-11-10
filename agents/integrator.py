"""Integrator agent — Stage 3 (Inference).

Receives the Stage 1 core conclusions and the Stage 2 QA instances, performs
cross-cutting reasoning over both, and produces a concise medical insight
report.

Input JSON::

    {
        "final_conclusions": ["...", "..."],
        "instances": [{"instruction", "input", "output"}, ...]
    }

Output JSON::

    {"final_answer": "<comprehensive medical insight report>"}
"""
from __future__ import annotations

import json
from typing import Any, Dict

from agents.base import BaseAgent


SYSTEM_PROMPT = (
    "Role: Medical Research Analyst\n"
    "Task: You are a top-tier medical research analyst. Given the core conclusions extracted in Stage 1 "
    "and the high-quality question-answer pairs generated in Stage 2, perform comprehensive, cross-cutting "
    "reasoning over this information from a global perspective and produce a concise 'Comprehensive Medical "
    "Insight Report'.\n"
    "Requirements: Synthesize the conclusions and QA pairs into a coherent, structured report written in Chinese. "
    "Highlight the key real-world medical problems, their implications, and any cross-cutting patterns."
)

_OUTPUT_FORMAT_HINT = (
    'Output your answer as a single JSON object like: {"final_answer": "..."}.'
)


class IntegratorAgent(BaseAgent):
    name = "integrator"

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        conclusions = input_data.get("final_conclusions") or []
        instances = input_data.get("instances") or []

        user_prompt = (
            "Stage 1 core conclusions:\n"
            f"{json.dumps(conclusions, ensure_ascii=False, indent=2)}\n\n"
            "Stage 2 high-quality question-answer pairs:\n"
            f"{json.dumps(instances, ensure_ascii=False, indent=2)}\n\n"
            f"{_OUTPUT_FORMAT_HINT}"
        )
        result = self.llm.chat_json(SYSTEM_PROMPT, user_prompt)

        final_answer = result.get("final_answer")
        if not isinstance(final_answer, str) or not final_answer.strip():
            raise ValueError(f"[integrator] LLM output missing 'final_answer': {result}")
        return {"final_answer": final_answer.strip()}
