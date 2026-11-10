"""Generator agent — Stage 2 (Training).

Reads the Stage 1 ``final_conclusion`` set and generates instruction-following
question-answer pairs (task instances) for fine-tuning an LLM.  If
``feedback_history`` is provided, the generation strategy is adjusted to
address it.

Input JSON::

    {
        "final_conclusions": ["...", "..."],
        "feedback_history": ["<feedback>", ...]   # optional; empty on the first round
    }

Output JSON::

    {"instances": [{"instruction": "...", "input": "...", "output": "..."}, ...]}
"""
from __future__ import annotations

from typing import Any, Dict, List

from agents.base import BaseAgent


SYSTEM_PROMPT = (
    "Role: Task Generator\n"
    "Task: Given a set of conclusions extracted from research papers, generate high-quality "
    "instruction-following question-answer pairs (task instances) suitable for fine-tuning a large language model.\n"
    "Requirements: Each instance must contain an 'instruction', an 'input', and an 'output'. "
    "Make the questions diverse and grounded in the given conclusions. "
    "If historical feedback is provided, adjust the generation strategy to address it."
)

_OUTPUT_FORMAT_HINT = (
    'Output your answer as a single JSON object like: '
    '{"instances": [{"instruction": "...", "input": "...", "output": "..."}]}.'
)


class GeneratorAgent(BaseAgent):
    name = "generator"

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        conclusions = input_data.get("final_conclusions") or []
        feedback_history = input_data.get("feedback_history") or []

        user_prompt = self._build_user_prompt(conclusions, feedback_history)
        result = self.llm.chat_json(SYSTEM_PROMPT, user_prompt)

        instances = result.get("instances")
        if not isinstance(instances, list):
            raise ValueError(f"[generator] LLM output missing 'instances' list: {result}")

        normalized = [
            {
                "instruction": str(inst.get("instruction", "")).strip(),
                "input": str(inst.get("input", "")).strip(),
                "output": str(inst.get("output", "")).strip(),
            }
            for inst in instances
            if isinstance(inst, dict)
        ]
        return {"instances": normalized}

    @staticmethod
    def _build_user_prompt(
        conclusions: List[str],
        feedback_history: List[str],
    ) -> str:
        lines = ["Final conclusions (from Stage 1):"]
        for i, conclusion in enumerate(conclusions, start=1):
            lines.append(f"{i}. {conclusion}")

        if feedback_history:
            lines.append("\nHistorical feedback (adjust your generation strategy accordingly):")
            for i, feedback in enumerate(feedback_history, start=1):
                lines.append(f"- {feedback}")

        lines.append(f"\n{_OUTPUT_FORMAT_HINT}")
        return "\n".join(lines)
