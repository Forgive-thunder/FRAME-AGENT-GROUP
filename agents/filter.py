"""Filter agent — Stage 3 (topic feasibility evaluation).

Before a new research topic is pursued, the Filter checks whether the topic has
sufficient historical Reflector/training-data support in the FAISS long-term
memory.  It retrieves the most relevant chunks, injects them as context, and
either approves the topic (producing an outline) or rejects it for insufficient
support.

Input JSON::

    {"topic": "<proposed research topic>"}

Output JSON::

    {
        "topic": "...",
        "supported": true | false,
        "verdict": "supported" | "insufficient",
        "reason": "...",
        "outline": "...",          # present when supported
        "retrieved": [ ... ]       # FAISS matches used as context
    }
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.base import BaseAgent


SYSTEM_PROMPT = (
    "Role: Research Topic Feasibility Filter\n"
    "Task: You are evaluating whether a proposed research topic has sufficient historical support "
    "from previously accumulated Reflector feedback and training data. Given the topic and the "
    "retrieved historical insights, judge whether the topic can be pursued.\n"
    "Requirements: If the retrieved insights are irrelevant or insufficient, return supported=false and "
    "recommend adjusting the topic. If they are sufficient, return supported=true and generate a detailed, "
    "structured research outline grounded in the retrieved insights."
)

_OUTPUT_FORMAT_HINT = (
    'Output your answer as a single JSON object like: '
    '{"supported": true, "reason": "...", "outline": "..."}.'
)


class FilterAgent(BaseAgent):
    name = "filter"

    def __init__(
        self,
        llm: Optional[Any] = None,
        memory: Optional[Any] = None,
        top_k: int = 3,
        min_score: float = 0.0,
        **config: Any,
    ) -> None:
        super().__init__(llm=llm, **config)
        self.memory = memory
        self.top_k = top_k
        self.min_score = min_score

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        topic = str(
            input_data.get("topic") or input_data.get("research_topic") or ""
        ).strip()
        if not topic:
            raise ValueError("[filter] input must contain a 'topic' string.")

        retrieved = self._retrieve(topic)
        if not retrieved:
            return self._reject(topic, retrieved, "检索不到相关历史数据，数据支撑不足，建议调整课题。")

        user_prompt = (
            f"Proposed research topic:\n{topic}\n\n"
            "Retrieved historical insights (from FAISS memory):\n"
            f"{self._format_retrieved(retrieved)}\n\n"
            f"{_OUTPUT_FORMAT_HINT}"
        )
        result = self.llm.chat_json(SYSTEM_PROMPT, user_prompt)

        supported = bool(result.get("supported", False))
        return {
            "topic": topic,
            "supported": supported,
            "verdict": "supported" if supported else "insufficient",
            "reason": str(result.get("reason", "")).strip(),
            "outline": str(result.get("outline", "")).strip(),
            "retrieved": retrieved,
        }

    # -- helpers ----------------------------------------------------------
    def _retrieve(self, topic: str) -> List[Dict[str, Any]]:
        if self.memory is None:
            return []
        try:
            results = self.memory.query_memory(topic, top_k=self.top_k)
        except Exception as exc:  # noqa: BLE001 — memory is best-effort at filter time
            print(f"[filter] memory query failed: {exc}")
            return []
        return [r for r in results if float(r.get("score", 0.0)) >= self.min_score]

    @staticmethod
    def _reject(
        topic: str,
        retrieved: List[Dict[str, Any]],
        reason: str,
    ) -> Dict[str, Any]:
        return {
            "topic": topic,
            "supported": False,
            "verdict": "insufficient",
            "reason": reason,
            "outline": "",
            "retrieved": retrieved,
        }

    @staticmethod
    def _format_retrieved(retrieved: List[Dict[str, Any]]) -> str:
        lines = []
        for i, r in enumerate(retrieved, start=1):
            lines.append(
                f"[{i}] score={float(r.get('score', 0.0)):.3f} source={r.get('source')}\n"
                f"    {str(r.get('text', ''))[:500]}"
            )
        return "\n".join(lines)
