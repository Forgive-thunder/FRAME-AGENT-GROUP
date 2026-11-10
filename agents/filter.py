"""Filter agent — Stage 1 (Dataset Construction).

Drops low-quality / redundant items from the refined list to produce the final
training dataset.  The stub keeps everything; the real implementation would
filter on Checker scores.

Input JSON::

    {"items": [{"id", "type", "content", "metadata"}, ...]}

Output JSON::

    {
        "kept_items": [...],
        "discarded_items": [...],
        "stats": {"total": int, "kept": int, "discarded": int}
    }
"""
from __future__ import annotations

from typing import Any, Dict

from agents.base import BaseAgent


class FilterAgent(BaseAgent):
    name = "filter"

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Superseded by the direct Extractor <-> Checker game loop in main.py.
        # Kept for parity with the original 7-agent design; not invoked now.
        raise NotImplementedError(
            "FilterAgent is not wired into the current Stage 1 game loop "
            "(see main.run_game_loop)."
        )
