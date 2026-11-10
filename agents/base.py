"""Base class shared by all seven FRAME agents."""
from __future__ import annotations

from typing import Any, Dict, Optional

from llm.client import LLMClient


class BaseAgent:
    """Common interface for every agent in the pipeline.

    Subclasses set a unique :attr:`name` (used by the mock LLM to pick the
    right dummy response) and implement :meth:`execute`.  ``main.py`` calls
    :meth:`run`, which logs the call and enforces the JSON-dict contract.
    """

    name: str = "base"

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        **config: Any,
    ) -> None:
        self.llm = llm if llm is not None else LLMClient()
        self.config = config

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Core logic.  Must return a JSON-serializable dict."""
        raise NotImplementedError

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Thin wrapper: log the invocation and validate the output."""
        print(f"  [{self.name}] <-- {list(input_data.keys())}")
        result = self.execute(input_data)
        self._validate(result)
        print(f"  [{self.name}] --> keys={list(result.keys())}")
        return result

    def _validate(self, result: Dict[str, Any]) -> None:
        if not isinstance(result, dict):
            raise TypeError(
                f"[{self.name}] output must be a JSON dict, got {type(result).__name__}"
            )
