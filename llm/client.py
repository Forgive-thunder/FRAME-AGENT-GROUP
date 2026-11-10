"""OpenAI-compatible LLM client (real calls, no dummies).

Reads ``API_KEY`` and ``BASE_URL`` (and optional ``MODEL``) from the
environment and issues chat-completion calls via the ``openai`` SDK.  JSON
mode is enforced so agents can rely on structured output.

Usage::

    export API_KEY="sk-..."
    export BASE_URL="https://api.example.com/v1"   # optional; omit for OpenAI
    export MODEL="gpt-4o-mini"                     # optional

Install the SDK if needed: ``pip install openai``.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional


class LLMClient:
    """Thin wrapper around the OpenAI chat-completions API."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model or os.getenv("MODEL", "gpt-4o-mini")
        self.api_key = api_key if api_key is not None else os.getenv("API_KEY")
        self.base_url = base_url if base_url is not None else os.getenv("BASE_URL")
        self.temperature = temperature
        self._client = None

    # -- internals --------------------------------------------------------
    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError(
                "LLM API key not set. Export the environment variable API_KEY "
                "before running (and BASE_URL if using a non-OpenAI endpoint)."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The 'openai' package is not installed. Run: pip install openai"
            ) from exc
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    # -- public API -------------------------------------------------------
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        json_mode: bool = True,
    ) -> str:
        """Return the assistant's reply text for a single-turn completion."""
        client = self._get_client()
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature if temperature is None else temperature,
        }
        if json_mode:
            # Force the model to emit valid JSON.  If a provider rejects this
            # argument, remove it and rely on the prompt's format instruction.
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        return (response.choices[0].message.content or "").strip()

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Call :meth:`chat` in JSON mode and parse the result into a dict."""
        content = self.chat(system_prompt, user_prompt, temperature, json_mode=True)
        return _parse_json(content)


def _parse_json(content: str) -> Dict[str, Any]:
    """Best-effort JSON parsing of an LLM reply (handles markdown fences)."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                raise ValueError(f"LLM returned unparseable JSON: {content!r}")
        else:
            raise ValueError(f"LLM returned unparseable JSON: {content!r}")

    if not isinstance(data, dict):
        raise ValueError(f"LLM JSON must be an object, got: {content!r}")
    return data
