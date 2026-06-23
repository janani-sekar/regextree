from __future__ import annotations

import os

from google import genai
from google.genai import types

from .base import LLMClient


class GeminiClient:
    """LLM client using Google's Gemini API (google-genai SDK)."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        thinking: bool = False,
    ):
        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError(
                "No API key provided. Set GOOGLE_API_KEY / GEMINI_API_KEY or pass api_key= to GeminiClient."
            )
        self._client = genai.Client(api_key=key)
        self._model = model
        self._thinking = thinking

    def generate(self, prompt: str) -> str:
        config = None
        if not self._thinking:
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        response = self._client.models.generate_content(
            model=self._model, contents=prompt, config=config
        )
        return response.text


def get_client(llm: str | LLMClient = "gemini", **kwargs) -> LLMClient:
    """Factory to resolve an LLM client from a string name or pass-through instance."""
    if isinstance(llm, str):
        if llm == "gemini":
            return GeminiClient(**kwargs)
        raise ValueError(f"Unknown LLM backend: {llm!r}. Use 'gemini' or pass an LLMClient instance.")
    return llm
