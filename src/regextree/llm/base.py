from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Protocol for LLM backends that generate regex candidates."""

    def generate(self, prompt: str) -> str:
        """Send a prompt and return the raw text response."""
        ...
