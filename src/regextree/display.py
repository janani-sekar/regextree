from __future__ import annotations

from typing import Sequence

import pandas as pd

from .llm import LLMClient, get_client
from .prompt import generate_description
from .rule import Rule, match_matrix


class DisplayMixin:
    """Mixin providing interpretability methods for fitted models."""

    _rules: list[Rule]
    _fitted: bool
    _llm_spec: str | LLMClient
    _llm_kwargs: dict
    _descriptions: dict[str, str]

    def rules(self) -> list[Rule]:
        """Return the list of fitted rules."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")
        return list(self._rules)

    def patterns(self) -> dict[str, str]:
        """Return {rule_id: regex_pattern} for all fitted rules."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")
        return {rule.id: rule.regex for rule in self._rules}

    def description(self, texts: Sequence[str] | None = None) -> dict[str, str]:
        """Return {rule_id: human_description} for all fitted rules.

        If descriptions haven't been generated yet and texts are provided,
        generates them via the LLM.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")

        if not hasattr(self, "_descriptions"):
            self._descriptions = {}

        # Return cached if available
        if self._descriptions:
            return dict(self._descriptions)

        # Generate descriptions if texts provided
        if texts is not None:
            client = get_client(self._llm_spec, **self._llm_kwargs)
            for rule in self._rules:
                desc = generate_description(
                    client, rule.regex, rule.target_class, texts
                )
                self._descriptions[rule.id] = desc
            return dict(self._descriptions)

        # Fallback: return regex as description
        return {rule.id: f"matches {rule.regex!r} → class {rule.target_class}" for rule in self._rules}

    def matches(self, x: Sequence[str], voids: bool = False) -> pd.DataFrame:
        """Return a DataFrame of boolean pattern matches (samples x rules).

        If voids=True, includes rules that matched no samples.
        If voids=False, drops columns where no sample matched.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")

        df = match_matrix(self._rules, x)

        if not voids:
            df = df.loc[:, df.any(axis=0)]

        return df
