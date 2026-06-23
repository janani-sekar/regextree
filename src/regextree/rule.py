from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rule:
    id: str
    regex: str
    target_class: int
    weight: float
    description: str = ""
    step: int = 0
    _compiled: re.Pattern = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_compiled", re.compile(self.regex))

    def matches(self, text: str) -> bool:
        return self._compiled.search(text) is not None

    def match_array(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([self.matches(t) for t in texts], dtype=bool)


def match_matrix(rules: list[Rule], texts: Sequence[str]) -> pd.DataFrame:
    """Build a (n_samples, n_rules) boolean DataFrame of pattern matches."""
    data = {rule.id: rule.match_array(texts) for rule in rules}
    return pd.DataFrame(data, index=range(len(texts)))


def compute_logits(rules: list[Rule], texts: Sequence[str], n_classes: int) -> np.ndarray:
    """Compute class logits for each sample by summing rule weights.

    Returns array of shape (n_samples, n_classes).
    """
    n = len(texts)
    logits = np.zeros((n, n_classes), dtype=np.float64)
    for rule in rules:
        mask = rule.match_array(texts)
        logits[mask, rule.target_class] += rule.weight
    return logits
