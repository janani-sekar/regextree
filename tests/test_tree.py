"""Tests for RegexTreeClassifier with a mock LLM."""

import numpy as np

from regextree import RegexTreeClassifier


class MockTreeLLM:
    """Mock LLM that returns regex patterns suitable for tree splitting."""

    def __init__(self):
        self._call_count = 0
        self._patterns = [
            [r"x=\d+/\d+", r"x=", r"\d+/\d+"],
            [r"^\d+.*x", r"\d+x[+-]", r"=.*x"],
            [r"\.\d+", r"\d+\.\d+", r"0\.\d+"],
        ]

    def generate(self, prompt: str) -> str:
        idx = min(self._call_count, len(self._patterns) - 1)
        patterns = self._patterns[idx]
        self._call_count += 1
        return "\n".join(patterns)


TEXTS = [
    "3.4x+0.4=-1x-6",
    "-28x+10=7x-17",
    "x=32/20",
    "x=25/0.8=17x",
    "x=-21/5",
    "x=1/1.5",
    "x=11.2/6.7",
    "x=42/2",
    "x=22/7",
    "-85x-60=19x-10",
]
LABELS = [0, 0, 1, 2, 1, 1, 2, 1, 1, 0]


def test_tree_fit_predict():
    model = RegexTreeClassifier(
        pool_size=3,
        max_depth=3,
        min_samples_leaf=1,
        llm=MockTreeLLM(),
        verbose=False,
    )
    model.fit(TEXTS, LABELS)
    preds = model.predict(TEXTS)
    assert len(preds) == len(TEXTS)
    assert all(p in [0, 1, 2] for p in preds)


def test_tree_predict_proba():
    model = RegexTreeClassifier(
        pool_size=3,
        max_depth=3,
        min_samples_leaf=1,
        llm=MockTreeLLM(),
        verbose=False,
    )
    model.fit(TEXTS, LABELS)
    proba = model.predict_proba(TEXTS)
    assert proba.shape == (len(TEXTS), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-7)


def test_tree_print():
    model = RegexTreeClassifier(
        pool_size=3,
        max_depth=2,
        min_samples_leaf=1,
        llm=MockTreeLLM(),
        verbose=False,
    )
    model.fit(TEXTS, LABELS)
    tree_str = model.print_tree()
    assert "IF matches" in tree_str or "LEAF" in tree_str


def test_tree_patterns():
    model = RegexTreeClassifier(
        pool_size=3,
        max_depth=2,
        min_samples_leaf=1,
        llm=MockTreeLLM(),
        verbose=False,
    )
    model.fit(TEXTS, LABELS)
    patterns = model.patterns()
    assert isinstance(patterns, dict)
    assert len(patterns) > 0
