"""Tests for the full RegexBoostClassifier using a mock LLM."""

import numpy as np

from regextree import RegexBoostClassifier


class MockLLM:
    """Deterministic mock LLM that returns regex patterns based on examples."""

    def __init__(self):
        self._call_count = 0
        # Pre-defined patterns that should work for the synthetic data
        self._patterns = [
            [r"x=\d+/\d+", r"x=\d+\.\d+", r"\d+/\d+$"],
            [r"^\d+\.\d+x", r"x[+-]", r"\d+x[+-]\d+"],
            [r"^-?\d+x[+-]", r"=-?\d+x", r"\d+x.*=.*\d+x"],
        ]

    def generate(self, prompt: str) -> str:
        idx = min(self._call_count, len(self._patterns) - 1)
        patterns = self._patterns[idx]
        self._call_count += 1
        return "\n".join(patterns)


# Synthetic dataset mimicking the user's math step data
TEXTS = [
    "3.4x+0.4=-1x-6",       # class 0: two-sided equation
    "-28x+10=7x-17",         # class 0
    "x=32/20",               # class 1: x = fraction
    "x=25/0.8=17x",          # class 2: chained equals
    "x=-21/5",               # class 1
    "x=1/1.5",               # class 1
    "x=11.2/6.7",            # class 2 (decimal division)
    "x=42/2",                # class 1
    "x=22/7",                # class 1
    "-85x-60=19x-10",        # class 0
]
LABELS = [0, 0, 1, 2, 1, 1, 2, 1, 1, 0]


def test_fit_predict_basic():
    model = RegexBoostClassifier(
        pool_size=3,
        step_count=3,
        eta=0.3,
        llm=MockLLM(),
        verbose=False,
    )
    model.fit(TEXTS, LABELS)
    preds = model.predict(TEXTS)
    assert len(preds) == len(TEXTS)
    assert all(p in [0, 1, 2] for p in preds)


def test_patterns_returns_dict():
    model = RegexBoostClassifier(pool_size=3, step_count=2, eta=0.3, llm=MockLLM(), verbose=False)
    model.fit(TEXTS, LABELS)
    patterns = model.patterns()
    assert isinstance(patterns, dict)
    for k, v in patterns.items():
        assert isinstance(k, str)
        assert isinstance(v, str)


def test_matches_returns_dataframe():
    model = RegexBoostClassifier(pool_size=3, step_count=2, eta=0.3, llm=MockLLM(), verbose=False)
    model.fit(TEXTS, LABELS)
    df = model.matches(TEXTS)
    assert df.shape[0] == len(TEXTS)
    assert df.dtypes.unique().tolist() == [np.dtype("bool")]


def test_rules_returns_list():
    model = RegexBoostClassifier(pool_size=3, step_count=2, eta=0.3, llm=MockLLM(), verbose=False)
    model.fit(TEXTS, LABELS)
    rules = model.rules()
    assert isinstance(rules, list)
    assert len(rules) > 0
    for rule in rules:
        assert hasattr(rule, "regex")
        assert hasattr(rule, "target_class")
        assert hasattr(rule, "weight")


def test_predict_proba_sums_to_one():
    model = RegexBoostClassifier(pool_size=3, step_count=3, eta=0.3, llm=MockLLM(), verbose=False)
    model.fit(TEXTS, LABELS)
    proba = model.predict_proba(TEXTS)
    assert proba.shape == (len(TEXTS), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-7)
