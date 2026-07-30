"""Tests for the full RegexBoostClassifier using a mock LLM."""

import numpy as np

from regextree import BoostRound, RegexBoostClassifier, SelectedCandidate
from regextree.scoring import negative_match_rate, passes_negative_match_gate


class MockLLM:
    """Deterministic mock LLM that returns regex patterns based on examples."""

    def __init__(self):
        self._call_count = 0
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


def _fit_model(**kwargs):
    defaults = dict(pool_size=3, step_count=3, eta=0.3, llm=MockLLM(), verbose=False)
    defaults.update(kwargs)
    model = RegexBoostClassifier(**defaults)
    model.fit(TEXTS, LABELS)
    return model


def test_fit_predict_basic():
    model = _fit_model()
    preds = model.predict(TEXTS)
    assert len(preds) == len(TEXTS)
    assert all(p in [0, 1, 2] for p in preds)


def test_patterns_returns_dict():
    model = _fit_model(step_count=2)
    patterns = model.patterns()
    assert isinstance(patterns, dict)
    for k, v in patterns.items():
        assert isinstance(k, str)
        assert isinstance(v, str)


def test_matches_returns_dataframe():
    model = _fit_model(step_count=2)
    df = model.matches(TEXTS)
    assert df.shape[0] == len(TEXTS)
    assert df.dtypes.unique().tolist() == [np.dtype("bool")]


def test_rules_returns_list():
    model = _fit_model(step_count=2)
    rules = model.rules()
    assert isinstance(rules, list)
    assert len(rules) > 0
    for rule in rules:
        assert hasattr(rule, "regex")
        assert hasattr(rule, "target_class")
        assert hasattr(rule, "weight")


def test_predict_proba_sums_to_one():
    model = _fit_model()
    proba = model.predict_proba(TEXTS)
    assert proba.shape == (len(TEXTS), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-7)


def test_generate_select_split():
    model = RegexBoostClassifier(pool_size=3, step_count=1, eta=0.3, llm=MockLLM(), verbose=False)
    labels = np.array(LABELS)
    logits = np.zeros((len(TEXTS), 3))
    round_data = model.generate_round(TEXTS, labels, logits, step=0)
    assert isinstance(round_data, BoostRound)
    assert round_data.candidates
    selected = model.select_candidate(round_data, TEXTS, labels, logits)
    assert isinstance(selected, SelectedCandidate)
    assert selected.regex in round_data.candidates


def test_negative_match_gate_rejects_broad_pattern():
    labels = np.array([0, 0, 0, 1, 1])
    matches = np.array([True, True, True, True, False])
    assert negative_match_rate(matches, labels, target_class=0) == 0.5
    assert passes_negative_match_gate(matches, labels, 0, max_negative_match_rate=0.25) is False
    assert passes_negative_match_gate(matches, labels, 0, max_negative_match_rate=0.5) is True


def test_precision_objective_prefers_tight_pattern():
    texts = ["x=1/2", "x=2/3", "x=3/4", "3x+1=0", "2x+1=0"]
    labels = np.array([1, 1, 1, 0, 0])
    logits = np.zeros((len(texts), 2))

    model = RegexBoostClassifier(
        pool_size=2,
        step_count=1,
        eta=0.5,
        llm=MockLLM(),
        verbose=False,
        objective="precision",
        max_negative_match_rate=0.5,
    )
    round_data = BoostRound(
        step=0,
        target_class=1,
        positives=["x=1/2", "x=2/3"],
        negatives=["3x+1=0", "2x+1=0"],
        candidates=[r"x=\d+/\d+", r".*x.*"],
    )
    selected = model.select_candidate(round_data, texts, labels, logits)
    assert selected is not None
    assert selected.regex == r"x=\d+/\d+"


def test_lasso_refit_sparsifies_rules():
    model = _fit_model(step_count=4, weight_method="lasso", lasso_C=0.05)
    assert len(model.rules()) <= 4


def test_subgroup_labels_sampling():
    subgroup = [0, 0, 1, 2, 1, 1, 2, 1, 1, 0]
    model = RegexBoostClassifier(pool_size=3, step_count=1, eta=0.3, llm=MockLLM(), verbose=False)
    labels = np.array(LABELS)
    logits = np.zeros((len(TEXTS), 3))
    model._subgroup_labels = np.array(subgroup)
    positives, negatives = model._sample_examples(
        TEXTS, labels, np.zeros_like(logits), target_class=1, subgroup_labels=np.array(subgroup)
    )
    assert positives
    assert negatives
