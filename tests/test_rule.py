"""Tests for regextree.rule module."""

import numpy as np
import pandas as pd

from regextree.rule import Rule, compute_logits, match_matrix


def test_rule_matches():
    rule = Rule(id="r0", regex=r"x=\d+", target_class=0, weight=1.0)
    assert rule.matches("x=42")
    assert not rule.matches("y=42")


def test_rule_match_array():
    rule = Rule(id="r0", regex=r"\d+/\d+", target_class=1, weight=0.5)
    texts = ["x=3/4", "y=10", "a=1/2"]
    result = rule.match_array(texts)
    np.testing.assert_array_equal(result, [True, False, True])


def test_match_matrix():
    rules = [
        Rule(id="r0", regex=r"x=", target_class=0, weight=1.0),
        Rule(id="r1", regex=r"/\d+", target_class=1, weight=0.5),
    ]
    texts = ["x=3/4", "y=10", "x=5"]
    df = match_matrix(rules, texts)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["r0", "r1"]
    assert df["r0"].tolist() == [True, False, True]
    assert df["r1"].tolist() == [True, False, False]


def test_compute_logits():
    rules = [
        Rule(id="r0", regex=r"x=", target_class=0, weight=1.0),
        Rule(id="r1", regex=r"/\d+", target_class=1, weight=0.5),
    ]
    texts = ["x=3/4", "y=10", "x=5"]
    logits = compute_logits(rules, texts, n_classes=2)
    assert logits.shape == (3, 2)
    # "x=3/4" matches both rules
    assert logits[0, 0] == 1.0
    assert logits[0, 1] == 0.5
    # "y=10" matches neither
    assert logits[1, 0] == 0.0
    assert logits[1, 1] == 0.0
    # "x=5" matches only r0
    assert logits[2, 0] == 1.0
    assert logits[2, 1] == 0.0
