"""Tests for LASSO weight refit."""

import numpy as np

from regextree.lasso import refit_lasso_weights
from regextree.rule import Rule


def test_refit_lasso_drops_zero_weight_rules():
    texts = ["x=1/2", "x=2/3", "3x+1=0", "2x+1=0", "x=9/10"]
    labels = np.array([1, 1, 0, 0, 1])
    rules = [
        Rule(id="r000", regex=r"x=\d+/\d+", target_class=1, weight=0.5, step=0),
        Rule(id="r001", regex=r".*", target_class=1, weight=0.5, step=1),
    ]

    refitted = refit_lasso_weights(rules, texts, labels, C=10.0, drop_zero=True)
    assert len(refitted) >= 1
    assert all(rule.weight != 0 for rule in refitted)
    assert refitted[0].regex == r"x=\d+/\d+"
