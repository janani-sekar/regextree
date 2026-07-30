from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from .rule import Rule, match_matrix


def refit_lasso_weights(
    rules: list[Rule],
    texts: Sequence[str],
    labels: np.ndarray,
    *,
    C: float = 1.0,
    max_rules: int | None = None,
    drop_zero: bool = True,
) -> list[Rule]:
    """Refit rule weights with L1-regularized logistic regression on match features.

    Rules are grouped by target_class and refit independently. Zero-weight rules
    are dropped when drop_zero=True.
    """
    if not rules:
        return []

    match_df = match_matrix(rules, texts).astype(float)
    updated: list[Rule] = []

    for target_class in sorted({rule.target_class for rule in rules}):
        class_rules = [rule for rule in rules if rule.target_class == target_class]
        columns = [rule.id for rule in class_rules]
        x = match_df[columns].to_numpy()
        y = (labels == target_class).astype(int)

        if y.sum() == 0 or y.sum() == len(y):
            updated.extend(class_rules)
            continue

        clf = LogisticRegression(
            l1_ratio=1.0,
            solver="saga",
            C=C,
            max_iter=1000,
            fit_intercept=True,
        )
        clf.fit(x, y)

        coefs = clf.coef_[0]
        for rule, coef in zip(class_rules, coefs):
            if drop_zero and abs(coef) < 1e-6:
                continue
            updated.append(
                Rule(
                    id=rule.id,
                    regex=rule.regex,
                    target_class=rule.target_class,
                    weight=float(coef),
                    description=rule.description,
                    step=rule.step,
                )
            )

    if max_rules is not None and len(updated) > max_rules:
        updated.sort(key=lambda rule: abs(rule.weight), reverse=True)
        updated = updated[:max_rules]

    return updated
