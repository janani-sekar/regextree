from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .rule import ManipulationRule


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def evaluate_rule_on_transitions(
    df: pd.DataFrame,
    rule: ManipulationRule,
    *,
    prev_expr_col: str = "prev_expr",
    curr_expr_col: str = "curr_expr",
    label_col: str = "label",
    step_correct_col: str | None = "step_correct",
) -> dict[str, float | int]:
    """Score a manipulation rule against binary cluster labels on transitions."""
    _require_columns(df, [prev_expr_col, curr_expr_col, label_col])

    labels = df[label_col].astype(int).to_numpy()
    predictions = []
    candidate_flags = []
    realized_flags = []
    false_positive_flags = []

    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        prev_expr = row_dict[prev_expr_col]
        curr_expr = row_dict[curr_expr_col]
        step_correct = row_dict.get(step_correct_col) if step_correct_col else None

        if prev_expr is None or curr_expr is None:
            candidate_flags.append(False)
            realized_flags.append(False)
            false_positive_flags.append(False)
            predictions.append(False)
            continue

        is_cand = rule.candidate(prev_expr)
        is_real = rule.realized(prev_expr, curr_expr, step_correct=step_correct)
        is_fp = (
            rule.false_positive(prev_expr, curr_expr, step_correct=bool(step_correct))
            if step_correct_col and step_correct is not None
            else False
        )

        candidate_flags.append(is_cand)
        realized_flags.append(is_real)
        false_positive_flags.append(is_fp)
        predictions.append(rule.predict(prev_expr, curr_expr, step_correct=step_correct))

    y_pred = np.array(predictions, dtype=int)
    y_true = labels.astype(int)
    candidate_mask = np.array(candidate_flags, dtype=bool)

    n_total = len(df)
    n_candidate = int(candidate_mask.sum())
    n_predicted_positive = int(y_pred.sum())
    n_realized = int(np.array(realized_flags, dtype=bool).sum())
    n_false_positive = int(np.array(false_positive_flags, dtype=bool).sum())

    if step_correct_col and step_correct_col in df.columns:
        correct_candidates = df.loc[candidate_mask, step_correct_col].astype(bool)
        n_correct_candidates = int(correct_candidates.sum()) if n_candidate else 0
        fp_rate = n_false_positive / n_correct_candidates if n_correct_candidates else float("nan")
    else:
        n_correct_candidates = 0
        fp_rate = float("nan")

    return {
        "n_total": n_total,
        "n_candidate": n_candidate,
        "n_realized": n_realized,
        "n_predicted_positive": n_predicted_positive,
        "n_false_positive": n_false_positive,
        "candidate_coverage": n_candidate / n_total if n_total else 0.0,
        "realized_rate": n_realized / n_candidate if n_candidate else 0.0,
        "false_positive_rate": fp_rate,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
