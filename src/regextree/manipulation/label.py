from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Sequence

import pandas as pd
import sympy

from .match import canonical_str
from .rule import ManipulationRule


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _as_named_rules(
    rules: Mapping[str, ManipulationRule] | Iterable[ManipulationRule],
) -> list[tuple[str, ManipulationRule]]:
    if isinstance(rules, Mapping):
        return list(rules.items())
    return [(rule.name, rule) for rule in rules]


def label_transitions(
    df: pd.DataFrame,
    rules: Mapping[str, ManipulationRule] | Iterable[ManipulationRule],
    *,
    transition_id_col: str = "transition_id",
    prev_expr_col: str = "prev_expr",
    curr_expr_col: str = "curr_expr",
    step_correct_col: str | None = "step_correct",
    realized_requires_incorrect: bool = True,
) -> pd.DataFrame:
    """Label every transition against every manipulation in a bank.

    Returns one row per (transition, manipulation) with candidate and realized
    flags. `manipulation_candidate` is needed downstream because the psychometric
    regressions run on the subpopulation where the previous expression is a
    candidate.

    When `realized_requires_incorrect` is True, a correct step is never realized
    (the s32 convention); such matches surface as `manipulation_false_positive`
    instead. Set it False to score realization purely structurally.
    """
    _require_columns(df, [transition_id_col, prev_expr_col, curr_expr_col])
    named_rules = _as_named_rules(rules)
    if not named_rules:
        raise ValueError("No manipulations supplied")

    duplicates = [name for name, count in Counter(name for name, _ in named_rules).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate manipulation ids: {sorted(duplicates)}")

    step_correct = (
        df[step_correct_col] if step_correct_col and step_correct_col in df.columns else None
    )

    # Expressions repeat heavily across the observed table, so apply each rule
    # once per distinct previous expression and canonicalize each step once.
    prev_keys = [None if expr is None else sympy.srepr(expr) for expr in df[prev_expr_col]]
    curr_canonical = {}
    for expr in df[curr_expr_col]:
        if expr is None:
            continue
        key = sympy.srepr(expr)
        if key not in curr_canonical:
            curr_canonical[key] = canonical_str(expr)
    curr_keys = [None if expr is None else sympy.srepr(expr) for expr in df[curr_expr_col]]

    prev_by_key: dict[str, sympy.Basic] = {}
    for key, expr in zip(prev_keys, df[prev_expr_col]):
        if key is not None and key not in prev_by_key:
            prev_by_key[key] = expr

    frames = []
    for manipulation_id, rule in named_rules:
        outputs_by_key = {key: set(rule.outputs(expr)) for key, expr in prev_by_key.items()}

        candidate_flags = []
        realized_flags = []
        false_positive_flags = []
        for row_index, (prev_key, curr_key) in enumerate(zip(prev_keys, curr_keys)):
            outputs = outputs_by_key.get(prev_key) if prev_key is not None else None
            if not outputs or curr_key is None:
                candidate_flags.append(bool(outputs))
                realized_flags.append(False)
                false_positive_flags.append(False)
                continue

            matched = curr_canonical[curr_key] in outputs
            is_correct = bool(step_correct.iloc[row_index]) if step_correct is not None else None

            candidate_flags.append(True)
            if realized_requires_incorrect and is_correct:
                realized_flags.append(False)
                false_positive_flags.append(matched)
            else:
                realized_flags.append(matched)
                false_positive_flags.append(False)

        frames.append(
            pd.DataFrame(
                {
                    transition_id_col: df[transition_id_col].to_numpy(),
                    "manipulation_id": manipulation_id,
                    "manipulation_name": rule.name,
                    "manipulation_candidate": candidate_flags,
                    "manipulation_realized": realized_flags,
                    "manipulation_false_positive": false_positive_flags,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def summarize_labels(
    labels: pd.DataFrame,
    *,
    transition_id_col: str = "transition_id",
) -> pd.DataFrame:
    """Per-manipulation coverage and realization rates from a label table."""
    grouped = labels.groupby(["manipulation_id", "manipulation_name"], as_index=False).agg(
        n_transitions=(transition_id_col, "count"),
        n_candidate=("manipulation_candidate", "sum"),
        n_realized=("manipulation_realized", "sum"),
        n_false_positive=("manipulation_false_positive", "sum"),
    )
    grouped["candidate_coverage"] = grouped["n_candidate"] / grouped["n_transitions"]
    grouped["realized_rate"] = (grouped["n_realized"] / grouped["n_candidate"]).where(
        grouped["n_candidate"] > 0, 0.0
    )
    return grouped
