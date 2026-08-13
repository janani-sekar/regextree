"""Tests for the manipulation bank and full-dataset labeling."""

import textwrap
from pathlib import Path

import pandas as pd
import pytest
import sympy

from regextree.manipulation import (
    ManipulationBank,
    ManipulationRule,
    label_transitions,
    summarize_labels,
)

SIGN_EXPORT = textwrap.dedent(
    """
    import sympy

    def apply_fn(prev_expr):
        if not isinstance(prev_expr, sympy.Equality):
            return []
        lhs, rhs = prev_expr.args
        if not lhs.is_Add:
            return []
        outputs = []
        for arg in lhs.args:
            if arg.is_Symbol:
                continue
            remaining = [a for a in lhs.args if a != arg]
            new_lhs = sympy.Add(*remaining, evaluate=False) if len(remaining) > 1 else remaining[0]
            outputs.append(sympy.Eq(new_lhs, sympy.Add(arg, rhs, evaluate=False), evaluate=False))
        return outputs

    NAME = "sign_error"
    DESCRIPTION = "Term moved across '=' without a sign change."
    """
).strip() + "\n"

NEVER_EXPORT = textwrap.dedent(
    """
    import sympy

    def apply_fn(prev_expr):
        return []

    NAME = "never_applies"
    DESCRIPTION = "Never a candidate."
    """
).strip() + "\n"


def apply_sign(prev_expr: sympy.Basic) -> list[sympy.Basic]:
    if not isinstance(prev_expr, sympy.Equality):
        return []
    lhs, rhs = prev_expr.args
    if not lhs.is_Add:
        return []
    outputs = []
    for arg in lhs.args:
        if arg.is_Symbol:
            continue
        remaining = [a for a in lhs.args if a != arg]
        new_lhs = sympy.Add(*remaining, evaluate=False) if len(remaining) > 1 else remaining[0]
        outputs.append(sympy.Eq(new_lhs, sympy.Add(arg, rhs, evaluate=False), evaluate=False))
    return outputs


@pytest.fixture
def transitions() -> pd.DataFrame:
    prev_expr = sympy.Eq(sympy.Symbol("x") + 3, 12, evaluate=False)
    wrong_curr = sympy.Eq(sympy.Symbol("x"), 15, evaluate=False)
    right_curr = sympy.Eq(sympy.Symbol("x"), 9, evaluate=False)
    not_candidate = sympy.Symbol("x") * 2
    return pd.DataFrame(
        {
            "transition_id": ["t1", "t2", "t3", "t4"],
            "prev_expr": [prev_expr, prev_expr, not_candidate, prev_expr],
            "curr_expr": [wrong_curr, right_curr, right_curr, wrong_curr],
            "step_correct": [False, True, False, True],
        }
    )


def test_bank_round_trip(tmp_path: Path):
    export = tmp_path / "results.py"
    export.write_text(SIGN_EXPORT)

    bank = ManipulationBank(tmp_path / "bank")
    entry = bank.add(
        export,
        provenance={"step_group": "Q7", "response_group": "R2"},
        stats={"val_precision": 0.26},
    )

    assert entry.manipulation_id.startswith("sign_error_")
    assert entry.manipulation_id in bank
    assert len(bank) == 1

    reloaded = bank.load(entry.manipulation_id)
    assert reloaded.name == "sign_error"
    prev_expr = sympy.Eq(sympy.Symbol("x") + 3, 12, evaluate=False)
    assert reloaded.candidate(prev_expr)

    manifest = bank.manifest()
    assert manifest.loc[0, "provenance_step_group"] == "Q7"
    assert manifest.loc[0, "stats_val_precision"] == 0.26


def test_bank_survives_reopen_and_dedupes(tmp_path: Path):
    export = tmp_path / "results.py"
    export.write_text(SIGN_EXPORT)

    first = ManipulationBank(tmp_path / "bank")
    entry = first.add(export)
    assert first.add(export).manipulation_id == entry.manipulation_id
    assert len(first) == 1

    second = ManipulationBank(tmp_path / "bank")
    assert len(second) == 1
    assert set(second.rules()) == {entry.manipulation_id}


def test_bank_missing_manipulation(tmp_path: Path):
    bank = ManipulationBank(tmp_path / "bank")
    assert bank.manifest().empty
    with pytest.raises(KeyError):
        bank.load("nope")


def test_label_transitions_long_table(transitions: pd.DataFrame):
    rules = {
        "m_sign": ManipulationRule("sign_error", "...", apply_sign),
        "m_never": ManipulationRule("never_applies", "...", lambda prev_expr: []),
    }
    labels = label_transitions(transitions, rules)

    assert len(labels) == len(transitions) * len(rules)
    assert list(labels.columns) == [
        "transition_id",
        "manipulation_id",
        "manipulation_name",
        "manipulation_candidate",
        "manipulation_realized",
        "manipulation_false_positive",
    ]

    sign = labels[labels["manipulation_id"] == "m_sign"].set_index("transition_id")
    # t3's previous expression is not a candidate at all.
    assert not sign.loc["t3", "manipulation_candidate"]
    assert sign.loc["t1", "manipulation_realized"]
    # t2 is a candidate but the observed step does not match an output.
    assert not sign.loc["t2", "manipulation_realized"]
    # t4 matches an output on a correct step, so it is a false positive.
    assert not sign.loc["t4", "manipulation_realized"]
    assert sign.loc["t4", "manipulation_false_positive"]

    never = labels[labels["manipulation_id"] == "m_never"]
    assert not never["manipulation_candidate"].any()


def test_realized_requires_incorrect_toggle(transitions: pd.DataFrame):
    rules = {"m_sign": ManipulationRule("sign_error", "...", apply_sign)}

    structural = label_transitions(transitions, rules, realized_requires_incorrect=False)
    structural = structural.set_index("transition_id")
    assert structural.loc["t4", "manipulation_realized"]
    assert not structural.loc["t4", "manipulation_false_positive"]

    s32_default = label_transitions(transitions, rules).set_index("transition_id")
    assert not s32_default.loc["t4", "manipulation_realized"]


def test_label_transitions_accepts_bank(tmp_path: Path, transitions: pd.DataFrame):
    bank = ManipulationBank(tmp_path / "bank")
    for name, source in (("sign.py", SIGN_EXPORT), ("never.py", NEVER_EXPORT)):
        export = tmp_path / name
        export.write_text(source)
        bank.add(export)

    labels = label_transitions(transitions, bank.rules())
    assert labels["manipulation_id"].nunique() == 2

    summary = summarize_labels(labels).set_index("manipulation_name")
    assert summary.loc["sign_error", "n_candidate"] == 3
    assert summary.loc["sign_error", "n_realized"] == 1
    assert summary.loc["never_applies", "n_candidate"] == 0
    assert summary.loc["never_applies", "realized_rate"] == 0.0


def test_label_transitions_rejects_duplicate_ids(transitions: pd.DataFrame):
    rule = ManipulationRule("sign_error", "...", apply_sign)
    with pytest.raises(ValueError, match="Duplicate manipulation ids"):
        label_transitions(transitions, [rule, rule])
