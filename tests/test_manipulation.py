"""Tests for regextree.manipulation."""

import textwrap
from pathlib import Path

import pandas as pd
import sympy

from regextree.manipulation import (
    ManipulationRule,
    evaluate_rule_on_transitions,
    is_realized,
    load_rule_from_module,
    manipulation_outputs,
)


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
        new_rhs = sympy.Add(arg, rhs, evaluate=False)
        outputs.append(sympy.Eq(new_lhs, new_rhs, evaluate=False))
    return outputs


def test_manipulation_outputs_sign_error():
    prev_expr = sympy.Eq(sympy.Symbol("x") + 3, 12, evaluate=False)
    outputs = manipulation_outputs(prev_expr, apply_sign)
    assert len(outputs) == 1
    assert outputs[0] == str(sympy.expand(sympy.Eq(sympy.Symbol("x"), 15, evaluate=False)))


def test_realized_on_incorrect_step():
    prev_expr = sympy.Eq(sympy.Symbol("x") + 3, 12, evaluate=False)
    curr_expr = sympy.Eq(sympy.Symbol("x"), 15, evaluate=False)
    assert is_realized(prev_expr, curr_expr, apply_sign, step_correct=False)


def test_false_positive_on_correct_step():
    prev_expr = sympy.Eq(sympy.Symbol("x") + 3, 12, evaluate=False)
    curr_expr = sympy.Eq(sympy.Symbol("x"), 15, evaluate=False)
    rule = ManipulationRule(name="sign_error", description="...", apply_fn=apply_sign)
    assert rule.false_positive(prev_expr, curr_expr, step_correct=True)
    assert not rule.realized(prev_expr, curr_expr, step_correct=True)


def test_evaluate_rule_on_transitions():
    prev_expr = sympy.Eq(sympy.Symbol("x") + 3, 12, evaluate=False)
    wrong_curr = sympy.Eq(sympy.Symbol("x"), 15, evaluate=False)
    right_curr = sympy.Eq(sympy.Symbol("x"), 9, evaluate=False)

    df = pd.DataFrame(
        {
            "prev_expr": [prev_expr, prev_expr, prev_expr],
            "curr_expr": [wrong_curr, right_curr, wrong_curr],
            "label": [1, 0, 1],
            "step_correct": [False, True, False],
        }
    )
    rule = ManipulationRule(name="sign_error", description="...", apply_fn=apply_sign)
    metrics = evaluate_rule_on_transitions(df, rule)

    assert metrics["n_candidate"] == 3
    assert metrics["n_realized"] == 2
    assert metrics["n_false_positive"] == 0
    assert metrics["n_predicted_positive"] == 2
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_load_rule_from_module(tmp_path: Path):
    module_path = tmp_path / "results.py"
    module_path.write_text(
        textwrap.dedent(
            """
            import sympy
            from regextree.manipulation import ManipulationRule

            def apply_fn(prev_expr):
                return []

            rule = ManipulationRule(
                name="empty",
                description="test rule",
                apply_fn=apply_fn,
            )
            """
        ).strip()
        + "\n"
    )

    loaded = load_rule_from_module(module_path)
    assert loaded.name == "empty"
    assert loaded.description == "test rule"
