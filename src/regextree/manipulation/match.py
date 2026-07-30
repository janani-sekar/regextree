from __future__ import annotations

from collections.abc import Callable

import sympy

ApplyFn = Callable[[sympy.Basic], list[sympy.Basic]]


def force_evaluate(expr: sympy.Basic) -> sympy.Basic:
    """Recursively re-evaluate a sympy expression (strips evaluate=False effects).

    Matches s32: on malformed student trees, return the original expression.
    """
    try:
        if not expr.args:
            return expr
        return expr.func(*[force_evaluate(arg) for arg in expr.args])
    except Exception:
        return expr


def canonical_str(expr: sympy.Basic) -> str:
    """Canonical string after forcing evaluation and expansion."""
    try:
        return str(sympy.expand(force_evaluate(expr)))
    except Exception:
        pass
    try:
        return str(expr)
    except Exception:
        return f"<unprintable:{type(expr).__name__}:{id(expr)}>"


def manipulation_outputs(prev_expr: sympy.Basic, apply_fn: ApplyFn) -> list[str]:
    """Apply a manipulation and return canonical output strings.

    Matches s32: empty list on apply failure or inapplicable input.
    """
    try:
        outputs = apply_fn(prev_expr)
        return [canonical_str(output) for output in outputs]
    except Exception:
        return []


def is_candidate(prev_expr: sympy.Basic, apply_fn: ApplyFn) -> bool:
    """True when the manipulation applies to the previous expression."""
    return len(manipulation_outputs(prev_expr, apply_fn)) > 0


def is_realized(
    prev_expr: sympy.Basic,
    curr_expr: sympy.Basic,
    apply_fn: ApplyFn,
    *,
    step_correct: bool | None = None,
) -> bool:
    """True when the observed step matches a manipulation output.

    When step_correct is provided, correct steps never count as realized
    (they may count as false positives via is_false_positive instead).
    """
    if step_correct is True:
        return False
    output_strs = manipulation_outputs(prev_expr, apply_fn)
    if not output_strs:
        return False
    return canonical_str(curr_expr) in output_strs


def is_false_positive(
    prev_expr: sympy.Basic,
    curr_expr: sympy.Basic,
    apply_fn: ApplyFn,
    *,
    step_correct: bool,
) -> bool:
    """True when a correct step matches a manipulation output."""
    if not step_correct:
        return False
    output_strs = manipulation_outputs(prev_expr, apply_fn)
    if not output_strs:
        return False
    return canonical_str(curr_expr) in output_strs


def predict_transition(
    prev_expr: sympy.Basic,
    curr_expr: sympy.Basic,
    apply_fn: ApplyFn,
    *,
    step_correct: bool | None = None,
) -> bool:
    """Predict that a transition exhibits the manipulation (candidate and realized)."""
    return is_candidate(prev_expr, apply_fn) and is_realized(
        prev_expr,
        curr_expr,
        apply_fn,
        step_correct=step_correct,
    )
