from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy

from .match import (
    ApplyFn,
    is_candidate,
    is_false_positive,
    is_realized,
    manipulation_outputs,
    predict_transition,
)


@dataclass
class ManipulationRule:
    """A manipulation spec: apply_fn(prev_expr) -> list of wrong next expressions."""

    name: str
    description: str
    apply_fn: ApplyFn

    def outputs(self, prev_expr: sympy.Basic) -> list[str]:
        return manipulation_outputs(prev_expr, self.apply_fn)

    def candidate(self, prev_expr: sympy.Basic) -> bool:
        return is_candidate(prev_expr, self.apply_fn)

    def realized(
        self,
        prev_expr: sympy.Basic,
        curr_expr: sympy.Basic,
        *,
        step_correct: bool | None = None,
    ) -> bool:
        return is_realized(
            prev_expr,
            curr_expr,
            self.apply_fn,
            step_correct=step_correct,
        )

    def false_positive(
        self,
        prev_expr: sympy.Basic,
        curr_expr: sympy.Basic,
        *,
        step_correct: bool,
    ) -> bool:
        return is_false_positive(
            prev_expr,
            curr_expr,
            self.apply_fn,
            step_correct=step_correct,
        )

    def predict(
        self,
        prev_expr: sympy.Basic,
        curr_expr: sympy.Basic,
        *,
        step_correct: bool | None = None,
    ) -> bool:
        return predict_transition(
            prev_expr,
            curr_expr,
            self.apply_fn,
            step_correct=step_correct,
        )

    def to_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "apply_fn": self.apply_fn,
        }

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> ManipulationRule:
        return cls(
            name=spec["name"],
            description=spec["description"],
            apply_fn=spec["apply_fn"],
        )
