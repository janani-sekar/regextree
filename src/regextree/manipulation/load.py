from __future__ import annotations

import importlib.util
from pathlib import Path

from .rule import ManipulationRule


def load_rule_from_module(path: str | Path, attr: str = "rule") -> ManipulationRule:
    """Load a ManipulationRule from a Python module written by an external agent.

    Accepted export shapes:
    1. `rule = ManipulationRule(...)`
    2. `apply_fn` plus `NAME`/`DESCRIPTION` (or `name`/`description`)
    """
    module_path = Path(path).resolve()
    if not module_path.is_file():
        raise FileNotFoundError(f"Rule module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("discovered_manipulation_rule", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    # Let exports import local scoring helpers if present
    scoring_dir = module_path.parent.parent / "scoring"
    if scoring_dir.is_dir():
        import sys

        sys.path.insert(0, str(scoring_dir))

    spec.loader.exec_module(module)

    if hasattr(module, attr):
        rule = getattr(module, attr)
        if isinstance(rule, ManipulationRule):
            return rule
        if hasattr(rule, "apply_fn") and hasattr(rule, "name"):
            return ManipulationRule(
                name=str(rule.name),
                description=str(getattr(rule, "description", "")),
                apply_fn=rule.apply_fn,
            )
        raise TypeError(
            f"Expected {attr!r} to be a ManipulationRule-like object, got {type(rule).__name__}"
        )

    if hasattr(module, "apply_fn"):
        name = getattr(module, "NAME", getattr(module, "name", None))
        description = getattr(module, "DESCRIPTION", getattr(module, "description", ""))
        if name is None:
            raise AttributeError(
                f"Module {module_path} defines apply_fn but no NAME/name"
            )
        return ManipulationRule(
            name=str(name),
            description=str(description),
            apply_fn=module.apply_fn,
        )

    raise AttributeError(
        f"Module {module_path} must define {attr!r} or apply_fn + NAME"
    )
