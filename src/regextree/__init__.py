"""regextree — Interpretable categorization via regex boosting and manipulation rules."""

from .boost import BoostRound, RegexBoostClassifier, SelectedCandidate
from .manipulation import (
    BankEntry,
    ManipulationBank,
    ManipulationRule,
    evaluate_rule_on_transitions,
    label_transitions,
    load_rule_from_module,
    summarize_labels,
)
from .rule import Rule
from .tree import RegexTreeClassifier

__all__ = [
    "BankEntry",
    "BoostRound",
    "ManipulationBank",
    "ManipulationRule",
    "RegexBoostClassifier",
    "RegexTreeClassifier",
    "Rule",
    "SelectedCandidate",
    "evaluate_rule_on_transitions",
    "label_transitions",
    "load_rule_from_module",
    "summarize_labels",
]
__version__ = "0.1.0"
