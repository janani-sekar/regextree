"""regextree — Interpretable categorization via regex boosting and manipulation rules."""

from .boost import BoostRound, RegexBoostClassifier, SelectedCandidate
from .manipulation import ManipulationRule, evaluate_rule_on_transitions, load_rule_from_module
from .rule import Rule
from .tree import RegexTreeClassifier

__all__ = [
    "BoostRound",
    "ManipulationRule",
    "RegexBoostClassifier",
    "RegexTreeClassifier",
    "Rule",
    "SelectedCandidate",
    "evaluate_rule_on_transitions",
    "load_rule_from_module",
]
__version__ = "0.1.0"
