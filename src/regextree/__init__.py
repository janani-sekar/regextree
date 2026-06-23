"""regextree — Interpretable text categorization via LLM-generated regex boosting."""

from .boost import RegexBoostClassifier
from .rule import Rule
from .tree import RegexTreeClassifier

__all__ = ["RegexBoostClassifier", "RegexTreeClassifier", "Rule"]
__version__ = "0.1.0"
