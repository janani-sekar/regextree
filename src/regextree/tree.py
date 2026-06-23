from __future__ import annotations

import re as _re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .display import DisplayMixin
from .llm import LLMClient, get_client
from .prompt import generate_candidates
from .rule import Rule


@dataclass
class TreeNode:
    """A node in the regex decision tree."""

    rule: Rule | None = None
    class_distribution: np.ndarray = field(default_factory=lambda: np.array([]))
    prediction: int = 0
    left: TreeNode | None = None   # regex matches
    right: TreeNode | None = None  # regex does not match
    depth: int = 0
    n_samples: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.rule is None


class RegexTreeClassifier(DisplayMixin):
    """Decision tree classifier where each split is a regex match/no-match."""

    def __init__(
        self,
        pool_size: int = 10,
        max_depth: int = 4,
        min_samples_leaf: int = 2,
        llm: str | LLMClient = "gemini",
        sample_size: int = 30,
        verbose: bool = True,
        **llm_kwargs,
    ):
        self.pool_size = pool_size
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.sample_size = sample_size
        self.verbose = verbose

        self._llm_spec = llm
        self._llm_kwargs = llm_kwargs
        self._client: LLMClient | None = None

        self._root: TreeNode | None = None
        self._rules: list[Rule] = []
        self._n_classes: int = 0
        self._classes: np.ndarray = np.array([])
        self._descriptions: dict[str, str] = {}
        self._fitted: bool = False
        self._rule_counter: int = 0

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = get_client(self._llm_spec, **self._llm_kwargs)
        return self._client

    def fit(
        self,
        x: Sequence[str],
        y: Sequence[int],
    ) -> RegexTreeClassifier:
        x_arr = list(x)
        y_arr = np.asarray(y, dtype=int)
        self._classes = np.unique(y_arr)
        self._n_classes = len(self._classes)

        label_map = {c: i for i, c in enumerate(self._classes)}
        labels = np.array([label_map[yi] for yi in y_arr])
        indices = np.arange(len(x_arr))

        self._root = self._build_tree(x_arr, labels, indices, depth=0)
        self._fitted = True
        return self

    def predict(self, x: Sequence[str]) -> np.ndarray:
        if not self._fitted or self._root is None:
            raise RuntimeError("Model has not been fitted yet.")
        preds = np.array([self._predict_one(t, self._root) for t in x])
        return self._classes[preds]

    def predict_proba(self, x: Sequence[str]) -> np.ndarray:
        if not self._fitted or self._root is None:
            raise RuntimeError("Model has not been fitted yet.")
        proba = np.array([self._predict_proba_one(t, self._root) for t in x])
        return proba

    def print_tree(self, node: TreeNode | None = None, indent: str = "") -> str:
        """Return a human-readable tree representation."""
        if node is None:
            node = self._root
        if node is None:
            return "(empty tree)"

        lines = []
        if node.is_leaf:
            pred_class = int(self._classes[node.prediction])
            lines.append(f"{indent}LEAF → class {pred_class} (n={node.n_samples})")
        else:
            assert node.rule is not None
            lines.append(f"{indent}IF matches r'{node.rule.regex}':  (n={node.n_samples})")
            if node.left:
                lines.append(self.print_tree(node.left, indent + "  "))
            lines.append(f"{indent}ELSE:")
            if node.right:
                lines.append(self.print_tree(node.right, indent + "  "))
        return "\n".join(lines)

    def _build_tree(
        self,
        x_arr: list[str],
        labels: np.ndarray,
        indices: np.ndarray,
        depth: int,
    ) -> TreeNode:
        distribution = np.bincount(labels[indices], minlength=self._n_classes).astype(float)
        prediction = int(distribution.argmax())
        n_samples = len(indices)

        # Stop conditions
        if (
            depth >= self.max_depth
            or n_samples <= self.min_samples_leaf
            or len(np.unique(labels[indices])) == 1
        ):
            return TreeNode(
                class_distribution=distribution / distribution.sum(),
                prediction=prediction,
                depth=depth,
                n_samples=n_samples,
            )

        # Find best regex split
        best_regex, best_gain, best_left, best_right = self._find_best_split(
            x_arr, labels, indices
        )

        if best_regex is None or best_left is None or best_right is None:
            return TreeNode(
                class_distribution=distribution / distribution.sum(),
                prediction=prediction,
                depth=depth,
                n_samples=n_samples,
            )

        # Create rule
        rule_id = f"t{self._rule_counter:03d}"
        self._rule_counter += 1
        rule = Rule(
            id=rule_id,
            regex=best_regex,
            target_class=prediction,
            weight=best_gain,
            step=depth,
        )
        self._rules.append(rule)

        if self.verbose:
            print(f"{'  ' * depth}depth={depth} split: r'{best_regex}' (gain={best_gain:.4f}, left={len(best_left)}, right={len(best_right)})")

        node = TreeNode(
            rule=rule,
            class_distribution=distribution / distribution.sum(),
            prediction=prediction,
            depth=depth,
            n_samples=n_samples,
        )
        node.left = self._build_tree(x_arr, labels, best_left, depth + 1)
        node.right = self._build_tree(x_arr, labels, best_right, depth + 1)
        return node

    def _find_best_split(
        self,
        x_arr: list[str],
        labels: np.ndarray,
        indices: np.ndarray,
    ) -> tuple[str | None, float, np.ndarray | None, np.ndarray | None]:
        """Find the best regex split for the given subset."""
        subset_labels = labels[indices]
        classes_present = np.unique(subset_labels)

        if len(classes_present) < 2:
            return None, 0.0, None, None

        # Pick the majority and minority classes for positive/negative sampling
        counts = np.bincount(subset_labels, minlength=self._n_classes)
        target_class = int(counts.argmax())

        pos_idx = indices[subset_labels == target_class]
        neg_idx = indices[subset_labels != target_class]

        n_pos = min(self.sample_size, len(pos_idx))
        n_neg = min(self.sample_size, len(neg_idx))

        rng = np.random.default_rng()
        sampled_pos = rng.choice(pos_idx, size=n_pos, replace=False)
        sampled_neg = rng.choice(neg_idx, size=n_neg, replace=False)

        positives = [x_arr[i] for i in sampled_pos]
        negatives = [x_arr[i] for i in sampled_neg]

        candidates = generate_candidates(
            self.client, int(self._classes[target_class]),
            positives, negatives, self.pool_size,
        )

        if not candidates:
            return None, 0.0, None, None

        parent_impurity = self._gini(subset_labels)
        best_gain = 0.0
        best_regex: str | None = None
        best_left_idx: np.ndarray | None = None
        best_right_idx: np.ndarray | None = None

        for regex in candidates:
            compiled = _re.compile(regex)
            matches = np.array([compiled.search(x_arr[i]) is not None for i in indices])

            left_idx = indices[matches]
            right_idx = indices[~matches]

            if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
                continue

            # Information gain (Gini)
            left_impurity = self._gini(labels[left_idx])
            right_impurity = self._gini(labels[right_idx])
            n = len(indices)
            weighted_impurity = (
                len(left_idx) / n * left_impurity + len(right_idx) / n * right_impurity
            )
            gain = parent_impurity - weighted_impurity

            if gain > best_gain:
                best_gain = gain
                best_regex = regex
                best_left_idx = left_idx
                best_right_idx = right_idx

        return best_regex, best_gain, best_left_idx, best_right_idx

    def _gini(self, labels: np.ndarray) -> float:
        """Gini impurity for a set of labels."""
        if len(labels) == 0:
            return 0.0
        counts = np.bincount(labels, minlength=self._n_classes).astype(float)
        probs = counts / counts.sum()
        return 1.0 - (probs ** 2).sum()

    def _predict_one(self, text: str, node: TreeNode) -> int:
        if node.is_leaf or node.rule is None:
            return node.prediction
        if node.rule.matches(text):
            return self._predict_one(text, node.left) if node.left else node.prediction
        return self._predict_one(text, node.right) if node.right else node.prediction

    def _predict_proba_one(self, text: str, node: TreeNode) -> np.ndarray:
        if node.is_leaf or node.rule is None:
            return node.class_distribution
        if node.rule.matches(text):
            return self._predict_proba_one(text, node.left) if node.left else node.class_distribution
        return self._predict_proba_one(text, node.right) if node.right else node.class_distribution
