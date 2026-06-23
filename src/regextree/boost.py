from __future__ import annotations

import logging
import re as _re
from typing import Sequence

import numpy as np

from .display import DisplayMixin
from .llm import LLMClient, get_client
from .prompt import generate_candidates
from .rule import Rule, compute_logits
from .scoring import (
    compute_gradients,
    compute_rule_weight,
    cross_entropy_loss,
    find_hardest_class,
    score_candidate,
    softmax,
)

logger = logging.getLogger(__name__)


class RegexBoostClassifier(DisplayMixin):
    """Gradient-boosted text classifier using LLM-generated regex patterns."""

    def __init__(
        self,
        pool_size: int = 20,
        step_count: int = 5,
        eta: float = 0.2,
        llm: str | LLMClient = "gemini",
        sample_size: int = 30,
        early_stopping_rounds: int | None = None,
        verbose: bool = True,
        **llm_kwargs,
    ):
        self.pool_size = pool_size
        self.step_count = step_count
        self.eta = eta
        self.sample_size = sample_size
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose

        self._llm_spec = llm
        self._llm_kwargs = llm_kwargs
        self._client: LLMClient | None = None

        self._rules: list[Rule] = []
        self._n_classes: int = 0
        self._classes: np.ndarray = np.array([])
        self._descriptions: dict[str, str] = {}
        self._fitted: bool = False

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = get_client(self._llm_spec, **self._llm_kwargs)
        return self._client

    def fit(
        self,
        x: Sequence[str],
        y: Sequence[int],
        eval_set: dict[str, tuple[Sequence[str], Sequence[int]]] | None = None,
    ) -> RegexBoostClassifier:
        x_arr = list(x)
        y_arr = np.asarray(y, dtype=int)
        self._classes = np.unique(y_arr)
        self._n_classes = len(self._classes)

        # Remap labels to 0..K-1 if needed
        label_map = {c: i for i, c in enumerate(self._classes)}
        labels = np.array([label_map[yi] for yi in y_arr])

        # Initialize logits from class priors
        counts = np.bincount(labels, minlength=self._n_classes).astype(float)
        priors = counts / counts.sum()
        logits = np.log(priors + 1e-12)[None, :].repeat(len(x_arr), axis=0)

        # Eval set
        eval_logits: dict[str, np.ndarray] = {}
        eval_labels: dict[str, np.ndarray] = {}
        if eval_set:
            for name, (x_e, y_e) in eval_set.items():
                eval_labels[name] = np.array([label_map[yi] for yi in y_e])
                eval_logits[name] = np.log(priors + 1e-12)[None, :].repeat(len(x_e), axis=0)

        best_val_loss = float("inf")
        rounds_without_improvement = 0

        for step in range(self.step_count):
            gradients = compute_gradients(logits, labels)
            target_class = find_hardest_class(gradients, labels)

            # Sample positives/negatives weighted by gradient magnitude
            positives, negatives = self._sample_examples(
                x_arr, labels, gradients, target_class
            )

            if not positives or not negatives:
                logger.warning(f"Step {step}: no valid samples for class {target_class}, skipping")
                continue

            # Generate candidate regexes
            candidates = generate_candidates(
                self.client, int(self._classes[target_class]),
                positives, negatives, self.pool_size,
            )

            if not candidates:
                logger.warning(f"Step {step}: LLM returned no valid regex candidates")
                continue

            # Score and select best candidate
            best_score = -float("inf")
            best_regex: str | None = None
            best_matches: np.ndarray | None = None

            for regex in candidates:
                compiled = _re.compile(regex)
                matches = np.array([compiled.search(t) is not None for t in x_arr], dtype=bool)
                score = score_candidate(matches, gradients, target_class)
                if score > best_score:
                    best_score = score
                    best_regex = regex
                    best_matches = matches

            if best_regex is None or best_matches is None or not best_matches.any():
                logger.warning(f"Step {step}: no candidate scored positively")
                continue

            # Compute weight and create rule
            weight = compute_rule_weight(best_matches, gradients, target_class, self.eta)
            if abs(weight) < 1e-8:
                continue

            rule = Rule(
                id=f"r{step:03d}",
                regex=best_regex,
                target_class=target_class,
                weight=weight,
                step=step,
            )
            self._rules.append(rule)

            # Update training logits
            logits[best_matches, target_class] += weight

            # Logging
            train_loss = cross_entropy_loss(logits, labels)
            train_acc = (logits.argmax(axis=1) == labels).mean()

            if self.verbose:
                msg = f"[step {step}] loss={train_loss:.4f} acc={train_acc:.3f} rule={best_regex!r} class={int(self._classes[target_class])} w={weight:.4f}"
                logger.info(msg)
                print(msg)

            # Eval set tracking
            if eval_set:
                for name, (x_e, _) in eval_set.items():
                    compiled = _re.compile(best_regex)
                    e_matches = np.array([compiled.search(t) is not None for t in x_e], dtype=bool)
                    eval_logits[name][e_matches, target_class] += weight

                    val_loss = cross_entropy_loss(eval_logits[name], eval_labels[name])
                    val_acc = (eval_logits[name].argmax(axis=1) == eval_labels[name]).mean()
                    if self.verbose:
                        print(f"  [{name}] loss={val_loss:.4f} acc={val_acc:.3f}")

                # Early stopping on first eval set
                first_name = next(iter(eval_set))
                val_loss = cross_entropy_loss(eval_logits[first_name], eval_labels[first_name])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1

                if (
                    self.early_stopping_rounds is not None
                    and rounds_without_improvement >= self.early_stopping_rounds
                ):
                    if self.verbose:
                        print(f"Early stopping at step {step}")
                    break

        self._fitted = True
        return self

    def predict(self, x: Sequence[str]) -> np.ndarray:
        """Predict class labels for input texts."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")
        logits = compute_logits(self._rules, x, self._n_classes)
        internal_preds = logits.argmax(axis=1)
        return self._classes[internal_preds]

    def predict_proba(self, x: Sequence[str]) -> np.ndarray:
        """Predict class probabilities for input texts."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")
        logits = compute_logits(self._rules, x, self._n_classes)
        return softmax(logits)

    def _sample_examples(
        self,
        x: list[str],
        labels: np.ndarray,
        gradients: np.ndarray,
        target_class: int,
    ) -> tuple[list[str], list[str]]:
        """Sample positive (target class) and negative examples, weighted by gradient magnitude."""
        pos_mask = labels == target_class
        neg_mask = ~pos_mask

        pos_indices = np.where(pos_mask)[0]
        neg_indices = np.where(neg_mask)[0]

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return [], []

        # Weight positives by how much they need this class (negative gradient)
        pos_weights = np.maximum(0.0, -gradients[pos_indices, target_class])
        if pos_weights.sum() < 1e-12:
            pos_weights = np.ones(len(pos_indices))
        pos_weights /= pos_weights.sum()

        # Weight negatives by how much they'd be hurt (positive gradient for target)
        neg_weights = np.maximum(0.0, gradients[neg_indices, target_class])
        if neg_weights.sum() < 1e-12:
            neg_weights = np.ones(len(neg_indices))
        neg_weights /= neg_weights.sum()

        n_pos = min(self.sample_size, len(pos_indices))
        n_neg = min(self.sample_size, len(neg_indices))

        rng = np.random.default_rng()
        sampled_pos = rng.choice(pos_indices, size=n_pos, replace=False, p=pos_weights)
        sampled_neg = rng.choice(neg_indices, size=n_neg, replace=False, p=neg_weights)

        positives = [x[i] for i in sampled_pos]
        negatives = [x[i] for i in sampled_neg]
        return positives, negatives
