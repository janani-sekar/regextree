from __future__ import annotations

import logging
import re as _re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .display import DisplayMixin
from .lasso import refit_lasso_weights
from .llm import LLMClient, get_client
from .prompt import generate_candidates
from .rule import Rule, compute_logits
from .scoring import (
    ObjectiveFn,
    ObjectiveName,
    compute_gradients,
    cross_entropy_loss,
    evaluate_candidate,
    find_hardest_class,
    softmax,
)

logger = logging.getLogger(__name__)


@dataclass
class BoostRound:
    """One generation step: sampled examples and LLM candidate regexes."""

    step: int
    target_class: int
    positives: list[str]
    negatives: list[str]
    candidates: list[str]


@dataclass
class SelectedCandidate:
    """Best candidate from a round after model-side selection."""

    regex: str
    matches: np.ndarray
    score: float
    weight: float
    target_class: int


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
        objective: ObjectiveName | ObjectiveFn = "cross_entropy",
        max_negative_match_rate: float | None = 0.5,
        weight_method: str = "gradient",
        lasso_C: float = 1.0,
        lasso_max_rules: int | None = None,
        prediction_threshold: float = 0.5,
        confuser_fraction: float = 0.5,
        **llm_kwargs,
    ):
        self.pool_size = pool_size
        self.step_count = step_count
        self.eta = eta
        self.sample_size = sample_size
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.objective = objective
        self.max_negative_match_rate = max_negative_match_rate
        self.weight_method = weight_method
        self.lasso_C = lasso_C
        self.lasso_max_rules = lasso_max_rules
        self.prediction_threshold = prediction_threshold
        self.confuser_fraction = confuser_fraction

        self._llm_spec = llm
        self._llm_kwargs = llm_kwargs
        self._client: LLMClient | None = None

        self._rules: list[Rule] = []
        self._n_classes: int = 0
        self._classes: np.ndarray = np.array([])
        self._descriptions: dict[str, str] = {}
        self._fitted: bool = False
        self._subgroup_labels: np.ndarray | None = None
        self._train_x: list[str] = []
        self._train_labels: np.ndarray = np.array([])

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
        subgroup_labels: Sequence[int] | None = None,
    ) -> RegexBoostClassifier:
        """Fit the model by looping generate -> select -> add, then optional LASSO refit."""
        x_arr = list(x)
        y_arr = np.asarray(y, dtype=int)
        self._classes = np.unique(y_arr)
        self._n_classes = len(self._classes)
        self._train_x = x_arr
        self._subgroup_labels = (
            np.asarray(subgroup_labels, dtype=int) if subgroup_labels is not None else None
        )

        label_map = {c: i for i, c in enumerate(self._classes)}
        labels = np.array([label_map[yi] for yi in y_arr])
        self._train_labels = labels

        counts = np.bincount(labels, minlength=self._n_classes).astype(float)
        priors = counts / counts.sum()
        logits = np.log(priors + 1e-12)[None, :].repeat(len(x_arr), axis=0)

        eval_logits: dict[str, np.ndarray] = {}
        eval_labels: dict[str, np.ndarray] = {}
        if eval_set:
            for name, (x_e, y_e) in eval_set.items():
                eval_labels[name] = np.array([label_map[yi] for yi in y_e])
                eval_logits[name] = np.log(priors + 1e-12)[None, :].repeat(len(x_e), axis=0)

        self._rules = []
        best_val_loss = float("inf")
        rounds_without_improvement = 0

        for step in range(self.step_count):
            round_data = self.generate_round(x_arr, labels, logits, step)
            if round_data is None:
                continue

            selected = self.select_candidate(round_data, x_arr, labels, logits)
            if selected is None:
                if self.verbose:
                    print(f"[step {step}] no candidate passed selection")
                continue

            rule = self.add_rule(selected, step)
            logits[selected.matches, selected.target_class] += selected.weight

            train_loss = cross_entropy_loss(logits, labels)
            train_acc = (logits.argmax(axis=1) == labels).mean()
            if self.verbose:
                msg = (
                    f"[step {step}] loss={train_loss:.4f} acc={train_acc:.3f} "
                    f"rule={selected.regex!r} class={int(self._classes[selected.target_class])} "
                    f"w={selected.weight:.4f} score={selected.score:.4f}"
                )
                logger.info(msg)
                print(msg)

            if eval_set:
                for name, (x_e, _) in eval_set.items():
                    compiled = _re.compile(selected.regex)
                    e_matches = np.array(
                        [compiled.search(t) is not None for t in x_e], dtype=bool
                    )
                    eval_logits[name][e_matches, selected.target_class] += selected.weight

                    val_loss = cross_entropy_loss(eval_logits[name], eval_labels[name])
                    val_acc = (eval_logits[name].argmax(axis=1) == eval_labels[name]).mean()
                    if self.verbose:
                        print(f"  [{name}] loss={val_loss:.4f} acc={val_acc:.3f}")

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

        if self.weight_method == "lasso" and self._rules:
            self.refit_weights(x_arr, labels)

        self._fitted = True
        return self

    def generate_round(
        self,
        x: Sequence[str],
        labels: np.ndarray,
        logits: np.ndarray,
        step: int,
    ) -> BoostRound | None:
        """Sample hard examples and prompt the LLM for candidate regexes."""
        x_arr = list(x)
        self._ensure_classes(labels)
        gradients = compute_gradients(logits, labels)
        target_class = find_hardest_class(gradients, labels)

        positives, negatives = self._sample_examples(
            x_arr, labels, gradients, target_class, self._subgroup_labels
        )
        if not positives or not negatives:
            logger.warning(f"Step {step}: no valid samples for class {target_class}, skipping")
            return None

        candidates = generate_candidates(
            self.client,
            int(self._classes[target_class]),
            positives,
            negatives,
            self.pool_size,
        )
        if not candidates:
            logger.warning(f"Step {step}: LLM returned no valid regex candidates")
            return None

        return BoostRound(
            step=step,
            target_class=target_class,
            positives=positives,
            negatives=negatives,
            candidates=candidates,
        )

    def select_candidate(
        self,
        round_data: BoostRound,
        x: Sequence[str],
        labels: np.ndarray,
        logits: np.ndarray,
    ) -> SelectedCandidate | None:
        """Score candidate regexes and return the best under the configured objective."""
        x_arr = list(x)
        gradients = compute_gradients(logits, labels)
        target_class = round_data.target_class

        best_score = -float("inf")
        best: SelectedCandidate | None = None

        for regex in round_data.candidates:
            compiled = _re.compile(regex)
            matches = np.array(
                [compiled.search(text) is not None for text in x_arr], dtype=bool
            )
            score, weight = evaluate_candidate(
                matches,
                logits,
                gradients,
                labels,
                target_class,
                self.eta,
                objective=self.objective,
                max_negative_match_rate=self.max_negative_match_rate,
            )
            if score > best_score and matches.any():
                best_score = score
                best = SelectedCandidate(
                    regex=regex,
                    matches=matches,
                    score=score,
                    weight=weight,
                    target_class=target_class,
                )

        return best

    def add_rule(self, selected: SelectedCandidate, step: int) -> Rule:
        """Append a selected candidate to the rule pool."""
        rule = Rule(
            id=f"r{step:03d}",
            regex=selected.regex,
            target_class=selected.target_class,
            weight=selected.weight,
            step=step,
        )
        self._rules.append(rule)
        return rule

    def refit_weights(
        self,
        x: Sequence[str] | None = None,
        labels: np.ndarray | None = None,
    ) -> RegexBoostClassifier:
        """Refit rule weights on the match matrix (currently LASSO only)."""
        if not self._rules:
            return self

        x_arr = list(x) if x is not None else self._train_x
        y_labels = labels if labels is not None else self._train_labels
        if not x_arr or y_labels.size == 0:
            raise RuntimeError("No training data available for weight refit.")

        if self.weight_method != "lasso":
            raise ValueError(f"Unsupported weight_method: {self.weight_method!r}")

        self._rules = refit_lasso_weights(
            self._rules,
            x_arr,
            y_labels,
            C=self.lasso_C,
            max_rules=self.lasso_max_rules,
        )
        if self.verbose:
            print(f"LASSO refit: kept {len(self._rules)} rules")
        return self

    def predict(self, x: Sequence[str]) -> np.ndarray:
        """Predict class labels for input texts."""
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")
        logits = compute_logits(self._rules, x, self._n_classes)

        if self._n_classes == 2 and self.prediction_threshold != 0.5:
            probs = softmax(logits)
            pos_class = 1
            internal_preds = (probs[:, pos_class] > self.prediction_threshold).astype(int)
            return self._classes[internal_preds]

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
        subgroup_labels: np.ndarray | None = None,
    ) -> tuple[list[str], list[str]]:
        """Sample positive and negative examples, optionally preferring confuser negatives."""
        pos_mask = labels == target_class
        neg_mask = ~pos_mask

        pos_indices = np.where(pos_mask)[0]
        neg_indices = np.where(neg_mask)[0]

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return [], []

        pos_weights = np.maximum(0.0, -gradients[pos_indices, target_class])
        if pos_weights.sum() < 1e-12:
            pos_weights = np.ones(len(pos_indices))
        pos_weights /= pos_weights.sum()

        n_pos = min(self.sample_size, len(pos_indices))
        n_neg = min(self.sample_size, len(neg_indices))

        rng = np.random.default_rng()
        sampled_pos = rng.choice(pos_indices, size=n_pos, replace=False, p=pos_weights)

        if subgroup_labels is not None and self.confuser_fraction > 0:
            pos_subgroups = np.unique(subgroup_labels[sampled_pos])
            confuser_mask = np.isin(subgroup_labels[neg_indices], pos_subgroups)
            confuser_indices = neg_indices[confuser_mask]
            other_indices = neg_indices[~confuser_mask]

            n_confusers = min(n_neg, int(round(n_neg * self.confuser_fraction)))
            n_other = n_neg - n_confusers

            sampled_neg_parts: list[np.ndarray] = []
            if len(confuser_indices) > 0 and n_confusers > 0:
                size = min(n_confusers, len(confuser_indices))
                confuser_weights = np.maximum(0.0, gradients[confuser_indices, target_class])
                if confuser_weights.sum() < 1e-12:
                    confuser_weights = np.ones(len(confuser_indices))
                confuser_weights /= confuser_weights.sum()
                sampled_neg_parts.append(
                    rng.choice(confuser_indices, size=size, replace=False, p=confuser_weights)
                )
            if len(other_indices) > 0 and n_other > 0:
                size = min(n_other, len(other_indices))
                other_weights = np.maximum(0.0, gradients[other_indices, target_class])
                if other_weights.sum() < 1e-12:
                    other_weights = np.ones(len(other_indices))
                other_weights /= other_weights.sum()
                sampled_neg_parts.append(
                    rng.choice(other_indices, size=size, replace=False, p=other_weights)
                )
            if sampled_neg_parts:
                sampled_neg = np.concatenate(sampled_neg_parts)
            else:
                sampled_neg = rng.choice(neg_indices, size=n_neg, replace=False)
        else:
            neg_weights = np.maximum(0.0, gradients[neg_indices, target_class])
            if neg_weights.sum() < 1e-12:
                neg_weights = np.ones(len(neg_indices))
            neg_weights /= neg_weights.sum()
            sampled_neg = rng.choice(neg_indices, size=n_neg, replace=False, p=neg_weights)

        positives = [x[i] for i in sampled_pos]
        negatives = [x[i] for i in sampled_neg]
        return positives, negatives

    def _ensure_classes(self, labels: np.ndarray) -> None:
        """Initialize class metadata when using split generate/select APIs."""
        if self._n_classes == 0:
            self._classes = np.unique(labels)
            self._n_classes = len(self._classes)
