from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

ObjectiveName = Literal["cross_entropy", "f1", "precision", "recall"]
ObjectiveFn = Callable[[np.ndarray, np.ndarray, int], float]


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax, numerically stable."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    """Mean cross-entropy loss. labels is 1D integer array."""
    probs = softmax(logits)
    n = len(labels)
    log_probs = np.log(probs[np.arange(n), labels] + 1e-12)
    return -log_probs.mean()


def compute_gradients(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Gradient of cross-entropy w.r.t. logits. Shape (n_samples, n_classes).

    grad[i, c] = p(c|x_i) - 1{y_i == c}
    Positive gradient for a class means the model is over-predicting it.
    Negative gradient means it's under-predicting (needs more weight there).
    """
    probs = softmax(logits)
    n = len(labels)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), labels] = 1.0
    return probs - one_hot


def find_hardest_class(gradients: np.ndarray, labels: np.ndarray) -> int:
    """Identify the class with highest total negative gradient mass.

    This is the class the model is most under-predicting — samples that
    belong to this class have large negative grad[i, c], meaning the model
    needs to push logits up for this class. Uses mean (not sum) so that
    minority classes aren't disadvantaged by having fewer samples.
    """
    n_classes = gradients.shape[1]
    class_need = np.zeros(n_classes)
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        class_need[c] = -gradients[mask, c].mean()
    return int(np.argmax(class_need))


def negative_match_rate(
    regex_matches: np.ndarray,
    labels: np.ndarray,
    target_class: int,
) -> float:
    """Fraction of non-target samples matched by a candidate regex."""
    neg_mask = labels != target_class
    if not neg_mask.any():
        return 0.0
    return float(regex_matches[neg_mask].mean())


def passes_negative_match_gate(
    regex_matches: np.ndarray,
    labels: np.ndarray,
    target_class: int,
    max_negative_match_rate: float | None,
) -> bool:
    """Reject candidates that fire on too large a share of negatives."""
    if max_negative_match_rate is None:
        return True
    return negative_match_rate(regex_matches, labels, target_class) <= max_negative_match_rate


def score_candidate(
    regex_matches: np.ndarray,
    gradients: np.ndarray,
    target_class: int,
) -> float:
    """Score a candidate regex by gradient-weighted gain.

    For samples where the regex matches, we'd add weight to target_class logit.
    The "gain" is the dot product of the match indicator with the negative
    gradient for target_class (i.e., how much the matching samples need this class).
    """
    need = -gradients[:, target_class]
    gain_if_match = np.where(regex_matches, need, 0.0).sum()
    penalty = np.where(regex_matches, np.maximum(0.0, -need), 0.0).sum()
    return gain_if_match - penalty


def compute_rule_weight(
    regex_matches: np.ndarray,
    gradients: np.ndarray,
    target_class: int,
    eta: float,
) -> float:
    """Compute the optimal weight for a rule using Newton's method step.

    For matched samples, optimal leaf value = -sum(grad) / sum(hess)
    where hess[i] = p_i * (1 - p_i) for the target class.
    """
    if not regex_matches.any():
        return 0.0

    grad_sum = gradients[regex_matches, target_class].sum()
    n_matched = regex_matches.sum()
    raw_weight = -grad_sum / max(n_matched, 1.0)
    return float(eta * np.clip(raw_weight, -2.0, 2.0))


def classification_scores(
    logits: np.ndarray,
    labels: np.ndarray,
    target_class: int,
) -> tuple[float, float, float]:
    """Return precision, recall, and F1 for a single target class."""
    preds = logits.argmax(axis=1)
    tp = int(((preds == target_class) & (labels == target_class)).sum())
    fp = int(((preds == target_class) & (labels != target_class)).sum())
    fn = int(((preds != target_class) & (labels == target_class)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_objective(
    objective: ObjectiveName | ObjectiveFn,
    logits: np.ndarray,
    labels: np.ndarray,
    target_class: int,
) -> float:
    """Score logits under an objective. Higher is better."""
    if callable(objective):
        return float(objective(logits, labels, target_class))

    if objective == "cross_entropy":
        return -cross_entropy_loss(logits, labels)
    if objective == "precision":
        return classification_scores(logits, labels, target_class)[0]
    if objective == "recall":
        return classification_scores(logits, labels, target_class)[1]
    if objective == "f1":
        return classification_scores(logits, labels, target_class)[2]
    raise ValueError(
        f"Unknown objective {objective!r}. "
        "Use 'cross_entropy', 'f1', 'precision', 'recall', or pass a callable."
    )


def evaluate_candidate(
    regex_matches: np.ndarray,
    logits: np.ndarray,
    gradients: np.ndarray,
    labels: np.ndarray,
    target_class: int,
    eta: float,
    objective: ObjectiveName | ObjectiveFn = "cross_entropy",
    max_negative_match_rate: float | None = None,
) -> tuple[float, float]:
    """Score a candidate regex and return (objective_score, rule_weight)."""
    if not passes_negative_match_gate(regex_matches, labels, target_class, max_negative_match_rate):
        return -float("inf"), 0.0

    weight = compute_rule_weight(regex_matches, gradients, target_class, eta)
    if abs(weight) < 1e-8:
        return -float("inf"), 0.0

    new_logits = logits.copy()
    new_logits[regex_matches, target_class] += weight
    score = evaluate_objective(objective, new_logits, labels, target_class)
    return score, weight
