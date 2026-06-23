from __future__ import annotations

import numpy as np


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
    # negative gradient = how much this sample needs more logit for target_class
    need = -gradients[:, target_class]
    # Only count positive need (samples that actually need this class boosted)
    gain_if_match = np.where(regex_matches, need, 0.0).sum()
    # Penalize false positives: samples that DON'T need this class but would get it
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
    # Hessian for softmax cross-entropy: p * (1 - p)
    # But we don't have probs stored separately, so use a simpler approach:
    # weight = eta * (-grad_sum / n_matched), clamped for stability
    n_matched = regex_matches.sum()
    raw_weight = -grad_sum / max(n_matched, 1.0)
    return float(eta * np.clip(raw_weight, -2.0, 2.0))
