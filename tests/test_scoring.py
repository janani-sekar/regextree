"""Tests for regextree.scoring module."""

import numpy as np

from regextree.scoring import (
    compute_gradients,
    compute_rule_weight,
    cross_entropy_loss,
    find_hardest_class,
    score_candidate,
    softmax,
)


def test_softmax_uniform():
    logits = np.zeros((3, 4))
    probs = softmax(logits)
    np.testing.assert_allclose(probs, 0.25, atol=1e-7)


def test_softmax_sums_to_one():
    logits = np.random.randn(10, 5)
    probs = softmax(logits)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-7)


def test_cross_entropy_perfect():
    logits = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    labels = np.array([0, 1])
    loss = cross_entropy_loss(logits, labels)
    assert loss < 0.01


def test_cross_entropy_random():
    logits = np.zeros((4, 3))
    labels = np.array([0, 1, 2, 0])
    loss = cross_entropy_loss(logits, labels)
    np.testing.assert_allclose(loss, np.log(3), atol=0.01)


def test_gradients_shape():
    logits = np.random.randn(10, 4)
    labels = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
    grads = compute_gradients(logits, labels)
    assert grads.shape == (10, 4)


def test_gradients_sum_to_zero():
    """For each sample, gradients across classes sum to zero."""
    logits = np.random.randn(5, 3)
    labels = np.array([0, 1, 2, 0, 1])
    grads = compute_gradients(logits, labels)
    np.testing.assert_allclose(grads.sum(axis=1), 0.0, atol=1e-7)


def test_find_hardest_class():
    # Class 2 has zero logits while others are correct → class 2 is hardest
    logits = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 0.0],  # should be class 2 but logits are flat
        [0.0, 0.0, 0.0],  # should be class 2 but logits are flat
    ])
    labels = np.array([0, 1, 2, 2])
    grads = compute_gradients(logits, labels)
    hardest = find_hardest_class(grads, labels)
    assert hardest == 2


def test_score_candidate_positive():
    # Candidate matches samples that need the target class
    grads = np.array([
        [-0.5, 0.25, 0.25],  # sample 0 is class 0, needs class 0 (grad is negative)
        [0.3, -0.5, 0.2],    # sample 1 is class 1
        [0.3, 0.2, -0.5],    # sample 2 is class 2
    ])
    matches = np.array([True, False, False])
    score = score_candidate(matches, grads, target_class=0)
    assert score > 0


def test_compute_rule_weight():
    grads = np.array([
        [-0.8, 0.4, 0.4],
        [-0.6, 0.3, 0.3],
        [0.5, -0.3, -0.2],
    ])
    matches = np.array([True, True, False])
    weight = compute_rule_weight(matches, grads, target_class=0, eta=0.2)
    # Both matched samples have negative gradient for class 0 → positive weight
    assert weight > 0
