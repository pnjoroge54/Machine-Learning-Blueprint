"""
Tests for cross_validation.scoring.probability_weighted_accuracy and
bet_sizing.ecdf (ECDF position sizing for meta-labels).
"""

import numpy as np
import pandas as pd
import pytest

from ..bet_sizing.ecdf import apply_ecdf_sizing, train_ecdf_sizing
from ..cross_validation.scoring import probability_weighted_accuracy

# --------------------------------------------------------------------------- PWA


def test_pwa_perfect_and_all_wrong():
    y = np.array([0, 1, 1, 0])
    confident_right = np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9], [0.7, 0.3]])
    assert probability_weighted_accuracy(y, confident_right) == pytest.approx(1.0)
    assert probability_weighted_accuracy(1 - y, confident_right) == pytest.approx(0.0)


def test_pwa_equals_accuracy_when_confidence_is_constant():
    y = np.array([0, 1, 1, 0, 1])
    p1 = np.array([0.3, 0.7, 0.3, 0.3, 0.7])  # same confidence (0.7) on every prediction
    acc = np.mean((p1 > 0.5).astype(int) == y)
    assert probability_weighted_accuracy(y, p1) == pytest.approx(acc)


def test_pwa_penalizes_confident_mistakes_more():
    y = np.array([1, 1])
    mild_mistake = np.array([[0.1, 0.9], [0.55, 0.45]])
    bold_mistake = np.array([[0.1, 0.9], [0.95, 0.05]])
    labels = [0, 1]  # y alone holds one class
    assert (probability_weighted_accuracy(y, bold_mistake, labels=labels)
            < probability_weighted_accuracy(y, mild_mistake, labels=labels))


def test_pwa_uninformative_predictions_score_half():
    y = np.array([0, 1, 0])
    assert probability_weighted_accuracy(y, np.full((3, 2), 0.5)) == 0.5


def test_pwa_sample_weights_and_label_order():
    y = np.array(["down", "up", "up"])
    proba = pd.DataFrame({"up": [0.2, 0.9, 0.4], "down": [0.8, 0.1, 0.6]})
    # Third prediction is wrong; giving it zero weight leaves only correct calls
    assert probability_weighted_accuracy(y, proba, sample_weight=[1, 1, 0], labels=["down", "up"]) == pytest.approx(1.0)
    weighted = probability_weighted_accuracy(y, proba, sample_weight=[1, 1, 5], labels=["down", "up"])
    unweighted = probability_weighted_accuracy(y, proba, labels=["down", "up"])
    assert weighted < unweighted


@pytest.mark.parametrize(
    "y, proba, kwargs",
    [
        (np.array([0, 1]), np.ones((2, 3)) / 3, {}),
        (np.array([0, 2]), np.array([[0.6, 0.4], [0.3, 0.7]]), {"labels": [0, 1]}),
        (np.array([0, 1]), np.array([[0.6, 0.4], [0.3, 0.7]]), {"sample_weight": [1, 2, 3]}),
    ],
)
def test_pwa_rejects_bad_inputs(y, proba, kwargs):
    with pytest.raises(ValueError):
        probability_weighted_accuracy(y, proba, **kwargs)


# --------------------------------------------------------------------------- ECDF sizing


@pytest.fixture
def ecdf():
    probs = pd.Series([0.55, 0.60, 0.65, 0.70, 0.80, 0.52, 0.58])
    outcomes = pd.Series([1, 1, 1, 1, 1, 0, 0])
    return train_ecdf_sizing(probs, outcomes)


def test_ecdf_maps_profitable_probabilities_to_their_rank(ecdf):
    # Profitable probabilities 0.55..0.80 map to ranks 1/5..5/5
    for p, rank in zip([0.55, 0.60, 0.65, 0.70, 0.80], [0.2, 0.4, 0.6, 0.8, 1.0]):
        assert float(ecdf(p)) == pytest.approx(rank)
    assert float(ecdf(0.9)) == pytest.approx(1.0)


def test_apply_ecdf_sizing_threshold_bounds_and_monotonicity(ecdf):
    assert apply_ecdf_sizing(ecdf, 0.45) == 0.0
    assert apply_ecdf_sizing(ecdf, 0.5) == 0.0  # at the threshold: no position
    sizes = [apply_ecdf_sizing(ecdf, p) for p in np.linspace(0.51, 0.99, 25)]
    assert all(0.0 <= s <= 1.0 for s in sizes)
    assert all(a <= b + 1e-12 for a, b in zip(sizes, sizes[1:]))
    assert apply_ecdf_sizing(ecdf, 0.7, threshold=0.75) == 0.0


def test_train_ecdf_requires_profitable_trades():
    with pytest.raises(ValueError, match="No profitable"):
        train_ecdf_sizing(pd.Series([0.6, 0.7]), pd.Series([0, 0]))
