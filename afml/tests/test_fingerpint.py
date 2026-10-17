"""
Test RegressionModelFingerprint and ClassificationModelFingerprint implementations.
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from ..feature_importance import (
    ClassificationModelFingerprint,
    RegressionModelFingerprint,
)

# pylint: disable=invalid-name
# pylint: disable=unsubscriptable-object


class TestModelFingerprint(unittest.TestCase):
    """
    Test model fingerprint functions
    """

    def setUp(self):
        """
        Regression data with known structure (load_boston was removed from
        scikit-learn 1.2): y depends linearly on 0 and 1, quadratically on 2, on the
        interaction of 3 and 4, and not at all on 5-7.
        """
        rng = np.random.default_rng(42)
        X = rng.normal(size=(300, 8))
        y = (3.0 * X[:, 0] + 1.0 * X[:, 1] + 2.0 * X[:, 2] ** 2 + 2.0 * X[:, 3] * X[:, 4]
             + rng.normal(scale=0.1, size=300))
        self.X, self.y = pd.DataFrame(X), pd.Series(y)

        self.reg_rf = RandomForestRegressor(n_estimators=50, random_state=42)
        self.reg_linear = LinearRegression(fit_intercept=True)
        self.reg_rf.fit(self.X, self.y)
        self.reg_linear.fit(self.X, self.y)

        self.reg_fingerprint = RegressionModelFingerprint()

    def test_linear_effect(self):
        """
        Test get_linear_effect for various regression models and num_values.
        """
        self.reg_fingerprint.fit(self.reg_linear, self.X, num_values=20)
        linear_effect, _, _ = self.reg_fingerprint.get_effects()

        # Normalized effects sum to one, and the largest coefficient dominates
        self.assertAlmostEqual(sum(linear_effect["norm"].values()), 1.0, delta=1e-9)
        self.assertEqual(max(linear_effect["norm"], key=linear_effect["norm"].get), 0)
        self.assertGreater(linear_effect["norm"][0], linear_effect["norm"][1])
        for noise in (5, 6, 7):
            self.assertLess(linear_effect["norm"][noise], linear_effect["norm"][1])

        self.reg_fingerprint.fit(self.reg_rf, self.X, num_values=20)
        rf_linear, _, _ = self.reg_fingerprint.get_effects()
        self.assertEqual(max(rf_linear["norm"], key=rf_linear["norm"].get), 0)

        # Increasing the number of samples doesn't change feature effect massively
        self.reg_fingerprint.fit(self.reg_linear, self.X, num_values=70)
        linear_effect_70, _, _ = self.reg_fingerprint.get_effects()
        for feature in range(self.X.shape[1]):
            self.assertAlmostEqual(
                linear_effect["norm"][feature], linear_effect_70["norm"][feature], delta=0.05
            )

    def test_non_linear_effect(self):
        """
        Test get_non_linear_effect for various regression models and num_values.
        """
        self.reg_fingerprint.fit(self.reg_rf, self.X, num_values=20)
        _, non_linear_effect, _ = self.reg_fingerprint.get_effects()

        # The quadratic feature carries the largest non-linear effect
        self.assertEqual(max(non_linear_effect["norm"], key=non_linear_effect["norm"].get), 2)

        self.reg_fingerprint.fit(self.reg_linear, self.X, num_values=20)
        _, non_linear_effect, _ = self.reg_fingerprint.get_effects()

        # Non-linear effect to linear model is zero
        for effect_value in non_linear_effect["raw"].values():
            self.assertAlmostEqual(effect_value, 0, delta=1e-8)

    def test_pairwise_effect(self):
        """
        Test compute_pairwise_effect for various regression models and num_values.
        """
        # Data whose only signal is the interaction of features 0 and 1, so a
        # forest's split artifacts around strong additive features cannot mask it
        rng = np.random.default_rng(7)
        X = pd.DataFrame(rng.normal(size=(400, 4)))
        y = pd.Series(2.0 * X[0] * X[1] + rng.normal(scale=0.1, size=400))
        rf = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y)
        combinations = [(0, 1), (2, 3), (0, 2)]
        self.reg_fingerprint.fit(rf, X, num_values=20, pairwise_combinations=combinations)
        _, _, pair_wise_effect = self.reg_fingerprint.get_effects()

        raw = pair_wise_effect["raw"]
        self.assertGreater(raw[str((0, 1))], 3 * raw[str((2, 3))])
        self.assertGreater(raw[str((0, 1))], 3 * raw[str((0, 2))])

        combinations = [(3, 4), (0, 1), (5, 6)]

        self.reg_fingerprint.fit(
            self.reg_linear, self.X, num_values=20, pairwise_combinations=combinations
        )
        _, _, pair_wise_effect = self.reg_fingerprint.get_effects()

        # Pairwise effect for linear model should be zero
        for pair in combinations:
            self.assertAlmostEqual(pair_wise_effect["raw"][str(pair)], 0, delta=1e-9)

    def test_classification_fingerpint(self):
        """
        Test model fingerprint values (linear, non-linear, pairwise) for classification model.
        """

        X, y = load_breast_cancer(return_X_y=True)
        X, y = pd.DataFrame(X), pd.Series(y)
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        clf_fingerpint = ClassificationModelFingerprint()
        clf_fingerpint.fit(clf, X, num_values=20, pairwise_combinations=[(0, 1), (2, 3), (8, 9)])

        linear_effect, non_linear_effect, pair_wise_effect = clf_fingerpint.get_effects()

        for feature, effect in zip([0, 2, 3, 8, 9], [0.0068, 0.0249, 0.014, 0]):
            self.assertAlmostEqual(linear_effect["raw"][feature], effect, delta=1e-3)

        for feature, effect in zip([0, 2, 3, 8, 9], [0.0062, 0.0217, 0.0155, 0.0013]):
            self.assertAlmostEqual(non_linear_effect["raw"][feature], effect, delta=1e-3)

        for comb, effect in zip([(0, 1), (2, 3), (8, 9)], [0.008, 0.0087, 0]):
            self.assertAlmostEqual(pair_wise_effect["raw"][str(comb)], effect, delta=1e-3)

    def test_plot_effects(self):
        """
        Test plot_effects function.
        """

        self.reg_fingerprint.fit(self.reg_rf, self.X, num_values=20)
        self.reg_fingerprint.plot_effects()

        self.reg_fingerprint.fit(
            self.reg_rf, self.X, num_values=20, pairwise_combinations=[(1, 2), (3, 5)]
        )
        self.reg_fingerprint.plot_effects()
