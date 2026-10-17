"""
Tests for the Phase 2 screening gates.

Unit tests check mechanics. Calibration tests, marked ``slow``, check that the
gates deliver the error rates they advertise: that the mutual-information screen
holds size on independent data, that it has power against a planted
relationship, and that the Westfall-Young correction controls the family-wise
error rate over a correlated feature family where Holm and Benjamini-Hochberg
make different tradeoffs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afml.screening.entropy_features import (
    bin_counts,
    entropy_screen,
    indicator_entropy,
    modal_occupancy,
)
from afml.screening.multiple_testing import (
    benjamini_hochberg,
    benjamini_yekutieli,
    bonferroni,
    effective_number_of_tests,
    holm_bonferroni,
    westfall_young_maxt,
)
from afml.screening.mutual_information import (
    mi_screen,
    miller_madow_bias,
    normalized_mutual_info,
)
from afml.screening.pipeline import screen_features
from afml.screening.stationarity import (
    assess_stationarity,
    rolling_stationarity,
    stationarity_gate,
)


# =============================================================================
# Helpers
# =============================================================================
def ar1(n, phi, rng, scale=1.0):
    e = rng.standard_normal(n) * scale
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def overlapping(n, h, rng):
    e = rng.standard_normal(n + h)
    return np.convolve(e, np.ones(h), mode="valid")[:n]


def feature_frame(n=3000, seed=0, phi=0.9):
    """A frame containing one informative feature and several controls."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    signal = ar1(n, phi, rng)
    return (
        pd.DataFrame(
            {
                "informative": signal,
                "noise_a": ar1(n, phi, rng),
                "noise_b": ar1(n, phi, rng),
                "saturated": np.clip(ar1(n, 0.99, rng), -0.05, 0.05),
                "constant": np.ones(n),
            },
            index=idx,
        ),
        signal,
        idx,
        rng,
    )


# =============================================================================
# Gate 2 — entropy
# =============================================================================
class TestEntropy:
    def test_constant_series_has_zero_entropy(self):
        assert indicator_entropy(np.ones(1000)) == 0.0
        assert modal_occupancy(np.ones(1000)) == 1.0

    def test_uniform_draw_approaches_maximum_entropy(self):
        rng = np.random.default_rng(1)
        assert indicator_entropy(rng.uniform(size=20_000), n_bins=16) > 0.99

    def test_quantile_binning_cannot_reject(self):
        """The reason ``uniform`` is the default.

        Equal-frequency bins give every non-degenerate feature an entropy of
        essentially 1.0, so a screen built on quantile binning passes everything,
        including a feature that spends 95% of its life in one state.
        """
        rng = np.random.default_rng(2)
        # Continuous but heavily skewed: uniform bins concentrate almost every
        # observation in the first bin, quantile bins spread them by construction.
        skewed = rng.lognormal(0.0, 3.0, 20_000)
        assert indicator_entropy(skewed, n_bins=16, method="quantile") > 0.9
        assert indicator_entropy(skewed, n_bins=16, method="uniform") < 0.2

    def test_modal_occupancy_catches_a_saturated_feature(self):
        rng = np.random.default_rng(3)
        saturated = np.r_[np.zeros(9000), rng.standard_normal(1000)]
        assert modal_occupancy(saturated, n_bins=16, method="uniform") > 0.85

    def test_bin_counts_conserve_mass(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal(5000)
        assert bin_counts(x, 20, "uniform").sum() == 5000
        assert bin_counts(x, 20, "quantile").sum() == 5000
        with pytest.raises(ValueError, match="uniform.*quantile"):
            bin_counts(x, 10, "gaussian")

    def test_screen_flags_degenerate_columns(self):
        features, _, _, _ = feature_frame()
        report = entropy_screen(features, min_entropy=0.5, max_modal_occupancy=0.5)
        assert not report.loc["constant", "passes"]
        assert not report.loc["saturated", "passes"]
        assert report.loc["informative", "passes"]
        assert report.index[0] in ("constant", "saturated")


# =============================================================================
# Multiple testing
# =============================================================================
class TestMultipleTesting:
    def test_ordering_of_conservatism(self):
        p = np.array([0.001, 0.01, 0.03, 0.2, 0.6])
        bonf, holm = bonferroni(p), holm_bonferroni(p)
        bh, by = benjamini_hochberg(p), benjamini_yekutieli(p)
        assert np.all(holm <= bonf + 1e-12)
        assert np.all(bh <= holm + 1e-12)
        assert np.all(by >= bh - 1e-12)

    def test_corrections_are_monotone_and_bounded(self):
        rng = np.random.default_rng(5)
        p = np.sort(rng.uniform(1e-4, 1, 40))
        for fn in (bonferroni, holm_bonferroni, benjamini_hochberg, benjamini_yekutieli):
            adj = fn(p)
            assert np.all((adj > 0) & (adj <= 1.0))
            assert np.all(np.diff(adj) >= -1e-12), fn.__name__

    def test_rejects_invalid_pvalues(self):
        with pytest.raises(ValueError, match="must lie in"):
            holm_bonferroni([0.0, 0.5])

    def test_westfall_young_uses_the_joint_maximum(self):
        rng = np.random.default_rng(6)
        null = rng.standard_normal((2000, 10))
        observed = np.full(10, 3.5)
        adj = westfall_young_maxt(observed, null)
        raw = np.array([(1 + np.sum(null[:, j] >= 3.5)) / 2001 for j in range(10)])
        assert np.all(adj >= raw - 1e-12), "the max null must dominate the marginal null"

    def test_westfall_young_exploits_correlation(self):
        """Perfectly correlated features cost nothing; independent ones cost K."""
        rng = np.random.default_rng(7)
        base = rng.standard_normal((3000, 1))
        identical = np.repeat(base, 12, axis=1)
        independent = rng.standard_normal((3000, 12))
        obs = np.full(12, 2.5)
        assert (
            westfall_young_maxt(obs, identical).mean()
            < westfall_young_maxt(obs, independent).mean()
        )

    def test_effective_number_of_tests_brackets_correctly(self):
        rng = np.random.default_rng(8)
        base = rng.standard_normal((3000, 1))
        assert effective_number_of_tests(np.repeat(base, 12, axis=1)) < 3.0
        assert effective_number_of_tests(rng.standard_normal((3000, 12))) > 6.0

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="feature columns"):
            westfall_young_maxt(np.zeros(5), np.zeros((100, 6)))


# =============================================================================
# Mutual information
# =============================================================================
class TestMutualInformation:
    def test_null_mi_is_positive_not_zero(self):
        """The finite-sample floor that makes a fixed threshold useless."""
        rng = np.random.default_rng(9)
        vals = [
            normalized_mutual_info(
                rng.standard_normal(2000), rng.standard_normal(2000), correction=None
            )
            for _ in range(100)
        ]
        assert np.mean(vals) > 0.002

    def test_miller_madow_reduces_the_floor(self):
        rng = np.random.default_rng(10)
        raw, corrected = [], []
        for _ in range(100):
            x, y = rng.standard_normal(2000), rng.standard_normal(2000)
            raw.append(normalized_mutual_info(x, y, correction=None))
            corrected.append(normalized_mutual_info(x, y, correction="miller_madow"))
        assert np.mean(corrected) < np.mean(raw)

    def test_miller_madow_bias_signs(self):
        """Positive on a sparse joint table, negative on a deterministic one.

        The negative branch is correct: for a near-deterministic relationship the
        plug-in estimator understates mutual information, so the correction adds
        rather than subtracts.
        """
        rng = np.random.default_rng(99)
        independent = rng.multinomial(4000, np.full(100, 0.01)).reshape(10, 10).astype(float)
        assert miller_madow_bias(independent) > 0.0
        assert miller_madow_bias(np.eye(10) * 5) < 0.0

    def test_detects_a_deterministic_relationship(self):
        rng = np.random.default_rng(11)
        x = rng.standard_normal(4000)
        assert normalized_mutual_info(x, np.abs(x)) > 0.2
        assert normalized_mutual_info(x, rng.standard_normal(4000)) < 0.05

    def test_screen_requires_bars_for_bar_permute(self):
        features, signal, idx, rng = feature_frame(1000)
        target = pd.Series(rng.standard_normal(1000), index=idx)
        with pytest.raises(ValueError, match="requires both"):
            mi_screen(features, target, scheme="bar_permute", n_perm=5)

    def test_screen_refuses_a_short_sample(self):
        rng = np.random.default_rng(12)
        idx = pd.date_range("2020-01-01", periods=40, freq="h")
        with pytest.raises(ValueError, match="refusing to calibrate"):
            mi_screen(
                pd.DataFrame({"a": rng.standard_normal(40)}, index=idx),
                pd.Series(rng.standard_normal(40), index=idx),
                n_perm=10,
            )

    def test_screen_returns_the_joint_null(self):
        features, signal, idx, rng = feature_frame(1500)
        target = pd.Series(overlapping(1500, 10, rng) + 0.5 * signal, index=idx)
        out = mi_screen(features[["informative", "noise_a"]], target, n_perm=50, seed=1)
        assert out.attrs["null_matrix"].shape == (50, 2)
        assert {"p_westfall_young", "p_holm", "p_benjamini_hochberg"} <= set(out.columns)
        assert not out["recomputed_feature"].any()


# =============================================================================
# Gate 1 — stationarity
# =============================================================================
class TestStationarity:
    def test_verdicts_on_known_processes(self):
        rng = np.random.default_rng(13)
        assert assess_stationarity(ar1(4000, 0.3, rng))["verdict"] == "stationary"
        assert assess_stationarity(np.cumsum(rng.standard_normal(4000)))["verdict"] in (
            "unit_root",
            "inconclusive",
        )

    def test_short_series_is_rejected(self):
        with pytest.raises(ValueError, match="at least 20"):
            assess_stationarity(np.arange(10.0))

    def test_rolling_reports_a_fraction(self):
        rng = np.random.default_rng(14)
        frac = rolling_stationarity(ar1(6000, 0.2, rng), n_windows=6, min_window=200)
        assert 0.0 <= frac <= 1.0 and frac > 0.5

    def test_rolling_returns_nan_when_windows_are_too_short(self):
        rng = np.random.default_rng(15)
        assert np.isnan(rolling_stationarity(rng.standard_normal(300), n_windows=8))

    def test_gate_separates_walks_from_noise(self):
        rng = np.random.default_rng(16)
        idx = pd.RangeIndex(4000)
        frame = pd.DataFrame(
            {"noise": ar1(4000, 0.2, rng), "walk": np.cumsum(rng.standard_normal(4000))},
            index=idx,
        )
        out = stationarity_gate(frame, n_windows=None)
        assert bool(out.loc["noise", "passes"])
        assert not bool(out.loc["walk", "passes"])
        assert set(out.columns) >= {"adf_p", "kpss_p", "verdict", "passes"}


# =============================================================================
# Pipeline
# =============================================================================
class TestPipeline:
    def test_gates_run_in_order_and_record_the_rejecting_gate(self):
        features, signal, idx, rng = feature_frame(2000, seed=17)
        features["walk"] = np.cumsum(rng.standard_normal(2000))
        target = pd.Series(overlapping(2000, 10, rng) + 1.5 * signal, index=idx)

        report = screen_features(features, target, n_perm=60, seed=2)
        assert report.rejected["walk"].startswith("stationarity")
        assert report.rejected["constant"] in ("entropy", "stationarity:inconclusive")
        summary = report.summary()
        assert "outcome" in summary.columns
        assert len(summary) == features.shape[1]

    def test_skip_stationarity_bypasses_gate_one(self):
        features, signal, idx, rng = feature_frame(1200, seed=18)
        target = pd.Series(overlapping(1200, 8, rng) + signal, index=idx)
        report = screen_features(features, target, n_perm=40, skip_stationarity=True)
        assert (report.stationarity["verdict"] == "skipped").all()

    def test_report_repr_is_informative(self):
        features, signal, idx, rng = feature_frame(1200, seed=19)
        target = pd.Series(overlapping(1200, 8, rng) + signal, index=idx)
        report = screen_features(features, target, n_perm=40)
        assert "selected" in repr(report)


# =============================================================================
# Calibration — the acceptance tests
# =============================================================================
@pytest.mark.slow
def test_acceptance_mi_screen_holds_size():
    """ACCEPTANCE: the null p-value distribution is uniform.

    Smoothed features against an overlapping label, the Blueprint configuration.
    Two checks. The rejection rate at nominal 0.05 is the headline, but with a
    tractable number of trials its Monte Carlo error is comparable to the
    quantity being measured. The mean p-value is the stronger test: under a
    correctly calibrated permutation null the p-value is uniform on (0, 1], so
    its mean is 0.5 regardless of the significance level chosen.

    Reference run at 200 trials: size 0.060 at alpha=0.05, 0.110 at 0.10,
    0.200 at 0.20, mean p 0.501.
    """
    trials = 150
    p_values = np.empty(trials)
    for t in range(trials):
        rng = np.random.default_rng(500 + t)
        n = 1500
        idx = pd.date_range("2020-01-01", periods=n, freq="h")
        features = pd.DataFrame({"f": ar1(n, 0.95, rng)}, index=idx)
        target = pd.Series(overlapping(n, 50, rng), index=idx)
        p_values[t] = float(mi_screen(features, target, n_perm=200, seed=t)["p_value"].iloc[0])

    size = float(np.mean(p_values < 0.05))
    mean_p = float(p_values.mean())
    assert 0.015 <= size <= 0.11, f"mi_screen size {size:.3f} outside [0.015, 0.11]"
    assert 0.43 <= mean_p <= 0.57, f"null p-values are not uniform; mean {mean_p:.3f}"


@pytest.mark.slow
def test_acceptance_mi_screen_has_power():
    """ACCEPTANCE: a planted relationship is detected at the adjusted level."""
    rng = np.random.default_rng(600)
    n = 3000
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    signal = ar1(n, 0.9, rng)
    features = pd.DataFrame(
        {"informative": signal, "noise_a": ar1(n, 0.9, rng), "noise_b": ar1(n, 0.9, rng)},
        index=idx,
    )
    target = pd.Series(3.0 * signal + overlapping(n, 20, rng), index=idx)
    out = mi_screen(features, target, n_perm=400, seed=1)
    assert bool(out.loc["informative", "passes"])
    assert out.index[0] == "informative"


@pytest.mark.slow
def test_acceptance_westfall_young_controls_family_wise_error():
    """ACCEPTANCE: FWER at or below nominal 0.05 over a correlated family.

    Twelve correlated but useless features against an overlapping label. A
    rejection of any feature is a family-wise error. The uncorrected rate is
    reported alongside to show what the correction is buying.
    """
    fwer_raw = fwer_wy = 0
    trials = 40
    for t in range(trials):
        rng = np.random.default_rng(700 + t)
        n = 1200
        idx = pd.date_range("2020-01-01", periods=n, freq="h")
        base = ar1(n, 0.95, rng)
        cols = {f"f{j}": base + 0.6 * ar1(n, 0.95, rng) for j in range(12)}
        features = pd.DataFrame(cols, index=idx)
        target = pd.Series(overlapping(n, 40, rng), index=idx)
        out = mi_screen(features, target, n_perm=200, seed=t)
        fwer_raw += bool((out["p_value"] < 0.05).any())
        fwer_wy += bool((out["p_westfall_young"] < 0.05).any())
    rate_wy = fwer_wy / trials
    assert rate_wy <= 0.15, f"Westfall-Young FWER {rate_wy:.3f} exceeds tolerance"
    assert fwer_wy <= fwer_raw, "correction must not reject more than the raw test"
