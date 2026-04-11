# tests/test_microstructure.py

"""
Pytest suite for src/features/microstructure.py and its dependencies.

Covers:
    - All Numba kernels directly (pure array inputs)
    - All public feature functions (Series / DataFrame inputs)
    - compute_all_features (integration test)
    - bar_microstructure_features (tick → bar mapping)
    - Edge cases: NaN propagation, zero volume, flat prices,
                  misaligned indices, empty inputs
    - Output dtype guarantees (float32 throughout)
    - Known-value regression tests where the math is tractable
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.microstructure import (
    _amihud_lambda_kernel,
    _bar_features_kernel,
    _corwin_schultz_kernel,
    _hasbrouck_lambda_kernel,
    _kyle_lambda_kernel,
    _roll_impact_kernel,
    _roll_measure_kernel,
    _to_float64,
    _vpin_kernel,
    _wrap,
    amihud_lambda,
    bar_microstructure_features,
    compute_all_features,
    corwin_schultz_spread,
    hasbrouck_lambda,
    kyle_lambda,
    roll_impact,
    roll_measure,
    vpin,
)
from src.bars.information_bars import _tick_rule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    """Seeded RNG for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def n() -> int:
    return 500


@pytest.fixture(scope="module")
def prices(rng, n) -> np.ndarray:
    """
    Simulated mid prices via a random walk.
    Guaranteed positive (starts at 100, steps are small).
    """
    steps = rng.normal(0, 0.05, size=n)
    return np.maximum(100.0 + np.cumsum(steps), 1.0)


@pytest.fixture(scope="module")
def volumes(rng, n) -> np.ndarray:
    """Strictly positive integer-valued volumes."""
    return rng.integers(1, 500, size=n).astype(np.float64)


@pytest.fixture(scope="module")
def directions(prices) -> np.ndarray:
    """Tick directions derived from simulated prices."""
    return _tick_rule(prices)


@pytest.fixture(scope="module")
def datetime_index(n) -> pd.DatetimeIndex:
    return pd.date_range("2023-01-02 09:00:00", periods=n, freq="1s", tz="UTC")


@pytest.fixture(scope="module")
def price_series(prices, datetime_index) -> pd.Series:
    return pd.Series(prices, index=datetime_index, name="close")


@pytest.fixture(scope="module")
def volume_series(volumes, datetime_index) -> pd.Series:
    return pd.Series(volumes, index=datetime_index, name="volume")


@pytest.fixture(scope="module")
def tick_df(prices, volumes, datetime_index, rng, n) -> pd.DataFrame:
    """
    Minimal tick DataFrame mimicking the output of a real data feed.
    Includes bid, ask, mid_price, spread, spread_bps, volume.
    """
    spread = rng.uniform(0.0001, 0.001, size=n)
    bid    = prices - spread / 2
    ask    = prices + spread / 2
    return pd.DataFrame(
        {
            "bid":        bid,
            "ask":        ask,
            "mid_price":  prices,
            "spread":     spread,
            "spread_bps": spread / prices * 10_000,
            "volume":     volumes,
        },
        index=datetime_index,
    )


@pytest.fixture(scope="module")
def ohlc_df(tick_df) -> pd.DataFrame:
    """
    Synthetic OHLC bars built by grouping ticks into fixed 50-tick bars.
    Index = last tick of each bar + 1 µs (matches make_bars convention).
    """
    bar_size = 50
    n        = len(tick_df)
    n_bars   = n // bar_size
    records  = []

    for i in range(n_bars):
        chunk = tick_df.iloc[i * bar_size : (i + 1) * bar_size]
        records.append(
            {
                "time":       chunk.index[-1] + pd.Timedelta(microseconds=1),
                "open":       chunk["mid_price"].iloc[0],
                "high":       chunk["mid_price"].max(),
                "low":        chunk["mid_price"].min(),
                "close":      chunk["mid_price"].iloc[-1],
                "spread":     chunk["spread"].mean(),
                "spread_bps": chunk["spread_bps"].mean(),
                "tick_volume": len(chunk),
                "volume":     chunk["volume"].sum(),
            }
        )

    return pd.DataFrame(records).set_index("time")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def assert_float32_series(s: pd.Series, name: str | None = None) -> None:
    """Assert that a Series has dtype float32."""
    assert s.dtype == np.float32, (
        f"Expected float32, got {s.dtype}"
        + (f" for '{name}'" if name else "")
    )


def assert_same_length(result, source, label: str = "") -> None:
    assert len(result) == len(source), (
        f"{label}: length mismatch {len(result)} vs {len(source)}"
    )


# ---------------------------------------------------------------------------
# 1. _tick_rule
# ---------------------------------------------------------------------------


class TestTickRule:
    def test_output_values_are_plus_minus_one(self, prices):
        b = _tick_rule(prices)
        unique = np.unique(b)
        assert set(unique).issubset({-1.0, 1.0}), f"Unexpected values: {unique}"

    def test_first_element_defaults_to_one_when_no_prior_move(self):
        # Flat then move: first b must be +1
        p = np.array([100.0, 100.0, 101.0, 100.5])
        b = _tick_rule(p)
        assert b[0] == 1.0

    def test_carry_forward_on_zero_diff(self):
        # Price goes up then stays flat: direction should carry forward as +1
        p = np.array([100.0, 101.0, 101.0, 101.0])
        b = _tick_rule(p)
        assert b[1] == 1.0   # Up tick
        assert b[2] == 1.0   # Carry
        assert b[3] == 1.0   # Carry

    def test_down_tick(self):
        p = np.array([100.0, 99.0, 99.0])
        b = _tick_rule(p)
        assert b[1] == -1.0
        assert b[2] == -1.0   # Carry forward

    def test_same_length_as_input(self, prices):
        b = _tick_rule(prices)
        assert len(b) == len(prices)

    def test_single_element(self):
        b = _tick_rule(np.array([100.0]))
        assert len(b) == 1
        assert b[0] == 1.0


# ---------------------------------------------------------------------------
# 2. _roll_measure_kernel
# ---------------------------------------------------------------------------


class TestRollMeasureKernel:
    WINDOW = 20

    def test_output_length(self, prices):
        out = _roll_measure_kernel(prices, self.WINDOW)
        assert len(out) == len(prices)

    def test_output_dtype(self, prices):
        out = _roll_measure_kernel(prices, self.WINDOW)
        assert out.dtype == np.float32

    def test_leading_nans(self, prices):
        out = _roll_measure_kernel(prices, self.WINDOW)
        assert np.all(np.isnan(out[: self.WINDOW]))

    def test_non_nan_after_window(self, prices):
        out = _roll_measure_kernel(prices, self.WINDOW)
        # Not all values after window must be non-NaN (positive cov → NaN)
        # but at least some should be finite for a random walk
        assert np.any(np.isfinite(out[self.WINDOW :]))

    def test_non_negative_values(self, prices):
        out = _roll_measure_kernel(prices, self.WINDOW)
        finite = out[np.isfinite(out)]
        assert np.all(finite >= 0.0), "Roll measure must be non-negative"

    def test_flat_prices_gives_nan(self):
        """Flat prices → zero variance → cov = 0 → not negative → NaN."""
        flat = np.full(50, 100.0, dtype=np.float64)
        out  = _roll_measure_kernel(flat, 10)
        # cov of constant diff = 0, which is not < 0 → NaN
        assert np.all(np.isnan(out[10:]))

    def test_known_negative_covariance(self):
        """
        Construct a perfectly alternating series so cov(Δp_t, Δp_{t-1}) < 0.
        [100, 101, 100, 101, ...] → Δp alternates +1, -1 → cov = -1.
        Roll spread = 2 * sqrt(1) = 2.
        """
        n   = 60
        p   = np.where(np.arange(n) % 2 == 0, 100.0, 101.0).astype(np.float64)
        out = _roll_measure_kernel(p, 20)
        # After window, values should be 2.0 (allowing float tolerance)
        finite = out[20:]
        finite = finite[np.isfinite(finite)]
        assert len(finite) > 0
        np.testing.assert_allclose(finite, 2.0, rtol=1e-4)


# ---------------------------------------------------------------------------
# 3. _roll_impact_kernel
# ---------------------------------------------------------------------------


class TestRollImpactKernel:
    WINDOW = 20

    def test_output_length(self, prices, volumes):
        out = _roll_impact_kernel(prices, volumes, self.WINDOW)
        assert len(out) == len(prices)

    def test_output_dtype(self, prices, volumes):
        out = _roll_impact_kernel(prices, volumes, self.WINDOW)
        assert out.dtype == np.float32

    def test_nan_when_roll_is_nan(self, prices, volumes):
        out = _roll_impact_kernel(prices, volumes, self.WINDOW)
        assert np.all(np.isnan(out[: self.WINDOW]))

    def test_zero_volume_gives_nan(self):
        p   = np.where(np.arange(50) % 2 == 0, 100.0, 101.0).astype(np.float64)
        v   = np.zeros(50, dtype=np.float64)
        out = _roll_impact_kernel(p, v, 10)
        # Roll spread is valid but dv=0 → NaN
        assert np.all(np.isnan(out[10:]))


# ---------------------------------------------------------------------------
# 4. _corwin_schultz_kernel
# ---------------------------------------------------------------------------


class TestCorwinSchultzKernel:
    def test_output_lengths(self, prices, rng):
        n    = len(prices)
        high = prices + rng.uniform(0, 0.5, size=n)
        low  = prices - rng.uniform(0, 0.5, size=n)
        spread, sigma = _corwin_schultz_kernel(high, low)
        assert len(spread) == n
        assert len(sigma)  == n

    def test_output_dtypes(self, prices, rng):
        n    = len(prices)
        high = prices + rng.uniform(0, 0.5, size=n)
        low  = prices - rng.uniform(0, 0.5, size=n)
        spread, sigma = _corwin_schultz_kernel(high, low)
        assert spread.dtype == np.float32
        assert sigma.dtype  == np.float32

    def test_first_element_is_nan(self, prices, rng):
        n    = len(prices)
        high = prices + 0.5
        low  = prices - 0.5
        spread, sigma = _corwin_schultz_kernel(high, low)
        assert np.isnan(spread[0])
        assert np.isnan(sigma[0])

    def test_spread_non_negative_where_finite(self, prices, rng):
        n    = len(prices)
        high = prices + rng.uniform(0.01, 0.5, size=n)
        low  = prices - rng.uniform(0.01, 0.5, size=n)
        spread, _ = _corwin_schultz_kernel(high, low)
        finite    = spread[np.isfinite(spread)]
        assert np.all(finite >= 0.0)

    def test_sigma_non_negative_where_finite(self, prices, rng):
        n   = len(prices)
        high = prices + 0.5
        low  = prices - 0.5
        _, sigma = _corwin_schultz_kernel(high, low)
        finite   = sigma[np.isfinite(sigma)]
        assert np.all(finite >= 0.0)

    def test_high_equals_low_produces_nan(self):
        """H = L → ln(H/L) = 0 → alpha may be invalid."""
        p      = np.full(10, 100.0, dtype=np.float64)
        spread, sigma = _corwin_schultz_kernel(p, p)
        # All should be NaN or zero (alpha = 0 → spread = 0)
        for val in spread[1:]:
            assert np.isnan(val) or val == pytest.approx(0.0, abs=1e-6)

    def test_wider_hl_gives_larger_sigma(self, prices):
        """Wider high-low range → larger estimated volatility."""
        n      = len(prices)
        narrow_high = prices + 0.1
        narrow_low  = prices - 0.1
        wide_high   = prices + 2.0
        wide_low    = prices - 2.0

        _, sigma_narrow = _corwin_schultz_kernel(narrow_high, narrow_low)
        _, sigma_wide   = _corwin_schultz_kernel(wide_high, wide_low)

        fn = sigma_narrow[np.isfinite(sigma_narrow)]
        fw = sigma_wide[np.isfinite(sigma_wide)]
        if len(fn) > 0 and len(fw) > 0:
            assert fw.mean() > fn.mean()


# ---------------------------------------------------------------------------
# 5. _kyle_lambda_kernel
# ---------------------------------------------------------------------------


class TestKyleLambdaKernel:
    WINDOW = 30

    def test_output_length(self, prices, volumes, directions):
        out = _kyle_lambda_kernel(prices, volumes, directions, self.WINDOW)
        assert len(out) == len(prices)

    def test_output_dtype(self, prices, volumes, directions):
        out = _kyle_lambda_kernel(prices, volumes, directions, self.WINDOW)
        assert out.dtype == np.float32

    def test_leading_nans(self, prices, volumes, directions):
        out = _kyle_lambda_kernel(prices, volumes, directions, self.WINDOW)
        assert np.all(np.isnan(out[: self.WINDOW]))

    def test_finite_values_after_window(self, prices, volumes, directions):
        out    = _kyle_lambda_kernel(prices, volumes, directions, self.WINDOW)
        finite = out[self.WINDOW :]
        assert np.any(np.isfinite(finite))

    def test_positive_lambda_for_trending_price_and_buy_pressure(self):
        """
        Construct a scenario where every tick is a buy (b=+1, v=1)
        and price strictly increases → OLS slope (λ) should be positive.
        """
        n      = 100
        p      = np.linspace(100.0, 110.0, n)
        v      = np.ones(n, dtype=np.float64)
        b      = np.ones(n, dtype=np.float64)
        out    = _kyle_lambda_kernel(p, v, b, 20)
        finite = out[20:]
        finite = finite[np.isfinite(finite)]
        assert np.all(finite > 0), "Trending buy pressure → positive Kyle λ"


# ---------------------------------------------------------------------------
# 6. _amihud_lambda_kernel
# ---------------------------------------------------------------------------


class TestAmihudLambdaKernel:
    WINDOW = 20

    def test_output_length(self, prices, volumes):
        out = _amihud_lambda_kernel(prices, volumes, self.WINDOW)
        assert len(out) == len(prices)

    def test_output_dtype(self, prices, volumes):
        out = _amihud_lambda_kernel(prices, volumes, self.WINDOW)
        assert out.dtype == np.float32

    def test_leading_nans(self, prices, volumes):
        out = _amihud_lambda_kernel(prices, volumes, self.WINDOW)
        assert np.all(np.isnan(out[: self.WINDOW]))

    def test_non_negative(self, prices, volumes):
        out    = _amihud_lambda_kernel(prices, volumes, self.WINDOW)
        finite = out[np.isfinite(out)]
        assert np.all(finite >= 0.0)

    def test_zero_volume_gives_nan(self):
        p   = np.linspace(100, 110, 50)
        v   = np.zeros(50, dtype=np.float64)
        out = _amihud_lambda_kernel(p, v, 10)
        assert np.all(np.isnan(out[10:]))

    def test_larger_volume_gives_smaller_illiq(self):
        """Higher volume → lower illiquidity (ILLIQ inversely proportional to V)."""
        n  = 100
        p  = np.linspace(100.0, 110.0, n)

        v_small = np.full(n, 1.0)
        v_large = np.full(n, 1_000_000.0)

        out_small = _amihud_lambda_kernel(p, v_small, 20)
        out_large = _amihud_lambda_kernel(p, v_large, 20)

        fs = out_small[np.isfinite(out_small)]
        fl = out_large[np.isfinite(out_large)]

        assert fs.mean() > fl.mean()


# ---------------------------------------------------------------------------
# 7. _hasbrouck_lambda_kernel
# ---------------------------------------------------------------------------


class TestHasbruckLambdaKernel:
    WINDOW = 30

    def test_output_length(self, prices, volumes, directions):
        out = _hasbrouck_lambda_kernel(prices, volumes, directions, self.WINDOW)
        assert len(out) == len(prices)

    def test_output_dtype(self, prices, volumes, directions):
        out = _hasbrouck_lambda_kernel(prices, volumes, directions, self.WINDOW)
        assert out.dtype == np.float32

    def test_leading_nans(self, prices, volumes, directions):
        out = _hasbrouck_lambda_kernel(prices, volumes, directions, self.WINDOW)
        assert np.all(np.isnan(out[: self.WINDOW]))

    def test_finite_after_window(self, prices, volumes, directions):
        out    = _hasbrouck_lambda_kernel(prices, volumes, directions, self.WINDOW)
        finite = out[self.WINDOW :]
        assert np.any(np.isfinite(finite))

    def test_positive_for_buy_driven_price_increase(self):
        """
        Same logic as Kyle: buy pressure + rising price → positive λ.
        """
        n   = 100
        p   = np.linspace(100.0, 110.0, n)
        v   = np.ones(n) * 100.0
        b   = np.ones(n)
        out = _hasbrouck_lambda_kernel(p, v, b, 20)
        f   = out[20:]
        f   = f[np.isfinite(f)]
        assert np.all(f > 0)


# ---------------------------------------------------------------------------
# 8. _vpin_kernel
# ---------------------------------------------------------------------------


class TestVPINKernel:
    def test_output_length_plausible(self, volumes, directions):
        bucket_size = float(volumes.sum() / 500)
        vpin_vals, ends = _vpin_kernel(volumes, directions, bucket_size, 50)
        # At least one VPIN value
        assert len(vpin_vals) >= 1

    def test_vpin_bounds(self, volumes, directions):
        """VPIN ∈ [0, 1] by construction."""
        bucket_size = float(volumes.sum() / 500)
        vpin_vals, _ = _vpin_kernel(volumes, directions, bucket_size, 50)
        assert np.all(vpin_vals >= 0.0)
        assert np.all(vpin_vals <= 1.0 + 1e-6), f"Max VPIN: {vpin_vals.max()}"

    def test_vpin_dtype(self, volumes, directions):
        bucket_size = float(volumes.sum() / 500)
        vpin_vals, ends = _vpin_kernel(volumes, directions, bucket_size, 50)
        assert vpin_vals.dtype == np.float32

    def test_all_buys_gives_vpin_one(self):
        """
        If all ticks are buys, V^B = bucket_size per bucket, V^S = 0.
        |V^B - V^S| = bucket_size → VPIN = 1.
        """
        v   = np.ones(1000, dtype=np.float64)
        b   = np.ones(1000, dtype=np.float64)      # All buys
        out, _ = _vpin_kernel(v, b, 10.0, 10)
        assert np.allclose(out, 1.0, atol=1e-5)

    def test_balanced_buys_sells_gives_low_vpin(self):
        """
        Perfectly alternating buys/sells → V^B ≈ V^S → VPIN ≈ 0.
        """
        n = 2000
        v = np.ones(n, dtype=np.float64)
        b = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        out, _ = _vpin_kernel(v, b, 10.0, 10)
        # Each bucket: 5 buys + 5 sells → |5-5| = 0 → VPIN = 0
        assert np.allclose(out, 0.0, atol=1e-5)

    def test_ends_array_length(self, volumes, directions):
        bucket_size = float(volumes.sum() / 500)
        vpin_vals, ends = _vpin_kernel(volumes, directions, bucket_size, 50)
        # ends should cover all buckets contributing to any VPIN value
        assert len(ends) >= len(vpin_vals)


# ---------------------------------------------------------------------------
# 9. _bar_features_kernel
# ---------------------------------------------------------------------------


class TestBarFeaturesKernel:
    def test_output_shapes(self, prices, volumes, directions):
        n_bars = 10
        bar_size = len(prices) // n_bars
        starts = np.arange(n_bars, dtype=np.int64) * bar_size
        ends   = starts + bar_size - 1

        t, v, d, f = _bar_features_kernel(directions, volumes, prices, starts, ends)
        assert len(t) == n_bars
        assert len(v) == n_bars
        assert len(d) == n_bars
        assert len(f) == n_bars

    def test_output_dtypes(self, prices, volumes, directions):
        starts = np.array([0, 10], dtype=np.int64)
        ends   = np.array([9, 19], dtype=np.int64)
        t, v, d, f = _bar_features_kernel(directions, volumes, prices, starts, ends)
        for arr in (t, v, d, f):
            assert arr.dtype == np.float32

    def test_buy_fraction_bounds(self, prices, volumes, directions):
        n_bars = 10
        bar_size = len(prices) // n_bars
        starts = np.arange(n_bars, dtype=np.int64) * bar_size
        ends   = starts + bar_size - 1
        _, _, _, f = _bar_features_kernel(directions, volumes, prices, starts, ends)
        assert np.all(f >= 0.0)
        assert np.all(f <= 1.0)

    def test_all_buy_ticks(self):
        """When all ticks are buys: buy_fraction=1, tick_imb=n, vol_imb=Σv."""
        n       = 100
        b       = np.ones(n, dtype=np.float64)
        v       = np.full(n, 10.0)
        p       = np.full(n, 100.0)
        starts  = np.array([0], dtype=np.int64)
        ends    = np.array([n - 1], dtype=np.int64)

        t, vol, d, f = _bar_features_kernel(b, v, p, starts, ends)

        assert f[0]   == pytest.approx(1.0)
        assert t[0]   == pytest.approx(float(n))
        assert vol[0] == pytest.approx(float(n * 10))
        assert d[0]   == pytest.approx(float(n * 10 * 100))

    def test_all_sell_ticks(self):
        n      = 100
        b      = np.full(n, -1.0)
        v      = np.full(n, 10.0)
        p      = np.full(n, 100.0)
        starts = np.array([0], dtype=np.int64)
        ends   = np.array([n - 1], dtype=np.int64)

        t, vol, d, f = _bar_features_kernel(b, v, p, starts, ends)

        assert f[0]   == pytest.approx(0.0)
        assert t[0]   == pytest.approx(float(-n))
        assert vol[0] == pytest.approx(float(-n * 10))


# ---------------------------------------------------------------------------
# 10. Public function: roll_measure
# ---------------------------------------------------------------------------


class TestRollMeasure:
    def test_returns_series(self, price_series):
        result = roll_measure(price_series, window=20)
        assert isinstance(result, pd.Series)

    def test_float32_dtype(self, price_series):
        result = roll_measure(price_series, window=20)
        assert_float32_series(result, "roll_measure")

    def test_preserves_index(self, price_series):
        result = roll_measure(price_series, window=20)
        pd.testing.assert_index_equal(result.index, price_series.index)

    def test_accepts_numpy_input(self, prices):
        result = roll_measure(prices, window=20)
        assert isinstance(result, pd.Series)
        assert len(result) == len(prices)

    def test_name(self, price_series):
        result = roll_measure(price_series)
        assert result.name == "roll_measure"


# ---------------------------------------------------------------------------
# 11. Public function: roll_impact
# ---------------------------------------------------------------------------


class TestRollImpact:
    def test_returns_series(self, price_series, volume_series):
        result = roll_impact(price_series, volume_series, window=20)
        assert isinstance(result, pd.Series)

    def test_float32_dtype(self, price_series, volume_series):
        result = roll_impact(price_series, volume_series, window=20)
        assert_float32_series(result, "roll_impact")

    def test_preserves_index(self, price_series, volume_series):
        result = roll_impact(price_series, volume_series)
        pd.testing.assert_index_equal(result.index, price_series.index)


# ---------------------------------------------------------------------------
# 12. Public function: corwin_schultz_spread
# ---------------------------------------------------------------------------


class TestCorwinSchultzSpread:
    @pytest.fixture
    def hl_series(self, price_series, rng, n):
        high = price_series + rng.uniform(0.01, 0.5, size=n)
        low  = price_series - rng.uniform(0.01, 0.5, size=n)
        return high, low

    def test_returns_dataframe(self, hl_series):
        high, low = hl_series
        result = corwin_schultz_spread(high, low)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, hl_series):
        high, low = hl_series
        result = corwin_schultz_spread(high, low)
        assert "cs_spread" in result.columns
        assert "cs_sigma"  in result.columns

    def test_float32_dtypes(self, hl_series):
        high, low = hl_series
        result = corwin_schultz_spread(high, low)
        assert result["cs_spread"].dtype == np.float32
        assert result["cs_sigma"].dtype  == np.float32

    def test_preserves_index(self, hl_series, price_series):
        high, low = hl_series
        result = corwin_schultz_spread(high, low)
        pd.testing.assert_index_equal(result.index, price_series.index)

    def test_length(self, hl_series, n):
        high, low = hl_series
        result = corwin_schultz_spread(high, low)
        assert len(result) == n


# ---------------------------------------------------------------------------
# 13. Public function: kyle_lambda
# ---------------------------------------------------------------------------


class TestKyleLambda:
    def test_returns_series(self, price_series, volume_series):
        result = kyle_lambda(price_series, volume_series, window=20)
        assert isinstance(result, pd.Series)

    def test_float32(self, price_series, volume_series):
        result = kyle_lambda(price_series, volume_series, window=20)
        assert_float32_series(result, "kyle_lambda")

    def test_preserves_index(self, price_series, volume_series):
        result = kyle_lambda(price_series, volume_series)
        pd.testing.assert_index_equal(result.index, price_series.index)

    def test_with_explicit_b(self, price_series, volume_series):
        b      = pd.Series(
            np.ones(len(price_series)), index=price_series.index
        )
        result = kyle_lambda(price_series, volume_series, b=b, window=20)
        assert isinstance(result, pd.Series)

    def test_auto_b_derivation(self, price_series, volume_series):
        """Passing b=None should not raise and should return valid Series."""
        result = kyle_lambda(price_series, volume_series, b=None, window=20)
        assert isinstance(result, pd.Series)
        assert len(result) == len(price_series)


# ---------------------------------------------------------------------------
# 14. Public function: amihud_lambda
# ---------------------------------------------------------------------------


class TestAmihudLambda:
    def test_returns_series(self, price_series, volume_series):
        result = amihud_lambda(price_series, volume_series, window=20)
        assert isinstance(result, pd.Series)

    def test_float32(self, price_series, volume_series):
        result = amihud_lambda(price_series, volume_series)
        assert_float32_series(result, "amihud_lambda")

    def test_non_negative(self, price_series, volume_series):
        result = amihud_lambda(price_series, volume_series)
        finite = result.dropna()
        assert (finite >= 0).all()

    def test_preserves_index(self, price_series, volume_series):
        result = amihud_lambda(price_series, volume_series)
        pd.testing.assert_index_equal(result.index, price_series.index)


# ---------------------------------------------------------------------------
# 15. Public function: hasbrouck_lambda
# ---------------------------------------------------------------------------


class TestHasbruckLambda:
    def test_returns_series(self, price_series, volume_series):
        result = hasbrouck_lambda(price_series, volume_series, window=20)
        assert isinstance(result, pd.Series)

    def test_float32(self, price_series, volume_series):
        result = hasbrouck_lambda(price_series, volume_series)
        assert_float32_series(result, "hasbrouck_lambda")

    def test_preserves_index(self, price_series, volume_series):
        result = hasbrouck_lambda(price_series, volume_series)
        pd.testing.assert_index_equal(result.index, price_series.index)

    def test_auto_b_derivation(self, price_series, volume_series):
        result = hasbrouck_lambda(price_series, volume_series, b=None)
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# 16. Public function: vpin
# ---------------------------------------------------------------------------


class TestVPIN:
    def test_returns_series(self, volume_series, price_series):
        result = vpin(volume_series, price_series, n_buckets=20)
        assert isinstance(result, pd.Series)

    def test_float32(self, volume_series, price_series):
        result = vpin(volume_series, price_series, n_buckets=20)
        assert result.dtype == np.float32

    def test_bounds(self, volume_series, price_series):
        result = vpin(volume_series, price_series, n_buckets=20)
        assert (result >= 0).all()
        assert (result <= 1.0 + 1e-6).all()

    def test_explicit_bucket_size(self, volume_series, price_series):
        result = vpin(
            volume_series, price_series, bucket_size=100.0, n_buckets=10
        )
        assert isinstance(result, pd.Series)
        assert len(result) >= 1

    def test_invalid_bucket_size_raises(self, volume_series, price_series):
        with pytest.raises(ValueError, match="bucket_size must be positive"):
            vpin(volume_series, price_series, bucket_size=0.0)

    def test_with_explicit_b(self, volume_series, price_series):
        b      = pd.Series(np.ones(len(volume_series)), index=volume_series.index)
        result = vpin(volume_series, price_series, b=b, n_buckets=10)
        # All buys → VPIN should be 1
        assert np.allclose(result.values, 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# 17. bar_microstructure_features
# ---------------------------------------------------------------------------


class TestBarMicrostructureFeatures:
    def test_returns_dataframe(self, tick_df, ohlc_df):
        result = bar_microstructure_features(tick_df, ohlc_df)
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self, tick_df, ohlc_df):
        result = bar_microstructure_features(tick_df, ohlc_df)
        expected = {
            "tick_imbalance",
            "volume_imbalance",
            "dollar_imbalance",
            "buy_fraction",
        }
        assert expected.issubset(set(result.columns))

    def test_index_matches_ohlc(self, tick_df, ohlc_df):
        result = bar_microstructure_features(tick_df, ohlc_df)
        pd.testing.assert_index_equal(result.index, ohlc_df.index)

    def test_float32_dtypes(self, tick_df, ohlc_df):
        result = bar_microstructure_features(tick_df, ohlc_df)
        for col in result.columns:
            assert result[col].dtype == np.float32, (
                f"Column '{col}' has dtype {result[col].dtype}, expected float32"
            )

    def test_buy_fraction_in_bounds(self, tick_df, ohlc_df):
        result = bar_microstructure_features(tick_df, ohlc_df)
        bf     = result["buy_fraction"].dropna()
        assert (bf >= 0.0).all()
        assert (bf <= 1.0).all()

    def test_missing_volume_raises(self, tick_df, ohlc_df):
        tick_no_vol = tick_df.drop(columns=["volume"])
        with pytest.raises(KeyError, match="volume"):
            bar_microstructure_features(tick_no_vol, ohlc_df)

    def test_custom_price_col(self, tick_df, ohlc_df):
        result = bar_microstructure_features(tick_df, ohlc_df, price_col="bid")
        assert isinstance(result, pd.DataFrame)

    def test_no_missing_bars(self, tick_df, ohlc_df):
        """Every bar in ohlc_df should appear in the result index."""
        result = bar_microstructure_features(tick_df, ohlc_df)
        assert len(result) == len(ohlc_df)


# ---------------------------------------------------------------------------
# 18. compute_all_features — integration tests
# ---------------------------------------------------------------------------


class TestComputeAllFeatures:
    def test_returns_dataframe(self, ohlc_df, tick_df):
        result = compute_all_features(ohlc_df, tick_df=tick_df)
        assert isinstance(result, pd.DataFrame)

    def test_index_matches_ohlc(self, ohlc_df, tick_df):
        result = compute_all_features(ohlc_df, tick_df=tick_df)
        pd.testing.assert_index_equal(result.index, ohlc_df.index)

    def test_expected_columns_present(self, ohlc_df, tick_df):
        result   = compute_all_features(ohlc_df, tick_df=tick_df)
        expected = {
            "roll_measure",
            "roll_impact",
            "cs_spread",
            "cs_sigma",
            "kyle_lambda",
            "amihud_lambda",
            "hasbrouck_lambda",
            "tick_imbalance",
            "volume_imbalance",
            "dollar_imbalance",
            "buy_fraction",
            "vpin",
        }
        missing = expected - set(result.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_all_float32(self, ohlc_df, tick_df):
        result = compute_all_features(ohlc_df, tick_df=tick_df)
        for col in result.columns:
            assert result[col].dtype == np.float32, (
                f"Column '{col}' dtype = {result[col].dtype}"
            )

    def test_without_tick_df(self, ohlc_df):
        """Without tick_df, bar features and VPIN should be omitted."""
        result = compute_all_features(ohlc_df, tick_df=None)
        assert isinstance(result, pd.DataFrame)

        tick_only_cols = {
            "tick_imbalance", "volume_imbalance",
            "dollar_imbalance", "buy_fraction", "vpin",
        }
        assert not tick_only_cols.intersection(set(result.columns)), (
            "tick-dependent columns should not appear when tick_df=None"
        )

    def test_without_volume_column(self, ohlc_df, tick_df):
        """ohlc_df without volume → volume-dependent features skipped."""
        ohlc_no_vol = ohlc_df.drop(columns=["volume"])
        result      = compute_all_features(ohlc_no_vol, tick_df=tick_df)
        vol_cols    = {"roll_impact", "kyle_lambda", "amihud_lambda", "hasbrouck_lambda"}
        assert not vol_cols.intersection(set(result.columns))

    def test_missing_close_raises(self, ohlc_df, tick_df):
        ohlc_no_close = ohlc_df.drop(columns=["close"])
        with pytest.raises(KeyError, match="close"):
            compute_all_features(ohlc_no_close, tick_df=tick_df)

    def test_custom_window(self, ohlc_df, tick_df):
        result = compute_all_features(ohlc_df, tick_df=tick_df, window=10)
        assert isinstance(result, pd.DataFrame)

    def test_include_flags(self, ohlc_df, tick_df):
        result = compute_all_features(
            ohlc_df,
            tick_df              = tick_df,
            include_bar_features = False,
            include_vpin         = False,
        )
        excluded = {"tick_imbalance", "volume_imbalance",
                    "dollar_imbalance", "buy_fraction", "vpin"}
        assert not excluded.intersection(set(result.columns))

    def test_no_inf_values(self, ohlc_df, tick_df):
        result = compute_all_features(ohlc_df, tick_df=tick_df)
        for col in result.columns:
            assert not np.any(np.isinf(result[col].to_numpy(na_value=0.0))), (
                f"Inf found in column '{col}'"
            )


# ---------------------------------------------------------------------------
# 19. Helper utilities
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_to_float64_from_series(self, price_series):
        arr = _to_float64(price_series)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert len(arr) == len(price_series)

    def test_to_float64_from_array(self, prices):
        arr = _to_float64(prices)
        assert arr.dtype == np.float64

    def test_to_float64_from_float32_array(self):
        x   = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        arr = _to_float64(x)
        assert arr.dtype == np.float64

    def test_wrap_with_series_source(self, price_series, prices):
        raw    = np.ones(len(prices), dtype=np.float32)
        result = _wrap(raw, price_series, "test")
        assert isinstance(result, pd.Series)
        assert result.name == "test"
        pd.testing.assert_index_equal(result.index, price_series.index)

    def test_wrap_with_array_source(self, prices):
        raw    = np.ones(len(prices), dtype=np.float32)
        result = _wrap(raw, prices, "test")
        assert isinstance(result, pd.Series)
        assert list(result.index) == list(range(len(prices)))


# ---------------------------------------------------------------------------
# 20. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_price_tick_rule(self):
        b = _tick_rule(np.array([100.0]))
        assert len(b) == 1
        assert b[0] == 1.0

    def test_two_prices_tick_rule(self):
        b = _tick_rule(np.array([100.0, 101.0]))
        assert b[1] == 1.0

        b2 = _tick_rule(np.array([100.0, 99.0]))
        assert b2[1] == -1.0

    def test_roll_measure_window_larger_than_data(self):
        """Window > data length → all NaN, no crash."""
        p   = np.linspace(100, 110, 10, dtype=np.float64)
        out = _roll_measure_kernel(p, 50)
        assert np.all(np.isnan(out))

    def test_corwin_schultz_minimum_data(self):
        """Two observations → one valid output at index 1."""
        high = np.array([101.0, 102.0], dtype=np.float64)
        low  = np.array([99.0, 98.0],   dtype=np.float64)
        spread, sigma = _corwin_schultz_kernel(high, low)
        assert len(spread) == 2
        assert np.isnan(spread[0])

    def test_vpin_n_buckets_larger_than_data(self):
        """Fewer buckets than n_buckets window → empty VPIN output."""
        v   = np.ones(5, dtype=np.float64)
        b   = np.ones(5, dtype=np.float64)
        out, ends = _vpin_kernel(v, b, 1.0, 100)
        # Can't form a full window → 0 VPIN values
        assert len(out) == 0

    def test_bar_features_single_bar(self, prices, volumes, directions):
        starts = np.array([0],              dtype=np.int64)
        ends   = np.array([len(prices) - 1], dtype=np.int64)
        t, v, d, f = _bar_features_kernel(directions, volumes, prices, starts, ends)
        assert len(t) == 1
        assert 0.0 <= f[0] <= 1.0

    def test_nan_prices_in_roll(self):
        """NaN in prices propagates gracefully (no crash)."""
        p      = np.array([100.0, np.nan, 101.0, 100.0, 102.0] * 10)
        result = _roll_measure_kernel(p, 5)
        assert len(result) == len(p)

    def test_misaligned_tick_ohlc_index(self, tick_df):
        """
        OHLC bars whose timestamps don't perfectly align with tick data
        should return NaN rows rather than raising.
        """
        # Shift ohlc index by 1 day — no tick will map to these bars
        shifted_ohlc = pd.DataFrame(
            {
                "close": [100.0, 101.0],
                "high":  [101.0, 102.0],
                "low":   [99.0, 100.0],
                "volume": [1000, 2000],
            },
            index=pd.date_range("2025-01-01", periods=2, freq="1h", tz="UTC"),
        )
        result = bar_microstructure_features(tick_df, shifted_ohlc)
        # All bars should be NaN (no tick maps to these times)
        assert result.isna().all().all()
