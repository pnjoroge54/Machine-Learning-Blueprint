"""
crisis_timing_models.py

Creative regime-detection / trade-timing toolkit for the USD-crisis scenario.

Framing: this treats the Dalio "Big Cycle" thesis as a SCENARIO to prepare detection
machinery for, not a forecast to be committed to. Every tool below is a two-sided
regime detector -- it will say "no regime shift here" just as readily as "regime
shift detected." None of it assumes the thesis is correct; it only tries to time
entries *if and when* the market itself starts behaving like the thesis predicts.

Sections
--------
1. AdaptiveTrendFilter   -- local-linear-trend Kalman filter (statsmodels). Replaces
                            fixed breakout levels ("DXY below 96") with a statistically
                            calibrated, self-adapting trend t-statistic.
2. KalmanHedgeRatio       -- dynamic hedge-ratio Kalman filter (Chan-style) for the pair
                            trades in the playbook (TIPS/nominals, financials/gold
                            miners, homebuilders/metals, 2s10s).
3. LatentCrisisIndex      -- hand-rolled sensor-fusion Kalman filter that replaces the
                            ad hoc points-based composite_score() with a principled
                            latent state estimated from several noisy indicators.
4. rolling_hurst          -- rescaled-range Hurst exponent; switches between momentum
                            and mean-reversion regimes.
5. rolling_sadf           -- simplified Sup-ADF explosiveness statistic (bubble
                            detection), illustrative -- prefer the existing, already
                            productionized structural-break test suite for real use.

Tested dependencies: numpy, pandas, scipy, statsmodels. No proprietary code is
imported; nothing here depends on the afml package. See the companion design
document for how these plug into the existing triple-barrier / trend-scanning /
meta-labeling / CPCV pipeline -- that integration is architecture, not code that can
be executed in this environment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents


# ---------------------------------------------------------------------------
# 1. Adaptive trend filter (local-linear-trend Kalman filter)
# ---------------------------------------------------------------------------
class AdaptiveTrendFilter:
    """
    Local-linear-trend Kalman filter for adaptive trend / momentum extraction.

    State-space form:
        y_t     = mu_t + eps_t                  (observation)
        mu_t    = mu_{t-1} + beta_{t-1} + eta_t  (level evolves with the trend)
        beta_t  = beta_{t-1} + zeta_t            (trend/velocity is itself a random walk)

    Unlike a fixed breakout level, both the level and its velocity are latent states
    updated optimally as each new observation arrives. The trend t-statistic (velocity
    divided by its own standard error) is a self-calibrating momentum signal -- the same
    underlying idea as trend-scanning's t-value-of-slope, but computed as a real-time
    filter instead of an in-sample label.
    """

    def __init__(self, stochastic_trend: bool = True):
        self.stochastic_trend = stochastic_trend
        self.results_ = None

    def fit(self, series: pd.Series) -> AdaptiveTrendFilter:
        model = UnobservedComponents(
            series.astype(float),
            level="local linear trend",
        )
        self.results_ = model.fit(disp=False)
        return self

    def trend_state(self) -> pd.DataFrame:
        """Smoothed level, trend velocity, its standard error, and the trend t-statistic."""
        if self.results_ is None:
            raise RuntimeError("call .fit() first")
        smoothed = self.results_.smoother_results
        # state order for local linear trend: [level, trend]
        level = smoothed.smoothed_state[0]
        trend = smoothed.smoothed_state[1]
        trend_var = smoothed.smoothed_state_cov[1, 1]
        trend_se = np.sqrt(np.clip(trend_var, 1e-12, None))
        t_stat = trend / trend_se
        idx = self.results_.data.row_labels
        return pd.DataFrame(
            {"level": level, "trend": trend, "trend_se": trend_se, "trend_tstat": t_stat},
            index=idx,
        )

    def regime_flag(self, t_threshold: float = 2.0, persistence: int = 5) -> pd.Series:
        """True where |trend t-stat| has exceeded t_threshold for `persistence` straight periods."""
        state = self.trend_state()
        exceeded = state["trend_tstat"].abs() > t_threshold
        return exceeded.rolling(persistence).sum() >= persistence


# ---------------------------------------------------------------------------
# 2. Dynamic hedge-ratio Kalman filter (Chan-style pairs regression)
# ---------------------------------------------------------------------------
class KalmanHedgeRatio:
    """
    Dynamic hedge-ratio Kalman filter for pair / spread trades.

    Treats [intercept, hedge_ratio] as a 2D latent state following a random walk,
    updated each period from the observed relationship y_t = a_t + b_t * x_t + noise.
    Produces a real-time hedge ratio and a z-scored spread for mean-reversion entries --
    a principled alternative to a fixed or once-estimated ratio, which matters directly
    here: the attached playbook itself flags "Correlation Breakdown" as a late-stage
    risk. A drifting hedge ratio is one way to notice that breakdown as it happens
    rather than after the pair trade has already lost money on a stale ratio.

    Applies to: TIPS vs. nominal Treasuries (breakeven widener), financials vs. gold
    miners, homebuilders vs. metals miners, 2s10s.

    delta: how fast the hedge ratio is allowed to drift (process noise). Smaller delta
    = slower, more stable ratio; larger delta = faster adaptation, noisier ratio.
    """

    def __init__(self, delta: float = 1e-4, measurement_noise: float = 1e-3):
        self.delta = delta
        self.measurement_noise = measurement_noise

    def run(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        n = len(y)
        theta = np.zeros(2)  # [intercept, hedge_ratio]
        P = np.eye(2)  # state covariance
        Vw = self.delta / (1 - self.delta) * np.eye(2)  # process noise covariance
        Ve = self.measurement_noise  # measurement noise variance

        intercepts = np.zeros(n)
        hedge_ratios = np.zeros(n)
        spreads = np.zeros(n)
        spread_vars = np.zeros(n)

        x_vals, y_vals = x.values, y.values
        for t in range(n):
            H = np.array([1.0, x_vals[t]])
            # predict
            P = P + Vw
            # innovation
            y_hat = H @ theta
            e = y_vals[t] - y_hat
            Q = H @ P @ H.T + Ve
            # update
            K = (P @ H) / Q
            theta = theta + K * e
            P = P - np.outer(K, H) @ P

            intercepts[t] = theta[0]
            hedge_ratios[t] = theta[1]
            spreads[t] = e
            spread_vars[t] = Q

        spread_z = spreads / np.sqrt(spread_vars)
        return pd.DataFrame(
            {
                "intercept": intercepts,
                "hedge_ratio": hedge_ratios,
                "spread": spreads,
                "spread_z": spread_z,
            },
            index=y.index,
        )


# ---------------------------------------------------------------------------
# 3. Latent crisis index (hand-rolled sensor-fusion Kalman filter)
# ---------------------------------------------------------------------------
class LatentCrisisIndex:
    """
    Kalman-filtered latent 'crisis intensity' index -- a principled replacement for the
    ad hoc points-based composite_score() in the attached USD Collapse Playbook, where
    each indicator crossing a fixed threshold adds a fixed number of points regardless
    of how noisy or informative that indicator actually is.

    Multiple indicators (DXY change, gold YoY change, real short rate, 30y yield change,
    foreign TIC flow, ...) are treated as noisy measurements of ONE latent state that
    evolves as a near-unit-root AR(1) process (phi close to 1, so a genuine regime shift
    is tracked durably rather than mean-reverting away spuriously). Each indicator's
    influence is weighted inversely to its own noise -- a shakier signal moves the index
    less. Missing observations (different indicators publish on different schedules --
    daily DXY vs. monthly TIC data) are handled naturally by skipping the update for any
    indicator not yet observed on a given date, no imputation required.

    Inputs must be pre-signed so that POSITIVE always means "more crisis-consistent"
    (e.g. use -DXY_zscore, +gold_YoY_zscore, -real_rate_zscore, +yield_change_zscore,
    -TIC_flow_zscore) and pre-standardized (z-scored) before passing in.
    """

    def __init__(self, phi: float = 0.98, process_noise: float = 0.05):
        self.phi = phi
        self.process_noise = process_noise

    def run(self, indicators: pd.DataFrame, noise_var: dict | None = None) -> pd.DataFrame:
        cols = indicators.columns
        noise_var = noise_var or {}
        R = {c: noise_var.get(c, 1.0) for c in cols}

        n = len(indicators)
        state, var = 0.0, 1.0
        states = np.zeros(n)
        state_vars = np.zeros(n)

        for t in range(n):
            # predict
            state = self.phi * state
            var = self.phi**2 * var + self.process_noise

            # sequential update -- fold in each available indicator one at a time
            row = indicators.iloc[t]
            for c in cols:
                y = row[c]
                if pd.isna(y):
                    continue
                Q = var + R[c]
                K = var / Q
                state = state + K * (y - state)
                var = (1 - K) * var

            states[t] = state
            state_vars[t] = var

        out = pd.DataFrame(
            {"crisis_index": states, "crisis_index_var": state_vars}, index=indicators.index
        )
        out["crisis_index_z"] = out["crisis_index"] / np.sqrt(out["crisis_index_var"])
        return out


# ---------------------------------------------------------------------------
# 4. Rolling Hurst exponent
# ---------------------------------------------------------------------------
def rolling_hurst(
    series: pd.Series, window: int = 100, min_lag: int = 2, max_lag: int = 20
) -> pd.Series:
    """
    Rolling Hurst exponent via rescaled-range-style analysis on the differenced series.

    H > 0.5: persistent / trending regime  -> momentum trades (DXY/gold breakout) more reliable
    H < 0.5: anti-persistent / mean-reverting -> pair trades more reliable
    H ~ 0.5: random walk -> no edge either way

    Use as an adaptive switch between the momentum-style trades and the mean-reversion
    pair trades in the playbook, instead of running both unconditionally.

    Note: this estimator has known finite-sample bias on short windows (a pure random
    walk can read noticeably below 0.5 with window <~100) -- documented in the Hurst
    estimation literature (e.g. Weron 2002). Read the *change* in rolling Hurst as the
    regime signal, not the absolute level against a textbook 0.5 threshold.
    """

    def _hurst(x: np.ndarray) -> float:
        lags = range(min_lag, max_lag)
        tau = [np.std(x[lag:] - x[:-lag]) for lag in lags]
        tau = [max(1e-08, t) for t in tau]
        slope, _ = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return slope

    return series.rolling(window).apply(_hurst, raw=True)


# ---------------------------------------------------------------------------
# 5. Simplified Sup-ADF explosiveness statistic
# ---------------------------------------------------------------------------
def rolling_sadf(series: pd.Series, min_window: int = 20, start_stride: int = 1) -> pd.Series:
    """
    Simplified Sup-ADF explosiveness statistic (Phillips-Wu-Yu 2011 style), computed on
    an expanding basis up to each point t. Right-tailed unit-root test: a RISING SADF
    signals explosive (bubble-like) behavior, the opposite framing from a standard ADF
    test's "is this stationary" question. No augmentation lags -- basic Sup-DF, not
    Sup-ADF, kept simple for illustration.

    This is a lightweight, O(n^2)-ish prototype for exploration. The existing
    structural-break test suite (StructuralBreakScanner / Feature Engineering Parts
    9-10) already implements the full GSADF with the double supremum over window start
    AND end points, is tested, and is almost certainly faster -- prefer that for
    anything beyond exploring this idea.
    """
    y = series.values.astype(float)
    n = len(y)
    sadf = np.full(n, np.nan)

    for t in range(min_window, n):
        y_t = y[: t + 1]
        best_tstat = -np.inf
        for start in range(0, len(y_t) - min_window, start_stride):
            window = y_t[start:]
            dy = np.diff(window)
            y_lag = window[:-1]
            X = np.column_stack([np.ones_like(y_lag), y_lag])
            beta, *_ = np.linalg.lstsq(X, dy, rcond=None)
            resid = dy - X @ beta
            dof = len(dy) - 2
            if dof <= 1:
                continue
            sigma2 = (resid @ resid) / dof
            XtX_inv = np.linalg.inv(X.T @ X)
            se_beta1 = np.sqrt(max(sigma2 * XtX_inv[1, 1], 1e-12))
            tstat = beta[1] / se_beta1
            best_tstat = max(best_tstat, tstat)
        sadf[t] = best_tstat

    return pd.Series(sadf, index=series.index)
