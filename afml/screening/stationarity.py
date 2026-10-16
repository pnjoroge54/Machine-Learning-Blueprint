"""
Stationarity gate for candidate features.

Supersedes ``afml.features.stationary.is_stationary``, which runs ADF alone in
twenty-eight lines and reports a boolean.

ADF alone cannot decide the question it is asked. Its null is a unit root, so
failure to reject is not evidence of stationarity, only absence of evidence
against a unit root. On a short or low-power sample the test fails to reject
almost everything, and a gate built on it passes non-stationary features whenever
the sample is small. KPSS inverts the null, so running both produces a four-way
outcome that separates the cases ADF conflates:

===================================  ==================================
ADF rejects, KPSS does not           stationary
ADF does not, KPSS rejects           unit root; route to fracdiff
both reject                          difference-stationary with a trend
neither rejects                      insufficient power; report, decide later
===================================  ==================================

The fourth cell is the one a single-test gate silently converts into a pass.

One important exclusion. This gate is for *raw* candidate features. It must not
be applied to the output of fractional differentiation. KPSS tests a
short-memory stationarity null, and an FFD series with ``0 < d < 1`` is
long-memory stationary by construction, so KPSS rejects it correctly and
uselessly. Measured on eight random walks, requiring both tests on FFD output
drives mean ``d`` from 0.20 to 0.78, destroying the memory the transform exists
to preserve. ``fracdiff_optimal`` therefore uses ADF alone, with a corrected lag
order; see ``FRACDIFF_FIXES.md``.

References
----------
Dickey, D. A., & Fuller, W. A. (1979).
Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

__all__ = ["VERDICTS", "assess_stationarity", "rolling_stationarity", "stationarity_gate"]

VERDICTS = ("stationary", "unit_root", "trend_stationary", "inconclusive")


def assess_stationarity(
    x,
    alpha: float = 0.05,
    adf_autolag: str | None = "AIC",
    adf_maxlag: int | None = None,
    kpss_nlags: str = "auto",
) -> dict:
    """Joint ADF and KPSS test on a single series.

    :param x: (pd.Series or np.ndarray) Series to test.
    :param alpha: (float) Significance level for both tests.
    :param adf_autolag: (str or None) Lag selection for ADF. ``"AIC"`` by default.
        A fixed low ``maxlag`` with ``autolag=None`` over-rejects the unit-root
        null when residuals remain serially correlated.
    :param adf_maxlag: (int or None) ADF lag ceiling. ``None`` applies the
        Schwert rule.
    :param kpss_nlags: (str or int) Bandwidth for the KPSS long-run variance.
    :return: (dict) ``adf_stat``, ``adf_p``, ``kpss_stat``, ``kpss_p``,
        ``verdict`` and ``n_obs``.

    .. note:: Both p-values are interpolated from lookup tables and are clipped at
       the table edges, so a KPSS p-value of 0.01 means "at most 0.01" and 0.10
       means "at least 0.10". The verdict logic uses only the comparison against
       ``alpha``, which the clipping does not affect.
    """
    values = np.asarray(getattr(x, "values", x), dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 20:
        raise ValueError(f"need at least 20 finite observations; got {values.size}")

    adf_stat, adf_p = adfuller(values, maxlag=adf_maxlag, regression="c", autolag=adf_autolag)[:2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # p-value clipped at the table edge
        kpss_stat, kpss_p = kpss(values, regression="c", nlags=kpss_nlags)[:2]

    adf_rejects = adf_p < alpha
    kpss_rejects = kpss_p < alpha
    if adf_rejects and not kpss_rejects:
        verdict = "stationary"
    elif not adf_rejects and kpss_rejects:
        verdict = "unit_root"
    elif adf_rejects and kpss_rejects:
        verdict = "trend_stationary"
    else:
        verdict = "inconclusive"

    return {
        "adf_stat": float(adf_stat),
        "adf_p": float(adf_p),
        "kpss_stat": float(kpss_stat),
        "kpss_p": float(kpss_p),
        "verdict": verdict,
        "n_obs": int(values.size),
    }


def rolling_stationarity(
    x, n_windows: int = 8, alpha: float = 0.05, min_window: int = 200, **kwargs
) -> float:
    """Fraction of contiguous subperiods that return a ``stationary`` verdict.

    A feature that is stationary in aggregate but fails in a third of its windows
    is structurally unstable, and aggregate testing hides that. The fraction is a
    diagnostic, not a gate: choose the threshold with the deployment horizon in
    mind, since a feature failing in one window out of eight may be fine for a
    model refit monthly and fatal for one fit once.

    :param x: (pd.Series or np.ndarray) Series to test.
    :param n_windows: (int) Number of non-overlapping subperiods.
    :param alpha: (float) Significance level.
    :param min_window: (int) Fewest observations for a window to be tested.
    :param kwargs: Forwarded to :func:`assess_stationarity`.
    :return: (float) Pass fraction in ``[0, 1]``, or ``np.nan`` if no window
        was long enough.
    """
    values = np.asarray(getattr(x, "values", x), dtype=float)
    values = values[np.isfinite(values)]
    size = values.size // n_windows
    if size < min_window:
        return float("nan")

    passes = 0
    tested = 0
    for i in range(n_windows):
        chunk = values[i * size : (i + 1) * size]
        try:
            passes += assess_stationarity(chunk, alpha=alpha, **kwargs)["verdict"] == "stationary"
            tested += 1
        except ValueError:
            continue
    return float(passes / tested) if tested else float("nan")


def stationarity_gate(
    features: pd.DataFrame,
    alpha: float = 0.05,
    n_windows: int | None = 8,
    auto_fracdiff: bool = False,
    fracdiff_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Joint ADF and KPSS verdicts for a feature matrix, with optional routing.

    :param features: (pd.DataFrame) One column per candidate feature.
    :param alpha: (float) Significance level.
    :param n_windows: (int) Subperiods for the rolling check. ``None`` skips it.
    :param auto_fracdiff: (bool) Route ``unit_root`` and ``trend_stationary``
        verdicts to :func:`afml.features.fracdiff.fracdiff_optimal` and record the
        recommended ``d``. Off by default.
    :param fracdiff_kwargs: (dict) Forwarded to ``fracdiff_optimal``.
    :return: (pd.DataFrame) Indexed by feature with the test statistics, the
        verdict, the rolling pass fraction, ``passes``, and when routing is on,
        ``recommended_d``.

    .. note:: ``auto_fracdiff`` transforms nothing. It records the ``d`` that
       would be required. Automatic transformation of a feature the researcher has
       not inspected is how silent look-ahead enters a pipeline: the recommended
       ``d`` is estimated on the full sample, so applying it without thought
       leaks information from the test period into the training features.
       Compute ``d`` on the training fold and carry it forward.
    """
    rows = []
    for name in features.columns:
        col = features[name]
        try:
            record = assess_stationarity(col, alpha=alpha)
        except ValueError as exc:
            rows.append(
                {"feature": name, "verdict": "inconclusive", "error": str(exc), "passes": False}
            )
            continue

        record["feature"] = name
        record["rolling_pass_fraction"] = (
            rolling_stationarity(col, n_windows=n_windows, alpha=alpha)
            if n_windows
            else float("nan")
        )
        record["passes"] = record["verdict"] == "stationary"

        if auto_fracdiff and record["verdict"] in ("unit_root", "trend_stationary"):
            from ..features.fracdiff import fracdiff_optimal

            try:
                _, d, _ = fracdiff_optimal(col.dropna(), **(fracdiff_kwargs or {}))
                record["recommended_d"] = d
            except Exception as exc:  # pragma: no cover - depends on caller data
                record["recommended_d"] = float("nan")
                record["error"] = str(exc)
        rows.append(record)

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).set_index("feature")
    order = {v: i for i, v in enumerate(VERDICTS)}
    return frame.sort_values("verdict", key=lambda s: s.map(order))
