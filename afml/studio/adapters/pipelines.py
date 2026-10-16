"""
The single mapping from studio specs to afml pipeline functions.

This branch has the older ``afml.production.model_development`` functions, not the
``FeatureEngineeringPipeline`` / ``ModelDevelopmentPipeline`` classes the studio plan
targets. Every afml call the studio makes goes through this module, so rewiring to
the newer pipelines touches nothing else. Each public step function below is pure:
it takes plain dicts and DataFrames, returns tables and a summary, and never touches
the ledger or the vault (the runner does that).

Steps
-----
load_bars            data spec -> OHLC bars
bars_report          bars -> bar count, bars/day, return distribution, ADF by d
compute_events       bars + label spec -> triple-barrier events with both weight series
labels_report        events -> label balance, holding times, concurrency, uniqueness
build_feature_matrix bars + feature set spec -> X and the provenance catalog
features_report      X + catalog -> NaN share per column
run_importance       X + events + importance spec -> MDI / MDA / SFI under PurgedKFold
run_model            X + events + model spec -> purged-CV grid search (clf_hyper_fit)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from annotated_types import Ge, Le
from loguru import logger

from afml.cache import cacheable

from ..progress import log_progress

from ..registry import call, get, register

Tables = Dict[str, pd.DataFrame]
Progress = Callable[[float, str], None]


def _noop_progress(p: float, msg: str = "") -> None:
    pass


_cache_baseline: Dict[str, Dict[str, int]] = {}


def cache_report(reset: bool = True) -> Dict[str, Dict[str, int]]:
    """afml.cache hits and misses per cached function since the last reset."""
    from afml.cache import cache_stats

    now = {k: dict(v) for k, v in cache_stats.get_stats().items()}
    out: Dict[str, Dict[str, int]] = {}
    for name, v in now.items():
        base = _cache_baseline.get(name, {"hits": 0, "misses": 0})
        delta = {"hits": v["hits"] - base["hits"], "misses": v["misses"] - base["misses"]}
        if delta["hits"] or delta["misses"]:
            out[name] = delta
    if reset:
        _cache_baseline.clear()
        _cache_baseline.update(now)
    return out


# =========================================================================== registry
# --------------------------------------------------------------------------- sources


@register("source", "mt5_ticks", n_inputs=1,
          description="Ticks from the local parquet store (MT5 fetch on miss) built into bars")
def source_mt5_ticks(data: Dict[str, Any]) -> pd.DataFrame:
    from afml.production.model_development import load_and_prepare_training_data

    if not data.get("account"):
        raise ValueError("mt5_ticks needs an account")
    return load_and_prepare_training_data(
        data["symbol"], str(data["start_date"]), str(data["end_date"]), data["account"],
        data["bar_type"], data["bar_size"], data["price"],
    )


@register("source", "bars_parquet", n_inputs=1, description="Pre-built OHLC bars from a parquet file")
def source_bars_parquet(data: Dict[str, Any], path: str = "") -> pd.DataFrame:
    if not path:
        raise ValueError("bars_parquet needs a path")
    bars = pd.read_parquet(path)
    start = pd.Timestamp(data["start_date"])
    end = pd.Timestamp(data["end_date"]) + pd.Timedelta(days=1)
    if bars.index.tz is not None:
        start, end = start.tz_localize(bars.index.tz), end.tz_localize(bars.index.tz)
    return bars.loc[(bars.index >= start) & (bars.index < end)]


@register("source", "synthetic", n_inputs=1,
          description="Seeded random-walk bars for tests and demos (no market data needed)")
def source_synthetic(
    data: Dict[str, Any],
    seed: int = 0,
    freq: str = "15min",
    sigma: Annotated[float, Ge(0)] = 0.001,
    signal_strength: Annotated[float, Ge(0), Le(1)] = 0.0,
) -> pd.DataFrame:
    """
    Geometric random walk on a fixed calendar. ``signal_strength`` plants an AR(1)
    drift in returns, so the battery and importance pages have something to find.
    """
    idx = pd.date_range(pd.Timestamp(data["start_date"]),
                        pd.Timestamp(data["end_date"]) + pd.Timedelta(days=1),
                        freq=freq, inclusive="left", name="time")
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, size=len(idx))
    ret = np.empty_like(eps)
    prev = 0.0
    for i, e in enumerate(eps):
        prev = signal_strength * prev + e
        ret[i] = prev
    close = 1.1 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    wiggle = np.abs(rng.normal(0, sigma / 2, size=len(idx)))
    bars = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * (1 + wiggle),
            "low": np.minimum(open_, close) * (1 - wiggle),
            "close": close,
            "spread": np.full(len(idx), 0.00001),
            "tick_volume": rng.integers(50, 500, size=len(idx)),
        },
        index=idx,
    )
    return bars


# --------------------------------------------------------------------------- targets


@register("target", "daily_vol", n_inputs=1, description="EWM volatility of daily returns (get_daily_vol)")
def target_daily_vol(bars: pd.DataFrame, lookback: Annotated[int, Ge(2)] = 100) -> pd.Series:
    from afml.util.volatility import get_daily_vol

    return get_daily_vol(bars["close"], lookback)


@register("target", "parkinson_vol", n_inputs=1, description="Parkinson high-low volatility")
def target_parkinson_vol(bars: pd.DataFrame, window: Annotated[int, Ge(2)] = 20) -> pd.Series:
    from afml.util.volatility import get_parkinson_vol

    return get_parkinson_vol(bars["high"], bars["low"], window)


# --------------------------------------------------------------------------- strategies


@register("strategy", "bollinger", description="Bollinger-band mean reversion (BollingerStrategy)")
def strategy_bollinger(window: Annotated[int, Ge(2)] = 20, num_std: Annotated[float, Ge(0)] = 2.0):
    from afml.strategies.trading_strategies import BollingerStrategy

    return BollingerStrategy(window=window, std=num_std)


@register("strategy", "ma_crossover", description="Moving-average crossover (MACrossoverStrategy)")
def strategy_ma_crossover(fast_window: Annotated[int, Ge(1)] = 10,
                          slow_window: Annotated[int, Ge(2)] = 30):
    from afml.strategies.trading_strategies import MACrossoverStrategy

    if fast_window >= slow_window:
        raise ValueError("fast_window must be below slow_window")
    return MACrossoverStrategy(fast_window=fast_window, slow_window=slow_window)


# --------------------------------------------------------------------------- estimators


@register("estimator", "random_forest", description="RandomForestClassifier, AFML defaults")
def estimator_random_forest(
    n_estimators: Annotated[int, Ge(1)] = 200,
    max_depth: Optional[int] = None,
    max_features: Any = 1,
    min_weight_fraction_leaf: Annotated[float, Ge(0), Le(0.5)] = 0.05,
    criterion: Literal["entropy", "gini", "log_loss"] = "entropy",
    class_weight: Literal["balanced_subsample", "balanced", "none"] = "balanced_subsample",
    random_state: int = 42,
    n_jobs: int = -1,
):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
        min_weight_fraction_leaf=min_weight_fraction_leaf, criterion=criterion,
        class_weight=None if class_weight == "none" else class_weight,
        random_state=random_state, n_jobs=n_jobs,
    )


@register("estimator", "extra_trees", description="ExtraTreesClassifier")
def estimator_extra_trees(
    n_estimators: Annotated[int, Ge(1)] = 200,
    max_depth: Optional[int] = None,
    max_features: Any = 1,
    min_weight_fraction_leaf: Annotated[float, Ge(0), Le(0.5)] = 0.05,
    class_weight: Literal["balanced_subsample", "balanced", "none"] = "balanced_subsample",
    random_state: int = 42,
    n_jobs: int = -1,
):
    from sklearn.ensemble import ExtraTreesClassifier

    return ExtraTreesClassifier(
        n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        class_weight=None if class_weight == "none" else class_weight,
        random_state=random_state, n_jobs=n_jobs,
    )


@register("estimator", "logistic", description="L2 logistic regression (scaled inputs)")
def estimator_logistic(C: Annotated[float, Ge(0)] = 1.0,
                       class_weight: Literal["balanced", "none"] = "balanced",
                       random_state: int = 42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight=None if class_weight == "none" else class_weight,
                           max_iter=1000, random_state=random_state),
    )


# --------------------------------------------------------------------------- features


@register("feature", "log_returns", n_inputs=1, description="Log returns over several lags")
def feature_log_returns(bars: pd.DataFrame, lags: List[int] = [1, 5, 20]) -> pd.DataFrame:
    logp = np.log(bars["close"])
    return pd.DataFrame({f"ret_{lag}": logp.diff(lag) for lag in lags}, index=bars.index)


@register("feature", "frac_diff_close", n_inputs=1, description="Fixed-width fractionally differentiated log close")
def feature_frac_diff_close(bars: pd.DataFrame, d: Annotated[float, Ge(0), Le(2)] = 0.4,
                            thres: Annotated[float, Ge(0)] = 1e-4) -> pd.DataFrame:
    from afml.features.fracdiff import frac_diff_ffd

    out = frac_diff_ffd(bars[["close"]], d, thres=thres, use_log=True)
    out = out if isinstance(out, pd.DataFrame) else out.to_frame()
    out.columns = [f"ffd_{d:g}"]
    return out.reindex(bars.index)


@register("feature", "fractals", n_inputs=1, description="Fractal analysis features (get_fractal_features)")
def feature_fractals(bars: pd.DataFrame, n: Annotated[int, Ge(1)] = 2,
                     lookback_period: Annotated[int, Ge(2)] = 20,
                     ma_period: Annotated[int, Ge(2)] = 20,
                     vol_lookback: Annotated[int, Ge(2)] = 100) -> pd.DataFrame:
    from afml.features.fractals import get_fractal_features
    from afml.util.volatility import get_daily_vol

    vol = get_daily_vol(bars["close"], vol_lookback).reindex(bars.index).ffill()
    return get_fractal_features(bars, vol, n, lookback_period, ma_period)


@register("feature", "return_dist", n_inputs=1, description="Rolling return-distribution moments")
def feature_return_dist(bars: pd.DataFrame, window: Annotated[int, Ge(3)] = 10) -> pd.DataFrame:
    from afml.features.returns import get_return_dist_features

    out = get_return_dist_features(bars["close"], window)
    return out if isinstance(out, pd.DataFrame) else out.to_frame()


@register("feature", "ma_differences", n_inputs=1, description="Normalized moving-average differences")
def feature_ma_differences(bars: pd.DataFrame, windows: List[int] = [5, 20, 50]) -> pd.DataFrame:
    from afml.features.moving_averages import calculate_ma_differences

    out = calculate_ma_differences(bars["close"], windows, drop=False, verbose=False)
    return out if isinstance(out, pd.DataFrame) else out.to_frame()


@register("feature", "volatility", n_inputs=1, description="Rolling realized and Parkinson volatility")
def feature_volatility(bars: pd.DataFrame, window: Annotated[int, Ge(2)] = 20) -> pd.DataFrame:
    from afml.util.volatility import get_parkinson_vol

    r = np.log(bars["close"]).diff()
    return pd.DataFrame(
        {
            f"rv_{window}": r.rolling(window).std(),
            f"parkinson_{window}": get_parkinson_vol(bars["high"], bars["low"], window),
        },
        index=bars.index,
    )


@register("feature", "time_features", n_inputs=1, description="Session and cyclical time encodings")
def feature_time(bars: pd.DataFrame, timeframe: str = "M15", n_terms: Annotated[int, Ge(1)] = 3,
                 bar_type: Literal["time", "tick", "volume", "dollar"] = "time",
                 forex: bool = True) -> pd.DataFrame:
    from afml.features.trading_session import get_time_features

    return get_time_features(bars, timeframe, n_terms=n_terms, bar_type=bar_type, forex=forex)


@register("feature", "debug_sleep", n_inputs=1, hidden=True,
          description="Test helper: sleeps (optionally in a child process) to exercise cancel")
def feature_debug_sleep(bars: pd.DataFrame, seconds: float = 60.0, spawn_child: bool = True) -> pd.DataFrame:
    child = None
    if spawn_child:
        child = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})"])
        logger.bind(child_pid=child.pid).info("debug_sleep spawned child {}", child.pid)
    time.sleep(seconds)
    if child is not None:
        child.wait()
    return pd.DataFrame({"debug": np.zeros(len(bars))}, index=bars.index)


# =========================================================================== steps
# --------------------------------------------------------------------------- bars


def load_bars(data: Dict[str, Any]) -> pd.DataFrame:
    """Load bars for a resolved DataSpec dict. The runner wraps this with the vault."""
    bars = call("source", data["source"], data, **data.get("source_params", {}))
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise TypeError("bars must have a DatetimeIndex")
    missing = {"open", "high", "low", "close"} - set(bars.columns)
    if missing:
        raise KeyError(f"bars missing columns {sorted(missing)}")
    return bars.sort_index()


def _adf(series: pd.Series) -> Tuple[float, float]:
    from statsmodels.tsa.stattools import adfuller

    s = series.dropna()
    if len(s) < 20:
        return float("nan"), float("nan")
    stat, pval, *_ = adfuller(s, maxlag=1, regression="c", autolag=None)
    return float(stat), float(pval)


def bars_report(bars: pd.DataFrame, d_grid: Optional[List[float]] = None) -> Tuple[Tables, Dict]:
    """Bar count, bars per day, the return distribution, and ADF on raw and FFD closes."""
    from afml.features.fracdiff import frac_diff_ffd

    d_grid = d_grid if d_grid is not None else [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    logc = np.log(bars["close"])
    ret = logc.diff().dropna()

    per_day = bars["close"].groupby(bars.index.normalize()).size()
    per_day = per_day.rename("bars").rename_axis("day").to_frame()

    rows = []
    for d in d_grid:
        if d == 0:
            s = logc
        else:
            s = frac_diff_ffd(bars[["close"]], d, thres=1e-4, use_log=True)
            s = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
        stat, pval = _adf(s)
        corr = float(pd.concat([s, logc], axis=1).dropna().corr().iloc[0, 1]) if d else 1.0
        rows.append({"d": d, "adf_stat": stat, "p_value": pval, "corr_with_log_close": corr})
    adf = pd.DataFrame(rows)

    hist_counts, edges = np.histogram(ret, bins=60)
    ret_hist = pd.DataFrame({"left": edges[:-1], "right": edges[1:], "count": hist_counts})

    stationary = adf[(adf["p_value"] < 0.05) & (adf["d"] > 0)]
    summary = {
        "n_bars": int(len(bars)),
        "start": str(bars.index[0]) if len(bars) else None,
        "end": str(bars.index[-1]) if len(bars) else None,
        "bars_per_day_mean": float(per_day["bars"].mean()) if len(per_day) else 0.0,
        "ret_std": float(ret.std()),
        "ret_skew": float(ret.skew()),
        "ret_kurtosis": float(ret.kurt()),
        "min_d_stationary": float(stationary["d"].min()) if len(stationary) else None,
    }
    return {"bars": bars, "bars_per_day": per_day, "adf_by_d": adf, "return_hist": ret_hist}, summary


# --------------------------------------------------------------------------- labels


@cacheable
def compute_events(bars: pd.DataFrame, labels: Dict[str, Any]) -> pd.DataFrame:
    """
    Triple-barrier events for a resolved LabelSpec dict, with both weight series:

    * ``train_w``: weights for the estimator fit (``train_weight_scheme``)
    * ``w``: return-attribution weights, always the CV scoring weights
    """
    from afml.labeling.triple_barrier import (
        add_vertical_barrier,
        get_event_weights,
        triple_barrier_labels,
    )
    from afml.strategies.signal_processing import get_entries

    if labels.get("from_run"):
        raise NotImplementedError(
            "Primary-to-secondary chaining needs LearnedStrategy.from_pipeline, "
            "which this afml branch does not have yet"
        )
    close = bars["close"]
    target = call("target", labels["target"]["name"], bars, **labels["target"]["params"])
    target = target.reindex(close.index).ffill()
    strategy = call("strategy", labels["strategy"]["name"], **labels["strategy"]["params"])

    threshold = target.mean() if labels.get("filter_events", True) else None
    side, t_events = get_entries(strategy, bars, filter_threshold=threshold)
    vb = add_vertical_barrier(t_events, close, **labels["max_holding_period"])
    events = triple_barrier_labels(
        close, target, t_events,
        vertical_barrier_times=vb,
        side_prediction=side,
        pt_sl=[labels["profit_target"], labels["stop_loss"]],
        min_ret=labels["min_ret"],
        min_pct=0.05,
        vertical_barrier_zero=labels["vertical_barrier_zero"],
        drop=True,
        verbose=False,
    )
    events = get_event_weights(events, close)

    scheme = labels["train_weight_scheme"]
    if scheme == "uniqueness":
        train_w = events["tW"]
    elif scheme == "time_decay":
        from afml.sample_weights.optimized_attribution import get_weights_by_time_decay_optimized

        train_w = get_weights_by_time_decay_optimized(
            events, close.index, last_weight=labels["time_decay"], linear=True,
            av_uniqueness=events["tW"], verbose=False,
        )
    else:
        train_w = pd.Series(1.0, index=events.index)
    events["train_w"] = (train_w / train_w.mean()).astype(float)
    events["w"] = (events["w"] / events["w"].mean()).astype(float)
    return events


def model_role(events: pd.DataFrame) -> str:
    """Derived, never chosen: secondary (meta-labeling) when events carry a side."""
    return "secondary" if "side" in events.columns else "primary"


def labels_report(events: pd.DataFrame, bars: pd.DataFrame) -> Tuple[Tables, Dict]:
    t1 =pd.to_datetime(events["t1"])
    hold = (t1 - events.index.to_series()).dt.total_seconds() / 3600.0
    balance = events["bin"].value_counts().sort_index().rename_axis("bin").rename("count").to_frame()

    # Concurrency: events alive at each bar
    starts = pd.Series(1, index=events.index).groupby(level=0).sum()
    ends = pd.Series(-1, index=t1.values).groupby(level=0).sum()
    conc = pd.concat([starts, ends]).groupby(level=0).sum().sort_index().cumsum()
    conc = conc.reindex(bars.index, method="ffill").fillna(0).rename("concurrent").to_frame()

    weights = events[["train_w", "w"]].rename(
        columns={"train_w": "fit_weight (train_weight_scheme)", "w": "score_weight (events['w'])"}
    )
    summary = {
        "n_events": int(len(events)),
        "role": model_role(events),
        "label_balance": {str(k): int(v) for k, v in balance["count"].items()},
        "holding_hours_median": float(hold.median()) if len(hold) else None,
        "avg_uniqueness": float(events["tW"].mean()) if "tW" in events else None,
        "max_concurrency": int(conc["concurrent"].max()) if len(conc) else 0,
    }
    tables = {
        "events": events,
        "label_balance": balance,
        "holding_hours": hold.rename("hours").to_frame(),
        "concurrency": conc,
        "weights": weights,
    }
    return tables, summary


# --------------------------------------------------------------------------- features


@cacheable
def compute_feature_step(bars: pd.DataFrame, name: str, params: Dict[str, Any]) -> pd.DataFrame:
    """One registered feature function on the bars, cached on bar content and params."""
    out = call("feature", name, bars, **params)
    if isinstance(out, pd.Series):
        out = out.to_frame(out.name or name)
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"feature {name} returned {type(out).__name__}, expected a DataFrame")
    return out.reindex(bars.index)


def build_feature_matrix(bars: pd.DataFrame, features: Dict[str, Any],
                         progress: Progress = _noop_progress) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compose a resolved FeatureSetSpec dict into one matrix.

    Returns ``(X, catalog)`` where the catalog maps each column back to the step that
    produced it (the provenance view on the Feature set page).
    """
    frames, catalog = [], []
    steps = features["steps"]
    for i, step in enumerate(steps):
        progress(i / len(steps), f"feature {step['name']}")
        df = compute_feature_step(bars, step["name"], step["params"])
        df = df.copy()
        prefix = step.get("prefix")
        if prefix:
            df.columns = [f"{prefix}{c}" for c in df.columns]
        df.columns = [str(c) for c in df.columns]
        for c in df.columns:
            catalog.append({"column": c, "step": i, "source": step["name"],
                            "params": repr(step["params"])})
        frames.append(df)
    X = pd.concat(frames, axis=1)
    dupes = X.columns[X.columns.duplicated()].tolist()
    if dupes:
        raise ValueError(f"duplicate feature columns {dupes}; give one of the steps a prefix")
    selected = features.get("selected_columns")
    if selected:
        missing = set(selected) - set(X.columns)
        if missing:
            raise KeyError(f"selected columns not produced by the steps: {sorted(missing)}")
        X = X[selected]
    X = X.replace([np.inf, -np.inf], np.nan).astype(float)
    cat = pd.DataFrame(catalog)
    cat["selected"] = cat["column"].isin(X.columns)
    progress(1.0, "features built")
    return X, cat


def features_report(X: pd.DataFrame, catalog: pd.DataFrame) -> Tuple[Tables, Dict]:
    nan_share = X.isna().mean().rename("nan_share").rename_axis("column").to_frame()
    desc = X.describe().T.rename_axis("column")
    summary = {
        "n_columns": int(X.shape[1]),
        "n_rows": int(X.shape[0]),
        "n_complete_rows": int(X.dropna().shape[0]),
        "max_nan_share": float(nan_share["nan_share"].max()) if len(nan_share) else 0.0,
    }
    return {"features": X, "catalog": catalog, "nan_share": nan_share, "describe": desc}, summary


# --------------------------------------------------------------------------- importance


def _align(X: pd.DataFrame, events: pd.DataFrame):
    idx = X.dropna().index.intersection(events.index)
    if len(idx) < 50:
        raise ValueError(f"only {len(idx)} rows with complete features and labels")
    ev = events.loc[idx]
    return X.loc[idx], ev["bin"].astype(int), pd.to_datetime(ev["t1"]), ev["train_w"], ev["w"]


def _score(clf, X, y, w, scoring: str) -> float:
    from sklearn.metrics import accuracy_score, log_loss

    if scoring == "neg_log_loss":
        prob = clf.predict_proba(X)
        return -log_loss(y, prob, sample_weight=w.values, labels=clf.classes_)
    return accuracy_score(y, clf.predict(X), sample_weight=w.values)


def _fit(est, X, y, w):
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline

    clf = clone(est)
    if isinstance(clf, Pipeline):
        clf.fit(X, y, **{f"{clf.steps[-1][0]}__sample_weight": w.values})
    else:
        clf.fit(X, y, sample_weight=w.values)
    return clf


def run_importance(X: pd.DataFrame, events: pd.DataFrame, importance: Dict[str, Any],
                   seed: int = 42, progress: Progress = _noop_progress) -> Tuple[Tables, Dict, Dict]:
    """
    MDI, MDA and SFI under PurgedKFold (AFML Ch. 8).

    Estimators fit with ``train_w``; every out-of-fold score uses the return-attribution
    weights ``w``. Returns ``(tables, summary, trial)`` where ``trial`` carries the
    baseline per-fold CV scores for the ledger.
    """
    from afml.cross_validation.cross_validation import PurgedKFold

    Xa, y, t1, w_fit, w_score = _align(X, events)
    if importance.get("split_mode"):
        cut = int(len(Xa) * importance["importance_fraction"])
        Xa, y, t1, w_fit, w_score = Xa.iloc[:cut], y.iloc[:cut], t1.iloc[:cut], w_fit.iloc[:cut], w_score.iloc[:cut]

    est_spec = importance["estimator"]
    est = call("estimator", est_spec["name"], **est_spec["params"])
    scoring = importance["scoring"]
    cv = PurgedKFold(n_splits=importance["n_splits"], t1=t1, pct_embargo=importance["pct_embargo"])
    folds = list(cv.split(Xa))
    methods = importance["methods"]
    rng = np.random.default_rng(seed)
    tables: Tables = {}
    n_tasks = len(folds) * (1 if "mda" in methods else 0) + (1 if "mdi" in methods else 0) + \
        (Xa.shape[1] if "sfi" in methods else 0)
    done = 0

    def tick(msg):
        nonlocal done
        done += 1
        progress(done / max(n_tasks, 1), msg)

    # Baseline fold scores (every method's reference, and the ledger trial)
    base_scores, mda_rows = [], []
    for k, (tr, te) in enumerate(folds):
        clf = _fit(est, Xa.iloc[tr], y.iloc[tr], w_fit.iloc[tr])
        s0 = _score(clf, Xa.iloc[te], y.iloc[te], w_score.iloc[te], scoring)
        base_scores.append(s0)
        if "mda" in methods:
            for col in Xa.columns:
                Xp = Xa.iloc[te].copy()
                Xp[col] = rng.permutation(Xp[col].values)
                s1 = _score(clf, Xp, y.iloc[te], w_score.iloc[te], scoring)
                denom = -s1 if scoring == "neg_log_loss" else (1.0 - s1)
                mda_rows.append({"fold": k, "feature": col, "base": s0, "permuted": s1,
                                 "importance": (s0 - s1) / denom if denom else np.nan})
            tick(f"MDA fold {k + 1}/{len(folds)}")

    tables["cv_fold_scores"] = pd.DataFrame({"fold": range(len(base_scores)), "score": base_scores})

    if "mda" in methods:
        mda_folds = pd.DataFrame(mda_rows)
        tables["mda_folds"] = mda_folds
        g = mda_folds.groupby("feature")["importance"]
        tables["mda"] = pd.DataFrame({"mean": g.mean(), "std_err": g.std() * len(folds) ** -0.5}) \
            .sort_values("mean", ascending=False).rename_axis("feature")

    if "mdi" in methods:
        clf = _fit(est, Xa, y, w_fit)
        final = clf.steps[-1][1] if hasattr(clf, "steps") else clf
        if hasattr(final, "estimators_"):
            imp = pd.DataFrame([t.feature_importances_ for t in final.estimators_], columns=Xa.columns)
            imp = imp.replace(0, np.nan)  # max_features=1: a zero means "never sampled"
            mdi = pd.DataFrame({"mean": imp.mean(), "std_err": imp.std() * len(imp) ** -0.5})
            mdi["mean"] /= mdi["mean"].sum()
            tables["mdi"] = mdi.sort_values("mean", ascending=False).rename_axis("feature")
        else:
            logger.warning("MDI skipped: {} is not a tree ensemble", type(final).__name__)
        tick("MDI")

    if "sfi" in methods:
        rows = []
        for col in Xa.columns:
            scores = []
            for tr, te in folds:
                clf = _fit(est, Xa.iloc[tr][[col]], y.iloc[tr], w_fit.iloc[tr])
                scores.append(_score(clf, Xa.iloc[te][[col]], y.iloc[te], w_score.iloc[te], scoring))
            rows.append({"feature": col, "mean": np.mean(scores),
                         "std_err": np.std(scores, ddof=1) * len(scores) ** -0.5})
            tick(f"SFI {col}")
        tables["sfi"] = pd.DataFrame(rows).set_index("feature").sort_values("mean", ascending=False)

    if "mdi" in tables and "mda" in tables:
        ranks = pd.DataFrame({"mdi_rank": tables["mdi"]["mean"].rank(ascending=False),
                              "mda_rank": tables["mda"]["mean"].rank(ascending=False)})
        tables["rank_compare"] = ranks.rename_axis("feature")

    summary = {
        "n_samples": int(len(Xa)),
        "n_features": int(Xa.shape[1]),
        "scoring": scoring,
        "cv_score_mean": float(np.mean(base_scores)),
        "cv_score_std_err": float(np.std(base_scores, ddof=1) / np.sqrt(len(base_scores))),
        "methods": methods,
        "split_mode": bool(importance.get("split_mode")),
        "sample_end": str(Xa.index[-1]),
    }
    trial = {"metric": scoring, "score": summary["cv_score_mean"], "fold_scores": base_scores}
    return tables, summary, trial


# --------------------------------------------------------------------------- model


def run_model(X: pd.DataFrame, events: pd.DataFrame, model: Dict[str, Any],
              seed: int = 42, progress: Progress = _noop_progress) -> Tuple[Tables, Dict, List[Dict]]:
    """
    Purged-CV grid search through ``afml.cross_validation.hyper_fit.clf_hyper_fit``.
    Each grid candidate is one ledger trial with its fold scores.

    The estimator fits with ``train_w``; every fold is scored with the
    return-attribution weights ``w``.
    """
    from afml.cross_validation.hyper_fit import MyPipeline, clf_hyper_fit

    Xa, y, t1, w_fit, w_score = _align(X, events)
    if model.get("window_fraction"):
        lo, hi = model["window_fraction"]
        a, b = int(len(Xa) * lo), int(len(Xa) * hi)
        Xa, y, t1 = Xa.iloc[a:b], y.iloc[a:b], t1.iloc[a:b]
        w_fit, w_score = w_fit.iloc[a:b], w_score.iloc[a:b]
    est_spec = model["estimator"]
    pipe = MyPipeline([("clf", call("estimator", est_spec["name"], **est_spec["params"]))])
    # clf_hyper_fit edits the grid in place; give it a copy
    grid = {k: list(v) for k, v in model["param_grid"].items()}
    if not grid:
        first = next(iter(est_spec["params"]))
        grid = {first: [est_spec["params"][first]]}
    progress(0.05, "grid search")
    best_model, cv_results = clf_hyper_fit(
        Xa, y, t1, pipe, grid,
        n_splits=model["cv_splits"],
        bagging_n_estimators=model["bagging_n_estimators"],
        bagging_max_samples=model["bagging_max_samples"],
        bagging_max_features=model["bagging_max_features"],
        rnd_search_iter=model["rnd_search_iter"],
        pct_embargo=model["pct_embargo"],
        random_state=seed,
        sample_weight_train=w_fit,
        sample_weight_score=w_score,
    )
    progress(0.95, "collecting results")
    res = cv_results["cv_results"]
    split_cols = sorted([c for c in res.columns if c.startswith("split") and c.endswith("_test_score")],
                        key=lambda c: int(c[5:].split("_")[0]))
    trials = []
    for i, row in res.iterrows():
        trials.append({"number": int(i), "metric": cv_results["scoring"],
                       "score": float(row["mean_test_score"]),
                       "params": row["params"],
                       "fold_scores": [float(row[c]) for c in split_cols]})
    table = res[["params", "mean_test_score", "std_test_score", "rank_test_score", *split_cols]].copy()
    table["params"] = table["params"].astype(str)
    summary = {
        "best_score": float(cv_results["best_score"]),
        "best_params": cv_results["best_params"],
        "scoring": cv_results["scoring"],
        "n_candidates": len(trials),
        "n_samples": int(len(Xa)),
        "score_weights_applied": True,
    }
    return {"cv_results": table}, summary, trials


# =========================================================================== pipeline kwargs


def to_pipeline_kwargs(spec) -> Dict[str, Any]:
    """
    Resolve a RunSpec into the keyword shape of the older ``develop_production_model``
    (removed from afml; ``FeatureEngineeringPipeline`` and ``ModelDevelopmentPipeline``
    are the rewiring targets).
    Registered names become live callables here, and only here.
    """
    d = spec.data
    out: Dict[str, Any] = {
        "symbol": d.symbol,
        "train_start": str(d.start_date),
        "train_end": str(d.end_date),
        "data_config": {"account_name": d.account, "bar_type": d.bar_type,
                        "bar_size": d.bar_size, "price": d.price},
    }
    if spec.features is not None:
        feats = spec.features.model_dump(mode="json")

        def feature_func(bars, **_):
            return build_feature_matrix(bars, feats)[0]

        out["feature_config"] = {"func": feature_func, "params": {}}
    if spec.labels is not None and spec.labels.target.name == "daily_vol":
        lab = spec.labels
        out["label_config"] = {
            "target_lookback": lab.target.params["lookback"],
            "strategy": call("strategy", lab.strategy.name, **lab.strategy.params),
            "profit_target": lab.profit_target,
            "stop_loss": lab.stop_loss,
            "max_holding_period": lab.max_holding_period,
            "min_ret": lab.min_ret,
            "vertical_barrier_zero": lab.vertical_barrier_zero,
        }
        out["sample_weight_params"] = {"attribution": "uniqueness" if lab.train_weight_scheme == "uniqueness" else None,
                                       "decay_factor": lab.time_decay}
    if spec.model is not None:
        m = spec.model
        out["model_params"] = {
            "param_grid": m.param_grid, "cv_splits": m.cv_splits, "pct_embargo": m.pct_embargo,
            "bagging_n_estimators": m.bagging_n_estimators, "bagging_max_samples": m.bagging_max_samples,
            "bagging_max_features": m.bagging_max_features, "rnd_search_iter": m.rnd_search_iter,
        }
    return out


def capabilities() -> Dict[str, Any]:
    """What this afml branch can do, so pages hide controls that have no backend."""
    return {
        "importance_methods": ["mdi", "mda", "sfi"],
        "missing_importance_methods": ["clustered_mdi", "clustered_mda", "pca", "orthogonal", "fingerprint"],
        "screening": False,  # afml.screening.screen_features exists; not wired yet
        "optuna": False,     # optuna not installed; model stage uses grid search
        "learned_strategy": False,
        "family_stats_source": "afml.studio.stats (local)",
    }


# Subpackages the steps above call into. prewarm_numba_in_package imports every module
# it walks, and walking all of afml would import afml.studio.__main__, the test
# suites, and modules such as afml.cache.startup_script that reconfigure logging at
# import time.
PREWARM_PACKAGES = ("afml.data_structures", "afml.features", "afml.filters", "afml.labeling",
                    "afml.sample_weights", "afml.sampling", "afml.util", "afml.cross_validation")


def prewarm() -> None:
    """Compile Numba kernels once per job (Windows spawns pay the import cost)."""
    if os.environ.get("AFML_STUDIO_SKIP_PREWARM"):
        return
    from afml.numba_warmup import prewarm_numba_in_package

    for pkg in PREWARM_PACKAGES:
        try:
            prewarm_numba_in_package(pkg, verbose=False)
        except BaseException as exc:  # warmup is an optimization, never a failure
            logger.debug("numba prewarm of {} skipped: {}", pkg, exc)
