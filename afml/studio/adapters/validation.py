"""
The Phase 3 validation battery: tests 1 to 6 against a frozen model run.

Validation can only reject. Each test answers one question and is judged against the
family's pre-registered threshold; every test runs and is reported even after one
fails, because a partial report invites rerunning only the test that failed.

Everything here works at the event level (one row per triple-barrier event). Strategy
returns come from out-of-fold probabilities through one frozen sizing rule: the
prop-firm sizer (``afml.prop_firms.PropFirmAwareSizer``) at a fresh account, without
concurrency averaging, applied to each event's raw return. For meta-labels the side
comes from the primary strategy; for primary labels it comes from the probability.

Like ``pipelines.py``, this module is pure: plain dicts and frames in, tables and a
summary out. The runner owns the ledger and the run folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from ..registry import call
from .pipelines import Progress, Tables, _block_length_in_events, _noop_progress

TEST_NAMES = {
    1: "Signal permutation",
    2: "CPCV path distribution",
    3: "Deflated Sharpe",
    4: "PBO",
    5: "Walk-forward degradation",
    6: "Sequence risk",
}
CPCV_FOLDS, CPCV_TEST_FOLDS = 8, 2  # 28 splits, 7 backtest paths
WALKFORWARD_WINDOWS = 5


# =========================================================================== inputs


def load_trial_oof(oof_dir: Path, index: pd.Index) -> pd.DataFrame:
    """Every complete trial's OOF probabilities as an events x trials frame."""
    cols = {}
    for f in sorted(Path(oof_dir).glob("trial_*.parquet")):
        cols[int(f.stem.split("_")[1])] = pd.read_parquet(f)["p"].astype(float)
    if not cols:
        raise FileNotFoundError(f"no per-trial OOF probabilities in {oof_dir}")
    return pd.DataFrame(cols).reindex(index)


def _rule_set(name: str):
    from afml.prop_firms import PRESETS

    return PRESETS[name]


def _is_meta(events: pd.DataFrame) -> bool:
    return "side" in events.columns


def raw_event_returns(events: pd.DataFrame) -> pd.Series:
    """Each event's price return from entry to exit, before any side is applied."""
    ret = events["ret"].astype(float)
    return ret * events["side"].astype(float) if _is_meta(events) else ret


def _unit_sizes(probs: np.ndarray, events: pd.DataFrame, rule_set) -> np.ndarray:
    """
    The frozen sizing rule as a function of probability alone: the position a long
    side would get at each probability. It is elementwise (no concurrency averaging),
    so a permuted probability vector maps to permuted sizes exactly.
    """
    from afml.prop_firms import Phase, PropFirmAccountState, PropFirmAwareSizer, WParamCalibrator

    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
    if len(p) != len(events):
        raise ValueError(f"{len(p)} probabilities for {len(events)} events")
    idx = events.index
    ev = events[["t1"]]
    meta = _is_meta(events)
    prob = pd.Series(p if meta else np.maximum(p, 1 - p), index=idx)
    state = PropFirmAccountState(100_000.0, Phase.CHALLENGE_PHASE_1, rule_set)
    out = PropFirmAwareSizer(WParamCalibrator()).size(
        events=ev, prob=prob, pred=pd.Series(1.0, index=idx), state=state, average_active=False)
    size = out["final_size"].to_numpy(dtype=float)
    return size if meta else size * np.sign(p - 0.5)


def positions(probs: pd.Series, events: pd.DataFrame, rule_set) -> pd.Series:
    """Signed position per event from its OOF probability."""
    ev = events.loc[probs.index]
    size = _unit_sizes(probs.to_numpy(), ev, rule_set)
    if _is_meta(ev):
        size = size * ev["side"].to_numpy(dtype=float)
    return pd.Series(size, index=probs.index)


def strategy_returns(probs: pd.Series, events: pd.DataFrame, rule_set) -> pd.Series:
    probs = probs.dropna()
    return positions(probs, events, rule_set) * raw_event_returns(events.loc[probs.index])


def _sharpe(r) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    sd = r.std(ddof=1) if r.size > 1 else 0.0
    return float(r.mean() / sd) if sd > 0 else 0.0


def _periods_per_year(index: pd.Index) -> float:
    from afml.backtest_statistics.performance_analysis import get_annualization_factors

    return float(get_annualization_factors(data_index=pd.DatetimeIndex(index))[1])


def frozen_estimator(model_cfg: Dict[str, Any], best_params: Dict[str, Any], events: pd.DataFrame,
                     data_index: pd.DatetimeIndex):
    """The frozen trial's estimator: the study's best params, including its fit weights."""
    from afml.cross_validation.optuna_hyper_fit import FinancialModelSuggester

    est = model_cfg["estimator"]
    params = {**est["params"], "n_jobs": 1} if "n_jobs" in est["params"] else dict(est["params"])
    base = call("estimator", est["name"], **params)
    return FinancialModelSuggester.apply_from_params(best_params, base, events, data_index)


def _score(model, X, y, w, metric: str) -> float:
    from sklearn.metrics import f1_score, log_loss

    if metric == "neg_log_loss":
        return float(-log_loss(y, model.predict_proba(X), sample_weight=w, labels=model.base_estimator.classes_))
    return float(f1_score(y, model.predict(X), sample_weight=w))


# =========================================================================== tests


def test_signal_permutation(probs: pd.Series, events: pd.DataFrame, rule_set, n_perm: int,
                            seed: int) -> Tuple[Tables, Dict]:
    """
    Test 1: does the model's trade selection beat a misaligned copy of itself?

    The frozen trial's OOF probabilities are circularly shifted against the events
    (keeping their serial dependence) and the strategy Sharpe recomputed.
    """
    from afml.screening.permutation import circular_shift_indices, permutation_pvalue

    probs = probs.dropna()
    ev = events.loc[probs.index]
    raw = raw_event_returns(ev).to_numpy()
    side = ev["side"].to_numpy(dtype=float) if _is_meta(ev) else 1.0
    unit = _unit_sizes(probs.to_numpy(), ev, rule_set)
    observed = _sharpe(side * unit * raw)
    idx = circular_shift_indices(len(unit), n_perm=n_perm, rng=np.random.default_rng(seed),
                                 min_shift=max(1, len(unit) // 20))
    null = np.array([_sharpe(side * unit[row] * raw) for row in idx])
    p = permutation_pvalue(observed, null)
    ann = np.sqrt(_periods_per_year(ev.index))
    tables = {"t1_null": pd.DataFrame({"sharpe": null * ann})}
    return tables, {"observed_sharpe": observed * ann, "p_value": float(p), "n_perm": int(n_perm),
                    "null_median": float(np.median(null) * ann)}


def run_cpcv(X: pd.DataFrame, y: pd.Series, events: pd.DataFrame, bars: pd.DataFrame, estimator,
             rule_set, pct_embargo: float, seed: int, account: Optional[Dict[str, Any]] = None):
    """CPCVDynamicBacktest on the frozen estimator (shared by tests 2 and 6)."""
    from afml.backtester.cpcv_dynamic_backtest import BacktestConfig, CPCVDynamicBacktest
    from afml.cross_validation.combinatorial import CombinatorialPurgedCV
    from afml.prop_firms import PropFirmAwareSizer, WParamCalibrator

    cv = CombinatorialPurgedCV(n_folds=CPCV_FOLDS, n_test_folds=CPCV_TEST_FOLDS, t1=events["t1"],
                               pct_embargo=pct_embargo)
    sides = events["side"] if _is_meta(events) else pd.Series(1, index=events.index)
    bt = CPCVDynamicBacktest(cv, estimator, PropFirmAwareSizer(WParamCalibrator()),
                             BacktestConfig(rule_set=rule_set, n_jobs=1, random_state=seed, **(account or {})),
                             bars, sides)
    bt.run(X, y, events)
    return bt


def test_cpcv_paths(bt) -> Tuple[Tables, Dict]:
    """Test 2: how spread out is performance across the CPCV backtest paths?"""
    rows, fan = [], []
    for r in bt.results_:
        rows.append({"path": r.path_id, "sharpe": r.sharpe, "max_drawdown": r.max_drawdown,
                     "final_equity": r.final_equity, "breach": r.breach or "", "n_trades": r.n_trades,
                     "hit_target": r.hit_target})
        eq = r.equity_curve / r.equity_curve.iloc[0]
        fan.append(pd.DataFrame({"path": r.path_id, "time": eq.index, "equity": eq.to_numpy()}))
    paths = pd.DataFrame(rows).set_index("path")
    sharpe = paths["sharpe"]
    return ({"t2_paths": paths, "t2_equity": pd.concat(fan, ignore_index=True)},
            {"median_sharpe": float(sharpe.median()), "positive_share": float((sharpe > 0).mean()),
             "n_paths": int(len(paths)), "p5_sharpe": float(sharpe.quantile(0.05)),
             "p95_sharpe": float(sharpe.quantile(0.95)), "breaches": int((paths["breach"] != "").sum()),
             "long_only": not bool((bt.primary_sides < 0).any())})


def test_deflated_sharpe(trial_returns: pd.DataFrame, best: int, n_eff: float, n_raw: int
                         ) -> Tuple[Tables, Dict]:
    """
    Test 3: does the best Sharpe survive the number of trials behind it?
    DSR through the units adapter, with the family's effective trial count.
    """
    from afml.backtest_statistics import deflated_sharpe_from_metrics
    from afml.backtest_statistics.performance_analysis import calculate_performance_metrics

    r = trial_returns[best].dropna()
    ppy = _periods_per_year(r.index)
    metrics = calculate_performance_metrics(r, r.index)
    ann_sharpes = trial_returns.apply(lambda c: _sharpe(c.dropna())) * np.sqrt(ppy)
    sr_std = float(ann_sharpes.std(ddof=1)) if len(ann_sharpes) > 1 else 0.0
    dsr = deflated_sharpe_from_metrics(metrics, ppy, len(r), sr_std, n_eff)
    n_max = int(max(n_raw, 2 * n_eff, 10))
    curve = pd.DataFrame({"n_trials": np.arange(1, n_max + 1)})
    curve["expected_max_sharpe"] = [
        deflated_sharpe_from_metrics(metrics, ppy, len(r), sr_std, n, benchmark_out=True) * np.sqrt(ppy)
        for n in curve["n_trials"]]
    return ({"t3_expected_max": curve, "t3_trial_sharpes": ann_sharpes.rename("sharpe").to_frame()},
            {"dsr": float(dsr), "observed_sharpe": float(metrics["sharpe_ratio"]), "n_eff": float(n_eff),
             "n_raw": int(n_raw), "sr_std": sr_std, "n_returns": int(len(r)), "periods_per_year": ppy,
             "skewness": float(metrics["skewness"]), "excess_kurtosis": float(metrics["kurtosis"])})


def test_pbo(trial_returns: pd.DataFrame, n_folds: int = 8) -> Tuple[Tables, Dict]:
    """
    Test 4: how often does the in-sample winner of the family fall below the
    out-of-sample median? CSCV across the trials' OOF strategy returns.
    """
    from afml.cross_validation.combinatorial import CombinatorialPurgedCV
    from afml.cross_validation.pbo import compute_pbo

    R = trial_returns.dropna(how="all").fillna(0.0).sort_index()
    t1 = pd.Series(R.index, index=R.index)
    res = compute_pbo(R, t1=t1, n_folds=n_folds)
    # The in-sample winner's IS and OOS Sharpe per split, for the scatter
    cv = CombinatorialPurgedCV(n_folds=n_folds, n_test_folds=n_folds // 2, t1=t1, pct_embargo=0.0)
    pts = []
    for train, test_list in cv.split(R):
        test = np.concatenate(test_list)
        is_s = R.iloc[train].apply(_sharpe)
        oos_s = R.iloc[test].apply(_sharpe)
        best = is_s.idxmax()
        pts.append({"is_sharpe": is_s[best], "oos_sharpe": oos_s[best],
                    "oos_median": float(oos_s.median())})
    ann = np.sqrt(_periods_per_year(R.index))
    scatter = pd.DataFrame(pts) * ann
    return ({"t4_logits": pd.DataFrame({"logit": res["logit_sr"]}),
             "t4_ranks": pd.DataFrame({"oos_rank": res["oos_ranks"]}), "t4_scatter": scatter},
            {"pbo": float(res["pbo"]), "n_splits": int(res["n_splits"]), "n_trials": int(R.shape[1])})


def test_walkforward(X: pd.DataFrame, y: pd.Series, events: pd.DataFrame, make_estimator: Callable,
                     metric: str, pct_embargo: float, n_boot: int, seed: int) -> Tuple[Tables, Dict]:
    """
    Test 5: does performance fall from in-sample to out-of-sample across anchored
    windows? In-sample is the purged CV score inside each training window (what the
    search saw); out-of-sample is the model fit on that window, scored on the next.
    """
    from afml.backtest_statistics.walkforward import walkforward_degradation
    from afml.cross_validation.cross_validation import PurgedKFold, PurgedWalkForwardCV
    from sklearn.base import clone

    w = events["w"]
    wf = PurgedWalkForwardCV(n_splits=WALKFORWARD_WINDOWS, t1=events["t1"], pct_embargo=pct_embargo,
                             expanding_window=True)
    rows = []
    for k, (train, test) in enumerate(wf.split(X)):
        Xtr, ytr, wtr = X.iloc[train], y.iloc[train], w.iloc[train]
        inner = PurgedKFold(n_splits=3, t1=events["t1"].iloc[train], pct_embargo=pct_embargo)
        is_scores = []
        for a, b in inner.split(Xtr):
            m = clone(make_estimator()).fit(Xtr.iloc[a], ytr.iloc[a])
            is_scores.append(_score(m, Xtr.iloc[b], ytr.iloc[b], wtr.iloc[b].to_numpy(), metric))
        m = clone(make_estimator()).fit(Xtr, ytr)
        oos = _score(m, X.iloc[test], y.iloc[test], w.iloc[test].to_numpy(), metric)
        rows.append({"window": k, "train_end": X.index[train[-1]], "test_start": X.index[test[0]],
                     "is_score": float(np.mean(is_scores)), "oos_score": oos})
    table = pd.DataFrame(rows).set_index("window")
    d = walkforward_degradation(table["is_score"], table["oos_score"], n_boot=n_boot, seed=seed)
    return {"t5_windows": table}, {**{k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
                                      for k, v in d.to_dict().items()}, "metric": metric}


def test_sequence_risk(bt, events: pd.DataFrame, rule_set, n_boot: int, seed: int) -> Tuple[Tables, Dict]:
    """
    Test 6: how bad is an unlucky ordering of the same trades? Block bootstrap of the
    median CPCV path's account returns against the rule set's loss limits.
    """
    from afml.backtest_statistics.bootstrap import bootstrap_drawdown_distribution, breach_probability

    by_sharpe = sorted(bt.results_, key=lambda r: r.sharpe)
    median_path = by_sharpe[len(by_sharpe) // 2]
    r = median_path.returns
    r = r[r != 0.0]  # events with no position change carry no ordering risk
    block = _block_length_in_events(events)
    days = max(pd.DatetimeIndex(events.index).normalize().nunique(), 1)
    per_day = max(int(round(len(r) / days)), 1)
    basis = "initial" if rule_set.max_loss_mode == "static" else "peak"
    dist = bootstrap_drawdown_distribution(r, mean_block=block, n_boot=n_boot, seed=seed)
    bp = breach_probability(r, mean_block=block, max_drawdown_limit=rule_set.overall_loss_limit_pct,
                            daily_loss_limit=rule_set.daily_loss_limit_pct, trades_per_day=per_day,
                            n_boot=n_boot, seed=seed, drawdown_basis=basis)
    return ({"t6_drawdowns": dist.to_frame()},
            {**{k: float(v) for k, v in bp.items()}, "path": int(median_path.path_id), "mean_block": block,
             "trades_per_day": per_day, "max_drawdown_limit": rule_set.overall_loss_limit_pct,
             "daily_loss_limit": rule_set.daily_loss_limit_pct, "drawdown_basis": basis})


# =========================================================================== battery


def _verdict(test: int, value: Optional[float], threshold: str, passed: Optional[bool],
             note: str = "") -> Dict[str, Any]:
    return {"test": test, "name": TEST_NAMES.get(test, "Process permutation"), "value": value,
            "threshold": threshold, "passed": passed, "note": note}


def run_battery(X: pd.DataFrame, events: pd.DataFrame, bars: pd.DataFrame, model_cfg: Dict[str, Any],
                source: Dict[str, Any], prereg: Dict[str, Any], family: Dict[str, Any], tests: List[int],
                seed: int = 42, progress: Progress = _noop_progress) -> Tuple[Tables, Dict]:
    """
    Run tests 1 to 6 in order. Every test runs and is reported; an error in one test
    is recorded as a failure of that test, never a reason to skip the rest.

    :param source: the frozen model run's summary (``best_params``, ``best_trial``,
        ``oof_dir``, ``scoring``).
    :param prereg: the pre-registration payload.
    :param family: the family statistics (``n_eigen``, ``n_raw``).
    """
    th = prereg["thresholds"]
    rule_set = _rule_set(prereg["rule_set"])
    n_boot = int(prereg.get("n_boot", 5000))

    # The frozen run's sample: complete features and labels, its window fraction
    idx = X.dropna().index.intersection(events.index)
    Xa, ev = X.loc[idx], events.loc[idx]
    if model_cfg.get("window_fraction"):
        lo, hi = model_cfg["window_fraction"]
        a, b = int(len(Xa) * lo), int(len(Xa) * hi)
        Xa, ev = Xa.iloc[a:b], ev.iloc[a:b]
    y = ev["bin"].astype(int)
    oof = load_trial_oof(Path(source["oof_dir"]), Xa.index)
    best = int(source["best_trial"])
    if best not in oof.columns:
        raise KeyError(f"no OOF probabilities saved for the frozen trial {best}")
    trial_returns = pd.DataFrame({n: strategy_returns(oof[n], ev, rule_set) for n in oof.columns})

    def make_estimator():
        return frozen_estimator(model_cfg, source["best_params"], ev, bars.index)

    tables: Tables = {}
    summary: Dict[str, Any] = {"tests": {}, "rule_set": prereg["rule_set"], "n_events": int(len(ev)),
                               "frozen_trial": best, "n_trials_with_oof": int(oof.shape[1])}
    verdicts: List[Dict[str, Any]] = []
    bt, bt_error = None, ""
    steps = len(tests)

    def run(test: int, fn: Callable[[], Tuple[Tables, Dict]]) -> Optional[Dict]:
        progress(len(summary["tests"]) / steps, f"test {test}: {TEST_NAMES[test]}")
        try:
            t, s = fn()
        except Exception as exc:  # recorded as a failure, never skipped
            logger.exception("validation test {} failed to run", test)
            summary["tests"][test] = {"error": f"{type(exc).__name__}: {exc}"}
            verdicts.append(_verdict(test, None, "", False, f"error: {exc}"))
            return None
        tables.update(t)
        summary["tests"][test] = s
        return s

    if 1 in tests:
        s = run(1, lambda: test_signal_permutation(oof[best], ev, rule_set, int(prereg["n_perm_signal"]), seed))
        if s:
            verdicts.append(_verdict(1, s["p_value"], f"p <= {th['permutation_p_max']}",
                                     s["p_value"] <= th["permutation_p_max"]))
    if 2 in tests or 6 in tests:
        try:
            progress(len(summary["tests"]) / steps, "CPCV backtest")
            bt = run_cpcv(Xa, y, ev, bars, make_estimator(), rule_set, float(model_cfg["pct_embargo"]), seed,
                          prereg.get("account"))
        except Exception as exc:
            logger.exception("CPCV backtest failed")
            bt_error = f"{type(exc).__name__}: {exc}"
        if 2 in tests:
            if bt is None:
                summary["tests"][2] = {"error": bt_error}
                verdicts.append(_verdict(2, None, "", False, f"error: {bt_error}"))
            else:
                s = run(2, lambda: test_cpcv_paths(bt))
                if s:
                    ok = s["median_sharpe"] > th["cpcv_median_sharpe_min"] and \
                        s["positive_share"] >= th["cpcv_positive_share_min"]
                    verdicts.append(_verdict(
                        2, s["median_sharpe"],
                        f"median Sharpe > {th['cpcv_median_sharpe_min']} and >= "
                        f"{th['cpcv_positive_share_min']:.0%} of paths positive", ok,
                        f"{s['positive_share']:.0%} of {s['n_paths']} paths positive"
                        + ("; long only (primary labels)" if s["long_only"] else "")))
    if 3 in tests:
        s = run(3, lambda: test_deflated_sharpe(trial_returns, best, float(family["n_eigen"] or 1),
                                                int(family["n_raw"])))
        if s:
            verdicts.append(_verdict(3, s["dsr"], f"DSR >= {th['dsr_min']}", s["dsr"] >= th["dsr_min"],
                                     f"N eff {s['n_eff']:.1f} of {s['n_raw']} trials"))
    if 4 in tests:
        s = run(4, lambda: test_pbo(trial_returns))
        if s:
            verdicts.append(_verdict(4, s["pbo"], f"PBO <= {th['pbo_max']}", s["pbo"] <= th["pbo_max"],
                                     f"{s['n_trials']} trials, {s['n_splits']} splits"))
    if 5 in tests:
        s = run(5, lambda: test_walkforward(Xa, y, ev, make_estimator, source["scoring"],
                                            float(model_cfg["pct_embargo"]), n_boot, seed))
        if s:
            verdicts.append(_verdict(5, s["mean_degradation"], "degradation interval includes 0",
                                     not s["overfits"],
                                     f"interval [{s['ci_low']:.4f}, {s['ci_high']:.4f}]"))
    if 6 in tests:
        if bt is None:
            summary["tests"][6] = {"error": bt_error}
            verdicts.append(_verdict(6, None, "", False, f"error: {bt_error}"))
        else:
            s = run(6, lambda: test_sequence_risk(bt, ev, rule_set, n_boot, seed))
            if s:
                verdicts.append(_verdict(6, s["p_any_breach"], f"breach probability <= {th['breach_probability_max']}",
                                         s["p_any_breach"] <= th["breach_probability_max"]))

    verdict = pd.DataFrame(verdicts).set_index("test").sort_index()
    tables["verdict"] = verdict
    summary["passed"] = bool(verdict["passed"].all()) if len(verdict) else False
    summary["n_failed"] = int((~verdict["passed"].astype(bool)).sum())
    progress(1.0, "battery done")
    return tables, summary


# =========================================================================== test 7


def development_process(labels_cfg: Dict[str, Any], features_cfg: Dict[str, Any], model_cfg: Dict[str, Any],
                        rule_set, seed: int) -> Callable[[pd.DataFrame], float]:
    """
    The whole development process as a function of bars: labels, features, the Optuna
    search with the frozen trial budget, then the selected trial's OOF strategy Sharpe
    under the frozen sizing rule. Seeds are fixed, so the same bars give the same value.
    """
    import shutil
    import tempfile

    from . import pipelines as P

    def develop(bars: pd.DataFrame) -> float:
        events = P.compute_events(bars, labels_cfg)
        X, _ = P.build_feature_matrix(bars, features_cfg)
        tmp = Path(tempfile.mkdtemp(prefix="afml_procperm_"))
        try:
            _, summary, _ = P.run_optuna(X, events, model_cfg, bars.index, seed=seed, storage_dir=tmp)
            probs = pd.read_parquet(tmp / "oof" / f"trial_{summary['best_trial']:04d}.parquet")["p"]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        r = strategy_returns(probs.astype(float), events, rule_set)
        return _sharpe(r) * np.sqrt(_periods_per_year(r.index))

    return develop


def run_process_permutation(bars: pd.DataFrame, cfg: Dict[str, Any], prereg: Dict[str, Any], seed: int = 42,
                            progress: Progress = _noop_progress,
                            develop_fn: Optional[Callable[[pd.DataFrame], float]] = None
                            ) -> Tuple[Tables, Dict]:
    """
    Test 7: does the whole development process manufacture edge from noise?

    ``permute_development_process`` on ``bar_permutation`` bars with the frozen trial
    budget. ``assert_cache_isolation`` runs first and the job refuses to start when
    it fails: a cache key that ignores bar content returns the real result for every
    permutation and makes a broken process look sound.
    """
    from afml.cross_validation.process_permutation import assert_cache_isolation, permute_development_process

    th = prereg["thresholds"]
    n_perm = int(prereg["n_perm_process"])
    develop = develop_fn or development_process(cfg["labels"], cfg["features"], cfg["model"],
                                                _rule_set(prereg["rule_set"]), seed)
    calls = {"n": 0}
    total = n_perm + 4  # isolation probes (3), the real bars, then the replicates

    def tracked(b: pd.DataFrame) -> float:
        calls["n"] += 1
        progress(min(calls["n"] / total, 0.99), f"development process {calls['n']}/{total}")
        return develop(b)

    progress(0.0, "checking cache isolation")
    assert_cache_isolation(tracked, bars, seed=seed)  # raises: the job refuses to start
    res = permute_development_process(tracked, bars, n_perm=n_perm, seed=seed,
                                      trial_budget=int(cfg["model"].get("n_trials", 0)),
                                      check_cache_isolation=False)
    lo, hi, _ = res.median_ci()
    center_above_zero = res.selection_effect_exceeds(0.0)
    p_ok = res.p_value <= th["permutation_p_max"]
    summary = {
        "observed_sharpe": float(res.observed), "spurious_center": float(res.spurious_center),
        "center_ci_low": float(lo), "center_ci_high": float(hi), "p_value": float(res.p_value),
        "n_perm": n_perm, "n_failed": len(res.failures), "trial_budget": res.trial_budget,
        "robust_z": float(res.robust_z), "passed": bool(p_ok and not center_above_zero),
        "cache_isolation": "passed",
    }
    note = (f"null median {res.spurious_center:.3f} (95% CI {lo:.3f} to {hi:.3f})"
            + ("; credibly above zero" if center_above_zero else ""))
    verdict = pd.DataFrame([_verdict(7, float(res.p_value),
                                     f"p <= {th['permutation_p_max']} and null median not credibly above 0",
                                     summary["passed"], note)]).set_index("test")
    progress(1.0, "process permutation done")
    return {"t7_null": pd.DataFrame({"sharpe": res.null_values}), "verdict": verdict}, summary
