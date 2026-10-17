"""
Sequential vs. standard bootstrap comparison toolkit.

This module makes the difference between standard (uniform, with-replacement)
bootstrapping and López de Prado's *sequential* bootstrap (Advances in Financial
Machine Learning, Ch. 4) explicit and visual, at two complementary levels:

1. **Sampling mechanism** (``compare_sampling``) — model-free and cheap.  Draws
   repeated bootstrap samples with each method and measures (a) the *average
   uniqueness* of the drawn set and (b) how often each observation is selected.
   On data with overlapping labels the sequential bootstrap achieves markedly
   higher average uniqueness because it down-weights observations that overlap
   already-selected ones.  This is the root cause of every downstream
   difference.

2. **Predictive effect** (``compare_predictions``) — fits a bagging ensemble with
   each sampler inside purged, embargoed cross-validation and compares the
   out-of-fold probabilities (Brier score, ECE, log-loss, PWA, accuracy) plus
   reliability curves.  The OOF predictions are produced by the sequential-aware
   ``CalibratorCV``, so the sequential sampler is genuinely exercised on every
   fold rather than silently collapsing to standard bagging.

The orchestrator ``compare_bootstrap_methods`` runs both levels and returns a
``BootstrapComparison`` result with ``.summary()`` and ``.plot()`` helpers.

Example
-------
>>> from sklearn.tree import DecisionTreeClassifier
>>> from afml.ensemble.bootstrap_comparison import compare_bootstrap_methods
>>> result = compare_bootstrap_methods(
...     base_estimator=DecisionTreeClassifier(max_depth=4),
...     X=features, y=events["bin"],
...     samples_info_sets=events["t1"], price_bars_index=bars.index,
...     n_estimators=100, n_splits=5, pct_embargo=0.01,
...     sample_weight=events["tW"],
... )
>>> print(result.summary())
>>> result.plot(save_path="bootstrap_comparison.png")

For a model already trained with ``ModelDevelopmentPipeline`` use
``compare_from_pipeline(pipeline)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline

from ..sampling.bootstrapping import (
    get_active_indices,
    get_ind_mat_average_uniqueness,
    get_ind_matrix,
    seq_bootstrap,
)
from ..util.pipelines import MyPipeline
from .sb_bagging import SequentiallyBootstrappedBaggingClassifier

STD_COLOR = "#d62728"  # red — standard bootstrap
SEQ_COLOR = "#1f77b4"  # blue — sequential bootstrap
_MAX_INT = np.iinfo(np.int32).max


# ───────────────────────────── result containers ─────────────────────────────


@dataclass
class SamplingComparison:
    """Model-free sampling diagnostics for both bootstrap methods."""

    uniqueness: pd.DataFrame  # n_repeats x {"standard", "sequential"}
    selection_counts: pd.DataFrame  # n_obs x {"standard", "sequential"}
    sample_length: int
    n_repeats: int

    def summary(self) -> pd.DataFrame:
        """Mean/std of average uniqueness and the uniqueness gain ratio."""
        desc = self.uniqueness.agg(["mean", "std"]).T
        desc["uniqueness_gain"] = (
            self.uniqueness["sequential"].mean() / self.uniqueness["standard"].mean()
        )
        return desc


@dataclass
class PredictionComparison:
    """Out-of-fold predictive diagnostics for both bootstrap methods."""

    metrics: pd.DataFrame  # rows=metric, cols={"standard", "sequential"}
    oof_probs: dict  # {"standard": ndarray, "sequential": ndarray}
    y_true: np.ndarray
    valid_mask: np.ndarray

    def summary(self) -> pd.DataFrame:
        return self.metrics


@dataclass
class BootstrapComparison:
    """Top-level result combining sampling and predictive comparisons."""

    avg_uniqueness: float
    n_estimators: int
    max_samples_standard: float
    sampling: SamplingComparison | None = None
    predictions: PredictionComparison | None = None
    _meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "Sequential vs. standard bootstrap comparison",
            "=" * 46,
            f"Dataset average uniqueness : {self.avg_uniqueness:.4f}",
            f"Ensemble size (n_estimators): {self.n_estimators}",
            f"Standard max_samples        : {self.max_samples_standard}",
        ]
        if self.sampling is not None:
            s = self.sampling.summary()
            gain = s["uniqueness_gain"].iloc[0]
            lines += [
                "",
                "Sampling (average uniqueness of the drawn set):",
                f"  standard   : {s.loc['standard', 'mean']:.4f} ± {s.loc['standard', 'std']:.4f}",
                f"  sequential : {s.loc['sequential', 'mean']:.4f} "
                f"± {s.loc['sequential', 'std']:.4f}",
                f"  gain (seq/std): {gain:.2f}x",
            ]
        if self.predictions is not None:
            m = self.predictions.metrics
            lines += ["", "Out-of-fold metrics (standard | sequential):"]
            for metric in m.index:
                lines.append(
                    f"  {metric:<12}: {m.loc[metric, 'standard']:.4f} | "
                    f"{m.loc[metric, 'sequential']:.4f}"
                )
        return "\n".join(lines)

    def plot(self, **kwargs):
        return plot_bootstrap_comparison(self, **kwargs)


# ───────────────────────────── sampling comparison ───────────────────────────


def _standard_draw(n_obs: int, sample_length: int, rng: np.random.RandomState) -> np.ndarray:
    """Standard bootstrap: sample observation ids uniformly with replacement."""
    return rng.randint(0, n_obs, size=sample_length)


def compare_sampling(
    samples_info_sets: pd.Series,
    price_bars_index,
    sample_length: int | None = None,
    n_repeats: int = 200,
    random_state: int | None = None,
    ind_mat: np.ndarray | None = None,
    active_indices: dict | None = None,
) -> SamplingComparison:
    """
    Compare the sampling behaviour of standard vs. sequential bootstrap.

    For ``n_repeats`` independent draws of length ``sample_length`` (default: the
    number of observations) under each method, this records the average
    uniqueness of the drawn set and accumulates how often each observation is
    selected.

    Parameters
    ----------
    samples_info_sets : pd.Series
        Triple-barrier label spans (``t1``): index is the event start time,
        value is the event end time.
    price_bars_index : pd.DatetimeIndex or array-like
        Sorted bar timestamps used to build the indicator matrix.
    sample_length : int, optional
        Draws per bootstrap sample.  Defaults to the number of observations.
    n_repeats : int, default 200
        Number of independent bootstrap samples per method.
    random_state : int, optional
        Seed for reproducibility.
    ind_mat, active_indices : optional
        Precomputed indicator matrix / active-index map (avoids recomputation
        when the caller already has them).

    Returns
    -------
    SamplingComparison
    """
    if ind_mat is None:
        ind_mat = get_ind_matrix(samples_info_sets, price_bars_index)
    if active_indices is None:
        active_indices = get_active_indices(samples_info_sets, price_bars_index)

    n_obs = ind_mat.shape[1]
    if sample_length is None:
        sample_length = n_obs

    rng = np.random.RandomState(random_state)
    seq_seeds = rng.randint(0, _MAX_INT, size=n_repeats)

    std_uniq = np.empty(n_repeats)
    seq_uniq = np.empty(n_repeats)
    std_counts = np.zeros(n_obs, dtype=np.int64)
    seq_counts = np.zeros(n_obs, dtype=np.int64)

    for i in range(n_repeats):
        phi_std = _standard_draw(n_obs, sample_length, rng)
        std_uniq[i] = float(get_ind_mat_average_uniqueness(ind_mat[:, phi_std]))
        std_counts += np.bincount(phi_std, minlength=n_obs)

        phi_seq = np.asarray(
            seq_bootstrap(
                active_indices, sample_length=sample_length, random_seed=int(seq_seeds[i])
            ),
            dtype=np.int64,
        )
        seq_uniq[i] = float(get_ind_mat_average_uniqueness(ind_mat[:, phi_seq]))
        seq_counts += np.bincount(phi_seq, minlength=n_obs)

    uniqueness = pd.DataFrame({"standard": std_uniq, "sequential": seq_uniq})
    selection_counts = pd.DataFrame({"standard": std_counts, "sequential": seq_counts})
    return SamplingComparison(
        uniqueness=uniqueness,
        selection_counts=selection_counts,
        sample_length=sample_length,
        n_repeats=n_repeats,
    )


# ──────────────────────────── prediction comparison ──────────────────────────


def _as_base_pipeline(base_estimator) -> Pipeline:
    """Wrap a bare estimator as a MyPipeline so it accepts sample_weight."""
    if isinstance(base_estimator, Pipeline):
        return MyPipeline(base_estimator.steps)
    return MyPipeline([("clf", base_estimator)])


def _build_seq_pipeline(
    base, n_estimators, max_features, samples_info_sets, price_bars_index, random_state
) -> MyPipeline:
    return MyPipeline(
        [
            (
                "seq_bag",
                SequentiallyBootstrappedBaggingClassifier(
                    estimator=MyPipeline(base.steps),
                    n_estimators=n_estimators,
                    max_samples=1.0,  # full draw; the sampler corrects overlap
                    max_features=max_features,
                    samples_info_sets=samples_info_sets,
                    price_bars_index=price_bars_index,
                    random_state=random_state,
                    n_jobs=1,
                ),
            )
        ]
    )


def _build_std_pipeline(base, n_estimators, max_samples, max_features, random_state) -> MyPipeline:
    return MyPipeline(
        [
            (
                "bag",
                BaggingClassifier(
                    estimator=MyPipeline(base.steps),
                    n_estimators=n_estimators,
                    max_samples=max_samples,
                    max_features=max_features,
                    random_state=random_state,
                    n_jobs=1,
                ),
            )
        ]
    )


def _oof_metrics(oof: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None):
    """Compute calibration/discrimination metrics on the valid OOF predictions."""
    from ..calibration.calibration import brier_score, expected_calibration_error
    from ..cross_validation.scoring import probability_weighted_accuracy

    valid = ~np.isnan(oof)
    p = np.clip(oof[valid], 1e-15, 1 - 1e-15)
    yt = y[valid]
    sw = None if sample_weight is None else sample_weight[valid]
    proba2d = np.column_stack([1 - p, p])

    metrics = {
        "brier": brier_score(yt, p),
        "ece": expected_calibration_error(yt, p),
        "log_loss": log_loss(yt, proba2d, labels=[0, 1], sample_weight=sw),
        "pwa": probability_weighted_accuracy(yt, proba2d, sample_weight=sw, labels=[0, 1]),
        "accuracy": accuracy_score(yt, (p >= 0.5).astype(int), sample_weight=sw),
    }
    return metrics, valid


def compare_predictions(
    base_estimator,
    X: pd.DataFrame,
    y: pd.Series,
    samples_info_sets: pd.Series,
    price_bars_index,
    n_estimators: int = 100,
    n_splits: int = 5,
    pct_embargo: float = 0.01,
    sample_weight: pd.Series | None = None,
    max_samples_standard: float = 1.0,
    max_features: float = 1.0,
    random_state: int | None = 0,
) -> PredictionComparison:
    """
    Compare out-of-fold predictive quality of the two bootstrap methods.

    A bagging ensemble is built with each sampler and evaluated with
    ``CalibratorCV`` over ``PurgedKFold``; the raw (pre-calibration) OOF
    probabilities are scored and returned so reliability curves can be drawn.

    Parameters mirror the production bagging configuration.  ``max_samples_standard``
    is the per-estimator sample fraction for the *standard* ``BaggingClassifier``
    (AFML §6.2 recommends the dataset average uniqueness here); the sequential
    ensemble always draws full-size samples because its sampler corrects overlap
    internally.
    """
    from ..calibration.calibration import CalibratorCV
    from ..cross_validation.cross_validation import PurgedKFold

    base = _as_base_pipeline(base_estimator)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(np.asarray(X))
    y = pd.Series(np.asarray(y), index=X.index)
    sw = None if sample_weight is None else pd.Series(np.asarray(sample_weight), index=X.index)

    cv = PurgedKFold(n_splits=n_splits, t1=samples_info_sets, pct_embargo=pct_embargo)

    pipelines = {
        "standard": _build_std_pipeline(
            base, n_estimators, max_samples_standard, max_features, random_state
        ),
        "sequential": _build_seq_pipeline(
            base, n_estimators, max_features, samples_info_sets, price_bars_index, random_state
        ),
    }

    oof_probs = {}
    metrics = {}
    valid_mask = None
    for name, pipe in pipelines.items():
        cal = CalibratorCV(estimator=pipe, cv=cv)
        cal.fit(X, y, sample_weight=sw)
        oof = cal.oof_probs_
        oof_probs[name] = oof
        m, valid = _oof_metrics(oof, y.to_numpy(), None if sw is None else sw.to_numpy())
        metrics[name] = m
        valid_mask = valid if valid_mask is None else (valid_mask & valid)

    metrics_df = pd.DataFrame(metrics)[["standard", "sequential"]]
    return PredictionComparison(
        metrics=metrics_df,
        oof_probs=oof_probs,
        y_true=y.to_numpy(),
        valid_mask=valid_mask,
    )


# ────────────────────────────────  orchestrator  ─────────────────────────────


def compare_bootstrap_methods(
    base_estimator,
    X: pd.DataFrame,
    y: pd.Series,
    samples_info_sets: pd.Series,
    price_bars_index,
    n_estimators: int = 100,
    n_splits: int = 5,
    pct_embargo: float = 0.01,
    sample_weight: pd.Series | None = None,
    max_samples_standard: float | None = None,
    max_features: float = 1.0,
    n_repeats: int = 200,
    random_state: int | None = 0,
    run_sampling: bool = True,
    run_predictions: bool = True,
) -> BootstrapComparison:
    """
    Run the full sequential-vs-standard bootstrap comparison.

    Parameters
    ----------
    base_estimator : estimator
        Unfitted base classifier (or pipeline) bagged by both methods.
    X, y : pd.DataFrame, pd.Series
        Feature matrix and binary labels, indexed like ``samples_info_sets``.
    samples_info_sets : pd.Series
        Triple-barrier ``t1`` label spans (index=t0, value=t1).
    price_bars_index : pd.DatetimeIndex or array-like
        Bar timestamps used to build the indicator matrix.
    n_estimators, n_splits, pct_embargo, sample_weight, max_features
        Standard bagging / cross-validation configuration.
    max_samples_standard : float, optional
        Per-estimator sample fraction for the standard ensemble. Defaults to
        the dataset average uniqueness (AFML §6.2).
    n_repeats : int, default 200
        Monte-Carlo repeats for the sampling comparison.
    run_sampling, run_predictions : bool
        Toggle each comparison level independently.

    Returns
    -------
    BootstrapComparison
    """
    ind_mat = get_ind_matrix(samples_info_sets, price_bars_index)
    avg_u = float(get_ind_mat_average_uniqueness(ind_mat))
    if max_samples_standard is None:
        max_samples_standard = round(avg_u, 2)

    sampling = None
    if run_sampling:
        active_indices = get_active_indices(samples_info_sets, price_bars_index)
        sampling = compare_sampling(
            samples_info_sets,
            price_bars_index,
            n_repeats=n_repeats,
            random_state=random_state,
            ind_mat=ind_mat,
            active_indices=active_indices,
        )

    predictions = None
    if run_predictions:
        predictions = compare_predictions(
            base_estimator,
            X,
            y,
            samples_info_sets,
            price_bars_index,
            n_estimators=n_estimators,
            n_splits=n_splits,
            pct_embargo=pct_embargo,
            sample_weight=sample_weight,
            max_samples_standard=max_samples_standard,
            max_features=max_features,
            random_state=random_state,
        )

    return BootstrapComparison(
        avg_uniqueness=avg_u,
        n_estimators=n_estimators,
        max_samples_standard=max_samples_standard,
        sampling=sampling,
        predictions=predictions,
        _meta={"n_obs": ind_mat.shape[1], "n_bars": ind_mat.shape[0]},
    )


def compare_from_pipeline(pipeline, base_estimator=None, **kwargs) -> BootstrapComparison:
    """
    Convenience wrapper that pulls inputs from a fitted ``ModelDevelopmentPipeline``.

    Reads ``preprocessed_features``, ``events`` (``bin``/``t1``/``tW``) and
    ``bar_data.index`` off the pipeline object.  If ``base_estimator`` is not
    supplied, the base estimator of the fitted bagging wrapper is used.
    """
    X = pipeline.preprocessed_features.loc[pipeline.events.index]
    y = pipeline.events["bin"]
    t1 = pipeline.events["t1"]
    price_bars_index = pipeline.bar_data.index
    sample_weight = pipeline.events.get("tW") if hasattr(pipeline.events, "get") else None

    if base_estimator is None:
        model = pipeline.best_model
        # Unwrap calibrator / pipeline / bagging wrapper to reach the base.
        if hasattr(model, "estimator_"):  # CalibratorCV
            model = model.estimator_
        if hasattr(model, "steps"):
            model = model.steps[-1][1]
        base_estimator = getattr(model, "estimator", model)

    n_estimators = kwargs.pop(
        "n_estimators", pipeline.model_params.get("bagging_n_estimators", 100) or 100
    )
    n_splits = kwargs.pop("n_splits", pipeline.n_splits)
    pct_embargo = kwargs.pop("pct_embargo", pipeline.model_params.get("pct_embargo", 0.01))
    sample_weight = kwargs.pop("sample_weight", sample_weight)

    return compare_bootstrap_methods(
        base_estimator=base_estimator,
        X=X,
        y=y,
        samples_info_sets=t1,
        price_bars_index=price_bars_index,
        sample_weight=sample_weight,
        n_estimators=n_estimators,
        n_splits=n_splits,
        pct_embargo=pct_embargo,
        **kwargs,
    )


# ────────────────────────────────  plotting  ─────────────────────────────────


def _plot_uniqueness(ax, sampling: SamplingComparison):
    u = sampling.uniqueness
    bins = np.linspace(min(u.min()), max(u.max()), 30)
    for col, color in (("standard", STD_COLOR), ("sequential", SEQ_COLOR)):
        ax.hist(u[col], bins=bins, alpha=0.55, color=color, label=col, edgecolor="white")
        ax.axvline(u[col].mean(), color=color, linestyle="--", linewidth=1.5)
    ax.set_title("Average uniqueness of drawn set")
    ax.set_xlabel("Average uniqueness (higher = less redundant)")
    ax.set_ylabel(f"Count ({sampling.n_repeats} draws)")
    ax.legend()


def _plot_selection_freq(ax, sampling: SamplingComparison):
    # Normalise to a per-draw selection rate so the two are comparable.
    total = sampling.n_repeats * sampling.sample_length
    rate = sampling.selection_counts / total
    bins = np.linspace(0, float(rate.max().max()), 30)
    for col, color in (("standard", STD_COLOR), ("sequential", SEQ_COLOR)):
        ax.hist(rate[col], bins=bins, alpha=0.55, color=color, label=col, edgecolor="white")
    ax.set_title("Per-observation selection rate")
    ax.set_xlabel("Fraction of draws selecting an observation")
    ax.set_ylabel("Number of observations")
    ax.legend()


def _plot_reliability_overlay(ax, predictions: PredictionComparison):
    from ..calibration.calibration import compute_reliability

    y = predictions.y_true
    for name, color in (("standard", STD_COLOR), ("sequential", SEQ_COLOR)):
        oof = predictions.oof_probs[name]
        valid = ~np.isnan(oof)
        df = compute_reliability(y[valid], oof[valid], n_bins=10)
        m = df["count"] > 0
        ax.plot(
            df.loc[m, "pred_mean"],
            df.loc[m, "true_frac"],
            "o-",
            color=color,
            label=name,
            markersize=5,
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="perfect")
    ax.set_title("Reliability diagram (OOF)")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()


def _plot_prob_hist(ax, predictions: PredictionComparison):
    for name, color in (("standard", STD_COLOR), ("sequential", SEQ_COLOR)):
        oof = predictions.oof_probs[name]
        ax.hist(
            oof[~np.isnan(oof)],
            bins=25,
            range=(0, 1),
            alpha=0.55,
            color=color,
            label=name,
            edgecolor="white",
        )
    ax.set_title("OOF predicted-probability distribution")
    ax.set_xlabel("P(positive class)")
    ax.set_ylabel("Count")
    ax.legend()


def _plot_metrics(ax, predictions: PredictionComparison, metrics, title):
    m = predictions.metrics.loc[metrics]
    yy = np.arange(len(metrics))
    h = 0.38
    ax.barh(yy - h / 2, m["standard"], height=h, color=STD_COLOR, label="standard")
    ax.barh(yy + h / 2, m["sequential"], height=h, color=SEQ_COLOR, label="sequential")
    ax.set_yticks(yy)
    ax.set_yticklabels(metrics)
    ax.invert_yaxis()
    ax.set_title(title)
    for j, metric in enumerate(metrics):
        ax.text(
            m.loc[metric, "standard"],
            j - h / 2,
            f" {m.loc[metric, 'standard']:.3f}",
            va="center",
            fontsize=8,
        )
        ax.text(
            m.loc[metric, "sequential"],
            j + h / 2,
            f" {m.loc[metric, 'sequential']:.3f}",
            va="center",
            fontsize=8,
        )
    ax.legend()


def plot_bootstrap_comparison(
    result: BootstrapComparison,
    figsize=(16, 9),
    save_path: str | None = None,
    show: bool = False,
):
    """
    Render a multi-panel comparison figure from a ``BootstrapComparison``.

    Panels are included for whichever comparisons were run: sampling
    (uniqueness + selection rate) on the top row, predictive (reliability,
    probability distribution, metrics) on the bottom row.  Returns the figure.
    """
    has_s = result.sampling is not None
    has_p = result.predictions is not None
    if not (has_s or has_p):
        raise ValueError("Nothing to plot: result has neither sampling nor predictions.")

    n_rows = int(has_s) + int(has_p)
    fig = plt.figure(figsize=(figsize[0], figsize[1] * n_rows / 2))
    row = 0

    if has_s:
        ax1 = fig.add_subplot(n_rows, 3, row * 3 + 1)
        ax2 = fig.add_subplot(n_rows, 3, row * 3 + 2)
        _plot_uniqueness(ax1, result.sampling)
        _plot_selection_freq(ax2, result.sampling)
        # Third cell on the sampling row: a compact text summary.
        ax3 = fig.add_subplot(n_rows, 3, row * 3 + 3)
        ax3.axis("off")
        s = result.sampling.summary()
        ax3.text(
            0.0,
            0.95,
            "Sampling summary\n"
            f"avg uniqueness (dataset): {result.avg_uniqueness:.3f}\n"
            f"standard   : {s.loc['standard', 'mean']:.3f}\n"
            f"sequential : {s.loc['sequential', 'mean']:.3f}\n"
            f"gain (seq/std): {s['uniqueness_gain'].iloc[0]:.2f}x\n"
            f"standard max_samples: {result.max_samples_standard}",
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
            transform=ax3.transAxes,
        )
        row += 1

    if has_p:
        ax4 = fig.add_subplot(n_rows, 3, row * 3 + 1)
        ax5 = fig.add_subplot(n_rows, 3, row * 3 + 2)
        ax6 = fig.add_subplot(n_rows, 3, row * 3 + 3)
        _plot_reliability_overlay(ax4, result.predictions)
        _plot_prob_hist(ax5, result.predictions)
        # Brier/ECE/log-loss are "lower is better"; PWA/accuracy "higher".
        avail = list(result.predictions.metrics.index)
        lower = [m for m in ("brier", "ece", "log_loss") if m in avail]
        higher = [m for m in ("pwa", "accuracy") if m in avail]
        # Show lower-is-better as the metrics panel; annotate higher in title.
        _plot_metrics(
            ax6,
            result.predictions,
            lower + higher,
            "OOF metrics  (brier/ece/log_loss ↓ better · pwa/accuracy ↑ better)",
        )

    fig.suptitle("Sequential vs. standard bootstrap", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
