"""
feature_engineering.py
----------------------
Independent feature-engineering and feature-importance pipeline.

Where ``ModelDevelopmentPipeline`` (model_development.py) is concerned with
*selecting a model*, this module is concerned with *understanding the features*
that feed it.  It shares the same data-loading, triple-barrier labeling and
optimal sample-weight machinery, but instead of HPO / calibration / ONNX export
it builds a feature matrix from an arbitrary *list* of feature functions and
runs the algorithms in ``afml.feature_importance`` over it, producing the same
style of self-contained HTML reports and a versioned artifact directory.

Design goals
------------
1. **Composable features.**  Features are supplied as a *list* of specs, each
   ``{"func": callable, "params": dict, "name": str}``.  Every function in
   ``afml.features`` (microstructure, fractals, returns, moving averages,
   fracdiff, …) — or any user callable that takes a bar DataFrame and returns a
   feature DataFrame — can be dropped in.  The pipeline concatenates their
   outputs, appends calendar/session features, and records *provenance*
   (which spec produced which columns) for the report.

2. **Reuse, don't duplicate.**  Data loading, triple-barrier labeling and the
   sample-weight search are imported directly from ``model_development`` so the
   feature-importance ranking is computed against exactly the labels and weights
   a model would be trained on.

3. **Selectable importance suite.**  Every method in ``afml.feature_importance``
   is wired in and toggled via ``importance_config['methods']``:

       'mdi'            Mean Decrease Impurity              (AFML 8.2, fast, IS)
       'mda'            Mean Decrease Accuracy              (AFML 8.3, OOS, CV)
       'sfi'            Single Feature Importance           (AFML 8.4, OOS, CV)
       'clustered_mdi'  Clustered MDI                       (MLAM 6.5.2)
       'clustered_mda'  Clustered MDA                       (MLAM 6.5.2)
       'pca'            PCA orthogonality / MDI correlation (AFML 8.6)
       'orthogonal'     MDI on PCA-orthogonalized features  (AFML 8.5)
       'fingerprint'    Linear / non-linear effect split    (Li et al. 2019)

   MDA and SFI run purged, embargoed cross-validation per feature (or per
   cluster) and dominate runtime; they are opt-in.

Workflow
--------
    1. load_training_data()    — ticks → bars (shared loader).
    2. engineer_features()     — apply every feature spec + calendar features.
    3. generate_labels()       — triple-barrier events (shared).
    4. compute_sample_weights()— optimal weighting-scheme search (shared).
    5. preprocess_features()   — align to events, drop NaN / constant / dup cols.
    6. analyze_importance()     — fit the importance model, run selected methods.
    7. _generate_reports()      — dark-themed HTML: catalog + importance + PCA.
    8. _save_artifacts()        — parquet feature matrix, per-method CSVs, configs.

Example
-------
    >>> from afml.features import compute_all_microfeatures, get_lagged_returns
    >>> from afml.features.fractals import get_fractal_features
    >>> pipe = FeatureEngineeringPipeline(
    ...     data_config=data_config,                       # same shape as MDP
    ...     feature_configs=[
    ...         {"func": compute_all_microfeatures, "params": {}, "name": "micro"},
    ...         {"func": get_lagged_returns,        "params": {"n_lags": 5}},
    ...         {"func": get_fractal_features,      "params": {}},
    ...     ],
    ...     target_config=target_config,
    ...     label_config=label_config,
    ...     importance_config={"methods": ["mdi", "mda", "clustered_mdi", "pca"]},
    ... )
    >>> importances, catalog, config = pipe.run()
"""

import base64
import inspect
import time
import warnings
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures
from loguru import logger
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.pipeline import Pipeline

from ..cache import cacheable, get_cache_monitor, print_contamination_report
from ..cross_validation.cross_validation import PurgedKFold, ml_cross_val_score
from ..data_structures import _INFO_BAR_TYPES
from ..feature_importance import (
    ClassificationModelFingerprint,
    feature_pca_analysis,
    get_orthogonal_features,
    mean_decrease_accuracy,
    mean_decrease_impurity,
    plot_feature_importance,
    single_feature_importance,
)
from ..features.trading_session import get_time_features
from ..mt5.tick_data_loader import tick_data_loader as loader
from ..strategies.trading_strategies import BaseStrategy

# Reuse the exact data / label / weight machinery the model pipeline uses, so
# importances are measured against the labels and weights a model would see.
from .model_development import (
    generate_events_triple_barrier,
    get_optimal_sample_weight,
    load_and_prepare_training_data,
)

warnings.filterwarnings(
    "ignore",
    message="X has feature names, but .* was fitted without feature names",
    category=UserWarning,
)
warnings.filterwarnings("ignore", "All-NaN slice encountered")

# Major currencies used to decide whether session/calendar features should be
# generated in forex mode. Mirrors model_development.create_feature_engineering_pipeline.
_MAJOR_CURRENCIES = ("USD", "EUR", "JPY", "GBP", "AUD", "NZD", "CAD", "CHF")

# Default importance suite: everything cheap + MDA, but not the per-feature SFI.
_DEFAULT_METHODS = ("mdi", "mda", "clustered_mdi", "pca")

_ALL_METHODS = (
    "mdi",
    "mda",
    "sfi",
    "clustered_mdi",
    "clustered_mda",
    "pca",
    "orthogonal",
    "fingerprint",
)


# ============================================================================
# Cached feature builder
# ============================================================================


@cacheable(time_aware=True)
def build_feature_matrix(
    data: pd.DataFrame,
    ticks: pd.DataFrame,
    feature_configs: list[dict],
    data_config: dict,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Compute the combined feature matrix from a list of feature specifications.

    Each spec contributes a block of columns; blocks are concatenated and a
    calendar/session block is appended.  A provenance map records which spec
    produced which columns so the report can attribute importance back to its
    source.

    Parameters
    ----------
    data : pd.DataFrame
        OHLCV bar data (the host frame passed to every feature function).
    ticks : pd.DataFrame
        Tick data, injected only into feature functions that declare a ``ticks``
        parameter and don't already receive one via ``params``.
    feature_configs : list of dict
        Feature specifications.  Each item:
            - func   : callable(data, **params) -> pd.DataFrame
            - params : dict, optional
            - name   : str, optional (defaults to ``func.__name__``)
    data_config : dict
        Must contain ``symbol``, ``bar_size`` and ``bar_type`` for the calendar
        feature block.

    Returns
    -------
    features : pd.DataFrame
        Concatenated feature matrix (duplicate column names de-duplicated,
        all-NaN rows dropped).
    provenance : dict[str, list[str]]
        Mapping ``source name -> list of column names`` contributed.

    Notes
    -----
    - Time-aware caching prevents look-ahead contamination across calls.
    """
    blocks: list[pd.DataFrame] = []
    provenance: dict[str, list[str]] = {}
    seen: set = set()

    for spec in feature_configs:
        func = spec["func"]
        name = spec.get("name", func.__name__)
        params = dict(spec.get("params", {}))

        sig = inspect.signature(func)
        if "ticks" in sig.parameters and "ticks" not in params:
            params["ticks"] = ticks

        block = func(data, **params)
        if not isinstance(block, pd.DataFrame):
            block = pd.DataFrame(block)

        # Drop intra-block duplicate columns, then columns already produced by an
        # earlier block, so the final concat has unique column labels.
        block = block.loc[:, ~block.columns.duplicated()]
        new_cols = [c for c in block.columns if c not in seen]
        if not new_cols:
            logger.warning(f"Feature source '{name}' produced only duplicate columns; skipped.")
            continue
        block = block[new_cols]
        seen.update(new_cols)

        provenance[name] = list(new_cols)
        blocks.append(block)

    # Calendar / session block (always appended once).
    symbol = data_config.get("symbol", "")
    is_forex = any(ccy in symbol for ccy in _MAJOR_CURRENCIES)
    time_feat = get_time_features(
        data,
        timeframe=data_config["bar_size"],
        bar_type=data_config["bar_type"],
        forex=is_forex,
    )
    time_cols = [c for c in time_feat.columns if c not in seen]
    if time_cols:
        provenance["time_features"] = time_cols
        blocks.append(time_feat[time_cols])

    features = pd.concat(blocks, axis=1).dropna(how="all")
    return features, provenance


def _resolve_scoring(label_space: set):
    """Pick the scoring callable: F1 for binary meta-labels, log-loss otherwise."""
    return f1_score if label_space == {0, 1} else log_loss


# ============================================================================
# FeatureEngineeringPipeline
# ============================================================================


class FeatureEngineeringPipeline:
    """
    Build a feature matrix from composable feature functions and analyse it with
    the ``afml.feature_importance`` suite, emitting ModelDevelopmentPipeline-style
    reports and a versioned artifact directory.

    Parameters
    ----------
    data_config : dict
        Same shape consumed by ``load_and_prepare_training_data`` /
        ``ModelDevelopmentPipeline`` (symbol, start_date, end_date, account_name,
        bar_type, bar_size, price, path, target_timeframe, …).
    feature_configs : list of dict
        Feature specifications, each ``{"func", "params", "name"}``.  This is the
        composable "add a function" surface; order is irrelevant because columns
        are concatenated and de-duplicated.
    target_config : dict
        Volatility-target spec ``{"func", "params"}`` for the triple barrier.
    label_config : dict
        Triple-barrier parameters (profit_target, stop_loss, max_holding_period,
        min_ret, vertical_barrier_zero, filter_as_series).
    importance_config : dict, optional
        Controls the importance analysis:
            - methods            : list[str] subset of _ALL_METHODS
                                   (default: MDI, MDA, clustered MDI, PCA).
            - n_splits           : int, PurgedKFold folds (default 5).
            - pct_embargo        : float (default 0.02).
            - estimator          : fitted-able classifier exposing
                                   ``estimators_`` for MDI (default a
                                   RandomForest with ``max_features=1`` per the
                                   AFML masking tip).
            - top_n              : int, max features per bar chart (default 30).
            - pca_variance_thresh: float (default 0.95).
            - clustering         : dict of get_feature_clusters kwargs
                                   (default ``{"dependence_metric": "linear"}``).
            - fingerprint_num_values   : int (default 50).
            - fingerprint_pairwise_top : int, top-k MDI features to test pairwise
                                         (default 0 = none).
    strategy : BaseStrategy or None, optional
        Entry/side signal.  ``None`` → every bar is a candidate entry (symmetric
        labels), matching ModelDevelopmentPipeline's default.
    base_dir : str, default "Features"
        Root of the artifact directory tree.
    """

    def __init__(
        self,
        data_config: dict,
        feature_configs: list[dict],
        target_config: dict,
        label_config: dict,
        importance_config: dict | None = None,
        strategy: BaseStrategy | None = None,
        base_dir: str = "Features",
    ):
        from .file_manager import ModelFileManager

        if not feature_configs:
            raise ValueError("feature_configs must contain at least one feature spec.")
        for spec in feature_configs:
            if "func" not in spec or not callable(spec["func"]):
                raise ValueError(f"Each feature spec needs a callable 'func'; got {spec!r}")

        # Mirror ModelDevelopmentPipeline's target_timeframe / bar_size handling.
        if data_config.get("target_timeframe") is not None:
            if data_config["bar_type"] in _INFO_BAR_TYPES:
                data_config["bar_size"] = data_config["target_timeframe"]
            else:
                data_config["target_timeframe"] = None

        self.data_config = data_config
        self.symbol = data_config["symbol"]
        self.train_start = data_config["start_date"]
        self.train_end = data_config["end_date"]
        self.account_name = data_config.get("account_name", "default")
        self.feature_configs = feature_configs
        self.target_config = target_config
        self.label_config = label_config
        self.strategy = strategy
        self.importance_config = dict(importance_config or {})
        self.pipeline_version = "1.0"

        methods = list(self.importance_config.get("methods", _DEFAULT_METHODS))
        invalid = [m for m in methods if m not in _ALL_METHODS]
        if invalid:
            raise ValueError(f"Unknown importance method(s) {invalid}; valid: {list(_ALL_METHODS)}")
        self.methods = methods

        self.imp_n_splits = self.importance_config.get("n_splits", 5)
        self.imp_pct_embargo = self.importance_config.get("pct_embargo", 0.02)
        self.top_n = self.importance_config.get("top_n", 30)
        self.pca_variance_thresh = self.importance_config.get("pca_variance_thresh", 0.95)
        self.clustering_kwargs = self.importance_config.get(
            "clustering", {"dependence_metric": "linear"}
        )
        self.fingerprint_num_values = self.importance_config.get("fingerprint_num_values", 50)
        self.fingerprint_pairwise_top = self.importance_config.get("fingerprint_pairwise_top", 0)

        # Default importance estimator: RandomForest with max_features=1 (AFML
        # Snippet 8.2 tip — one random feature per split avoids masking and keeps
        # MDI honest). estimators_ are bare trees, so MDI works directly.
        self.importance_estimator = self.importance_config.get(
            "estimator",
            RandomForestClassifier(
                n_estimators=self.importance_config.get("n_estimators", 200),
                max_features=1,
                criterion="entropy",
                class_weight="balanced_subsample",
                min_weight_fraction_leaf=0.0,
                oob_score=True,
                bootstrap=True,
                n_jobs=self.importance_config.get("n_jobs", -1),
                random_state=self.importance_config.get("random_state", 42),
            ),
        )

        # ── config for the file manager / report headers ──────────────────────
        self.config = data_config.copy()
        self.config.pop("path", None)
        self.config["training_start"] = self.config.pop("start_date")
        self.config["training_end"] = self.config.pop("end_date")
        self.config["strategy"] = (
            strategy.get_strategy_name() if strategy is not None else "ml_driven"
        )
        self.config["feature_sources"] = [
            spec.get("name", spec["func"].__name__) for spec in feature_configs
        ]
        self.config["target_func"] = target_config["func"].__name__
        self.config["target_params"] = target_config["params"]
        self.config["importance_methods"] = self.methods
        self.config.update(label_config)

        # ── state ──────────────────────────────────────────────────────────────
        self.bar_data = None
        self.features = None
        self.provenance = None
        self.events = None
        self.sample_weight = None
        self.best_weighting_scheme = None
        self.weight_cv_results = None
        self.preprocessed_features = None
        self.preprocessor = None
        self.dropped_columns_ = None
        self.is_primary = None
        self.importance_model_ = None
        self.oob_score_ = None
        self.oos_score_ = None
        self.scoring_name_ = None
        self.feature_clusters_ = None
        self.importances: dict[str, pd.DataFrame] = {}
        self.pca_corr_ = None
        self.fingerprint_ = None
        self.catalog_ = None
        # method name -> {"status": ok|failed|empty, "detail": str}. Every
        # requested method gets an entry whether or not it produced a table, so
        # the report can distinguish "not requested" from "requested and failed".
        self.method_status_: dict[str, dict] = {}
        self.denoise_info_: dict[str, float] = {}
        self.substitution_flags_: dict[str, pd.DataFrame] = {}

        self.model_type = "featimp_rf"
        self.file_manager = ModelFileManager(base_dir)
        self.file_paths = self.file_manager.setup_model_directory(self.config, self.model_type)

        self.completed_steps = {
            "data_loading": False,
            "feature_engineering": False,
            "label_generation": False,
            "weight_computation": False,
            "preprocessing": False,
            "importance_analysis": False,
        }

        self._setup_logging()

    # ── logging ────────────────────────────────────────────────────────────────

    def _setup_logging(self):
        self.logger = logger.bind(context=self.__class__.__name__)
        self._log_handler_id = logger.add(
            self.file_paths["logs"] / "feature_engineering.log",
            level="DEBUG",
            rotation="50 MB",
            filter=lambda record: record["extra"].get("context") == "FeatureEngineeringPipeline",
        )

    # ── run() ──────────────────────────────────────────────────────────────────

    def run(
        self,
        generate_reports: bool = True,
        save: bool = True,
        cache_reports: bool = False,
        display: bool = False,
        verbose: bool = True,
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
        """
        Execute the full feature-engineering + importance analysis.

        Returns
        -------
        tuple
            (importances, catalog, config) where ``importances`` is a dict of
            per-method importance DataFrames and ``catalog`` is the feature
            provenance/coverage table.
        """
        time0 = time.time()
        if verbose:
            print("\n" + "=" * 70)
            print("FEATURE ENGINEERING & IMPORTANCE PIPELINE")
            print(f"Methods: {', '.join(self.methods)}")
            print("=" * 70)

        try:
            self.load_training_data()
            self.engineer_features()
            self.generate_labels()
            self.compute_sample_weights()
            self.preprocess_features()
            self.analyze_importance()
            self._build_catalog()

            if generate_reports:
                self._generate_reports(display=display)
            if cache_reports:
                self._display_cache_reports()
            if save:
                self._save_artifacts()

            if verbose:
                elapsed = str(pd.Timedelta(seconds=time.time() - time0).round("1s")).replace(
                    "0 days ", ""
                )
                print(f"\n✓ Completed in {elapsed}")

            return self.importances, self.catalog_, self.config

        except Exception as e:
            self.logger.error(f"Feature-engineering pipeline failed: {e}")
            raise

    # ── steps ──────────────────────────────────────────────────────────────────

    def load_training_data(self):
        self.bar_data = load_and_prepare_training_data(**self.data_config)
        self.completed_steps["data_loading"] = True

    def engineer_features(self):
        ticks = loader.get_tick_data(
            self.data_config["symbol"],
            self.data_config["start_date"],
            self.data_config["end_date"],
            self.data_config["account_name"],
        )
        self.features, self.provenance = build_feature_matrix(
            self.bar_data, ticks, self.feature_configs, self.data_config
        )
        self.logger.info(
            f"features: {self.features.shape} from {len(self.provenance)} source(s) "
            f"({', '.join(self.provenance)})"
        )
        self.completed_steps["feature_engineering"] = True

    def generate_labels(self):
        self.events = generate_events_triple_barrier(
            self.bar_data,
            self.strategy,
            self.target_config,
            **self.label_config,
        )
        self.is_primary = "side" not in self.events.columns
        self.config["model_role"] = "primary" if self.is_primary else "secondary"
        self.logger.info(
            f"Model role: {self.config['model_role']} | "
            f"Label space: {np.unique(self.events['bin']).tolist()} | "
            f"avgU={self.events['tW'].mean():.4f}"
        )
        self.completed_steps["label_generation"] = True

    def compute_sample_weights(self):
        self.sample_weight, self.weight_cv_results = get_optimal_sample_weight(
            self.bar_data.index,
            self.events,
            self.features,
            self.imp_n_splits,
            None,
        )
        self.best_weighting_scheme = self.weight_cv_results["best_scheme"]
        self.logger.info(
            f"best_weighting_scheme: {self.best_weighting_scheme} "
            f"({self.weight_cv_results['scoring']}={self.weight_cv_results['best_score']:.4f})"
        )
        self.completed_steps["weight_computation"] = True

    def preprocess_features(self):
        """Align features to events, drop NaN rows, then constant/duplicate cols."""
        combined = self.features

        all_nan_cols = combined.columns[combined.isna().all()].tolist()
        if all_nan_cols:
            self.logger.warning(f"Dropping {len(all_nan_cols)} all-NaN column(s): {all_nan_cols}")
            combined = combined.drop(columns=all_nan_cols)

        valid_index = combined.dropna().index.intersection(self.events.index)
        combined = combined.loc[valid_index]
        if combined.empty:
            raise ValueError(
                "No samples remain after aligning features to events and dropping NaN rows. "
                "Check feature warmup windows."
            )

        self.preprocessor = Pipeline(
            [("dcf", DropConstantFeatures()), ("ddf", DropDuplicateFeatures())]
        )
        self.preprocessed_features = self.preprocessor.fit_transform(combined)

        constant_dropped = list(
            getattr(self.preprocessor.named_steps["dcf"], "features_to_drop_", [])
        )
        duplicate_dropped = list(
            getattr(self.preprocessor.named_steps["ddf"], "features_to_drop_", [])
        )
        self.dropped_columns_ = {"constant": constant_dropped, "duplicate": duplicate_dropped}

        self.events = self.events.loc[self.preprocessed_features.index]
        self.logger.info(
            f"Preprocessed features: {self.preprocessed_features.shape} "
            f"(dropped {combined.shape[1] - self.preprocessed_features.shape[1]} columns)"
        )
        self.completed_steps["preprocessing"] = True

    # ── importance analysis ─────────────────────────────────────────────────────

    def analyze_importance(self):
        """Fit the importance model and run every selected importance method."""
        X = self.preprocessed_features.loc[self.events.index]
        y = self.events["bin"]
        # Two weight forms are needed because the AFML kernels disagree on type:
        #   mean_decrease_accuracy indexes weights positionally as a numpy array
        #   (sample_weight[train]); ml_cross_val_score (used by SFI and the
        #   reference score) indexes them as a pandas Series (.iloc[train]).
        sw_train_s = self.sample_weight.loc[X.index]
        sw_score_s = self.events["w"].loc[X.index]
        sw_train = sw_train_s.to_numpy()
        sw_score = sw_score_s.to_numpy()

        scoring = _resolve_scoring(set(y.unique()))
        self.scoring_name_ = "f1" if scoring is f1_score else "neg_log_loss"

        # Fit the reference model once (used by MDI, PCA correlation, fingerprint).
        self.importance_model_ = clone(self.importance_estimator).fit(X, y, sample_weight=sw_train)
        self.oob_score_ = float(getattr(self.importance_model_, "oob_score_", np.nan))

        cv = PurgedKFold(
            n_splits=self.imp_n_splits, t1=self.events["t1"], pct_embargo=self.imp_pct_embargo
        )

        # One reference OOS score for the importance plots' annotation.
        try:
            self.oos_score_ = float(
                ml_cross_val_score(
                    clone(self.importance_estimator),
                    X,
                    y,
                    cv_gen=cv,
                    sample_weight_train=sw_train_s,
                    sample_weight_score=sw_score_s,
                    scoring=scoring,
                ).mean()
            )
        except Exception as e:
            self.logger.warning(f"Reference OOS score failed: {e}")
            self.oos_score_ = float("nan")

        needs_clusters = any(m.startswith("clustered") for m in self.methods)
        if needs_clusters:
            self.feature_clusters_ = self._safe("feature clustering", self._compute_clusters, X)

        runners = {
            "mdi": lambda: self._run_mdi(),
            # MDA / clustered MDA take numpy weights (positional indexing);
            # SFI takes Series weights (routed through ml_cross_val_score).
            "mda": lambda: self._run_mda(X, y, cv, sw_train, sw_score, scoring),
            "sfi": lambda: self._run_sfi(X, y, cv, sw_train_s, sw_score_s, scoring),
            "clustered_mdi": lambda: self._run_clustered_mdi(),
            "clustered_mda": lambda: self._run_clustered_mda(X, y, cv, sw_train, sw_score, scoring),
            "pca": lambda: self._run_pca(X),
            "orthogonal": lambda: self._run_orthogonal(X, y, sw_train),
            "fingerprint": lambda: self._run_fingerprint(X),
        }
        for method in self.methods:
            self.logger.info(f"Running importance method: {method}")
            self.method_status_.setdefault(method, {"status": "ok", "detail": ""})
            result = self._safe(method, runners[method])
            if method == "pca":
                self.pca_corr_ = result
            elif method == "fingerprint":
                self.fingerprint_ = result
            elif result is not None:
                self.importances[method] = result

            if result is None and self.method_status_[method]["status"] == "ok":
                # No exception, but nothing came back. The clustered methods
                # return None when clustering itself failed upstream.
                self.method_status_[method] = {
                    "status": "empty",
                    "detail": (
                        "returned no table; feature clustering unavailable"
                        if method.startswith("clustered")
                        else "returned no table"
                    ),
                }

        self._compute_substitution_flags()
        self.completed_steps["importance_analysis"] = True

    # ── substitution-effect diff ────────────────────────────────────────────

    def _compute_substitution_flags(self, dilution_threshold: float = 2.0):
        """
        Quantify how far each cluster's credit is spread across its members.

        For a cluster whose members are near-duplicates, a naive per-feature
        method splits one predictor's credit among them, so the cluster total is
        much larger than any single member's score. The ratio

            dilution = sum(naive over members) / max(naive over members)

        is 1.0 when a single member carries the cluster and approaches the
        member count when credit is split evenly. Anything above
        ``dilution_threshold`` means the naive ranking understates every member.

        This replaces a per-feature rank-shift diff, which is unusable on coarse
        partitions: all members of a cluster share one clustered score and
        therefore one rank, so the shift measures the tie structure rather than
        substitution.
        """
        if not self.feature_clusters_:
            return
        for naive_name, clustered_name in (("mdi", "clustered_mdi"), ("mda", "clustered_mda")):
            naive = self.importances.get(naive_name)
            clustered = self.importances.get(clustered_name)
            if naive is None or clustered is None:
                continue
            rows = []
            for i, cl in enumerate(self.feature_clusters_, start=1):
                members = [c for c in cl if c in naive.index]
                if not members:
                    continue
                vals = naive.loc[members, "mean"].astype(float)
                total, top = float(vals.sum()), float(vals.max())
                rows.append(
                    {
                        "cluster": f"Cluster {i}",
                        "size": len(members),
                        "clustered_score": float(clustered.loc[members[0], "mean"]),
                        "naive_total": total,
                        "naive_best_member": top,
                        "top_member": str(vals.idxmax()),
                        "dilution": (total / top) if top else np.nan,
                    }
                )
            df = (
                pd.DataFrame(rows)
                .set_index("cluster")
                .sort_values("clustered_score", ascending=False)
            )
            self.substitution_flags_[naive_name] = df
            diluted = df[df["dilution"] > dilution_threshold]
            if not diluted.empty:
                self.logger.warning(
                    f"{len(diluted)} cluster(s) show {naive_name} credit split across "
                    f"members (dilution > {dilution_threshold}): {list(diluted.index)}"
                )

    def _safe(self, label, fn, *args):
        """
        Run ``fn``; log and swallow failures so one method can't kill the run.

        The failure is recorded in ``method_status_`` as well as logged. A
        swallowed exception that leaves no trace in the report is worse than a
        crash, because the resulting report looks complete.
        """
        try:
            return fn(*args)
        except Exception as e:
            self.logger.warning(f"Importance step '{label}' failed: {e}")
            self.method_status_[label] = {
                "status": "failed",
                "detail": f"{type(e).__name__}: {e}",
            }
            return None

    def _compute_clusters(self, X):
        from ..clustering.feature_clusters import get_feature_clusters
        from ..denoising.denoising_detoning import EffectiveSampleSizeError

        kwargs = dict(self.clustering_kwargs)
        if kwargs.get("denoise"):
            # Ask for the diagnostics so the report can state which spectrum the
            # partition came from rather than leaving it implicit.
            try:
                clusters, self.denoise_info_ = get_feature_clusters(X, return_info=True, **kwargs)
            except EffectiveSampleSizeError as err:
                # The panel is too persistent for a Marcenko-Pastur fit. Cluster
                # on the raw correlation the exception carries rather than on a
                # ceiling the data cannot support. Nothing is recomputed: the
                # correlation matrix comes back on the exception, and the fit
                # was never reached.
                self.logger.warning(
                    f"Denoising skipped: {err} Clustering will run on the raw "
                    "sample correlation matrix, so the partition carries noise "
                    "eigenvalues and the clustered scores should be read with that "
                    "in mind."
                )
                kwargs.pop("denoise", None)
                kwargs.pop("detone", None)
                self.denoise_info_ = dict(err.info)
                self.denoise_info_.update({"skipped": True, "reason": str(err)})
                clusters = get_feature_clusters(X, corr=err.corr, **kwargs)
                self.logger.info(f"Feature clustering produced {len(clusters)} cluster(s).")
                return clusters
            i = self.denoise_info_
            self.logger.info(
                f"Correlation cleaned before clustering | T={i['t_raw']} "
                f"T_eff={i['t_eff']:.0f} (rho_bar={i['rho_bar']:.3f}) "
                f"q={i['q']:.2f} ceiling={i['noise_ceiling']:.3f} "
                f"factors={i['n_facts']}/{i['n_features']}"
                + (f" | detoned ({i['market_components']} component(s))" if i["detoned"] else "")
            )
        else:
            clusters = get_feature_clusters(X, **kwargs)
            self.logger.warning(
                "Clustering ran on the raw sample correlation matrix. Set "
                "importance_config['clustering']['denoise'] = True to shrink the "
                "noise eigenvalues first."
            )
        self.logger.info(f"Feature clustering produced {len(clusters)} cluster(s).")
        return clusters

    def _run_mdi(self):
        return mean_decrease_impurity(self.importance_model_, self._feat_names())

    def _run_mda(self, X, y, cv, sw_train, sw_score, scoring):
        return mean_decrease_accuracy(
            clone(self.importance_estimator),
            X,
            y,
            cv,
            sample_weight_train=sw_train,
            sample_weight_score=sw_score,
            scoring=scoring,
        )

    def _run_sfi(self, X, y, cv, sw_train, sw_score, scoring):
        return single_feature_importance(
            clone(self.importance_estimator),
            X,
            y,
            cv,
            sample_weight_train=sw_train,
            sample_weight_score=sw_score,
            scoring=scoring,
        )

    def _run_clustered_mdi(self):
        if not self.feature_clusters_:
            return None
        return mean_decrease_impurity(
            self.importance_model_, self._feat_names(), clustered_subsets=self.feature_clusters_
        )

    def _run_clustered_mda(self, X, y, cv, sw_train, sw_score, scoring):
        if not self.feature_clusters_:
            return None
        return mean_decrease_accuracy(
            clone(self.importance_estimator),
            X,
            y,
            cv,
            clustered_subsets=self.feature_clusters_,
            sample_weight_train=sw_train,
            sample_weight_score=sw_score,
            scoring=scoring,
        )

    def _run_pca(self, X):
        """PCA-vs-MDI correlation (needs an MDI table; compute if not already run)."""
        mdi = self.importances.get("mdi")
        if mdi is None:
            mdi = mean_decrease_impurity(self.importance_model_, self._feat_names())
        return feature_pca_analysis(X, mdi, variance_thresh=self.pca_variance_thresh)

    def _run_orthogonal(self, X, y, sw_train):
        """
        MDI on PCA-orthogonalized features (AFML 8.5, Snippet 8.5).

        Orthogonalization removes the linear substitution effects that dilute MDI/MDA
        across correlated columns: a fresh forest is fit on the principal components
        (PC_*) explaining ``pca_variance_thresh`` of variance, and MDI is computed on
        them. Importance concentrated in a few PCs signals a low-dimensional signal;
        importance spread evenly across many PCs signals a high-dimensional / noisy one.
        """
        z_arr = get_orthogonal_features(X, variance_thresh=self.pca_variance_thresh)
        z = pd.DataFrame(
            z_arr, index=X.index, columns=[f"PC_{i + 1}" for i in range(z_arr.shape[1])]
        )
        model = clone(self.importance_estimator).fit(z, y, sample_weight=sw_train)
        return mean_decrease_impurity(model, z.columns.tolist())

    def _run_fingerprint(self, X):
        fp = ClassificationModelFingerprint()
        pairwise = None
        if self.fingerprint_pairwise_top and "mdi" in self.importances:
            top = (
                self.importances["mdi"]["mean"]
                .nlargest(self.fingerprint_pairwise_top)
                .index.tolist()
            )
            pairwise = [(a, b) for i, a in enumerate(top) for b in top[i + 1 :]]
        fp.fit(
            self.importance_model_,
            X,
            num_values=self.fingerprint_num_values,
            pairwise_combinations=pairwise,
        )
        return fp

    def _feat_names(self):
        return self.preprocessed_features.columns.tolist()

    # ── feature catalog ─────────────────────────────────────────────────────────

    def _build_catalog(self):
        """One row per surviving feature: source, NaN coverage, and mean MDI/MDA."""
        col_to_source = {}
        for source, cols in (self.provenance or {}).items():
            for c in cols:
                col_to_source[c] = source

        kept = self._feat_names()
        mdi = self.importances.get("mdi")
        mda = self.importances.get("mda")
        coverage = self.preprocessed_features.notna().mean()

        rows = []
        for col in kept:
            rows.append(
                {
                    "feature": col,
                    "source": col_to_source.get(col, "unknown"),
                    "coverage": float(coverage.get(col, np.nan)),
                    "mdi": float(mdi.loc[col, "mean"])
                    if mdi is not None and col in mdi.index
                    else np.nan,
                    "mda": float(mda.loc[col, "mean"])
                    if mda is not None and col in mda.index
                    else np.nan,
                }
            )
        self.catalog_ = pd.DataFrame(rows).sort_values(
            "mdi" if mdi is not None else "feature",
            ascending=False if mdi is not None else True,
        )

    # ── reporting ────────────────────────────────────────────────────────────────

    def _report_subtitle(self, sep: str = " &nbsp;|&nbsp; ") -> str:
        bar_label = (
            f"{self.data_config.get('bar_size', '')} {self.data_config.get('bar_type', '')}".strip()
        )
        return sep.join(
            [self.symbol, f"{bar_label} bars", f"{self.train_start} to {self.train_end}"]
        )

    def _report_title_suffix(self) -> str:
        return self._report_subtitle(sep=" | ").replace("&nbsp;", "")

    def _importance_png_b64(
        self, importance_df: pd.DataFrame, title: str, baseline: float | None = None
    ) -> str | None:
        """
        Render a horizontal importance bar chart (top-N) to a base64 PNG.

        ``baseline`` draws a vertical reference line. SFI reports a raw
        cross-validated score rather than a decrement, so its bars are only
        interpretable against the score an uninformative feature would earn.
        """
        import matplotlib.pyplot as plt

        df = importance_df.dropna(subset=["mean"]).copy()
        if df.empty:
            return None
        df = df.astype(float).sort_values("mean").tail(self.top_n)

        original_backend = plt.get_backend()
        plt.switch_backend("agg")
        try:
            with plt.ioff():
                fig, ax = plt.subplots(figsize=(10, max(2.5, len(df) / 3.2)))
                ax.barh(
                    df.index,
                    df["mean"],
                    xerr=df["std"],
                    color="#38bdf8",
                    alpha=0.85,
                    error_kw={"ecolor": "#f97316", "alpha": 0.6},
                )
                if baseline is not None:
                    ax.axvline(baseline, color="#f97316", linestyle="--", linewidth=1.2)
                    ax.text(
                        baseline,
                        len(df) - 0.4,
                        " uninformative baseline",
                        color="#f97316",
                        fontsize=8,
                        va="top",
                    )
                ax.set_title(
                    f"{title}  (OOB={self.oob_score_:.3f}, OOS {self.scoring_name_}={self.oos_score_:.3f})",
                    color="#f1f5f9",
                )
                ax.set_facecolor("#0f172a")
                fig.set_facecolor("#0f172a")
                ax.tick_params(colors="#94a3b8")
                for spine in ax.spines.values():
                    spine.set_color("#334155")
                fig.tight_layout()
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=140, facecolor="#0f172a", bbox_inches="tight")
                plt.close("all")
        finally:
            plt.switch_backend(original_backend)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def _generate_reports(self, display: bool = False):
        """Emit the dark-themed HTML summary; failures are non-fatal."""
        try:
            self._generate_summary_html()
            # Also persist the AFML reference bar chart per method to plots/.
            for method, df in self.importances.items():
                try:
                    plot_feature_importance(
                        df.copy(),
                        self.oob_score_,
                        self.oos_score_,
                        save_fig=True,
                        output_path=str(self.file_paths["plots"] / f"importance_{method}.png"),
                    )
                except Exception as e:
                    self.logger.warning(f"plot_feature_importance({method}) failed: {e}")
            if display:
                try:
                    from IPython.display import IFrame
                    from IPython.display import display as _display

                    report = self.file_paths["reports"] / "feature_engineering_summary.html"
                    _display(IFrame(src=report.as_uri(), width="100%", height=800))
                except Exception:
                    pass
        except Exception as e:
            self.logger.warning(f"Report generation failed: {e}")

    def _generate_summary_html(self):
        report_path = self.file_paths["reports"] / "feature_engineering_summary.html"

        # ── source coverage cards ────────────────────────────────────────────
        source_rows = []
        for source, cols in (self.provenance or {}).items():
            kept = [c for c in cols if c in self._feat_names()]
            source_rows.append(
                f"<tr><td>{source}</td><td>{len(cols)}</td><td>{len(kept)}</td></tr>"
            )
        source_table = (
            "<table><thead><tr><th>Source</th><th>Produced</th><th>Kept</th></tr></thead>"
            f"<tbody>{''.join(source_rows)}</tbody></table>"
        )

        # ── dropped columns ──────────────────────────────────────────────────
        dropped = self.dropped_columns_ or {}
        dropped_bits = []
        for reason in ("constant", "duplicate"):
            cols = dropped.get(reason, [])
            if cols:
                dropped_bits.append(
                    f"<tr><td>{reason.capitalize()}</td><td>{len(cols)}</td>"
                    f"<td>{', '.join(map(str, cols))}</td></tr>"
                )
        dropped_table = (
            "<table><thead><tr><th>Reason</th><th>Count</th><th>Columns</th></tr></thead>"
            f"<tbody>{''.join(dropped_bits)}</tbody></table>"
            if dropped_bits
            else '<p class="label">No columns dropped in preprocessing.</p>'
        )

        # ── method status ────────────────────────────────────────────────────
        # Rendered before anything else: a report that silently omits a failed
        # method is indistinguishable from one where it was never requested.
        status_rows = []
        for method in self.methods:
            st = self.method_status_.get(method, {"status": "not run", "detail": ""})
            badge = {
                "ok": "&#10003; ok",
                "failed": "&#10007; failed",
                "empty": "&#9888; empty",
            }.get(st["status"], st["status"])
            cls = "ok" if st["status"] == "ok" else "bad"
            status_rows.append(
                f'<tr><td>{method}</td><td class="{cls}">{badge}</td>'
                f'<td class="label">{st["detail"] or "&mdash;"}</td></tr>'
            )
        n_ok = sum(1 for m in self.methods if self.method_status_.get(m, {}).get("status") == "ok")
        status_section = f"""
            <div class="card">
                <h2>Method Status</h2>
                <p class="label">{n_ok} of {len(self.methods)} requested method(s) produced
                   a result. A method that fails is logged and skipped, so this table is the
                   only place a failure is visible.</p>
                <table><thead><tr><th>Method</th><th>Status</th><th>Detail</th></tr></thead>
                <tbody>{"".join(status_rows)}</tbody></table>
            </div>"""

        # ── clusters ─────────────────────────────────────────────────────────
        cluster_section = ""
        if self.feature_clusters_:
            crows = []
            for i, cl in enumerate(self.feature_clusters_, start=1):
                crows.append(
                    f"<tr><td>Cluster {i}</td><td>{len(cl)}</td>"
                    f'<td class="label">{", ".join(map(str, cl))}</td></tr>'
                )
            cluster_section = f"""
                <div class="card">
                    <h2>Feature Clusters (ONC)</h2>
                    <p class="label">Clustered MDI and MDA assign one score to a whole cluster
                       and broadcast it to every member, so cluster membership determines the
                       result. Very unequal cluster sizes make clustered MDI hard to compare
                       across clusters, because a cluster's score is the sum of its members'
                       impurity reductions.</p>
                    <table><thead><tr><th>Cluster</th><th>Size</th><th>Members</th></tr></thead>
                    <tbody>{"".join(crows)}</tbody></table>
                </div>"""

        # ── substitution-effect diff ─────────────────────────────────────────
        subst_section = ""
        for naive, flags in self.substitution_flags_.items():
            if flags is None or flags.empty:
                subst_section += (
                    f'<div class="card"><h2>Substitution Effects &mdash; {naive.upper()}</h2>'
                    f'<p class="label">No clusters available to compare against {naive}.</p></div>'
                )
                continue
            tbl = flags.rename_axis(None).to_html(
                classes="cal-table", border=0, float_format=lambda v: f"{v:.4f}"
            )
            worst = flags["dilution"].max()
            subst_section += f"""
                <div class="card">
                    <h2>Substitution Effects &mdash; {naive.upper()}</h2>
                    <p class="label">Dilution is the cluster's total {naive} score divided by its
                       best single member's score. A value near 1 means one feature carries the
                       cluster and the naive ranking is trustworthy for it. A value near the
                       cluster size means the credit is split evenly across substitutes, so every
                       member is understated by roughly that factor. Highest here:
                       {worst:.2f}&times;.</p>
                    {tbl}
                </div>"""

        # ── importance plots + tables ────────────────────────────────────────
        method_titles = {
            "mdi": "MDI — Mean Decrease Impurity",
            "mda": "MDA — Mean Decrease Accuracy",
            "clustered_mdi": "Clustered MDI",
            "clustered_mda": "Clustered MDA",
            "orthogonal": "MDI on PCA-Orthogonal Features (AFML 8.5)",
        }
        method_notes = {
            "mdi": "In-sample impurity reduction, normalized to sum to 1. Diluted across "
            "correlated features: two copies of one predictor each receive about half "
            "its true credit.",
            "mda": "Out-of-sample degradation when a column is permuted, measured over purged "
            "and embargoed folds. Also diluted by substitution, but on the correct side "
            "of the train/test split.",
            "clustered_mdi": "Cluster impurity totals broadcast to every member. Scales with "
            "cluster size, so compare clusters of similar size only.",
            "clustered_mda": "Degradation when every column in a cluster is permuted together. "
            "Size-independent, which makes it the safer of the two clustered "
            "methods when cluster sizes are unequal.",
            "orthogonal": "MDI recomputed on principal components. Importance concentrated in "
            "a few components indicates a low-dimensional signal.",
        }
        importance_sections = []
        for method, df in self.importances.items():
            if method == "sfi":
                continue
            title = method_titles.get(method, method)
            img = self._importance_png_b64(df, title)
            img_html = (
                f'<img src="data:image/png;base64,{img}" alt="{title}">'
                if img
                else '<p class="label">No plottable importances.</p>'
            )
            table_html = (
                df.sort_values("mean", ascending=False)
                .head(self.top_n)
                .to_html(classes="cal-table", border=0, float_format=lambda v: f"{v:.5f}")
            )
            importance_sections.append(
                f'<div class="card"><h2>{title}</h2>'
                f'<p class="label">{method_notes.get(method, "")}</p>{img_html}'
                f'<details><summary class="label">Top {self.top_n} (table)</summary>{table_html}</details></div>'
            )

        # ── SFI, on its own scale ────────────────────────────────────────────
        sfi_section = ""
        if "sfi" in self.importances:
            sfi_df = self.importances["sfi"]
            n_classes = int(self.events["bin"].nunique())
            baseline = None
            if self.scoring_name_ == "neg_log_loss" and n_classes > 1:
                baseline = -float(np.log(n_classes))
            img = self._importance_png_b64(
                sfi_df, "SFI — Single Feature Importance", baseline=baseline
            )
            img_html = (
                f'<img src="data:image/png;base64,{img}" alt="SFI">'
                if img
                else '<p class="label">No plottable scores.</p>'
            )
            base_txt = (
                f"An uninformative feature scores about {baseline:.3f}, the entropy of the "
                f"{n_classes}-class base rate. Scores far below that line are not necessarily "
                "uninformative: a single-feature model that fits its column closely produces "
                "confident probabilities and is penalized heavily by log-loss whenever it is "
                "wrong, so SFI under log-loss partly measures calibration."
                if baseline is not None
                else "Scores are raw cross-validated values on the configured metric."
            )
            table_html = (
                sfi_df.sort_values("mean", ascending=False)
                .head(self.top_n)
                .to_html(classes="cal-table", border=0, float_format=lambda v: f"{v:.5f}")
            )
            sfi_section = f"""
                <div class="card">
                    <h2>SFI &mdash; Single Feature Importance</h2>
                    <p class="label"><span class="important">Different scale.</span> SFI reports the
                       raw out-of-sample score of a model fitted on one feature alone, not a
                       decrement, so its numbers are not comparable with MDI or MDA. {base_txt}</p>
                    {img_html}
                    <details><summary class="label">Top {self.top_n} (table)</summary>{table_html}</details>
                </div>"""

        # ── PCA orthogonality ────────────────────────────────────────────────
        pca_section = ""
        if self.pca_corr_:
            pca_rows = []
            for k, v in self.pca_corr_.items():
                if isinstance(v, tuple):
                    corr, pval = float(v[0]), float(v[1])
                    pca_rows.append(f"<tr><td>{k}</td><td>{corr:+.4f}</td><td>{pval:.3g}</td></tr>")
            pca_section = f"""
                <div class="card">
                    <h2>PCA Orthogonality (AFML 8.6)</h2>
                    <p class="label">Correlation between supervised MDI importance and unsupervised
                       PCA eigen-structure. High positive correlation suggests the model's signal is
                       not pure overfitting.</p>
                    <table><thead><tr><th>Correlation</th><th>Coefficient</th><th>p-value</th></tr></thead>
                    <tbody>{"".join(pca_rows)}</tbody></table>
                </div>"""

        # ── fingerprint ──────────────────────────────────────────────────────
        fingerprint_section = ""
        if self.fingerprint_ is not None:
            try:
                lin, nonlin, pair = self.fingerprint_.get_effects()
                lin_top = pd.Series(lin["norm"]).sort_values(ascending=False).head(self.top_n)
                nl_top = pd.Series(nonlin["norm"]).sort_values(ascending=False).head(self.top_n)
                fp_table = pd.DataFrame({"linear": lin_top, "non_linear": nl_top}).to_html(
                    classes="cal-table", border=0, float_format=lambda v: f"{v:.4f}"
                )
                fingerprint_section = f"""
                    <div class="card">
                        <h2>Model Fingerprint</h2>
                        <p class="label">Normalised linear vs non-linear partial-dependence effects
                           (Li, Turkington & Yazdani 2019).</p>
                        {fp_table}
                    </div>"""
            except Exception as e:
                self.logger.warning(f"Fingerprint section failed: {e}")

        cluster_note = (
            f"<p class='label'>Feature clusters: {len(self.feature_clusters_)}</p>"
            if self.feature_clusters_
            else ""
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Feature Engineering Report — {self._report_title_suffix()}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background:#0f172a; color:#f1f5f9;
                padding:40px; line-height:1.6; }}
        .container {{ max-width:1100px; margin:auto; }}
        h1 {{ color:#38bdf8; border-bottom:2px solid #334155; padding-bottom:10px; }}
        h2 {{ color:#7dd3fc; margin-top:8px; }}
        .card {{ background:#1e293b; border-radius:12px; padding:24px; margin-bottom:24px;
                 border:1px solid #334155; }}
        table {{ width:100%; border-collapse:collapse; margin-top:10px; }}
        th, td {{ text-align:left; padding:10px; border-bottom:1px solid #334155; }}
        th {{ color:#94a3b8; font-weight:600; text-transform:uppercase; font-size:0.78rem; }}
        .label {{ color:#94a3b8; font-size:0.9rem; }}
        .ok {{ color:#22c55e; font-weight:600; }}
        .bad {{ color:#f97316; font-weight:600; }}
        .important {{ color:#fbbf24; font-weight:600; }}
        .metric {{ font-size:1.5rem; font-weight:700; color:#22c55e; }}
        img {{ width:100%; max-width:1050px; border-radius:8px; margin-top:8px; }}
        details summary {{ cursor:pointer; margin-top:10px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Feature Engineering &amp; Importance: {self.symbol}</h1>
        <p class="label">{self._report_subtitle()}</p>
        <p class="label">Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>

        <div class="card">
            <h2>Snapshot</h2>
            <table>
                <tr>
                    <td><span class="label">Features (kept)</span><br>
                        <span class="metric">{len(self._feat_names())}</span></td>
                    <td><span class="label">Training Samples</span><br>
                        <strong>{len(self.events)}</strong></td>
                    <td><span class="label">Model Role</span><br>
                        <strong>{self.config.get("model_role", "N/A").capitalize()}</strong></td>
                </tr>
                <tr>
                    <td><span class="label">OOB Score</span><br>
                        <strong>{self.oob_score_:.4f}</strong></td>
                    <td><span class="label">OOS ({self.scoring_name_})</span><br>
                        <strong>{self.oos_score_:.4f}</strong></td>
                    <td><span class="label">Weighting Scheme</span><br>
                        <strong>{self.best_weighting_scheme or "N/A"}</strong></td>
                </tr>
            </table>
            {cluster_note}
        </div>

        <div class="card">
            <h2>Feature Sources</h2>
            <p class="label">Columns contributed by each feature function and how many survived
               preprocessing (constant/duplicate removal).</p>
            {source_table}
        </div>

        <div class="card">
            <h2>Dropped Features (Preprocessing)</h2>
            {dropped_table}
        </div>

        {status_section}
        {cluster_section}
        {"".join(importance_sections)}
        {sfi_section}
        {subst_section}
        {pca_section}
        {fingerprint_section}
    </div>
</body>
</html>"""
        report_path.write_text(html, encoding="utf-8")
        self.logger.info(f"Generated feature-engineering report: {report_path}")

    def _display_cache_reports(self):
        print("\n" + "=" * 70)
        print("CACHE PERFORMANCE REPORT")
        print("=" * 70)
        monitor = get_cache_monitor()
        if monitor:
            monitor.print_report()

    # ── persistence ──────────────────────────────────────────────────────────────

    def _save_artifacts(self):
        self.file_manager.save_dataframe(self.preprocessed_features, "features")
        self.file_manager.save_dataframe(self.events, "events")
        if self.sample_weight is not None:
            self.file_manager.save_dataframe(self.sample_weight.to_frame("weights"), "weights")

        # Per-method importance tables → CSV next to the report.
        for method, df in self.importances.items():
            df.to_csv(self.file_paths["reports"] / f"importance_{method}.csv")
        if self.catalog_ is not None:
            self.catalog_.to_csv(self.file_paths["reports"] / "feature_catalog.csv", index=False)

        metrics = {
            "feature_count": len(self._feat_names()),
            "feature_sources": self.config["feature_sources"],
            "training_samples": len(self.events),
            "oob_score": self.oob_score_,
            "oos_score": self.oos_score_,
            "scoring": self.scoring_name_,
            "best_weighting_scheme": self.best_weighting_scheme,
            "methods": self.methods,
            "n_clusters": len(self.feature_clusters_) if self.feature_clusters_ else 0,
            "completed_steps": self.completed_steps,
        }
        self.file_manager.save_object(metrics, "metrics")
        self.file_manager.save_object(self.feature_configs, "feature_config")
        self.file_manager.save_object(self._feat_names(), "feature_names")
        if self.preprocessor is not None:
            self.file_manager.save_object(self.preprocessor, "preprocessor")
        if self.target_config is not None:
            self.file_manager.save_object(self.target_config, "target_config")

        self.logger.info(f"Saved all artifacts to {self.file_paths['base_dir']}")

    # ── diagnostics ──────────────────────────────────────────────────────────────

    def check_contamination(self):
        print("\n" + "=" * 70)
        print("DATA CONTAMINATION CHECK")
        print("=" * 70)
        print_contamination_report()

    def get_data_summary(self) -> pd.DataFrame:
        rows = []
        for name, data in [
            ("bar_data", self.bar_data),
            ("features", self.features),
            ("preprocessed_features", self.preprocessed_features),
            ("events", self.events),
            ("sample_weight", self.sample_weight),
        ]:
            if data is None:
                continue
            shape = data.shape if hasattr(data, "shape") else (len(data),)
            rows.append(
                {
                    "Component": name,
                    "Rows": shape[0],
                    "Columns": shape[1] if len(shape) > 1 else 1,
                }
            )
        return pd.DataFrame(rows)
