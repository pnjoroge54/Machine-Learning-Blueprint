"""
Pydantic specs at the UI boundary.

Each spec resolves to the exact dicts the afml pipelines accept through
``to_pipeline_kwargs()``, so the pipelines do not change. Two rules come straight
from the codebase:

* Two weight fields, never one. ``train_weight_scheme`` (weights for the estimator
  fit) and ``score_weight_source`` (return-attribution weights from ``events['w']``,
  always used for CV scoring) are separate fields with separate widgets.
* The model role is derived, not chosen. Primary versus secondary follows from
  whether ``side`` is in ``events.columns``; no spec field can disagree with it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import registry

Stage = Literal["bars", "labels", "features", "screening", "importance", "model"]
Purpose = Literal["research", "holdout"]

STAGES: List[str] = ["bars", "labels", "features", "screening", "importance", "model"]

# Sections each stage needs. Sections a stage does not use are dropped from the
# canonical config, so they cannot change its hash.
STAGE_SECTIONS: Dict[str, List[str]] = {
    "bars": ["data"],
    "labels": ["data", "labels"],
    "features": ["data", "features"],
    "screening": ["data", "labels", "features", "screening"],
    "importance": ["data", "labels", "features", "importance"],
    "model": ["data", "labels", "features", "model"],
}


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class CallSpec(_Spec):
    """A registered callable: a stable name plus its params."""

    name: str
    params: Dict[str, Any] = Field(default_factory=dict)

    def resolved(self, kind: str) -> "CallSpec":
        """Fill defaults and coerce types from the registered signature."""
        fields = {f.name: f for f in registry.param_fields(kind, self.name)}
        unknown = set(self.params) - set(fields)
        if unknown:
            raise ValueError(f"{kind}:{self.name} has no params {sorted(unknown)}")
        params: Dict[str, Any] = {}
        for name, f in fields.items():
            if name in self.params:
                value = self.params[name]
            elif f.required:
                raise ValueError(f"{kind}:{self.name} requires param {name!r}")
            else:
                value = f.default
            if f.widget == "float" and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            if isinstance(value, tuple):
                value = list(value)
            params[name] = value
        return CallSpec(name=registry.get(kind, self.name).name, params=params)


class DataSpec(_Spec):
    symbol: str
    source: str = "mt5_ticks"
    source_params: Dict[str, Any] = Field(default_factory=dict)
    account: Optional[str] = None
    bar_type: Literal["tick", "time", "volume", "dollar"] = "tick"
    bar_size: Union[int, str] = "M5"
    price: Literal["bid", "ask", "mid_price", "bid_ask"] = "mid_price"
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def _dates_ordered(self):
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self

    @property
    def bar_spec(self) -> str:
        return f"{self.bar_type}:{self.bar_size}:{self.price}"


class LabelSpec(_Spec):
    strategy: CallSpec = Field(default_factory=lambda: CallSpec(name="bollinger"))
    # A completed primary run used as the strategy of a secondary run.
    from_run: Optional[str] = None
    target: CallSpec = Field(default_factory=lambda: CallSpec(name="daily_vol"))
    profit_target: float = Field(1.0, gt=0)
    stop_loss: float = Field(1.0, gt=0)
    max_holding_period: Dict[str, int] = Field(default_factory=lambda: {"days": 1})
    min_ret: float = Field(0.0, ge=0)
    vertical_barrier_zero: bool = True
    filter_events: bool = True
    # Weights for the estimator fit. Never used for scoring.
    train_weight_scheme: Literal["uniqueness", "time_decay", "none"] = "uniqueness"
    time_decay: float = Field(1.0, ge=-1, le=1)
    # Weights for CV scoring: always return attribution, events["w"].
    score_weight_source: Literal["return_attribution"] = "return_attribution"

    @field_validator("max_holding_period")
    @classmethod
    def _holding_keys(cls, v):
        allowed = {"days", "hours", "minutes", "seconds", "num_bars"}
        bad = set(v) - allowed
        if bad:
            raise ValueError(f"max_holding_period keys must be in {sorted(allowed)}")
        return v


class FeatureStep(CallSpec):
    prefix: Optional[str] = None


class FeatureSetSpec(_Spec):
    steps: List[FeatureStep] = Field(min_length=1)
    # Set when a saved feature set keeps only some columns.
    selected_columns: Optional[List[str]] = None


class ScreeningSpec(_Spec):
    """
    The three-gate screen of ``afml.screening.screen_features``: stationarity, entropy,
    then mutual information against a permutation null with Westfall-Young adjustment.

    The null resamples the target with a scheme that keeps its serial dependence;
    the naive shuffle is not offered because overlapping labels are autocorrelated.
    Set the thresholds once, before looking: tuning them until the screen selects
    what you hoped for makes its p-values worthless.
    """

    target: Literal["label", "event_return"] = "label"
    scheme: Literal["circular_shift", "stationary_bootstrap"] = "circular_shift"
    n_perm: int = Field(500, ge=50, le=10_000)
    alpha: float = Field(0.05, gt=0, lt=0.5)
    min_entropy: float = Field(0.5, ge=0, le=1)
    max_modal_occupancy: float = Field(0.5, gt=0, le=1)
    # KPSS is the wrong test for fractionally differenced features; skip gate 1 for them
    skip_stationarity: bool = False


class ImportanceSpec(_Spec):
    methods: List[Literal["mdi", "mda", "sfi"]] = Field(default_factory=lambda: ["mdi", "mda"])
    n_splits: int = Field(5, ge=2, le=20)
    pct_embargo: float = Field(0.01, ge=0, lt=0.5)
    estimator: CallSpec = Field(default_factory=lambda: CallSpec(name="random_forest"))
    scoring: Literal["neg_log_loss", "accuracy"] = "neg_log_loss"
    # Split mode: importance runs on the first part of the research window, model
    # development on the rest, so model CV scores do not reuse the folds that chose
    # the features.
    split_mode: bool = False
    importance_fraction: float = Field(0.5, gt=0.1, lt=0.9)

    @field_validator("methods")
    @classmethod
    def _non_empty(cls, v):
        if not v:
            raise ValueError("choose at least one importance method")
        return sorted(set(v), key=v.index)


def _check_search_space(space: Dict[str, Any]) -> Dict[str, Any]:
    """
    A search space is a dict of estimator params. Each value is one of:

    * a list: categorical choices
    * ``{"int": [low, high]}``: integers, both ends included
    * ``{"float": [low, high], "log": false}``: floats, log scale when ``log`` is true
    """
    for name, dim in space.items():
        if isinstance(dim, list):
            if not dim:
                raise ValueError(f"search space {name!r}: empty choice list")
            continue
        if not isinstance(dim, dict):
            raise ValueError(f"search space {name!r}: use a list or an int/float range")
        kinds = {"int", "float"} & set(dim)
        if len(kinds) != 1 or set(dim) - {"int", "float", "log"}:
            raise ValueError(f"search space {name!r}: give exactly one of 'int' or 'float' (plus 'log')")
        kind = kinds.pop()
        bounds = dim[kind]
        if not (isinstance(bounds, list) and len(bounds) == 2 and bounds[0] < bounds[1]):
            raise ValueError(f"search space {name!r}: {kind} range must be [low, high] with low < high")
        if dim.get("log") and bounds[0] <= 0:
            raise ValueError(f"search space {name!r}: a log range needs low > 0")
    return space


class ModelSpec(_Spec):
    estimator: CallSpec = Field(default_factory=lambda: CallSpec(name="random_forest"))
    # "optuna": a TPE study through afml's optimize_trading_model, one ledger trial per
    # Optuna trial. "grid": purged-CV grid search through clf_hyper_fit.
    search: Literal["grid", "optuna"] = "grid"
    param_grid: Dict[str, List[Any]] = Field(default_factory=dict)
    search_space: Dict[str, Any] = Field(default_factory=dict)
    n_trials: int = Field(50, ge=1, le=5000)
    timeout_s: int = Field(3600, ge=10)
    pruner: Literal["hyperband", "median"] = "hyperband"
    metric: Literal["neg_log_loss", "f1"] = "neg_log_loss"
    cv_splits: int = Field(5, ge=2, le=20)
    pct_embargo: float = Field(0.01, ge=0, lt=0.5)
    bagging_n_estimators: int = Field(0, ge=0)
    bagging_max_samples: float = Field(1.0, gt=0, le=1)
    bagging_max_features: float = Field(1.0, gt=0, le=1)
    rnd_search_iter: int = Field(0, ge=0)
    # Use the later part of the window when the importance run used split mode.
    window_fraction: Optional[List[float]] = None

    @field_validator("search_space")
    @classmethod
    def _valid_space(cls, v):
        return _check_search_space(v)

    @model_validator(mode="after")
    def _space_for_optuna(self):
        # The space must be in the config so the hash covers what the study searched
        if self.search == "optuna" and not self.search_space:
            raise ValueError("an Optuna search needs a search_space")
        return self


class RunSpec(_Spec):
    stage: Stage
    purpose: Purpose = "research"
    data: DataSpec
    labels: Optional[LabelSpec] = None
    features: Optional[FeatureSetSpec] = None
    screening: Optional[ScreeningSpec] = None
    importance: Optional[ImportanceSpec] = None
    model: Optional[ModelSpec] = None
    seed: int = 42
    # Display-only fields, excluded from the config hash.
    notes: str = ""
    family_label: Optional[str] = None

    @model_validator(mode="after")
    def _sections_present(self):
        missing = [s for s in STAGE_SECTIONS[self.stage] if getattr(self, s) is None]
        if missing:
            raise ValueError(f"stage {self.stage!r} needs sections {missing}")
        return self

    # ------------------------------------------------------------------ identity

    def resolved(self) -> "RunSpec":
        """Copy with unused sections dropped and registry params filled in."""
        keep = set(STAGE_SECTIONS[self.stage])
        out = self.model_copy(deep=True)
        for section in ("labels", "features", "screening", "importance", "model"):
            if section not in keep:
                object.__setattr__(out, section, None)
        if out.labels is not None:
            if out.labels.from_run is None:
                out.labels.strategy = out.labels.strategy.resolved("strategy")
            out.labels.target = out.labels.target.resolved("target")
        if out.features is not None:
            steps = []
            for step in out.features.steps:
                r = step.resolved("feature")
                steps.append(FeatureStep(name=r.name, params=r.params, prefix=step.prefix))
            out.features.steps = steps
        if out.importance is not None:
            out.importance.estimator = out.importance.estimator.resolved("estimator")
        if out.model is not None:
            out.model.estimator = out.model.estimator.resolved("estimator")
        return out

    def canonical_dict(self) -> Dict[str, Any]:
        d = self.resolved().model_dump(mode="json", exclude={"notes", "family_label"})
        return {k: v for k, v in d.items() if v is not None}

    def canonical_json(self) -> str:
        return canonical_json(self.canonical_dict())

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def family_key(self) -> Dict[str, Any]:
        """The research question this run belongs to; N is counted per family."""
        d = self.data
        key: Dict[str, Any] = {
            "symbol": d.symbol,
            "source": d.source,
            "bar_spec": d.bar_spec,
        }
        labels = self.resolved().labels
        if labels is None:
            key["label_spec"] = None
            key["strategy"] = None
        else:
            key["strategy"] = (
                {"from_run": labels.from_run}
                if labels.from_run
                else labels.strategy.model_dump(mode="json")
            )
            key["label_spec"] = labels.model_dump(
                mode="json",
                include={
                    "target",
                    "profit_target",
                    "stop_loss",
                    "max_holding_period",
                    "min_ret",
                    "vertical_barrier_zero",
                    "filter_events",
                },
            )
        return key

    def family_id(self) -> str:
        return "f-" + hashlib.sha256(canonical_json(self.family_key()).encode()).hexdigest()[:12]

    def family_name(self) -> str:
        if self.family_label:
            return self.family_label
        d = self.data
        strat = "unlabeled"
        if self.labels is not None and "labels" in STAGE_SECTIONS[self.stage]:
            strat = self.labels.from_run or self.labels.strategy.name
        return f"{d.symbol} · {d.bar_spec} · {strat}"

    # ------------------------------------------------------------------ pipelines

    def to_pipeline_kwargs(self) -> Dict[str, Any]:
        """
        Resolve to the dicts the existing afml pipelines take
        (in the keyword shape of the older ``develop_production_model``): ``data_config``,
        ``feature_config``, ``label_config``, ``model_params`` and
        ``sample_weight_params``. Registered names become live callables here, and
        only here.
        """
        from .adapters import pipelines

        return pipelines.to_pipeline_kwargs(self.resolved())


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
