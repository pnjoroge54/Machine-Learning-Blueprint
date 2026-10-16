"""
The screening pipeline: three gates in a fixed order.

The order is not arbitrary and the gates are not interchangeable.

1. **Stationarity.** Runs first because a non-stationary feature invalidates
   every statistic computed downstream, including the permutation null itself,
   which assumes the series is stationary under rotation or resampling.

2. **Entropy.** Runs second because it is O(N), needs no null, and its verdicts
   are one-directional. A feature that does not vary cannot be informative, and
   discarding it here saves the permutation budget for candidates that might be.

3. **Mutual information.** Runs last because it is the only gate that costs
   ``n_perm`` full evaluations, and it is the only one that can endorse rather
   than merely reject.

Each gate is a filter, so the feature set entering gate ``k+1`` is the survivors
of gate ``k``. This has a consequence worth stating plainly: the multiple-testing
correction in gate 3 is applied over the *surviving* family, not the original
one. The earlier gates are not hypothesis tests and do not consume alpha, but
they were chosen by someone who had seen the data. If the entropy threshold is
tuned until the MI screen produces the desired survivors, the reported p-values
are worthless. Set the thresholds once, before looking.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from .entropy_features import entropy_screen
from .multiple_testing import effective_number_of_tests
from .mutual_information import mi_screen
from .stationarity import stationarity_gate

__all__ = ["ScreeningReport", "screen_features"]


@dataclass
class ScreeningReport:
    """Outcome of the three-gate screen.

    :param stationarity: (pd.DataFrame) Gate 1 output over all input features.
    :param entropy: (pd.DataFrame) Gate 2 output over gate 1 survivors.
    :param mutual_information: (pd.DataFrame) Gate 3 output over gate 2 survivors.
    :param selected: (list) Features clearing all three gates.
    :param rejected: (dict) Feature name to the gate that rejected it.
    :param effective_tests: (float) Independent-test equivalent of the screened
        family, from the joint permutation null.
    """

    stationarity: pd.DataFrame
    entropy: pd.DataFrame
    mutual_information: pd.DataFrame
    selected: list[str]
    rejected: dict[str, str] = field(default_factory=dict)
    effective_tests: float = float("nan")

    def summary(self) -> pd.DataFrame:
        """One row per input feature with its outcome at each gate."""
        rows = []
        for name in self.stationarity.index:
            row = {
                "feature": name,
                "verdict": self.stationarity.loc[name, "verdict"],
                "entropy": self.entropy.loc[name, "entropy"]
                if name in self.entropy.index
                else float("nan"),
                "mi": self.mutual_information.loc[name, "statistic"]
                if name in self.mutual_information.index
                else float("nan"),
                "mi_null_mean": self.mutual_information.loc[name, "null_mean"]
                if name in self.mutual_information.index
                else float("nan"),
                "p_adjusted": self.mutual_information.loc[name, "p_westfall_young"]
                if name in self.mutual_information.index
                else float("nan"),
                "outcome": "selected"
                if name in self.selected
                else self.rejected.get(name, "unknown"),
            }
            rows.append(row)
        return pd.DataFrame(rows).set_index("feature")

    def __repr__(self) -> str:
        n_in = len(self.stationarity)
        return (
            f"ScreeningReport({len(self.selected)}/{n_in} selected, "
            f"effective_tests={self.effective_tests:.1f})"
        )


def screen_features(
    features: pd.DataFrame,
    target: pd.Series,
    alpha: float = 0.05,
    min_entropy: float = 0.5,
    max_modal_occupancy: float = 0.5,
    scheme: str = "circular_shift",
    n_perm: int = 500,
    seed: int = 0,
    bars: pd.DataFrame | None = None,
    feature_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    skip_stationarity: bool = False,
    scheme_kwargs: dict[str, float] | None = None,
) -> ScreeningReport:
    """Run the three-gate screen over a feature matrix.

    :param features: (pd.DataFrame) Candidate features, aligned with ``target``.
    :param target: (pd.Series) Forward return or label.
    :param alpha: (float) Level for the stationarity and mutual-information gates.
    :param min_entropy: (float) Normalized-entropy floor.
    :param max_modal_occupancy: (float) Modal-share ceiling.
    :param scheme: (str) Null scheme for the mutual-information gate.
    :param n_perm: (int) Permutation replicates.
    :param seed: (int) Random seed.
    :param bars: (pd.DataFrame) Required for ``scheme="bar_permute"``.
    :param feature_fn: (callable) Required for ``scheme="bar_permute"``.
    :param skip_stationarity: (bool) Bypass gate 1. Use when the features are
        already fractionally differenced, where KPSS is the wrong test; see
        :mod:`afml.screening.stationarity`.
    :param scheme_kwargs: (dict) Extra scheme arguments, e.g. ``mean_block``.
    :return: (ScreeningReport)
    """
    rejected: dict[str, str] = {}

    # -- Gate 1: stationarity -------------------------------------------------
    if skip_stationarity:
        stat = pd.DataFrame({"verdict": "skipped", "passes": True}, index=features.columns)
        stat.index.name = "feature"
        survivors = list(features.columns)
    else:
        stat = stationarity_gate(features, alpha=alpha, n_windows=None)
        survivors = [n for n in stat.index if bool(stat.loc[n, "passes"])]
        for name in stat.index:
            if name not in survivors:
                rejected[name] = f"stationarity:{stat.loc[name, 'verdict']}"

    if not survivors:
        return ScreeningReport(stat, pd.DataFrame(), pd.DataFrame(), [], rejected)

    # -- Gate 2: entropy ------------------------------------------------------
    ent = entropy_screen(
        features[survivors],
        min_entropy=min_entropy,
        max_modal_occupancy=max_modal_occupancy,
    )
    passed_entropy = [n for n in ent.index if bool(ent.loc[n, "passes"])]
    for name in ent.index:
        if name not in passed_entropy:
            rejected[name] = "entropy"

    if not passed_entropy:
        return ScreeningReport(stat, ent, pd.DataFrame(), [], rejected)

    # -- Gate 3: mutual information ------------------------------------------
    mi = mi_screen(
        features[passed_entropy],
        target,
        scheme=scheme,
        n_perm=n_perm,
        seed=seed,
        bars=bars,
        feature_fn=feature_fn,
        alpha=alpha,
        scheme_kwargs=scheme_kwargs,
    )
    selected = [n for n in mi.index if bool(mi.loc[n, "passes"])]
    for name in mi.index:
        if name not in selected:
            rejected[name] = "mutual_information"

    null_matrix = mi.attrs.get("null_matrix")
    eff = effective_number_of_tests(null_matrix) if null_matrix is not None else float("nan")

    return ScreeningReport(stat, ent, mi, selected, rejected, eff)
