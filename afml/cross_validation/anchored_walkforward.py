import numpy as np
import pandas as pd
from sklearn.base import clone

from ..cross_validation.cross_validation import PurgedWalkForwardCV


def vin_v_anchored_walkforward(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    estimator,
    n_splits: int = 5,
    inner_val_pct: float = 0.20,
    final_test_pct: float = 0.20,
    pct_embargo: float = 0.01,
    scorer=None,
) -> dict:
    """
    Three-layer V-in-V pipeline with anchored expanding-window walkforward.

    Data is partitioned once, strictly in temporal order:
        [─── Outer Training (60%) ───][─ Inner Val (20%) ─][─ Final Test (20%) ─]

    The outer training set is further split by PurgedWalkForwardCV into
    n_splits expanding windows, each producing its own purged train/test pair.
    The inner validation set is touched only for shortlisting; the final test
    set is opened exactly once after a written commitment to a single config.

    Parameters
    ----------
    X, y, t1 : aligned DataFrame, Series, Series
    estimator  : sklearn-compatible estimator
    n_splits   : number of anchored walkforward windows within outer training
    inner_val_pct, final_test_pct : fractions of total data reserved
    pct_embargo : embargo fraction passed to PurgedWalkForwardCV
    scorer     : callable(y_true, y_pred) -> float, or None (returns raw preds)

    Returns
    -------
    dict with keys:
        outer_scores    – per-window score within outer training
        inner_val_score – score on the inner validation set (shortlisting only)
        final_test_score– score on the final test set (opened once, at the end)
        outer_cv        – the fitted PurgedWalkForwardCV object
    """
    n = len(X)

    # ── 1. Partition the data into three zones ───────────────────────────────
    outer_end   = int(n * (1.0 - inner_val_pct - final_test_pct))
    val_end     = int(n * (1.0 - final_test_pct))

    X_outer,     y_outer,     t1_outer     = X.iloc[:outer_end],  y.iloc[:outer_end],  t1.iloc[:outer_end]
    X_inner_val, y_inner_val, t1_inner_val = X.iloc[outer_end:val_end], y.iloc[outer_end:val_end], t1.iloc[outer_end:val_end]
    X_final,     y_final                   = X.iloc[val_end:],    y.iloc[val_end:]

    # ── 2. Phase 1 — exhaustive search within outer training ─────────────────
    # PurgedWalkForwardCV(expanding_window=True) anchors the start at index 0
    # and grows the training window forward — this IS anchored walkforward.
    outer_cv = PurgedWalkForwardCV(
        n_splits=n_splits,
        t1=t1_outer,
        pct_embargo=pct_embargo,
        expanding_window=True,   # ← anchored, not rolling
    )

    outer_scores = []
    outer_models = []

    for train_idx, test_idx in outer_cv.split(X_outer, y_outer):
        model = clone(estimator)
        model.fit(X_outer.iloc[train_idx], y_outer.iloc[train_idx])
        preds = model.predict(X_outer.iloc[test_idx])

        score = scorer(y_outer.iloc[test_idx], preds) if scorer else None
        outer_scores.append(score)
        outer_models.append(model)

    # At this point the researcher reviews outer_scores, shortlists candidates,
    # and selects a small number of finalist configurations. The inner
    # validation set is NOT touched yet.

    # ── 3. Phase 2 — shortlist on inner validation (sparingly) ───────────────
    # Only finalist models are exposed here. This set must not be used
    # to refine parameters — observation terminates candidate selection.
    finalist_model = outer_models[-1]   # placeholder: researcher selects this
    val_preds      = finalist_model.predict(X_inner_val)
    inner_val_score = scorer(y_inner_val, val_preds) if scorer else None

    # ── 4. Phase 3 — final test (opened exactly once) ────────────────────────
    # The researcher commits in writing to finalist_model before this line.
    # Any adjustment after seeing this result invalidates the test.
    final_preds      = finalist_model.predict(X_final)
    final_test_score = scorer(y_final, final_preds) if scorer else None

    return {
        "outer_scores":     outer_scores,
        "inner_val_score":  inner_val_score,
        "final_test_score": final_test_score,
        "outer_cv":         outer_cv,
    }


# ── Example usage ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    dates  = pd.date_range("2018-01-01", periods=1000, freq="B")
    X_demo = pd.DataFrame(np.random.randn(1000, 5), index=dates)
    y_demo = pd.Series(np.random.randint(0, 2, 1000), index=dates)
    t1_demo = pd.Series(dates + pd.Timedelta(days=5), index=dates)

    results = vin_v_anchored_walkforward(
        X=X_demo, y=y_demo, t1=t1_demo,
        estimator=RandomForestClassifier(n_estimators=50, random_state=42),
        n_splits=5,
        scorer=accuracy_score,
    )

    print("Outer window scores:", results["outer_scores"])
    print("Inner validation:  ", results["inner_val_score"])
    print("Final test score:  ", results["final_test_score"])


