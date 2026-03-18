import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
  

def train_bagging_ensemble(
    X,
    y,
    base_estimator_unfitted,
    bagging_params,
    sequential_params=None,
    sample_weight=None,
    random_state=None,
):
    """
    Train a bagging ensemble (standard or sequential) and return a pipeline
    containing the fitted ensemble, ready for ONNX export.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data.
    y : array-like, shape (n_samples,)
        Target values.
    base_estimator_unfitted : estimator object
        An unfitted scikit-learn estimator (e.g., a DecisionTreeClassifier)
        that will be used as the base for each bagged model.
    bagging_params : dict
        Parameters for the bagging ensemble. Expected keys:
        - n_estimators : int
        - max_samples : float or int (default=1.0)
        - max_features : float, int, or str (e.g., "sqrt", "log2") (default=1.0)
        - bootstrap_features : bool (default=False)
        - oob_score : bool (default=False)
        - n_jobs : int (default=-1)
    sequential_params : dict or None
        If provided and contains `sequential=True`, sequential bootstrapping is used.
        Must also contain:
        - samples_info_sets : pd.Series
        - price_bars_index : pd.Index or pd.DatetimeIndex
    sample_weight : array-like, optional
        Sample weights passed to the fit method.
    random_state : int, RandomState instance, or None
        Controls the randomness of the bagging process.

    Returns
    -------
    pipeline : Pipeline
        A pipeline with a single step `('bag', fitted_ensemble)`. The fitted_ensemble
        is either a standard BaggingClassifier (if sequential=False) or a standard
        BaggingClassifier populated with the sequentially bootstrapped estimators
        (if sequential=True), ensuring ONNX compatibility.
    """
    # Determine if sequential bagging is requested
    use_sequential = (
        sequential_params is not None and sequential_params.get("sequential", False)
    )

    # Extract common bagging settings
    n_estimators = bagging_params.get("n_estimators", 10)
    max_samples = bagging_params.get("max_samples", 1.0)
    max_features = bagging_params.get("max_features", 1.0)
    bootstrap_features = bagging_params.get("bootstrap_features", False)
    oob_score = bagging_params.get("oob_score", False)
    n_jobs = bagging_params.get("n_jobs", -1)

    if use_sequential:
        from .sequential_bagging import SequentiallyBootstrappedBaggingClassifier

        # Validate required sequential parameters
        required = ["samples_info_sets", "price_bars_index"]
        for key in required:
            if key not in sequential_params:
                raise ValueError(
                    f"sequential_params must contain '{key}' when sequential=True"
                )

        # Instantiate the custom sequentially bootstrapped bagging classifier
        bag = SequentiallyBootstrappedBaggingClassifier(
            samples_info_sets=sequential_params["samples_info_sets"],
            price_bars_index=sequential_params["price_bars_index"],
            estimator=base_estimator_unfitted,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,          # can be int, float, or str
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=False,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=0,
        )
        bag.fit(X, y, sample_weight=sample_weight)

        # --- Convert to standard BaggingClassifier for ONNX compatibility ---
        standard_bag = BaggingClassifier(
            estimator=base_estimator_unfitted,
            n_estimators=len(bag.estimators_),
            max_samples=1.0,          # not used after fitting
            max_features=bag.max_features,
            bootstrap=bag.bootstrap,   # should be True
            bootstrap_features=bag.bootstrap_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        # Attach the fitted components
        standard_bag.estimators_ = bag.estimators_
        standard_bag.estimators_features_ = bag.estimators_features_

        # Copy essential metadata required for prediction
        if hasattr(bag, "classes_"):
            standard_bag.classes_ = bag.classes_
            standard_bag.n_classes_ = bag.n_classes_
        standard_bag.n_features_in_ = bag.n_features_in_

        # Optionally copy OOB attributes (if needed later)
        if hasattr(bag, "oob_score_"):
            standard_bag.oob_score_ = bag.oob_score_
        if hasattr(bag, "oob_decision_function_"):
            standard_bag.oob_decision_function_ = bag.oob_decision_function_
        if hasattr(bag, "oob_prediction_"):
            standard_bag.oob_prediction_ = bag.oob_prediction_

        fitted_ensemble = standard_bag
    else:
        # Standard bagging using sklearn's BaggingClassifier
        bag = BaggingClassifier(
            estimator=base_estimator_unfitted,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=True,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=False,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=0,
        )
        bag.fit(X, y, sample_weight=sample_weight)
        fitted_ensemble = bag

    # Return a pipeline (matching the structure of your existing code)
    return Pipeline([("bag", fitted_ensemble)])