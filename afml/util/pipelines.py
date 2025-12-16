from loguru import logger
from sklearn.pipeline import Pipeline


class MyPipeline(Pipeline):
    """Allows for a sample_weight in fit method"""

    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + "__sample_weight"] = sample_weight
        return super().fit(X, y, **fit_params)
    

def make_custom_pipeline(pipe_clf):
    """
    Construct a custom pipeline wrapper that supports passing `sample_weight`
    to the `fit` method.

    This function ensures that the provided classifier or pipeline is wrapped
    in a `MyPipeline` object, which extends scikit-learn's `Pipeline` to
    handle `sample_weight` during training. If a bare estimator is provided,
    it is wrapped as the final step in the pipeline. If a scikit-learn
    `Pipeline` is provided, its steps are preserved inside `MyPipeline`.

    Parameters
    ----------
    pipe_clf : sklearn.base.BaseEstimator or sklearn.pipeline.Pipeline
        The classifier or pipeline to wrap. Can be:
        - A single estimator (e.g., LogisticRegression, RandomForestClassifier).
        - A scikit-learn `Pipeline` object containing preprocessing and
          estimator steps.

    Returns
    -------
    MyPipeline
        A `MyPipeline` instance containing the provided estimator or pipeline
        steps, extended to support `sample_weight` in the `fit` method.

    Notes
    -----
    - If `pipe_clf` is not a `Pipeline`, it will be wrapped as a single
      step named `"clf"`.
    - If `pipe_clf` is already a `Pipeline`, its steps are reused directly.
    - This helper is useful when performing model selection or training
      workflows that require weighted samples.
    """
    if not isinstance(pipe_clf, Pipeline):
        return MyPipeline([("clf", pipe_clf)])
    elif isinstance(pipe_clf, Pipeline):
        return MyPipeline(pipe_clf.steps)
    else:
        return pipe_clf
    

def set_pipeline_params(pipeline, **kwargs):
    """
    Safely set one or more parameters for all estimators in a pipeline.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The pipeline to modify.
    **kwargs : dict
        Parameter names and values to set. For example:
        n_jobs=-1, random_state=42

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Modified pipeline.
    """
    current_params = pipeline.get_params()
    for step_name, estimator in pipeline.named_steps.items():
        for param_name, value in kwargs.items():
            param_key = f"{step_name}__{param_name}"
            try:
                if param_key in current_params:
                    pipeline.set_params(**{param_key: value})
            except Exception as e:
                logger.error(
                    "Failed to set %s for step %s: %s",
                    param_name,
                    step_name,
                    str(e),
                )
    return pipeline
