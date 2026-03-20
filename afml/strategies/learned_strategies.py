"""
learned_strategy.py
-------------------
Bridges Stage 1 (primary model training) and Stage 2 (secondary / meta-labeling
model training) by wrapping a fitted primary ModelDevelopmentPipeline output as
a BaseStrategy.

Once a primary model has been trained, wrap it in LearnedStrategy and pass it
as the strategy argument to a new ModelDevelopmentPipeline. The secondary pipeline
will call generate_signals() at label-generation time to obtain directional side
predictions, which are then passed to triple_barrier_labels as side_prediction.
This causes get_bins() to enter the meta-labeling path, producing {0, 1} labels
and retaining the 'side' column in events — which in turn tells the pipeline it
is training a secondary model and enables rolling meta-features.

The two-stage workflow
----------------------
    # ── Stage 1: primary model ────────────────────────────────────────────────
    primary_pipeline = ModelDevelopmentPipeline(
        strategy=BollingerBandStrategy(window=20, std=1.5),
        data_config=data_config,
        feature_config=feature_config,
        target_config=target_config,
        label_config=label_config,         # no side_prediction → primary labels
        model_params=model_params,
    )
    primary_pipeline.run()

    # ── Stage 2: secondary model ───────────────────────────────────────────────
    learned = LearnedStrategy.from_pipeline(primary_pipeline)

    secondary_pipeline = ModelDevelopmentPipeline(
        strategy=learned,                  # generates side predictions
        data_config=data_config,           # same data window
        feature_config=feature_config,
        target_config=target_config,
        label_config=secondary_label_config,   # side_prediction provided internally
        model_params=secondary_model_params,
    )
    secondary_pipeline.run()

Constraints
-----------
- LearnedStrategy is designed for primary models only.
  Primary models have no meta-features in their training set (skipped when
  'side' is absent from events). If a secondary model is wrapped here, the
  meta-feature columns will be absent at inference time and prediction will fail.

- The primary pipeline must have been run with save=True, or best_model must
  remain in memory, before constructing LearnedStrategy.

- best_model must include the fitted preprocessor as its first step
  (pipeline version >= 4.0). This is guaranteed when the model was produced
  by the current ModelDevelopmentPipeline which prepends the preprocessor in
  train_model() after the HPO dispatch.
"""

import pandas as pd

from ..strategies.trading_strategies import BaseStrategy


class LearnedStrategy(BaseStrategy):
    """
    Wraps a fitted primary-model Pipeline as a BaseStrategy so it can generate
    side predictions for a secondary model's triple-barrier labeling step.

    generate_signals() applies the same feature engineering used during training,
    then calls best_model.predict(). Because best_model now includes the fitted
    DropConstantFeatures and DropDuplicateFeatures steps as its first component,
    column alignment is handled internally — exactly the same columns that were
    present after preprocessing at training time will be selected at inference
    time, regardless of what the raw feature function produces on new data.
    """

    def __init__(
        self,
        fitted_pipeline,
        feature_config: dict,
        strategy_name: str,
    ):
        """
        Parameters
        ----------
        fitted_pipeline : sklearn Pipeline
            The best_model from a completed primary ModelDevelopmentPipeline.
            Must include the preprocessor as step 0 (pipeline version >= 4.0).
        feature_config : dict
            The same feature_config used to train the primary model.
            Required keys: 'func' (callable), 'params' (dict).
        strategy_name : str
            Human-readable identifier embedded in the secondary pipeline's
            study_name and artifact directory path.
        """
        self.fitted_pipeline = fitted_pipeline
        self.feature_config = feature_config
        self._strategy_name = strategy_name

    @classmethod
    def from_pipeline(cls, pipeline, strategy_name: str = None):
        """
        Convenience constructor that reads directly from a completed pipeline.

        Parameters
        ----------
        pipeline : ModelDevelopmentPipeline
            A pipeline on which .run() has been called successfully.
        strategy_name : str, optional
            Override the auto-generated name. Defaults to
            "<strategy>_<symbol>_Learned".

        Returns
        -------
        LearnedStrategy
        """
        if pipeline.best_model is None:
            raise ValueError(
                "pipeline.best_model is None. Call pipeline.run() before "
                "constructing LearnedStrategy."
            )
        if not getattr(pipeline, 'is_primary', True):
            raise ValueError(
                "LearnedStrategy wraps primary models only. The supplied pipeline "
                "trained a secondary (meta-labeling) model — its meta-features "
                "depend on a prior model's predictions and cannot be reproduced "
                "at inference time without that model being in scope."
            )

        name = strategy_name or (
            f"{pipeline.strategy.get_strategy_name()}"
            f"_{pipeline.symbol}_Learned"
        )
        return cls(
            fitted_pipeline=pipeline.best_model,
            feature_config=pipeline.feature_config,
            strategy_name=name,
        )

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Produce directional side predictions for each bar in `data`.

        The method applies feature_config['func'] to the bar data, drops NaNs,
        and calls fitted_pipeline.predict(). The preprocessor inside the pipeline
        handles column alignment — columns that were constant or duplicate at
        training time are dropped before the classifier sees the data.

        Parameters
        ----------
        data : pd.DataFrame
            Bar data (OHLCV) indexed by timestamp.

        Returns
        -------
        pd.Series
            Integer {-1, 1} series indexed by bar timestamp.
            The label 0 (vertical barrier hit during primary training) is mapped
            to 1 to ensure every bar has a side. The secondary model then decides
            whether to act on each signal — the primary model's only role here is
            to provide direction.
        """
        func = self.feature_config['func']
        features = func(data, **self.feature_config['params']).dropna()
        predictions = self.fitted_pipeline.predict(features)
        signals = pd.Series(predictions, index=features.index, dtype='int8')
        # Map 0 (vertical barrier class) to 1 for triple_barrier_labels compatibility.
        signals = signals.replace(0, 1)
        return signals

    def get_strategy_name(self) -> str:
        return self._strategy_name

    def get_objective(self) -> str:
        # Declaring meta_labeling here causes get_entries() to use the predicted
        # side from generate_signals() as the position direction for labeling.
        return 'meta_labeling'
