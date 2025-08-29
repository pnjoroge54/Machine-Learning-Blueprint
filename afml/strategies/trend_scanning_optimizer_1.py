class TrendScanningEvaluator:
    """Evaluator for triple barrier parameters with performance caching"""

    def __init__(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        trading_days_per_year: int = 252,
        trading_hours_per_day: int = 24,
        on_crossover: bool = True,
    ):
        """
        Initialize the triple barrier evaluator

        :param strategy: Trading strategy instance (Bollinger, MA, etc.)
        :param data: Market data with ('open', 'high', 'low', 'close') prices
        :param trading_days_per_year: Trading days per year for annualization
        :param trading_hours_per_day: Trading hours per day for annualization
        """
        # Validate input data
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DateTimeIndex")
        if "close" not in data:
            raise ValueError("Data must contain 'close' prices")

        self.strategy = strategy
        self.data = data.copy()
        self.close = data["close"]

        self.trading_days_per_year = trading_days_per_year
        self.trading_hours_per_day = trading_hours_per_day
        self.on_crossover = on_crossover
        self.objective = strategy.get_objective()

        # Generate primary signals
        logger.info("Generating primary signals...")
        self.primary_signals, self.t_events = get_entries(strategy, data, on_crossover)

        # Strategy-specific objective weights
        # NOTE: ParetoOptimizer expects to minimize the last weight.
        self.objective_weights = {
            "mean_reversion": {"sortino_ratio": 0.4, "win_rate": 0.3, "ulcer_index": -0.3},
            "trend_following": {"calmar_ratio": 0.4, "profit_factor": 0.3, "ulcer_index": -0.3},
        }

        # Create data fingerprint for cache invalidation
        self._data_fingerprint = self._create_data_fingerprint()

        # Cache the expensive performance evaluation
        self._cached_evaluate_performance = memory.cache(
            self._evaluate_performance_impl, ignore=["self"]
        )
        self._cached_calculate_strategy_metrics = memory.cache(
            self._calculate_strategy_metrics_impl, ignore=["self"]
        )

    def _create_data_fingerprint(self) -> str:
        """Create a unique fingerprint for the dataset to ensure cache validity"""
        # Combine key data characteristics into a hash
        data_info = {
            "data_shape": self.data.shape,
            "data_start": str(self.data.index[0]),
            "data_end": str(self.data.index[-1]),
            "close_hash": hashlib.md5(self.close.values.tobytes()).hexdigest()[:16],
            "target_hash": hashlib.md5(self.target.values.tobytes()).hexdigest()[:16],
            "strategy_name": self.strategy.get_strategy_name(),
            "strategy_params": str(sorted(self.strategy.__dict__.items())),
            "t_events_count": len(self.t_events),
            "vertical_barrier_zero": self.vertical_barrier_zero,
        }

        fingerprint_str = str(sorted(data_info.items()))
        return hashlib.md5(fingerprint_str.encode()).hexdigest()

    def _evaluate_performance_impl(
        self, pt: float, sl: float, time_horizon: int, data_fingerprint: str
    ) -> pd.DataFrame:
        """
        Internal implementation of performance evaluation (cached)

        Note: data_fingerprint parameter ensures cache invalidation when data changes
        """
        # Get labeled trades
        events = get_bins_from_trend(
            close=self.close,
            target=self.target,
            t_events=self.t_events,
            pt_sl=[pt, sl],
            min_ret=0,
            vertical_barrier_times=vertical_barriers,
            side_prediction=self.primary_signals,
            vertical_barrier_zero=self.vertical_barrier_zero,
            verbose=False,
        )

        return events

    def evaluate_performance(self, pt: float, sl: float, time_horizon: int) -> pd.DataFrame:
        """
        Evaluate barrier parameters with persistent caching

        :param pt: Profit-taking multiple
        :param sl: Stop-loss multiple
        :param time_horizon: Vertical barrier in bars
        :return: Labeled trades DataFrame
        """
        return self._cached_evaluate_performance(pt, sl, time_horizon, self._data_fingerprint)

    def _calculate_strategy_metrics_impl(
        self, events: pd.DataFrame, data_fingerprint: str
    ) -> pd.Series:
        """Compute performance metrics from strategy returns"""
        return calculate_label_metrics(
            self.data, self.target, self.primary_signals, events, self.trading_hours_per_day
        )

    def calculate_strategy_metrics(self, events: pd.DataFrame) -> pd.Series:
        return self._cached_calculate_strategy_metrics(events, self._data_fingerprint)

    def get_objective_score(self, metrics: pd.Series) -> float:
        """Calculate single objective score based on strategy type"""
        if self.objective not in self.objective_weights:
            return metrics.get("sharpe_ratio", 0)

        objective = 0
        for metric, weight in self.objective_weights[self.objective].items():
            score = metrics.get(metric, 0)
            if metric in lower_is_better and np.sign(score) == -1:
                score *= -1
            objective += score * weight
        return objective

    def get_multi_objectives(self, metrics: pd.Series) -> tuple:
        """Return strategy-specific objectives for Pareto optimization"""
        objectives = []
        for metric in self.objective_weights[self.objective].keys():
            score = metrics[metric]
            if metric in lower_is_better and np.sign(score) == -1:
                score *= -1  # Ensure we minimize correctly
            objectives.append(score)
        return tuple(objectives)

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "triple-barrier evaluators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self):
        """
        Get parameters for this estimator.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params()

        for key, value in params.items():
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            setattr(self, key, value)
            valid_params[key] = value

        return self.__init__(**valid_params)
