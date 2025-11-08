import hashlib
import inspect
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from multiprocessing import Pool
from os import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from loguru import logger
from numba import get_num_threads, set_num_threads

from ..backtest_statistics.performance_analysis import (
    calculate_performance_metrics,
    get_positions_from_events,
    lower_is_better,
)
from ..cache import memory
from ..labeling.triple_barrier import add_vertical_barrier, triple_barrier_labels
from ..util.volatility import get_period_vol
from .signal_processing import get_entries
from .signals import BaseStrategy


def get_dynamic_seed() -> int:
    return int(time.time())


def save_optimization_results(
    config: "OptimizationConfig",
    best_solution: "GAOptimal",
    pareto_front: list = None,
):
    results_dict = {
        "config": asdict(config),
        "best_solution": asdict(best_solution),
        "pareto_front": pareto_front if pareto_front else [],
    }
    dirpath = Path("GA_logs_triple_barrier")
    dirpath.mkdir(exist_ok=True)
    filename = Path(dirpath, f"{config.run_id}.json")
    logger.info(f"Saving results to {filename.stem}...")
    with open(filename, "w") as f:

        def default_serializer(o):
            if isinstance(o, (np.ndarray, np.generic)):
                return o.tolist()
            if isinstance(o, (datetime, pd.Timestamp)):
                return o.isoformat()
            if isinstance(o, base.Fitness):
                return o.values
            if not isinstance(o, (dict, list, str, int, float, bool, type(None))):
                return str(o)
            return o

        json.dump(results_dict, f, indent=4, default=default_serializer)
    logger.info("Save complete.")


@dataclass
class GAOptimal:
    """Represents the optimal solution found by the genetic algorithm."""

    profit_taking: float
    stop_loss: float
    time_horizon: int
    fitness: Union[float, tuple] = 0.0


@dataclass
class OptimizationConfig:
    """Configuration object to track all optimization parameters."""

    strategy_name: str
    period: str
    bar_size: str
    strategy_params: Dict
    target_vol_params: Dict
    ga_params: Dict
    bounds: Dict
    trading_hours_per_day: int
    objective: Optional[str]
    timestamp: str
    run_id: str


class TripleBarrierEvaluator:
    """Evaluator for triple barrier parameters with performance caching"""

    def __init__(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        target_vol_params: dict = {"days": 1, "lookback": 100},
        target_vol_multiplier: int = 1,
        filter_events: bool = False,
        filter_as_series: bool = False,
        vertical_barrier_zero: bool = True,
        trading_days_per_year: int = 252,
        trading_hours_per_day: int = 24,
        on_crossover: bool = True,
    ):
        """
        Initialize the triple barrier evaluator

        :param strategy: Trading strategy instance (Bollinger, MA, etc.)
        :param data: Market data with ('open', 'high', 'low', 'close') prices
        :param target_vol_params: Parameters to get target volatility series for setting horizontal barriers
            :param days: (int) Number of days
            :param hours: (int) Number of hours
            :param minutes: (int) Number of minutes
            :param lookback: (int) Lookback window
        :param target_vol_multiplier: Multiplier of target volatility
        :param filter_events: Apply CUSUM filter to close prices using target volatility series as threshold
        :param filter_as_series: Whether to pass a float or Series to the CUSUM filter
        :param vertical_barrier_zero: If True, set label to 0 for events that touch vertical barrier, else set to sign of the return
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
        self.target_vol_multiplier = target_vol_multiplier
        self.target_vol_params = target_vol_params
        self.target = (
            get_period_vol(self.close, **target_vol_params).dropna() * target_vol_multiplier
        )
        self.trading_days_per_year = trading_days_per_year
        self.trading_hours_per_day = trading_hours_per_day
        self.filter_events = filter_events
        self.filter_as_series = filter_as_series
        self.vertical_barrier_zero = vertical_barrier_zero
        self.on_crossover = on_crossover
        self.objective = strategy.get_objective()

        # Generate primary signals
        logger.info("Generating primary signals...")
        if filter_events:
            self.threshold = self.target if filter_as_series else self.target.mean()
        else:
            self.threshold = None

        self.primary_signals, self.t_events = get_entries(
            strategy, data, self.threshold, on_crossover
        )

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
        # Create vertical barriers
        vertical_barriers = add_vertical_barrier(self.t_events, self.close, num_bars=time_horizon)

        # Get labeled trades
        events = triple_barrier_labels(
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
        returns = events["ret"]
        positions = get_positions_from_events(self.target.index, events)
        return calculate_performance_metrics(
            returns,
            self.target.index,
            positions,
            self.trading_days_per_year,
            self.trading_hours_per_day,
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


class SingleObjectiveOptimizer:
    """Genetic Algorithm for single objective optimization"""

    def __init__(
        self,
        population_size=50,
        generations=50,
        crossover_rate=0.8,
        mutation_rate=0.2,
        bounds=dict(pt=[0.5, 5], sl=[0.5, 5], time_horizon=[5, 100]),
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.bounds = bounds
        self.setup_deap()

    def setup_deap(self):
        """DEAP framework initialization for single objective"""
        # Clear existing classes to avoid conflicts
        for cls in ["FitnessSingle", "Individual"]:
            if cls in creator.__dict__:
                del creator.__dict__[cls]

        # Create fitness class with single objective
        creator.create("FitnessSingle", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessSingle)

        # Initialize toolbox
        self.toolbox = base.Toolbox()

        # Attribute generators
        self.toolbox.register("attr_pt", random.uniform, self.bounds["pt"][0], self.bounds["pt"][1])
        self.toolbox.register("attr_sl", random.uniform, self.bounds["sl"][0], self.bounds["sl"][1])
        self.toolbox.register(
            "attr_th",
            random.randint,
            self.bounds["time_horizon"][0],
            self.bounds["time_horizon"][1],
        )

        # Individual and population
        self.toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            (self.toolbox.attr_pt, self.toolbox.attr_sl, self.toolbox.attr_th),
            n=1,
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Genetic operators
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", self.custom_mutation)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def custom_mutation(self, individual):
        """Custom mutation respecting parameter bounds"""
        # Mutate profit_taking
        if random.random() < self.mutation_rate:
            individual[0] += random.gauss(0, 0.01)
            individual[0] = np.clip(individual[0], self.bounds["pt"][0], self.bounds["pt"][1])

        # Mutate stop_loss
        if random.random() < self.mutation_rate:
            individual[1] += random.gauss(0, 0.01)
            individual[1] = np.clip(individual[1], self.bounds["sl"][0], self.bounds["sl"][1])

        # Mutate time_horizon
        if random.random() < self.mutation_rate:
            individual[2] += random.randint(-5, 6)
            individual[2] = np.clip(individual[2], *self.bounds["time_horizon"])

        return (individual,)

    @staticmethod
    def init_worker():
        """Helper to isolate setting of threads used in Numba"""
        before = get_num_threads()
        set_num_threads(2)
        logger.debug(f"Threads: {before} → {get_num_threads()}")

    def parallel_evaluation(self, population, evaluator):
        """Evaluate population in parallel"""
        with Pool(processes=cpu_count() // 2, initializer=self.init_worker) as pool:
            results = pool.starmap(
                self.evaluate_individual, [(ind, evaluator) for ind in population]
            )
        return results

    def evaluate_individual(self, individual: list, evaluator: TripleBarrierEvaluator):
        pt, sl, th = individual
        trades = evaluator.evaluate_performance(pt, sl, int(th))
        metrics = evaluator.calculate_strategy_metrics(trades)
        fitness = evaluator.get_objective_score(metrics)
        return (fitness,)

    def optimize(self, evaluator: TripleBarrierEvaluator) -> GAOptimal:
        """Run single-objective optimization"""
        self.toolbox.register("evaluate", self.evaluate_individual, evaluator=evaluator)

        # Create initial population
        population = self.toolbox.population(n=self.population_size)

        # Evaluate initial population
        fitnesses = self.parallel_evaluation(population, evaluator)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Hall of Fame for elitism
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Run algorithm
        population, logbook = algorithms.eaSimple(
            population,
            self.toolbox,
            cxpb=self.crossover_rate,
            mutpb=self.mutation_rate,
            ngen=self.generations,
            stats=stats,
            halloffame=hof,
            verbose=True,
        )

        # Return best individual
        best = hof[0]
        return GAOptimal(
            profit_taking=best[0],
            stop_loss=best[1],
            time_horizon=int(best[2]),
            fitness=best.fitness.values[0],
        )


class ParetoOptimizer:
    """Multi-objective Pareto optimization"""

    def __init__(
        self,
        population_size=50,
        generations=50,
        crossover_rate=0.8,
        mutation_rate=0.2,
        bounds=dict(pt=[0.5, 5], sl=[0.5, 5], time_horizon=[5, 100]),
    ):
        # Set population_size to multiple of 4 to avoid error with `tools.selTournamentDCD`
        self.population_size = population_size - (population_size % 4)
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.bounds = bounds
        self.setup_deap()

    def setup_deap(self):
        """DEAP framework initialization for multi-objective"""
        # Clear existing classes to avoid conflicts
        for cls in ["FitnessMulti", "Individual"]:
            if cls in creator.__dict__:
                del creator.__dict__[cls]

        # Create fitness class with multiple objectives
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        # Initialize toolbox
        self.toolbox = base.Toolbox()

        # Attribute generators
        self.toolbox.register("attr_pt", random.uniform, self.bounds["pt"][0], self.bounds["pt"][1])
        self.toolbox.register("attr_sl", random.uniform, self.bounds["sl"][0], self.bounds["sl"][1])
        self.toolbox.register(
            "attr_th",
            random.randint,
            self.bounds["time_horizon"][0],
            self.bounds["time_horizon"][1],
        )

        # Individual and population
        self.toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            (self.toolbox.attr_pt, self.toolbox.attr_sl, self.toolbox.attr_th),
            n=1,
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Genetic operators
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", self.custom_mutation)
        self.toolbox.register("select", tools.selNSGA2)

    def custom_mutation(self, individual):
        """Custom mutation with bounds checking"""
        if random.random() < self.mutation_rate:
            individual[0] += random.gauss(0, 0.01)
            individual[0] = np.clip(individual[0], self.bounds["pt"][0], self.bounds["pt"][1])

        if random.random() < self.mutation_rate:
            individual[1] += random.gauss(0, 0.01)
            individual[1] = np.clip(individual[1], self.bounds["sl"][0], self.bounds["sl"][1])

        if random.random() < self.mutation_rate:
            individual[2] += random.randint(-5, 6)
            individual[2] = np.clip(individual[2], *self.bounds["time_horizon"])

        return (individual,)

    @staticmethod
    def init_worker():
        """Helper to isolate setting of threads used in Numba"""
        before = get_num_threads()
        set_num_threads(2)
        logger.debug(f"Threads: {before} → {get_num_threads()}")

    def parallel_evaluation(self, population, evaluator):
        """Evaluate population in parallel"""
        with Pool(processes=cpu_count() // 2, initializer=self.init_worker) as pool:
            results = pool.starmap(
                self.evaluate_individual, [(ind, evaluator) for ind in population]
            )
        return results

    def evaluate_individual(self, individual: list, evaluator: TripleBarrierEvaluator):
        """Evaluate individual for multi-objective optimization"""
        pt, sl, th = individual
        trades = evaluator.evaluate_performance(pt, sl, int(th))
        metrics = evaluator.calculate_strategy_metrics(trades)
        return evaluator.get_multi_objectives(metrics)

    def optimize(self, evaluator: TripleBarrierEvaluator) -> List[Dict]:
        """Run Pareto optimization"""
        self.toolbox.register("evaluate", self.evaluate_individual, evaluator=evaluator)

        # Create initial population
        population = self.toolbox.population(n=self.population_size)

        # Evaluate initial population
        fitnesses = self.parallel_evaluation(population, evaluator)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Run NSGA-II algorithm
        population = self.toolbox.select(population, len(population))
        no_improve_count = 0
        best_front_size = 0

        for gen in range(self.generations):
            # Create offspring
            offspring = tools.selTournamentDCD(population, len(population))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # ADDED: Explicit bounds checking after crossover and mutation
            for child in offspring:
                child[0] = np.clip(child[0], self.bounds["pt"][0], self.bounds["pt"][1])
                child[1] = np.clip(child[1], self.bounds["sl"][0], self.bounds["sl"][1])
                child[2] = int(np.clip(child[2], *self.bounds["time_horizon"]))

            # Evaluate new individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.parallel_evaluation(invalid_ind, evaluator)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Combine parents and offspring
            population = self.toolbox.select(population + offspring, self.population_size)

            # Check for convergence
            front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
            if len(front) > best_front_size:
                best_front_size = len(front)
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Early stopping
            if no_improve_count >= 5:
                logger.info(f"Early stopping at generation {gen}")
                break

            # Print progress
            if gen % 10 == 0:
                logger.info(f"Gen {gen}: Pareto front size = {len(front)}")

        # Extract Pareto front
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

        # Prepare results
        results = []
        for ind in pareto_front:
            pt, sl, th = ind
            results.append(
                {
                    "profit_taking": pt,
                    "stop_loss": sl,
                    "time_horizon": int(th),
                    "objectives": ind.fitness.values,
                }
            )

        return results


def select_knee_point(pareto_front: List[Dict]) -> GAOptimal:
    """
    Select the knee point solution from Pareto front

    :param pareto_front: List of Pareto optimal solutions
    :return: Selected knee point solution
    """
    if len(pareto_front) < 3:
        return pareto_front[0] if pareto_front else None

    # Extract objectives
    objectives = np.array([sol["objectives"] for sol in pareto_front])

    # Normalize objectives
    mins = objectives.min(axis=0)
    maxs = objectives.max(axis=0)
    norm_obj = (objectives - mins) / (maxs - mins + 1e-10)

    # Calculate distance to ideal point (1,1,1)
    ideal = np.array([1] * norm_obj.shape[1])
    distances = np.linalg.norm(norm_obj - ideal, axis=1)

    # Find knee point (maximum curvature)
    curvatures = np.zeros(len(distances))
    for i in range(1, len(distances) - 1):
        a = distances[i - 1]
        b = distances[i]
        c = distances[i + 1]
        curvatures[i] = (a - 2 * b + c) / (1 + (a - c) ** 2) ** 1.5

    knee_point = pareto_front[np.argmax(curvatures)]
    return GAOptimal(
        profit_taking=knee_point["profit_taking"],
        stop_loss=knee_point["stop_loss"],
        time_horizon=int(knee_point["time_horizon"]),
        fitness=knee_point["objectives"],
    )


def get_optimal_triple_barrier_labels(
    strategy: BaseStrategy,
    optimizer: ParetoOptimizer,
    data: pd.DataFrame,
    bar_size: str,
    target_vol_params: dict = {"days": 1, "lookback": 100},
    **kwargs,
) -> Tuple[pd.DataFrame, pd.Series]:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_seed = get_dynamic_seed()
    random.seed(run_seed)
    logger.info(f"Running optimization with seed: {run_seed}")
    config = OptimizationConfig(
        strategy_name=strategy.get_strategy_name(),
        bar_size=bar_size,
        period=" to ".join(map(str, data.index[[0, -1]])),
        strategy_params={k: v for k, v in strategy.__dict__.items() if not k.startswith("_")},
        target_vol_params=target_vol_params,
        ga_params={
            "population_size": optimizer.population_size,
            "generations": optimizer.generations,
            "crossover_rate": optimizer.crossover_rate,
            "mutation_rate": optimizer.mutation_rate,
            "seed": run_seed,
        },
        bounds=optimizer.bounds,
        trading_hours_per_day=kwargs.get("trading_hours_per_day", 24),
        objective=strategy.get_objective(),
        timestamp=run_timestamp,
        run_id=f"{bar_size}_{strategy.get_strategy_name()}_{strategy.get_objective()}_{run_timestamp}",
    )

    evaluator = TripleBarrierEvaluator(strategy, data, target_vol_params, **kwargs)
    objectives = list(evaluator.objective_weights[evaluator.objective].keys())
    pareto_front = optimizer.optimize(evaluator)
    best_params = select_knee_point(pareto_front)

    if best_params:
        msg = (
            f"{' '.join(evaluator.objective.title().split('_'))} Fitness Parameters: {objectives}"
            f"\nOptimal solution (Knee Point): \n{best_params}"
        )
        logger.info(msg)
        save_optimization_results(config, best_params, pareto_front)
        events = evaluator.evaluate_performance(
            pt=best_params.profit_taking,
            sl=best_params.stop_loss,
            time_horizon=best_params.time_horizon,
        )
        trade_metrics = evaluator.calculate_strategy_metrics(events)
        return events, trade_metrics
    else:
        logger.error("Optimization failed to find a solution.")
        return pd.DataFrame(), pd.Series()
