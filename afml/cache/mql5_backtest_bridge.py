"""
AFML Backtesting Bridge: Enhanced Python-MQL5 Integration
Based on "Price Action Analysis Toolkit Development (Part 36)"
Optimized for backtesting with AFML cache system integration.
"""

import socket
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

# Import AFML cache components
from afml.cache import get_backtest_cache, get_data_tracker, robust_cacheable


@dataclass
class BacktestConfig:
    """Configuration for backtest runs."""

    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_balance: float = 10000.0
    risk_percent: float = 1.0


@dataclass
class BarData:
    """Single bar of OHLCV data from MQL5."""

    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: float


@dataclass
class BacktestSignal:
    """Trading signal for backtest execution."""

    timestamp: datetime
    symbol: str
    signal_type: str  # "BUY", "SELL", "CLOSE"
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    strategy_name: str
    features: Dict[str, float]  # Feature values that generated this signal


@dataclass
class BacktestTrade:
    """Completed trade from backtest."""

    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    pnl: float
    pnl_percent: float
    exit_reason: str  # "TP", "SL", "SIGNAL", "EOD"
    confidence: float


class AFMLBacktestBridge:
    """
    Enhanced backtesting bridge integrating AFML cache with MQL5 data.

    Key Features:
    - Socket-based communication with MQL5
    - Cached feature engineering and ML predictions
    - Historical data persistence in Parquet format
    - Comprehensive backtest result tracking
    - Data access logging for contamination detection
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 80,
        cache_dir: Optional[Path] = None,
        data_dir: Optional[Path] = None,
    ):
        """
        Initialize AFML Backtest Bridge.

        Args:
            host: Server host for MQL5 connection
            port: Server port
            cache_dir: Directory for cache storage
            data_dir: Directory for historical data (Parquet files)
        """
        from afml.cache import CACHE_DIRS

        self.host = host
        self.port = port
        self.cache_dir = cache_dir or CACHE_DIRS["base"] / "backtest_bridge"
        self.data_dir = data_dir or self.cache_dir / "historical_data"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Communication
        self.server_socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None
        self.is_running = False
        self.lock = threading.Lock()

        # Backtest state
        self.current_config: Optional[BacktestConfig] = None
        self.historical_bars: Dict[str, pd.DataFrame] = {}
        self.trades: List[BacktestTrade] = []
        self.signals: List[BacktestSignal] = []

        # Cache and tracking
        self.backtest_cache = get_backtest_cache()
        self.data_tracker = get_data_tracker()

        logger.info(f"AFML Backtest Bridge initialized: {self.cache_dir}")

    # =========================================================================
    # Historical Data Management (Parquet-based, inspired by Article)
    # =========================================================================

    def bootstrap_historical_data(
        self, symbol: str, days: int = 60, timeframe: str = "M1"
    ) -> pd.DataFrame:
        """
        Bootstrap historical data and save to Parquet.

        Similar to Article's approach but using socket instead of MT5 library.
        """
        parquet_file = self.data_dir / f"{symbol}_{timeframe}_hist.parquet"

        # Check if already bootstrapped
        if parquet_file.exists():
            logger.info(f"Loading existing data from {parquet_file}")
            df = pd.read_parquet(parquet_file)
            return df

        logger.info(f"Bootstrapping {days} days of {symbol} {timeframe} data...")

        # Request data from MQL5
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Send bootstrap request to MQL5
        request = {
            "type": "bootstrap_request",
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }

        self._send_message(request)

        # Receive bars (this would need proper implementation in MQL5 side)
        bars_data = self._receive_bars_batch()

        if bars_data:
            df = pd.DataFrame(bars_data)
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)

            # Save to compressed Parquet
            df.to_parquet(parquet_file, compression="zstd")

            logger.info(f"Bootstrapped {len(df)} bars and saved to {parquet_file}")

            # Track data access
            self.data_tracker.log_access(
                dataset_name=f"{symbol}_{timeframe}",
                start_date=df.index[0],
                end_date=df.index[-1],
                purpose="bootstrap",
                data_shape=df.shape,
            )

            return df

        raise RuntimeError("Failed to bootstrap historical data")

    def append_new_bars(self, symbol: str, timeframe: str = "M1") -> pd.DataFrame:
        """Append only new bars to existing Parquet file."""
        parquet_file = self.data_dir / f"{symbol}_{timeframe}_hist.parquet"

        if not parquet_file.exists():
            raise FileNotFoundError("No historical data found. Run bootstrap first.")

        df = pd.read_parquet(parquet_file)
        last_time = df.index[-1]

        logger.info(f"Last bar in storage: {last_time}")

        # Request new bars from MQL5
        request = {
            "type": "update_request",
            "symbol": symbol,
            "timeframe": timeframe,
            "since": last_time.isoformat(),
        }

        self._send_message(request)
        new_bars = self._receive_bars_batch()

        if new_bars:
            new_df = pd.DataFrame(new_bars)
            new_df["time"] = pd.to_datetime(new_df["time"])
            new_df.set_index("time", inplace=True)

            # Merge and remove duplicates
            merged = pd.concat([df, new_df])
            merged = merged[~merged.index.duplicated(keep="last")]

            # Save back
            merged.to_parquet(parquet_file, compression="zstd")

            logger.info(f"Appended {len(new_bars)} new bars")
            return merged

        logger.info("No new bars to append")
        return df

    # =========================================================================
    # Feature Engineering (Cached)
    # =========================================================================

    @robust_cacheable
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for ML model (cached for speed).

        Based on Article's feature engineering pipeline:
        - z-spike normalization
        - MACD histogram difference
        - RSI (14-period)
        - ATR (14-period)
        - EMA envelope bands
        """
        import pandas_ta as ta

        df = df.copy()

        # Z-spike (20-bar rolling std) - Article's method
        df["returns"] = df["close"].diff()
        df["z_spike"] = df["returns"] / (df["returns"].rolling(20).std() + 1e-9)

        # MACD histogram difference
        df["macd_hist"] = ta.trend.macd_diff(df["close"])

        # RSI (14-period)
        df["rsi"] = ta.momentum.rsi(df["close"], window=14)

        # ATR (14-period) for volatility
        df["atr"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"], window=14
        )

        # EMA envelope bands
        ema_20 = df["close"].ewm(span=20).mean()
        df["ema_lower"] = ema_20 * 0.997
        df["ema_upper"] = ema_20 * 1.003

        # Distance from envelope
        df["env_position"] = (df["close"] - ema_20) / ema_20

        return df.dropna()

    @robust_cacheable
    def generate_ml_signal(
        self, features: pd.DataFrame, model_path: Optional[Path] = None
    ) -> BacktestSignal:
        """
        Generate ML trading signal (cached).

        Args:
            features: DataFrame with engineered features
            model_path: Path to trained model file

        Returns:
            BacktestSignal with prediction and confidence
        """
        import joblib

        if model_path is None:
            model_path = self.cache_dir / "model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)

        # Get last row for prediction
        X = features[
            [
                "z_spike",
                "macd_hist",
                "rsi",
                "atr",
                "ema_lower",
                "ema_upper",
                "env_position",
            ]
        ].iloc[-1:]

        # Predict probabilities
        probs = model.predict_proba(X)[0]

        # Determine signal (using Article's threshold of 0.55)
        p_buy = probs[1]  # Assuming class 1 = BUY
        p_sell = probs[2] if len(probs) > 2 else 0  # Class 2 = SELL

        if p_buy > 0.55:
            signal_type = "BUY"
            confidence = p_buy
        elif p_sell > 0.55:
            signal_type = "SELL"
            confidence = p_sell
        else:
            signal_type = "WAIT"
            confidence = max(p_buy, p_sell)

        # Calculate SL/TP based on ATR (Article's method)
        current_price = features["close"].iloc[-1]
        atr = features["atr"].iloc[-1]

        if signal_type == "BUY":
            stop_loss = current_price - atr
            take_profit = current_price + (2 * atr)
        elif signal_type == "SELL":
            stop_loss = current_price + atr
            take_profit = current_price - (2 * atr)
        else:
            stop_loss = 0.0
            take_profit = 0.0

        return BacktestSignal(
            timestamp=features.index[-1],
            symbol=self.current_config.symbol if self.current_config else "UNKNOWN",
            signal_type=signal_type,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            strategy_name="AFML_ML_Strategy",
            features=X.iloc[0].to_dict(),
        )

    # =========================================================================
    # Backtest Execution
    # =========================================================================

    def run_backtest(
        self,
        config: BacktestConfig,
        model_path: Optional[Path] = None,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Run complete backtest with cached feature engineering.

        Args:
            config: Backtest configuration
            model_path: Path to trained model
            save_results: Whether to save results to cache

        Returns:
            Dict with backtest metrics and trade list
        """
        logger.info(
            f"Starting backtest: {config.symbol} "
            f"{config.start_date} to {config.end_date}"
        )

        self.current_config = config
        self.trades = []
        self.signals = []

        # Load historical data
        symbol_key = f"{config.symbol}_{config.timeframe}"

        if symbol_key not in self.historical_bars:
            parquet_file = self.data_dir / f"{symbol_key}_hist.parquet"

            if not parquet_file.exists():
                raise FileNotFoundError(
                    f"No historical data found. Run bootstrap first."
                )

            df = pd.read_parquet(parquet_file)
            self.historical_bars[symbol_key] = df
        else:
            df = self.historical_bars[symbol_key]

        # Filter to backtest period
        backtest_data = df[
            (df.index >= config.start_date) & (df.index <= config.end_date)
        ].copy()

        logger.info(f"Backtest data: {len(backtest_data)} bars")

        # Track data access for contamination detection
        self.data_tracker.log_access(
            dataset_name=symbol_key,
            start_date=backtest_data.index[0],
            end_date=backtest_data.index[-1],
            purpose="backtest",
            data_shape=backtest_data.shape,
        )

        # Engineer features (CACHED!)
        logger.info("Engineering features (checking cache)...")
        features = self.engineer_features(backtest_data)

        # Simulate trading
        logger.info("Running backtest simulation...")
        results = self._simulate_trading(features, model_path)

        # Save results to backtest cache
        if save_results and self.trades:
            metrics = self._calculate_metrics(results)

            trades_df = pd.DataFrame([asdict(t) for t in self.trades])

            self.backtest_cache.cache_backtest(
                strategy_name="AFML_ML_Strategy",
                parameters=asdict(config),
                data=backtest_data,
                metrics=metrics,
                trades=trades_df,
                equity_curve=results["equity_curve"],
            )

            logger.info(f"Backtest results cached")

        return results

    def _simulate_trading(
        self, features: pd.DataFrame, model_path: Optional[Path]
    ) -> Dict[str, Any]:
        """Simulate trading on historical data."""

        balance = self.current_config.initial_balance
        equity_curve = []
        current_position = None

        # Walk forward through data
        for i in range(100, len(features)):  # Start after warmup period
            current_time = features.index[i]
            window = features.iloc[: i + 1]

            # Generate signal (CACHED!)
            signal = self.generate_ml_signal(window, model_path)
            self.signals.append(signal)

            # Check if we should exit current position
            if current_position:
                current_price = features["close"].iloc[i]

                # Check SL/TP
                if current_position["direction"] == "LONG":
                    if current_price <= current_position["stop_loss"]:
                        # Stop loss hit
                        pnl = (
                            current_position["stop_loss"]
                            - current_position["entry_price"]
                        )
                        balance += pnl * current_position["size"]
                        self._record_trade(
                            current_position,
                            current_time,
                            current_position["stop_loss"],
                            "SL",
                            pnl,
                        )
                        current_position = None
                    elif current_price >= current_position["take_profit"]:
                        # Take profit hit
                        pnl = (
                            current_position["take_profit"]
                            - current_position["entry_price"]
                        )
                        balance += pnl * current_position["size"]
                        self._record_trade(
                            current_position,
                            current_time,
                            current_position["take_profit"],
                            "TP",
                            pnl,
                        )
                        current_position = None

                elif current_position["direction"] == "SHORT":
                    if current_price >= current_position["stop_loss"]:
                        # Stop loss hit
                        pnl = (
                            current_position["entry_price"]
                            - current_position["stop_loss"]
                        )
                        balance += pnl * current_position["size"]
                        self._record_trade(
                            current_position,
                            current_time,
                            current_position["stop_loss"],
                            "SL",
                            pnl,
                        )
                        current_position = None
                    elif current_price <= current_position["take_profit"]:
                        # Take profit hit
                        pnl = (
                            current_position["entry_price"]
                            - current_position["take_profit"]
                        )
                        balance += pnl * current_position["size"]
                        self._record_trade(
                            current_position,
                            current_time,
                            current_position["take_profit"],
                            "TP",
                            pnl,
                        )
                        current_position = None

            # Enter new position if signal
            if signal.signal_type in ["BUY", "SELL"] and current_position is None:
                direction = "LONG" if signal.signal_type == "BUY" else "SHORT"

                # Calculate position size based on risk
                risk_amount = balance * (self.current_config.risk_percent / 100)
                atr = features["atr"].iloc[i]
                position_size = risk_amount / atr if atr > 0 else 0.01

                current_position = {
                    "entry_time": current_time,
                    "direction": direction,
                    "entry_price": signal.entry_price,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "size": position_size,
                    "confidence": signal.confidence,
                }

            # Record equity
            equity_curve.append(
                {"time": current_time, "balance": balance, "equity": balance}
            )

        # Close any remaining position at end
        if current_position:
            final_price = features["close"].iloc[-1]
            if current_position["direction"] == "LONG":
                pnl = final_price - current_position["entry_price"]
            else:
                pnl = current_position["entry_price"] - final_price

            balance += pnl * current_position["size"]
            self._record_trade(
                current_position, features.index[-1], final_price, "EOD", pnl
            )

        return {
            "final_balance": balance,
            "equity_curve": pd.DataFrame(equity_curve).set_index("time"),
            "trades": self.trades,
            "signals": self.signals,
        }

    def _record_trade(
        self,
        position: Dict,
        exit_time: datetime,
        exit_price: float,
        exit_reason: str,
        pnl: float,
    ):
        """Record completed trade."""
        trade = BacktestTrade(
            entry_time=position["entry_time"],
            exit_time=exit_time,
            symbol=self.current_config.symbol,
            direction=position["direction"],
            entry_price=position["entry_price"],
            exit_price=exit_price,
            stop_loss=position["stop_loss"],
            take_profit=position["take_profit"],
            position_size=position["size"],
            pnl=pnl * position["size"],
            pnl_percent=(pnl / position["entry_price"]) * 100,
            exit_reason=exit_reason,
            confidence=position["confidence"],
        )

        self.trades.append(trade)

    def _calculate_metrics(self, results: Dict) -> Dict[str, float]:
        """Calculate backtest performance metrics."""
        trades_df = pd.DataFrame([asdict(t) for t in self.trades])

        if len(trades_df) == 0:
            return {}

        total_pnl = trades_df["pnl"].sum()
        win_rate = (trades_df["pnl"] > 0).sum() / len(trades_df)
        avg_win = (
            trades_df[trades_df["pnl"] > 0]["pnl"].mean()
            if any(trades_df["pnl"] > 0)
            else 0
        )
        avg_loss = (
            trades_df[trades_df["pnl"] < 0]["pnl"].mean()
            if any(trades_df["pnl"] < 0)
            else 0
        )
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Calculate max drawdown
        equity_curve = results["equity_curve"]
        cummax = equity_curve["equity"].cummax()
        drawdown = (equity_curve["equity"] - cummax) / cummax
        max_drawdown = drawdown.min()

        return {
            "total_pnl": total_pnl,
            "total_trades": len(trades_df),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "final_balance": results["final_balance"],
            "return_percent": (
                (results["final_balance"] - self.current_config.initial_balance)
                / self.current_config.initial_balance
            )
            * 100,
        }

    # =========================================================================
    # Socket Communication (simplified helpers)
    # =========================================================================

    def _send_message(self, message: Dict):
        """Send JSON message to MQL5."""
        # Implementation would be similar to your MQL5Bridge
        pass

    def _receive_bars_batch(self) -> List[Dict]:
        """Receive batch of bars from MQL5."""
        # Implementation would receive and parse bars
        pass


# =============================================================================
# Convenience Functions
# =============================================================================


def run_quick_backtest(
    symbol: str,
    days_back: int = 30,
    initial_balance: float = 10000.0,
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Quick backtest runner.

    Usage:
        results = run_quick_backtest("EURUSD", days_back=30)
        print(f"Win Rate: {results['metrics']['win_rate']:.2%}")
    """
    bridge = AFMLBacktestBridge()

    # Bootstrap if needed
    try:
        bridge.bootstrap_historical_data(symbol, days=days_back + 10)
    except Exception as e:
        logger.info(f"Bootstrap skipped: {e}")

    # Configure backtest
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    config = BacktestConfig(
        symbol=symbol,
        timeframe="M1",
        start_date=start_date,
        end_date=end_date,
        initial_balance=initial_balance,
        risk_percent=1.0,
    )

    # Run backtest
    results = bridge.run_backtest(config, model_path=model_path)

    # Calculate metrics
    metrics = bridge._calculate_metrics(results)

    return {
        "config": asdict(config),
        "metrics": metrics,
        "trades": [asdict(t) for t in bridge.trades],
        "equity_curve": results["equity_curve"],
    }


__all__ = [
    "AFMLBacktestBridge",
    "BacktestConfig",
    "BacktestSignal",
    "BacktestTrade",
    "run_quick_backtest",
]
