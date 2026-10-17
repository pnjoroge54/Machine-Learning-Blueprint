"""
global_liquidity_pipeline.py
----------------------------
A clean, reusable module based on the concepts from
"Mining Central Bank Balance Sheet Data to Get a Picture of Global Liquidity"
(MQL5 article #18355).

Classes
-------
GlobalLiquidityMiner
    Fetch & normalise central bank balance sheet data.
CompositeLiquidityIndex
    Build a weighted composite liquidity index and derived features.
ForexLiquidityForecaster
    Prepare ML features, train forecasts, and generate predictions.
"""


import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Optional dependencies – gracefully degrade if not installed
try:
    from fredapi import Fred
except ImportError:
    Fred = None

try:
    import yfinance as yf
except ImportError:
    yf = None


# ---------------------------------------------------------------------------
# 1. GlobalLiquidityMiner – data collection & standardisation
# ---------------------------------------------------------------------------
class GlobalLiquidityMiner:
    """
    Collects and normalises central bank balance sheet data from
    multiple sources (FRED, yfinance proxies, etc.).
    """

    def __init__(
        self,
        fred_api_key: str | None = None,
        start_date: str = "2000-01-01",
        end_date: str | None = None,
    ):
        """
        Parameters
        ----------
        fred_api_key : str, optional
            Your FRED API key (obtain from https://fred.stlouisfed.org/docs/api/api_key.html).
            If not provided, FRED data will be skipped.
        start_date : str
            Start date in 'YYYY-MM-DD' format (default '2000-01-01').
        end_date : str, optional
            End date in 'YYYY-MM-DD' format. If None, uses today's date.
        """
        if Fred is not None and fred_api_key:
            self.fred = Fred(api_key=fred_api_key)
        else:
            self.fred = None
            if Fred is None:
                logger.warning("fredapi not installed. Install with: pip install fredapi")
            if not fred_api_key:
                logger.warning("No FRED API key provided. FRED data will be unavailable.")

        self.start_date = start_date
        self.end_date = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
        self.data_cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # 1.1 Fetch Federal Reserve data (FRED)
    # ------------------------------------------------------------------
    def fetch_fed_data(self) -> pd.DataFrame:
        """Retrieve Fed balance sheet data from FRED."""
        if self.fred is None:
            logger.error("FRED client not initialised. Provide a valid FRED API key.")
            return pd.DataFrame()

        try:
            fed_total = self.fred.get_series("WALCL", start=self.start_date, end=self.end_date)
            fed_securities = self.fred.get_series(
                "WSHOSHO", start=self.start_date, end=self.end_date
            )
            fed_loans = self.fred.get_series("WLRRAL", start=self.start_date, end=self.end_date)

            df = pd.DataFrame(
                {
                    "date": pd.to_datetime(fed_total.index),
                    "total_assets": fed_total.values,
                    "securities_held": fed_securities.reindex(
                        fed_total.index, method="ffill"
                    ).values,
                    "loans_and_repos": fed_loans.reindex(fed_total.index, method="ffill").values,
                }
            )
            df["currency"] = "USD"
            df["assets_growth_rate"] = df["total_assets"].pct_change(periods=52)
            df["securities_share"] = df["securities_held"] / df["total_assets"]
            df.set_index("date", inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error loading Fed data: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 1.2 Fetch ECB proxy data (EURUSD + Euro Stoxx 50)
    # ------------------------------------------------------------------
    def fetch_ecb_proxy_data(self) -> pd.DataFrame:
        """
        Build a proxy ECB balance sheet using market indicators.
        Uses EURUSD and Euro Stoxx 50 if yfinance is available.
        """
        if yf is None:
            logger.warning("yfinance not installed. ECB proxy data unavailable.")
            return pd.DataFrame()

        try:
            eur_usd = yf.download(
                "EURUSD=X", start=self.start_date, end=self.end_date, progress=False
            )
            stoxx = yf.download(
                "^STOXX50E", start=self.start_date, end=self.end_date, progress=False
            )

            proxy = pd.DataFrame(index=eur_usd.index)
            proxy["eur_strength"] = 1.0 / eur_usd["Close"]
            proxy["equity_liquidity"] = stoxx["Close"] / stoxx["Close"].rolling(252).mean()
            proxy["synthetic_balance"] = (
                proxy["eur_strength"].rolling(30).mean() * proxy["equity_liquidity"] * 1e6
            )
            return proxy
        except Exception as e:
            logger.error(f"Error loading ECB proxy data: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 1.3 Fetch BOJ proxy data (USDJPY + Nikkei 225)
    # ------------------------------------------------------------------
    def fetch_boj_proxy_data(self) -> pd.DataFrame:
        """Build a proxy BOJ balance sheet using USDJPY and Nikkei 225."""
        if yf is None:
            logger.warning("yfinance not installed. BOJ proxy data unavailable.")
            return pd.DataFrame()

        try:
            usdjpy = yf.download(
                "USDJPY=X", start=self.start_date, end=self.end_date, progress=False
            )
            nikkei = yf.download("^N225", start=self.start_date, end=self.end_date, progress=False)

            proxy = pd.DataFrame(index=usdjpy.index)
            proxy["jpy_strength"] = 1.0 / usdjpy["Close"]
            proxy["equity_liquidity"] = nikkei["Close"] / nikkei["Close"].rolling(252).mean()
            proxy["synthetic_balance"] = (
                proxy["jpy_strength"].rolling(30).mean() * proxy["equity_liquidity"] * 1e6
            )
            return proxy
        except Exception as e:
            logger.error(f"Error loading BOJ proxy data: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 1.4 Fetch PBoC proxy data (Chinese bonds, simplified)
    # ------------------------------------------------------------------
    def fetch_pboc_proxy_data(self) -> pd.DataFrame:
        """
        Build a minimal PBoC proxy using China bond ETF (CBON).
        In a production system you could add additional China-specific
        indicators (e.g., SHCOMP, CNH).
        """
        if yf is None:
            logger.warning("yfinance not installed. PBoC proxy data unavailable.")
            return pd.DataFrame()

        try:
            cbon = yf.download("CBON", start=self.start_date, end=self.end_date, progress=False)
            proxy = pd.DataFrame(index=cbon.index)
            proxy["bond_liquidity"] = cbon["Close"] / cbon["Close"].rolling(252).mean()
            proxy["synthetic_balance"] = proxy["bond_liquidity"] * 1e7
            return proxy
        except Exception as e:
            logger.error(f"Error loading PBoC proxy data: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 1.5 Main method: fetch all central bank data
    # ------------------------------------------------------------------
    def fetch_central_bank_balance_sheets(self) -> dict[str, pd.DataFrame]:
        """
        Populate self.data_cache['balance_sheets'] with a DataFrame for each
        central bank (FED, ECB, BOJ, PBOC) and return the dictionary.
        """
        balance_sheets = {}

        # --- Federal Reserve (real data) ---
        fed_df = self.fetch_fed_data()
        if not fed_df.empty:
            balance_sheets["FED"] = fed_df
            logger.info(f"Loaded {len(fed_df)} Federal Reserve data entries.")

        # --- ECB (proxy) ---
        ecb_df = self.fetch_ecb_proxy_data()
        if not ecb_df.empty:
            balance_sheets["ECB"] = ecb_df
            logger.info(f"Loaded {len(ecb_df)} ECB proxy entries.")

        # --- BOJ (proxy) ---
        boj_df = self.fetch_boj_proxy_data()
        if not boj_df.empty:
            balance_sheets["BOJ"] = boj_df
            logger.info(f"Loaded {len(boj_df)} BOJ proxy entries.")

        # --- PBoC (proxy) ---
        pboc_df = self.fetch_pboc_proxy_data()
        if not pboc_df.empty:
            balance_sheets["PBOC"] = pboc_df
            logger.info(f"Loaded {len(pboc_df)} PBoC proxy entries.")

        self.data_cache["balance_sheets"] = balance_sheets
        return balance_sheets


# ---------------------------------------------------------------------------
# 2. CompositeLiquidityIndex – weighted index & derived features
# ---------------------------------------------------------------------------
class CompositeLiquidityIndex:
    """
    Creates a weighted composite liquidity index from central bank data
    and enriches it with short-term, long-term, acceleration, and
    volatility features.
    """

    DEFAULT_WEIGHTS = {
        "FED_balance": 0.35,
        "ECB_balance": 0.25,
        "BOJ_balance": 0.15,
        "PBOC_balance": 0.20,
    }

    def __init__(self, weights: dict[str, float] | None = None):
        """
        Parameters
        ----------
        weights : dict, optional
            Mapping from bank_code + '_balance' to weight.
            If None, DEFAULT_WEIGHTS is used.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS

    # ------------------------------------------------------------------
    # 2.1 Main index calculation
    # ------------------------------------------------------------------
    def calculate_liquidity_index(
        self, miner: GlobalLiquidityMiner, min_periods: int = 252
    ) -> pd.DataFrame:
        """
        Compute the composite liquidity index.

        Parameters
        ----------
        miner : GlobalLiquidityMiner
            An instance with populated data_cache['balance_sheets'].
        min_periods : int
            Minimum number of observations for rolling normalisation (default 252).

        Returns
        -------
        pd.DataFrame
            DataFrame with a single column 'liquidity_index'.
        """
        balance_sheets = miner.data_cache.get("balance_sheets", {})
        if not balance_sheets:
            logger.warning(
                "No balance sheet data found. Run fetch_central_bank_balance_sheets first."
            )
            return pd.DataFrame()

        all_series = {}
        for bank, df in balance_sheets.items():
            series_name = f"{bank}_balance"
            # Use the 'synthetic_balance' column for proxy banks, 'total_assets' for FED
            if "synthetic_balance" in df.columns:
                raw = df["synthetic_balance"]
            elif "total_assets" in df.columns:
                raw = df["total_assets"]
            else:
                continue

            # Z-score normalisation
            roll_mean = raw.rolling(window=min_periods, min_periods=1).mean()
            roll_std = raw.rolling(window=min_periods, min_periods=1).std().replace(0, np.nan)
            normalized = (raw - roll_mean) / roll_std
            all_series[series_name] = normalized

        if not all_series:
            return pd.DataFrame()

        combined = pd.DataFrame(all_series).ffill()

        # Apply weights
        weight_series = pd.Series(self.weights)
        available_weights = weight_series[weight_series.index.isin(combined.columns)]
        if available_weights.empty:
            logger.error("No matching weight columns. Check your weight keys.")
            return pd.DataFrame()

        # Normalise weights to sum to 1
        available_weights /= available_weights.sum()
        liquidity_index = combined[available_weights.index].dot(available_weights)

        return pd.DataFrame({"liquidity_index": liquidity_index}, index=combined.index)

    # ------------------------------------------------------------------
    # 2.2 Derived features
    # ------------------------------------------------------------------
    def enhance_liquidity_index(self, base_index: pd.Series) -> pd.DataFrame:
        """
        Create a DataFrame of advanced liquidity indicators.

        Parameters
        ----------
        base_index : pd.Series
            The composite liquidity index (output of calculate_liquidity_index).

        Returns
        -------
        pd.DataFrame
            Columns: base_liquidity_index, short_term_liquidity (30d),
            long_term_trend (252d), liquidity_acceleration (diff of diff),
            liquidity_volatility (60d std).
        """
        enhanced = pd.DataFrame(index=base_index.index)
        enhanced["base_liquidity_index"] = base_index
        enhanced["short_term_liquidity"] = base_index.rolling(window=30).mean()
        enhanced["long_term_trend"] = base_index.rolling(window=252).mean()
        enhanced["liquidity_acceleration"] = base_index.diff().diff()
        enhanced["liquidity_volatility"] = base_index.rolling(window=60).std()
        return enhanced


# ---------------------------------------------------------------------------
# 3. ForexLiquidityForecaster – ML feature engineering & forecasting
# ---------------------------------------------------------------------------
class ForexLiquidityForecaster:
    """
    Combines technical indicators and liquidity features to train a
    RandomForestRegressor for multiple return horizons.
    """

    def __init__(
        self,
        liquidity_miner: GlobalLiquidityMiner,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        liquidity_miner : GlobalLiquidityMiner
            Miner that holds the balance sheet data cache.
        n_estimators : int
            Number of trees in the RandomForest.
        max_depth : int
            Maximum depth of each tree.
        random_state : int
            Seed for reproducibility.
        """
        self.liquidity_miner = liquidity_miner
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.models: dict[str, dict] = {}  # {symbol: {horizon: {model, scaler, r2}}}
        self.scalers: dict[str, StandardScaler] = {}

    # ------------------------------------------------------------------
    # 3.1 Technical indicator helpers
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0.0).rolling(window=period).mean()
        rs = gain / loss
        return 100.0 - (100.0 / (1.0 + rs))

    # ------------------------------------------------------------------
    # 3.2 Feature preparation
    # ------------------------------------------------------------------
    def prepare_features(
        self,
        symbol: str,
        historical_data: pd.DataFrame,
        feature_df: pd.DataFrame,
        include_liquidity: bool = True,
    ) -> pd.DataFrame:
        """
        Build a feature DataFrame for a single symbol.

        Parameters
        ----------
        symbol : str
            Instrument code (e.g., 'EURUSD').
        historical_data : pd.DataFrame
            Must contain at least a 'close' column.
        feature_df : pd.DataFrame
            Feature matrix from external functions.
        include_liquidity : bool
            Whether to join central bank balance data.

        Returns
        -------
        pd.DataFrame
            Feature matrix with no NaN values.
        """
        df = historical_data.copy()
        required_cols = {"close"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"historical_data must contain column 'close'. Found {df.columns}")

        # Add features
        df = df.join(feature_df, how="left")

        # Attach liquidity data
        if include_liquidity:
            balance_sheets = self.liquidity_miner.data_cache.get("balance_sheets", {})
            for bank, bs_data in balance_sheets.items():
                col_name = f"{bank}_balance"
                # Use first available liquidity column
                if "synthetic_balance" in bs_data.columns:
                    series = bs_data["synthetic_balance"]
                elif "total_assets" in bs_data.columns:
                    series = bs_data["total_assets"]
                else:
                    continue
                temp = series.rename(col_name)
                df = df.join(temp, how="left")
                df[col_name].fillna(method="ffill", inplace=True)

        return df.dropna()

    # ------------------------------------------------------------------
    # 3.3 Model training
    # ------------------------------------------------------------------
    def build_prediction_model(
        self,
        symbol: str,
        feature_df: pd.DataFrame,
        horizons: tuple = (1, 5, 10, 20),
        train_ratio: float = 0.8,
    ) -> dict[str, dict]:
        """
        Train a RandomForest for each forward-return horizon.

        Parameters
        ----------
        symbol : str
            Instrument identifier.
        feature_df : pd.DataFrame
            Feature matrix as returned by prepare_features().
        horizons : tuple of int
            Forecast horizons in trading days (e.g., 1, 5, 10, 20).
        train_ratio : float
            Fraction of data used for training.

        Returns
        -------
        dict
            {horizon_label: {'model', 'scaler', 'r2'}}.
        """
        # Build target dictionary
        targets = {}
        for h in horizons:
            label = f"return_{h}d"
            targets[label] = feature_df["close"].shift(-h) / feature_df["close"] - 1.0

        # Feature columns (exclude any existing target columns)
        feature_cols = [c for c in feature_df.columns if not c.startswith("return_")]
        X_all = feature_df[feature_cols].dropna()

        # Split
        train_size = int(len(X_all) * train_ratio)
        X_train_raw, X_test_raw = X_all.iloc[:train_size], X_all.iloc[train_size:]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)

        models = {}
        for label, target_series in targets.items():
            # Align indices
            y_all = target_series.dropna()
            common_idx = X_all.index.intersection(y_all.index)
            X_aligned = X_all.loc[common_idx]
            y_aligned = y_all.loc[common_idx]

            if len(X_aligned) < 50:
                logger.warning(f"Not enough data for {label} (size={len(X_aligned)}). Skipping.")
                continue

            t_size = int(len(X_aligned) * train_ratio)
            X_train = scaler.fit_transform(X_aligned.iloc[:t_size])
            X_test = scaler.transform(X_aligned.iloc[t_size:])

            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
            model.fit(X_train, y_aligned.iloc[:t_size])

            test_pred = model.predict(X_test)
            test_r2 = r2_score(y_aligned.iloc[t_size:], test_pred)

            models[label] = {
                "model": model,
                "scaler": scaler,  # NOTE: fitted on the aligned training set
                "r2": test_r2,
            }

        self.models[symbol] = models
        self.scalers[symbol] = scaler
        return models

    # ------------------------------------------------------------------
    # 3.4 Prediction (ML pipeline interface)
    # ------------------------------------------------------------------
    def predict(
        self,
        symbol: str,
        feature_df: pd.DataFrame,
        horizon_label: str = "return_5d",
    ) -> np.ndarray:
        """
        Generate forecasts for a trained horizon.

        Parameters
        ----------
        symbol : str
            Instrument identifier.
        feature_df : pd.DataFrame
            Feature matrix (same structure as training).
        horizon_label : str
            Which horizon model to use (e.g., 'return_5d').

        Returns
        -------
        np.ndarray
            Predicted returns.
        """
        if symbol not in self.models:
            raise ValueError(f"No model trained for {symbol}. Call build_prediction_model first.")
        if horizon_label not in self.models[symbol]:
            raise ValueError(
                f"Horizon {horizon_label} not available. Choices: {list(self.models[symbol].keys())}"
            )

        model_info = self.models[symbol][horizon_label]
        model = model_info["model"]
        scaler = model_info["scaler"]

        feature_cols = [c for c in feature_df.columns if not c.startswith("return_")]
        X = feature_df[feature_cols].dropna()
        X_scaled = scaler.transform(X)
        return model.predict(X_scaled)


# ---------------------------------------------------------------------------
# Optional: TradingIntegration (MetaTrader 5 bridge)
# ---------------------------------------------------------------------------
class TradingIntegration:
    """
    (Optional) Bridge to MetaTrader 5 for live data and signal generation.
    Requires MetaTrader5 installed.
    """

    def __init__(self, forecaster: ForexLiquidityForecaster):
        self.forecaster = forecaster
        self.mt5_initialized = False
        try:
            import MetaTrader5 as mt5

            self.mt5 = mt5
            if self.mt5.initialize():
                self.mt5_initialized = True
                logger.info("MetaTrader5 initialized successfully.")
            else:
                logger.warning("MetaTrader5 initialization failed.")
        except ImportError:
            logger.warning("MetaTrader5 not installed. TradingIntegration will be inactive.")
            self.mt5 = None

    def fetch_forex_data(self, symbol: str, days: int = 1460) -> pd.DataFrame:
        """Pull daily bars from MetaTrader5."""
        if not self.mt5_initialized:
            raise RuntimeError("MetaTrader5 is not initialized.")
        rates = self.mt5.copy_rates_from_pos(symbol, self.mt5.TIMEFRAME_D1, 0, days)
        if rates is None:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df["date"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("date", inplace=True)
        return df

    def generate_trading_signals(self, symbol: str, forecasts: dict) -> dict:
        """Crude signal logic based on short-term average return."""
        short_returns = [
            f["return"]
            for h, f in forecasts.get("forecasts", {}).items()
            if h in ["return_1d", "return_2d", "return_3d"]
        ]
        avg_return = np.mean(short_returns) if short_returns else 0.0

        signals = {
            "symbol": symbol,
            "avg_short_return": avg_return,
            "signal": ("BUY" if avg_return > 0.005 else "SELL" if avg_return < -0.005 else "HOLD"),
            "strength": min(abs(avg_return) * 100, 100),
        }
        return signals


# ---------------------------------------------------------------------------
# Quick demo (run as: python global_liquidity_pipeline.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Mine central bank data
    miner = GlobalLiquidityMiner(fred_api_key=None, start_date="2020-01-01")
    miner.fetch_central_bank_balance_sheets()

    # 2. Compute composite index
    index_calc = CompositeLiquidityIndex()
    liq_df = index_calc.calculate_liquidity_index(miner)
    if not liq_df.empty:
        enhanced = index_calc.enhance_liquidity_index(liq_df["liquidity_index"])
        logger.info(f"Composite liquidity index created. Shape: {enhanced.shape}")

    # 3. Build forecaster & train on a dummy dataset
    forecaster = ForexLiquidityForecaster(liquidity_miner=miner)
    # Generate a simple random walk for demonstration
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    demo_data = pd.DataFrame({"close": 100 + np.cumsum(np.random.randn(500))}, index=dates)
    features = forecaster.prepare_features("EURUSD", demo_data)
    models = forecaster.build_prediction_model("EURUSD", features)

    if models:
        logger.info("Trained models:")
        for horizon, info in models.items():
            logger.info(f"  {horizon}: R² = {info['r2']:.4f}")

    # 4. Predict using the trained model
    predictions = forecaster.predict("EURUSD", features, horizon_label="return_5d")
    logger.info(f"Sample predictions (return_5d): {predictions[:5]}")
