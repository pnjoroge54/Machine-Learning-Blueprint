import pandas as pd
from fredapi import Fred
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


class GlobalLiquidityMiner:
    def __init__(self, fred_api_key: str, start_date: str, end_date: str):
        self.fred = Fred(api_key=fred_api_key) if fred_api_key else None
        self.start_date = start_date
        self.end_date = end_date
        self.data_cache = {}

    def fetch_central_bank_balance_sheets(self) -> dict[str, pd.DataFrame]:
        """Obtaining central bank balance sheet data."""
        balance_sheets = {}

        if self.fred:
            logger.info("Loading Federal Reserve data...")
            try:
                fed_total_assets = self.fred.get_series(
                    "WALCL", start=self.start_date, end=self.end_date
                )
                fed_securities = self.fred.get_series(
                    "WSHOSHO", start=self.start_date, end=self.end_date
                )
                fed_loans = self.fred.get_series("WLRRAL", start=self.start_date, end=self.end_date)

                balance_sheets["FED"] = pd.DataFrame(
                    {
                        "date": fed_total_assets.index,
                        "total_assets": fed_total_assets.values,
                        "securities_held": fed_securities.reindex(
                            fed_total_assets.index, method="ffill"
                        ).values,
                        "loans_and_repos": fed_loans.reindex(
                            fed_total_assets.index, method="ffill"
                        ).values,
                        "currency": "USD",
                    }
                )

                balance_sheets["FED"]["assets_growth_rate"] = balance_sheets["FED"][
                    "total_assets"
                ].pct_change(periods=52)
                balance_sheets["FED"]["securities_share"] = (
                    balance_sheets["FED"]["securities_held"] / balance_sheets["FED"]["total_assets"]
                )

                logger.info(f"Loaded {len(fed_total_assets)} Federal Reserve data entries")
            except Exception as e:
                logger.error(f"Error loading Fed data: {e}")

        self.data_cache["balance_sheets"] = balance_sheets
        return balance_sheets


class ForexLiquidityForecaster:
    def __init__(self, liquidity_miner: GlobalLiquidityMiner):
        self.liquidity_miner = liquidity_miner
        self.models = {}
        self.scalers = {}
        self.forecasts = {}

    def prepare_features(self, symbol: str, historical_data: pd.DataFrame) -> pd.DataFrame:
        """Create features for forecasting."""
        df = historical_data.copy()

        # Technical indicators
        df["rsi_14"] = self.calculate_rsi(df["close"], 14)
        df["ema_50"] = df["close"].ewm(span=50).mean()
        df["volatility_20d"] = df["close"].pct_change().rolling(20).std()

        # Attaching liquidity data
        if "balance_sheets" in self.liquidity_miner.data_cache:
            for bank, bs_data in self.liquidity_miner.data_cache["balance_sheets"].items():
                df = df.join(
                    bs_data[["total_assets"]].rename(columns={"total_assets": f"{bank}_balance"}),
                    how="left",
                )
                df[f"{bank}_balance"].fillna(method="ffill", inplace=True)

        return df.dropna()

    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """RSI calculation."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def build_prediction_model(self, symbol: str, feature_df: pd.DataFrame):
        """Training a forecasting model."""
        targets = {
            f"return_{h}d": feature_df["close"].shift(-h) / feature_df["close"] - 1
            for h in [1, 3, 5, 8]
        }

        feature_columns = [col for col in feature_df.columns if not col.startswith("return_")]
        X = feature_df[feature_columns].dropna()

        train_size = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]

        models = {}
        for target_name, target_series in targets.items():
            y = target_series.dropna()
            common_idx = X.index.intersection(y.index)
            X_aligned, y_aligned = X.loc[common_idx], y.loc[common_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_aligned.iloc[:train_size])
            X_test_scaled = scaler.transform(X_aligned.iloc[train_size:])

            model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
            model.fit(X_train_scaled, y_aligned.iloc[:train_size])

            test_pred = model.predict(X_test_scaled)
            test_r2 = r2_score(y_aligned.iloc[train_size:], test_pred)

            models[target_name] = {"model": model, "scaler": scaler, "r2": test_r2}

        self.models[symbol] = models
        self.scalers[symbol] = scaler
