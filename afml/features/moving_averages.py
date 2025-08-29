import warnings
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
from feature_engine.selection import DropCorrelatedFeatures
from loguru import logger
from matplotlib import pyplot as plt
from scipy import stats

# Optional SHAP import (with fallback)
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. Feature importance will use correlation-based method.")


def get_MA_diffs(
    close: pd.Series, windows: Union[List, Tuple], threshold: float = 0.8, verbose: bool = False
):
    """
    Moving average differences.

    param close: (pd.Series) Close prices
    param windows: (Union[List, Tuple]) Windows to create differences for, e.g. (10, 20, 50)
    param threshold: (float)
    param verbose: (bool) If True, display heatmap of correlation matrix
    returns df: (pd.DataFrame)
    """
    df = pd.DataFrame(index=close.index)
    sma = {window: close.rolling(window, closed="left").mean() for window in windows}

    # Create differences of all unique combinations of windows
    for win in combinations(windows, 2):
        fast_window, slow_window = sorted(win)
        df[f"sma_diff_{fast_window}_{slow_window}"] = sma[fast_window] - sma[slow_window]

    dcf = DropCorrelatedFeatures(threshold=threshold)
    out = dcf.fit_transform(df)
    dropped = df.columns.difference(out.columns).to_list()
    if len(dropped) > 0:
        logger.info(
            f"\nDropped features with correlation > {threshold}: \n\t{dropped}"
            f"\nKept features: \n\t{out.columns.to_list()}"
        )
        if verbose:
            corr_matrix = df.corr()
            # Set the figure size for better readability
            plt.figure(figsize=(12, 4))

            # Create the heatmap with the mask
            sns.heatmap(
                corr_matrix,
                cmap="coolwarm",  # Choose a colormap
                linewidths=0.5,  # Add lines to separate the cells
                annot=True,  # Annotate with the correlation values
                fmt=".2f",  # Format the annotations to two decimal places
                cbar_kws={"shrink": 0.8},  # Shrink the color bar
            )

            plt.title("Correlation Matrix")
            plt.show()

    return out


def calculate_ma_differences(
    close: pd.Series,
    windows: Union[List[int], Tuple[int, ...]],
    threshold: float = 0.8,
    include_ratios: bool = True,
    normalize_differences: bool = True,
    verbose: bool = False,
    return_metadata: bool = False,
    target_series: Optional[pd.Series] = None,
    importance_method: str = "correlation",
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Calculate moving average differences and ratios with intelligent correlation handling.

    This function creates moving average difference features and automatically removes
    redundant features based on correlation thresholds. It includes options for ratio
    calculations, normalization, and feature importance estimation.

    Parameters:
    -----------
    close : pd.Series
        Close price series with datetime index
    windows : Union[List[int], Tuple[int, ...]]
        Windows to create differences for, e.g. [10, 20, 50]
    threshold : float, default 0.8
        Correlation threshold for feature removal (0-1)
    include_ratios : bool, default True
        Whether to include ratio features in addition to differences
    normalize_differences : bool, default True
        Whether to normalize differences by the slower MA
    verbose : bool, default False
        If True, display detailed logging and visualizations
    return_metadata : bool, default False
        If True, return additional metadata about the feature selection process
    target_series : Optional[pd.Series], default None
        Target series for feature importance calculation (e.g., future returns)
    importance_method : str, default "correlation"
        Method for calculating feature importance ("correlation", "shap", or "mutual_info")

    Returns:
    --------
    Union[pd.DataFrame, Dict[str, Any]]
        If return_metadata is False: DataFrame with selected features
        If return_metadata is True: Dictionary with features and metadata

    Raises:
    -------
    ValueError
        If windows has less than 2 elements or threshold is not between 0-1
    """
    # Input validation
    if not isinstance(windows, (list, tuple)) or len(windows) < 2:
        raise ValueError("windows must be a list or tuple with at least 2 elements")

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    # Calculate moving averages
    df = pd.DataFrame(index=close.index)
    sma_dict = {}

    for window in windows:
        sma_dict[window] = close.rolling(window, min_periods=1).mean()
        df[f"sma_{window}"] = sma_dict[window]

    # Create all unique combinations of windows
    feature_candidates = {}

    for fast, slow in combinations(windows, 2):
        if fast >= slow:
            continue

        # Difference features
        diff_feature = sma_dict[fast] - sma_dict[slow]
        feature_candidates[f"ma_diff_{fast}_{slow}"] = diff_feature

        # Ratio features (if enabled)
        if include_ratios:
            ratio_feature = sma_dict[fast] / sma_dict[slow]
            feature_candidates[f"ma_ratio_{fast}_{slow}"] = ratio_feature

        # Normalized differences (if enabled)
        if normalize_differences:
            norm_diff_feature = diff_feature / sma_dict[slow]
            feature_candidates[f"ma_norm_diff_{fast}_{slow}"] = norm_diff_feature

    # Add all candidate features to DataFrame
    for feature_name, feature_values in feature_candidates.items():
        df[feature_name] = feature_values

    # Remove any features with zero variance
    initial_features = df.columns.tolist()
    df = df.loc[:, df.std() > 1e-8]
    zero_variance_features = set(initial_features) - set(df.columns)

    if zero_variance_features and verbose:
        logger.info(f"Removed zero-variance features: {zero_variance_features}")

    # Calculate correlation matrix
    corr_matrix = df.corr().abs()

    # Identify highly correlated features to remove
    upper_tri = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

    # Keep only non-redundant features
    selected_features = df.columns.difference(to_drop)
    result_df = df[selected_features]

    # Calculate feature importance if target is provided
    feature_importance = None
    if target_series is not None:
        feature_importance = calculate_feature_importance(
            result_df, target_series, method=importance_method
        )

    # Logging and visualization
    if verbose:
        logger.info(f"Initial features: {len(df.columns)}")
        logger.info(f"Selected features: {len(selected_features)}")
        logger.info(f"Dropped features: {to_drop}")

        # Plot correlation matrix
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap="RdBu_r",
            center=0,
            square=True,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Moving Average Feature Correlations", fontsize=16)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        # Plot feature importance if available
        if feature_importance is not None:
            plt.figure(figsize=(10, 6))
            feature_importance.sort_values().plot(kind="barh")
            plt.title("Feature Importance Scores")
            plt.tight_layout()
            plt.show()

    # Return based on metadata flag
    if return_metadata:
        return {
            "features": result_df,
            "dropped_features": to_drop,
            "correlation_matrix": corr_matrix,
            "feature_importance": feature_importance,
            "parameters": {
                "windows": windows,
                "threshold": threshold,
                "include_ratios": include_ratios,
                "normalize_differences": normalize_differences,
            },
        }
    else:
        return result_df


def calculate_feature_importance(
    features: pd.DataFrame, target: pd.Series, method: str = "correlation"
) -> pd.Series:
    """
    Calculate feature importance using various methods.

    Parameters:
    -----------
    features : pd.DataFrame
        Feature matrix
    target : pd.Series
        Target variable
    method : str, default "correlation"
        Importance calculation method ("correlation", "shap", or "mutual_info")

    Returns:
    --------
    pd.Series
        Feature importance scores
    """
    # Align features and target
    aligned_data = pd.concat([features, target], axis=1).dropna()
    X = aligned_data[features.columns]
    y = aligned_data[target.name]

    if method == "correlation":
        # Simple correlation-based importance
        importance = X.corrwith(y).abs()
        return importance / importance.sum()  # Normalize to sum to 1

    elif method == "shap" and SHAP_AVAILABLE:
        # SHAP-based importance
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split

        # Train a model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Calculate mean absolute SHAP values
        importance = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
        return importance / importance.sum()  # Normalize to sum to 1

    elif method == "mutual_info":
        # Mutual information-based importance
        from sklearn.feature_selection import mutual_info_regression

        # Calculate mutual information
        mi = mutual_info_regression(X, y, random_state=42)
        importance = pd.Series(mi, index=X.columns)
        return importance / importance.sum()  # Normalize to sum to 1

    else:
        if method == "shap" and not SHAP_AVAILABLE:
            warnings.warn("SHAP not available, falling back to correlation method")
        elif method not in ["correlation", "shap", "mutual_info"]:
            warnings.warn(f"Unknown method {method}, falling back to correlation method")

        # Fallback to correlation
        importance = X.corrwith(y).abs()
        return importance / importance.sum()  # Normalize to sum to 1


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    prices = 100 + np.cumsum(np.random.randn(500) * 0.5)
    close = pd.Series(prices, index=dates)

    # Create a target (future returns)
    future_returns = close.pct_change(5).shift(-5)  # 5-day forward returns

    # Calculate MA differences
    result = calculate_ma_differences(
        close=close,
        windows=[10, 20, 50, 100],
        threshold=0.7,
        include_ratios=True,
        normalize_differences=True,
        verbose=True,
        return_metadata=True,
        target_series=future_returns,
        importance_method="correlation",
    )

    print(f"Selected {len(result['features'].columns)} features")
    print("Feature importance:")
    print(result["feature_importance"].sort_values(ascending=False))
