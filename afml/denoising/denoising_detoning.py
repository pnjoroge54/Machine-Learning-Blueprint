"""Denoising and detoning of correlation matrices.

Implements the Marcenko-Pastur fitting, effective sample size, denoising,
and detoning procedures described in Machine Learning for Asset Managers,
Chapter 2.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity


class EffectiveSampleSizeError(ValueError):
    """Raised when the effective sample size leaves too few observations per variable.

    The Marcenko-Pastur density is defined for q = T/N greater than 1. A panel
    persistent enough to push T_eff below N carries no usable noise spectrum, and
    fitting one anyway would produce a ceiling the data does not support.

    Everything computed before the failure is attached, so a caller that wants to
    continue without denoising does not repeat any of it:

        corr      the sample correlation matrix
        e_val     eigenvalues, descending
        e_vec     eigenvectors, matching column order
        info      t_raw, t_eff, rho_bar, n_features, q

    The expensive Marcenko-Pastur fit is never reached, so nothing is wasted.
    """

    def __init__(self, message, corr=None, e_val=None, e_vec=None, info=None):
        super().__init__(message)
        self.corr = corr
        self.e_val = e_val
        self.e_vec = e_vec
        self.info = info or {}


def get_pca(corr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return eigenvalues and eigenvectors of a symmetric correlation matrix.

    Eigenvalues are sorted in descending order; the eigenvector columns follow
    the same order.
    """
    e_val, e_vec = np.linalg.eigh(corr)
    idx = np.argsort(e_val)[::-1]
    return e_val[idx], e_vec[:, idx]


def mp_pdf(var: float, q: float, pts: int = 1000) -> pd.Series:
    """Theoretical Marcenko-Pastur density.

    Parameters
    ----------
    var : float
        Noise variance, sigma^2.
    q : float
        Observations per variable, T/N. Note: q = T/N, not N/T.

    Returns
    -------
    pd.Series indexed by the eigenvalue grid.
    """
    e_min = var * (1 - (1.0 / q) ** 0.5) ** 2
    e_max = var * (1 + (1.0 / q) ** 0.5) ** 2
    e_val = np.linspace(e_min, e_max, pts)
    pdf = q / (2 * np.pi * var * e_val) * ((e_max - e_val) * (e_val - e_min)) ** 0.5
    return pd.Series(pdf, index=e_val)


def fit_kde(
    obs: np.ndarray,
    b_width: float = 0.15,
    x: np.ndarray | None = None,
) -> pd.Series:
    """Absolute-bandwidth Gaussian KDE of empirical eigenvalues.

    The bandwidth is in eigenvalue units for sklearn's KernelDensity. It is
    NOT a scipy.stats.gaussian_kde ``bw_method`` relative scale factor.
    """
    obs = np.asarray(obs).reshape(-1, 1)
    kde = KernelDensity(kernel="gaussian", bandwidth=b_width).fit(obs)

    if x is None:
        x = np.unique(obs).reshape(-1, 1)
    else:
        x = np.asarray(x).reshape(-1, 1)

    log_density = kde.score_samples(x)
    return pd.Series(np.exp(log_density), index=x.flatten())


def err_pdfs(
    var: np.ndarray,
    e_val: np.ndarray,
    q: float,
    b_width: float,
    pts: int = 1000,
) -> float:
    """Squared error between theoretical MP pdf and empirical KDE."""
    pdf0 = mp_pdf(var[0], q, pts)
    pdf1 = fit_kde(e_val, b_width, x=pdf0.index.values)
    return float(np.sum((pdf1 - pdf0) ** 2))


def find_max_eval(
    e_val: np.ndarray,
    q: float,
    b_width: float = 0.15,
) -> tuple[float, float]:
    """Fit noise variance and return the noise ceiling and fitted sigma^2.

    q is T/N, the number of observations per variable.
    """
    out = minimize(
        lambda v: err_pdfs(v, e_val, q, b_width),
        x0=np.array([0.5]),
        bounds=((1e-5, 1 - 1e-5),),
    )
    var = out["x"][0] if out["success"] else 1.0
    e_max = var * (1 + (1.0 / q) ** 0.5) ** 2
    return e_max, var


def effective_sample_size(
    X: np.ndarray,
    t_raw: int,
) -> tuple[float, float]:
    """AR(1) effective sample size based on average lag-1 autocorrelation.

    Parameters
    ----------
    X : ndarray of shape (T, N)
        Feature panel.
    t_raw : int
        Raw number of observations.

    Returns
    -------
    (t_eff, rho_bar)
    """
    rhos = [np.corrcoef(X[:-1, j], X[1:, j])[0, 1] for j in range(X.shape[1])]
    rho_bar = float(np.mean(rhos))
    t_eff = t_raw * (1 - rho_bar) / (1 + rho_bar)
    return t_eff, rho_bar


def cov2corr(cov: np.ndarray) -> np.ndarray:
    """Convert covariance to correlation and clip numerical errors to [-1, 1]."""
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return np.clip(corr, -1, 1)


def denoised_corr(
    e_val: np.ndarray,
    e_vec: np.ndarray,
    n_facts: int,
    labels: Sequence[str] | None = None,
) -> np.ndarray | pd.DataFrame:
    """Constant residual eigenvalue denoising (MLAM eq. 2.5).

    Keeps the top ``n_facts`` eigenvectors unchanged and replaces all noise
    eigenvalues by their average, preserving the trace.
    """
    e_val_ = e_val.copy()

    if n_facts < e_val_.shape[0]:
        e_val_[n_facts:] = e_val_[n_facts:].sum() / (e_val_.shape[0] - n_facts)

    corr1 = e_vec @ np.diag(e_val_) @ e_vec.T
    corr1 = cov2corr(corr1)

    if labels is not None:
        return pd.DataFrame(corr1, index=labels, columns=labels)

    return corr1


def detoned_corr(
    corr: pd.DataFrame,
    market_components: int = 1,
) -> pd.DataFrame:
    """Remove the market-mode component from a denoised correlation matrix.

    The market component is usually the first eigenvector. The result is
    rescaled to a proper correlation matrix.
    """
    e_val, e_vec = get_pca(corr)

    # e_val is a 1D vector. Build a diagonal matrix from the top
    # market_components eigenvalues instead of using 2D slicing.
    e_val_mkt = np.diag(e_val[:market_components])
    e_vec_mkt = e_vec[:, :market_components]

    corr_mkt = e_vec_mkt @ e_val_mkt @ e_vec_mkt.T
    corr1 = corr.values - corr_mkt

    return pd.DataFrame(cov2corr(corr1), index=corr.index, columns=corr.columns)


def clean_corr(
    X: pd.DataFrame,
    detone: bool = False,
    market_components: int = 1,
    kde_bwidth: float = 0.15,
    use_effective_sample_size: bool = True,
    corr: pd.DataFrame | None = None,
    min_q: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """Denoise, and optionally detone, the correlation matrix of a feature panel.

    Wraps the Chapter 2 procedure into the single call a downstream consumer
    needs: fit the Marcenko-Pastur noise ceiling, count the eigenvalues above
    it, shrink the rest to their common average, and optionally strip the
    market mode.

    Parameters
    ----------
    X : pd.DataFrame of shape (T, N)
        Feature panel. Used for the correlation matrix and, when
        ``use_effective_sample_size`` is set, for the AR(1) correction.
    detone : bool
        Remove the top ``market_components`` eigenvectors after denoising.
    market_components : int
        Number of leading eigenvectors treated as the market mode.
    kde_bwidth : float
        Absolute bandwidth, in eigenvalue units, for the empirical KDE.
    use_effective_sample_size : bool
        Fit the noise ceiling against T_eff rather than T. Persistent features
        carry fewer independent observations than rows, which widens the noise
        band and lowers the factor count. Leave this on for financial panels.
    corr : pd.DataFrame, optional
        Precomputed correlation matrix. Defaults to ``X.corr()``.
    min_q : float
        Smallest acceptable T_eff/N. Below this the Marcenko-Pastur density is
        undefined or degenerate and ``EffectiveSampleSizeError`` is raised,
        carrying the correlation matrix and eigendecomposition so the caller can
        continue without recomputing them.

    Raises
    ------
    EffectiveSampleSizeError
        When q is not above ``min_q``.

    Returns
    -------
    (corr_clean, info)
        ``info`` carries the diagnostics a report needs: t_raw, t_eff, rho_bar,
        q, the fitted noise variance, the noise ceiling, and n_facts.
    """
    corr_raw = X.corr() if corr is None else corr
    t_raw, n_features = X.shape

    rho_bar = np.nan
    t_eff = float(t_raw)
    if use_effective_sample_size:
        t_eff, rho_bar = effective_sample_size(X.to_numpy(dtype=float), t_raw)

    q = t_eff / n_features
    e_val, e_vec = get_pca(corr_raw)

    if q <= min_q:
        # Raise before the Marcenko-Pastur fit rather than clamping q. Clamping
        # would return a denoised matrix built on a ceiling fitted at a q the
        # data does not support, and it would do so silently, which is worse
        # than either failing or skipping the denoising.
        raise EffectiveSampleSizeError(
            f"q = T_eff/N = {q:.3f} is not above min_q = {min_q:.3f}. "
            f"T_eff = {t_eff:.1f} from T = {t_raw} at rho_bar = {rho_bar:.3f}, "
            f"against N = {n_features} features. The panel is too persistent for a "
            f"Marcenko-Pastur fit; cluster without denoising, drop features, or pass "
            f"use_effective_sample_size=False to fit against T instead.",
            corr=corr_raw,
            e_val=e_val,
            e_vec=e_vec,
            info={
                "t_raw": int(t_raw),
                "t_eff": float(t_eff),
                "rho_bar": float(rho_bar),
                "n_features": int(n_features),
                "q": float(q),
            },
        )

    e_max, var = find_max_eval(e_val, q, b_width=kde_bwidth)
    n_facts = int((e_val > e_max).sum())

    corr_clean = denoised_corr(e_val, e_vec, n_facts, labels=corr_raw.index)
    if detone:
        corr_clean = detoned_corr(corr_clean, market_components=market_components)

    info = {
        "t_raw": int(t_raw),
        "t_eff": float(t_eff),
        "rho_bar": float(rho_bar),
        "n_features": int(n_features),
        "q": float(q),
        "noise_var": float(var),
        "noise_ceiling": float(e_max),
        "n_facts": n_facts,
        "detoned": bool(detone),
        "market_components": int(market_components) if detone else 0,
    }
    return corr_clean, info
