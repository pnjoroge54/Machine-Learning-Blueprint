"""
Tests for afml.denoising (Marcenko-Pastur denoising and detoning, MLAM Ch. 2).
"""

import numpy as np
import pandas as pd
import pytest

from ..denoising.denoising_detoning import (
    EffectiveSampleSizeError,
    clean_corr,
    cov2corr,
    effective_sample_size,
    get_pca,
    mp_pdf,
)


def _factor_panel(t=2000, n=40, n_factors=2, seed=0):
    """Returns driven by a few strong factors plus idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(t, n_factors))
    loadings = rng.normal(0.0, 1.0, size=(n_factors, n))
    X = factors @ loadings + rng.normal(size=(t, n))
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(n)])


def test_cov2corr_matches_numpy():
    X = _factor_panel(t=500, n=6).to_numpy()
    corr = cov2corr(np.cov(X, rowvar=False))
    np.testing.assert_allclose(corr, np.corrcoef(X, rowvar=False), atol=1e-12)
    np.testing.assert_allclose(np.diag(corr), 1.0)


def test_get_pca_sorted_and_reconstructs():
    corr = np.corrcoef(_factor_panel(t=500, n=8).to_numpy(), rowvar=False)
    e_val, e_vec = get_pca(corr)
    assert np.all(np.diff(e_val) <= 1e-12)
    np.testing.assert_allclose(e_vec @ np.diag(e_val) @ e_vec.T, corr, atol=1e-10)


@pytest.mark.parametrize("q, var", [(10.0, 1.0), (4.0, 0.5)])
def test_marcenko_pastur_density_integrates_to_one(q, var):
    pdf = mp_pdf(var, q, pts=20_000)
    e = pdf.index.to_numpy()
    assert e.min() == pytest.approx(var * (1 - q ** -0.5) ** 2)
    assert e.max() == pytest.approx(var * (1 + q ** -0.5) ** 2)
    assert np.trapz(pdf.to_numpy(), e) == pytest.approx(1.0, abs=0.01)


def test_effective_sample_size_ar1():
    rng = np.random.default_rng(1)
    t, n = 20_000, 3
    iid = rng.normal(size=(t, n))
    t_eff, rho = effective_sample_size(iid, t)
    assert abs(rho) < 0.03 and t_eff == pytest.approx(t, rel=0.06)

    ar = np.zeros((t, n))
    for i in range(1, t):
        ar[i] = 0.5 * ar[i - 1] + rng.normal(size=n)
    t_eff, rho = effective_sample_size(ar, t)
    assert rho == pytest.approx(0.5, abs=0.03)
    assert t_eff == pytest.approx(t / 3, rel=0.1)  # (1 - 0.5) / (1 + 0.5)


def test_clean_corr_keeps_a_valid_correlation_matrix_and_finds_the_factors():
    X = _factor_panel(n_factors=2)
    corr, info = clean_corr(X)
    assert corr.shape == (40, 40)
    np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)
    np.testing.assert_allclose(corr.to_numpy(), corr.to_numpy().T, atol=1e-12)
    assert np.linalg.eigvalsh(corr.to_numpy()).min() > -1e-8
    assert 2 <= info["n_facts"] <= 3
    for key in ("t_raw", "t_eff", "q"):
        assert key in info


def test_detoning_removes_the_market_mode():
    X = _factor_panel(n_factors=1, seed=3)
    raw_top = np.linalg.eigvalsh(np.corrcoef(X.to_numpy(), rowvar=False)).max()
    detoned, _ = clean_corr(X, detone=True, market_components=1)
    assert np.linalg.eigvalsh(detoned.to_numpy()).max() < 0.5 * raw_top


def test_clean_corr_refuses_too_few_effective_observations():
    X = _factor_panel(t=30, n=40)
    with pytest.raises(EffectiveSampleSizeError):
        clean_corr(X, min_q=1.0)
