"""
Size calibration for the null schemes in :mod:`afml.screening.permutation`.

Answers plan open question 1: what block length does the stationary bootstrap
need to achieve nominal test size, and does it get there at all?

Design. Feature and target are generated independently, so the null hypothesis
is true by construction and every rejection is a false positive. The feature is
AR(1) with coefficient ``phi``, standing in for a smoothed indicator. The target
is an ``h``-bar overlapping forward return, standing in for a triple-barrier
label. The test statistic is absolute Pearson correlation; any bivariate
dependence measure shows the same distortion.

Run::

    python -m afml.screening.notes.size_calibration
"""

from __future__ import annotations

import numpy as np

from ..permutation import build_null

N_TRIALS = 200
N_PERM = 200
N_OBS = 2000
ALPHA = 0.05


def ar1(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def overlapping_forward_return(n: int, h: int, rng: np.random.Generator) -> np.ndarray:
    e = rng.standard_normal(n + h)
    return np.convolve(e, np.ones(h), mode="valid")[:n]


def abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    sx, sy = x.std(), y.std()
    return (
        0.0
        if sx == 0 or sy == 0
        else float(abs(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy)))
    )


def size(scheme: str, phi: float, h: int, seed: int, **kw) -> float:
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(N_TRIALS):
        x = ar1(N_OBS, phi, rng)
        y = overlapping_forward_return(N_OBS, h, rng)
        res = build_null(
            abs_corr,
            x,
            y,
            scheme=scheme,
            n_perm=N_PERM,
            seed=int(rng.integers(0, 2**31)),
            scheme_kwargs=kw,
        )
        hits += res.p_value < ALPHA
    return hits / N_TRIALS


def main() -> None:
    print(f"nominal alpha = {ALPHA} | {N_TRIALS} trials | {N_PERM} permutations | n = {N_OBS}\n")

    print("A. Scheme comparison, phi = 0.99, label horizon h = 100")
    print(f"{'scheme':<34}{'size':>8}")
    print(f"{'naive_shuffle':<34}{size('naive_shuffle', 0.99, 100, 11):>8.3f}")
    print(f"{'circular_shift':<34}{size('circular_shift', 0.99, 100, 12):>8.3f}")
    for mult in (1, 2, 3, 5, 8, 12):
        block = float(mult * 100)
        label = f"stationary_bootstrap (L = {mult}x span)"
        print(f"{label:<34}{size('stationary_bootstrap', 0.99, 100, 13, mean_block=block):>8.3f}")

    print("\nB. circular_shift across configurations")
    print(f"{'phi':>6}{'h':>6}{'size':>8}")
    for phi in (0.0, 0.90, 0.99):
        for h in (1, 20, 100):
            print(f"{phi:>6.2f}{h:>6d}{size('circular_shift', phi, h, 21):>8.3f}")


if __name__ == "__main__":
    main()
