"""Figures 1-3 for MetaTrader 5 Machine Learning Blueprint (Part 26).

Run process_permutation_study.py first; this reads the study_results.json it
writes to the same folder. Real EURUSD H1 data, 500 replicates per null.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_BG, _AX_BG, _AX_EDGE, _GRID_C = "#ffffff", "#f6f8fa", "#d0d7de", "#eaeef2"
_TEXT, _SUBTEXT = "#1f2328", "#57606a"
C = {"blue": "#0969da", "green": "#1a7f37", "orange": "#bc4c00", "red": "#cf222e", "grey": "#57606a"}
FS_AX_LABEL, FS_TICK, FS_LEGEND, FS_ANNOT = 10, 9, 9, 9
plt.rcParams.update({
    "figure.facecolor": _BG, "axes.facecolor": _AX_BG, "axes.edgecolor": _AX_EDGE,
    "axes.labelcolor": _TEXT, "axes.grid": True, "grid.color": _GRID_C, "grid.linewidth": 0.6,
    "xtick.color": _SUBTEXT, "ytick.color": _SUBTEXT, "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK, "text.color": _TEXT, "legend.facecolor": _AX_BG,
    "legend.edgecolor": _AX_EDGE, "legend.labelcolor": _TEXT, "legend.fontsize": FS_LEGEND,
    "font.family": "monospace", "axes.labelsize": FS_AX_LABEL,
    "axes.spines.top": False, "axes.spines.right": False,
})
DPI, _W = 100, 7.5
SAVE_KW = dict(dpi=DPI, facecolor=_BG)
HERE = Path(__file__).parent


def _save(fig, name: str) -> Path:
    path = HERE / name
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    img = Image.open(path)
    if img.width != 750:
        img.resize((750, round(img.height * 750 / img.width)), Image.LANCZOS).save(path)
    return path


def _ordinal(n: int) -> str:
    """1 -> 1st, 2 -> 2nd, 11 -> 11th, 41 -> 41st."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _tag(ax, s: str) -> None:
    ax.text(-0.09, 1.02, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def fig_curve(r: dict) -> Path:
    c = r["curve"]
    b = np.array(c["trial_budget"], float)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(_W, 7.0), constrained_layout=True,
                                 gridspec_kw={"height_ratios": [1.5, 1.0]})
    a1.fill_between(b, c["median_ci_lower"], c["median_ci_upper"], color=C["red"], alpha=0.18, lw=0,
                    label="null median, 95% CI")
    a1.plot(b, c["spurious_median"], "o-", color=C["red"], lw=2.0, label="null median")
    a1.fill_between(b, c["q90_ci_lower"], c["q90_ci_upper"], color=C["orange"], alpha=0.14, lw=0,
                    label="null 90th percentile, 95% CI")
    a1.plot(b, c["spurious_q90"], "s--", color=C["orange"], lw=1.5, label="null 90th percentile")
    a1.plot(b, c["observed"], "D-", color=C["blue"], lw=2.0, ms=7, label="observed on real bars")
    for x, y, pct in zip(b, c["observed"], c["null_percentile"]):
        a1.annotate(f"{_ordinal(round(100 * pct))} pct", (x, y), xytext=(0, -17), textcoords="offset points",
                    ha="center", fontsize=FS_ANNOT, color=C["blue"],
                    bbox=dict(boxstyle="round,pad=0.15", fc=_AX_BG, ec="none", alpha=0.9))
    a1.axhline(0, color=_SUBTEXT, lw=1.0, ls="--")
    a1.set_xscale("log"); a1.set_xticks(b, [str(int(x)) for x in b])
    a1.set_ylabel("annualized Sharpe"); a1.set_xlabel("trial budget (log scale)")
    a1.set_ylim(-0.15, 1.2)
    a1.legend(loc="upper left", ncol=2, framealpha=0.92)
    _tag(a1, "(a)")
    a2.bar(np.arange(b.size), np.array(c["fraction_positive"]) * 100, color=C["red"], alpha=0.8,
           edgecolor=_AX_EDGE, width=0.6)
    for i, f in enumerate(c["fraction_positive"]):
        a2.text(i, f * 100 + 2, f"{100 * f:.0f}%", ha="center", fontsize=FS_ANNOT)
    a2.set_xticks(np.arange(b.size), [str(int(x)) for x in b])
    a2.set_ylim(0, 100); a2.set_ylabel("null replicates with\nSharpe > 0 (%)")
    a2.set_xlabel("trial budget")
    _tag(a2, "(b)")
    return _save(fig, "spurious_curve.png")


def fig_nulls(r: dict) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(_W, 7.0), constrained_layout=True)
    for ax, key, col, lab, tag in ((axes[0], "search", C["red"], "100-trial search", "(a)"),
                                   (axes[1], "fixed", C["green"], "fixed MA(10, 50), no search", "(b)")):
        d = r["point"][key]
        v = np.asarray(d["null_values"], float)
        ax.hist(v, bins=40, density=True, alpha=0.65, color=col, edgecolor=_BG,
                label=f"{lab}: null, n = {v.size}")
        ax.axvspan(d["median_ci_lower"], d["median_ci_upper"], color=_SUBTEXT, alpha=0.15,
                   label="null median, 95% CI")
        ax.axvline(d["spurious_center"], color=_TEXT, lw=1.5, ls="--",
                   label=f"null median {d['spurious_center']:.3f}")
        ax.axvline(d["observed"], color=C["blue"], lw=2.0,
                   label=f"observed {d['observed']:.3f} ({_ordinal(round(100 * d['null_percentile']))} pct, p = {d['p_value']:.2f})")
        ax.set_xlabel("annualized Sharpe"); ax.set_ylabel("density")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.75)
        ax.legend(loc="upper right", framealpha=0.92)
        _tag(ax, tag)
    return _save(fig, "null_distributions.png")


def fig_sensitivity(r: dict) -> Path:
    rows = r["sensitivity"]["rows"]
    s = np.array([x["grid_seed"] for x in rows]); pct = np.array([x["percentile"] for x in rows]) * 100
    fig, ax = plt.subplots(figsize=(_W, 3.9), constrained_layout=True)
    ax.axhspan(0, 50, color=C["red"], alpha=0.05, lw=0)
    ax.axhline(50, color=_TEXT, lw=1.0, ls="--", label="null median (50th pct)")
    ax.axhline(95, color=C["orange"], lw=1.5, ls=":", label="95th pct (one-sided 5% level)")
    cols = [C["blue"] if g != 1 else C["red"] for g in s]
    ax.scatter(s, pct, c=cols, s=55, zorder=3, edgecolor=_TEXT, lw=0.6)
    ax.annotate("grid seed 1\n(Figures 1-2)", (1, pct[0]), xytext=(18, 26), textcoords="offset points",
                fontsize=FS_ANNOT, arrowprops=dict(arrowstyle="-", color=_SUBTEXT, lw=1.0))
    ax.set_xticks(s); ax.set_xlim(0.3, 20.7); ax.set_ylim(0, 100)
    ax.set_xlabel("grid seed (100 (fast, slow) pairs drawn per seed)")
    ax.set_ylabel("observed percentile\nwithin its null (%)")
    ax.legend(loc="lower right", framealpha=0.92)
    return _save(fig, "grid_seed_sensitivity.png")


if __name__ == "__main__":
    res = json.loads((HERE / "study_results.json").read_text())
    for p in (fig_curve(res), fig_nulls(res), fig_sensitivity(res)):
        w, h = Image.open(p).size
        print(("OK " if w == 750 else "BAD"), p.name, w, h)
