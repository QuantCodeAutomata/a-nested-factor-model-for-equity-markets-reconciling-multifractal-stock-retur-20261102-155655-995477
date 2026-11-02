"""
Plotting utilities for N-SfFM replication results.

Produces figures corresponding to Figures 2-8 in the paper.
All figures are saved to results/ directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

RESULTS_DIR = Path("results")


def _save(fig: plt.Figure, name: str) -> Path:
    """Save a figure to results/ directory."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_hurst_distributions(
    stock_hurst: NDArray,
    index_hurst: NDArray,
    H_true: float,
    H_i_true: float,
    filename: str = "exp1_hurst_distributions.png",
    title: str = "Figure 2: Hurst exponent distributions (synthetic)",
) -> Path:
    """
    Plot empirical distributions of Hurst exponents for stocks and index (Figure 2).

    Parameters
    ----------
    stock_hurst : NDArray of shape (N*MC_REPS,)
        Hurst estimates for individual stocks.
    index_hurst : NDArray of shape (MC_REPS,)
        Hurst estimates for the index.
    H_true : float
        True factor Hurst exponent (expected center of index distribution).
    H_i_true : float
        True idiosyncratic Hurst exponent (expected center of stock distribution).
    filename : str
        Output filename (saved to results/).
    title : str
        Figure title.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(0, max(0.25, stock_hurst.max() * 1.1, index_hurst.max() * 1.1), 40)

    ax.hist(stock_hurst, bins=bins, alpha=0.6, color="steelblue", label=f"Single stocks (true H_i={H_i_true})")
    ax.hist(index_hurst, bins=bins, alpha=0.6, color="black", label=f"Index (true H={H_true})")
    ax.axvline(H_true, color="black", linestyle="--", linewidth=1.5, label=f"H={H_true}")
    ax.axvline(H_i_true, color="steelblue", linestyle="--", linewidth=1.5, label=f"H_i={H_i_true}")

    ax.set_xlabel("Estimated Hurst exponent", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return _save(fig, filename)


def plot_factor_h_vs_true_h(
    H_true_vals: list[float],
    H_index_means: list[float],
    H_index_ci_low: list[float],
    H_index_ci_high: list[float],
    filename: str = "exp2_factor_h_vs_true.png",
    title: str = "Figure 3: Estimated index H vs true factor H",
) -> Path:
    """
    Plot estimated index Hurst exponent vs true factor H (Figure 3).

    Parameters
    ----------
    H_true_vals : list of float
        True factor Hurst values tested.
    H_index_means : list of float
        Mean estimated index Hurst for each true H.
    H_index_ci_low : list of float
        Lower 95% CI for estimated index Hurst.
    H_index_ci_high : list of float
        Upper 95% CI for estimated index Hurst.
    filename : str
        Output filename.
    title : str
        Figure title.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    H_range = np.linspace(min(H_true_vals) * 0.9, max(H_true_vals) * 1.1, 100)
    ax.plot(H_range, H_range, "k--", linewidth=1, label="First bisector")

    ax.errorbar(
        H_true_vals,
        H_index_means,
        yerr=[
            [m - lo for m, lo in zip(H_index_means, H_index_ci_low)],
            [hi - m for m, hi in zip(H_index_means, H_index_ci_high)],
        ],
        fmt="o",
        color="purple",
        capsize=5,
        label="Estimated H_I (mean ± 95% CI)",
    )

    ax.set_xlabel("True factor H", fontsize=12)
    ax.set_ylabel("Estimated index Hurst H_I", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return _save(fig, filename)


def plot_hurst_vs_N(
    N_values: list[int],
    H_index_means: NDArray,
    H_index_ci: NDArray,
    H_factor_means: NDArray,
    H_factor_ci: NDArray,
    H_true: float,
    filename: str = "exp3_hurst_vs_N.png",
    title: str = "Figure 4: Estimated Hurst exponents vs number of assets",
) -> Path:
    """
    Plot estimated index and factor Hurst vs N (Figure 4).

    Parameters
    ----------
    N_values : list of int
        Asset counts N.
    H_index_means : NDArray of shape (len(N_values),)
        Mean estimated index H.
    H_index_ci : NDArray of shape (2, len(N_values))
        Lower/upper 95% CI rows for index H.
    H_factor_means : NDArray of shape (len(N_values),)
        Mean estimated factor H.
    H_factor_ci : NDArray of shape (2, len(N_values))
        Lower/upper 95% CI rows for factor H.
    H_true : float
        Theoretical factor H (horizontal reference line).
    filename : str
        Output filename.
    title : str
        Figure title.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    Nx = np.array(N_values)

    ax.axhline(H_true, color="gray", linestyle="--", linewidth=1.5, label=f"True H={H_true}")

    ax.fill_between(Nx, H_index_ci[0], H_index_ci[1], alpha=0.2, color="steelblue")
    ax.plot(Nx, H_index_means, "o-", color="steelblue", label="Estimated index H_I")

    ax.fill_between(Nx, H_factor_ci[0], H_factor_ci[1], alpha=0.2, color="sienna")
    ax.plot(Nx, H_factor_means, "s-", color="sienna", label="Estimated factor H")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of assets N", fontsize=12)
    ax.set_ylabel("Estimated Hurst exponent", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return _save(fig, filename)


def plot_idiosyncratic_hurst_histograms(
    H_i_scenario_a: NDArray,
    H_i_true_a: float,
    H_i_scenario_b: NDArray,
    H_i_true_b: float,
    filename: str = "exp4_idiosyncratic_hurst.png",
    title: str = "Figure 5: Calibration of idiosyncratic Hurst exponents",
) -> Path:
    """
    Plot histograms of estimated idiosyncratic Hurst exponents (Figure 5).

    Parameters
    ----------
    H_i_scenario_a : NDArray
        Estimated H_i values for scenario A.
    H_i_true_a : float
        True H_i for scenario A.
    H_i_scenario_b : NDArray
        Estimated H_i values for scenario B.
    H_i_true_b : float
        True H_i for scenario B.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, h_est, h_true, label in [
        (axes[0], H_i_scenario_a, H_i_true_a, f"H_i={H_i_true_a}"),
        (axes[1], H_i_scenario_b, H_i_true_b, f"H_i={H_i_true_b}"),
    ]:
        bins = np.linspace(0, max(0.2, h_est.max() * 1.2), 30)
        ax.hist(h_est, bins=bins, color="steelblue", alpha=0.7, label="Estimated H_i")
        ax.axvline(h_true, color="red", linestyle="--", linewidth=2, label=f"True H_i={h_true}")
        ax.set_xlabel("Estimated idiosyncratic Hurst H_i", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"True {label}", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    return _save(fig, filename)


def plot_empirical_factor_hurst(
    N_values: list[int],
    H_factor_means: NDArray,
    H_factor_ci_low: NDArray,
    H_factor_ci_high: NDArray,
    H_index: float,
    filename: str = "exp5_empirical_factor_hurst.png",
    title: str = "Figure 6: Estimated factor Hurst H vs number of assets (empirical)",
) -> Path:
    """
    Plot empirical factor Hurst vs N_s with S&P 500 index benchmark (Figure 6).

    Parameters
    ----------
    N_values : list of int
    H_factor_means : NDArray
    H_factor_ci_low : NDArray
    H_factor_ci_high : NDArray
    H_index : float
        Estimated S&P 500 index Hurst (horizontal reference line, ~0.13).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    Nx = np.array(N_values)
    ax.axhline(H_index, color="black", linestyle="-", linewidth=2.0,
               label=f"S&P 500 index H ≈ {H_index:.3f}")

    ax.errorbar(
        Nx,
        H_factor_means,
        yerr=[H_factor_means - H_factor_ci_low, H_factor_ci_high - H_factor_means],
        fmt="o",
        color="steelblue",
        capsize=5,
        label="Estimated factor H (mean ± 95% CI)",
    )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of assets N_s", fontsize=12)
    ax.set_ylabel("Estimated Hurst exponent", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    return _save(fig, filename)


def plot_idiosyncratic_hurst_empirical(
    H_i_approach_a: NDArray,
    H_i_approach_b: NDArray,
    filename: str = "exp6_idiosyncratic_hurst_empirical.png",
    title: str = "Figure 8: Idiosyncratic Hurst exponents (empirical)",
) -> Path:
    """
    Plot empirical idiosyncratic Hurst distributions from two calibration approaches
    (Figure 8a and 8b in the paper).

    Parameters
    ----------
    H_i_approach_a : NDArray
        H_i estimates from full calibration (Approach A).
    H_i_approach_b : NDArray
        H_i estimates from index-sourced factor (Approach B).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, h_est, label in [
        (axes[0], H_i_approach_a, "(a) Full calibration"),
        (axes[1], H_i_approach_b, "(b) S&P 500 GK as factor"),
    ]:
        valid = h_est[~np.isnan(h_est)]
        if len(valid) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue
        bins = np.linspace(0, min(0.5, np.percentile(valid, 99) * 1.2), 40)
        ax.hist(valid, bins=bins, color="steelblue", alpha=0.7)
        ax.axvline(0.0, color="red", linestyle="--", linewidth=1.5, label="H_i=0")
        ax.set_xlabel("Estimated idiosyncratic Hurst H_i", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    return _save(fig, filename)
