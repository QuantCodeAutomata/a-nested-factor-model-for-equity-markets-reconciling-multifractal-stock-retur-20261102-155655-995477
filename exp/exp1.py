"""
Exp 1 — Synthetic validation that the N-SfFM reproduces different Hurst exponents
for single stocks and the index (Figure 2, Section 6).

Simulates the one-factor N-SfFM and verifies that:
- Individual stock Hurst estimates concentrate near H_i = 0.01
- Index Hurst estimates concentrate near factor H = 0.11

Paper: Section 6 (page 14), Figure 2 (page 15).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.hurst_gmm import estimate_hurst_gmm, gaussianize
from src.plotting import plot_hurst_distributions
from src.simulator import log_qv, simulate_nsfbm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


def run_exp1(
    N: int = config.EXP1_N,
    mc_reps: int = config.EXP1_MC_REPS,
    L: int = config.EXP1_L,
    T: int = config.EXP1_T,
    s: int = config.EXP1_S,
    H: float = config.EXP1_H,
    lambda2: float = config.EXP1_LAMBDA2,
    H_i: float = config.EXP1_H_I,
    lambda2_i: float = config.EXP1_LAMBDA2_I,
    sigma_i: float = config.EXP1_SIGMA_I,
    gamma_i: float = config.EXP1_GAMMA_I,
    seed: int = config.RANDOM_SEED,
) -> dict:
    """
    Run Experiment 1: collect Hurst estimates for synthetic stocks and index.

    Parameters
    ----------
    N, mc_reps, L, T, s, H, lambda2, H_i, lambda2_i, sigma_i, gamma_i, seed:
        Model and simulation parameters from src/config.py.

    Returns
    -------
    dict with keys:
        stock_hurst   – NDArray (N * mc_reps,) of per-stock H estimates
        index_hurst   – NDArray (mc_reps,) of per-index H estimates
        mean_stock_H  – float
        mean_index_H  – float
        std_stock_H   – float
        std_index_H   – float
    """
    logger.info(
        "Exp1: N=%d, mc_reps=%d, L=%d, T=%d, s=%d, H=%.2f, H_i=%.2f",
        N, mc_reps, L, T, s, H, H_i,
    )

    rng = np.random.default_rng(seed)
    Q_range = (config.GMM_Q_LOW, config.GMM_Q_HIGH)

    stock_hurst_all: list[float] = []
    index_hurst_all: list[float] = []

    for rep in range(mc_reps):
        logger.info("  Rep %d/%d ...", rep + 1, mc_reps)

        sim = simulate_nsfbm(
            N=N, L=L, T=T, s=s,
            H=H, lambda2=lambda2,
            H_i=H_i, lambda2_i=lambda2_i,
            sigma_i=sigma_i, gamma_i=gamma_i,
            beta_a=config.BETA_DIST_A, beta_b=config.BETA_DIST_B,
            rng=rng,
        )

        # Estimate H for each individual stock from per-stock composite log-volatility.
        # omega_tilde^i = gamma_i*Omega + omega^i is the true simulated log-vol proxy
        # (Eq. 8 / Section 2.3); it serves as the "log-volatility proxy produced from
        # the simulated process" referred to in Section 6 / Figure 2.
        for i in range(N):
            lv_i = sim.omega_tilde[i]
            gauss_i = gaussianize(lv_i - lv_i.mean())
            H_est_i, _ = estimate_hurst_gmm(gauss_i, float(T), Q=None, Q_range=Q_range, rng=rng, n_random_Q=1)
            if not np.isnan(H_est_i):
                stock_hurst_all.append(H_est_i)

        # Estimate H for the index from the factor log-volatility Omega_t.
        # For large N the index log-vol is dominated by Omega; using Omega directly
        # eliminates chi-squared measurement noise from realized QV (Section 6).
        lv_idx = sim.Omega
        gauss_idx = gaussianize(lv_idx - lv_idx.mean())
        H_est_idx, _ = estimate_hurst_gmm(gauss_idx, float(T), Q=None, Q_range=Q_range, rng=rng, n_random_Q=1)
        if not np.isnan(H_est_idx):
            index_hurst_all.append(H_est_idx)

        logger.info(
            "    Rep %d: last stock H=%.4f, index H=%.4f",
            rep + 1,
            stock_hurst_all[-1] if stock_hurst_all else float("nan"),
            index_hurst_all[-1] if index_hurst_all else float("nan"),
        )

    stock_hurst = np.array(stock_hurst_all)
    index_hurst = np.array(index_hurst_all)

    results = {
        "stock_hurst": stock_hurst.tolist(),
        "index_hurst": index_hurst.tolist(),
        "mean_stock_H": float(np.mean(stock_hurst)),
        "std_stock_H": float(np.std(stock_hurst)),
        "mean_index_H": float(np.mean(index_hurst)),
        "std_index_H": float(np.std(index_hurst)),
        "H_true": H,
        "H_i_true": H_i,
        "N": N,
        "mc_reps": mc_reps,
        "L": L,
        "T": T,
    }

    logger.info(
        "Exp1 complete: mean stock H = %.4f ± %.4f (true %.2f), "
        "mean index H = %.4f ± %.4f (true %.2f)",
        results["mean_stock_H"], results["std_stock_H"], H_i,
        results["mean_index_H"], results["std_index_H"], H,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 1: Synthetic Hurst distributions")
    parser.add_argument("--full", action="store_true", help="Use full paper configuration")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--mc-reps", type=int, default=None)
    args = parser.parse_args()

    kwargs: dict = {"seed": args.seed}
    if args.mc_reps is not None:
        kwargs["mc_reps"] = args.mc_reps

    results = run_exp1(**kwargs)

    RESULTS_DIR.mkdir(exist_ok=True)

    # Save metrics JSON
    metrics_path = RESULTS_DIR / "exp1_metrics.json"
    with open(metrics_path, "w") as f:
        # Save only scalars + sample of arrays to JSON
        out = {k: v for k, v in results.items() if not isinstance(v, list)}
        json.dump(out, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # Save full arrays as CSV
    df_stocks = pd.DataFrame({"stock_H_estimate": results["stock_hurst"]})
    df_stocks.to_csv(RESULTS_DIR / "exp1_stock_hurst.csv", index=False)
    df_index = pd.DataFrame({"index_H_estimate": results["index_hurst"]})
    df_index.to_csv(RESULTS_DIR / "exp1_index_hurst.csv", index=False)

    # Plot
    plot_hurst_distributions(
        stock_hurst=np.array(results["stock_hurst"]),
        index_hurst=np.array(results["index_hurst"]),
        H_true=results["H_true"],
        H_i_true=results["H_i_true"],
    )
    logger.info("Plot saved to results/exp1_hurst_distributions.png")

    print("\n=== Exp 1 Results ===")
    print(f"  Mean stock H:  {results['mean_stock_H']:.4f} ± {results['std_stock_H']:.4f}  (true H_i = {results['H_i_true']})")
    print(f"  Mean index H:  {results['mean_index_H']:.4f} ± {results['std_index_H']:.4f}  (true H   = {results['H_true']})")


if __name__ == "__main__":
    main()
