"""
Exp 3 — Synthetic study of convergence with number of assets (Figure 4, Section 6).

For each true factor Hurst H in EXP3_H_VALUES and each basket size N in
EXP3_N_VALUES, simulates EXP3_MC_REPS replications of the N-SfFM.  For each
replication, estimates both the index Hurst (from the equal-weight index QV)
and the factor Hurst via the calibration proxy (Eq. 44).  Summarises convergence
across N with mean and empirical 95% CI.

Paper: Section 6 (page 16-17), Figure 4.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.calibration import estimate_beta, factor_qv_proxy
from src.hurst_gmm import estimate_hurst_gmm, gaussianize
from src.plotting import plot_hurst_vs_N
from src.simulator import log_qv, simulate_nsfbm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


def run_exp3(seed: int = config.RANDOM_SEED) -> dict:
    """
    Run Experiment 3: convergence of estimated Hurst exponents with basket size N.

    For each (H_true, N) combination, runs EXP3_MC_REPS replications of the
    N-SfFM and estimates:
      - Index Hurst via GMM on Gaussianized log index QV.
      - Factor Hurst via the calibration proxy (beta estimation → factor QV →
        Gaussianize → GMM).

    Parameters
    ----------
    seed : int
        Base random seed for reproducibility.

    Returns
    -------
    dict
        Nested dict keyed by H_true (str) → N (str) → summary stats.
    """
    rng = np.random.default_rng(seed)
    Q_range = (config.GMM_Q_LOW, config.GMM_Q_HIGH)

    # Outer structure: H_true → N → stats
    metrics: dict[str, dict[str, dict]] = {}

    for H_true in config.EXP3_H_VALUES:
        h_key = str(H_true)
        metrics[h_key] = {}

        for N in config.EXP3_N_VALUES:
            n_key = str(N)
            logger.info(
                "Exp3: H_true=%.2f  N=%d — starting %d MC reps",
                H_true, N, config.EXP3_MC_REPS,
            )

            h_index_estimates: list[float] = []
            h_factor_estimates: list[float] = []

            for rep in range(config.EXP3_MC_REPS):
                logger.info(
                    "  H_true=%.2f  N=%d  rep %d/%d",
                    H_true, N, rep + 1, config.EXP3_MC_REPS,
                )

                sim = simulate_nsfbm(
                    N=N,
                    L=config.EXP3_L,
                    T=config.EXP3_T,
                    s=config.EXP3_S,
                    H=H_true,
                    lambda2=config.EXP3_LAMBDA2,
                    H_i=config.EXP3_H_I,
                    lambda2_i=config.EXP3_LAMBDA2_I,
                    sigma_i=config.EXP3_SIGMA_I,
                    gamma_i=config.EXP3_GAMMA_I,
                    beta_a=config.BETA_DIST_A,
                    beta_b=config.BETA_DIST_B,
                    rng=rng,
                )

                # --- Index Hurst estimate ---
                lqv_idx = log_qv(sim.qv_index)
                gauss_idx = gaussianize(lqv_idx - lqv_idx.mean())
                H_index_est, _ = estimate_hurst_gmm(
                    gauss_idx,
                    T=float(config.EXP3_T),
                    Q=None,
                    Q_range=Q_range,
                    rng=rng,
                    n_random_Q=1,
                )

                # --- Factor Hurst estimate via proxy ---
                beta_hat = estimate_beta(sim.returns_stocks)
                fqv = factor_qv_proxy(sim.qv_stocks, beta_hat)
                log_fqv = log_qv(fqv)
                omega = gaussianize(log_fqv - log_fqv.mean())
                H_factor_est, _ = estimate_hurst_gmm(
                    omega,
                    T=float(config.EXP3_T),
                    Q=None,
                    Q_range=Q_range,
                    rng=rng,
                    n_random_Q=1,
                )

                if not np.isnan(H_index_est):
                    h_index_estimates.append(float(H_index_est))
                else:
                    logger.warning(
                        "    rep %d: NaN H_index_est — skipped", rep + 1
                    )

                if not np.isnan(H_factor_est):
                    h_factor_estimates.append(float(H_factor_est))
                else:
                    logger.warning(
                        "    rep %d: NaN H_factor_est — skipped", rep + 1
                    )

                logger.info(
                    "    H_index_est=%.4f  H_factor_est=%.4f",
                    H_index_est if not np.isnan(H_index_est) else float("nan"),
                    H_factor_est if not np.isnan(H_factor_est) else float("nan"),
                )

            # Summarise across reps
            arr_idx = np.array(h_index_estimates)
            arr_fac = np.array(h_factor_estimates)

            if len(arr_idx) == 0:
                idx_mean = idx_ci_low = idx_ci_high = float("nan")
            else:
                idx_mean = float(np.mean(arr_idx))
                idx_ci_low = float(np.percentile(arr_idx, 2.5))
                idx_ci_high = float(np.percentile(arr_idx, 97.5))

            if len(arr_fac) == 0:
                fac_mean = fac_ci_low = fac_ci_high = float("nan")
            else:
                fac_mean = float(np.mean(arr_fac))
                fac_ci_low = float(np.percentile(arr_fac, 2.5))
                fac_ci_high = float(np.percentile(arr_fac, 97.5))

            metrics[h_key][n_key] = {
                "H_index_mean": idx_mean,
                "H_index_ci_low": idx_ci_low,
                "H_index_ci_high": idx_ci_high,
                "H_factor_mean": fac_mean,
                "H_factor_ci_low": fac_ci_low,
                "H_factor_ci_high": fac_ci_high,
                "H_index_estimates": h_index_estimates,
                "H_factor_estimates": h_factor_estimates,
            }
            logger.info(
                "  N=%d summary: H_index=%.4f [%.4f,%.4f]  H_factor=%.4f [%.4f,%.4f]",
                N, idx_mean, idx_ci_low, idx_ci_high,
                fac_mean, fac_ci_low, fac_ci_high,
            )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 3: Convergence of Hurst estimates with number of assets (Figure 4)"
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                        help="Random seed (default: %(default)s)")
    args = parser.parse_args()

    metrics = run_exp3(seed=args.seed)

    RESULTS_DIR.mkdir(exist_ok=True)

    # Save metrics JSON
    metrics_path = RESULTS_DIR / "exp3_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # Save CSV table
    rows = []
    for h_key, n_dict in metrics.items():
        for n_key, vals in n_dict.items():
            rows.append({
                "H_true": float(h_key),
                "N": int(n_key),
                "H_index_mean": vals["H_index_mean"],
                "H_index_ci_low": vals["H_index_ci_low"],
                "H_index_ci_high": vals["H_index_ci_high"],
                "H_factor_mean": vals["H_factor_mean"],
                "H_factor_ci_low": vals["H_factor_ci_low"],
                "H_factor_ci_high": vals["H_factor_ci_high"],
            })
    df = pd.DataFrame(rows).sort_values(["H_true", "N"])
    csv_path = RESULTS_DIR / "exp3_table.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Table saved to %s", csv_path)

    # Plot one figure per H_true value
    for H_true in config.EXP3_H_VALUES:
        h_key = str(H_true)
        N_values = config.EXP3_N_VALUES

        H_index_means = np.array([
            metrics[h_key][str(N)]["H_index_mean"] for N in N_values
        ])
        H_index_ci_lows = np.array([
            metrics[h_key][str(N)]["H_index_ci_low"] for N in N_values
        ])
        H_index_ci_highs = np.array([
            metrics[h_key][str(N)]["H_index_ci_high"] for N in N_values
        ])
        H_factor_means = np.array([
            metrics[h_key][str(N)]["H_factor_mean"] for N in N_values
        ])
        H_factor_ci_lows = np.array([
            metrics[h_key][str(N)]["H_factor_ci_low"] for N in N_values
        ])
        H_factor_ci_highs = np.array([
            metrics[h_key][str(N)]["H_factor_ci_high"] for N in N_values
        ])

        plot_hurst_vs_N(
            N_values=N_values,
            H_index_means=H_index_means,
            H_index_ci=np.array([H_index_ci_lows, H_index_ci_highs]),
            H_factor_means=H_factor_means,
            H_factor_ci=np.array([H_factor_ci_lows, H_factor_ci_highs]),
            H_true=H_true,
            filename=f"exp3_hurst_vs_N_H{H_true:.2f}.png",
            title=f"Figure 4: Estimated Hurst vs N (H_true={H_true})",
        )
        logger.info("Plot saved to results/exp3_hurst_vs_N_H%.2f.png", H_true)

    # Summary table
    print("\n=== Exp 3 Results ===")
    for H_true in config.EXP3_H_VALUES:
        h_key = str(H_true)
        print(f"\n  H_true = {H_true}")
        print(f"  {'N':>5}  {'H_idx_mean':>12}  {'idx_CI':>20}  {'H_fac_mean':>12}  {'fac_CI':>20}")
        print("  " + "-" * 78)
        for N in config.EXP3_N_VALUES:
            v = metrics[h_key][str(N)]
            print(
                f"  {N:5d}  {v['H_index_mean']:12.4f}  "
                f"[{v['H_index_ci_low']:.4f},{v['H_index_ci_high']:.4f}]  "
                f"{v['H_factor_mean']:12.4f}  "
                f"[{v['H_factor_ci_low']:.4f},{v['H_factor_ci_high']:.4f}]"
            )


if __name__ == "__main__":
    main()
