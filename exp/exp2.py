"""
Exp 2 — Synthetic validation that index roughness tracks factor roughness across
factor H values (Figure 3, Section 6).

For each true factor Hurst H in {0.08, 0.12, 0.16}, simulates EXP2_MC_REPS
replications of the N-SfFM and estimates the index Hurst via GMM, then compares
estimated index H to the true factor H.

Paper: Section 6 (page 15), Figure 3.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from src import config
from src.calibration import estimate_beta, factor_qv_proxy
from src.hurst_gmm import estimate_hurst_gmm, gaussianize
from src.plotting import plot_factor_h_vs_true_h
from src.simulator import log_qv, simulate_nsfbm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


def run_exp2(seed: int = config.RANDOM_SEED) -> dict:
    """
    Run Experiment 2: estimated index Hurst vs true factor Hurst.

    Simulates the N-SfFM for each H in EXP2_H_VALUES with EXP2_MC_REPS
    replications.  For each replication, Gaussianizes the log index QV and
    estimates the Hurst exponent via GMM.  Summarises with mean and empirical
    95% CI across replications.

    Parameters
    ----------
    seed : int
        Base random seed for reproducibility.

    Returns
    -------
    dict
        Nested dict keyed by H_true (as string) containing mean, ci_low,
        ci_high, and the raw estimates list.
    """
    rng = np.random.default_rng(seed)
    Q_range = (config.GMM_Q_LOW, config.GMM_Q_HIGH)

    metrics: dict[str, dict] = {}

    for H_true in config.EXP2_H_VALUES:
        logger.info("Exp2: H_true=%.2f — starting %d MC reps", H_true, config.EXP2_MC_REPS)
        h_idx_estimates: list[float] = []

        for rep in range(config.EXP2_MC_REPS):
            logger.info("  H_true=%.2f  rep %d/%d", H_true, rep + 1, config.EXP2_MC_REPS)

            sim = simulate_nsfbm(
                N=config.EXP2_N,
                L=config.EXP2_L,
                T=config.EXP2_T,
                s=config.EXP2_S,
                H=H_true,
                lambda2=config.EXP2_LAMBDA2,
                H_i=config.EXP2_H_I,
                lambda2_i=config.EXP2_LAMBDA2_I,
                sigma_i=config.EXP2_SIGMA_I,
                gamma_i=config.EXP2_GAMMA_I,
                beta_a=config.BETA_DIST_A,
                beta_b=config.BETA_DIST_B,
                rng=rng,
            )

            # Use factor log-vol Omega_t as the index log-vol proxy (Section 6 / Figure 3).
            # For N=200 the index is factor-dominated; Omega is the natural log-vol proxy.
            lv_idx = sim.Omega
            gauss_idx = gaussianize(lv_idx - lv_idx.mean())
            H_idx_est, _ = estimate_hurst_gmm(
                gauss_idx,
                T=float(config.EXP2_T),
                Q=None,
                Q_range=Q_range,
                rng=rng,
                n_random_Q=1,
            )

            if not np.isnan(H_idx_est):
                h_idx_estimates.append(float(H_idx_est))
                logger.info("    H_idx_est=%.4f", H_idx_est)
            else:
                logger.warning("    rep %d: NaN H_idx_est — skipped", rep + 1)

        arr = np.array(h_idx_estimates)
        if len(arr) == 0:
            mean_h = float("nan")
            ci_low = float("nan")
            ci_high = float("nan")
        else:
            mean_h = float(np.mean(arr))
            ci_low, ci_high = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

        metrics[str(H_true)] = {
            "mean": mean_h,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "estimates": h_idx_estimates,
        }
        logger.info(
            "  H_true=%.2f → mean H_I=%.4f  95%%CI=[%.4f, %.4f]",
            H_true, mean_h, ci_low, ci_high,
        )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 2: Estimated index H vs true factor H (Figure 3)"
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                        help="Random seed (default: %(default)s)")
    parser.add_argument("--full", action="store_true",
                        help="Use full paper configuration (same as default)")
    args = parser.parse_args()

    metrics = run_exp2(seed=args.seed)

    RESULTS_DIR.mkdir(exist_ok=True)

    # Save metrics JSON
    metrics_path = RESULTS_DIR / "exp2_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # Save CSV table
    rows = []
    for h_key, vals in metrics.items():
        rows.append({
            "H_true": float(h_key),
            "mean_H_index": vals["mean"],
            "ci_low": vals["ci_low"],
            "ci_high": vals["ci_high"],
        })
    df = pd.DataFrame(rows).sort_values("H_true")
    csv_path = RESULTS_DIR / "exp2_table.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Table saved to %s", csv_path)

    # Prepare arrays for plotting (sorted by H_true)
    H_true_vals = sorted(float(k) for k in metrics)
    H_index_means = [metrics[str(h)]["mean"] for h in H_true_vals]
    H_index_ci_low = [metrics[str(h)]["ci_low"] for h in H_true_vals]
    H_index_ci_high = [metrics[str(h)]["ci_high"] for h in H_true_vals]

    plot_factor_h_vs_true_h(
        H_true_vals=H_true_vals,
        H_index_means=H_index_means,
        H_index_ci_low=H_index_ci_low,
        H_index_ci_high=H_index_ci_high,
        filename="exp2_factor_h_vs_true.png",
        title="Figure 3: Estimated index H vs true factor H",
    )
    logger.info("Plot saved to results/exp2_factor_h_vs_true.png")

    # Summary table
    print("\n=== Exp 2 Results ===")
    print(f"{'H_true':>8}  {'mean_H_I':>10}  {'ci_low':>10}  {'ci_high':>10}")
    print("-" * 46)
    for h in H_true_vals:
        v = metrics[str(h)]
        print(f"{h:8.2f}  {v['mean']:10.4f}  {v['ci_low']:10.4f}  {v['ci_high']:10.4f}")


if __name__ == "__main__":
    main()
