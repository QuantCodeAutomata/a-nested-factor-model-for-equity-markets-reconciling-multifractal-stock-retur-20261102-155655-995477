"""
Exp 5 — Empirical estimation of factor roughness from S&P500 constituent panels
(Figure 6, Section 7.1).

For each basket size N_s in {2, 4, 8, 16, 32, 64, 128, 256}, draw 20 random
combinations of N_s stocks from the available universe.  For each combination
estimate the factor Hurst exponent 20 times (one per random Q draw) then take
the mean; this yields one H value per combination.  Collect 20 such values per
N_s and compute 95 % confidence intervals.  Also estimates the S&P 500 index
Hurst from the same GMM.

Paper: Section 7.1 (page 18), Figure 6.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from data.data_loader import (
    SP500_LONG_TERM_TICKERS,
    build_gk_panel,
    build_return_panel,
    garman_klass,
    load_sp500_data_exp5,
)
from src import config
from src.calibration import estimate_beta, factor_qv_proxy
from src.hurst_gmm import estimate_hurst_gmm, gaussianize
from src.plotting import plot_empirical_factor_hurst

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


def run_exp5(seed: int = config.RANDOM_SEED) -> dict:
    """
    Run Experiment 5: empirical factor Hurst estimation from S&P500 panels.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        N_s, mean_H, ci_low, ci_high, H_index
    """
    api_key = os.environ.get("DATALOADER_API_KEY", "")
    if not api_key:
        raise RuntimeError("DATALOADER_API_KEY not set. Set it in .env file.")

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    logger.info("exp5: loading S&P500 data %s to %s ...",
                config.EXP5_START_DATE, config.EXP5_END_DATE)
    stocks_ohlcv, index_ohlcv = load_sp500_data_exp5(
        config.EXP5_START_DATE,
        config.EXP5_END_DATE,
        SP500_LONG_TERM_TICKERS[:150],
    )

    gk_panel = build_gk_panel(stocks_ohlcv, config.EXP5_START_DATE, config.EXP5_END_DATE)
    ret_panel = build_return_panel(stocks_ohlcv, config.EXP5_START_DATE, config.EXP5_END_DATE)

    # Drop any ticker with NaN GK values
    gk_panel = gk_panel.dropna(axis=1)
    ret_panel = ret_panel[gk_panel.columns].dropna(axis=1)

    # Align on common dates
    common_dates = gk_panel.index.intersection(ret_panel.index)
    gk_panel = gk_panel.loc[common_dates]
    ret_panel = ret_panel.loc[common_dates]

    available_tickers = list(gk_panel.columns)
    logger.info("Available tickers after NaN filter: %d", len(available_tickers))

    # S&P500 index GK
    gk_index = garman_klass(index_ohlcv).reindex(common_dates).dropna()
    T_empirical = len(gk_index)
    logger.info("exp5: T_empirical=%d", T_empirical)

    # Estimate S&P500 index Hurst exponent
    log_gk_idx = np.log(gk_index.values.clip(min=1e-12))
    gauss_idx = gaussianize(log_gk_idx - log_gk_idx.mean())
    H_index, _ = estimate_hurst_gmm(
        gauss_idx,
        T=float(T_empirical),
        Q=None,
        Q_range=(config.EXP5_GMM_Q_LOW, config.EXP5_GMM_Q_HIGH),
        rng=rng,
        n_random_Q=20,
    )
    logger.info("exp5: S&P500 index H = %.4f", H_index)

    # ------------------------------------------------------------------
    # Step 2: Loop over basket sizes and combinations
    # ------------------------------------------------------------------
    mean_H_list: list[float] = []
    ci_low_list: list[float] = []
    ci_high_list: list[float] = []

    for N_s in config.EXP5_N_VALUES:
        if N_s > len(available_tickers):
            logger.warning("N_s=%d exceeds available tickers (%d); skipping.",
                           N_s, len(available_tickers))
            mean_H_list.append(float("nan"))
            ci_low_list.append(float("nan"))
            ci_high_list.append(float("nan"))
            continue

        logger.info("exp5: N_s = %d", N_s)
        H_factor_combos: list[float] = []

        for combo in range(config.EXP5_N_COMBOS):
            sampled = rng.choice(available_tickers, size=N_s, replace=False).tolist()

            gk_sub = gk_panel[sampled].values.T      # (N_s, T)
            ret_sub = ret_panel[sampled].values.T     # (N_s, T)

            beta = estimate_beta(ret_sub)
            fqv = factor_qv_proxy(gk_sub, beta)

            log_fqv = np.log(fqv.clip(min=1e-12))
            Omega = gaussianize(log_fqv - log_fqv.mean())

            # 20 random Q draws → mean H for this combination
            H_factor_list: list[float] = []
            for _ in range(20):
                q = int(rng.integers(config.EXP5_GMM_Q_LOW, config.EXP5_GMM_Q_HIGH + 1))
                H_q, _ = estimate_hurst_gmm(
                    Omega, T=float(T_empirical), Q=q, rng=rng, n_random_Q=1
                )
                if not np.isnan(H_q):
                    H_factor_list.append(H_q)

            H_factor_combo = float(np.mean(H_factor_list)) if H_factor_list else float("nan")
            H_factor_combos.append(H_factor_combo)
            logger.info("  N_s=%d combo %d/%d: H_factor=%.4f",
                        N_s, combo + 1, config.EXP5_N_COMBOS, H_factor_combo)

        valid = np.array([v for v in H_factor_combos if not np.isnan(v)])
        if len(valid) == 0:
            logger.warning("N_s=%d: all combos returned NaN.", N_s)
            mean_H_list.append(float("nan"))
            ci_low_list.append(float("nan"))
            ci_high_list.append(float("nan"))
        else:
            mean_H_list.append(float(np.mean(valid)))
            lo, hi = np.percentile(valid, [2.5, 97.5])
            ci_low_list.append(float(lo))
            ci_high_list.append(float(hi))
            logger.info("N_s=%d: mean H=%.4f  [%.4f, %.4f]",
                        N_s, mean_H_list[-1], ci_low_list[-1], ci_high_list[-1])

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(exist_ok=True)

    metrics = {
        "N_s": config.EXP5_N_VALUES,
        "mean_H": mean_H_list,
        "ci_low": ci_low_list,
        "ci_high": ci_high_list,
        "H_index": float(H_index),
    }
    metrics_path = RESULTS_DIR / "exp5_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    table_df = pd.DataFrame({
        "N_s": config.EXP5_N_VALUES,
        "mean_H_factor": mean_H_list,
        "ci_low": ci_low_list,
        "ci_high": ci_high_list,
    })
    table_path = RESULTS_DIR / "exp5_table.csv"
    table_df.to_csv(table_path, index=False)
    logger.info("Table saved to %s", table_path)

    # Plot
    plot_empirical_factor_hurst(
        N_values=config.EXP5_N_VALUES,
        H_factor_means=np.array(mean_H_list),
        H_factor_ci_low=np.array(ci_low_list),
        H_factor_ci_high=np.array(ci_high_list),
        H_index=float(H_index),
    )
    logger.info("Plot saved to results/exp5_empirical_factor_hurst.png")

    print("\n=== Exp 5 Results ===")
    print(f"  S&P500 index H = {H_index:.4f}")
    for i, N_s in enumerate(config.EXP5_N_VALUES):
        print(f"  N_s={N_s:>3d}: mean H = {mean_H_list[i]:.4f}  "
              f"[{ci_low_list[i]:.4f}, {ci_high_list[i]:.4f}]")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 5: Empirical factor roughness from S&P500 panels"
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                        help="Random seed (default: %(default)s)")
    parser.add_argument("--full", action="store_true",
                        help="Use full paper configuration (no effect; all params from config)")
    args = parser.parse_args()

    run_exp5(seed=args.seed)


if __name__ == "__main__":
    main()
