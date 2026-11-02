"""
Exp 6 — Empirical calibration of idiosyncratic roughness for long-horizon S&P500 stocks
(Figure 8, Section 7.2).

Loads 1994-2026 daily OHLC for the 243 long-term S&P500 constituents and runs
two calibration approaches:
  A) Full pipeline — estimate beta, factor QV, Omega, H_factor, residuals, H_i.
  B) Use S&P500 GK log-vol as Omega directly (bypass Steps 1-3), then
     estimate residuals → omega^i → H_i.

Paper: Section 7.2 (pages 19-21), Figure 8.
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
    load_sp500_data_exp6,
)
from src import config
from src.calibration import run_calibration
from src.hurst_gmm import gaussianize
from src.plotting import plot_idiosyncratic_hurst_empirical

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


def run_exp6(seed: int = config.RANDOM_SEED) -> dict:
    """
    Run Experiment 6: empirical idiosyncratic Hurst calibration from long-horizon
    S&P500 data (1994-2026).

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        N_stocks, T, approach_A, approach_B, Omega_correlation
    """
    api_key = os.environ.get("DATALOADER_API_KEY", "")
    if not api_key:
        raise RuntimeError("DATALOADER_API_KEY not set. Set it in .env file.")

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    logger.info("exp6: loading S&P500 data %s to %s ...",
                config.EXP6_START_DATE, config.EXP6_END_DATE)
    stocks_ohlcv, index_ohlcv = load_sp500_data_exp6(
        config.EXP6_START_DATE,
        config.EXP6_END_DATE,
        SP500_LONG_TERM_TICKERS,
    )

    gk_panel = build_gk_panel(stocks_ohlcv, config.EXP6_START_DATE, config.EXP6_END_DATE)
    ret_panel = build_return_panel(stocks_ohlcv, config.EXP6_START_DATE, config.EXP6_END_DATE)

    # Drop any ticker with any NaN GK value
    gk_panel = gk_panel.dropna(axis=1)
    ret_panel = ret_panel[gk_panel.columns].dropna(axis=1)

    # Align on common dates, then drop any row with NaN
    common_dates = gk_panel.index.intersection(ret_panel.index)
    gk_panel = gk_panel.loc[common_dates].dropna(axis=0)
    ret_panel = ret_panel.loc[common_dates].dropna(axis=0)
    common_dates = gk_panel.index.intersection(ret_panel.index)
    gk_panel = gk_panel.loc[common_dates]
    ret_panel = ret_panel.loc[common_dates]

    T_long = len(common_dates)

    # S&P500 index GK → Omega_index
    gk_index = garman_klass(index_ohlcv).reindex(common_dates)
    log_gk_idx = np.log(gk_index.values.clip(min=1e-12))
    Omega_index = gaussianize(log_gk_idx - log_gk_idx.mean())

    gk_arr = gk_panel.values.T    # (N, T)
    ret_arr = ret_panel.values.T  # (N, T)
    tickers = list(gk_panel.columns)

    N_stocks = gk_arr.shape[0]
    logger.info("exp6: N_stocks=%d, T=%d", N_stocks, T_long)

    # ------------------------------------------------------------------
    # Step 2: Approach A — full calibration
    # ------------------------------------------------------------------
    logger.info("exp6: running Approach A (full calibration) ...")
    cal_A = run_calibration(
        ret_arr,
        gk_arr,
        T=float(T_long),
        Omega_external=None,
        Q_range=(config.GMM_Q_LOW, config.GMM_Q_HIGH),
        rng=rng,
    )
    H_i_A: np.ndarray = cal_A.H_i  # (N,)

    corr_Omega = float(np.corrcoef(cal_A.Omega, Omega_index)[0, 1])
    logger.info("exp6: Omega correlation with S&P500 GK: %.4f", corr_Omega)

    # ------------------------------------------------------------------
    # Step 3: Approach B — use index GK as factor log-vol
    # ------------------------------------------------------------------
    logger.info("exp6: running Approach B (index GK as factor) ...")
    cal_B = run_calibration(
        ret_arr,
        gk_arr,
        T=float(T_long),
        Omega_external=Omega_index,
        Q_range=(config.GMM_Q_LOW, config.GMM_Q_HIGH),
        rng=rng,
    )
    H_i_B: np.ndarray = cal_B.H_i  # (N,)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(exist_ok=True)

    def _summarize(h_arr: np.ndarray) -> dict:
        valid = h_arr[~np.isnan(h_arr)]
        return {
            "mean_H_i": float(np.mean(valid)) if len(valid) else float("nan"),
            "std_H_i": float(np.std(valid)) if len(valid) else float("nan"),
            "pct_near_zero": float(np.mean(valid < 0.05)) if len(valid) else float("nan"),
        }

    metrics = {
        "N_stocks": N_stocks,
        "T": T_long,
        "approach_A": _summarize(H_i_A),
        "approach_B": _summarize(H_i_B),
        "Omega_correlation": corr_Omega,
    }
    metrics_path = RESULTS_DIR / "exp6_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    df_A = pd.DataFrame({"ticker": tickers, "H_i": H_i_A.tolist()})
    df_A.to_csv(RESULTS_DIR / "exp6_approach_A.csv", index=False)

    df_B = pd.DataFrame({"ticker": tickers, "H_i": H_i_B.tolist()})
    df_B.to_csv(RESULTS_DIR / "exp6_approach_B.csv", index=False)

    logger.info("CSVs saved to results/exp6_approach_A.csv and exp6_approach_B.csv")

    # Plot
    plot_idiosyncratic_hurst_empirical(
        H_i_approach_a=H_i_A,
        H_i_approach_b=H_i_B,
    )
    logger.info("Plot saved to results/exp6_idiosyncratic_hurst_empirical.png")

    print("\n=== Exp 6 Results ===")
    print(f"  N_stocks = {N_stocks}, T = {T_long}")
    print(f"  Omega correlation (A vs index GK): {corr_Omega:.4f}")
    for label, summ in [("Approach A (full)", metrics["approach_A"]),
                        ("Approach B (index GK)", metrics["approach_B"])]:
        print(f"  {label}: mean H_i={summ['mean_H_i']:.4f}  "
              f"std={summ['std_H_i']:.4f}  "
              f"pct_near_zero={summ['pct_near_zero']:.3f}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 6: Empirical idiosyncratic roughness for long-horizon S&P500 stocks"
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                        help="Random seed (default: %(default)s)")
    parser.add_argument("--full", action="store_true",
                        help="Use full paper configuration (no effect; all params from config)")
    args = parser.parse_args()

    run_exp6(seed=args.seed)


if __name__ == "__main__":
    main()
