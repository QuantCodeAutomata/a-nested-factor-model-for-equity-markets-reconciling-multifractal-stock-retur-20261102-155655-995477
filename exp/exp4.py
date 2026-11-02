"""
Exp 4 — Synthetic calibration accuracy for idiosyncratic Hurst exponents (Figure 5, Section 6).

Simulates the one-factor N-SfFM for two idiosyncratic Hurst scenarios (H_i = 0.03 and 0.07)
and runs the full calibration pipeline to assess how accurately the per-asset idiosyncratic
Hurst exponents are recovered from synthetic data.

Paper: Section 6 (page 17), Figure 5.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src import config
from src.calibration import run_calibration
from src.plotting import plot_idiosyncratic_hurst_histograms
from src.simulator import simulate_nsfbm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


def run_exp4(seed: int = config.RANDOM_SEED) -> dict:
    """
    Run Experiment 4: calibration accuracy for idiosyncratic Hurst exponents.

    For each of two H_i scenarios (0.03 and 0.07), simulates the N-SfFM model
    with N=300 assets and L=32768 time steps, then runs the full calibration
    pipeline (including 300 individual GMM calls) to estimate per-asset H_i.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys "scenario_A" and "scenario_B", each containing:
        mean_H_i  – float, mean of estimated H_i across assets
        std_H_i   – float, std of estimated H_i across assets
        H_i_true  – float, the true idiosyncratic Hurst exponent used
    """
    rng = np.random.default_rng(seed)

    scenarios: list[tuple[float, str]] = [
        (config.EXP4_H_I_VALUES[0], "A"),
        (config.EXP4_H_I_VALUES[1], "B"),
    ]

    collected_H_i: dict[str, NDArray] = {}

    for H_i_true, label in scenarios:
        logger.info(
            "Exp4 Scenario %s: H_i_true=%.2f, N=%d, L=%d, T=%d, MC_REPS=%d",
            label, H_i_true, config.EXP4_N, config.EXP4_L, config.EXP4_T,
            config.EXP4_MC_REPS,
        )

        reps_H_i: list[NDArray] = []

        for rep in range(config.EXP4_MC_REPS):
            logger.info("  Scenario %s rep %d/%d — simulating ...", label, rep + 1, config.EXP4_MC_REPS)

            sim = simulate_nsfbm(
                N=config.EXP4_N,
                L=config.EXP4_L,
                T=config.EXP4_T,
                s=config.EXP4_S,
                H=config.EXP4_H,
                lambda2=config.EXP4_LAMBDA2,
                H_i=H_i_true,
                lambda2_i=config.EXP4_LAMBDA2_I,
                sigma_i=config.EXP4_SIGMA_I,
                gamma_i=config.EXP4_GAMMA_I,
                beta_a=config.BETA_DIST_A,
                beta_b=config.BETA_DIST_B,
                rng=rng,
            )

            logger.info(
                "  Scenario %s rep %d/%d — calibrating (%d GMM calls) ...",
                label, rep + 1, config.EXP4_MC_REPS, config.EXP4_N,
            )

            cal = run_calibration(
                sim.returns_stocks,
                sim.qv_stocks,
                T=float(config.EXP4_T),
                Q_range=(config.GMM_Q_LOW, config.GMM_Q_HIGH),
                rng=rng,
            )

            reps_H_i.append(cal.H_i)

            logger.info(
                "  Scenario %s rep %d/%d done: mean H_i=%.4f, std=%.4f",
                label, rep + 1, config.EXP4_MC_REPS,
                float(np.mean(cal.H_i)), float(np.std(cal.H_i)),
            )

        scenario_H_i = np.concatenate(reps_H_i)
        collected_H_i[label] = scenario_H_i

        logger.info(
            "Scenario %s complete: mean H_i=%.4f ± %.4f (true=%.2f, n=%d)",
            label, float(np.mean(scenario_H_i)), float(np.std(scenario_H_i)),
            H_i_true, len(scenario_H_i),
        )

    scenario_A_H_i = collected_H_i["A"]
    scenario_B_H_i = collected_H_i["B"]
    H_i_true_a = config.EXP4_H_I_VALUES[0]
    H_i_true_b = config.EXP4_H_I_VALUES[1]

    # --- Save results ---
    RESULTS_DIR.mkdir(exist_ok=True)

    metrics = {
        "scenario_A": {
            "H_i_true": H_i_true_a,
            "mean_H_i": float(np.mean(scenario_A_H_i)),
            "std_H_i": float(np.std(scenario_A_H_i)),
            "N": len(scenario_A_H_i),
        },
        "scenario_B": {
            "H_i_true": H_i_true_b,
            "mean_H_i": float(np.mean(scenario_B_H_i)),
            "std_H_i": float(np.std(scenario_B_H_i)),
            "N": len(scenario_B_H_i),
        },
    }

    metrics_path = RESULTS_DIR / "exp4_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    pd.DataFrame({"H_i_estimate": scenario_A_H_i}).to_csv(
        RESULTS_DIR / "exp4_scenario_A.csv", index=False
    )
    pd.DataFrame({"H_i_estimate": scenario_B_H_i}).to_csv(
        RESULTS_DIR / "exp4_scenario_B.csv", index=False
    )

    plot_idiosyncratic_hurst_histograms(
        H_i_scenario_a=scenario_A_H_i,
        H_i_true_a=H_i_true_a,
        H_i_scenario_b=scenario_B_H_i,
        H_i_true_b=H_i_true_b,
    )
    logger.info("Plot saved to results/exp4_idiosyncratic_hurst.png")

    # --- Summary table ---
    print("\n=== Exp 4 Results: Idiosyncratic Hurst Calibration Accuracy ===")
    print(f"{'Scenario':<12} {'H_i true':>10} {'mean H_i':>10} {'std H_i':>10} {'N':>6}")
    print("-" * 52)
    for sc_label, sc_key, h_true in [("A", "scenario_A", H_i_true_a), ("B", "scenario_B", H_i_true_b)]:
        m = metrics[sc_key]
        print(
            f"{'Scenario ' + sc_label:<12} {h_true:>10.3f} {m['mean_H_i']:>10.4f} {m['std_H_i']:>10.4f} {m['N']:>6}"
        )

    return {
        "scenario_A": {
            "mean_H_i": metrics["scenario_A"]["mean_H_i"],
            "std_H_i": metrics["scenario_A"]["std_H_i"],
            "H_i_true": H_i_true_a,
        },
        "scenario_B": {
            "mean_H_i": metrics["scenario_B"]["mean_H_i"],
            "std_H_i": metrics["scenario_B"]["std_H_i"],
            "H_i_true": H_i_true_b,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 4: Synthetic calibration accuracy for idiosyncratic Hurst exponents"
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed")
    parser.add_argument(
        "--full", action="store_true",
        help="Use full paper configuration (no effect: Exp 4 always uses paper parameters)",
    )
    args = parser.parse_args()

    run_exp4(seed=args.seed)


if __name__ == "__main__":
    main()
