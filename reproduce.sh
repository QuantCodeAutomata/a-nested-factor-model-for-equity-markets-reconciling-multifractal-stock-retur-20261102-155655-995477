#!/usr/bin/env bash
# reproduce.sh — run all experiments and tests for the N-SfFM replication.
# Usage: bash reproduce.sh
# Experiments 1-4 require no API keys.
# Experiments 5-6 require DATALOADER_API_KEY and MASSIVE_API_KEY set in .env.
set -u

echo "=== N-SfFM Replication ==="
echo "Results will be written to results/"

echo ""
echo "--- Unit tests (no API key required) ---"
python -m pytest -m "not market_data" -v

echo ""
echo "--- Exp 1: Figure 2 — stock vs index Hurst distributions ---"
python -m exp.exp1

echo ""
echo "--- Exp 2: Figure 3 — index Hurst vs factor H ---"
python -m exp.exp2

echo ""
echo "--- Exp 3: Figure 4 — convergence with number of assets ---"
python -m exp.exp3

echo ""
echo "--- Exp 4: Figure 5 — idiosyncratic Hurst calibration ---"
python -m exp.exp4

if [ -n "${DATALOADER_API_KEY:-}" ] && [ -n "${MASSIVE_API_KEY:-}" ]; then
    echo ""
    echo "--- Exp 5: Figure 6 — factor roughness vs panel size (empirical) ---"
    python -m exp.exp5

    echo ""
    echo "--- Exp 6: Figure 8 — idiosyncratic Hurst for S&P 500 stocks (empirical) ---"
    python -m exp.exp6
else
    echo ""
    echo "Skipping Exp 5 and Exp 6: DATALOADER_API_KEY and/or MASSIVE_API_KEY not set."
fi

echo ""
echo "=== Done. Check results/RESULTS.md for comparison with paper values. ==="
