# A Nested Factor Model for Equity Markets: Reconciling Multifractal Stock Returns and Rough Index Volatilities

**Authors:** Othmane Zarhali, Cecilia Aubrun, Emmanuel Bacry, Jean-Philippe Bouchaud, Jean-François Muzy  
**arXiv:** [2505.02678v5](https://arxiv.org/abs/2505.02678)

This repository reproduces the synthetic validation experiments (Sections 6, Figures 2–5) and the empirical experiments (Section 7, Figures 6 and 8) from the above paper. The paper introduces the Nested Log S-fBM Factor Model (N-SfFM), which reconciles the empirical observation that single-stock volatilities are rougher than index volatilities by embedding idiosyncratic rough volatility within a one-factor structure driven by a smoother common factor.

---

## Repository layout

```
src/                # Library modules (sfbm, simulator, hurst_gmm, calibration, plotting, config)
exp/                # Experiment scripts (exp1.py – exp6.py)
tests/              # Pytest test suite
data/               # Data-loading code (data_loader.py); downloaded data never committed
results/            # All result artifacts (plots, CSVs, JSON metrics, logs) and RESULTS.md
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then add your DATALOADER_API_KEY and MASSIVE_API_KEY
```

Experiments 1–4 use only synthetic data and need no API keys. Experiments 5 and 6 load daily OHLC data via the QuantX Dataloader and require both `DATALOADER_API_KEY` (QuantX) and `MASSIVE_API_KEY` (Massive).

---

## Running experiments

All commands are run from the repository root.

### Synthetic experiments (no API key required)

```bash
# Exp 1 – Figure 2: stock vs index Hurst distributions (N=100, 50 replications)
python -m exp.exp1

# Exp 2 – Figure 3: index Hurst vs factor H for H ∈ {0.08, 0.12, 0.16} (N=200, 20 replications)
python -m exp.exp2

# Exp 3 – Figure 4: convergence with N ∈ {2,4,8,16,32,64,128,256} (20 replications)
python -m exp.exp3

# Exp 4 – Figure 5: idiosyncratic Hurst calibration for H_i ∈ {0.03, 0.07} (N=300)
python -m exp.exp4
```

To run the full paper configuration for exp 3 (all N values, two H values, 20 replications per cell):

```bash
python -m exp.exp3 --full
```

### Empirical experiments (DATALOADER_API_KEY and MASSIVE_API_KEY required)

```bash
# Exp 5 – Figure 6: factor roughness vs N_s from S&P 500 panels (2016–2020)
python -m exp.exp5

# Exp 6 – Figure 8: idiosyncratic Hurst for long-horizon S&P 500 stocks (1994–2026)
python -m exp.exp6
```

### Tests

```bash
# Unit tests only (no API key required)
python -m pytest -m "not market_data" -v

# Full suite including market-data integration tests
python -m pytest -v
```

### Reproduce all experiments

```bash
bash reproduce.sh
```

---

## Results

All output files are written to `results/`:

| Experiment | Figures / Tables | Key output files |
|---|---|---|
| Exp 1 | Figure 2 | `exp1_hurst_distributions.png`, `exp1_metrics.json` |
| Exp 2 | Figure 3 | `exp2_factor_h_vs_true.png`, `exp2_metrics.json` |
| Exp 3 | Figure 4 | `exp3_hurst_vs_N_H0.08.png`, `exp3_hurst_vs_N_H0.11.png`, `exp3_metrics.json` |
| Exp 4 | Figure 5 | `exp4_idiosyncratic_hurst.png`, `exp4_metrics.json` |
| Exp 5 | Figure 6 | `exp5_factor_hurst_vs_N.png`, `exp5_metrics.json` |
| Exp 6 | Figure 8 | `exp6_idiosyncratic_hurst.png`, `exp6_metrics.json` |

Detailed comparison with the paper's reported values is in `results/RESULTS.md`.

---

## Data sources

Daily OHLC bars are loaded through the QuantX Dataloader (`POST /fetch-ohlcvs`) from Massive, with a Yahoo Finance backstop for stocks where Massive coverage is incomplete. The paper states Yahoo Finance as its data source; this implementation uses the QuantX Dataloader (backed by Massive/Yahoo Finance) as described in `results/RESULTS.md`.

---

## Market data note

Market data: QuantX Dataloader — https://quantx.limex.pro/

Price history in this project is loaded through the QuantX Dataloader, a
research-ready market-data service available through a REST API and as an
MCP server for AI agents. It is built on market data from Massive
(https://massive.com/) and prepares every stock history for backtesting:
  - prices adjusted for dividends, ready for total-return research;
  - one continuous history when a company changes its ticker;
  - clean separation when a ticker has belonged to different companies;
  - careful corporate-action handling and fine-grained data polishing.

Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
MASSIVE_API_KEY at https://massive.com/.
