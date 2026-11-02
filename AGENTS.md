# AGENTS.md — Guide for coding agents and contributors

## What this repository implements

Replication of the synthetic and empirical experiments from:

> **A Nested Factor Model for Equity Markets: Reconciling Multifractal Stock Returns and Rough Index Volatilities**  
> Othmane Zarhali, Cecilia Aubrun, Emmanuel Bacry, Jean-Philippe Bouchaud, Jean-François Muzy  
> arXiv: [2505.02678v5](https://arxiv.org/abs/2505.02678)

The paper introduces the Nested Log S-fBM Factor Model (N-SfFM), in which individual stock log-volatilities are driven by an idiosyncratic rough S-fBM process plus a smoother common factor S-fBM. This structure reconciles the empirical observation that stock index volatility is smoother (higher Hurst exponent) than single-stock volatility.

This repository reproduces:
- **Exp 1** — Figure 2: Synthetic validation that the N-SfFM generates different Hurst exponents for stocks and the index (Section 6).
- **Exp 2** — Figure 3: Synthetic validation that estimated index Hurst tracks the factor Hurst H across H ∈ {0.08, 0.12, 0.16} (Section 6).
- **Exp 3** — Figure 4: Convergence of estimated index and factor Hurst exponents with the number of assets N (Section 6).
- **Exp 4** — Figure 5: Calibration accuracy for idiosyncratic Hurst exponents H_i ∈ {0.03, 0.07} (Section 6).
- **Exp 5** — Figure 6: Empirical factor roughness from S&P 500 constituent panels vs number of stocks N_s (Section 7.1).
- **Exp 6** — Figure 8: Empirical idiosyncratic Hurst exponents for long-horizon S&P 500 stocks, two factor-volatility proxies (Section 7.2).

---

## Repository layout

```
src/
    config.py          All experiment parameters as module-level variables
    sfbm.py            S-fBM covariance kernel and Gaussian path sampler
    simulator.py       N-SfFM path simulation (one factor + N idiosyncratic processes)
    hurst_gmm.py       GMM Hurst estimation (Wu et al. [26] method)
    calibration.py     Full N-SfFM calibration pipeline (Sections 5 and 7)
    plotting.py        Shared plotting helpers (histograms, line plots)

exp/
    exp1.py – exp6.py  One script per experiment; run as python -m exp.expN

tests/
    conftest.py        Pytest fixtures (synthetic data, session-scoped market data)
    test_sfbm.py       Unit tests for src/ modules

data/
    data_loader.py     All market-data loading (QuantX Dataloader + Yahoo backstop)

results/
    RESULTS.md         Comparison with paper, deviations, not-reproduced items
    *.png, *.json, *.csv, *.log   Experiment artifacts (overwritten on each run)
```

`src/`, `exp/`, and `data/` each contain an empty `__init__.py`.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

Experiments 1–4 require no API keys. Experiments 5 and 6 require both keys.

---

## Running experiments

Run all commands from the repository root.

### Synthetic experiments (no API keys needed)

```bash
python -m exp.exp1    # Figure 2 — stock vs index Hurst distributions
python -m exp.exp2    # Figure 3 — index Hurst vs factor H
python -m exp.exp3    # Figure 4 — convergence with N (reduced N values, H=0.08,0.11)
python -m exp.exp4    # Figure 5 — idiosyncratic Hurst calibration
```

Full paper configuration for exp3 (all N values, 20 replications, H=0.08 and H=0.11):

```bash
python -m exp.exp3 --full
```

### Empirical experiments (DATALOADER_API_KEY and MASSIVE_API_KEY required)

```bash
python -m exp.exp5    # Figure 6 — factor roughness vs panel size N_s (2016–2020)
python -m exp.exp6    # Figure 8 — idiosyncratic Hurst for 243 S&P 500 stocks (1994–2026)
```

### All experiments via reproduce.sh

```bash
bash reproduce.sh
```

---

## Running tests

```bash
# Unit tests only (no API keys required)
python -m pytest -m "not market_data" -v

# Full suite including market-data integration tests (API keys required)
python -m pytest -v
```

---

## Conventions

1. **All experiment parameters live in `src/config.py`** as module-level `UPPER_CASE` variables with a comment stating their meaning, unit, and paper source. Do not add parameters anywhere else.

2. **All data loading goes through `data/data_loader.py`**. No experiment script or src module fetches data directly; no synthetic generators are placed in `src/` or `exp/`.

3. **API keys** are read from the environment via `python-dotenv`. Keep them in an uncommitted `.env`; never put them in source code, logs, or results files.

4. **Result artifacts** are written directly to `results/`. After a rerun, the experiment commands overwrite the files in `results/`. The numbers and verdicts in `results/RESULTS.md` must be checked against the new files and updated by hand; they are not updated automatically.

5. **Tests** are in `tests/`. Market-data tests are marked `@pytest.mark.market_data` and are skipped when `DATALOADER_API_KEY` is not set. Unit tests use synthetic fixtures only.

6. **No downloaded data is committed.** `data/cache/` (local Dataloader cache) is gitignored. Result files in `results/` (plots, metrics, logs) are committed; they contain derived statistics, not raw OHLCV data.

---

## Key src/ signatures

### `src/sfbm.py`
```python
sfbm_covariance(tau: np.ndarray, H: float, lam2: float, T: int) -> np.ndarray
sample_sfbm_path(L: int, H: float, lam2: float, T: int, rng: np.random.Generator) -> np.ndarray
```

### `src/simulator.py`
```python
simulate_nsfbm(
    N: int, L: int, H: float, lam2: float, T: int, s: int,
    H_i: float, lam2_i: float, gamma_i: float, sigma_i: float,
    beta: np.ndarray, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]  # (returns [L×N], log_var_stocks [L×N])
```

### `src/hurst_gmm.py`
```python
estimate_hurst_gmm(
    log_vol: np.ndarray, lags: np.ndarray | None = None,
) -> tuple[float, float]  # (H_est, lam2_est)
```

### `src/calibration.py`
```python
estimate_betas(returns: np.ndarray) -> np.ndarray          # Eq. 43
estimate_factor_qv(returns: np.ndarray, betas: np.ndarray) -> np.ndarray  # Eq. 44
gaussianize(x: np.ndarray) -> np.ndarray                  # Eq. 76
estimate_idiosyncratic_logvol(
    returns: np.ndarray, betas: np.ndarray,
    factor_logvol: np.ndarray,
) -> np.ndarray                                            # Eq. 46 + regression
run_full_calibration(
    returns: np.ndarray,
) -> dict                                                  # full pipeline Section 5
```

### `data/data_loader.py`
```python
load_sp500_data_exp5(start_date, end_date, tickers, index_ticker="SPY") -> tuple[dict, pd.DataFrame]
load_sp500_data_exp6(start_date, end_date, tickers) -> tuple[dict, pd.DataFrame]
garman_klass(ohlcv: pd.DataFrame) -> pd.Series            # Eq. 47
```
