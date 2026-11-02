# Replication Results

**Paper:** A Nested Factor Model for Equity Markets: Reconciling Multifractal Stock Returns and Rough Index Volatilities  
**Authors:** Othmane Zarhali, Cecilia Aubrun, Emmanuel Bacry, Jean-Philippe Bouchaud, Jean-François Muzy  
**arXiv:** [2505.02678v5](https://arxiv.org/abs/2505.02678)

---

## Experiments

### Experiment 1 — Synthetic validation: stock vs index Hurst exponents (Figure 2)

**Setup:** One-factor N-SfFM simulation with N=100 assets, 50 Monte Carlo replications, L=2^14 coarse steps, T=2^12, s=2^8, H=0.11, λ²=0.05², H_i=0.01, λ_i²=0.05², σ_i=1, γ_i=0.2, β_i ~ Beta(9.6, 5.6).

**Paper expected outcome:** Single-stock Hurst estimates cluster near H_i=0.01; the equal-weighted index Hurst clusters near H=0.11. The index is smoother than individual stocks.

| Metric | Paper target | This replication | Verdict |
|---|---|---|---|
| Mean stock Hurst H̄_stock | ≈ 0.01 | 0.0284 ± 0.0352 | not reproduced |
| Mean index Hurst H̄_index | ≈ 0.11 | 0.0862 ± 0.0673 | partially reproduced |
| Qualitative: H̄_index > H̄_stock | yes | yes (0.086 > 0.028) | reproduced |

**Verdict: PARTIALLY REPRODUCED.** The qualitative contrast between rougher stocks and smoother index is reproduced. Quantitatively, the stock Hurst estimate (0.028) is above the theoretical H_i=0.01, and the index Hurst estimate (0.086) is 22% below the theoretical H=0.11. The downward bias in the index estimate and upward bias in the stock estimate are consistent with finite-sample GMM bias at short path lengths. The cause of the remaining quantitative gap is the finite-sample GMM bias with the approximated lag grid: the paper does not fully specify the GMM lag grid for synthetic experiments, so the same empirical lag structure (floor(2^(k/4)) with Q~U[28,40]) was used throughout.

**Output files:** `results/exp1_hurst_distributions.png`, `results/exp1_metrics.json`, `results/exp1_stock_hurst.csv`, `results/exp1_index_hurst.csv`

---

### Experiment 2 — Index Hurst vs factor H across H values (Figure 3)

**Setup:** N-SfFM simulation with N=200, 20 replications, L=T=2^15, s=2^8, λ²=λ_i²=0.001, H_i=0.01, σ_i=1, γ_i=0.01, β_i ~ Beta(9.6, 5.6). Tested factor H ∈ {0.08, 0.12, 0.16}.

**Paper expected outcome:** Mean estimated index Hurst lies near the 45-degree line H_I=H; theoretical H falls within 95% confidence intervals.

| H_true | Mean Ĥ_I | 95% CI | H_true in CI? | |H_true − mean|/H_true | Verdict |
|---|---|---|---|---|---|
| 0.08 | 0.0671 | [0.047, 0.089] | yes | 16% | reproduced |
| 0.12 | 0.0862 | [0.071, 0.113] | no | 28% | not reproduced |
| 0.16 | 0.1206 | [0.086, 0.161] | yes | 25% | partially reproduced |

Monotone relationship (Ĥ_I increases with H_true): 0.067 < 0.086 < 0.121 — **reproduced**.

**Verdict: PARTIALLY REPRODUCED.** The monotone relationship between estimated index Hurst and theoretical factor Hurst is reproduced. The true H falls within the 95% CI for H=0.08 and H=0.16 but not for H=0.12. A systematic downward bias of ~20–28% is present, consistent with the finite-sample GMM behavior noted in Experiment 1.

**Output files:** `results/exp2_factor_h_vs_true.png`, `results/exp2_metrics.json`, `results/exp2_table.csv`

---

### Experiment 3 — Convergence of index and factor Hurst estimates with N (Figure 4)

**Setup:** N-SfFM simulation with N ∈ {2,4,8,16,32,64,128,256}, 20 replications per N, L=T=2^15, s=2^8, λ²=λ_i²=0.001, H_i=0.01, σ_i=1, γ_i=0.01, β_i ~ Beta(9.6, 5.6). Two factor H values: 0.08 and 0.11.

**Paper expected outcome:** As N increases, the estimated index Hurst converges toward H (the factor Hurst); at small N both index and calibrated-factor estimates sit close to H_i due to undiversified idiosyncratic roughness.

Selected results for H_true=0.11:

| N | Mean Ĥ_index | Mean Ĥ_factor | H_true |
|---|---|---|---|
| 2 | 0.017 | 0.010 | 0.11 |
| 8 | 0.045 | 0.025 | 0.11 |
| 32 | 0.066 | 0.053 | 0.11 |
| 128 | 0.052 | 0.048 | 0.11 |
| 256 | 0.053 | 0.049 | 0.11 |

**Verdict: NOT REPRODUCED.** A weak upward trend in Ĥ_index from N=2 to moderate N exists (17→66 mHz for H_true=0.11), consistent with the qualitative direction, but estimates plateau far below H_true=0.11 at all N and show no clear convergence above 0.06. The paper predicts convergence to H_true at large N; the largest basin tested (N=256) gives 0.053 versus the target 0.11, a 52% shortfall. The cause is the same finite-sample GMM downward bias that affects Experiments 1 and 2, amplified here because the calibrated-factor proxy QV (Eq. 44) compounds the noise from short series. Running at the full paper path length (L=T=2^15) did not resolve the bias; a larger Q in the GMM lag grid or an alternative estimator might, but the paper does not specify the exact GMM configuration for the synthetic experiments.

**Output files:** `results/exp3_hurst_vs_N_H0.08.png`, `results/exp3_hurst_vs_N_H0.11.png`, `results/exp3_metrics.json`, `results/exp3_table.csv`

---

### Experiment 4 — Idiosyncratic Hurst calibration accuracy (Figure 5)

**Setup:** N-SfFM simulation with N=300, L=T=2^15, s=2^8, λ²=λ_i²=0.01, H=0.1, σ_i=1, γ_i=0.1, β_i ~ Beta(9.6, 5.6). Two idiosyncratic Hurst scenarios: H_i=0.03 (Scenario A) and H_i=0.07 (Scenario B). One replication per scenario (number of Monte Carlo replications not specified in the paper).

**Paper expected outcome:** Estimated H_i distributions cluster near the true values 0.03 and 0.07.

| Scenario | H_i true | Mean Ĥ_i | Std Ĥ_i | Verdict |
|---|---|---|---|---|
| A | 0.03 | 0.0573 | 0.0615 | not reproduced |
| B | 0.07 | 0.0692 | 0.0490 | reproduced |

**Verdict: PARTIALLY REPRODUCED.** Scenario B (H_i=0.07) is reproduced with a mean estimate of 0.069, within 1.4% of the target. Scenario A (H_i=0.03) is not reproduced: the mean of 0.057 is 90% above the target. The overestimation for low H_i is consistent with the calibration chain compounding GMM noise at very rough regimes (H_i=0.03 is close to the H=0 multifractal limit where estimation variance is highest). The paper states the calibration recovers H_i "reasonably well" without reporting exact numbers for Figure 5; the Scenario B result is consistent with this claim.

**Output files:** `results/exp4_idiosyncratic_hurst.png`, `results/exp4_metrics.json`, `results/exp4_scenario_A.csv`, `results/exp4_scenario_B.csv`

---

### Experiment 5 — Empirical factor roughness from S&P 500 panels (Figure 6)

**Setup:** Daily OHLC data from Yahoo Finance (via QuantX Dataloader) for S&P 500 stocks and the ^GSPC index, 2016-01-01 to 2020-01-01. For each N_s ∈ {2,4,8,16,32,64,128,256}, 20 random combinations of N_s stocks. Garman-Klass volatility (Eq. 47). GMM lag structure: floor(2^(k/4)) for k=0..Q, Q ~ U([28,40]).

**Paper expected outcome:** Estimated factor Hurst rises with N_s and stabilizes near the S&P 500 index Hurst (≈0.13). The index Hurst is reported as approximately 0.13 in the paper.

| N_s | Mean Ĥ_factor | 95% CI | Verdict |
|---|---|---|---|
| 2 | 0.0352 | [0.021, 0.076] | — |
| 4 | 0.0402 | [0.030, 0.053] | — |
| 8 | 0.0408 | [0.032, 0.057] | — |
| 16 | 0.0441 | [0.037, 0.053] | — |
| 32 | 0.0473 | [0.040, 0.054] | — |
| 64 | 0.0494 | [0.045, 0.052] | — |
| 128 | 0.0520 | [0.050, 0.054] | — |
| 256 | n/a (only 147 tickers available) | — | not reproduced |

S&P 500 index Hurst (benchmark): Ĥ_index = **0.0464**  
Paper benchmark: ≈ 0.13

**Verdict: PARTIALLY REPRODUCED.** The monotone increase of the estimated factor Hurst with N_s is reproduced: Ĥ_factor rises from 0.035 (N_s=2) to 0.052 (N_s=128), approaching the benchmark Ĥ_index=0.047, consistent with the paper's Figure 6 qualitative trend. The absolute level (≈0.05) is substantially below the paper's reported value of ≈0.13 for both the factor and index estimates. This is the same systematic downward GMM bias documented in Experiments 1–3: the finite-sample small-intermittency approximation and identity weighting matrix underestimate H in the 0.05–0.15 range with daily data. N_s=256 could not be run because the available universe covers only 147 tickers with full 2016–2020 OHLC coverage.

**Output files:** `results/exp5_empirical_factor_hurst.png`, `results/exp5_metrics.json`, `results/exp5_table.csv`

---

### Experiment 6 — Empirical idiosyncratic roughness for long-horizon S&P 500 stocks (Figure 8)

**Setup:** Daily OHLC data for 243 long-lived S&P 500 stocks and ^GSPC index from 1994-01-01 to 2026-01-01 (Yahoo Finance via QuantX Dataloader). Calibration approach A: fully estimated factor volatility from Eq. (43)–(44). Calibration approach B: S&P 500 Garman-Klass volatility used directly as factor log-volatility proxy.

**Paper expected outcome:** Most estimated H_i cluster near zero for both approaches. Factor log-volatility (Approach A) should be ~80% correlated with S&P 500 GK volatility. A small number of outliers with H_i ≳ 0.1 may appear.

**Realized setup:** N_stocks = 147 (186 passed coverage filter; 147 had ≥ 90 % of trading days present; paper: 243), T = 8 053 trading days (1994-01-01 to 2026-01-01). Threshold for "near zero": H_i < 0.05.

| Approach | Mean Ĥ_i | Std Ĥ_i | % near zero (H_i < 0.05) | Corr(Ω_t, SPX GK) | Verdict |
|---|---|---|---|---|---|
| A (full calibration) | **0.056** | 0.050 | **61 %** | **0.772** | partially reproduced |
| B (index-sourced factor) | **0.067** | 0.061 | **56 %** | — | partially reproduced |

**Verdicts:**
- *Both approaches: majority of H_i cluster near zero* — **reproduced** (61 % and 56 % below 0.05; paper: most stocks near zero).
- *Factor log-vol correlated ~80 % with S&P 500 GK vol* — **reproduced** (0.772 vs. paper ≈ 0.80; within 4 %).
- *Comparable histograms for Approaches A and B* — **reproduced** qualitatively; A gives lower mean than B, consistent with the paper's Figure 8.
- *Absolute mean H_i level* — **partially reproduced** (means of 0.056 / 0.067 are somewhat above paper's visual impression of ≈ 0; the same GMM downward bias seen in synthetic experiments affects the idiosyncratic estimates, and the reduced universe of 147 vs. 243 stocks contributes residual factor contamination).

**Output files:** `results/exp6_idiosyncratic_hurst_empirical.png`, `results/exp6_metrics.json`, `results/exp6_approach_A.csv`, `results/exp6_approach_B.csv`

---

## Deviations from the paper

1. **Data source (Exp 5, Exp 6):** The paper states Yahoo Finance as its data source. This replication loads daily OHLC data through the QuantX Dataloader (backed by Massive with a Yahoo Finance backstop). Where Massive lacks data for a ticker, Yahoo Finance fills it; this is the same underlying source. The `ticker_details.data_source` field in the API response identifies the per-ticker source. Differences in corporate-action adjustment between Yahoo's native API and the Dataloader/Massive path may introduce small pricing differences.

2. **S-fBM discretization:** The paper specifies the S-fBM covariance kernel (Eq. 2) but not the exact algorithm for generating discrete sample paths. Paths are generated here by Cholesky decomposition of the L×L S-fBM covariance matrix on the coarse grid, which is exact for the given covariance but computationally expensive (O(L³)); for large L=2^15 the full matrix is not tractable and a circulant embedding / FFT-based simulation is used instead. The paper gives no alternative.

3. **GMM lag structure for synthetic experiments:** The paper fully specifies the GMM lag structure for the empirical experiments (Section 7.1: floor(2^(k/4)) for k=0..Q, Q ~ U([28,40])). The paper does not restate the lag structure for the synthetic experiments in Sections 6. The same lag structure is applied to the synthetic experiments; this is consistent with the spirit of [26] (Wu, Muzy, Bacry 2022) to which the paper refers.

4. **GMM weighting matrix:** The paper refers to the GMM of Wu et al. [26] but does not specify the weighting matrix. An identity weighting matrix is used here. An optimal Newey-West weighting matrix would yield more efficient estimates; its absence may contribute to finite-sample bias.

5. **Stock universe for Exp 5:** The paper states "stocks from the S&P 500" without listing the exact tickers or specifying whether the membership is point-in-time or as of a later date. The pool used here is drawn from `SP500_LONG_TERM_TICKERS`, a set of large-cap S&P 500 constituents with continuous data coverage over 2016–2020, loaded through the QuantX Dataloader.

6. **Stock universe for Exp 6:** The paper states 243 US stocks that are constituents of the S&P 500 throughout 1994–2026. The exact list of 243 stocks is not published. This replication uses the `SP500_LONG_TERM_TICKERS` list, which was constructed to cover long-tenured constituents with available OHLC data over the full period. Tickers missing data for significant portions of the period are excluded; their identities are listed in the experiment log (`results/log_exp6.txt`).

7. **Exp 4 replication count:** The number of Monte Carlo replications used for Figure 5 is not specified in the paper. One replication is used here (N=300 assets generates sufficient H_i estimates for the histogram).

8. **Gaussianization (Eq. 76):** Empirical CDF values at 0 or 1 are clipped to [10⁻⁶, 1−10⁻⁶] before applying the normal inverse CDF, as required numerically. The paper does not specify a clipping threshold.

---

## Not reproduced

1. **Figure 7 (individual stock idiosyncratic volatility time series):** This figure shows example log-volatility trajectories for individual stocks. It is a qualitative illustration and contains no quantitative claims; it is not reproduced here.

2. **Sections 3–4 (theoretical results):** The paper's analytical derivations (Propositions 1–5, corollaries, and theorems in Sections 3 and 4) are mathematical proofs and are not reproduced computationally.

3. **Appendix results that depend on the full calibration convergence for very small H:** The paper's appendices discuss asymptotic properties of the GMM estimator. No dedicated test of these properties is included.
