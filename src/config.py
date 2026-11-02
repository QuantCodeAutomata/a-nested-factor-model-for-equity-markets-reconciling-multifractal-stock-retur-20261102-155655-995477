"""
Configuration for all experiments in the N-SfFM replication.
All experiment parameters are defined here as module-level variables.
"""

# ---------------------------------------------------------------------------
# S-fBM covariance kernel (Section 2.1, Eq. 2 and 5)
# ---------------------------------------------------------------------------
# nu^2 = lambda^2 / (H*(1-2H)); covariance C(tau) = nu^2/2 * (1-(tau/T)^(2H)) for |tau|<T
# Mean for E[exp(omega)]=1: mu = -nu^2/4  (Eq. 50)
# Intermittency: lambda^2 = H*(1-2H)*nu^2 (Eq. 5)

# ---------------------------------------------------------------------------
# Beta-distribution parameters for exposures beta_i (Appendix H.2)
# ---------------------------------------------------------------------------
BETA_DIST_A = 9.6   # shape parameter a for Beta(a, b)
BETA_DIST_B = 5.6   # shape parameter b for Beta(a, b)

# ---------------------------------------------------------------------------
# Exp 1 — Synthetic validation: stock vs index Hurst exponents (Figure 2)
# Section 6; parameters from Figure 2 caption (page 14-15)
# ---------------------------------------------------------------------------
EXP1_N = 100           # number of synthetic assets
EXP1_MC_REPS = 50      # Monte Carlo replications
EXP1_L = 2**14         # total coarse time steps (length of simulated paths)
EXP1_T = 2**12         # decorrelation scale of S-fBM
EXP1_S = 2**8          # fine-grid sub-steps per coarse period
EXP1_H = 0.11          # factor Hurst exponent
EXP1_LAMBDA2 = 0.05**2  # factor intermittency lambda^2 (lambda=0.05)
EXP1_H_I = 0.01        # idiosyncratic Hurst exponent (same for all i)
EXP1_LAMBDA2_I = 0.05**2  # idiosyncratic intermittency (lambda_i=0.05)
EXP1_SIGMA_I = 1.0     # idiosyncratic vol scale sigma_i
EXP1_GAMMA_I = 0.2     # vol-vol exposure gamma_i (same for all i)

# ---------------------------------------------------------------------------
# Exp 2 — Factor roughness vs index roughness (Figure 3)
# Section 6; parameters from Figure 3 caption (page 15)
# ---------------------------------------------------------------------------
EXP2_N = 200           # number of synthetic assets
EXP2_MC_REPS = 20      # Monte Carlo replications
EXP2_H_VALUES = [0.08, 0.12, 0.16]  # tested factor Hurst values
EXP2_L = 2**15         # total coarse time steps (T=L=2^15)
EXP2_T = 2**15         # decorrelation scale
EXP2_S = 2**8          # fine-grid sub-steps per coarse period
EXP2_LAMBDA2 = 0.001   # factor intermittency lambda^2
EXP2_LAMBDA2_I = 0.001  # idiosyncratic intermittency
EXP2_H_I = 0.01        # idiosyncratic Hurst exponent
EXP2_SIGMA_I = 1.0     # idiosyncratic vol scale
EXP2_GAMMA_I = 0.01    # vol-vol exposure

# ---------------------------------------------------------------------------
# Exp 3 — Convergence with number of assets (Figure 4)
# Section 6; parameters from Figure 4 caption / surrounding text (page 16-17)
# ---------------------------------------------------------------------------
EXP3_N_VALUES = [2, 4, 8, 16, 32, 64, 128, 256]  # basket sizes
EXP3_MC_REPS = 20      # Monte Carlo replications per N
EXP3_H_VALUES = [0.08, 0.11]  # two tested factor Hurst values
EXP3_L = 2**15         # total coarse time steps
EXP3_T = 2**15         # decorrelation scale
EXP3_S = 2**8          # fine-grid sub-steps
EXP3_LAMBDA2 = 0.001   # factor intermittency
EXP3_LAMBDA2_I = 0.001  # idiosyncratic intermittency
EXP3_H_I = 0.01        # idiosyncratic Hurst
EXP3_SIGMA_I = 1.0     # idiosyncratic vol scale
EXP3_GAMMA_I = 0.01    # vol-vol exposure

# ---------------------------------------------------------------------------
# Exp 4 — Idiosyncratic Hurst calibration accuracy (Figure 5)
# Section 6; parameters from Figure 5 caption (page 17)
# ---------------------------------------------------------------------------
EXP4_N = 300           # number of synthetic assets
EXP4_H = 0.1           # factor Hurst exponent
EXP4_LAMBDA2 = 0.01    # factor intermittency
EXP4_LAMBDA2_I = 0.01  # idiosyncratic intermittency
EXP4_H_I_VALUES = [0.03, 0.07]  # two idiosyncratic Hurst scenarios
EXP4_SIGMA_I = 1.0     # idiosyncratic vol scale
EXP4_GAMMA_I = 0.1     # vol-vol exposure
EXP4_L = 2**15         # total coarse time steps
EXP4_T = 2**15         # decorrelation scale
EXP4_S = 2**8          # fine-grid sub-steps
# Number of Monte Carlo replications not specified in paper; assumed 1 per scenario
EXP4_MC_REPS = 1       # not specified in the paper

# ---------------------------------------------------------------------------
# Exp 5 — Empirical factor roughness from S&P 500 (Figure 6)
# Section 7.1; parameters from pp. 17-18
# ---------------------------------------------------------------------------
EXP5_START_DATE = "2016-01-01"  # sample start (Section 7.1, page 18)
EXP5_END_DATE = "2020-01-01"    # sample end (Section 7.1, page 18)
EXP5_N_VALUES = [2, 4, 8, 16, 32, 64, 128, 256]  # basket sizes
EXP5_N_COMBOS = 20     # random combinations per N_s (Section 7.1, page 18)
# GMM lag sequence: floor(2^(k/4)) for k=0..Q, Q ~ U([28,40]) (Section 7.1, page 18)
EXP5_GMM_Q_LOW = 28    # lower bound for Q
EXP5_GMM_Q_HIGH = 40   # upper bound for Q
EXP5_INDEX_TICKER = "SPY"  # SPY ETF as S&P 500 proxy; ^GSPC is rejected by the Massive backend
# Minimum S&P 500 tickers pool — paper says "stocks from S&P500" without fixed list
EXP5_TICKER_POOL_SIZE = 400   # pool size before subsampling; not specified in paper

# ---------------------------------------------------------------------------
# Exp 6 — Empirical idiosyncratic roughness for long-horizon S&P 500 (Figure 8)
# Section 7.2; parameters from pp. 19-21
# ---------------------------------------------------------------------------
EXP6_START_DATE = "1994-01-01"  # sample start (Section 7.2, page 19)
EXP6_END_DATE = "2026-01-01"    # sample end (Section 7.2, page 19)
EXP6_N_STOCKS = 243    # survivorship-stable S&P 500 stocks (Section 7.2, page 19)
EXP6_INDEX_TICKER = "SPY"  # SPY ETF as S&P 500 proxy; ^GSPC is rejected by the Massive backend

# ---------------------------------------------------------------------------
# GMM configuration (Appendix C.2; Section 7.1 for lag structure)
# ---------------------------------------------------------------------------
# Lag structure: floor(2^(k/4)) for k=0..Q, Q ~ U([GMM_Q_LOW, GMM_Q_HIGH])
# Paper specifies this structure for empirical experiments; same applied to synthetic
GMM_Q_LOW = 28         # lower bound for Q (Section 7.1, page 18)
GMM_Q_HIGH = 40        # upper bound for Q (Section 7.1, page 18)
GMM_H_BOUNDS = (1e-4, 0.49)   # bounds for H in optimization
GMM_LAMBDA2_BOUNDS = (1e-8, 2.0)  # bounds for lambda^2 in optimization

# ---------------------------------------------------------------------------
# Random seed (not specified in paper; fixed for reproducibility)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42       # not specified in the paper
