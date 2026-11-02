"""
Simulation of the one-factor N-SfFM model (Section 2.3, Eq. 8).

The model dynamics are:
    dx^i_t = beta_i * df_t + d_epsilon^i_t
    df_t   = exp(Omega_t / 2) * dW^0_t
    d_epsilon^i_t = sigma_i * exp(omega_tilde^i_t / 2) * dB^i_t
    omega_tilde^i_t = gamma_i * Omega_t + omega^i_t

where:
    Omega_t ~ S-fBM(H, lambda^2, T)      [factor log-volatility]
    omega^i_t ~ S-fBM(H_i, lambda_i^2, T) [idiosyncratic log-volatility]

Returns and quadratic variations are computed over coarse time steps with
s fine sub-steps per coarse period.

Custom implementation (paper Eq. 8 / Section 2.3 / Section 6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.sfbm import simulate_sfbm


@dataclass
class SimulationResult:
    """
    Container for one N-SfFM simulation replication.

    Attributes
    ----------
    qv_stocks : NDArray of shape (N, L)
        Per-asset quadratic variation proxy for each coarse period.
    qv_index : NDArray of shape (L,)
        Index (equal-weight) quadratic variation proxy.
    returns_stocks : NDArray of shape (N, L)
        Coarse returns for each asset.
    returns_index : NDArray of shape (L,)
        Coarse returns for the equal-weight index.
    Omega : NDArray of shape (L,)
        Factor log-volatility path.
    omega_tilde : NDArray of shape (N, L)
        Per-asset composite log-volatility paths.
    beta : NDArray of shape (N,)
        Factor exposure vector.
    """

    qv_stocks: NDArray
    qv_index: NDArray
    returns_stocks: NDArray
    returns_index: NDArray
    Omega: NDArray
    omega_tilde: NDArray
    beta: NDArray


def simulate_nsfbm(
    N: int,
    L: int,
    T: int,
    s: int,
    H: float,
    lambda2: float,
    H_i: float | NDArray,
    lambda2_i: float | NDArray,
    sigma_i: float | NDArray,
    gamma_i: float | NDArray,
    beta: NDArray | None = None,
    beta_a: float = 9.6,
    beta_b: float = 5.6,
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    """
    Simulate one Monte Carlo replication of the one-factor N-SfFM (Eq. 8).

    Generates:
    - Omega_t: factor log-volatility as S-fBM(H, lambda^2, T)
    - omega^i_t: idiosyncratic log-volatility as S-fBM(H_i, lambda_i^2, T)
    - omega_tilde^i_t = gamma_i * Omega_t + omega^i_t
    - Asset returns and quadratic variations with s fine sub-steps

    Parameters
    ----------
    N : int
        Number of synthetic assets.
    L : int
        Number of coarse time steps.
    T : int
        Decorrelation scale for S-fBM.
    s : int
        Number of fine sub-steps per coarse period.
    H : float
        Factor Hurst exponent.
    lambda2 : float
        Factor intermittency lambda^2.
    H_i : float or NDArray of shape (N,)
        Idiosyncratic Hurst exponents (scalar = same for all).
    lambda2_i : float or NDArray of shape (N,)
        Idiosyncratic intermittencies (scalar = same for all).
    sigma_i : float or NDArray of shape (N,)
        Idiosyncratic vol scale (scalar = same for all).
    gamma_i : float or NDArray of shape (N,)
        Vol-vol exposure gamma_i (scalar = same for all).
    beta : NDArray of shape (N,) or None
        Factor exposure vector. If None, draw from Beta(beta_a, beta_b).
    beta_a : float
        Beta distribution parameter a (Appendix H.2).
    beta_b : float
        Beta distribution parameter b (Appendix H.2).
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    SimulationResult
    """
    if rng is None:
        rng = np.random.default_rng()

    # Broadcast scalar parameters to arrays
    H_i_arr = np.broadcast_to(np.asarray(H_i, dtype=float), (N,)).copy()
    lam2_i_arr = np.broadcast_to(np.asarray(lambda2_i, dtype=float), (N,)).copy()
    sigma_i_arr = np.broadcast_to(np.asarray(sigma_i, dtype=float), (N,)).copy()
    gamma_i_arr = np.broadcast_to(np.asarray(gamma_i, dtype=float), (N,)).copy()

    # Draw beta if not provided (Beta distribution, Appendix H.2)
    if beta is None:
        beta = rng.beta(beta_a, beta_b, size=N)

    # --- Simulate factor log-volatility Omega_t (S-fBM with mean) ---
    Omega = simulate_sfbm(L, H, lambda2, T, n_samples=1, rng=rng, add_mean=True)[0]

    # --- Simulate idiosyncratic log-volatilities omega^i_t ---
    # For efficiency, batch assets with same parameters
    # If all H_i and lam2_i are the same, batch in one call
    if np.all(H_i_arr == H_i_arr[0]) and np.all(lam2_i_arr == lam2_i_arr[0]):
        omega_raw = simulate_sfbm(
            L, H_i_arr[0], lam2_i_arr[0], T,
            n_samples=N, rng=rng, add_mean=True
        )  # shape (N, L)
    else:
        omega_raw = np.zeros((N, L))
        for i in range(N):
            omega_raw[i] = simulate_sfbm(
                L, H_i_arr[i], lam2_i_arr[i], T,
                n_samples=1, rng=rng, add_mean=True
            )[0]

    # omega_tilde^i = gamma_i * Omega + omega^i
    # Both Omega and omega^i already have their correct means for E[exp(.)]=1
    # The sum: E[exp(gamma_i Omega + omega^i)] = 1 is satisfied automatically
    # because gamma_i * mean(Omega) + mean(omega^i) = -gamma_i*nu_Omega^2/4 - nu_i^2/4
    # and Var(gamma_i Omega + omega^i)/2 = (gamma_i^2 * nu_Omega^2/2 + nu_i^2/2)/2
    # So E[exp] = exp(mean + Var/2) = exp(-gamma_i^2*nu_Omega^2/4 - nu_i^2/4
    #                                      + gamma_i^2*nu_Omega^2/4 + nu_i^2/4) = 1 ✓
    omega_tilde = gamma_i_arr[:, None] * Omega[None, :] + omega_raw  # (N, L)

    # --- Simulate quadratic variations and returns ---
    # For each coarse step n, with s fine sub-steps:
    #   Df_n = exp(Omega_n/2) * Z0_n       [Z0_n ~ N(0,1)]
    #   De_n^i = sigma_i * exp(omega_tilde_n^i/2) * Zi_n  [Zi_n ~ N(0,1) indep]
    #   Dx_n^i = beta_i * Df_n + De_n^i
    #
    # Quadratic variation (chi-squared approximation valid for s=256):
    #   QV_n^i = beta_i^2 * exp(Omega_n) * chi2_factor_n/s +
    #            sigma_i^2 * exp(omega_tilde_n^i) * chi2_i_n/s
    # where chi2_factor_n ~ chi^2(s), chi2_i_n ~ chi^2(s)
    #
    # Factor chi-squared (shared among all assets via common W^0)
    chi2_factor = rng.chisquare(s, size=L) / s  # shape (L,)

    # Asset quadratic variations (N x L)
    # For each asset, compute factor + idiosyncratic contributions
    qv_stocks = np.zeros((N, L))
    for i in range(N):
        chi2_i = rng.chisquare(s, size=L) / s
        qv_stocks[i] = (
            beta[i] ** 2 * np.exp(Omega) * chi2_factor
            + sigma_i_arr[i] ** 2 * np.exp(omega_tilde[i]) * chi2_i
        )

    # Coarse returns (normal approximation: sum of s N(0,1/s) is N(0,1))
    Z_factor = rng.standard_normal(L)  # shared factor normal
    returns_stocks = np.zeros((N, L))
    for i in range(N):
        Z_i = rng.standard_normal(L)
        returns_stocks[i] = (
            beta[i] * np.exp(Omega / 2) * Z_factor
            + sigma_i_arr[i] * np.exp(omega_tilde[i] / 2) * Z_i
        )

    # Equal-weight index
    returns_index = np.mean(returns_stocks, axis=0)

    # Index quadratic variation: derived from the INDEX RETURN process (Eq. 8 / Section 6).
    # For ΔI = (1/N)*Σ_i ΔX_i, the QV of the index is:
    #   QV(I) = mean(β)² * exp(Ω) * χ²_factor/s
    #           + (1/N) * mean_i(σ_i² * exp(ω̃_i)) * χ²_eff/s
    # The idiosyncratic chi-squared noise scales as 1/N² per asset and 1/N in total,
    # correctly diversifying away for large N while leaving the factor contribution intact.
    # For small N the idiosyncratic term contributes, giving intermediate H estimates.
    bar_beta_sq: float = float(np.mean(beta) ** 2)
    sigma_eff_sq: NDArray = np.mean(
        sigma_i_arr[:, None] ** 2 * np.exp(omega_tilde), axis=0
    )  # shape (L,)
    chi2_eff_idx: NDArray = rng.chisquare(s, size=L) / s  # independent of chi2_factor
    qv_index = (
        bar_beta_sq * np.exp(Omega) * chi2_factor
        + (1.0 / N) * sigma_eff_sq * chi2_eff_idx
    )

    return SimulationResult(
        qv_stocks=qv_stocks,
        qv_index=qv_index,
        returns_stocks=returns_stocks,
        returns_index=returns_index,
        Omega=Omega,
        omega_tilde=omega_tilde,
        beta=beta,
    )


def log_qv(qv: NDArray, clip_min: float = 1e-12) -> NDArray:
    """
    Compute log of quadratic variation, clipping to avoid log(0).

    Parameters
    ----------
    qv : NDArray
        Quadratic variation values.
    clip_min : float
        Minimum value before log.

    Returns
    -------
    NDArray
        log(max(qv, clip_min)).
    """
    return np.log(np.maximum(qv, clip_min))
