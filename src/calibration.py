"""
Calibration pipeline for the N-SfFM (Section 5, Eqs. 43-46).

Steps:
    1. Estimate beta from off-diagonal Frobenius minimization (Eq. 43)
    2. Compute factor quadratic variation proxy (Eq. 44)
    3. Gaussianize log-factor-QV → Omega_t (Appendix H.1 / Eq. 76)
    4. GMM estimation of factor H and lambda^2 (Appendix C.2)
    5. Compute residual QV (Eq. 46)
    6. Gaussianize log-residual-QV → omega_tilde^i_t
    7. Regress omega_tilde^i on Omega_t → omega^i_t (Section 5 Step 5)
    8. GMM estimation of H_i and lambda_i^2 for each asset (Section 5 Step 5)

Custom implementation (paper Section 5, Eqs. 43-46, 76 / Appendix H.1 C.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from src.hurst_gmm import (
    estimate_hurst_gmm,
    gaussianize,
)


# ---------------------------------------------------------------------------
# Step 1: Beta estimation
# ---------------------------------------------------------------------------


def _beta_objective_and_grad(
    beta: NDArray, C_off: NDArray
) -> tuple[float, NDArray]:
    """
    Objective and gradient for off-diagonal Frobenius norm (Eq. 43):
        f(beta) = ||C_off - beta beta^T||_{F,off-diag}^2
        grad_k  = -4 [C_off beta]_k + 4 beta_k (||beta||^2 - beta_k^2)

    Parameters
    ----------
    beta : NDArray of shape (N,)
    C_off : NDArray of shape (N, N) with diagonal zeroed

    Returns
    -------
    (float, NDArray)
        Objective value and gradient.
    """
    Cb = C_off @ beta          # (N,)
    beta2_sum = np.dot(beta, beta)
    obj = np.sum((C_off - np.outer(beta, beta)) ** 2)
    grad = -4 * Cb + 4 * beta * (beta2_sum - beta ** 2)
    return float(obj), grad


def estimate_beta(
    returns: NDArray,
    max_iter: int = 2000,
) -> NDArray:
    """
    Estimate the exposure vector beta by minimizing the off-diagonal Frobenius
    norm between the sample return covariance and the rank-1 matrix beta beta^T
    (Eq. 43).

        beta_hat = argmin ||E(X X^T) - beta beta^T||_{F, off-diag}

    Uses the leading eigenvector of the off-diagonal covariance as initializer,
    then refines with L-BFGS-B.

    Parameters
    ----------
    returns : NDArray of shape (N, L)
        Return time series for N assets over L time steps.
    max_iter : int
        Maximum L-BFGS-B iterations.

    Returns
    -------
    NDArray of shape (N,)
        Estimated beta vector.
    """
    N, L = returns.shape

    # Sample covariance
    R = returns - returns.mean(axis=1, keepdims=True)
    C = R @ R.T / L  # (N, N)

    # Off-diagonal covariance (zero diagonal)
    C_off = C.copy()
    np.fill_diagonal(C_off, 0.0)

    # Initial guess: leading eigenvector of C_off scaled by sqrt(eigenvalue)
    eigvals, eigvecs = np.linalg.eigh(C_off)
    idx = np.argmax(eigvals)
    lam1 = max(eigvals[idx], 1e-8)
    beta0 = np.sqrt(lam1) * eigvecs[:, idx]

    # Ensure beta is positive (sign convention: most entries positive)
    if np.mean(beta0) < 0:
        beta0 = -beta0

    result = minimize(
        lambda b: _beta_objective_and_grad(b, C_off),
        beta0,
        method="L-BFGS-B",
        jac=True,
        bounds=[(1e-6, None)] * N,  # beta_i > 0
        options={"maxiter": max_iter, "ftol": 1e-14, "gtol": 1e-8},
    )

    beta_hat = np.maximum(result.x, 1e-6)
    return beta_hat


# ---------------------------------------------------------------------------
# Step 2: Factor quadratic variation proxy
# ---------------------------------------------------------------------------


def factor_qv_proxy(
    qv_assets: NDArray,
    beta: NDArray,
) -> NDArray:
    """
    Compute the factor quadratic variation proxy using Eq. 44:

        <f_hat>_t = (sum_i beta_i^2 * <x^i>_t) / (sum_i beta_i^4)

    Parameters
    ----------
    qv_assets : NDArray of shape (N, L)
        Per-asset quadratic variation proxy over L time periods.
    beta : NDArray of shape (N,)
        Factor exposure vector.

    Returns
    -------
    NDArray of shape (L,)
        Factor quadratic variation proxy.
    """
    beta2 = beta ** 2
    beta4 = beta ** 4
    return (beta2[:, None] * qv_assets).sum(axis=0) / beta4.sum()


# ---------------------------------------------------------------------------
# Step 4 (residuals): Residual QV
# ---------------------------------------------------------------------------


def residual_qv(
    qv_assets: NDArray,
    factor_qv: NDArray,
    beta: NDArray,
) -> NDArray:
    """
    Compute per-asset residual quadratic variation from Eq. 46:

        <epsilon^i>_t = <x^i>_t - beta_i^2 * <f_hat>_t

    Clips at a small positive value to avoid negative residuals from noise.

    Parameters
    ----------
    qv_assets : NDArray of shape (N, L)
        Per-asset quadratic variation.
    factor_qv : NDArray of shape (L,)
        Factor quadratic variation proxy.
    beta : NDArray of shape (N,)
        Factor exposures.

    Returns
    -------
    NDArray of shape (N, L)
        Residual quadratic variation per asset.
    """
    res = qv_assets - (beta[:, None] ** 2) * factor_qv[None, :]
    return np.maximum(res, 1e-12)


# ---------------------------------------------------------------------------
# Full calibration pipeline
# ---------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    """
    Container for the full N-SfFM calibration result.

    Attributes
    ----------
    beta : NDArray (N,)
        Estimated factor exposures.
    factor_qv : NDArray (L,)
        Factor QV proxy.
    Omega : NDArray (L,)
        Gaussianized log-factor-QV (factor log-volatility estimate).
    H_factor : float
        Estimated factor Hurst exponent.
    lambda2_factor : float
        Estimated factor intermittency.
    H_i : NDArray (N,)
        Per-asset estimated idiosyncratic Hurst exponents.
    lambda2_i : NDArray (N,)
        Per-asset estimated idiosyncratic intermittencies.
    gamma_i : NDArray (N,)
        Per-asset vol-vol regression coefficients.
    omega_i : NDArray (N, L)
        Per-asset idiosyncratic log-volatility time series.
    """

    beta: NDArray
    factor_qv: NDArray
    Omega: NDArray
    H_factor: float
    lambda2_factor: float
    H_i: NDArray
    lambda2_i: NDArray
    gamma_i: NDArray
    omega_i: NDArray = field(default_factory=lambda: np.array([]))


def run_calibration(
    returns: NDArray,
    qv_assets: NDArray,
    T: float,
    Omega_external: NDArray | None = None,
    Q_range: tuple[int, int] = (28, 40),
    rng: np.random.Generator | None = None,
) -> CalibrationResult:
    """
    Run the full N-SfFM calibration pipeline (Section 5, Steps 1-5).

    Parameters
    ----------
    returns : NDArray of shape (N, L)
        Asset return time series.
    qv_assets : NDArray of shape (N, L)
        Per-asset quadratic variation proxies (e.g., GK estimator or fine-grid QV).
    T : float
        Decorrelation scale for the S-fBM processes.
    Omega_external : NDArray of shape (L,) or None
        If provided, use this directly as the factor log-volatility (Approach B in
        Section 7.2: sourcing S&P 500 GK log-vol as factor).
    Q_range : tuple
        Uniform range [low, high] for random GMM lag Q.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    CalibrationResult
    """
    if rng is None:
        rng = np.random.default_rng()

    N, L = returns.shape

    # Step 1: Estimate beta (Eq. 43)
    beta = estimate_beta(returns)

    # Step 2: Factor QV proxy (Eq. 44)
    fqv = factor_qv_proxy(qv_assets, beta)

    if Omega_external is not None:
        # Approach B: use external factor log-volatility (Section 7.2)
        Omega = Omega_external - np.mean(Omega_external)
        H_factor = float("nan")
        lambda2_factor = float("nan")
    else:
        # Step 3: Gaussianize log-factor-QV (Appendix H.1, Eq. 76)
        log_fqv = np.log(np.maximum(fqv, 1e-12))
        log_fqv_centered = log_fqv - np.mean(log_fqv)
        Omega = gaussianize(log_fqv_centered)

        # Step 4: GMM on Omega_t to estimate H_factor, lambda2_factor (Appendix C.2)
        H_factor, lambda2_factor = estimate_hurst_gmm(
            Omega, T, Q=None, Q_range=Q_range, rng=rng, n_random_Q=20
        )

    # Step 5: Residual QV (Eq. 46)
    res_qv = residual_qv(qv_assets, fqv, beta)  # (N, L)

    # Step 5 continued: Gaussianize residual log-QV → omega_tilde^i
    H_i_arr = np.empty(N)
    lambda2_i_arr = np.empty(N)
    gamma_i_arr = np.empty(N)
    omega_i_arr = np.empty((N, L))

    for i in range(N):
        log_res_qv_i = np.log(np.maximum(res_qv[i], 1e-12))
        log_res_centered = log_res_qv_i - np.mean(log_res_qv_i)
        omega_tilde_i = gaussianize(log_res_centered)  # ~ gamma_i Omega + omega^i

        # Step 5: Regress omega_tilde^i on Omega to extract omega^i (Section 5)
        # omega_tilde^i = gamma_i * Omega + omega^i
        # OLS: gamma_i = Cov(omega_tilde^i, Omega) / Var(Omega)
        var_Omega = np.var(Omega)
        if var_Omega > 1e-12:
            gamma_i = float(np.cov(omega_tilde_i, Omega)[0, 1] / var_Omega)
        else:
            gamma_i = 0.0
        omega_i = omega_tilde_i - gamma_i * Omega

        gamma_i_arr[i] = gamma_i
        omega_i_arr[i] = omega_i

        # Step 5: GMM on omega^i to estimate H_i, lambda_i^2
        H_i, lam2_i = estimate_hurst_gmm(
            omega_i, T, Q=None, Q_range=Q_range, rng=rng, n_random_Q=1
        )
        H_i_arr[i] = H_i if not np.isnan(H_i) else 0.0
        lambda2_i_arr[i] = lam2_i if not np.isnan(lam2_i) else 0.01

    return CalibrationResult(
        beta=beta,
        factor_qv=fqv,
        Omega=Omega,
        H_factor=H_factor,
        lambda2_factor=lambda2_factor,
        H_i=H_i_arr,
        lambda2_i=lambda2_i_arr,
        gamma_i=gamma_i_arr,
        omega_i=omega_i_arr,
    )
