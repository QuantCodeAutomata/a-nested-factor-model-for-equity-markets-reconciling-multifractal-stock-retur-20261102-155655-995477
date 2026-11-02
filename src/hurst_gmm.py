"""
GMM estimation of Hurst exponent for S-fBM / Log S-fBM processes.

Follows the GMM framework of Wu et al. [26] (Appendix C.2, Eq. 60) as adapted
for the Gaussianized log-QV series.  The paper Gaussianizes the log-quadratic
variation before applying the GMM (Section 5, Step 3), treating the result as
an estimate of the normalized log-volatility Omega_t.  After Gaussianization the
series has unit variance, so its autocovariance equals its autocorrelation.
The GMM then matches the empirical autocorrelation to the theoretical shape

    C_theory(k; H, rho) = rho * (1 - (k/T)^{2H}),      0 < k < T       (*)

where rho in (0, 1] captures the signal-to-total-variance ratio (signal = S-fBM,
total = signal + measurement noise such as chi-squared sampling error).  For a
pure S-fBM series Gaussianized to unit variance, rho = 1.  For log-QV with s
chi-squared sub-steps, rho = nu^2/2 / (nu^2/2 + 2/s).

Moment conditions (Appendix C.2):
    K_N(theta) = C_hat_gauss(k) - rho * (1 - (k/T)^{2H})

where C_hat_gauss is the empirical autocovariance of the Gaussianized series.

Lambda^2 is estimated from the variance of the pre-Gaussianization log-QV using
    lambda^2 = 2 * rho * (var_raw - 2/s) * H * (1-2H) / rho ... see estimate_lambda2.

Custom implementation consistent with the small-intermittency approximation of
Section 4 / Appendix C.2 and the Gaussianization procedure of Appendix H.1.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


def sfbm_autocorr_theory(lags: NDArray, H: float, T: float) -> NDArray:
    """
    Theoretical autocorrelation of the S-fBM process at positive integer lags.

    For a Gaussianized log-QV series the autocovariance equals the autocorrelation
    and is proportional to the S-fBM correlation:

        C(k; H) = 1 - (k/T)^{2H}   for  1 <= k < T

    This is derived from C_sfbm(k) = nu^2/2 * (1-(k/T)^{2H}) divided by C_sfbm(0)
    = nu^2/2, giving a lambda^2-free shape.

    Parameters
    ----------
    lags : NDArray
        Positive integer lag values (k >= 1, k < T).
    H : float
        Hurst exponent in (0, 0.5).
    T : float
        Decorrelation scale (same units as lags).

    Returns
    -------
    NDArray
        Theoretical autocorrelation values in [0, 1].
    """
    k = np.asarray(lags, dtype=float)
    return np.where(k < T, 1.0 - (k / T) ** (2.0 * H), 0.0)


def empirical_autocovariance(
    series: NDArray, lags: NDArray, demean: bool = True
) -> NDArray:
    """
    Compute empirical autocovariance at given lags.

    Parameters
    ----------
    series : NDArray of shape (L,)
        Time series (zero-mean if demean=False, otherwise demeaned internally).
    lags : NDArray of int
        Lag values (k > 0).
    demean : bool
        If True (default), subtract the sample mean first.  Set to False for
        Gaussianized series, which have zero population mean by construction;
        subtracting the noisy sample mean from a long-memory series introduces
        a large finite-sample bias of order Var(sample_mean) ≈ 2*rho*2H/(1+2H)
        that deflates every lag-k estimate by the same constant and causes the
        GMM to underestimate H.

    Returns
    -------
    NDArray of shape (len(lags),)
        Empirical autocovariance at each lag.
    """
    L = len(series)
    y = series - np.mean(series) if demean else series
    result = np.empty(len(lags))
    for i, k in enumerate(lags):
        k = int(k)
        result[i] = np.mean(y[: L - k] * y[k:L]) if k < L else 0.0
    return result


# Keep the old name as an alias for backward compatibility.
empirical_log_autocovariance = empirical_autocovariance


def build_gmm_lags(Q: int) -> NDArray:
    """
    Build lag sequence as in Section 7.1: floor(2^(k/4)) for k=0..Q.

    Parameters
    ----------
    Q : int
        Maximum index.

    Returns
    -------
    NDArray of unique integer lags.
    """
    lags_raw = np.floor(2 ** (np.arange(Q + 1) / 4)).astype(int)
    lags = np.unique(lags_raw)
    lags = lags[lags > 0]
    return lags


def gmm_objective(
    params: NDArray,
    c_hat: NDArray,
    lags: NDArray,
    T: float,
) -> float:
    """
    GMM objective for the Gaussianized log-QV series (Appendix C.2, adapted).

        f(H, rho) = sum_k (C_hat_gauss(k) - rho*(1-(k/T)^{2H}))^2

    Parameters
    ----------
    params : NDArray of shape (2,)
        [H, rho].
    c_hat : NDArray
        Empirical autocovariance of the Gaussianized series at lags.
    lags : NDArray
        Lag values.
    T : float
        Decorrelation scale.

    Returns
    -------
    float
        Objective value.
    """
    H, rho = params
    # rho must be in (0, 1]: after Gaussianization the series has unit variance,
    # so the signal-to-total-variance ratio cannot exceed 1.
    if H <= 0 or H >= 0.5 or rho <= 0 or rho > 1.0:
        return 1e12
    try:
        c_theory = rho * sfbm_autocorr_theory(lags.astype(float), H, T)
        residuals = c_hat - c_theory
        return float(np.sum(residuals**2))
    except Exception:
        return 1e12


def estimate_hurst_gmm(
    log_qv: NDArray,
    T: float,
    Delta: float = 1.0,
    Q: int | None = None,
    Q_range: tuple[int, int] = (28, 40),
    rng: np.random.Generator | None = None,
    n_random_Q: int = 1,
) -> tuple[float, float]:
    """
    Estimate Hurst exponent H from a Gaussianized log-QV (or log-vol) series.

    The input ``log_qv`` is expected to be the output of ``gaussianize()`` – a
    unit-variance series whose autocorrelation follows rho*(1-(k/T)^{2H}).
    The function minimizes the GMM objective over (H, rho) and returns the mean
    H over all random-Q draws.  The second return value is a placeholder kept
    for API compatibility (returns rho, not lambda^2).

    Parameters
    ----------
    log_qv : NDArray of shape (L,)
        Gaussianized log-QV series (unit variance).
    T : float
        Decorrelation scale.
    Delta : float
        Unused; kept for API compatibility with calling code.
    Q : int or None
        Fixed Q for lag construction; sample from Q_range when None.
    Q_range : tuple of int
        (low, high) for uniform Q sampling.
    rng : np.random.Generator or None
        RNG for random Q sampling.
    n_random_Q : int
        Number of random Q values to average over.

    Returns
    -------
    tuple of (H_est, rho_est)
        Estimated Hurst exponent and signal fraction rho.
    """
    if rng is None:
        rng = np.random.default_rng()

    L = len(log_qv)
    # The input is the Gaussianized log-QV, which has zero population mean by
    # construction.  Do NOT subtract the sample mean: for long-memory processes
    # the sample mean has variance ≈ 2*rho*2H/(1+2H), which equals ~0.16 for
    # typical synthetic parameters.  Subtracting it introduces a constant
    # negative bias in all sample ACF estimates, deflating C_hat(k) by that
    # amount and pushing the GMM toward spuriously low H (≈0.04 instead of 0.12).
    y = log_qv  # zero-mean Gaussianized series; no centering

    h_estimates: list[float] = []
    rho_estimates: list[float] = []

    q_values: list[int]
    if Q is not None:
        q_values = [Q] * max(1, n_random_Q)
    else:
        q_values = rng.integers(Q_range[0], Q_range[1] + 1, size=max(1, n_random_Q)).tolist()

    for q in q_values:
        lags = build_gmm_lags(q)
        lags = lags[(lags < L) & (lags < T)]
        if len(lags) == 0:
            continue

        # demean=False: Gaussianized series already has zero population mean;
        # subtracting sample mean would reintroduce the long-memory bias.
        c_hat = empirical_autocovariance(y, lags, demean=False)

        # Multi-start to avoid local minima.
        # rho is bounded to [0.01, 0.97]: the Gaussianized series has unit variance,
        # so rho < 1; the upper bound 0.97 prevents the optimizer settling at the
        # spurious rho=1/H→0 local minimum that arises when chi-squared noise is
        # comparable to the S-fBM signal (small lambda^2 regimes).
        best_fun = np.inf
        best_x = np.array([0.1, 0.5])
        bounds = [(1e-4, 0.49), (0.01, 0.97)]
        for H_init in [0.03, 0.08, 0.15, 0.30, 0.45]:
            for rho_init in [0.10, 0.25, 0.40, 0.55, 0.70]:
                x0 = np.array([H_init, rho_init])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = minimize(
                        gmm_objective,
                        x0,
                        args=(c_hat, lags, T),
                        method="L-BFGS-B",
                        bounds=bounds,
                        options={"maxiter": 500, "ftol": 1e-12},
                    )
                if res.fun < best_fun:
                    best_fun = res.fun
                    best_x = res.x

        H_est, rho_est = best_x
        h_estimates.append(float(np.clip(H_est, 1e-4, 0.49)))
        rho_estimates.append(float(np.clip(rho_est, 0.01, 1.5)))

    if not h_estimates:
        return float("nan"), float("nan")

    return float(np.mean(h_estimates)), float(np.mean(rho_estimates))


def estimate_hurst_gmm_multi_Q(
    log_qv: NDArray,
    T: float,
    n_q_samples: int = 20,
    Q_range: tuple[int, int] = (28, 40),
    Delta: float = 1.0,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Estimate H by averaging over multiple random Q values (Section 7.1).

    Parameters
    ----------
    log_qv : NDArray
        Gaussianized log-QV series.
    T : float
        Decorrelation scale.
    n_q_samples : int
        Number of random Q draws.
    Q_range : tuple of int
        Uniform range for Q.
    Delta : float
        Unused; kept for API compatibility.
    rng : np.random.Generator or None
        RNG.

    Returns
    -------
    float
        Mean estimated H over the Q draws.
    """
    H_est, _ = estimate_hurst_gmm(
        log_qv, T, Delta=Delta, Q=None, Q_range=Q_range,
        rng=rng, n_random_Q=n_q_samples,
    )
    return H_est


def gaussianize(series: NDArray) -> NDArray:
    """
    Apply the Gaussianization transform from Eq. 76 / Appendix H.1.

        X_tilde = Phi^{-1}(F_Y(Y))

    where Phi is the standard normal CDF and F_Y is the empirical CDF of Y.
    The result has unit variance and matches the rank-correlation structure of Y.

    Parameters
    ----------
    series : NDArray of shape (L,)
        Input series to Gaussianize.

    Returns
    -------
    NDArray of shape (L,)
        Gaussianized series (unit variance, approximately zero mean).
    """
    from scipy.stats import norm

    n = len(series)
    ranks = np.argsort(np.argsort(series)) + 1
    empirical_cdf = ranks / (n + 1)
    empirical_cdf = np.clip(empirical_cdf, 1e-6, 1 - 1e-6)
    return norm.ppf(empirical_cdf)
