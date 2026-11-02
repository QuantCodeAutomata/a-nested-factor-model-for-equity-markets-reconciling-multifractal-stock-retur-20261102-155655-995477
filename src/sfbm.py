"""
Stationary fractional Brownian Motion (S-fBM) simulation and covariance kernel.

The S-fBM is a stationary Gaussian process with autocovariance (Eq. 2):
    C(tau) = nu^2/2 * (1 - (tau/T)^(2H))  for |tau| < T
             0                              otherwise

where nu^2 = lambda^2 / (H*(1-2H)) (Eq. 5).

The mean is chosen so E[exp(omega)] = 1 (Eq. 3):
    mu = -nu^2/4  (derived from Eq. 50)

Simulation uses the Davies-Harte circulant-embedding method for efficient
generation of stationary Gaussian processes.

Custom implementation (paper Eqs. 2, 5, Section 2.1).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sfbm_variance(H: float, lambda2: float) -> float:
    """
    Compute the point variance of the S-fBM, nu^2/2 (Eq. 2 at tau=0).

    Parameters
    ----------
    H : float
        Hurst exponent in (0, 0.5).
    lambda2 : float
        Intermittency coefficient lambda^2 (Eq. 5).

    Returns
    -------
    float
        Variance = nu^2/2 where nu^2 = lambda^2 / (H*(1-2H)).
    """
    nu2 = lambda2 / (H * (1 - 2 * H))
    return nu2 / 2


def sfbm_mean(H: float, lambda2: float) -> float:
    """
    Compute the mean of the S-fBM process so that E[exp(omega)] = 1 (Eq. 50).

    mu = -lambda^2 / (4*H*(1-2H)) = -nu^2/4

    Parameters
    ----------
    H : float
        Hurst exponent in (0, 0.5).
    lambda2 : float
        Intermittency coefficient lambda^2.

    Returns
    -------
    float
        Mean value mu.
    """
    nu2 = lambda2 / (H * (1 - 2 * H))
    return -nu2 / 4


def sfbm_covariance(tau: NDArray, H: float, lambda2: float, T: float) -> NDArray:
    """
    Compute the autocovariance of S-fBM at lag tau (Eq. 2).

    C(tau) = nu^2/2 * (1 - (|tau|/T)^(2H))  for |tau| < T
             0                                 otherwise

    Parameters
    ----------
    tau : NDArray
        Array of time lags.
    H : float
        Hurst exponent in (0, 0.5).
    lambda2 : float
        Intermittency coefficient.
    T : float
        Decorrelation time scale.

    Returns
    -------
    NDArray
        Autocovariance values.
    """
    nu2 = lambda2 / (H * (1 - 2 * H))
    tau_abs = np.abs(tau)
    cov = np.where(
        tau_abs < T,
        nu2 / 2 * (1 - (tau_abs / T) ** (2 * H)),
        0.0,
    )
    return cov


def simulate_sfbm(
    L: int,
    H: float,
    lambda2: float,
    T: int,
    n_samples: int = 1,
    rng: np.random.Generator | None = None,
    add_mean: bool = True,
) -> NDArray:
    """
    Simulate S-fBM paths using the Davies-Harte circulant-embedding method.

    Generates a stationary Gaussian process with covariance from Eq. 2,
    optionally adding the mean so that E[exp(omega)] = 1 (Eq. 3/50).

    Parameters
    ----------
    L : int
        Length of each simulated path (coarse time steps).
    H : float
        Hurst exponent in (0, 0.5).
    lambda2 : float
        Intermittency coefficient lambda^2.
    T : int
        Decorrelation scale (coarse periods); covariance is zero for |tau| >= T.
    n_samples : int
        Number of independent paths to generate.
    rng : np.random.Generator or None
        Random number generator; uses default if None.
    add_mean : bool
        If True, add the mean -nu^2/4 so that E[exp(omega)] = 1.

    Returns
    -------
    NDArray of shape (n_samples, L)
        Simulated S-fBM paths.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Build covariance sequence for circulant embedding
    # Size of circulant must be >= L + T to capture full correlation range.
    # Use next power of 2 >= 2*max(L, T) for efficiency.
    min_size = L + min(L, T)
    m = 1
    while m < min_size:
        m <<= 1
    m = max(m, 2 * L)  # ensure at least 2L

    lags = np.arange(m)
    cov_seq = sfbm_covariance(lags, H, lambda2, float(T))
    # Build symmetric circulant: [c0, c1, ..., c_{m/2}, c_{m/2-1}, ..., c1]
    circ = np.zeros(m)
    circ[0] = cov_seq[0]
    if m > 1:
        circ[1 : m // 2 + 1] = cov_seq[1 : m // 2 + 1]
        circ[m // 2 + 1 :] = cov_seq[m // 2 - 1 : 0 : -1]

    # Eigenvalues of circulant via FFT
    eigvals = np.real(np.fft.fft(circ))

    # Handle small negative eigenvalues due to discretization error
    neg_frac = np.sum(eigvals < 0) / m
    if neg_frac > 0.05:
        raise RuntimeError(
            f"Too many negative eigenvalues ({neg_frac:.1%}) in circulant embedding. "
            "Try larger m or different parameters."
        )
    eigvals = np.maximum(eigvals, 0.0)

    # Generate n_samples paths at once
    # Each path: z ~ CN(0, diag(eigvals)) in frequency domain, then IFFT
    # For real output: use the real-valued trick
    sqrt_eig = np.sqrt(eigvals / m)  # scale for IFFT normalization

    # Generate (n_samples, m) complex random normals
    z_real = rng.standard_normal((n_samples, m))
    z_imag = rng.standard_normal((n_samples, m))
    z = (z_real + 1j * z_imag) / np.sqrt(2)

    # Multiply by sqrt(eigenvalues) and inverse FFT.
    # Scaling: Var(Re(IFFT(z_scaled))[t]) = sum(eigvals)/(2m^3) = c[0]/(2m^2).
    # Multiplying by m*sqrt(2) gives Var = c[0] = nu^2/2.  (Custom derivation.)
    z_scaled = z * sqrt_eig[None, :]
    paths_full = np.real(np.fft.ifft(z_scaled, axis=1)) * (m * np.sqrt(2))

    # Extract first L time steps
    paths = paths_full[:, :L]

    if add_mean:
        mu = sfbm_mean(H, lambda2)
        paths += mu

    return paths  # shape (n_samples, L)
