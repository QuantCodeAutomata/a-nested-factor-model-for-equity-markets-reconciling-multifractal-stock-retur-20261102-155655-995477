"""Unit tests for src/sfbm.py — S-fBM covariance kernel and simulation."""

from __future__ import annotations

import numpy as np
import pytest

from src.sfbm import (
    sfbm_covariance,
    sfbm_mean,
    sfbm_variance,
    simulate_sfbm,
)


class TestSfbmCovariance:
    """Tests for the S-fBM covariance kernel (Eq. 2)."""

    def test_variance_at_zero_lag(self) -> None:
        """C(0) = nu^2/2 = lambda^2 / (2*H*(1-2H))."""
        H, lam2, T = 0.1, 0.01, 100.0
        expected = lam2 / (2 * H * (1 - 2 * H))
        assert abs(sfbm_covariance(np.array([0.0]), H, lam2, T)[0] - expected) < 1e-10

    def test_zero_beyond_T(self) -> None:
        """C(tau) = 0 for |tau| >= T."""
        H, lam2, T = 0.1, 0.01, 50.0
        cov = sfbm_covariance(np.array([T, T + 10, 2 * T]), H, lam2, T)
        assert np.all(cov == 0.0)

    def test_covariance_decreasing(self) -> None:
        """C(tau) should decrease as |tau| increases from 0 to T."""
        H, lam2, T = 0.1, 0.01, 100.0
        lags = np.arange(0, 80, 10, dtype=float)
        cov = sfbm_covariance(lags, H, lam2, T)
        assert np.all(np.diff(cov) <= 0), "Covariance should be non-increasing"

    def test_negative_lags_symmetric(self) -> None:
        """C(tau) = C(-tau) for the stationary process."""
        H, lam2, T = 0.15, 0.02, 100.0
        lags = np.array([1.0, 5.0, 20.0])
        cov_pos = sfbm_covariance(lags, H, lam2, T)
        cov_neg = sfbm_covariance(-lags, H, lam2, T)
        np.testing.assert_allclose(cov_pos, cov_neg)

    def test_variance_equals_sfbm_variance(self) -> None:
        """sfbm_variance should equal C(0)."""
        H, lam2, T = 0.1, 0.01, 100.0
        var = sfbm_variance(H, lam2)
        cov0 = sfbm_covariance(np.array([0.0]), H, lam2, T)[0]
        assert abs(var - cov0) < 1e-10


class TestSfbmMean:
    """Tests for the mean of the S-fBM process."""

    def test_mean_is_negative(self) -> None:
        """Mean should be -nu^2/4 < 0."""
        H, lam2 = 0.1, 0.01
        mu = sfbm_mean(H, lam2)
        assert mu < 0

    def test_mean_formula(self) -> None:
        """mu = -lambda^2 / (4*H*(1-2H))."""
        H, lam2 = 0.1, 0.01
        expected = -lam2 / (4 * H * (1 - 2 * H))
        assert abs(sfbm_mean(H, lam2) - expected) < 1e-12


class TestSimulateSfbm:
    """Tests for S-fBM path simulation."""

    def test_shape(self, tiny_sfbm_params: dict, rng: np.random.Generator) -> None:
        """Output shape matches (n_samples, L)."""
        p = tiny_sfbm_params
        n_samples = 5
        paths = simulate_sfbm(p["L"], p["H"], p["lambda2"], p["T"], n_samples=n_samples, rng=rng)
        assert paths.shape == (n_samples, p["L"])

    def test_not_all_zero(self, tiny_sfbm_params: dict, rng: np.random.Generator) -> None:
        """Simulated paths are not constant."""
        p = tiny_sfbm_params
        paths = simulate_sfbm(p["L"], p["H"], p["lambda2"], p["T"], n_samples=3, rng=rng)
        assert np.std(paths) > 1e-6

    def test_mean_of_exp_close_to_one(self, rng: np.random.Generator) -> None:
        """E[exp(omega_t)] ≈ 1 when add_mean=True (Eq. 3)."""
        H, lam2, T, L = 0.1, 0.005, 128, 512
        # Average over many paths and time steps
        n_samples = 100
        paths = simulate_sfbm(L, H, lam2, T, n_samples=n_samples, rng=rng, add_mean=True)
        mean_exp = np.mean(np.exp(paths))
        # Should be close to 1 with large enough samples
        assert 0.3 < mean_exp < 3.0, f"E[exp(omega)] = {mean_exp:.3f} far from 1"

    def test_stationarity_variance(self, rng: np.random.Generator) -> None:
        """Sample variance across a path should be close to nu^2/2."""
        H, lam2, T, L = 0.1, 0.01, 128, 1024
        n_samples = 20
        paths = simulate_sfbm(L, H, lam2, T, n_samples=n_samples, rng=rng, add_mean=False)
        # Sample variance at a single time step (averaged over paths)
        var_sample = np.var(paths)
        expected_var = lam2 / (2 * H * (1 - 2 * H))
        # Within 50% is sufficient for finite samples
        assert abs(var_sample - expected_var) / expected_var < 0.5, (
            f"Sample variance {var_sample:.4f} too far from expected {expected_var:.4f}"
        )

    def test_single_sample(self, tiny_sfbm_params: dict, rng: np.random.Generator) -> None:
        """Single path has correct length."""
        p = tiny_sfbm_params
        paths = simulate_sfbm(p["L"], p["H"], p["lambda2"], p["T"], n_samples=1, rng=rng)
        assert paths.shape == (1, p["L"])
        assert np.isfinite(paths).all()

    def test_reproducible_with_same_seed(self, tiny_sfbm_params: dict) -> None:
        """Same RNG seed yields identical paths."""
        p = tiny_sfbm_params
        paths1 = simulate_sfbm(p["L"], p["H"], p["lambda2"], p["T"], n_samples=2,
                                rng=np.random.default_rng(7))
        paths2 = simulate_sfbm(p["L"], p["H"], p["lambda2"], p["T"], n_samples=2,
                                rng=np.random.default_rng(7))
        np.testing.assert_array_equal(paths1, paths2)
