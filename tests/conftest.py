"""
Pytest configuration: session-scoped fixtures and marker registration.

Market-data tests are marked `market_data` and skipped when
DATALOADER_API_KEY is not set in the environment.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "market_data: tests that call the QuantX Dataloader API and require DATALOADER_API_KEY",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("DATALOADER_API_KEY"):
        return  # API key present; run market_data tests
    skip_market = pytest.mark.skip(reason="DATALOADER_API_KEY not set")
    for item in items:
        if "market_data" in item.keywords:
            item.add_marker(skip_market)


# ---------------------------------------------------------------------------
# Small synthetic fixtures for unit tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Fixed-seed RNG for reproducible unit tests."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def tiny_sfbm_params() -> dict:
    """Minimal S-fBM parameters for fast unit tests."""
    return {"H": 0.1, "lambda2": 0.05**2, "T": 128, "L": 256}


@pytest.fixture(scope="session")
def tiny_sim_params() -> dict:
    """Minimal N-SfFM simulation parameters for fast unit tests."""
    return {
        "N": 10,
        "L": 256,
        "T": 128,
        "s": 16,
        "H": 0.11,
        "lambda2": 0.05**2,
        "H_i": 0.01,
        "lambda2_i": 0.05**2,
        "sigma_i": 1.0,
        "gamma_i": 0.2,
    }
