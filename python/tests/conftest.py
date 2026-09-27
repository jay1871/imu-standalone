"""Shared fixtures for imu_nbk tests."""

from pathlib import Path

import pytest
import numpy as np

from imu_nbk.params import load_config, ImuNbkParams


@pytest.fixture
def config_path():
    """Path to the shared ADIS16607 YAML config."""
    return Path(__file__).parent.parent.parent / "config" / "adis16607.yaml"


@pytest.fixture
def params(config_path):
    """Load ImuNbkParams from YAML."""
    return load_config(config_path)


@pytest.fixture
def fixed_draws_one_step():
    """Fixed random draws for deterministic Python testing.

    Injects known values [1.0, 2.0, 3.0] per axis for exact numeric verification.
    The C++ side achieves determinism via a fixed seed instead.
    Order: 3 draws per axis × 6 axes = 18 total.
    """
    return np.array([1.0, 2.0, 3.0] * 6)
