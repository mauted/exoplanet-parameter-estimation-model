"""Relativistic Doppler conversion helpers."""

from __future__ import annotations

import numpy as np

from exoplanet_est.constants import SPEED_OF_LIGHT


def velocity_to_doppler_shift(velocity: np.ndarray | float) -> np.ndarray:
    """Convert radial velocity in m/s to Doppler shift."""
    beta = np.asarray(velocity, dtype=float) / SPEED_OF_LIGHT
    return np.sqrt((1.0 + beta) / (1.0 - beta)) - 1.0


def doppler_shift_to_velocity(shift: np.ndarray | float) -> np.ndarray:
    """Convert Doppler shift to radial velocity in m/s."""
    shift = np.asarray(shift, dtype=float)
    k = (1.0 + shift) ** 2
    return SPEED_OF_LIGHT * (k - 1.0) / (k + 1.0)
