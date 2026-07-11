"""Keplerian radial-velocity modeling utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq

from exoplanet_est.constants import DAY_IN_SECONDS, G


@dataclass(frozen=True)
class OrbitalParameters:
    """Parameters for an edge-on single-planet radial-velocity model."""

    semi_amplitude_ms: float
    period_days: float
    eccentricity: float
    omega_rad: float
    t_periastron_days: float
    gamma_ms: float


def solve_kepler_equation(
    mean_anomaly_rad: np.ndarray | float,
    eccentricity: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 64,
) -> np.ndarray:
    """Solve Kepler's equation for the eccentric anomaly."""

    mean_anomaly = np.asarray(mean_anomaly_rad, dtype=float)
    eccentric_anomaly = np.where(
        eccentricity < 0.8,
        mean_anomaly,
        np.pi * np.ones_like(mean_anomaly),
    )

    for _ in range(max_iterations):
        delta = (
            eccentric_anomaly
            - eccentricity * np.sin(eccentric_anomaly)
            - mean_anomaly
        ) / (1.0 - eccentricity * np.cos(eccentric_anomaly))
        eccentric_anomaly -= delta
        if np.max(np.abs(delta)) < tolerance:
            break

    return eccentric_anomaly


def true_anomaly_from_eccentric_anomaly(
    eccentric_anomaly_rad: np.ndarray | float,
    eccentricity: float,
) -> np.ndarray:
    """Convert eccentric anomaly to true anomaly."""

    eccentric_anomaly = np.asarray(eccentric_anomaly_rad, dtype=float)
    factor = np.sqrt((1.0 + eccentricity) / (1.0 - eccentricity))
    return 2.0 * np.arctan2(
        factor * np.sin(eccentric_anomaly / 2.0),
        np.cos(eccentric_anomaly / 2.0),
    )


def radial_velocity_curve(
    parameters: OrbitalParameters,
    times_days: np.ndarray | float,
) -> np.ndarray:
    """Compute the stellar radial-velocity curve in m/s."""

    times_days = np.asarray(times_days, dtype=float)
    mean_motion = 2.0 * np.pi / parameters.period_days
    mean_anomaly = mean_motion * (times_days - parameters.t_periastron_days)
    eccentric_anomaly = solve_kepler_equation(mean_anomaly, parameters.eccentricity)
    true_anomaly = true_anomaly_from_eccentric_anomaly(
        eccentric_anomaly,
        parameters.eccentricity,
    )
    return (
        parameters.gamma_ms
        + parameters.semi_amplitude_ms
        * (
            np.cos(parameters.omega_rad + true_anomaly)
            + parameters.eccentricity * np.cos(parameters.omega_rad)
        )
    )


def semi_major_axis_from_period(
    period_days: float,
    total_mass_kg: float,
) -> float:
    """Compute orbital semi-major axis in meters from period and total mass."""

    period_seconds = period_days * DAY_IN_SECONDS
    return ((G * total_mass_kg * period_seconds**2) / (4.0 * np.pi**2)) ** (1.0 / 3.0)


def semi_amplitude_from_masses(
    star_mass_kg: float,
    planet_mass_kg: float,
    period_days: float,
    eccentricity: float,
    *,
    inclination_rad: float = np.pi / 2.0,
) -> float:
    """Compute radial-velocity semi-amplitude in m/s."""

    period_seconds = period_days * DAY_IN_SECONDS
    numerator = (2.0 * np.pi * G / period_seconds) ** (1.0 / 3.0)
    return (
        numerator
        * planet_mass_kg
        * np.sin(inclination_rad)
        / (star_mass_kg + planet_mass_kg) ** (2.0 / 3.0)
        / np.sqrt(1.0 - eccentricity**2)
    )


def estimate_planet_mass(
    star_mass_kg: float,
    semi_amplitude_ms: float,
    period_days: float,
    eccentricity: float,
    *,
    inclination_rad: float = np.pi / 2.0,
) -> float:
    """Estimate the planet mass from the RV semi-amplitude."""

    target = abs(semi_amplitude_ms)
    if target <= 0.0:
        raise ValueError("Semi-amplitude must be positive.")

    def residual(planet_mass_kg: float) -> float:
        return semi_amplitude_from_masses(
            star_mass_kg,
            planet_mass_kg,
            period_days,
            eccentricity,
            inclination_rad=inclination_rad,
        ) - target

    lower = 1e20
    upper = star_mass_kg * 0.2
    while residual(upper) < 0.0:
        upper *= 2.0
        if upper >= star_mass_kg:
            raise RuntimeError("Could not bracket a planet mass for the fitted amplitude.")

    return brentq(residual, lower, upper, maxiter=256)


def relative_state_vectors(
    semi_major_axis_m: float,
    eccentricity: float,
    true_anomaly_rad: float,
    total_mass_kg: float,
    *,
    omega_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return relative position and velocity vectors in the x-z plane."""

    mu = G * total_mass_kg
    p = semi_major_axis_m * (1.0 - eccentricity**2)
    radius = p / (1.0 + eccentricity * np.cos(true_anomaly_rad))

    x_peri = radius * np.cos(true_anomaly_rad)
    z_peri = radius * np.sin(true_anomaly_rad)

    speed_scale = np.sqrt(mu / p)
    vx_peri = -speed_scale * np.sin(true_anomaly_rad)
    vz_peri = speed_scale * (eccentricity + np.cos(true_anomaly_rad))

    cos_omega = math.cos(omega_rad)
    sin_omega = math.sin(omega_rad)

    position = np.array(
        [
            cos_omega * x_peri - sin_omega * z_peri,
            0.0,
            sin_omega * x_peri + cos_omega * z_peri,
        ],
        dtype=float,
    )
    velocity = np.array(
        [
            cos_omega * vx_peri - sin_omega * vz_peri,
            0.0,
            sin_omega * vx_peri + cos_omega * vz_peri,
        ],
        dtype=float,
    )
    return position, velocity
