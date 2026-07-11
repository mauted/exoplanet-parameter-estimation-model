"""Two-body orbit integration for visualization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from exoplanet_est.constants import G
from exoplanet_est.keplerian import relative_state_vectors


@dataclass(frozen=True)
class TwoBodySimulation:
    """Integrated barycentric star/planet trajectories."""

    times_seconds: np.ndarray
    positions_m: np.ndarray
    velocities_ms: np.ndarray


def _accelerations(
    star_position_m: np.ndarray,
    planet_position_m: np.ndarray,
    star_mass_kg: float,
    planet_mass_kg: float,
) -> tuple[np.ndarray, np.ndarray]:
    relative_position = planet_position_m - star_position_m
    distance = np.linalg.norm(relative_position)
    if distance == 0.0:
        raise ValueError("Star and planet positions overlap.")

    scale = G / distance**3
    star_acc = scale * planet_mass_kg * relative_position
    planet_acc = -scale * star_mass_kg * relative_position
    return star_acc, planet_acc


def barycentric_state_from_orbit(
    star_mass_kg: float,
    planet_mass_kg: float,
    semi_major_axis_m: float,
    eccentricity: float,
    true_anomaly_rad: float,
    *,
    omega_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct barycentric initial conditions from orbital elements."""

    relative_position, relative_velocity = relative_state_vectors(
        semi_major_axis_m,
        eccentricity,
        true_anomaly_rad,
        star_mass_kg + planet_mass_kg,
        omega_rad=omega_rad,
    )
    mass_ratio = planet_mass_kg / (star_mass_kg + planet_mass_kg)
    star_position = -mass_ratio * relative_position
    planet_position = (1.0 - mass_ratio) * relative_position
    star_velocity = -mass_ratio * relative_velocity
    planet_velocity = (1.0 - mass_ratio) * relative_velocity
    return star_position, star_velocity, planet_position, planet_velocity


def integrate_two_body(
    star_mass_kg: float,
    star_position_m: np.ndarray,
    star_velocity_ms: np.ndarray,
    planet_mass_kg: float,
    planet_position_m: np.ndarray,
    planet_velocity_ms: np.ndarray,
    *,
    total_time_s: float,
    n_steps: int = 2000,
) -> TwoBodySimulation:
    """Integrate a two-body orbit with the velocity Verlet method."""

    times = np.linspace(0.0, total_time_s, n_steps + 1)
    dt = total_time_s / n_steps

    positions = np.zeros((2, n_steps + 1, 3), dtype=float)
    velocities = np.zeros((2, n_steps + 1, 3), dtype=float)
    positions[0, 0] = np.asarray(star_position_m, dtype=float)
    positions[1, 0] = np.asarray(planet_position_m, dtype=float)
    velocities[0, 0] = np.asarray(star_velocity_ms, dtype=float)
    velocities[1, 0] = np.asarray(planet_velocity_ms, dtype=float)

    star_acc, planet_acc = _accelerations(
        positions[0, 0],
        positions[1, 0],
        star_mass_kg,
        planet_mass_kg,
    )

    for index in range(n_steps):
        positions[0, index + 1] = (
            positions[0, index] + velocities[0, index] * dt + 0.5 * star_acc * dt**2
        )
        positions[1, index + 1] = (
            positions[1, index] + velocities[1, index] * dt + 0.5 * planet_acc * dt**2
        )

        next_star_acc, next_planet_acc = _accelerations(
            positions[0, index + 1],
            positions[1, index + 1],
            star_mass_kg,
            planet_mass_kg,
        )
        velocities[0, index + 1] = (
            velocities[0, index] + 0.5 * (star_acc + next_star_acc) * dt
        )
        velocities[1, index + 1] = (
            velocities[1, index] + 0.5 * (planet_acc + next_planet_acc) * dt
        )
        star_acc = next_star_acc
        planet_acc = next_planet_acc

    return TwoBodySimulation(
        times_seconds=times,
        positions_m=positions,
        velocities_ms=velocities,
    )
