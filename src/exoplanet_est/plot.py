"""Plotting helpers for the exoplanet showcase."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exoplanet_est.constants import AU_IN_METERS, DAY_IN_SECONDS, MASS_JUPITER, MASS_SUN
from exoplanet_est.data import RadialVelocityDataset
from exoplanet_est.keplerian import (
    OrbitalParameters,
    solve_kepler_equation,
    true_anomaly_from_eccentric_anomaly,
)
from exoplanet_est.nbody import barycentric_state_from_orbit, integrate_two_body
from exoplanet_est.optimize import RVFitResult, evaluate_on_dense_grid


@dataclass(frozen=True)
class ShowcaseTruth:
    """Optional truth metadata for synthetic demos."""

    parameters: OrbitalParameters
    planet_mass_kg: float
    star_mass_kg: float


def _orbit_simulation(
    fit_result: RVFitResult,
    *,
    reference_day: float,
    n_steps: int = 2000,
):
    mean_anomaly = (
        2.0
        * np.pi
        / fit_result.parameters.period_days
        * (reference_day - fit_result.parameters.t_periastron_days)
    )
    eccentric_anomaly = solve_kepler_equation(
        mean_anomaly,
        fit_result.parameters.eccentricity,
    )
    true_anomaly = float(
        true_anomaly_from_eccentric_anomaly(
            eccentric_anomaly,
            fit_result.parameters.eccentricity,
        )
    )

    star_position, star_velocity, planet_position, planet_velocity = (
        barycentric_state_from_orbit(
            fit_result.star_mass_kg,
            fit_result.planet_mass_kg,
            fit_result.semi_major_axis_m,
            fit_result.parameters.eccentricity,
            true_anomaly,
            omega_rad=fit_result.parameters.omega_rad,
        )
    )
    return integrate_two_body(
        fit_result.star_mass_kg,
        star_position,
        star_velocity,
        fit_result.planet_mass_kg,
        planet_position,
        planet_velocity,
        total_time_s=fit_result.parameters.period_days * DAY_IN_SECONDS,
        n_steps=n_steps,
    )


def plot_fit_summary(
    dataset: RadialVelocityDataset,
    fit_result: RVFitResult,
    *,
    truth: ShowcaseTruth | None = None,
    title: str = "Exoplanet Parameter Estimation",
):
    """Create the 16:9 showcase figure."""

    dense_times, dense_curve = evaluate_on_dense_grid(
        fit_result.parameters,
        start_day=float(dataset.times_days.min()),
        stop_day=float(dataset.times_days.max()),
    )
    orbit = _orbit_simulation(
        fit_result,
        reference_day=float(dataset.times_days.min()),
    )

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.075, right=0.97, top=0.9, bottom=0.18, wspace=0.12)

    axes[0].errorbar(
        dataset.times_days,
        dataset.radial_velocity_ms,
        yerr=dataset.uncertainty_ms,
        fmt="o",
        color="#0f172a",
        ecolor="#94a3b8",
        elinewidth=1.2,
        capsize=2,
        ms=5,
        label="observations",
    )
    axes[0].plot(dense_times, dense_curve, color="#2563eb", lw=2.5, label="best fit")
    if truth is not None:
        truth_curve = evaluate_on_dense_grid(
            truth.parameters,
            start_day=float(dataset.times_days.min()),
            stop_day=float(dataset.times_days.max()),
        )[1]
        axes[0].plot(
            dense_times,
            truth_curve,
            color="#16a34a",
            lw=1.6,
            ls="--",
            label="ground truth",
        )

    axes[0].set_title("Radial Velocity Fit")
    axes[0].set_xlabel("Time (days)")
    axes[0].set_ylabel("Stellar radial velocity (m/s)")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, loc="upper right")

    star_path = orbit.positions_m[0]
    planet_path = orbit.positions_m[1]
    axes[1].plot(
        planet_path[:, 0] / AU_IN_METERS,
        planet_path[:, 2] / AU_IN_METERS,
        color="#2563eb",
        lw=2.2,
        label="planet",
    )
    axes[1].plot(
        star_path[:, 0] / AU_IN_METERS,
        star_path[:, 2] / AU_IN_METERS,
        color="#f59e0b",
        lw=2.0,
        label="star",
    )
    axes[1].scatter(
        [planet_path[0, 0] / AU_IN_METERS],
        [planet_path[0, 2] / AU_IN_METERS],
        color="#1d4ed8",
        s=48,
    )
    axes[1].scatter(
        [star_path[0, 0] / AU_IN_METERS],
        [star_path[0, 2] / AU_IN_METERS],
        color="#d97706",
        s=48,
    )
    axes[1].set_title("Reflex Orbit (Velocity Verlet)")
    axes[1].set_xlabel("x (AU)")
    axes[1].set_ylabel("z (AU)")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False, loc="upper right")

    metrics = [
        f"P = {fit_result.parameters.period_days:.1f} d",
        f"K = {fit_result.parameters.semi_amplitude_ms:.1f} m/s",
        f"e = {fit_result.parameters.eccentricity:.3f}",
        f"a = {fit_result.semi_major_axis_m / AU_IN_METERS:.3f} AU",
        f"M_p = {fit_result.planet_mass_kg / MASS_JUPITER:.3f} Mj",
        f"M_* = {fit_result.star_mass_kg / MASS_SUN:.2f} Msun",
        f"RMS = {fit_result.residual_rms_ms:.2f} m/s",
    ]
    metric_lines = [
        "   |   ".join(metrics[:4]),
        "   |   ".join(metrics[4:]),
    ]
    fig.suptitle(title, fontsize=22, fontweight="bold")
    fig.text(
        0.5,
        0.07,
        "\n".join(metric_lines),
        ha="center",
        va="center",
        fontsize=10.5,
        linespacing=1.35,
        color="#334155",
    )
    return fig, axes


def save_figure(fig, path: str | Path) -> None:
    """Persist a Matplotlib figure."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
