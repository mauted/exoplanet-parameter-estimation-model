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
class PlotTheme:
    """Visual theme for showcase or print figures."""

    name: str
    figure_bg: str
    axes_bg: str
    text: str
    muted_text: str
    spine: str
    grid: str
    observation: str
    observation_error: str
    fit: str
    truth: str
    planet: str
    star: str
    legend_frame: bool


DARK_THEME = PlotTheme(
    name="dark",
    figure_bg="#020617",
    axes_bg="#0f172a",
    text="#e2e8f0",
    muted_text="#cbd5e1",
    spine="#475569",
    grid="#94a3b8",
    observation="#f8fafc",
    observation_error="#64748b",
    fit="#2563eb",
    truth="#16a34a",
    planet="#2563eb",
    star="#f59e0b",
    legend_frame=True,
)

PRINT_THEME = PlotTheme(
    name="print",
    figure_bg="#ffffff",
    axes_bg="#ffffff",
    text="#111827",
    muted_text="#374151",
    spine="#6b7280",
    grid="#9ca3af",
    observation="#111827",
    observation_error="#9ca3af",
    fit="#1d4ed8",
    truth="#15803d",
    planet="#1d4ed8",
    star="#b45309",
    legend_frame=False,
)

THEMES = {
    "dark": DARK_THEME,
    "print": PRINT_THEME,
}


@dataclass(frozen=True)
class ShowcaseTruth:
    """Optional truth metadata for synthetic demos."""

    parameters: OrbitalParameters
    planet_mass_kg: float
    star_mass_kg: float


@dataclass(frozen=True)
class PlotSeries:
    """Arrays needed to rebuild figures in LaTeX/PGFPlots."""

    times_days: np.ndarray
    radial_velocity_ms: np.ndarray
    uncertainty_ms: np.ndarray
    model_times_days: np.ndarray
    model_velocity_ms: np.ndarray
    truth_velocity_ms: np.ndarray | None
    star_x_au: np.ndarray
    star_z_au: np.ndarray
    planet_x_au: np.ndarray
    planet_z_au: np.ndarray


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


def build_plot_series(
    dataset: RadialVelocityDataset,
    fit_result: RVFitResult,
    *,
    truth: ShowcaseTruth | None = None,
) -> PlotSeries:
    """Build the arrays used by Matplotlib and PGFPlots exporters."""

    dense_times, dense_curve = evaluate_on_dense_grid(
        fit_result.parameters,
        start_day=float(dataset.times_days.min()),
        stop_day=float(dataset.times_days.max()),
    )
    orbit = _orbit_simulation(
        fit_result,
        reference_day=float(dataset.times_days.min()),
    )
    truth_curve = None
    if truth is not None:
        truth_curve = evaluate_on_dense_grid(
            truth.parameters,
            start_day=float(dataset.times_days.min()),
            stop_day=float(dataset.times_days.max()),
        )[1]

    star_path = orbit.positions_m[0]
    planet_path = orbit.positions_m[1]
    return PlotSeries(
        times_days=np.asarray(dataset.times_days, dtype=float),
        radial_velocity_ms=np.asarray(dataset.radial_velocity_ms, dtype=float),
        uncertainty_ms=np.asarray(dataset.uncertainty_ms, dtype=float),
        model_times_days=dense_times,
        model_velocity_ms=dense_curve,
        truth_velocity_ms=None if truth_curve is None else np.asarray(truth_curve, dtype=float),
        star_x_au=star_path[:, 0] / AU_IN_METERS,
        star_z_au=star_path[:, 2] / AU_IN_METERS,
        planet_x_au=planet_path[:, 0] / AU_IN_METERS,
        planet_z_au=planet_path[:, 2] / AU_IN_METERS,
    )


def export_plot_series(series: PlotSeries, stem: str | Path) -> dict[str, Path]:
    """Write CSV tables for native LaTeX/PGFPlots figure reconstruction."""

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    obs_path = stem.with_name(f"{stem.name}_rv_obs.csv")
    model_path = stem.with_name(f"{stem.name}_rv_model.csv")
    orbit_path = stem.with_name(f"{stem.name}_orbit.csv")

    np.savetxt(
        obs_path,
        np.column_stack(
            [series.times_days, series.radial_velocity_ms, series.uncertainty_ms]
        ),
        delimiter=",",
        header="time_days,radial_velocity_ms,uncertainty_ms",
        comments="",
    )

    if series.truth_velocity_ms is None:
        model_array = np.column_stack(
            [series.model_times_days, series.model_velocity_ms]
        )
        model_header = "time_days,model_velocity_ms"
    else:
        model_array = np.column_stack(
            [
                series.model_times_days,
                series.model_velocity_ms,
                series.truth_velocity_ms,
            ]
        )
        model_header = "time_days,model_velocity_ms,truth_velocity_ms"
    np.savetxt(
        model_path,
        model_array,
        delimiter=",",
        header=model_header,
        comments="",
    )

    # Downsample the orbit path for compact PGFPlots tables.
    stride = max(len(series.planet_x_au) // 400, 1)
    np.savetxt(
        orbit_path,
        np.column_stack(
            [
                series.star_x_au[::stride],
                series.star_z_au[::stride],
                series.planet_x_au[::stride],
                series.planet_z_au[::stride],
            ]
        ),
        delimiter=",",
        header="star_x_au,star_z_au,planet_x_au,planet_z_au",
        comments="",
    )
    return {"obs": obs_path, "model": model_path, "orbit": orbit_path}


def plot_fit_summary(
    dataset: RadialVelocityDataset,
    fit_result: RVFitResult,
    *,
    truth: ShowcaseTruth | None = None,
    title: str = "Exoplanet Parameter Estimation",
    theme: str | PlotTheme = "dark",
):
    """Create the 16:9 showcase figure."""

    theme = THEMES[theme] if isinstance(theme, str) else theme
    series = build_plot_series(dataset, fit_result, truth=truth)

    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.patch.set_facecolor(theme.figure_bg)
    fig.subplots_adjust(left=0.075, right=0.97, top=0.9, bottom=0.18, wspace=0.12)

    for axis in axes:
        axis.set_facecolor(theme.axes_bg)
        axis.tick_params(colors=theme.text)
        axis.xaxis.label.set_color(theme.text)
        axis.yaxis.label.set_color(theme.text)
        axis.title.set_color(theme.text)
        for spine in axis.spines.values():
            spine.set_color(theme.spine)

    axes[0].errorbar(
        series.times_days,
        series.radial_velocity_ms,
        yerr=series.uncertainty_ms,
        fmt="o",
        color=theme.observation,
        ecolor=theme.observation_error,
        elinewidth=1.2,
        capsize=2,
        ms=5,
        label="observations",
    )
    axes[0].plot(
        series.model_times_days,
        series.model_velocity_ms,
        color=theme.fit,
        lw=2.5,
        label="best fit",
    )
    if series.truth_velocity_ms is not None:
        axes[0].plot(
            series.model_times_days,
            series.truth_velocity_ms,
            color=theme.truth,
            lw=1.6,
            ls="--",
            label="ground truth",
        )

    axes[0].set_title("Radial Velocity Fit")
    axes[0].set_xlabel("Time (days)")
    axes[0].set_ylabel("Stellar radial velocity (m/s)")
    axes[0].grid(color=theme.grid, alpha=0.25 if theme.name == "print" else 0.14)
    legend = axes[0].legend(frameon=theme.legend_frame, loc="upper right")
    if theme.legend_frame:
        legend.get_frame().set_facecolor(theme.axes_bg)
        legend.get_frame().set_edgecolor(theme.spine)
        for text in legend.get_texts():
            text.set_color(theme.text)

    axes[1].plot(
        series.planet_x_au,
        series.planet_z_au,
        color=theme.planet,
        lw=2.2,
        label="planet",
    )
    axes[1].plot(
        series.star_x_au,
        series.star_z_au,
        color=theme.star,
        lw=2.0,
        label="star",
    )
    axes[1].scatter(
        [series.planet_x_au[0]],
        [series.planet_z_au[0]],
        color=theme.planet,
        s=48,
    )
    axes[1].scatter(
        [series.star_x_au[0]],
        [series.star_z_au[0]],
        color=theme.star,
        s=48,
    )
    axes[1].set_title("Reflex Orbit (Velocity Verlet)")
    axes[1].set_xlabel("x (AU)")
    axes[1].set_ylabel("z (AU)")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(color=theme.grid, alpha=0.25 if theme.name == "print" else 0.14)
    legend = axes[1].legend(frameon=theme.legend_frame, loc="upper right")
    if theme.legend_frame:
        legend.get_frame().set_facecolor(theme.axes_bg)
        legend.get_frame().set_edgecolor(theme.spine)
        for text in legend.get_texts():
            text.set_color(theme.text)

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
    fig.suptitle(title, fontsize=22, fontweight="bold", color=theme.text)
    fig.text(
        0.5,
        0.07,
        "\n".join(metric_lines),
        ha="center",
        va="center",
        fontsize=10.5,
        linespacing=1.35,
        color=theme.muted_text,
    )
    return fig, axes, series


def save_figure(fig, path: str | Path) -> None:
    """Persist a Matplotlib figure as PNG or vector PDF."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "bbox_inches": "tight",
        "facecolor": fig.get_facecolor(),
    }
    if path.suffix.lower() == ".pdf":
        fig.savefig(path, format="pdf", **save_kwargs)
    else:
        fig.savefig(path, dpi=180, **save_kwargs)
