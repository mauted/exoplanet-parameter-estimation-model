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
    radial_velocity_curve,
    solve_kepler_equation,
    true_anomaly_from_eccentric_anomaly,
)
from exoplanet_est.nbody import barycentric_state_from_orbit, integrate_two_body
from exoplanet_est.optimize import (
    OptimizationHistory,
    OptimizationTracePoint,
    RVFitResult,
    evaluate_on_dense_grid,
    fit_result_from_trace_point,
)


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
    phase_fold: bool = False
    x_label: str = "Time (days)"
    rv_title: str = "Radial Velocity Fit"


def should_phase_fold(
    dataset: RadialVelocityDataset,
    period_days: float,
    *,
    min_cycles: float = 8.0,
) -> bool:
    """Recommend phase-folding when many orbital cycles span the baseline."""

    time_span = float(dataset.times_days.max() - dataset.times_days.min())
    if period_days <= 0.0:
        return False
    return (time_span / period_days) >= min_cycles


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
    n_orbit_steps: int = 2000,
    phase_fold: bool | None = None,
) -> PlotSeries:
    """Build the arrays used by Matplotlib and PGFPlots exporters."""

    period = float(fit_result.parameters.period_days)
    fold = (
        should_phase_fold(dataset, period)
        if phase_fold is None
        else bool(phase_fold)
    )

    orbit = _orbit_simulation(
        fit_result,
        reference_day=float(dataset.times_days.min()),
        n_steps=n_orbit_steps,
    )
    star_path = orbit.positions_m[0]
    planet_path = orbit.positions_m[1]

    if fold:
        t0 = float(fit_result.parameters.t_periastron_days)
        obs_phase = np.mod(
            (np.asarray(dataset.times_days, dtype=float) - t0) / period,
            1.0,
        )
        order = np.argsort(obs_phase)
        model_phase = np.linspace(0.0, 1.0, 400)
        model_times = t0 + model_phase * period
        model_curve = radial_velocity_curve(fit_result.parameters, model_times)
        truth_curve = None
        if truth is not None:
            truth_curve = radial_velocity_curve(truth.parameters, model_times)
        return PlotSeries(
            times_days=obs_phase[order],
            radial_velocity_ms=np.asarray(dataset.radial_velocity_ms, dtype=float)[
                order
            ],
            uncertainty_ms=np.asarray(dataset.uncertainty_ms, dtype=float)[order],
            model_times_days=model_phase,
            model_velocity_ms=np.asarray(model_curve, dtype=float),
            truth_velocity_ms=(
                None if truth_curve is None else np.asarray(truth_curve, dtype=float)
            ),
            star_x_au=star_path[:, 0] / AU_IN_METERS,
            star_z_au=star_path[:, 2] / AU_IN_METERS,
            planet_x_au=planet_path[:, 0] / AU_IN_METERS,
            planet_z_au=planet_path[:, 2] / AU_IN_METERS,
            phase_fold=True,
            x_label="Orbital phase",
            rv_title="Phase-Folded Radial Velocity",
        )

    dense_times, dense_curve = evaluate_on_dense_grid(
        fit_result.parameters,
        start_day=float(dataset.times_days.min()),
        stop_day=float(dataset.times_days.max()),
    )
    truth_curve = None
    if truth is not None:
        truth_curve = evaluate_on_dense_grid(
            truth.parameters,
            start_day=float(dataset.times_days.min()),
            stop_day=float(dataset.times_days.max()),
        )[1]

    return PlotSeries(
        times_days=np.asarray(dataset.times_days, dtype=float),
        radial_velocity_ms=np.asarray(dataset.radial_velocity_ms, dtype=float),
        uncertainty_ms=np.asarray(dataset.uncertainty_ms, dtype=float),
        model_times_days=dense_times,
        model_velocity_ms=dense_curve,
        truth_velocity_ms=(
            None if truth_curve is None else np.asarray(truth_curve, dtype=float)
        ),
        star_x_au=star_path[:, 0] / AU_IN_METERS,
        star_z_au=star_path[:, 2] / AU_IN_METERS,
        planet_x_au=planet_path[:, 0] / AU_IN_METERS,
        planet_z_au=planet_path[:, 2] / AU_IN_METERS,
        phase_fold=False,
        x_label="Time (days)",
        rv_title="Radial Velocity Fit",
    )


def export_plot_series(series: PlotSeries, stem: str | Path) -> dict[str, Path]:
    """Write CSV tables for native LaTeX/PGFPlots figure reconstruction."""

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    obs_path = stem.with_name(f"{stem.name}_rv_obs.csv")
    model_path = stem.with_name(f"{stem.name}_rv_model.csv")
    orbit_path = stem.with_name(f"{stem.name}_orbit.csv")
    x_name = "phase" if series.phase_fold else "time_days"

    np.savetxt(
        obs_path,
        np.column_stack(
            [series.times_days, series.radial_velocity_ms, series.uncertainty_ms]
        ),
        delimiter=",",
        header=f"{x_name},radial_velocity_ms,uncertainty_ms",
        comments="",
    )

    if series.truth_velocity_ms is None:
        model_array = np.column_stack(
            [series.model_times_days, series.model_velocity_ms]
        )
        model_header = f"{x_name},model_velocity_ms"
    else:
        model_array = np.column_stack(
            [
                series.model_times_days,
                series.model_velocity_ms,
                series.truth_velocity_ms,
            ]
        )
        model_header = f"{x_name},model_velocity_ms,truth_velocity_ms"
    np.savetxt(
        model_path,
        model_array,
        delimiter=",",
        header=model_header,
        comments="",
    )

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
    status_line: str | None = None,
    n_orbit_steps: int = 2000,
    fixed_limits: dict[str, tuple[float, float]] | None = None,
    phase_fold: bool | None = None,
):
    """Create the 16:9 showcase figure."""

    theme = THEMES[theme] if isinstance(theme, str) else theme
    series = build_plot_series(
        dataset,
        fit_result,
        truth=truth,
        n_orbit_steps=n_orbit_steps,
        phase_fold=phase_fold,
    )

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
        zorder=3,
    )
    axes[0].plot(
        series.model_times_days,
        series.model_velocity_ms,
        color=theme.fit,
        lw=2.5,
        label="best fit",
        zorder=2,
    )
    if series.truth_velocity_ms is not None:
        axes[0].plot(
            series.model_times_days,
            series.truth_velocity_ms,
            color=theme.truth,
            lw=1.6,
            ls="--",
            label="ground truth",
            zorder=1,
        )

    axes[0].set_title(series.rv_title)
    axes[0].set_xlabel(series.x_label)
    axes[0].set_ylabel("Stellar radial velocity (m/s)")
    if series.phase_fold:
        axes[0].set_xlim(0.0, 1.0)
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

    if fixed_limits is not None:
        if "rv_x" in fixed_limits:
            axes[0].set_xlim(*fixed_limits["rv_x"])
        if "rv_y" in fixed_limits:
            axes[0].set_ylim(*fixed_limits["rv_y"])
        if "orbit_x" in fixed_limits:
            axes[1].set_xlim(*fixed_limits["orbit_x"])
        if "orbit_z" in fixed_limits:
            axes[1].set_ylim(*fixed_limits["orbit_z"])

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
    if status_line:
        metric_lines.append(status_line)
    fig.suptitle(title, fontsize=22, fontweight="bold", color=theme.text)
    fig.text(
        0.5,
        0.055 if status_line else 0.07,
        "\n".join(metric_lines),
        ha="center",
        va="center",
        fontsize=10.5,
        linespacing=1.35,
        color=theme.muted_text,
    )
    return fig, axes, series


def select_history_frames(
    history: OptimizationHistory,
    *,
    max_frames: int = 48,
) -> list[OptimizationTracePoint]:
    """Pick a compact subsequence of optimizer snapshots for animation."""

    points = list(history.points)
    if len(points) <= max_frames:
        return points

    sample_count = max_frames - 1
    positions = np.unique(
        np.round(np.geomspace(1, len(points) - 1, num=sample_count)).astype(int) - 1
    )
    selected = [points[index] for index in positions]
    if points[-1] not in selected:
        selected.append(points[-1])
    return selected


def fixed_limits_from_fit(
    dataset: RadialVelocityDataset,
    fit_result: RVFitResult,
    *,
    n_orbit_steps: int = 400,
    phase_fold: bool | None = None,
) -> dict[str, tuple[float, float]]:
    """Freeze axis ranges from the final fit so the GIF does not jump around."""

    series = build_plot_series(
        dataset,
        fit_result,
        n_orbit_steps=n_orbit_steps,
        phase_fold=phase_fold,
    )
    rv_pad = 0.08 * max(
        float(np.ptp(series.radial_velocity_ms)),
        float(np.ptp(series.model_velocity_ms)),
        1.0,
    )
    rv_ymin = min(
        float(series.radial_velocity_ms.min()),
        float(series.model_velocity_ms.min()),
    )
    rv_ymax = max(
        float(series.radial_velocity_ms.max()),
        float(series.model_velocity_ms.max()),
    )
    orbit_x = np.concatenate([series.planet_x_au, series.star_x_au])
    orbit_z = np.concatenate([series.planet_z_au, series.star_z_au])
    orbit_pad = 0.08 * max(float(np.ptp(orbit_x)), float(np.ptp(orbit_z)), 0.1)
    if series.phase_fold:
        rv_x = (0.0, 1.0)
    else:
        rv_x = (float(dataset.times_days.min()), float(dataset.times_days.max()))
    return {
        "rv_x": rv_x,
        "rv_y": (rv_ymin - rv_pad, rv_ymax + rv_pad),
        "orbit_x": (float(orbit_x.min()) - orbit_pad, float(orbit_x.max()) + orbit_pad),
        "orbit_z": (float(orbit_z.min()) - orbit_pad, float(orbit_z.max()) + orbit_pad),
    }


def render_fit_evolution_gif(
    dataset: RadialVelocityDataset,
    history: OptimizationHistory,
    *,
    star_mass_kg: float,
    output_path: str | Path,
    title: str = "Radial Velocity Fit Evolution",
    theme: str | PlotTheme = "dark",
    max_frames: int = 48,
    frame_duration_ms: int = 120,
    final_hold_frames: int = 12,
    n_orbit_steps: int = 400,
    phase_fold: bool | None = None,
) -> Path:
    """Render a dark-theme GIF of the best-so-far fit across optimizer generations."""

    import io

    from PIL import Image

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = select_history_frames(history, max_frames=max_frames)
    final_result = fit_result_from_trace_point(
        frames[-1],
        star_mass_kg=star_mass_kg,
        dataset=dataset,
    )
    limits = fixed_limits_from_fit(
        dataset,
        final_result,
        n_orbit_steps=n_orbit_steps,
        phase_fold=phase_fold,
    )

    images: list[Image.Image] = []
    for point in frames:
        snapshot = fit_result_from_trace_point(
            point,
            star_mass_kg=star_mass_kg,
            dataset=dataset,
        )
        fig, _, _ = plot_fit_summary(
            dataset,
            snapshot,
            title=title,
            theme=theme,
            status_line=_status_line(point),
            n_orbit_steps=n_orbit_steps,
            fixed_limits=limits,
            phase_fold=phase_fold,
        )
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=110,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
        )
        plt.close(fig)
        buffer.seek(0)
        images.append(Image.open(buffer).convert("RGB"))

    if not images:
        raise ValueError("No frames available to render a fit-evolution GIF.")

    held = images + [images[-1]] * final_hold_frames
    held[0].save(
        output_path,
        save_all=True,
        append_images=held[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )
    return output_path


def _status_line(point: OptimizationTracePoint) -> str:
    stage = "DE" if point.stage == "de" else "LS polish"
    return (
        f"{stage}  |  generation {point.generation}  |  "
        f"chi^2 = {point.chi2:.1f}  |  RMS = {point.residual_rms_ms:.2f} m/s"
    )


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
