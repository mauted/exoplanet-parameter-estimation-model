"""Global optimization for radial-velocity fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, OptimizeResult, differential_evolution, least_squares

from exoplanet_est.data import RadialVelocityDataset
from exoplanet_est.keplerian import (
    OrbitalParameters,
    estimate_planet_mass,
    radial_velocity_curve,
    semi_major_axis_from_period,
)


@dataclass(frozen=True)
class OptimizationTracePoint:
    """Best-so-far state after one optimizer step."""

    generation: int
    stage: str
    chi2: float
    residual_rms_ms: float
    parameters: OrbitalParameters


@dataclass(frozen=True)
class OptimizationHistory:
    """Convergence trace for differential evolution plus local polish."""

    points: tuple[OptimizationTracePoint, ...]

    @property
    def generations(self) -> np.ndarray:
        return np.asarray([point.generation for point in self.points], dtype=int)

    @property
    def chi2(self) -> np.ndarray:
        return np.asarray([point.chi2 for point in self.points], dtype=float)

    @property
    def residual_rms_ms(self) -> np.ndarray:
        return np.asarray([point.residual_rms_ms for point in self.points], dtype=float)


@dataclass(frozen=True)
class RVFitResult:
    """Final fit outputs and derived quantities."""

    parameters: OrbitalParameters
    model_velocity_ms: np.ndarray
    residual_rms_ms: float
    weighted_sse: float
    star_mass_kg: float
    planet_mass_kg: float
    semi_major_axis_m: float
    success: bool
    optimizer_message: str
    history: OptimizationHistory | None = None


def default_bounds(dataset: RadialVelocityDataset) -> Bounds:
    """Construct broad but stable bounds from the observation window."""

    time_span = float(dataset.times_days.max() - dataset.times_days.min())
    velocity_span = float(
        dataset.radial_velocity_ms.max() - dataset.radial_velocity_ms.min()
    )
    gamma_guess = float(np.median(dataset.radial_velocity_ms))

    min_period = max(time_span / 8.0, 5.0)
    max_period = max(time_span * 1.5, min_period * 2.0)
    max_amplitude = max(abs(velocity_span), 20.0) * 1.5

    lower = np.array(
        [
            1.0,
            min_period,
            0.0,
            -np.pi,
            dataset.times_days.min() - max_period,
            gamma_guess - max_amplitude,
        ],
        dtype=float,
    )
    upper = np.array(
        [
            max_amplitude,
            max_period,
            0.85,
            np.pi,
            dataset.times_days.min() + max_period,
            gamma_guess + max_amplitude,
        ],
        dtype=float,
    )
    return Bounds(lower, upper)


def _vector_to_parameters(vector: np.ndarray) -> OrbitalParameters:
    return OrbitalParameters(
        semi_amplitude_ms=float(vector[0]),
        period_days=float(vector[1]),
        eccentricity=float(vector[2]),
        omega_rad=float(vector[3]),
        t_periastron_days=float(vector[4]),
        gamma_ms=float(vector[5]),
    )


def _weighted_residuals(
    vector: np.ndarray,
    dataset: RadialVelocityDataset,
) -> np.ndarray:
    parameters = _vector_to_parameters(vector)
    model = radial_velocity_curve(parameters, dataset.times_days)
    return (dataset.radial_velocity_ms - model) / dataset.uncertainty_ms


def _metrics_for_vector(
    vector: np.ndarray,
    dataset: RadialVelocityDataset,
) -> tuple[OrbitalParameters, float, float]:
    parameters = _vector_to_parameters(vector)
    model = radial_velocity_curve(parameters, dataset.times_days)
    residuals = dataset.radial_velocity_ms - model
    chi2 = float(np.sum((residuals / dataset.uncertainty_ms) ** 2))
    rms = float(np.sqrt(np.mean(residuals**2)))
    return parameters, chi2, rms


def export_optimization_history_csv(
    history: OptimizationHistory,
    path: Path,
    *,
    truth: OrbitalParameters | None = None,
) -> Path:
    """Write a convergence trace CSV for plotting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "generation,stage,stage_code,chi2,rms_ms,K_ms,P_days,e,"
        "rel_err_K,rel_err_P"
    )
    rows: list[str] = [header]
    for point in history.points:
        parameters = point.parameters
        stage_code = "0" if point.stage == "de" else "1"
        if truth is None:
            rel_err_k = "nan"
            rel_err_p = "nan"
        else:
            rel_err_k = (
                abs(parameters.semi_amplitude_ms - truth.semi_amplitude_ms)
                / abs(truth.semi_amplitude_ms)
            )
            rel_err_p = abs(parameters.period_days - truth.period_days) / abs(
                truth.period_days
            )
            rel_err_k = f"{rel_err_k:.10e}"
            rel_err_p = f"{rel_err_p:.10e}"
        rows.append(
            ",".join(
                [
                    str(point.generation),
                    point.stage,
                    stage_code,
                    f"{point.chi2:.10e}",
                    f"{point.residual_rms_ms:.10e}",
                    f"{parameters.semi_amplitude_ms:.10e}",
                    f"{parameters.period_days:.10e}",
                    f"{parameters.eccentricity:.10e}",
                    rel_err_k,
                    rel_err_p,
                ]
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def fit_radial_velocity_curve(
    dataset: RadialVelocityDataset,
    *,
    star_mass_kg: float,
    bounds: Bounds | None = None,
    seed: int = 0,
    max_iterations: int = 300,
    record_history: bool = True,
) -> RVFitResult:
    """Fit a single-planet Keplerian RV model using DE + local polish."""

    bounds = bounds or default_bounds(dataset)
    history_points: list[OptimizationTracePoint] = []

    def objective(vector: np.ndarray) -> float:
        return float(np.sum(_weighted_residuals(vector, dataset) ** 2))

    def de_callback(intermediate_result: OptimizeResult) -> None:
        if not record_history:
            return
        parameters, chi2, rms = _metrics_for_vector(intermediate_result.x, dataset)
        history_points.append(
            OptimizationTracePoint(
                generation=len(history_points) + 1,
                stage="de",
                chi2=chi2,
                residual_rms_ms=rms,
                parameters=parameters,
            )
        )

    global_result = differential_evolution(
        objective,
        bounds=list(zip(bounds.lb, bounds.ub, strict=True)),
        seed=seed,
        maxiter=max_iterations,
        polish=False,
        updating="deferred",
        callback=de_callback if record_history else None,
    )

    local_result = least_squares(
        lambda vector: _weighted_residuals(vector, dataset),
        x0=global_result.x,
        bounds=(bounds.lb, bounds.ub),
        max_nfev=20_000,
    )

    parameters, weighted_sse, residual_rms = _metrics_for_vector(
        local_result.x, dataset
    )
    model_velocity = radial_velocity_curve(parameters, dataset.times_days)

    if record_history:
        history_points.append(
            OptimizationTracePoint(
                generation=len(history_points) + 1,
                stage="polish",
                chi2=weighted_sse,
                residual_rms_ms=residual_rms,
                parameters=parameters,
            )
        )

    planet_mass_kg = estimate_planet_mass(
        star_mass_kg,
        parameters.semi_amplitude_ms,
        parameters.period_days,
        parameters.eccentricity,
    )
    semi_major_axis_m = semi_major_axis_from_period(
        parameters.period_days,
        star_mass_kg + planet_mass_kg,
    )

    history = (
        OptimizationHistory(points=tuple(history_points)) if record_history else None
    )

    return RVFitResult(
        parameters=parameters,
        model_velocity_ms=model_velocity,
        residual_rms_ms=residual_rms,
        weighted_sse=weighted_sse,
        star_mass_kg=star_mass_kg,
        planet_mass_kg=planet_mass_kg,
        semi_major_axis_m=semi_major_axis_m,
        success=bool(global_result.success and local_result.success),
        optimizer_message=f"{global_result.message}; {local_result.message}",
        history=history,
    )


def evaluate_on_dense_grid(
    parameters: OrbitalParameters,
    *,
    start_day: float,
    stop_day: float,
    samples: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the model on a dense, sorted time grid."""

    times = np.linspace(start_day, stop_day, samples)
    return times, radial_velocity_curve(parameters, times)
