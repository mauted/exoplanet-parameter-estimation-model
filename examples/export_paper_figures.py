"""Export print-friendly vector figures and CSV tables for the LaTeX paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exoplanet_est.constants import MASS_JUPITER, MASS_SUN
from exoplanet_est.data import (
    generate_synthetic_dataset,
    load_archived_star_dataset,
    save_synthetic_dataset_csv,
)
from exoplanet_est.keplerian import OrbitalParameters, semi_amplitude_from_masses
from exoplanet_est.optimize import (
    export_optimization_history_csv,
    fit_radial_velocity_curve,
)
from exoplanet_est.plot import (
    ShowcaseTruth,
    export_plot_series,
    plot_fit_summary,
    save_figure,
)
from scipy.optimize import Bounds

FIGURES_DIR = Path("docs/figures")
DATA_DIR = FIGURES_DIR / "plotdata"


def bounds_for_dataset(dataset, star_index: int) -> Bounds:
    time_span = float(dataset.times_days.max() - dataset.times_days.min())
    velocity_span = float(
        dataset.radial_velocity_ms.max() - dataset.radial_velocity_ms.min()
    )
    amplitude_upper = max(abs(velocity_span), 20.0) * 2.0
    period_lower = max(time_span / 40.0, 0.75)
    period_upper = max(time_span * 1.2, period_lower * 3.0)
    if star_index == 1:
        period_upper = max(period_upper, 6000.0)
    gamma_guess = float(dataset.radial_velocity_ms.mean())
    return Bounds(
        [
            1.0,
            period_lower,
            0.0,
            -3.14159,
            dataset.times_days.min() - period_upper,
            gamma_guess - amplitude_upper,
        ],
        [
            amplitude_upper,
            period_upper,
            0.85,
            3.14159,
            dataset.times_days.min() + period_upper,
            gamma_guess + amplitude_upper,
        ],
    )


def export_one(
    *,
    stem: str,
    dataset,
    fit_result,
    title: str,
    truth: ShowcaseTruth | None = None,
) -> None:
    fig, _, series = plot_fit_summary(
        dataset,
        fit_result,
        truth=truth,
        title=title,
        theme="print",
    )
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    save_figure(fig, pdf_path)
    export_plot_series(series, DATA_DIR / stem)
    plt.close(fig)
    print(f"wrote {pdf_path}")


def export_synthetic() -> None:
    star_mass_kg = 1.08 * MASS_SUN
    planet_mass_kg = 0.82 * MASS_JUPITER
    truth = OrbitalParameters(
        semi_amplitude_ms=semi_amplitude_from_masses(
            star_mass_kg,
            planet_mass_kg,
            period_days=147.5,
            eccentricity=0.19,
        ),
        period_days=147.5,
        eccentricity=0.19,
        omega_rad=1.05,
        t_periastron_days=18.0,
        gamma_ms=4.5,
    )
    rng = np.random.default_rng(12)
    observation_times = np.sort(rng.uniform(0.0, 420.0, size=92))
    dataset = generate_synthetic_dataset(
        truth,
        times_days=observation_times,
        noise_std_ms=2.2,
        jitter_std_ms=0.7,
        seed=12,
        label="synthetic-hd-demo",
    )
    save_synthetic_dataset_csv(dataset, Path("outputs/synthetic_rv_dataset.csv"))
    fit_result = fit_radial_velocity_curve(dataset, star_mass_kg=star_mass_kg, seed=12)
    if fit_result.history is not None:
        history_path = export_optimization_history_csv(
            fit_result.history,
            DATA_DIR / "rv_fit_preview_convergence.csv",
            truth=truth,
        )
        print(f"wrote {history_path} ({len(fit_result.history.points)} steps)")
    export_one(
        stem="rv_fit_preview",
        dataset=dataset,
        fit_result=fit_result,
        title="Exoplanet Parameter Estimation from Radial Velocity",
        truth=ShowcaseTruth(
            parameters=truth,
            planet_mass_kg=planet_mass_kg,
            star_mass_kg=star_mass_kg,
        ),
    )


def export_archived(star_index: int) -> None:
    dataset, star_mass_kg = load_archived_star_dataset(star_index)
    fit_result = fit_radial_velocity_curve(
        dataset,
        star_mass_kg=star_mass_kg,
        bounds=bounds_for_dataset(dataset, star_index),
        max_iterations=400,
        seed=star_index,
    )
    export_one(
        stem=f"archived_star{star_index}_fit",
        dataset=dataset,
        fit_result=fit_result,
        title=f"Archived Star {star_index} Radial Velocity Fit",
    )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    export_synthetic()
    for star_index in range(4):
        export_archived(star_index)


if __name__ == "__main__":
    main()
