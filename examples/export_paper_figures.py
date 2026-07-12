"""Export print-friendly vector figures and CSV tables for the LaTeX paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exoplanet_est.constants import MASS_JUPITER, MASS_SUN
from exoplanet_est.data import (
    generate_synthetic_dataset,
    list_public_targets,
    load_public_target_dataset,
    save_synthetic_dataset_csv,
)
from exoplanet_est.keplerian import OrbitalParameters, semi_amplitude_from_masses
from exoplanet_est.optimize import (
    export_optimization_history_csv,
    fit_radial_velocity_curve,
    periodogram_bounds,
)
from exoplanet_est.plot import (
    ShowcaseTruth,
    export_plot_series,
    plot_fit_summary,
    save_figure,
)

FIGURES_DIR = Path("docs/figures")
DATA_DIR = FIGURES_DIR / "plotdata"

TARGET_SEEDS = {
    "51_peg": 0,
    "hd_209458": 1,
    "70_vir": 2,
    "hd_3651": 3,
}


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


def export_public(target_key: str) -> dict:
    dataset, star_mass_kg, target = load_public_target_dataset(target_key)
    bounds = periodogram_bounds(
        dataset,
        eccentricity_upper=target.eccentricity_upper,
    )
    fit_result = fit_radial_velocity_curve(
        dataset,
        star_mass_kg=star_mass_kg,
        bounds=bounds,
        max_iterations=400,
        seed=TARGET_SEEDS.get(target_key, 0),
    )
    stem = f"{target_key}_fit"
    export_one(
        stem=stem,
        dataset=dataset,
        fit_result=fit_result,
        title=f"{target.host_name} Radial Velocity Fit",
    )
    params = fit_result.parameters
    summary = {
        "key": target.key,
        "host_name": target.host_name,
        "planet_name": target.planet_name,
        "reference": target.reference,
        "bibcode": target.bibcode,
        "archive_url": target.archive_url,
        "n_points": int(len(dataset.times_days)),
        "star_mass_solar": target.star_mass_solar,
        "period_days": params.period_days,
        "semi_amplitude_ms": params.semi_amplitude_ms,
        "eccentricity": params.eccentricity,
        "omega_rad": params.omega_rad,
        "t_periastron_days": params.t_periastron_days,
        "gamma_ms": params.gamma_ms,
        "semi_major_axis_au": fit_result.semi_major_axis_m / 1.495978707e11,
        "planet_mass_mjup": fit_result.planet_mass_kg / MASS_JUPITER,
        "residual_rms_ms": fit_result.residual_rms_ms,
        "chi2": fit_result.weighted_sse,
        "success": fit_result.success,
        "K_bound": [float(bounds.lb[0]), float(bounds.ub[0])],
        "P_bound": [float(bounds.lb[1]), float(bounds.ub[1])],
        "e_bound": [float(bounds.lb[2]), float(bounds.ub[2])],
        "t0_bound": [float(bounds.lb[4]), float(bounds.ub[4])],
        "gamma_bound": [float(bounds.lb[5]), float(bounds.ub[5])],
    }
    print(
        f"{target.key}: P={params.period_days:.3f} d  "
        f"K={params.semi_amplitude_ms:.2f} m/s  "
        f"e={params.eccentricity:.3f}  "
        f"Mp={summary['planet_mass_mjup']:.3f} Mjup  "
        f"RMS={fit_result.residual_rms_ms:.2f} m/s"
    )
    return summary


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    export_synthetic()
    summaries = [export_public(target.key) for target in list_public_targets()]
    summary_path = DATA_DIR / "public_fit_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
