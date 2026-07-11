"""Fit a synthetic exoplanet radial-velocity dataset and save a showcase figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exoplanet_est.constants import MASS_JUPITER, MASS_SUN
from exoplanet_est.data import generate_synthetic_dataset, save_synthetic_dataset_csv
from exoplanet_est.keplerian import OrbitalParameters, semi_amplitude_from_masses
from exoplanet_est.optimize import fit_radial_velocity_curve
from exoplanet_est.plot import ShowcaseTruth, plot_fit_summary, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("outputs/rv_fit_preview.png"),
        help="Path for the showcase plot.",
    )
    parser.add_argument(
        "--dataset-out",
        type=Path,
        default=Path("outputs/synthetic_rv_dataset.csv"),
        help="Path for the generated synthetic CSV.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the Matplotlib figure after saving it.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12,
        help="Random seed for observation cadence and noise.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    rng = np.random.default_rng(args.seed)
    observation_times = np.sort(rng.uniform(0.0, 420.0, size=92))
    dataset = generate_synthetic_dataset(
        truth,
        times_days=observation_times,
        noise_std_ms=2.2,
        jitter_std_ms=0.7,
        seed=args.seed,
        label="synthetic-hd-demo",
    )
    save_synthetic_dataset_csv(dataset, args.dataset_out)

    fit_result = fit_radial_velocity_curve(
        dataset,
        star_mass_kg=star_mass_kg,
        seed=args.seed,
    )

    fig, _ = plot_fit_summary(
        dataset,
        fit_result,
        truth=ShowcaseTruth(
            parameters=truth,
            planet_mass_kg=planet_mass_kg,
            star_mass_kg=star_mass_kg,
        ),
        title="Exoplanet Parameter Estimation from Radial Velocity",
    )
    save_figure(fig, args.save)

    print("Synthetic truth")
    print(f"  star mass:   {star_mass_kg / MASS_SUN:.3f} Msun")
    print(f"  planet mass: {planet_mass_kg / MASS_JUPITER:.3f} Mjup")
    print(f"  period:      {truth.period_days:.3f} days")
    print(f"  K:           {truth.semi_amplitude_ms:.3f} m/s")
    print(f"  eccentricity:{truth.eccentricity:.3f}")
    print("")
    print("Recovered fit")
    print(f"  planet mass: {fit_result.planet_mass_kg / MASS_JUPITER:.3f} Mjup")
    print(f"  period:      {fit_result.parameters.period_days:.3f} days")
    print(f"  K:           {fit_result.parameters.semi_amplitude_ms:.3f} m/s")
    print(f"  eccentricity:{fit_result.parameters.eccentricity:.3f}")
    print(f"  RMS:         {fit_result.residual_rms_ms:.3f} m/s")
    print(f"  figure:      {args.save}")
    print(f"  dataset:     {args.dataset_out}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
