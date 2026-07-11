"""Fit one of the archived radial-velocity datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.optimize import Bounds

from exoplanet_est.constants import MASS_JUPITER, MASS_SUN
from exoplanet_est.data import load_archived_star_dataset
from exoplanet_est.optimize import fit_radial_velocity_curve
from exoplanet_est.plot import plot_fit_summary, save_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--star-index",
        type=int,
        default=0,
        choices=(0, 1, 2, 3),
        help="Archived star index.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output path for the figure.",
    )
    parser.add_argument(
        "--uncertainty-ms",
        type=float,
        default=1.0,
        help="Fallback per-point uncertainty when the legacy CSV lacks one.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=400,
        help="Maximum differential evolution iterations.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the Matplotlib figure after saving it.",
    )
    return parser.parse_args()


def bounds_for_dataset(dataset, star_index: int) -> Bounds:
    """Choose broad period bounds tuned for the archived datasets."""

    time_span = float(dataset.times_days.max() - dataset.times_days.min())
    velocity_span = float(
        dataset.radial_velocity_ms.max() - dataset.radial_velocity_ms.min()
    )
    amplitude_upper = max(abs(velocity_span), 20.0) * 2.0
    period_lower = max(time_span / 40.0, 0.75)
    period_upper = max(time_span * 1.2, period_lower * 3.0)

    # The star-1 archive spans many years, so widen the search slightly.
    if star_index == 1:
        period_upper = max(period_upper, 6000.0)

    gamma_guess = float(dataset.radial_velocity_ms.mean())
    return Bounds(
        [1.0, period_lower, 0.0, -3.14159, dataset.times_days.min() - period_upper, gamma_guess - amplitude_upper],
        [amplitude_upper, period_upper, 0.85, 3.14159, dataset.times_days.min() + period_upper, gamma_guess + amplitude_upper],
    )


def main() -> None:
    args = parse_args()
    dataset, star_mass_kg = load_archived_star_dataset(
        args.star_index,
        uncertainty_ms=args.uncertainty_ms,
    )
    fit_result = fit_radial_velocity_curve(
        dataset,
        star_mass_kg=star_mass_kg,
        bounds=bounds_for_dataset(dataset, args.star_index),
        max_iterations=args.max_iterations,
        seed=args.star_index,
    )

    output_path = args.save or Path(f"outputs/archived_star{args.star_index}_fit.png")
    fig, _ = plot_fit_summary(
        dataset,
        fit_result,
        title=f"Archived Star {args.star_index} Radial Velocity Fit",
    )
    save_figure(fig, output_path)

    print(f"Dataset:      {dataset.label}")
    print(f"Star mass:    {star_mass_kg / MASS_SUN:.3f} Msun")
    print(f"Planet mass:  {fit_result.planet_mass_kg / MASS_JUPITER:.3f} Mjup")
    print(f"Period:       {fit_result.parameters.period_days:.3f} days")
    print(f"Semi-ampl.:   {fit_result.parameters.semi_amplitude_ms:.3f} m/s")
    print(f"Eccentricity: {fit_result.parameters.eccentricity:.3f}")
    print(f"Gamma:        {fit_result.parameters.gamma_ms:.3f} m/s")
    print(f"RMS:          {fit_result.residual_rms_ms:.3f} m/s")
    print(f"Success:      {fit_result.success}")
    print(f"Figure:       {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
