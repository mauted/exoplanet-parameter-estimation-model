"""Fit one of the public NASA Exoplanet Archive radial-velocity series."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from exoplanet_est.constants import MASS_JUPITER, MASS_SUN
from exoplanet_est.data import list_public_targets, load_public_target_dataset
from exoplanet_est.optimize import fit_radial_velocity_curve, periodogram_bounds
from exoplanet_est.plot import plot_fit_summary, save_figure

TARGET_SEEDS = {
    "51_peg": 0,
    "hd_209458": 1,
    "70_vir": 2,
    "hd_3651": 3,
}


def parse_args() -> argparse.Namespace:
    targets = list_public_targets()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=tuple(target.key for target in targets),
        default=targets[0].key,
        help="Public catalog target key.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output path for the figure.",
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
    parser.add_argument(
        "--theme",
        choices=("dark", "print"),
        default="dark",
        help="Figure theme: dark for web/showcase, print for light vector paper figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, star_mass_kg, target = load_public_target_dataset(args.target)
    fit_result = fit_radial_velocity_curve(
        dataset,
        star_mass_kg=star_mass_kg,
        bounds=periodogram_bounds(
            dataset,
            eccentricity_upper=target.eccentricity_upper,
        ),
        max_iterations=args.max_iterations,
        seed=TARGET_SEEDS.get(args.target, 0),
    )

    output_path = args.save or Path(f"outputs/{args.target}_fit.png")
    fig, _, _ = plot_fit_summary(
        dataset,
        fit_result,
        title=f"{target.host_name} Radial Velocity Fit",
        theme=args.theme,
        phase_fold=True,
    )
    save_figure(fig, output_path)

    print(f"Target:       {target.host_name} ({target.key})")
    print(f"Reference:    {target.reference}")
    print(f"Archive URL:  {target.archive_url}")
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
