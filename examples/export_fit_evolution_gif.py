"""Export a GIF of optimizer evolution for an archived star fit."""

from __future__ import annotations

import argparse
from pathlib import Path

from scipy.optimize import Bounds

from exoplanet_est.data import load_archived_star_dataset
from exoplanet_est.optimize import fit_radial_velocity_curve
from exoplanet_est.plot import render_fit_evolution_gif


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--star-index", type=int, default=3, choices=(0, 1, 2, 3))
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("docs/archived_star3_fit_evolution.gif"),
        help="Output GIF path.",
    )
    parser.add_argument("--max-iterations", type=int, default=400)
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--frame-duration-ms", type=int, default=110)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, star_mass_kg = load_archived_star_dataset(args.star_index)
    fit_result = fit_radial_velocity_curve(
        dataset,
        star_mass_kg=star_mass_kg,
        bounds=bounds_for_dataset(dataset, args.star_index),
        max_iterations=args.max_iterations,
        seed=args.star_index,
        record_history=True,
    )
    if fit_result.history is None:
        raise RuntimeError("Optimizer history was not recorded.")

    output_path = render_fit_evolution_gif(
        dataset,
        fit_result.history,
        star_mass_kg=star_mass_kg,
        output_path=args.save,
        title=f"Archived Star {args.star_index} Fit Evolution",
        theme="dark",
        max_frames=args.max_frames,
        frame_duration_ms=args.frame_duration_ms,
    )
    print(f"History steps: {len(fit_result.history.points)}")
    print(f"GIF:           {output_path}")
    print(f"Final RMS:     {fit_result.residual_rms_ms:.3f} m/s")


if __name__ == "__main__":
    main()
