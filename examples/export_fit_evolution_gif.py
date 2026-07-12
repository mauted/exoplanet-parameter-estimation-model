"""Export a GIF of optimizer evolution for a public NASA RVC fit."""

from __future__ import annotations

import argparse
from pathlib import Path

from exoplanet_est.data import list_public_targets, load_public_target_dataset
from exoplanet_est.optimize import fit_radial_velocity_curve, periodogram_bounds
from exoplanet_est.plot import render_fit_evolution_gif

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
        default="51_peg",
        help="Public catalog target key.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("docs/51_peg_fit_evolution.gif"),
        help="Output GIF path.",
    )
    parser.add_argument("--max-iterations", type=int, default=400)
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--frame-duration-ms", type=int, default=110)
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
        record_history=True,
    )
    if fit_result.history is None:
        raise RuntimeError("Optimizer history was not recorded.")

    output_path = render_fit_evolution_gif(
        dataset,
        fit_result.history,
        star_mass_kg=star_mass_kg,
        output_path=args.save,
        title=f"{target.host_name} Fit Evolution",
        theme="dark",
        max_frames=args.max_frames,
        frame_duration_ms=args.frame_duration_ms,
        phase_fold=True,
    )
    print(f"History steps: {len(fit_result.history.points)}")
    print(f"GIF:           {output_path}")
    print(f"Final RMS:     {fit_result.residual_rms_ms:.3f} m/s")


if __name__ == "__main__":
    main()
