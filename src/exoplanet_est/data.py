"""Data loading and synthetic radial-velocity generation."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np

from exoplanet_est.constants import DAY_IN_SECONDS, MASS_SUN
from exoplanet_est.doppler import doppler_shift_to_velocity, velocity_to_doppler_shift
from exoplanet_est.keplerian import OrbitalParameters, radial_velocity_curve

ARCHIVED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "archived_observations"


@dataclass(frozen=True)
class RadialVelocityDataset:
    """Observed or synthetic radial-velocity time series."""

    times_days: np.ndarray
    radial_velocity_ms: np.ndarray
    uncertainty_ms: np.ndarray
    label: str = "dataset"


def load_radial_velocity_csv(
    path: str | Path,
    *,
    value_kind: str = "velocity",
    time_column: str = "time_days",
    value_column: str | None = None,
    uncertainty_column: str = "uncertainty_ms",
    time_unit: str = "days",
) -> RadialVelocityDataset:
    """Load a radial-velocity dataset from a CSV file.

    Supported CSV shapes:
    - named columns such as `time_days`, `radial_velocity_ms`, `uncertainty_ms`
    - unnamed numeric columns ordered as time, value, optional uncertainty
    """

    path = Path(path)
    rows: list[list[str]] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        rows = [row for row in reader if row]

    if not rows:
        raise ValueError(f"CSV file is empty: {path}")

    header = rows[0]
    try:
        [float(value) for value in header[:2]]
        has_header = False
    except ValueError:
        has_header = True

    if has_header:
        with path.open(newline="") as handle:
            records = list(csv.DictReader(handle))
        if not records:
            raise ValueError(f"CSV file contains a header but no data rows: {path}")

        value_column = value_column or (
            "radial_velocity_ms" if value_kind == "velocity" else "doppler_shift"
        )
        times = np.array([float(row[time_column]) for row in records], dtype=float)
        values = np.array([float(row[value_column]) for row in records], dtype=float)
        uncertainties = np.array(
            [float(row.get(uncertainty_column) or 1.0) for row in records],
            dtype=float,
        )
    else:
        array = np.array(rows, dtype=float)
        if array.shape[1] < 2:
            raise ValueError("CSV file must contain at least two columns: time and value.")
        times = array[:, 0]
        values = array[:, 1]
        uncertainties = (
            array[:, 2] if array.shape[1] >= 3 else np.full_like(times, 1.0)
        )

    if time_unit == "days":
        times_days = times
    elif time_unit == "seconds":
        times_days = times / DAY_IN_SECONDS
    else:
        raise ValueError("time_unit must be either 'days' or 'seconds'.")

    radial_velocity_ms = (
        values if value_kind == "velocity" else doppler_shift_to_velocity(values)
    )
    return RadialVelocityDataset(
        times_days=times_days,
        radial_velocity_ms=radial_velocity_ms,
        uncertainty_ms=uncertainties,
        label=path.stem,
    )


def load_archived_star_masses(path: str | Path | None = None) -> np.ndarray:
    """Load the archived stellar mass catalog in solar masses."""

    path = Path(path) if path is not None else ARCHIVED_DATA_DIR / "star-masses.csv"
    rows = np.genfromtxt(path, delimiter=",", skip_header=1)
    if rows.ndim == 1:
        rows = rows[None, :]
    return rows[:, 1]


def load_archived_star_dataset(
    star_index: int,
    *,
    data_dir: str | Path | None = None,
    uncertainty_ms: float = 1.0,
) -> tuple[RadialVelocityDataset, float]:
    """Load one archived star dataset and return its stellar mass."""

    data_dir = Path(data_dir) if data_dir is not None else ARCHIVED_DATA_DIR
    dataset = load_radial_velocity_csv(
        data_dir / f"star{star_index}.csv",
        value_kind="doppler_shift",
        time_column="Seconds",
        value_column="Doppler Shift",
        time_unit="seconds",
    )
    dataset = RadialVelocityDataset(
        times_days=dataset.times_days,
        radial_velocity_ms=dataset.radial_velocity_ms,
        uncertainty_ms=np.full_like(dataset.times_days, uncertainty_ms, dtype=float),
        label=f"archived-star{star_index}",
    )
    star_mass_solar = load_archived_star_masses(data_dir / "star-masses.csv")[star_index]
    return dataset, float(star_mass_solar * MASS_SUN)


def save_synthetic_dataset_csv(
    dataset: RadialVelocityDataset,
    path: str | Path,
    *,
    include_doppler_shift: bool = True,
) -> None:
    """Persist a dataset to CSV so fits can be reproduced."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["time_days", "radial_velocity_ms", "uncertainty_ms"]
        if include_doppler_shift:
            header.append("doppler_shift")
        writer.writerow(header)

        doppler_shift = velocity_to_doppler_shift(dataset.radial_velocity_ms)
        for index, time_day in enumerate(dataset.times_days):
            row = [
                float(time_day),
                float(dataset.radial_velocity_ms[index]),
                float(dataset.uncertainty_ms[index]),
            ]
            if include_doppler_shift:
                row.append(float(doppler_shift[index]))
            writer.writerow(row)


def generate_synthetic_dataset(
    parameters: OrbitalParameters,
    *,
    times_days: np.ndarray,
    noise_std_ms: float,
    jitter_std_ms: float = 0.0,
    seed: int = 0,
    label: str = "synthetic-rv",
) -> RadialVelocityDataset:
    """Generate a noisy synthetic radial-velocity dataset."""

    rng = np.random.default_rng(seed)
    noiseless_curve = radial_velocity_curve(parameters, times_days)
    noise = rng.normal(loc=0.0, scale=noise_std_ms, size=times_days.shape)
    if jitter_std_ms > 0.0:
        noise += rng.normal(loc=0.0, scale=jitter_std_ms, size=times_days.shape)

    return RadialVelocityDataset(
        times_days=np.asarray(times_days, dtype=float),
        radial_velocity_ms=noiseless_curve + noise,
        uncertainty_ms=np.full_like(times_days, noise_std_ms, dtype=float),
        label=label,
    )
