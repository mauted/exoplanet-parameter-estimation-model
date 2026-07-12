"""Data loading and synthetic radial-velocity generation."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path

import numpy as np

from exoplanet_est.constants import DAY_IN_SECONDS, MASS_SUN
from exoplanet_est.doppler import doppler_shift_to_velocity, velocity_to_doppler_shift
from exoplanet_est.keplerian import OrbitalParameters, radial_velocity_curve

PUBLIC_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "nasa_exoplanet_archive"
# Backward-compatible alias used by earlier package exports.
ARCHIVED_DATA_DIR = PUBLIC_DATA_DIR


@dataclass(frozen=True)
class RadialVelocityDataset:
    """Observed or synthetic radial-velocity time series."""

    times_days: np.ndarray
    radial_velocity_ms: np.ndarray
    uncertainty_ms: np.ndarray
    label: str = "dataset"


@dataclass(frozen=True)
class PublicTarget:
    """Metadata for one NASA Exoplanet Archive RVC series."""

    key: str
    host_name: str
    planet_name: str
    star_mass_solar: float
    csv_file: str
    archive_url: str
    overview_url: str
    eccentricity_upper: float
    reference: str
    bibcode: str
    n_points: int
    notes: str = ""


def load_public_catalog(path: str | Path | None = None) -> dict:
    """Load the packaged NASA Exoplanet Archive catalog metadata."""

    path = Path(path) if path is not None else PUBLIC_DATA_DIR / "catalog.json"
    with path.open() as handle:
        return json.load(handle)


def list_public_targets(path: str | Path | None = None) -> tuple[PublicTarget, ...]:
    """Return the ordered public observational targets."""

    catalog = load_public_catalog(path)
    return tuple(
        PublicTarget(
            key=entry["key"],
            host_name=entry["host_name"],
            planet_name=entry["planet_name"],
            star_mass_solar=float(entry["star_mass_solar"]),
            csv_file=entry["csv_file"],
            archive_url=entry["archive_url"],
            overview_url=entry["overview_url"],
            eccentricity_upper=float(entry.get("eccentricity_upper", 0.85)),
            reference=entry["reference"],
            bibcode=entry["bibcode"],
            n_points=int(entry["n_points"]),
            notes=entry.get("notes", ""),
        )
        for entry in catalog["targets"]
    )


def get_public_target(key: str, path: str | Path | None = None) -> PublicTarget:
    """Look up one public target by catalog key."""

    for target in list_public_targets(path):
        if target.key == key:
            return target
    known = ", ".join(target.key for target in list_public_targets(path))
    raise KeyError(f"Unknown public target {key!r}. Known keys: {known}")


def load_radial_velocity_csv(
    path: str | Path,
    *,
    value_kind: str = "velocity",
    time_column: str = "time_days",
    value_column: str | None = None,
    uncertainty_column: str = "uncertainty_ms",
    time_unit: str = "days",
    label: str | None = None,
) -> RadialVelocityDataset:
    """Load a radial-velocity dataset from a CSV file.

    Supported CSV shapes:
    - named columns such as `time_days`, `radial_velocity_ms`, `uncertainty_ms`
    - NASA-style `time_jd`, `radial_velocity_ms`, `uncertainty_ms`
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

        header_names = {name.lower(): name for name in records[0]}
        if time_column not in records[0] and "time_jd" in header_names:
            time_column = header_names["time_jd"]
            time_unit = "days"
        if value_column is None:
            if value_kind == "velocity":
                value_column = header_names.get(
                    "radial_velocity_ms", "radial_velocity_ms"
                )
            else:
                value_column = header_names.get("doppler_shift", "doppler_shift")
        if uncertainty_column not in records[0]:
            uncertainty_column = header_names.get("uncertainty_ms", uncertainty_column)

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
        label=label or path.stem,
    )


def load_public_target_dataset(
    key: str,
    *,
    data_dir: str | Path | None = None,
    relative_times: bool = True,
) -> tuple[RadialVelocityDataset, float, PublicTarget]:
    """Load one public NASA Exoplanet Archive RVC series and its stellar mass."""

    data_dir = Path(data_dir) if data_dir is not None else PUBLIC_DATA_DIR
    target = get_public_target(key, data_dir / "catalog.json")
    dataset = load_radial_velocity_csv(
        data_dir / target.csv_file,
        value_kind="velocity",
        time_column="time_jd",
        value_column="radial_velocity_ms",
        uncertainty_column="uncertainty_ms",
        time_unit="days",
        label=target.key,
    )
    times = np.asarray(dataset.times_days, dtype=float)
    if relative_times:
        times = times - times.min()
    dataset = RadialVelocityDataset(
        times_days=times,
        radial_velocity_ms=np.asarray(dataset.radial_velocity_ms, dtype=float),
        uncertainty_ms=np.asarray(dataset.uncertainty_ms, dtype=float),
        label=target.key,
    )
    return dataset, float(target.star_mass_solar * MASS_SUN), target


def load_archived_star_dataset(
    star_index: int,
    *,
    data_dir: str | Path | None = None,
    uncertainty_ms: float = 1.0,
) -> tuple[RadialVelocityDataset, float]:
    """Backward-compatible wrapper mapping index 0..3 onto public targets."""

    del uncertainty_ms  # public CSVs already include published uncertainties
    targets = list_public_targets(
        None if data_dir is None else Path(data_dir) / "catalog.json"
    )
    if star_index < 0 or star_index >= len(targets):
        raise IndexError(
            f"star_index must be in 0..{len(targets) - 1}; got {star_index}"
        )
    dataset, star_mass_kg, _target = load_public_target_dataset(
        targets[star_index].key,
        data_dir=data_dir,
    )
    return dataset, star_mass_kg


def load_archived_star_masses(path: str | Path | None = None) -> np.ndarray:
    """Backward-compatible stellar-mass vector for the public catalog order."""

    if path is not None:
        rows = np.genfromtxt(path, delimiter=",", skip_header=1)
        if rows.ndim == 1:
            rows = rows[None, :]
        return rows[:, 1]
    return np.asarray(
        [target.star_mass_solar for target in list_public_targets()],
        dtype=float,
    )


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
