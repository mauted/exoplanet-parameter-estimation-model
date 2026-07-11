# Archived observation data

This directory contains four Doppler-shift time series and a stellar-mass
catalog used for observational validation.

Included files:

- `star0.csv` through `star3.csv`: Doppler-shift time series
- `star-masses.csv`: stellar masses in solar-mass units

## Provenance

These files were provided by an independent research mentor and are understood
to originate from a public NASA exoplanet/radial-velocity archive. The specific
catalog, survey program, or DOI was not retained with the working copies, so
the series are treated here as anonymized archival Doppler data rather than as
named discoveries.

## Usage

The package loads these files through `load_archived_star_dataset(...)`, which
converts:

- time from seconds to days
- Doppler shift to radial velocity in meters per second

Example:

```bash
MPLBACKEND=Agg uv run python examples/fit_archived_star.py --star-index 0 \
  --save outputs/archived_star0_fit.png
```
