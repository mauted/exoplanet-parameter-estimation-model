# Polygence archive data

This directory vendors the recovered observation files from the older Polygence
version of the project.

Included files:

- `star0.csv` through `star3.csv`: Doppler-shift time series
- `star-masses.csv`: stellar masses in solar-mass units

The new package uses these files through `load_polygence_star_dataset(...)`,
which converts:

- time from seconds to days
- Doppler shift to radial velocity in meters per second

Example:

```bash
MPLBACKEND=Agg uv run python examples/fit_polygence_star.py --star-index 0 \
  --save outputs/polygence_star0_fit.png
```
