# NASA Exoplanet Archive radial-velocity series

Public literature radial-velocity (RVC) time series downloaded from the
[NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) and used
for observational validation in this project.

## Included targets

| Key | Host | Literature source | Archive file |
| --- | --- | --- | --- |
| `51_peg` | 51 Peg | Marcy et al. 1997 | `UID_0113357_RVC_002.tbl` |
| `hd_209458` | HD 209458 | Butler et al. 2006 | `UID_0108859_RVC_001.tbl` |
| `70_vir` | 70 Vir | Butler et al. 2006 | `UID_0065721_RVC_001.tbl` |
| `hd_3651` | HD 3651 | Butler et al. 2006 | `UID_0003093_RVC_002.tbl` |

Provenance for each series (URL, bibcode, instrument, stellar mass) is recorded
in `catalog.json`. Original IPAC table downloads for the selected series are kept
under `raw/tbl/` for auditability:

- `51_peg.tbl` → `51_peg.csv`
- `hd_209458_a.tbl` → `hd_209458.csv`
- `70_vir_a.tbl` → `70_vir.csv`
- `hd_3651_b.tbl` → `hd_3651.csv`

## CSV columns

- `time_jd`: observation time in Julian days (as published)
- `radial_velocity_ms`: barycentric radial velocity in m/s
- `uncertainty_ms`: published per-point uncertainty in m/s

## Acknowledgment

This research has made use of the NASA Exoplanet Archive, which is operated by
the California Institute of Technology, under contract with the National
Aeronautics and Space Administration under the Exoplanet Exploration Program.

When using a specific series, also cite the literature reference listed in
`catalog.json`.
