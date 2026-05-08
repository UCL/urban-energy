"""
Per-year projection of fuel emission and price factors.

Loaded at runtime from the consolidated parquet built by
[data/build_projections.py](../../data/build_projections.py) from
hand-tabulated CSVs at [data/projections/](../../data/projections/).

Each CSV row carries an explicit source citation so any factor in the
Atlas can be traced to a specific NESO/DESNZ publication. See
[data/projections/README.md](../../data/projections/README.md) for the
update workflow.

The kWh values themselves are *not* projected — only the multipliers that
turn kWh into CO2 or £ change with year. This captures grid decarbonisation
and fuel-price evolution; it does not model demand-side electrification.

Public API (frozen across data-source changes):
    - PROJECTION_YEARS:        list[int]
    - fuel_factors(year, kind): dict[fuel_name → float]
    - SCENARIO:                 str  (currently 'central')

Future expansion (planned but not yet exposed):
    - per-scenario factors (low / central / high / counterfactual)
    - sensitivity envelopes on the dashboard
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pandas as pd

from urban_energy.paths import DATA_DIR

# Default scenario for the central case; the data file carries low/high too.
SCENARIO: str = "central"

PARQUET_PATH = DATA_DIR / "projections" / "projections.parquet"

# Repo-bundled CSV fallback so a fresh checkout works without running the
# build script. Production runs should rebuild the parquet whenever CSVs
# change so all consumers see consistent data.
_CSV_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "projections"
_CSV_FALLBACKS = (
    (_CSV_DIR / "carbon_factors.csv", "kgco2_per_kwh"),
    (_CSV_DIR / "fuel_prices.csv", "gbp_per_kwh"),
)


@cache
def _load() -> pd.DataFrame:
    """Load the consolidated long-form projections DataFrame."""
    if PARQUET_PATH.exists():
        return pd.read_parquet(PARQUET_PATH)

    # Fallback: read the source CSVs directly. Logged once at import time so
    # users notice they are running without the canonical build artefact.
    print(
        f"  [projections] {PARQUET_PATH} not found; "
        f"loading from bundled CSVs. "
        f"Run `uv run python data/build_projections.py` to rebuild."
    )
    parts = []
    for path, _metric in _CSV_FALLBACKS:
        if path.exists():
            parts.append(pd.read_csv(path))
    if not parts:
        raise FileNotFoundError(
            f"No projection data found at {PARQUET_PATH} or in {_CSV_DIR}"
        )
    return pd.concat(parts, ignore_index=True)


@cache
def projection_years() -> list[int]:
    """Distinct years present in the projection data, sorted."""
    df = _load()
    return sorted(int(y) for y in df["year"].unique())


@cache
def fuel_factors(year: int, kind: str, scenario: str = SCENARIO) -> dict[str, float]:
    """
    Return per-fuel factor table for `year` × `kind` × `scenario`.

    Parameters
    ----------
    year : int
        Must be present in `projection_years()`.
    kind : {"kgco2", "gbp"}
        Which metric to look up.
    scenario : str, default 'central'
        Which scenario column. Falls back to 'central' if requested
        scenario is not present for a given fuel/year (e.g. counterfactual
        only published for electricity).

    Returns
    -------
    dict
        Maps fuel name ('elec', 'gas', 'petrol', 'diesel') to factor (float).
    """
    metric = {"kgco2": "kgco2_per_kwh", "gbp": "gbp_per_kwh"}.get(kind)
    if metric is None:
        raise ValueError(f"Unknown factor kind: {kind!r}")

    df = _load()
    out: dict[str, float] = {}
    for fuel in ("elec", "gas", "petrol", "diesel"):
        sub = df[(df["year"] == year) & (df["metric"] == metric) & (df["fuel"] == fuel)]
        if len(sub) == 0:
            raise KeyError(f"No projection for year={year}, fuel={fuel}, metric={metric}")
        # Prefer requested scenario; fall back to 'central'
        match = sub[sub["scenario"] == scenario]
        if len(match) == 0:
            match = sub[sub["scenario"] == "central"]
        if len(match) == 0:
            match = sub.iloc[[0]]
        out[fuel] = float(match.iloc[0]["value"])
    return out


# Module-level constants used by some tests / consumers expecting the prior
# eager-import API. Computed at first access via the cached loader.
PROJECTION_YEARS: list[int] = projection_years()
