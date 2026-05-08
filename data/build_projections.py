"""
Consolidate the projection input CSVs into a single parquet for runtime use.

Reads:
    data/projections/carbon_factors.csv
    data/projections/fuel_prices.csv

Writes:
    $DATA_DIR/projections/projections.parquet

Schema (long form):
    year      int      e.g. 2025
    scenario str       e.g. 'central', 'low', 'high', 'counterfactual'
    fuel     str       'elec' | 'gas' | 'petrol' | 'diesel'
    metric   str       'kgco2_per_kwh' | 'gbp_per_kwh'
    value    float
    source   str
    notes    str

Re-run after editing either CSV.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from urban_energy.paths import DATA_DIR

REPO_PROJECTIONS_DIR = Path(__file__).parent / "projections"
OUT_PATH = DATA_DIR / "projections" / "projections.parquet"

EXPECTED_SCENARIOS = {"central", "low", "high", "counterfactual"}
EXPECTED_FUELS = {"elec", "gas", "petrol", "diesel"}
EXPECTED_METRICS = {"kgco2_per_kwh", "gbp_per_kwh"}


def _load_csv(path: Path, expected_metric: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"year", "scenario", "fuel", "metric", "value", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    bad_metrics = set(df["metric"]) - {expected_metric}
    if bad_metrics:
        raise ValueError(
            f"{path.name}: unexpected metric values {bad_metrics} "
            f"(file should contain only {expected_metric})"
        )
    return df


def main() -> None:
    print(f"Reading CSVs from {REPO_PROJECTIONS_DIR}")
    carbon = _load_csv(REPO_PROJECTIONS_DIR / "carbon_factors.csv", "kgco2_per_kwh")
    prices = _load_csv(REPO_PROJECTIONS_DIR / "fuel_prices.csv", "gbp_per_kwh")
    print(f"  carbon_factors.csv: {len(carbon)} rows")
    print(f"  fuel_prices.csv:    {len(prices)} rows")

    df = pd.concat([carbon, prices], ignore_index=True)

    # ---- validation ----
    bad_scenarios = set(df["scenario"]) - EXPECTED_SCENARIOS
    if bad_scenarios:
        raise ValueError(f"Unexpected scenarios: {bad_scenarios}")
    bad_fuels = set(df["fuel"]) - EXPECTED_FUELS
    if bad_fuels:
        raise ValueError(f"Unexpected fuels: {bad_fuels}")

    df["year"] = df["year"].astype("int32")
    if "notes" not in df.columns:
        df["notes"] = ""
    df["notes"] = df["notes"].fillna("")

    # ---- summary ----
    print()
    print("Coverage by metric × scenario × year:")
    cov = df.groupby(["metric", "scenario", "year"]).size().unstack(fill_value=0)
    print(cov.to_string())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        OUT_PATH.unlink()
    df.to_parquet(OUT_PATH, index=False)
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"\n  Wrote {OUT_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
