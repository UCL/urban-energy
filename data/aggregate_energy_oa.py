"""
Aggregate postcode-level DESNZ energy data to Output Area (OA) level.

Uses the postcode-to-OA lookup (from build_postcode_oa_lookup.py) to aggregate
metered gas and electricity consumption from postcode level to OA level via
meter-weighted means.

Inputs:
    - temp/statistics/postcode_energy_consumption.parquet
    - temp/statistics/postcode_oa_lookup.parquet

Output:
    - temp/statistics/oa_energy_consumption.parquet
        Columns: OA21CD, LSOA21CD, oa_elec_mean_kwh, oa_gas_mean_kwh,
                 oa_total_mean_kwh, oa_gas_share, oa_num_meters,
                 oa_num_postcodes
"""

import pandas as pd

from urban_energy.paths import DATA_DIR

OUTPUT_DIR = DATA_DIR / "statistics"

POSTCODE_ENERGY_PATH = DATA_DIR / "statistics" / "postcode_energy_consumption.parquet"
POSTCODE_OA_LOOKUP_PATH = DATA_DIR / "statistics" / "postcode_oa_lookup.parquet"

# Minimum total meters per OA for a stable estimate
MIN_METERS_PER_OA = 5


def normalise_postcode(s: pd.Series) -> pd.Series:
    """
    Normalise postcode format for matching.

    Strips whitespace, converts to uppercase. Does NOT remove the internal
    space — both DESNZ and Code-Point use the standard "AB10 1AU" format.

    Parameters
    ----------
    s : pd.Series
        Raw postcode strings.

    Returns
    -------
    pd.Series
        Normalised postcode strings.
    """
    return s.str.strip().str.upper()


def aggregate_postcode_to_oa(
    energy: pd.DataFrame,
    lookup: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate postcode energy to OA level using meter-weighted means.

    Parameters
    ----------
    energy : pd.DataFrame
        Postcode-level energy with columns: Postcode, elec_num_meters,
        elec_mean_kwh, gas_num_meters, gas_mean_kwh, total_mean_kwh.
    lookup : pd.DataFrame
        Postcode-to-OA mapping with columns: Postcode, OA21CD, LSOA21CD.

    Returns
    -------
    pd.DataFrame
        OA-level energy consumption.
    """
    # Normalise postcodes in both datasets
    energy = energy.copy()
    lookup = lookup.copy()
    energy["Postcode"] = normalise_postcode(energy["Postcode"])
    lookup["Postcode"] = normalise_postcode(lookup["Postcode"])

    # Join energy to OA lookup
    lookup_cols = ["Postcode", "OA21CD", "LSOA21CD"]
    merged = energy.merge(lookup[lookup_cols], on="Postcode", how="inner")
    n_matched = len(merged)
    n_total = len(energy)
    pct = n_matched / n_total
    print(f"  Matched {n_matched:,}/{n_total:,} postcodes ({pct:.1%})")

    # Compute meter counts per postcode (use max of gas/elec meters)
    elec_meters = pd.to_numeric(
        merged.get("elec_num_meters", 0), errors="coerce"
    ).fillna(0)
    gas_meters = pd.to_numeric(merged.get("gas_num_meters", 0), errors="coerce").fillna(
        0
    )
    merged["_meters"] = elec_meters.clip(lower=0) + gas_meters.clip(lower=0)
    # If both are zero, use 1 to avoid division by zero
    merged["_meters"] = merged["_meters"].replace(0, 1)

    # Weighted mean: sum(value * weight) / sum(weight) per OA
    results: dict[str, pd.Series] = {}

    for metric, weight_col in [
        ("elec_mean_kwh", "elec_num_meters"),
        ("gas_mean_kwh", "gas_num_meters"),
        ("total_mean_kwh", "_meters"),
    ]:
        if metric not in merged.columns:
            continue
        val = pd.to_numeric(merged[metric], errors="coerce")
        raw_wt = merged.get(weight_col, merged["_meters"])
        wt = pd.to_numeric(raw_wt, errors="coerce").fillna(1)
        wt = wt.clip(lower=0).replace(0, 1)

        weighted_sum = (val * wt).groupby(merged["OA21CD"]).sum()
        weight_total = wt.groupby(merged["OA21CD"]).sum()
        oa_name = f"oa_{metric}" if not metric.startswith("oa_") else metric
        results[oa_name] = weighted_sum / weight_total

    # Count metrics
    results["oa_num_meters"] = merged.groupby("OA21CD")["_meters"].sum()
    results["oa_num_postcodes"] = merged.groupby("OA21CD")["Postcode"].count()

    # LSOA21CD lookup (first per OA — all postcodes in same OA map to same LSOA)
    oa_lsoa = merged.groupby("OA21CD")["LSOA21CD"].first()

    # Assemble
    oa_energy = pd.DataFrame(results)
    oa_energy["LSOA21CD"] = oa_lsoa
    oa_energy = oa_energy.reset_index()

    # Gas share
    has_gas = "oa_gas_mean_kwh" in oa_energy.columns
    has_total = "oa_total_mean_kwh" in oa_energy.columns
    if has_gas and has_total:
        oa_energy["oa_gas_share"] = oa_energy["oa_gas_mean_kwh"] / oa_energy[
            "oa_total_mean_kwh"
        ].replace(0, pd.NA)

    # Filter: minimum meters
    n_before = len(oa_energy)
    oa_energy = oa_energy[oa_energy["oa_num_meters"] >= MIN_METERS_PER_OA].copy()
    n_filtered = n_before - len(oa_energy)
    if n_filtered > 0:
        print(f"  Filtered {n_filtered:,} OAs with < {MIN_METERS_PER_OA} meters")

    return oa_energy


def main() -> None:
    """Aggregate postcode energy to OA level."""
    print("=" * 60)
    print("Aggregate Postcode Energy → Output Area (OA)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load postcode energy
    print("\n[1/3] Loading postcode energy data...")
    if not POSTCODE_ENERGY_PATH.exists():
        msg = (
            f"Postcode energy not found at {POSTCODE_ENERGY_PATH}. "
            "Run: uv run python data/download_energy_postcode.py"
        )
        raise FileNotFoundError(msg)
    energy = pd.read_parquet(POSTCODE_ENERGY_PATH)
    print(f"  Loaded {len(energy):,} postcodes")

    # 2. Load lookup
    print("\n[2/3] Loading postcode → OA lookup...")
    if not POSTCODE_OA_LOOKUP_PATH.exists():
        msg = (
            f"Postcode-OA lookup not found at {POSTCODE_OA_LOOKUP_PATH}. "
            "Run: uv run python data/build_postcode_oa_lookup.py"
        )
        raise FileNotFoundError(msg)
    lookup = pd.read_parquet(POSTCODE_OA_LOOKUP_PATH)
    print(f"  Loaded {len(lookup):,} postcode-OA mappings")

    # 3. Aggregate
    print("\n[3/3] Aggregating to OA level...")
    oa_energy = aggregate_postcode_to_oa(energy, lookup)

    # Save
    output_path = OUTPUT_DIR / "oa_energy_consumption.parquet"
    oa_energy.to_parquet(output_path, index=False)
    print(f"\n  Saved {len(oa_energy):,} OAs to {output_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary:")
    for col in oa_energy.columns:
        if col in ("OA21CD", "LSOA21CD"):
            continue
        if oa_energy[col].dtype in ("float64", "int64", "Float64"):
            valid = oa_energy[col].notna().sum()
            mean_val = oa_energy[col].mean()
            if "share" in col:
                print(f"  {col}: {valid:,} valid, mean = {mean_val:.3f}")
            else:
                print(f"  {col}: {valid:,} valid, mean = {mean_val:,.1f}")
    n_lsoas = oa_energy["LSOA21CD"].nunique()
    print(f"\n  Covers {len(oa_energy):,} OAs across {n_lsoas:,} LSOAs")
    print("=" * 60)


if __name__ == "__main__":
    main()
