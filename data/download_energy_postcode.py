"""
Download DESNZ postcode-level domestic gas and electricity consumption.

Downloads postcode-level metered energy consumption data from the Department
for Energy Security and Net Zero (DESNZ). This is the same underlying meter
data as the LSOA-level statistics but at full postcode granularity (~15-20
addresses per postcode on average).

Data source:
- Electricity: https://www.gov.uk/government/collections/sub-national-electricity-consumption-data
- Gas: https://www.gov.uk/government/collections/sub-national-gas-consumption-data
License: UK Open Government Licence (OGL)

Key features:
- Actual metered consumption (not SAP-modelled like EPC)
- Gas is weather-corrected; electricity is NOT
- Domestic meters only
- Postcodes with fewer than 5 meters are suppressed
- Postcodes where top-2 consumers exceed 90% of total are suppressed
- Meters consuming < 100 kWh/year are excluded

Output:
    - temp/statistics/postcode_energy_consumption.parquet
        Columns: Postcode, elec_num_meters, elec_total_kwh, elec_mean_kwh,
                 elec_median_kwh, gas_num_meters, gas_total_kwh,
                 gas_mean_kwh, gas_median_kwh, total_mean_kwh, gas_share
"""

from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from urban_energy.paths import CACHE_DIR as _CACHE_ROOT
from urban_energy.paths import DATA_DIR

OUTPUT_DIR = DATA_DIR / "statistics"
CACHE_DIR = _CACHE_ROOT / "desnz_postcode"

# DESNZ Postcode-Level Domestic Consumption URLs (2024 data, published 2025-12-18)
# Note: media IDs are not predictable — hardcoded per year.
TARGET_YEAR = 2024

ELEC_URL = (
    "https://assets.publishing.service.gov.uk/media/"
    "694282a1fdbd8404f9e1f1da/"
    "Postcode_level_all_meters_electricity_2024.csv"
)
GAS_URL = (
    "https://assets.publishing.service.gov.uk/media/"
    "6942a4e2501cdd438f4cf502/"
    "Postcode_level_gas_2024.csv"
)

ELEC_FILENAME = f"postcode_all_meters_electricity_{TARGET_YEAR}.csv"
GAS_FILENAME = f"postcode_gas_{TARGET_YEAR}.csv"


def download_file(url: str, dest: Path, timeout: int = 600) -> None:
    """
    Download a file with progress bar, streaming directly to disk.

    Parameters
    ----------
    url : str
        URL to download.
    dest : Path
        Destination file path.
    timeout : int
        Request timeout in seconds.
    """
    headers = {"User-Agent": "urban-energy-research/1.0"}
    response = requests.get(url, stream=True, timeout=timeout, headers=headers)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with (
        open(dest, "wb") as f,
        tqdm(total=total_size, unit="B", unit_scale=True, desc=dest.name) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)
            pbar.update(len(chunk))


def download_and_cache(url: str, filename: str) -> Path:
    """
    Download a file with caching. Returns path to cached file.

    Parameters
    ----------
    url : str
        URL to download.
    filename : str
        Cache filename.

    Returns
    -------
    Path
        Path to the cached file.
    """
    cache_path = CACHE_DIR / filename

    if cache_path.exists():
        print(f"  Loading cached {filename} ({cache_path.stat().st_size / 1e6:.1f} MB)")
        return cache_path

    print(f"  Downloading {filename}...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    download_file(url, cache_path)
    print(f"  Cached to {cache_path}")
    return cache_path


def parse_postcode_csv(path: Path, prefix: str) -> pd.DataFrame:
    """
    Parse a DESNZ postcode-level consumption CSV.

    Filters to individual postcodes only (removes outcode summary rows
    and unallocated entries). Filters to England postcodes only.

    Parameters
    ----------
    path : Path
        Path to the CSV file.
    prefix : str
        Column prefix, e.g. "elec" or "gas".

    Returns
    -------
    pd.DataFrame
        Consumption data indexed by Postcode.
    """
    print(f"  Reading {path.name}...")
    df = pd.read_csv(path, dtype={"Postcode": str, "Outcode": str})
    print(f"  Raw rows: {len(df):,}")

    # Standardise column names
    df.columns = df.columns.str.strip()

    # Filter to individual postcodes only (remove summary/aggregate rows)
    # Summary rows have Postcode like "All postcodes" or null
    valid_postcode = (
        df["Postcode"].notna()
        & ~df["Postcode"].str.contains("All", case=False, na=False)
        & ~df["Postcode"].str.contains("Unallocated", case=False, na=False)
        & (df["Postcode"].str.len() >= 5)  # Valid UK postcodes are 5-8 chars
    )
    df = df[valid_postcode].copy()
    print(f"  After filtering to individual postcodes: {len(df):,}")

    # Filter to England postcodes (exclude Scotland and Wales)
    # Postcode areas that are entirely in Scotland or Wales:
    _scottish_areas = {
        "AB",
        "DD",
        "DG",
        "EH",
        "FK",
        "G",
        "HS",
        "IV",
        "KA",
        "KW",
        "KY",
        "ML",
        "PA",
        "PH",
        "TD",
        "ZE",
    }
    _welsh_areas = {"CF", "LD", "LL", "NP", "SA"}
    _exclude = _scottish_areas | _welsh_areas
    area = df["Postcode"].str.extract(r"^([A-Z]{1,2})", expand=False)
    england_mask = ~area.isin(_exclude)
    n_before = len(df)
    df = df[england_mask].copy()
    print(f"  After filtering to England: {len(df):,} (removed {n_before - len(df):,})")

    # Rename metric columns with prefix
    rename_map = {
        "Num_meters": f"{prefix}_num_meters",
        "Total_cons_kwh": f"{prefix}_total_kwh",
        "Mean_cons_kwh": f"{prefix}_mean_kwh",
        "Median_cons_kwh": f"{prefix}_median_kwh",
    }
    # Handle case variations
    for orig_col in list(df.columns):
        col_lower = orig_col.lower().replace(" ", "_")
        if "num" in col_lower and "meter" in col_lower:
            rename_map[orig_col] = f"{prefix}_num_meters"
        elif "total" in col_lower and ("cons" in col_lower or "kwh" in col_lower):
            rename_map[orig_col] = f"{prefix}_total_kwh"
        elif "mean" in col_lower and ("cons" in col_lower or "kwh" in col_lower):
            rename_map[orig_col] = f"{prefix}_mean_kwh"
        elif "median" in col_lower and ("cons" in col_lower or "kwh" in col_lower):
            rename_map[orig_col] = f"{prefix}_median_kwh"

    df = df.rename(columns=rename_map)

    # Keep only Postcode and metric columns
    keep_cols = ["Postcode"] + [c for c in df.columns if c.startswith(f"{prefix}_")]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Strip whitespace from postcodes and normalise to uppercase
    df["Postcode"] = df["Postcode"].str.strip().str.upper()

    # Coerce metrics to numeric (suppressed values will become NaN)
    for col in df.columns:
        if col != "Postcode":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows where all metrics are NaN (fully suppressed)
    metric_cols = [c for c in df.columns if c != "Postcode"]
    df = df.dropna(subset=metric_cols, how="all")
    print(f"  After dropping fully suppressed: {len(df):,}")

    return df


def join_gas_electricity(elec_df: pd.DataFrame, gas_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join gas and electricity data on Postcode and compute derived metrics.

    Parameters
    ----------
    elec_df : pd.DataFrame
        Electricity consumption by postcode.
    gas_df : pd.DataFrame
        Gas consumption by postcode (weather-corrected).

    Returns
    -------
    pd.DataFrame
        Merged data with total energy and gas share columns.
    """
    merged = elec_df.merge(gas_df, on="Postcode", how="outer")

    # Derived: total mean consumption (gas + electricity)
    gas_mean = merged.get("gas_mean_kwh")
    elec_mean = merged.get("elec_mean_kwh")
    if gas_mean is not None and elec_mean is not None:
        merged["total_mean_kwh"] = gas_mean.fillna(0) + elec_mean.fillna(0)
        merged["gas_share"] = gas_mean / merged["total_mean_kwh"].replace(0, pd.NA)

    print(f"  Joined: {len(merged):,} postcodes")
    print(f"    Both gas & elec: {(elec_mean.notna() & gas_mean.notna()).sum():,}")
    print(f"    Elec only: {(elec_mean.notna() & gas_mean.isna()).sum():,}")
    print(f"    Gas only: {(elec_mean.isna() & gas_mean.notna()).sum():,}")
    return merged


def main() -> None:
    """Download and process DESNZ postcode-level energy statistics."""
    print("=" * 60)
    print("DESNZ Postcode-Level Domestic Energy Consumption")
    print(f"Target year: {TARGET_YEAR}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download electricity
    print("\n[1/4] Electricity consumption data...")
    elec_path = download_and_cache(ELEC_URL, ELEC_FILENAME)

    # 2. Download gas
    print("\n[2/4] Gas consumption data...")
    gas_path = download_and_cache(GAS_URL, GAS_FILENAME)

    # 3. Parse and join
    print("\n[3/4] Parsing data...")
    print("\n  --- Electricity ---")
    elec_df = parse_postcode_csv(elec_path, "elec")
    print("\n  --- Gas (weather-corrected) ---")
    gas_df = parse_postcode_csv(gas_path, "gas")

    print("\n  --- Joining ---")
    energy_df = join_gas_electricity(elec_df, gas_df)

    # 4. Save
    print("\n[4/4] Saving output...")
    output_path = OUTPUT_DIR / "postcode_energy_consumption.parquet"
    energy_df.to_parquet(output_path, index=False)
    print(f"  Saved {len(energy_df):,} postcodes to {output_path}")

    # Summary statistics
    print(f"\n{'=' * 60}")
    print("Summary:")
    for col in energy_df.columns:
        if col == "Postcode":
            continue
        if energy_df[col].dtype in ("float64", "int64", "Float64", "Int64"):
            valid = energy_df[col].notna().sum()
            mean_val = energy_df[col].mean()
            print(f"  {col}: {valid:,} valid, mean = {mean_val:,.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
