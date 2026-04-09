"""
Download DVLA vehicle licensing statistics at LSOA level.

Downloads three datasets from DfT/DVLA:
- VEH0125: All vehicles by body type, keepership, licence status, LSOA
- VEH0135: Ultra-low emission vehicles (ULEVs) by fuel type, LSOA
- VEH0145: Plug-in vehicles (PiVs) by fuel type, LSOA

Data source: https://www.gov.uk/government/statistical-data-sets/vehicle-licensing-statistics-data-files
Publisher: Department for Transport / DVLA
License: UK Open Government Licence (OGL)

Key features:
- 2021 Census LSOA boundaries (historical data retrospectively mapped)
- Quarterly data from Q4 2009 (VEH0125) or Q4 2011 (VEH0135/0145)
- Wide format: each quarter is a separate column
- Suppression: [c] marker for 1-4 vehicles; LSOAs with <5 vehicles excluded
- Vehicle counts allocated by registered keeper's postcode

Output:
    - temp/statistics/lsoa_vehicles.parquet
        Licensed cars and ULEVs per LSOA for a target quarter, with columns:
        LSOA21CD, cars_total, cars_private, cars_company, ulev_total,
        ulev_battery_electric, ulev_plug_in_hybrid, ulev_hybrid
"""

from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from urban_energy.paths import CACHE_DIR as _CACHE_ROOT
from urban_energy.paths import DATA_DIR

OUTPUT_DIR = DATA_DIR / "statistics"
CACHE_DIR = _CACHE_ROOT / "dvla"

# Target quarter to extract (most recent full year)
TARGET_QUARTER = "2024 Q4"

# DVLA Vehicle Licensing Statistics CSV URLs
VEH0125_URL = (
    "https://assets.publishing.service.gov.uk/media/"
    "696648a3e8b93f59c3aecd93/df_VEH0125.csv"
)
VEH0135_URL = (
    "https://assets.publishing.service.gov.uk/media/"
    "696648958d599f4c09e1fffe/df_VEH0135.csv"
)

VEH0125_FILENAME = "df_VEH0125.csv"
VEH0135_FILENAME = "df_VEH0135.csv"


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


def _clean_vehicle_count(value: object) -> float | None:
    """
    Clean a vehicle count value, handling [c] suppression markers.

    Parameters
    ----------
    value : object
        Raw cell value (int, float, str, or NaN).

    Returns
    -------
    float | None
        Numeric count, or None if suppressed/missing.
        Suppressed [c] values (1-4 vehicles) are returned as NaN.
    """
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s == "[c]" or s == "c":
        return None  # Suppressed: 1-4 vehicles
    if s == "0" or s == "":
        return 0.0
    try:
        return float(s)
    except ValueError:
        return None


def parse_veh0125(path: Path, quarter: str) -> pd.DataFrame:
    """
    Parse VEH0125 (all vehicles by body type, keepership, licence status, LSOA).

    Extracts licensed cars for a target quarter, pivoted by keepership.

    Parameters
    ----------
    path : Path
        Path to the VEH0125 CSV file.
    quarter : str
        Target quarter column name, e.g. "2024 Q4".

    Returns
    -------
    pd.DataFrame
        LSOA-level car counts with columns: LSOA21CD, cars_total,
        cars_private, cars_company.
    """
    print(f"  Reading {path.name} (this is ~224 MB, may take a moment)...")
    df = pd.read_csv(path, dtype=str)
    print(f"  Raw rows: {len(df):,}")

    # Validate quarter column exists
    if quarter not in df.columns:
        available = [c for c in df.columns if "Q" in c]
        raise ValueError(
            f"Quarter '{quarter}' not found. Available: {available[:5]}..."
        )

    # Filter to: Licensed cars only, England LSOAs
    mask = (
        (df["BodyType"] == "Cars")
        & (df["LicenceStatus"] == "Licensed")
        & df["LSOA21CD"].str.startswith("E", na=False)
        & (df["LSOA21CD"] != "Miscellaneous")
    )
    df = df[mask].copy()
    print(f"  Licensed cars, England: {len(df):,} rows")

    # Clean the target quarter values
    df["count"] = df[quarter].apply(_clean_vehicle_count)

    # Pivot by keepership
    result_parts = []

    # Total cars
    total = df[df["Keepership"] == "Total"][["LSOA21CD", "count"]].copy()
    total = total.rename(columns={"count": "cars_total"})
    result_parts.append(total.set_index("LSOA21CD"))

    # Private cars
    private = df[df["Keepership"] == "PRIVATE"][["LSOA21CD", "count"]].copy()
    private = private.rename(columns={"count": "cars_private"})
    result_parts.append(private.set_index("LSOA21CD"))

    # Company cars
    company = df[df["Keepership"] == "COMPANY"][["LSOA21CD", "count"]].copy()
    company = company.rename(columns={"count": "cars_company"})
    result_parts.append(company.set_index("LSOA21CD"))

    result = pd.concat(result_parts, axis=1).reset_index()
    print(f"  Extracted {len(result):,} LSOAs for {quarter}")
    return result


def parse_veh0135(path: Path, quarter: str) -> pd.DataFrame:
    """
    Parse VEH0135 (ULEVs by fuel type, keepership, LSOA).

    Extracts ULEV counts for a target quarter, aggregated across keepership.

    Parameters
    ----------
    path : Path
        Path to the VEH0135 CSV file.
    quarter : str
        Target quarter column name, e.g. "2024 Q4".

    Returns
    -------
    pd.DataFrame
        LSOA-level ULEV counts with columns: LSOA21CD, ulev_total,
        ulev_battery_electric, ulev_plug_in_hybrid, ulev_hybrid.
    """
    print(f"  Reading {path.name}...")
    df = pd.read_csv(path, dtype=str)
    print(f"  Raw rows: {len(df):,}")

    # Validate quarter column exists
    if quarter not in df.columns:
        available = [c for c in df.columns if "Q" in c]
        raise ValueError(
            f"Quarter '{quarter}' not found. Available: {available[:5]}..."
        )

    # Filter to England LSOAs, Total keepership (to avoid double-counting)
    mask = (
        df["LSOA21CD"].str.startswith("E", na=False)
        & (df["LSOA21CD"] != "Miscellaneous")
        & (df["Keepership"] == "Total")
    )
    df = df[mask].copy()
    print(f"  England, Total keepership: {len(df):,} rows")

    # Clean the target quarter values
    df["count"] = df[quarter].apply(_clean_vehicle_count)

    # Pivot by fuel type
    result_parts = []

    # Total ULEVs
    total = df[df["Fuel"] == "Total"][["LSOA21CD", "count"]].copy()
    total = total.rename(columns={"count": "ulev_total"})
    result_parts.append(total.set_index("LSOA21CD"))

    # Battery electric
    bev = df[df["Fuel"] == "BATTERY ELECTRIC"][["LSOA21CD", "count"]].copy()
    bev = bev.rename(columns={"count": "ulev_battery_electric"})
    result_parts.append(bev.set_index("LSOA21CD"))

    # Plug-in hybrids (combine petrol and diesel)
    phev_mask = df["Fuel"].str.startswith("PLUG-IN HYBRID", na=False)
    phev = df[phev_mask].groupby("LSOA21CD")["count"].sum().reset_index()
    phev = phev.rename(columns={"count": "ulev_plug_in_hybrid"})
    result_parts.append(phev.set_index("LSOA21CD"))

    # Hybrid electric (non-plug-in, combine petrol and diesel)
    hybrid_mask = df["Fuel"].str.startswith("HYBRID ELECTRIC", na=False)
    hybrid = df[hybrid_mask].groupby("LSOA21CD")["count"].sum().reset_index()
    hybrid = hybrid.rename(columns={"count": "ulev_hybrid"})
    result_parts.append(hybrid.set_index("LSOA21CD"))

    result = pd.concat(result_parts, axis=1).reset_index()
    print(f"  Extracted {len(result):,} LSOAs for {quarter}")
    return result


def main() -> None:
    """Download and process DVLA vehicle licensing statistics."""
    print("=" * 60)
    print("DVLA Vehicle Licensing Statistics (LSOA Level)")
    print(f"Target quarter: {TARGET_QUARTER}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download VEH0125 (all vehicles)
    print("\n[1/4] All vehicles (VEH0125)...")
    veh0125_path = download_and_cache(VEH0125_URL, VEH0125_FILENAME)

    # 2. Download VEH0135 (ULEVs)
    print("\n[2/4] Ultra-low emission vehicles (VEH0135)...")
    veh0135_path = download_and_cache(VEH0135_URL, VEH0135_FILENAME)

    # 3. Parse
    print("\n[3/4] Parsing data...")
    print("\n  --- Cars (VEH0125) ---")
    cars_df = parse_veh0125(veh0125_path, TARGET_QUARTER)
    print("\n  --- ULEVs (VEH0135) ---")
    ulev_df = parse_veh0135(veh0135_path, TARGET_QUARTER)

    # Join cars and ULEVs
    print("\n  --- Joining ---")
    vehicles_df = cars_df.merge(ulev_df, on="LSOA21CD", how="left")

    # Derived: EV share of total cars
    vehicles_df["ulev_share"] = vehicles_df["ulev_total"] / vehicles_df[
        "cars_total"
    ].replace(0, pd.NA)
    vehicles_df["bev_share"] = vehicles_df["ulev_battery_electric"] / vehicles_df[
        "cars_total"
    ].replace(0, pd.NA)

    print(f"  Joined: {len(vehicles_df):,} LSOAs")

    # 4. Save
    print("\n[4/4] Saving output...")
    output_path = OUTPUT_DIR / "lsoa_vehicles.parquet"
    vehicles_df.to_parquet(output_path, index=False)
    print(f"  Saved {len(vehicles_df):,} LSOAs to {output_path}")

    # Summary statistics
    print(f"\n{'=' * 60}")
    print("Summary:")
    for col in vehicles_df.columns:
        if col == "LSOA21CD":
            continue
        if vehicles_df[col].dtype in ("float64", "int64", "Float64"):
            valid = vehicles_df[col].notna().sum()
            mean_val = vehicles_df[col].mean()
            if "share" in col:
                print(f"  {col}: {valid:,} valid, mean = {mean_val:.3f}")
            else:
                print(f"  {col}: {valid:,} valid, mean = {mean_val:,.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
