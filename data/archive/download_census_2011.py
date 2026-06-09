"""
Download Census 2011 commute data at OA level for pre-pandemic validation.

Downloads QS701EW (method of travel to work) and QS702EW (distance travelled
to work) at Output Area level from Nomis. Many OA codes are unchanged between
2011 and 2021; we do a direct join on matching codes and drop the rest.

Usage:
    uv run python data/download_census_2011.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from urban_energy.paths import CACHE_DIR, DATA_DIR

OUTPUT_DIR = DATA_DIR / "statistics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "urban-energy-research/1.0"}

# Energy intensities (kWh per passenger-km) — same as 2021 analysis
ROAD_KWH_PKM = 0.399
RAIL_KWH_PKM = 0.178
WORKDAYS = 220

# Census 2011 distance bands and midpoints (km)
DISTANCE_BANDS_2011: dict[str, float] = {
    "Less than 2km": 1.0,
    "2km to less than 5km": 3.5,
    "5km to less than 10km": 7.5,
    "10km to less than 20km": 15.0,
    "20km to less than 30km": 25.0,
    "30km to less than 40km": 35.0,
    "40km to less than 60km": 50.0,
    "60km and over": 80.0,
}


def download_oa_table(table_id: str, desc: str) -> pd.DataFrame:
    """
    Download a Census 2011 table at OA level from Nomis bulk download.

    Uses the pre-packaged OA-level ZIP files from Nomis, which contain the
    full dataset split by region. Concatenates all English region files.
    """
    import zipfile

    cache_zip = CACHE_DIR / f"census_2011_{table_id}_oa.zip"
    cache_parquet = CACHE_DIR / f"census_2011_{table_id}_oa_england.parquet"

    if cache_parquet.exists():
        print(f"  Using cached {table_id}")
        return pd.read_parquet(cache_parquet)

    # Download bulk ZIP — try two URL patterns
    if not cache_zip.exists():
        urls = [
            f"https://www.nomisweb.co.uk/output/census/2011/{table_id.lower()}_oa.zip",
            f"https://www.nomisweb.co.uk/output/census/2011/{table_id.lower()}_2011_oa.zip",
        ]
        for url in urls:
            print(f"  Trying {url.split('/')[-1]}...")
            resp = requests.get(url, headers=HEADERS, timeout=600)
            if resp.status_code == 200 and len(resp.content) > 100:
                cache_zip.write_bytes(resp.content)
                print(f"  Saved {cache_zip.name} ({len(resp.content) / 1e6:.1f} MB)")
                break
        else:
            raise RuntimeError(f"Could not download {table_id} from Nomis")

    # Read all English region data files from ZIP
    # Region suffixes: A, B, D, E, F, G, H, J, K = English regions
    # W = Wales (exclude)
    with zipfile.ZipFile(cache_zip) as z:
        data_files = [
            f for f in z.namelist()
            if f.upper().endswith(".CSV")
            and "DATA" in f.upper()
            and not f.upper().endswith("_W.CSV")  # exclude Wales
        ]
        print(f"  Found {len(data_files)} data CSV files")

        parts = []
        for csv_file in sorted(data_files):
            df = pd.read_csv(z.open(csv_file))
            # Filter to OA codes (E00xxxxxx), skip national/regional totals
            if "GeographyCode" in df.columns:
                df = df[df["GeographyCode"].str.startswith("E00")]
            parts.append(df)
            print(f"    {csv_file.split('/')[-1]}: {len(df):,} OAs")

    combined = pd.concat(parts, ignore_index=True)
    combined.to_parquet(cache_parquet, index=False)
    print(f"  Total: {len(combined):,} English OAs")
    return combined


def process_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process QS702EW bulk data into per-OA commute distance metrics.

    Column mapping (from CODE file):
        QS702EW0001: Total
        QS702EW0002: Less than 2km          (midpoint: 1.0)
        QS702EW0003: 2km to less than 5km   (midpoint: 3.5)
        QS702EW0004: 5km to less than 10km  (midpoint: 7.5)
        QS702EW0005: 10km to less than 20km (midpoint: 15.0)
        QS702EW0006: 20km to less than 30km (midpoint: 25.0)
        QS702EW0007: 30km to less than 40km (midpoint: 35.0)
        QS702EW0008: 40km to less than 60km (midpoint: 50.0)
        QS702EW0009: 60km and over          (midpoint: 80.0)
        QS702EW0010: Work mainly from home
        QS702EW0011: Other
        QS702EW0013: Average distance (km)
    """
    df = df.rename(columns={"GeographyCode": "OA11CD"})

    # Distance bands: code → midpoint
    band_cols = {
        "QS702EW0002": 1.0,
        "QS702EW0003": 3.5,
        "QS702EW0004": 7.5,
        "QS702EW0005": 15.0,
        "QS702EW0006": 25.0,
        "QS702EW0007": 35.0,
        "QS702EW0008": 50.0,
        "QS702EW0009": 80.0,
    }

    num = pd.Series(0.0, index=df.index)
    den = pd.Series(0.0, index=df.index)
    for col, mid in band_cols.items():
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
            num += vals * mid
            den += vals

    df["avg_commute_km_2011"] = num / den.replace(0, np.nan)
    df["n_travelling_2011"] = den

    # WFH share
    total = pd.to_numeric(df.get("QS702EW0001", 0), errors="coerce").fillna(0)
    wfh = pd.to_numeric(df.get("QS702EW0010", 0), errors="coerce").fillna(0)
    df["wfh_share_2011"] = wfh / total.replace(0, np.nan)

    return df[["OA11CD", "avg_commute_km_2011", "n_travelling_2011", "wfh_share_2011"]]


def process_mode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process QS701EW bulk data into per-OA commute mode shares.

    Column mapping (from Nomis):
        QS701EW0001: All categories (total in employment + not in employment)
        QS701EW0002: Work mainly at or from home
        QS701EW0003: Underground, metro, light rail, tram
        QS701EW0004: Train
        QS701EW0005: Bus, minibus or coach
        QS701EW0006: Taxi
        QS701EW0007: Motorcycle, scooter or moped
        QS701EW0008: Driving a car or van
        QS701EW0009: Passenger in a car or van
        QS701EW0010: Bicycle
        QS701EW0011: On foot
        QS701EW0012: Other
        QS701EW0013: Not in employment
    """
    df = df.rename(columns={"GeographyCode": "OA11CD"})

    def _col(code: str) -> pd.Series:
        return pd.to_numeric(df.get(code, 0), errors="coerce").fillna(0)

    total = _col("QS701EW0001") - _col("QS701EW0013")  # employed only
    total = total.replace(0, np.nan)

    car_driver = _col("QS701EW0008")
    car_pass = _col("QS701EW0009")
    bus = _col("QS701EW0005")
    train = _col("QS701EW0004")
    metro = _col("QS701EW0003")
    walk = _col("QS701EW0011")
    cycle = _col("QS701EW0010")
    taxi = _col("QS701EW0006")
    motor = _col("QS701EW0007")

    private = car_driver + car_pass + taxi + motor
    public = bus + train + metro
    active = walk + cycle

    df["car_share_2011"] = car_driver / total
    df["private_share_2011"] = private / total
    df["public_share_2011"] = public / total
    df["active_share_2011"] = active / total

    return df[["OA11CD", "car_share_2011", "private_share_2011",
               "public_share_2011", "active_share_2011"]]


def main() -> None:
    """Download and process Census 2011 commute data at OA level."""
    print("=" * 70)
    print("CENSUS 2011 COMMUTE DATA — OA LEVEL (PRE-PANDEMIC VALIDATION)")
    print("=" * 70)

    # Step 1: Download
    print("\n[1/4] Downloading QS702EW (distance to work, OA level)")
    dist_raw = download_oa_table("QS702EW", "Distance travelled to work")
    print(f"  Rows: {len(dist_raw):,}")

    print("\n[2/4] Downloading QS701EW (mode of travel, OA level)")
    mode_raw = download_oa_table("QS701EW", "Method of travel to work")
    print(f"  Rows: {len(mode_raw):,}")

    # Step 2: Process
    print("\n[3/4] Processing...")
    dist = process_distance(dist_raw)
    mode = process_mode(mode_raw)

    combined = dist.merge(mode, on="OA11CD", how="outer")
    combined = combined[combined["OA11CD"].str.startswith("E")].copy()
    print(f"  England OAs (2011): {len(combined):,}")

    # Step 3: Estimate commute energy (same method as 2021)
    avg_km = combined["avg_commute_km_2011"].fillna(0)
    annual_km = avg_km * 2 * WORKDAYS

    private_share = combined["private_share_2011"].fillna(0)
    public_share = combined["public_share_2011"].fillna(0)

    combined["private_commute_kwh_2011"] = annual_km * private_share * ROAD_KWH_PKM
    combined["public_commute_kwh_2011"] = annual_km * public_share * RAIL_KWH_PKM
    combined["commute_kwh_2011"] = (
        combined["private_commute_kwh_2011"] + combined["public_commute_kwh_2011"]
    )
    # Overall scenario (6.04x)
    combined["transport_kwh_2011_overall"] = combined["commute_kwh_2011"] * 6.04

    # Step 4: Map to OA21 via ONS OA11→OA21 exact fit lookup
    print("\n[4/4] Mapping to OA 2021 via ONS lookup...")
    lookup_path = CACHE_DIR / "oa11_to_oa21_lookup.parquet"
    if not lookup_path.exists():
        raise FileNotFoundError(
            f"OA11→OA21 lookup not found at {lookup_path}. "
            "Run the download script or check CACHE_DIR."
        )
    lookup = pd.read_parquet(lookup_path)
    lookup = lookup[lookup["OA11CD"].str.startswith("E")]
    print(f"  Lookup: {len(lookup):,} England OA mappings")

    # For unchanged OAs (U): direct 1:1 mapping
    # For split OAs (S): one OA11 → multiple OA21 → duplicate 2011 data to each
    # For merged OAs (M): multiple OA11 → one OA21 → aggregate (weighted by commuters)
    combined = combined.merge(
        lookup[["OA11CD", "OA21CD", "CHNGIND"]],
        on="OA11CD",
        how="inner",
    )
    print(f"  Matched: {len(combined):,} rows")
    print(f"    Unchanged (U): {(combined['CHNGIND'] == 'U').sum():,}")
    print(f"    Split (S):     {(combined['CHNGIND'] == 'S').sum():,}")
    print(f"    Merged (M):    {(combined['CHNGIND'] == 'M').sum():,}")

    # For split: the 2011 values apply to all child OA21s (same neighbourhood)
    # For merged: aggregate by OA21CD, weighted by n_travelling_2011
    numeric_cols = [
        "avg_commute_km_2011", "car_share_2011", "private_share_2011",
        "public_share_2011", "active_share_2011", "commute_kwh_2011",
        "transport_kwh_2011_overall",
    ]

    # Group by OA21CD and take weighted mean (weight = n_travelling)
    def _weighted_agg(group: pd.DataFrame) -> pd.Series:
        w = group["n_travelling_2011"].fillna(0)
        total_w = w.sum()
        if total_w == 0:
            return group[numeric_cols].mean()
        result_row = {}
        for col in numeric_cols:
            result_row[col] = (group[col].fillna(0) * w).sum() / total_w
        result_row["n_travelling_2011"] = total_w
        return pd.Series(result_row)

    result = combined.groupby("OA21CD").apply(_weighted_agg, include_groups=False).reset_index()
    # Add WFH if present
    if "wfh_share_2011" in combined.columns:
        wfh = combined.groupby("OA21CD")["wfh_share_2011"].mean()
        result = result.merge(wfh, on="OA21CD", how="left")

    # Summary
    print(f"\n  Result columns: {list(result.columns)}")
    print(f"\n  Summary (Census 2011, {len(result):,} OAs):")
    for col in result.columns:
        if col == "OA21CD":
            continue
        val = result[col].median()
        if "share" in col:
            print(f"    {col}: {val:.1%}")
        elif "kwh" in col:
            print(f"    {col}: {val:,.0f}")
        else:
            print(f"    {col}: {val:.1f}")

    # Save
    out_cols = [
        "OA21CD", "avg_commute_km_2011", "n_travelling_2011",
        "car_share_2011", "private_share_2011", "public_share_2011",
        "active_share_2011", "commute_kwh_2011", "transport_kwh_2011_overall",
    ]
    if "wfh_share_2011" in result.columns:
        out_cols.append("wfh_share_2011")

    out_path = OUTPUT_DIR / "census_2011_commute_oa.parquet"
    result[out_cols].to_parquet(out_path, index=False)
    print(f"\n  Saved {out_path} ({len(result):,} OAs)")


if __name__ == "__main__":
    main()
