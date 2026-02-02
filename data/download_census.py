"""
Download and preprocess Census 2021 Output Area data for England and Wales.

Downloads:
- OA boundaries (GeoPackage)
- OA to LSOA/MSOA/LAD lookup table
- Census topic summary tables

Outputs joined data to temp/statistics/
"""

import zipfile
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from tqdm import tqdm

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent / "temp" / "statistics"
CACHE_DIR = Path(__file__).parent / ".cache"

# Topic summary tables to download (those available at OA level)
TOPIC_SUMMARIES = {
    1: "Usual resident population",
    6: "Population density",
    11: "Households by deprivation dimensions",
    17: "Household size",
    44: "Accommodation type",
    54: "Tenure",
    61: "Method of travel to work",
    62: "NS-SeC (socio-economic classification)",
}

# URLs
OA_BOUNDARIES_URL = (
    "https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items/"
    "6beafcfd9b9c4c9993a06b6b199d7e6d/geoPackage?layers=0"
)
OA_LOOKUP_URL = (
    "https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items/"
    "d2e894a81ae040a6a4cfb9d6c14d6285/csv?layers=0"
)
NOMISWEB_TS_URL = "https://www.nomisweb.co.uk/output/census/2021/census2021-ts{:03d}.zip"


def download_file(url: str, desc: str, timeout: int = 300) -> bytes:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    content = BytesIO()

    with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            content.write(chunk)
            pbar.update(len(chunk))

    return content.getvalue()


def download_oa_boundaries() -> gpd.GeoDataFrame:
    """Download OA boundaries from ONS Geoportal."""
    cache_path = CACHE_DIR / "oa_boundaries.gpkg"

    if cache_path.exists():
        print(f"Loading cached OA boundaries from {cache_path}")
        return gpd.read_file(cache_path)

    print("Downloading OA boundaries (~150 MB)...")
    content = download_file(OA_BOUNDARIES_URL, "OA Boundaries")

    # Save to cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(content)

    return gpd.read_file(cache_path)


def download_oa_lookup() -> pd.DataFrame:
    """Download OA to LSOA/MSOA/LAD lookup table."""
    cache_path = CACHE_DIR / "oa_lookup.csv"

    if cache_path.exists():
        print(f"Loading cached lookup from {cache_path}")
        return pd.read_csv(cache_path)

    print("Downloading OA lookup table...")
    content = download_file(OA_LOOKUP_URL, "OA Lookup")

    # Save to cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(content)

    return pd.read_csv(cache_path)


def download_topic_summary(ts_number: int) -> pd.DataFrame:
    """Download a topic summary table and extract OA-level data."""
    cache_path = CACHE_DIR / f"ts{ts_number:03d}_oa.parquet"

    if cache_path.exists():
        print(f"Loading cached TS{ts_number:03d} from {cache_path}")
        return pd.read_parquet(cache_path)

    url = NOMISWEB_TS_URL.format(ts_number)
    print(f"Downloading TS{ts_number:03d}: {TOPIC_SUMMARIES.get(ts_number, '')}...")

    try:
        content = download_file(url, f"TS{ts_number:03d}")
    except requests.exceptions.HTTPError as e:
        print(f"  Warning: Could not download TS{ts_number:03d}: {e}")
        return pd.DataFrame()

    # Extract OA-level CSV from zip
    with zipfile.ZipFile(BytesIO(content)) as zf:
        # Find the OA file (filename contains '-oa.')
        oa_files = [f for f in zf.namelist() if "-oa." in f.lower()]

        if not oa_files:
            print(f"  Warning: No OA-level data in TS{ts_number:03d}")
            return pd.DataFrame()

        oa_file = oa_files[0]
        print(f"  Extracting {oa_file}")

        with zf.open(oa_file) as f:
            # Handle potential BOM and encoding issues
            df = pd.read_csv(f, encoding="utf-8-sig")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Rename geography columns for consistency
    if "geography code" in df.columns:
        df = df.rename(columns={"geography code": "OA21CD"})
    if "geography" in df.columns:
        df = df.drop(columns=["geography"])  # Remove name column, keep code

    # Add prefix to data columns to identify source table
    data_cols = [c for c in df.columns if c not in ["OA21CD", "date"]]
    rename_map = {c: f"ts{ts_number:03d}_{c}" for c in data_cols}
    df = df.rename(columns=rename_map)

    # Drop date column if present
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    # Cache as parquet
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def main():
    """Main download and processing pipeline."""
    print("=" * 60)
    print("Census 2021 Output Area Data Download")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download OA boundaries
    print("\n[1/4] Downloading OA boundaries...")
    oa_gdf = download_oa_boundaries()
    print(f"  Loaded {len(oa_gdf):,} Output Areas")

    # Standardise column name
    if "OA21CD" not in oa_gdf.columns:
        # Find the OA code column
        oa_col = [c for c in oa_gdf.columns if "OA21CD" in c.upper()]
        if oa_col:
            oa_gdf = oa_gdf.rename(columns={oa_col[0]: "OA21CD"})

    # Save raw boundaries
    oa_boundaries_path = OUTPUT_DIR / "oa_boundaries.gpkg"
    print(f"  Saving to {oa_boundaries_path}")
    oa_gdf.to_file(oa_boundaries_path, driver="GPKG")

    # 2. Download lookup table
    print("\n[2/4] Downloading OA lookup table...")
    lookup_df = download_oa_lookup()
    print(f"  Loaded {len(lookup_df):,} rows")

    # Standardise column names
    lookup_df.columns = lookup_df.columns.str.strip()

    # Save as parquet
    lookup_path = OUTPUT_DIR / "oa_lookup.parquet"
    print(f"  Saving to {lookup_path}")
    lookup_df.to_parquet(lookup_path, index=False)

    # 3. Download topic summaries
    print("\n[3/4] Downloading Census topic summaries...")
    ts_dataframes = {}

    for ts_num, ts_desc in TOPIC_SUMMARIES.items():
        df = download_topic_summary(ts_num)
        if not df.empty:
            ts_dataframes[ts_num] = df

            # Save individual table
            ts_path = OUTPUT_DIR / f"census_ts{ts_num:03d}_oa.parquet"
            df.to_parquet(ts_path, index=False)

    # 4. Join everything together
    print("\n[4/4] Joining all data...")

    # Start with boundaries (keep only OA21CD and geometry)
    keep_cols = ["OA21CD", "geometry"]
    extra_cols = [c for c in oa_gdf.columns if c not in keep_cols and "Shape" not in c]
    if extra_cols:
        print(f"  Dropping boundary metadata columns: {extra_cols[:5]}...")
    joined_gdf = oa_gdf[keep_cols].copy()

    # Join lookup
    lookup_cols = ["OA21CD", "LSOA21CD", "LSOA21NM", "MSOA21CD", "MSOA21NM", "LAD22CD", "LAD22NM"]
    available_cols = [c for c in lookup_cols if c in lookup_df.columns]
    joined_gdf = joined_gdf.merge(lookup_df[available_cols], on="OA21CD", how="left")
    print(f"  Joined lookup: {len(joined_gdf):,} rows")

    # Join each topic summary
    for ts_num, ts_df in ts_dataframes.items():
        if "OA21CD" in ts_df.columns:
            joined_gdf = joined_gdf.merge(ts_df, on="OA21CD", how="left")
            print(f"  Joined TS{ts_num:03d}: {len(ts_df.columns) - 1} columns")

    # Save final joined dataset
    output_path = OUTPUT_DIR / "census_oa_joined.gpkg"
    print(f"\nSaving joined dataset to {output_path}")
    print(f"  {len(joined_gdf):,} rows x {len(joined_gdf.columns)} columns")
    joined_gdf.to_file(output_path, driver="GPKG")

    # Also save as parquet (without geometry, for faster tabular analysis)
    parquet_path = OUTPUT_DIR / "census_oa_joined.parquet"
    joined_df = pd.DataFrame(joined_gdf.drop(columns=["geometry"]))
    joined_df.to_parquet(parquet_path, index=False)
    print(f"  Also saved tabular data to {parquet_path}")

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Output files in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
