"""
Download and process NaPTAN (National Public Transport Access Nodes) data for England.

Downloads the full national dataset from the DfT and filters to England-only stops,
including rail stations, bus stops, tram/metro, ferry terminals, and airports.

Data source: https://beta-naptan.dft.gov.uk/
License: UK Open Government Licence (OGL)

Outputs GeoPackage to temp/transport/
"""

from io import BytesIO
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point
from tqdm import tqdm

# Configuration
TEMP_DIR = Path(__file__).parent.parent / "temp"
OUTPUT_DIR = TEMP_DIR / "transport"
CACHE_DIR = Path(__file__).parent / ".cache" / "naptan"

# NaPTAN download URL (direct CSV download)
NAPTAN_URL = "https://beta-naptan.dft.gov.uk/download/national/csv"

# ATCO Area Code ranges by country
# England: 010-499, Scotland: 601-690, Wales: 511-582
# National services (all GB): 910-940 (air, coach, ferry, rail/tram)
ENGLAND_ATCO_MIN = 10
ENGLAND_ATCO_MAX = 499

# National service codes (need geographic filtering)
NATIONAL_ATCO_CODES = {910, 920, 930, 940}

# England bounding box (approximate, in WGS84)
# Used to filter national services to England only
ENGLAND_BBOX = {
    "min_lon": -6.5,
    "max_lon": 2.0,
    "min_lat": 49.8,
    "max_lat": 56.0,
}

# Stop types of interest for accessibility analysis
STOP_TYPE_DESCRIPTIONS = {
    "BCT": "Bus/Coach bay/stop",
    "BCS": "Bus/Coach station",
    "BCE": "Bus/Coach station entrance",
    "BST": "Shared bus/tram stop",
    "FER": "Ferry terminal",
    "FTD": "Ferry terminal dock entrance",
    "GAT": "Airport entrance",
    "MET": "Metro/Underground entrance",
    "PLT": "Metro/Underground platform",
    "RLY": "Rail station",
    "RSE": "Rail station entrance",
    "RPL": "Rail platform",
    "TMU": "Tram/Metro/Underground station",
}


def download_file(url: str, desc: str, timeout: int = 600) -> bytes:
    """
    Download a file with progress bar.

    Parameters
    ----------
    url : str
        URL to download from.
    desc : str
        Description for progress bar.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    bytes
        Downloaded file content.
    """
    headers = {"User-Agent": "urban-energy-research/1.0"}
    response = requests.get(url, stream=True, timeout=timeout, headers=headers)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    content = BytesIO()

    with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            content.write(chunk)
            pbar.update(len(chunk))

    return content.getvalue()


def extract_atco_area(atco_code: str) -> int | None:
    """
    Extract the ATCO area code (first 3 digits) from a full ATCO code.

    Parameters
    ----------
    atco_code : str
        Full ATCO code (e.g., "490000001A").

    Returns
    -------
    int | None
        Three-digit area code as integer, or None if invalid.
    """
    if pd.isna(atco_code) or len(str(atco_code)) < 3:
        return None
    try:
        return int(str(atco_code)[:3])
    except ValueError:
        return None


def is_in_england_bbox(lon: float, lat: float) -> bool:
    """
    Check if coordinates fall within England's bounding box.

    Parameters
    ----------
    lon : float
        Longitude (WGS84).
    lat : float
        Latitude (WGS84).

    Returns
    -------
    bool
        True if point is within England bounding box.
    """
    if pd.isna(lon) or pd.isna(lat):
        return False
    return (
        ENGLAND_BBOX["min_lon"] <= lon <= ENGLAND_BBOX["max_lon"]
        and ENGLAND_BBOX["min_lat"] <= lat <= ENGLAND_BBOX["max_lat"]
    )


def download_naptan() -> pd.DataFrame:
    """
    Download NaPTAN data, using cache if available.

    Returns
    -------
    pd.DataFrame
        Full NaPTAN dataset.
    """
    cache_path = CACHE_DIR / "naptan_raw.parquet"

    if cache_path.exists():
        print(f"Loading cached NaPTAN from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Downloading NaPTAN national dataset...")
    content = download_file(NAPTAN_URL, "NaPTAN CSV")

    # Parse CSV
    df = pd.read_csv(BytesIO(content), low_memory=False)
    print(f"  Downloaded {len(df):,} stops")

    # Cache as parquet
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)

    return df


def filter_england_stops(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter NaPTAN data to England-only stops.

    Includes:
    - All stops with England ATCO area codes (010-499)
    - National service stops (910-940) within England's bounding box

    Parameters
    ----------
    df : pd.DataFrame
        Full NaPTAN dataset.

    Returns
    -------
    pd.DataFrame
        Filtered to England stops only.
    """
    # Extract ATCO area code
    df = df.copy()
    df["ATCOArea"] = df["ATCOCode"].apply(extract_atco_area)

    # Filter 1: England ATCO area codes
    england_atco_mask = (df["ATCOArea"] >= ENGLAND_ATCO_MIN) & (
        df["ATCOArea"] <= ENGLAND_ATCO_MAX
    )

    # Filter 2: National services within England bbox
    national_mask = df["ATCOArea"].isin(NATIONAL_ATCO_CODES)
    in_england_bbox = df.apply(
        lambda row: is_in_england_bbox(row.get("Longitude"), row.get("Latitude")),
        axis=1,
    )
    national_england_mask = national_mask & in_england_bbox

    # Combine filters
    combined_mask = england_atco_mask | national_england_mask

    n_england_atco = england_atco_mask.sum()
    n_national_england = national_england_mask.sum()
    print(f"  England ATCO stops: {n_england_atco:,}")
    print(f"  National services in England: {n_national_england:,}")

    return df[combined_mask].copy()


def create_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Convert NaPTAN DataFrame to GeoDataFrame with proper CRS.

    Parameters
    ----------
    df : pd.DataFrame
        NaPTAN data with Longitude/Latitude columns.

    Returns
    -------
    gpd.GeoDataFrame
        Spatial data in British National Grid (EPSG:27700).
    """
    # Remove rows without valid coordinates
    valid_coords = df["Longitude"].notna() & df["Latitude"].notna()
    n_invalid = (~valid_coords).sum()
    if n_invalid > 0:
        print(f"  Removing {n_invalid:,} stops without valid coordinates")
    df = df[valid_coords].copy()

    # Create geometry from coordinates
    geometry = [Point(lon, lat) for lon, lat in zip(df["Longitude"], df["Latitude"])]

    # Create GeoDataFrame in WGS84
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Transform to British National Grid for UK analysis
    gdf = gdf.to_crs(epsg=27700)

    return gdf


def select_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Select and rename relevant columns for analysis.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Full NaPTAN GeoDataFrame.

    Returns
    -------
    gpd.GeoDataFrame
        Simplified with key columns only.
    """
    # Columns to keep (mapping from NaPTAN names to cleaner names)
    column_mapping = {
        "ATCOCode": "atco_code",
        "ATCOArea": "atco_area",
        "CommonName": "name",
        "LocalityName": "locality",
        "ParentLocalityName": "parent_locality",
        "StopType": "stop_type",
        "Status": "status",
        "Easting": "easting",
        "Northing": "northing",
        "Longitude": "longitude",
        "Latitude": "latitude",
    }

    # Keep only columns that exist
    available_cols = {k: v for k, v in column_mapping.items() if k in gdf.columns}
    gdf = gdf[list(available_cols.keys()) + ["geometry"]].copy()
    gdf = gdf.rename(columns=available_cols)

    return gdf


def summarise_by_stop_type(gdf: gpd.GeoDataFrame) -> None:
    """Print summary statistics by stop type."""
    print("\nStop type summary:")
    print("-" * 50)

    counts = gdf["stop_type"].value_counts()
    for stop_type, count in counts.items():
        desc = STOP_TYPE_DESCRIPTIONS.get(stop_type, "Other")
        print(f"  {stop_type:4s}: {count:>8,}  ({desc})")

    print("-" * 50)
    print(f"  Total: {len(gdf):,} stops")


def main() -> None:
    """Main download and processing pipeline."""
    print("=" * 60)
    print("NaPTAN Transport Access Points Download")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download NaPTAN data
    print("\n[1/4] Downloading NaPTAN data...")
    df = download_naptan()
    print(f"  Total stops in GB: {len(df):,}")

    # 2. Filter to England
    print("\n[2/4] Filtering to England...")
    df_england = filter_england_stops(df)
    print(f"  England stops: {len(df_england):,}")

    # 3. Convert to GeoDataFrame
    print("\n[3/4] Creating spatial data...")
    gdf = create_geodataframe(df_england)
    gdf = select_columns(gdf)
    print(f"  Final GeoDataFrame: {len(gdf):,} stops")

    # Summary by stop type
    summarise_by_stop_type(gdf)

    # 4. Save output
    print("\n[4/4] Saving output...")

    # Save as GeoPackage
    gpkg_path = OUTPUT_DIR / "naptan_england.gpkg"
    print(f"  Saving to {gpkg_path}")
    gdf.to_file(gpkg_path, driver="GPKG")

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Output files in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
