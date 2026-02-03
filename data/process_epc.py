"""
Process Energy Performance Certificate (EPC) data for spatial analysis.

Joins EPC records to spatial locations via direct UPRN linkage.
Only uses records with valid UPRN (available since November 2021).

Input:
    - EPC bulk download CSV(s) from https://epc.opendatacommunities.org/
    - OS Open UPRN GeoPackage

Output:
    - temp/epc/epc_domestic_cleaned.parquet (cleaned tabular data, no geometry)
    - temp/epc/epc_domestic_spatial.gpkg (with geometry)
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

# Configuration
BASE_DIR = Path(__file__).parent.parent
TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR = TEMP_DIR / "epc"

# Input paths
EPC_INPUT_DIR = (
    TEMP_DIR / "all-domestic-certificates"
)  # Directory containing EPC CSV files
UPRN_PATH = TEMP_DIR / "osopenuprn_202601_gpkg" / "osopenuprn_202601.gpkg"

# Key EPC columns to retain
EPC_COLUMNS = [
    # Identifiers
    "LMK_KEY",
    "UPRN",
    "POSTCODE",
    # Energy metrics
    "CURRENT_ENERGY_RATING",
    "CURRENT_ENERGY_EFFICIENCY",
    "ENERGY_CONSUMPTION_CURRENT",
    # Building characteristics
    "PROPERTY_TYPE",
    "BUILT_FORM",
    "TOTAL_FLOOR_AREA",
    "NUMBER_HABITABLE_ROOMS",
    "CONSTRUCTION_AGE_BAND",
    # Heating
    "MAIN_FUEL",
    "HEATING_COST_CURRENT",
    # Walls/roof
    "WALLS_DESCRIPTION",
    "ROOF_DESCRIPTION",
    # Tenure and dates
    "TENURE",
    "LODGEMENT_DATE",
]


def load_epc_certificates(input_dir: Path) -> pd.DataFrame:
    """
    Load and concatenate EPC certificate CSVs.

    Parameters
    ----------
    input_dir : Path
        Directory containing EPC CSV files (organized by local authority).

    Returns
    -------
    pd.DataFrame
        Combined EPC records.
    """
    if not input_dir.exists():
        raise FileNotFoundError(
            f"EPC input directory not found: {input_dir}\n"
            f"Download from: https://epc.opendatacommunities.org/"
        )

    # Find all certificates.csv files in subdirectories
    csv_files = list(input_dir.glob("*/certificates.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No certificates.csv files found in {input_dir}")

    print(f"  Found {len(csv_files)} local authority files")

    # Load and concatenate
    dfs = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, usecols=EPC_COLUMNS, low_memory=False)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(combined):,} total records")

    return combined


def clean_epc_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise EPC data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw EPC records.

    Returns
    -------
    pd.DataFrame
        Cleaned records with standardised types.
    """
    df = df.copy()

    # Ensure string columns are string type (avoid mixed type issues with parquet)
    df["LMK_KEY"] = df["LMK_KEY"].astype(str)

    # Parse lodgement date
    df["LODGEMENT_DATE"] = pd.to_datetime(df["LODGEMENT_DATE"], errors="coerce")

    # Clean UPRN to int64 (filter out invalid)
    df["UPRN"] = pd.to_numeric(df["UPRN"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["UPRN"])
    df["UPRN"] = df["UPRN"].astype("int64")
    print(f"  Filtered to UPRN records: {n_before:,} -> {len(df):,}")

    # Clean numeric columns
    numeric_cols = [
        "CURRENT_ENERGY_EFFICIENCY", "ENERGY_CONSUMPTION_CURRENT",
        "TOTAL_FLOOR_AREA", "HEATING_COST_CURRENT", "NUMBER_HABITABLE_ROOMS",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def deduplicate_by_uprn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the most recent certificate per UPRN.

    Parameters
    ----------
    df : pd.DataFrame
        EPC records with UPRN and LODGEMENT_DATE.

    Returns
    -------
    pd.DataFrame
        One record per UPRN (most recent).
    """
    n_before = len(df)
    df = df.sort_values("LODGEMENT_DATE", ascending=False)
    df = df.drop_duplicates(subset=["UPRN"], keep="first")
    print(f"  Deduplicated: {n_before:,} -> {len(df):,}")
    return df


def join_to_uprn(epc_df: pd.DataFrame, uprn_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Join EPC records to UPRN coordinates.

    Parameters
    ----------
    epc_df : pd.DataFrame
        EPC records with UPRN column (int64).
    uprn_gdf : gpd.GeoDataFrame
        OS Open UPRN with geometry.

    Returns
    -------
    gpd.GeoDataFrame
        EPC records with point geometry.
    """
    # Ensure matching dtypes
    uprn_gdf = uprn_gdf.copy()
    uprn_gdf["UPRN"] = uprn_gdf["UPRN"].astype("int64")

    # Inner join on UPRN
    merged = uprn_gdf[["UPRN", "geometry"]].merge(epc_df, on="UPRN", how="inner")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=uprn_gdf.crs)


def main() -> None:
    """Main processing pipeline."""
    print("=" * 60)
    print("EPC Data Processing (UPRN linkage only)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and clean
    print("\n[1/4] Loading EPC certificates...")
    epc_raw = load_epc_certificates(EPC_INPUT_DIR)

    print("\n[2/4] Cleaning and filtering to UPRN records...")
    epc_clean = clean_epc_data(epc_raw)
    epc_dedup = deduplicate_by_uprn(epc_clean)
    print(f"  {len(epc_dedup):,} unique properties with UPRN")

    # Spatial join
    print("\n[3/4] Loading UPRN and joining...")
    uprn = gpd.read_file(UPRN_PATH)
    epc_spatial = join_to_uprn(epc_dedup, uprn)
    print(f"  Matched {len(epc_spatial):,} records")

    # Save
    print("\n[4/4] Saving outputs...")
    epc_dedup.to_parquet(OUTPUT_DIR / "epc_domestic_cleaned.parquet")
    epc_spatial.to_file(OUTPUT_DIR / "epc_domestic_spatial.gpkg", driver="GPKG")

    print("\nDone.")


if __name__ == "__main__":
    main()
