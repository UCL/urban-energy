"""
Process Energy Performance Certificate (EPC) data for spatial analysis.

Joins EPC records to spatial locations via direct UPRN linkage.
Only uses records with valid UPRN (available since November 2021).

Input:
    - EPC bulk download CSV(s) from https://epc.opendatacommunities.org/
    - OS Open UPRN GeoPackage

Output:
    - temp/epc/epc_domestic_cleaned.parquet (cleaned tabular data)
    - temp/epc/epc_domestic_spatial.parquet (with geometry)
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

# Configuration
BASE_DIR = Path(__file__).parent.parent
TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR = TEMP_DIR / "epc"

# Input paths
EPC_INPUT_DIR = TEMP_DIR / "epc_raw"  # Directory containing EPC CSV files
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
        Directory containing EPC CSV files.

    Returns
    -------
    pd.DataFrame
        Combined EPC records.
    """
    # TODO: Implement based on actual download structure
    raise NotImplementedError(
        "Implement based on your EPC download structure. "
        "Expected location: temp/epc_raw/"
    )


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
    # TODO: Implement cleaning
    # - Select columns from EPC_COLUMNS
    # - Parse LODGEMENT_DATE
    # - Clean UPRN to int64
    # - Filter to records with valid UPRN
    raise NotImplementedError("Implement EPC cleaning logic")


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
    # TODO: Sort by date descending, drop duplicates keeping first
    raise NotImplementedError("Implement deduplication")


def join_to_uprn(
    epc_df: pd.DataFrame, uprn_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
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
    # TODO: Ensure matching dtypes, inner join on UPRN
    raise NotImplementedError("Implement UPRN join")


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
    epc_spatial.to_parquet(OUTPUT_DIR / "epc_domestic_spatial.parquet")

    print("\nDone.")


if __name__ == "__main__":
    main()
