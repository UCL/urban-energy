"""
Process Energy Performance Certificate (EPC) data for spatial analysis.

Scope: DOMESTIC (residential) buildings only. Non-domestic EPCs are a separate
dataset not included in this analysis.

Joins EPC records to spatial locations via direct UPRN linkage.
Only uses records with valid UPRN (available since November 2021).
Filters to England only (excludes Welsh local authorities).

Memory-efficient: processes in disk-backed phases to avoid holding the full
EPC dataset and UPRN GeoPackage in memory simultaneously.

Input:
    - EPC bulk download CSV(s) from https://epc.opendatacommunities.org/
      (domestic certificates only - the "all-domestic-certificates" download)
    - OS Open UPRN GeoPackage

Output:
    - temp/epc/epc_domestic_spatial.parquet (GeoParquet with UPRN point geometry)
    - temp/epc/epc_domestic_cleaned.parquet (tabular, no geometry — intermediate)
"""

import gc
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyogrio

from urban_energy.paths import TEMP_DIR

OUTPUT_DIR = TEMP_DIR / "epc"

# Input paths
EPC_INPUT_DIR = (
    TEMP_DIR / "all-domestic-certificates"
)  # Directory containing EPC CSV files
UPRN_PATH = TEMP_DIR / "osopenuprn_202601_gpkg" / "osopenuprn_202601.gpkg"

# EPC columns to extract from bulk download CSVs
EPC_COLUMNS = [
    # === Identifiers ===
    "LMK_KEY",
    "UPRN",
    "UPRN_SOURCE",
    "POSTCODE",
    # === Energy metrics (current) ===
    "CURRENT_ENERGY_RATING",
    "CURRENT_ENERGY_EFFICIENCY",
    "ENERGY_CONSUMPTION_CURRENT",
    # === Energy metrics (potential) ===
    "POTENTIAL_ENERGY_RATING",
    "POTENTIAL_ENERGY_EFFICIENCY",
    "ENERGY_CONSUMPTION_POTENTIAL",
    # === CO2 and environmental impact ===
    "CO2_EMISSIONS_CURRENT",
    "CO2_EMISS_CURR_PER_FLOOR_AREA",
    "ENVIRONMENT_IMPACT_CURRENT",
    "ENVIRONMENT_IMPACT_POTENTIAL",
    # === Building characteristics ===
    "PROPERTY_TYPE",
    "BUILT_FORM",
    "TOTAL_FLOOR_AREA",
    "NUMBER_HABITABLE_ROOMS",
    "NUMBER_HEATED_ROOMS",
    "CONSTRUCTION_AGE_BAND",
    "EXTENSION_COUNT",
    "TRANSACTION_TYPE",
    "REPORT_TYPE",
    # === Floor position (flat stratification) ===
    "FLOOR_LEVEL",
    "FLAT_TOP_STOREY",
    "FLAT_STOREY_COUNT",
    "FLOOR_HEIGHT",
    "HEAT_LOSS_CORRIDOR",
    "UNHEATED_CORRIDOR_LENGTH",
    # === Building fabric ===
    "MAINS_GAS_FLAG",
    "GLAZED_TYPE",
    "MULTI_GLAZE_PROPORTION",
    "LOW_ENERGY_LIGHTING",
    "NUMBER_OPEN_FIREPLACES",
    "MECHANICAL_VENTILATION",
    # === Heating ===
    "MAIN_FUEL",
    "MAINHEAT_DESCRIPTION",
    "MAINHEATCONT_DESCRIPTION",
    "MAIN_HEATING_CONTROLS",
    "SECONDHEAT_DESCRIPTION",
    # === Hot water ===
    "HOTWATER_DESCRIPTION",
    # === Envelope descriptions ===
    "WALLS_DESCRIPTION",
    "ROOF_DESCRIPTION",
    "FLOOR_DESCRIPTION",
    "WINDOWS_DESCRIPTION",
    "LIGHTING_DESCRIPTION",
    # === Component energy efficiency ratings ===
    "WALLS_ENERGY_EFF",
    "WINDOWS_ENERGY_EFF",
    "ROOF_ENERGY_EFF",
    "FLOOR_ENERGY_EFF",
    "MAINHEAT_ENERGY_EFF",
    "HOT_WATER_ENERGY_EFF",
    "LIGHTING_ENERGY_EFF",
    "MAINHEATC_ENERGY_EFF",
    "SHEATING_ENERGY_EFF",
    # === Component environmental efficiency ratings ===
    "WALLS_ENV_EFF",
    "WINDOWS_ENV_EFF",
    "ROOF_ENV_EFF",
    "FLOOR_ENV_EFF",
    "MAINHEAT_ENV_EFF",
    "HOT_WATER_ENV_EFF",
    "LIGHTING_ENV_EFF",
    # === Renewables ===
    "PHOTO_SUPPLY",
    "SOLAR_WATER_HEATING_FLAG",
    "WIND_TURBINE_COUNT",
    # === Costs (current) ===
    "HEATING_COST_CURRENT",
    "LIGHTING_COST_CURRENT",
    "HOT_WATER_COST_CURRENT",
    # === Costs (potential) ===
    "HEATING_COST_POTENTIAL",
    "LIGHTING_COST_POTENTIAL",
    "HOT_WATER_COST_POTENTIAL",
    # === Tenure and dates ===
    "TENURE",
    "LODGEMENT_DATE",
    "INSPECTION_DATE",
]

NUMERIC_COLS = [
    # Energy metrics
    "CURRENT_ENERGY_EFFICIENCY",
    "ENERGY_CONSUMPTION_CURRENT",
    "POTENTIAL_ENERGY_EFFICIENCY",
    "ENERGY_CONSUMPTION_POTENTIAL",
    # CO2 / environmental
    "CO2_EMISSIONS_CURRENT",
    "CO2_EMISS_CURR_PER_FLOOR_AREA",
    "ENVIRONMENT_IMPACT_CURRENT",
    "ENVIRONMENT_IMPACT_POTENTIAL",
    # Building dimensions
    "TOTAL_FLOOR_AREA",
    "NUMBER_HABITABLE_ROOMS",
    "NUMBER_HEATED_ROOMS",
    "FLOOR_LEVEL",
    "FLOOR_HEIGHT",
    "FLAT_STOREY_COUNT",
    "UNHEATED_CORRIDOR_LENGTH",
    "EXTENSION_COUNT",
    # Fabric
    "MULTI_GLAZE_PROPORTION",
    "LOW_ENERGY_LIGHTING",
    "NUMBER_OPEN_FIREPLACES",
    # Renewables
    "PHOTO_SUPPLY",
    "WIND_TURBINE_COUNT",
    # Costs
    "HEATING_COST_CURRENT",
    "LIGHTING_COST_CURRENT",
    "HOT_WATER_COST_CURRENT",
    "HEATING_COST_POTENTIAL",
    "LIGHTING_COST_POTENTIAL",
    "HOT_WATER_COST_POTENTIAL",
]

STRING_COLS = [
    "FLAT_TOP_STOREY",
    "MAINS_GAS_FLAG",
    "SOLAR_WATER_HEATING_FLAG",
    "GLAZED_TYPE",
    "MECHANICAL_VENTILATION",
    "HEAT_LOSS_CORRIDOR",
    "CURRENT_ENERGY_RATING",
    "POTENTIAL_ENERGY_RATING",
    "TRANSACTION_TYPE",
    "REPORT_TYPE",
    "UPRN_SOURCE",
    # Efficiency ratings
    "WALLS_ENERGY_EFF",
    "WINDOWS_ENERGY_EFF",
    "ROOF_ENERGY_EFF",
    "FLOOR_ENERGY_EFF",
    "MAINHEAT_ENERGY_EFF",
    "HOT_WATER_ENERGY_EFF",
    "LIGHTING_ENERGY_EFF",
    "MAINHEATC_ENERGY_EFF",
    "SHEATING_ENERGY_EFF",
    "WALLS_ENV_EFF",
    "WINDOWS_ENV_EFF",
    "ROOF_ENV_EFF",
    "FLOOR_ENV_EFF",
    "MAINHEAT_ENV_EFF",
    "HOT_WATER_ENV_EFF",
    "LIGHTING_ENV_EFF",
]


def _find_csv_files(input_dir: Path) -> list[Path]:
    """
    Find England-only EPC certificate CSV files.

    Parameters
    ----------
    input_dir : Path
        Directory containing EPC CSV files (organized by local authority).

    Returns
    -------
    list[Path]
        Sorted list of CSV file paths (Welsh LAs excluded).
    """
    if not input_dir.exists():
        raise FileNotFoundError(
            f"EPC input directory not found: {input_dir}\n"
            f"Download from: https://epc.opendatacommunities.org/"
        )

    all_csv_files = sorted(input_dir.glob("*/certificates.csv"))
    if not all_csv_files:
        raise FileNotFoundError(f"No certificates.csv files found in {input_dir}")

    # Exclude Welsh LAs (codes starting W0)
    csv_files = [f for f in all_csv_files if "-W0" not in f.parent.name]
    n_excluded = len(all_csv_files) - len(csv_files)
    print(f"  Found {len(csv_files)} England LA files (excluded {n_excluded} Welsh)")
    return csv_files


def _clean_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a batch of EPC records in-place.

    Coerces types, drops records without valid UPRN, strips whitespace
    from categorical columns. Operates on a single batch to limit memory.

    Parameters
    ----------
    df : pd.DataFrame
        Raw EPC records for one batch.

    Returns
    -------
    pd.DataFrame
        Cleaned records with standardised types.
    """
    df["LMK_KEY"] = df["LMK_KEY"].astype(str)

    # Parse dates
    df["LODGEMENT_DATE"] = pd.to_datetime(df["LODGEMENT_DATE"], errors="coerce")
    if "INSPECTION_DATE" in df.columns:
        df["INSPECTION_DATE"] = pd.to_datetime(df["INSPECTION_DATE"], errors="coerce")

    # Filter to valid UPRN
    df["UPRN"] = pd.to_numeric(df["UPRN"], errors="coerce")
    df = df.dropna(subset=["UPRN"])
    df["UPRN"] = df["UPRN"].astype("int64")

    # Numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # String columns — strip whitespace
    for col in STRING_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    # Catch-all: convert any remaining object columns to string
    # (descriptions, free-text fields not in STRING_COLS)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("").astype(str)

    return df


def write_epc_to_parquet(
    input_dir: Path, output_path: Path, batch_size: int = 40
) -> int:
    """
    Read EPC CSVs in batches, clean, and write to a single parquet file.

    Parameters
    ----------
    input_dir : Path
        Directory containing EPC CSV files.
    output_path : Path
        Path for the intermediate parquet file.
    batch_size : int
        Number of LA CSV files per batch.

    Returns
    -------
    int
        Total number of records written.
    """
    csv_files = _find_csv_files(input_dir)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    total_with_uprn = 0

    for i in range(0, len(csv_files), batch_size):
        batch_files = csv_files[i : i + batch_size]
        batch_end = min(i + batch_size, len(csv_files))
        print(f"  Batch {i // batch_size + 1}: files {i + 1}-{batch_end}")

        dfs = []
        for csv_path in batch_files:
            df = pd.read_csv(csv_path, usecols=EPC_COLUMNS, low_memory=False)
            dfs.append(df)

        batch_df = pd.concat(dfs, ignore_index=True)
        del dfs
        total_rows += len(batch_df)

        batch_df = _clean_batch(batch_df)
        total_with_uprn += len(batch_df)

        table = pa.Table.from_pandas(batch_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(output_path), table.schema)
        writer.write_table(table)

        del batch_df, table
        gc.collect()

    if writer is not None:
        writer.close()

    print(f"  Total records: {total_rows:,}")
    print(f"  With valid UPRN: {total_with_uprn:,}")
    return total_with_uprn


def deduplicate_on_disk(input_path: Path, output_path: Path) -> int:
    """
    Deduplicate parquet by UPRN, keeping the most recent certificate.

    Reads only key columns to find winners, then reads full data filtered
    to those keys.

    Parameters
    ----------
    input_path : Path
        Intermediate parquet with all records.
    output_path : Path
        Output parquet with one record per UPRN.

    Returns
    -------
    int
        Number of deduplicated records.
    """
    # Pass 1: read only dedup keys to find winners
    print("  Pass 1: finding most recent certificate per UPRN...")
    keys = pd.read_parquet(input_path, columns=["UPRN", "LODGEMENT_DATE", "LMK_KEY"])
    keys = keys.sort_values("LODGEMENT_DATE", ascending=False)
    keys = keys.drop_duplicates(subset=["UPRN"], keep="first")
    keep_keys = set(keys["LMK_KEY"])
    n_dedup = len(keys)
    del keys
    gc.collect()

    # Pass 2: read full data, filter to winners
    print(f"  Pass 2: reading {n_dedup:,} winning records...")
    df = pd.read_parquet(input_path)
    df = df[df["LMK_KEY"].isin(keep_keys)]
    del keep_keys
    gc.collect()

    df.to_parquet(output_path, index=False)
    print(f"  Deduplicated: {len(df):,} unique UPRNs")

    n = len(df)
    del df
    gc.collect()
    return n


def read_uprn_filtered(
    uprn_path: Path, needed_uprns: set[int], chunk_size: int = 2_000_000
) -> gpd.GeoDataFrame:
    """
    Read UPRN GeoPackage in chunks, keeping only needed UPRNs.

    Parameters
    ----------
    uprn_path : Path
        Path to OS Open UPRN GeoPackage.
    needed_uprns : set[int]
        Set of UPRN values to retain.
    chunk_size : int
        Number of features to read per chunk.

    Returns
    -------
    gpd.GeoDataFrame
        Filtered UPRN records with geometry.
    """
    info = pyogrio.read_info(uprn_path)
    total_features = info["features"]
    if total_features <= 0:
        # Fallback: force count
        info = pyogrio.read_info(uprn_path, force_feature_count=True)
        total_features = info["features"]

    print(f"  UPRN file: {total_features:,} total features")
    n_chunks = (total_features + chunk_size - 1) // chunk_size

    matched_chunks: list[gpd.GeoDataFrame] = []
    total_matched = 0

    for i in range(n_chunks):
        start = i * chunk_size
        chunk = gpd.read_file(
            uprn_path,
            rows=slice(start, start + chunk_size),
            columns=["UPRN", "geometry"],
        )
        chunk["UPRN"] = chunk["UPRN"].astype("int64")
        matched = chunk[chunk["UPRN"].isin(needed_uprns)]
        total_matched += len(matched)

        if len(matched) > 0:
            matched_chunks.append(matched)

        del chunk
        gc.collect()

        print(
            f"    Chunk {i + 1}/{n_chunks}: "
            f"matched {len(matched):,} "
            f"(running total: {total_matched:,})"
        )

    if not matched_chunks:
        raise ValueError("No UPRN matches found")

    result = pd.concat(matched_chunks, ignore_index=True)
    result = gpd.GeoDataFrame(result, geometry="geometry", crs=info["crs"])
    del matched_chunks
    gc.collect()

    print(f"  Matched {len(result):,} UPRNs with geometry")
    return result


def main() -> None:
    """Main processing pipeline (disk-backed phases)."""
    print("=" * 60)
    print("EPC Data Processing (disk-backed, memory-efficient)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    intermediate_path = OUTPUT_DIR / "_epc_intermediate.parquet"
    cleaned_path = OUTPUT_DIR / "epc_domestic_cleaned.parquet"

    # Phase 1: Read CSVs in batches → intermediate parquet
    print("\n[1/5] Loading EPC certificates (batched)...")
    write_epc_to_parquet(EPC_INPUT_DIR, intermediate_path)

    # Phase 2: Deduplicate by UPRN
    print("\n[2/5] Deduplicating by UPRN...")
    n_dedup = deduplicate_on_disk(intermediate_path, cleaned_path)
    intermediate_path.unlink()  # free disk space
    print(f"  {n_dedup:,} unique properties with UPRN")

    # Phase 3: Read UPRN geometry (chunked, filtered)
    print("\n[3/5] Loading UPRN geometry (chunked)...")
    needed_uprns = set(pd.read_parquet(cleaned_path, columns=["UPRN"])["UPRN"].values)
    print(f"  Need geometry for {len(needed_uprns):,} UPRNs")
    uprn_gdf = read_uprn_filtered(UPRN_PATH, needed_uprns)
    del needed_uprns
    gc.collect()

    # Phase 4: Join EPC to geometry
    print("\n[4/5] Joining EPC data to geometry...")
    epc_df = pd.read_parquet(cleaned_path)
    merged = uprn_gdf[["UPRN", "geometry"]].merge(epc_df, on="UPRN", how="inner")
    epc_spatial = gpd.GeoDataFrame(merged, geometry="geometry", crs=uprn_gdf.crs)
    del epc_df, uprn_gdf, merged
    gc.collect()
    print(f"  Joined: {len(epc_spatial):,} records with geometry")

    # Phase 5: Save outputs (parquet only — GPKG is too slow for this volume)
    print("\n[5/5] Saving output...")
    output_parquet = OUTPUT_DIR / "epc_domestic_spatial.parquet"
    epc_spatial.to_parquet(output_parquet, index=False)
    print(f"  Saved {len(epc_spatial):,} records to {output_parquet}")

    print(f"\n  Columns ({len(epc_spatial.columns)}): {list(epc_spatial.columns)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
