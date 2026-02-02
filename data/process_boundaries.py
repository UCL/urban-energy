"""
Process OS Open Built Up Areas into analysis-ready boundaries.

Reads the raw OS Open Built Up Areas download and produces:
- Cleaned individual built-up area polygons
- Merged conurbations (adjacent areas combined)

Input: temp/OS_Open_Built_Up_Areas_GeoPackage/os_open_built_up_areas.gpkg
Output: temp/boundaries/
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from shapely.validation import make_valid

# Configuration
BASE_DIR = Path(__file__).parent.parent
TEMP_DIR = BASE_DIR / "temp"
INPUT_PATH = (
    TEMP_DIR / "OS_Open_Built_Up_Areas_GeoPackage" / "os_open_built_up_areas.gpkg"
)
OUTPUT_DIR = TEMP_DIR / "boundaries"

# Buffer distance (metres) for merging adjacent built-up areas
MERGE_BUFFER_DISTANCE = 100


def load_built_up_areas(path: Path) -> gpd.GeoDataFrame:
    """
    Load OS Open Built Up Areas from GeoPackage.

    Parameters
    ----------
    path : Path
        Path to the GeoPackage file.

    Returns
    -------
    gpd.GeoDataFrame
        Built-up areas with standardised column names.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            f"Download from: https://osdatahub.os.uk/downloads/open/BuiltUpAreas"
        )

    print(f"Loading built-up areas from {path}")
    gdf = gpd.read_file(path)
    print(f"  Loaded {len(gdf):,} polygons")

    # Standardise column names (OS uses various conventions)
    col_mapping = {}
    for col in gdf.columns:
        col_lower = col.lower()
        if "bua" in col_lower and "cd" in col_lower:
            col_mapping[col] = "BUA22CD"
        elif "bua" in col_lower and "nm" in col_lower:
            col_mapping[col] = "BUA22NM"

    if col_mapping:
        gdf = gdf.rename(columns=col_mapping)

    return gdf


def validate_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Validate and fix invalid geometries.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with valid geometries.
    """
    invalid_count = (~gdf.geometry.is_valid).sum()
    if invalid_count > 0:
        print(f"  Fixing {invalid_count} invalid geometries")
        gdf["geometry"] = gdf.geometry.apply(make_valid)

    return gdf


def merge_adjacent_areas(
    gdf: gpd.GeoDataFrame, buffer_distance: float = MERGE_BUFFER_DISTANCE
) -> gpd.GeoDataFrame:
    """
    Merge adjacent built-up areas into conurbations.

    Uses a buffer-dissolve-negative buffer approach to merge polygons
    that are within buffer_distance of each other.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Built-up areas in projected CRS (metres).
    buffer_distance : float
        Distance in metres to buffer before dissolving.

    Returns
    -------
    gpd.GeoDataFrame
        Merged conurbations with area statistics.
    """
    print(f"Merging adjacent areas (buffer={buffer_distance}m)...")

    # Buffer all geometries
    buffered = gdf.geometry.buffer(buffer_distance)

    # Union all buffered geometries
    merged_geom = unary_union(buffered)

    # Negative buffer to restore original boundaries
    if merged_geom.geom_type == "MultiPolygon":
        restored = [geom.buffer(-buffer_distance) for geom in merged_geom.geoms]
    else:
        restored = [merged_geom.buffer(-buffer_distance)]

    # Filter out any empty geometries that result from small areas disappearing
    restored = [g for g in restored if not g.is_empty and g.area > 0]

    # Create new GeoDataFrame
    merged_gdf = gpd.GeoDataFrame(
        {"conurbation_id": range(len(restored))},
        geometry=restored,
        crs=gdf.crs,
    )

    # Calculate area in hectares
    merged_gdf["area_ha"] = merged_gdf.geometry.area / 10_000

    # Spatial join to get constituent BUA names
    joined = gpd.sjoin(
        gdf[["BUA22CD", "BUA22NM", "geometry"]], merged_gdf, predicate="within"
    )

    # Aggregate names for each conurbation
    name_agg = (
        joined.groupby("conurbation_id")["BUA22NM"]
        .agg(lambda x: "; ".join(sorted(set(x))))
        .reset_index()
        .rename(columns={"BUA22NM": "constituent_names"})
    )

    # Count constituents
    count_agg = (
        joined.groupby("conurbation_id")["BUA22CD"]
        .count()
        .reset_index()
        .rename(columns={"BUA22CD": "n_areas"})
    )

    merged_gdf = merged_gdf.merge(name_agg, on="conurbation_id", how="left")
    merged_gdf = merged_gdf.merge(count_agg, on="conurbation_id", how="left")

    # Generate a simple name from the largest constituent
    def get_primary_name(names: str | None) -> str:
        if pd.isna(names) or names is None:
            return "Unknown"
        name_list = names.split("; ")
        # Return the longest name (often the main city)
        return str(max(name_list, key=len))

    merged_gdf["primary_name"] = merged_gdf["constituent_names"].apply(get_primary_name)

    print(f"  Created {len(merged_gdf):,} merged conurbations")

    # Ensure return type is GeoDataFrame
    result: gpd.GeoDataFrame = merged_gdf  # type: ignore[assignment]
    return result


def main() -> None:
    """Main processing pipeline."""
    print("=" * 60)
    print("OS Open Built Up Areas Processing")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/4] Loading built-up areas...")
    bua = load_built_up_areas(INPUT_PATH)

    # Validate geometries
    print("\n[2/4] Validating geometries...")
    bua = validate_geometries(bua)

    # Save cleaned individual areas
    print("\n[3/4] Saving individual built-up areas...")
    individual_path = OUTPUT_DIR / "built_up_areas.gpkg"
    bua.to_file(individual_path, driver="GPKG")
    print(f"  Saved to {individual_path}")

    # Also save as parquet (faster for non-spatial analysis)
    parquet_path = OUTPUT_DIR / "built_up_areas.parquet"
    bua.to_parquet(parquet_path)
    print(f"  Saved to {parquet_path}")

    # Merge adjacent areas
    print("\n[4/4] Creating merged conurbations...")
    merged = merge_adjacent_areas(bua)

    merged_path = OUTPUT_DIR / "built_up_areas_merged.gpkg"
    merged.to_file(merged_path, driver="GPKG")
    print(f"  Saved to {merged_path}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
    print(f"\nInput:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"\nIndividual areas: {len(bua):,}")
    print(f"Merged conurbations: {len(merged):,}")
    print("\nLargest conurbations by area:")

    top_5 = merged.nlargest(5, "area_ha")[["primary_name", "n_areas", "area_ha"]]
    for _, row in top_5.iterrows():
        name = row["primary_name"]
        n = row["n_areas"]
        area = row["area_ha"]
        print(f"  {name}: {n} areas, {area:,.0f} ha")


if __name__ == "__main__":
    main()
