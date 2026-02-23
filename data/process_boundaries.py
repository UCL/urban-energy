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
from shapely import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

# Configuration
from urban_energy.paths import TEMP_DIR

INPUT_PATH = (
    TEMP_DIR / "OS_Open_Built_Up_Areas_GeoPackage" / "os_open_built_up_areas.gpkg"
)
OUTPUT_DIR = TEMP_DIR / "boundaries"

# Buffer distance (metres) for merging adjacent built-up areas
MERGE_BUFFER_DISTANCE = 100

# Simplification tolerance (metres) for reducing vertex count
SIMPLIFY_TOLERANCE = 10


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
    print(f"  Columns: {list(gdf.columns)}")

    # Standardise column names (OS uses various conventions)
    col_mapping = {}
    for col in gdf.columns:
        col_lower = col.lower()
        if "bua" in col_lower and "cd" in col_lower:
            col_mapping[col] = "BUA22CD"
        elif "bua" in col_lower and "nm" in col_lower:
            col_mapping[col] = "BUA22NM"
        # Also check for gsscode/name patterns used in some OS releases
        elif col_lower in ("gsscode", "code"):
            col_mapping[col] = "BUA22CD"
        elif col_lower in ("name", "name1_text"):
            col_mapping[col] = "BUA22NM"

    if col_mapping:
        print(f"  Renaming columns: {col_mapping}")
        gdf = gdf.rename(columns=col_mapping)

    # Validate required columns exist
    required = ["BUA22CD", "BUA22NM"]
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise KeyError(
            f"Required columns not found: {missing}\n"
            f"Available columns: {list(gdf.columns)}\n"
            f"Please check the OS Open Built Up Areas data format."
        )

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


def _count_vertices(geom: Polygon | MultiPolygon) -> int:
    """Count vertices in a Polygon or MultiPolygon."""
    if geom.geom_type == "Polygon":
        return len(geom.exterior.coords)
    elif geom.geom_type == "MultiPolygon":
        return sum(len(p.exterior.coords) for p in geom.geoms)
    return 0


def _remove_holes(geom: Polygon | MultiPolygon) -> Polygon | MultiPolygon:
    """Remove interior holes from a Polygon or MultiPolygon."""
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    elif geom.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    return geom


def simplify_geometries(
    gdf: gpd.GeoDataFrame, tolerance: float = SIMPLIFY_TOLERANCE
) -> gpd.GeoDataFrame:
    """
    Simplify geometries by removing holes and reducing vertices.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with polygon geometries.
    tolerance : float
        Simplification tolerance in CRS units (metres for BNG).

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with simplified geometries.
    """
    original_vertices = gdf.geometry.apply(_count_vertices).sum()

    # Remove interior holes (keep only exterior ring)
    gdf["geometry"] = gdf.geometry.apply(_remove_holes)

    # Simplify using Douglas-Peucker algorithm
    gdf["geometry"] = gdf.geometry.simplify(tolerance=tolerance)

    simplified_vertices = gdf.geometry.apply(_count_vertices).sum()

    print(f"  Removed holes, simplified (tolerance={tolerance}m)")
    print(f"  Vertices: {original_vertices:,} -> {simplified_vertices:,}")

    return gdf


def remove_contained_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Remove polygons that are fully contained within larger polygons.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with polygon geometries.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with nested polygons removed.
    """
    # Sort by area descending so we check smaller against larger
    gdf = gdf.copy()
    gdf["_area"] = gdf.geometry.area
    gdf = gdf.sort_values("_area", ascending=False).reset_index(drop=True)

    # Build spatial index
    sindex = gdf.sindex

    to_remove = set()
    for idx, row in gdf.iterrows():
        if idx in to_remove:
            continue

        # Find candidates that might be contained (smaller polygons)
        candidates = list(sindex.intersection(row.geometry.bounds))

        for candidate_idx in candidates:
            if candidate_idx <= idx or candidate_idx in to_remove:
                continue

            candidate_geom = gdf.loc[candidate_idx, "geometry"]
            if row.geometry.contains(candidate_geom):
                to_remove.add(candidate_idx)

    gdf = gdf.drop(index=list(to_remove)).drop(columns=["_area"])
    print(f"  Removed {len(to_remove):,} contained polygons")

    return gdf.reset_index(drop=True)


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
    print("\n[1/6] Loading built-up areas...")
    bua = load_built_up_areas(INPUT_PATH)

    # Filter to England only (project scope; LiDAR coverage is England only)
    n_before = len(bua)
    bua = bua[bua["BUA22CD"].str.startswith("E")]
    print(f"  Filtered to England: {n_before:,} -> {len(bua):,}")

    # Validate geometries
    print("\n[2/6] Validating geometries...")
    bua = validate_geometries(bua)

    # Simplify geometries
    print("\n[3/6] Simplifying geometries...")
    bua = simplify_geometries(bua)

    # Remove nested polygons
    print("\n[4/6] Removing contained polygons...")
    bua = remove_contained_polygons(bua)

    # Save cleaned individual areas
    print("\n[5/6] Saving individual built-up areas...")
    individual_path = OUTPUT_DIR / "built_up_areas.gpkg"
    bua.to_file(individual_path, driver="GPKG")
    print(f"  Saved to {individual_path}")

    # Merge adjacent areas
    print("\n[6/6] Creating merged conurbations...")
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
