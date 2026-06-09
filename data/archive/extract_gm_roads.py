"""
Extract OS Open Roads network for Greater Manchester + 20km buffer.

Derives the GM Combined Authority boundary by dissolving Census Output Area
geometries for the 10 GM Local Authority Districts, buffers by 20km, and
extracts road_link and road_node layers from the national OS Open Roads
GeoPackage.

Input:  temp/statistics/census_oa_joined.gpkg   (Census OAs with LAD22CD)
        temp/oproad_gpkg_gb/Data/oproad_gb.gpkg (national OS Open Roads)
Output: temp/roads/gm_boundary.gpkg             (dissolved GM boundary)
        temp/roads/gm_boundary_buffered.gpkg    (20km buffered boundary)
        temp/roads/gm_roads.gpkg                (road_link + road_node)

Usage:
    uv run python data/extract_gm_roads.py
"""

from pathlib import Path

import geopandas as gpd
from shapely.ops import unary_union

from urban_energy.paths import DATA_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = DATA_DIR / "roads"

CENSUS_GPKG = DATA_DIR / "statistics" / "census_oa_joined.gpkg"
ROADS_GPKG = DATA_DIR / "oproad_gpkg_gb" / "Data" / "oproad_gb.gpkg"

BUFFER_DISTANCE_M = 20_000  # 20 km

# Greater Manchester Local Authority District codes (LAD22CD)
GM_LAD_CODES: list[str] = [
    "E08000001",  # Bolton
    "E08000002",  # Bury
    "E08000003",  # Manchester
    "E08000004",  # Oldham
    "E08000005",  # Rochdale
    "E08000006",  # Salford
    "E08000007",  # Stockport
    "E08000008",  # Tameside
    "E08000009",  # Trafford
    "E08000010",  # Wigan
]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def load_gm_boundary(census_path: Path) -> gpd.GeoDataFrame:
    """
    Load and dissolve the Greater Manchester boundary from Census OA data.

    Parameters
    ----------
    census_path : Path
        Path to the Census OA GeoPackage containing LAD22CD column.

    Returns
    -------
    gpd.GeoDataFrame
        Single-row GeoDataFrame with the dissolved GM boundary in EPSG:27700.

    Raises
    ------
    FileNotFoundError
        If the Census GeoPackage does not exist.
    ValueError
        If not all 10 GM LADs are found in the data.
    """
    if not census_path.exists():
        raise FileNotFoundError(f"Census GeoPackage not found: {census_path}")

    print("  Loading Census OA geometries...")
    oas = gpd.read_file(census_path, columns=["OA21CD", "LAD22CD"])

    gm_oas = oas[oas["LAD22CD"].isin(GM_LAD_CODES)].copy()
    found_lads = set(gm_oas["LAD22CD"].unique())
    missing = set(GM_LAD_CODES) - found_lads
    if missing:
        raise ValueError(f"Missing GM LADs in census data: {missing}")

    print(f"  Found {len(gm_oas):,} OAs across {len(found_lads)} GM boroughs")

    boundary_geom = unary_union(gm_oas.geometry)
    gm_boundary = gpd.GeoDataFrame(
        {"name": ["Greater Manchester"]},
        geometry=[boundary_geom],
        crs=oas.crs,
    )
    return gm_boundary


def extract_roads_layer(
    gpkg_path: Path,
    layer: str,
    clip_geom: object,
) -> gpd.GeoDataFrame:
    """
    Extract a road layer from the national GeoPackage using spatial filtering.

    Applies a two-stage filter: bounding box query (leveraging the GeoPackage
    R-tree index), then precise geometric intersection.

    Parameters
    ----------
    gpkg_path : Path
        Path to the OS Open Roads national GeoPackage.
    layer : str
        Layer name ("road_link" or "road_node").
    clip_geom : shapely.geometry.base.BaseGeometry
        Geometry defining the extraction area.

    Returns
    -------
    gpd.GeoDataFrame
        Extracted features intersecting the clip geometry.

    Raises
    ------
    FileNotFoundError
        If the GeoPackage does not exist.
    """
    if not gpkg_path.exists():
        raise FileNotFoundError(f"OS Open Roads GeoPackage not found: {gpkg_path}")

    bbox = clip_geom.bounds  # type: ignore[union-attr]
    print(f"  Reading {layer} (bbox filter)...")
    gdf = gpd.read_file(gpkg_path, layer=layer, bbox=bbox)
    print(f"    After bbox: {len(gdf):,} features")

    print(f"  Clipping {layer} to buffered boundary...")
    gdf = gdf[gdf.intersects(clip_geom)].copy()
    print(f"    After clip: {len(gdf):,} features")

    return gdf.reset_index(drop=True)


def main() -> None:
    """Main extraction pipeline."""
    print("=" * 60)
    print("OS Open Roads — Greater Manchester Extraction")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: GM boundary
    print("\n[1/5] Loading GM boundary from census data...")
    gm_boundary = load_gm_boundary(CENSUS_GPKG)
    boundary_path = OUTPUT_DIR / "gm_boundary.gpkg"
    gm_boundary.to_file(boundary_path, driver="GPKG")
    print(f"  Saved: {boundary_path}")

    # Step 2: Buffer
    print(f"\n[2/5] Buffering boundary ({BUFFER_DISTANCE_M / 1000:.0f} km)...")
    buffered_geom = gm_boundary.geometry.iloc[0].buffer(BUFFER_DISTANCE_M)
    gm_buffered = gpd.GeoDataFrame(
        {"name": ["Greater Manchester (20km buffer)"]},
        geometry=[buffered_geom],
        crs=gm_boundary.crs,
    )
    buffered_path = OUTPUT_DIR / "gm_boundary_buffered.gpkg"
    gm_buffered.to_file(buffered_path, driver="GPKG")
    print(f"  Saved: {buffered_path}")

    # Step 3: Extract road_link
    print("\n[3/5] Extracting road_link layer...")
    road_link = extract_roads_layer(ROADS_GPKG, "road_link", buffered_geom)

    # Step 4: Extract road_node
    print("\n[4/5] Extracting road_node layer...")
    road_node = extract_roads_layer(ROADS_GPKG, "road_node", buffered_geom)

    # Step 5: Save
    print("\n[5/5] Saving output...")
    output_path = OUTPUT_DIR / "gm_roads.gpkg"
    road_link.to_file(output_path, driver="GPKG", layer="road_link")
    road_node.to_file(output_path, driver="GPKG", layer="road_node", mode="a")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")

    # Summary
    print("\n" + "=" * 60)
    print("Extraction complete!")
    print("=" * 60)
    print(f"  road_link: {len(road_link):,} features")
    print(f"  road_node: {len(road_node):,} features")
    print(f"  Output:    {output_path}")


if __name__ == "__main__":
    main()
