"""
Compute building morphology metrics from height-enriched building footprints.

Processes boundary-by-boundary in same order as LiDAR script for consistency.
Supports resumable processing via per-boundary caching.

Input:
    - temp/lidar/building_heights.gpkg (buildings with LiDAR-derived heights)
    - temp/boundaries/built_up_areas.gpkg (study area boundaries)

Output:
    - temp/morphology/buildings_morphology.gpkg (buildings with morphology metrics)
    - temp/morphology/cache/{BUA22CD}.gpkg (per-boundary cache)

Metrics computed:
    Geometry:
        - footprint_area_m2: Building footprint area
        - perimeter_m: Building perimeter

    Shape (via momepy):
        - orientation: Deviation from cardinal directions (0-45°, 0=N-S/E-W aligned)
        - convexity: Area / convex hull area (1=simple, <1=L-shapes/courtyards)
        - compactness: Circular compactness (1=circle, lower=elongated/complex)
        - elongation: Ratio of longest to shortest axis (1=square, higher=elongated)

    Adjacency (via momepy):
        - shared_wall_length_m: Total length of walls shared with other buildings
        - shared_wall_ratio: shared_wall_length / perimeter (0 = detached, ~0.5 = terraced)
"""

from pathlib import Path

import geopandas as gpd
import momepy
import numpy as np
import pandas as pd
from tqdm import tqdm

# Configuration
BASE_DIR = Path(__file__).parent.parent
TEMP_DIR = BASE_DIR / "temp"
BUILDINGS_PATH = TEMP_DIR / "lidar" / "building_heights.gpkg"
BOUNDARIES_PATH = TEMP_DIR / "boundaries" / "built_up_areas.gpkg"
OUTPUT_DIR = TEMP_DIR / "morphology"
CACHE_DIR = OUTPUT_DIR / "cache"


def compute_geometry_metrics(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute basic geometry metrics for buildings.

    Parameters
    ----------
    buildings : gpd.GeoDataFrame
        Buildings with geometry.

    Returns
    -------
    gpd.GeoDataFrame
        Buildings with added geometry metrics.
    """
    buildings = buildings.copy()

    # Footprint area and perimeter
    buildings["footprint_area_m2"] = buildings.geometry.area
    buildings["perimeter_m"] = buildings.geometry.length

    return buildings


def compute_shape_metrics(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute building shape metrics using momepy.

    Parameters
    ----------
    buildings : gpd.GeoDataFrame
        Buildings with geometry.

    Returns
    -------
    gpd.GeoDataFrame
        Buildings with shape metrics added:
        - orientation: Deviation from cardinal directions (0-45°)
        - convexity: Area / convex hull area (envelope complexity)
        - compactness: Circular compactness (shape efficiency)
        - elongation: Longest / shortest axis ratio
    """
    buildings = buildings.copy()

    if len(buildings) == 0:
        buildings["orientation"] = []
        buildings["convexity"] = []
        buildings["compactness"] = []
        buildings["elongation"] = []
        return buildings

    # Orientation: deviation from N-S/E-W (0 = aligned, 45 = diagonal)
    try:
        buildings["orientation"] = momepy.orientation(buildings).values
    except Exception as e:
        tqdm.write(f"      Warning: momepy.orientation failed: {e}")
        buildings["orientation"] = np.nan

    # Convexity: area / convex hull area (1 = convex, <1 = indented/complex)
    try:
        buildings["convexity"] = momepy.convexity(buildings).values
    except Exception as e:
        tqdm.write(f"      Warning: momepy.convexity failed: {e}")
        buildings["convexity"] = np.nan

    # Circular compactness: comparison to circle (1 = circle, <1 = less compact)
    try:
        buildings["compactness"] = momepy.circular_compactness(buildings).values
    except Exception as e:
        tqdm.write(f"      Warning: momepy.circular_compactness failed: {e}")
        buildings["compactness"] = np.nan

    # Elongation: longest axis / shortest axis (1 = square, >1 = elongated)
    try:
        buildings["elongation"] = momepy.elongation(buildings).values
    except Exception as e:
        tqdm.write(f"      Warning: momepy.elongation failed: {e}")
        buildings["elongation"] = np.nan

    return buildings


def compute_shared_walls(
    buildings: gpd.GeoDataFrame, tolerance: float = 1.5
) -> gpd.GeoDataFrame:
    """
    Compute shared wall metrics using momepy.

    OS Open Map Local has intentional gaps (typically 0.5-1m) between adjacent
    buildings for cartographic purposes. We use momepy's non-strict mode with
    a tolerance to detect "virtual" shared walls across these gaps.

    Parameters
    ----------
    buildings : gpd.GeoDataFrame
        Buildings with geometry.
    tolerance : float
        Distance tolerance in metres for detecting adjacency. Default 1.5m
        handles typical OS Open Map Local gaps.

    Returns
    -------
    gpd.GeoDataFrame
        Buildings with shared wall metrics added.
    """
    buildings = buildings.copy()

    if len(buildings) == 0:
        buildings["shared_wall_length_m"] = []
        buildings["shared_wall_ratio"] = []
        return buildings

    if len(buildings) == 1:
        # Single building - no shared walls possible
        buildings["shared_wall_length_m"] = 0.0
        buildings["shared_wall_ratio"] = 0.0
        return buildings

    # Use momepy to compute shared walls with tolerance for OS data gaps
    # strict=False allows detection of nearby (not just touching) buildings
    try:
        shared_walls = momepy.shared_walls(buildings, strict=False, tolerance=tolerance)
        buildings["shared_wall_length_m"] = shared_walls.values
    except Exception as e:
        # If momepy fails, fall back to zero
        tqdm.write(f"      Warning: momepy.shared_walls failed: {e}")
        buildings["shared_wall_length_m"] = 0.0

    # Compute ratio
    buildings["shared_wall_ratio"] = np.where(
        buildings["perimeter_m"] > 0,
        buildings["shared_wall_length_m"] / buildings["perimeter_m"],
        0.0,
    )

    # Clamp ratio to [0, 1]
    buildings["shared_wall_ratio"] = buildings["shared_wall_ratio"].clip(0, 1)

    return buildings


def process_boundary(
    boundary_row: pd.Series,
    all_buildings: gpd.GeoDataFrame,
    verbose: bool = False,
) -> gpd.GeoDataFrame:
    """
    Process buildings within a single boundary.

    Parameters
    ----------
    boundary_row : pd.Series
        Row from boundaries GeoDataFrame.
    all_buildings : gpd.GeoDataFrame
        All buildings with heights (will be filtered to boundary).
    verbose : bool
        Print progress details.

    Returns
    -------
    gpd.GeoDataFrame
        Buildings with morphology metrics. Empty if no buildings in boundary.
    """
    bua_name = boundary_row.get("BUA22NM", "Unknown")
    boundary_geom = boundary_row.geometry
    bounds = boundary_geom.bounds

    # Filter buildings to boundary bbox first (fast)
    buildings = all_buildings.cx[bounds[0] : bounds[2], bounds[1] : bounds[3]].copy()

    if len(buildings) == 0:
        if verbose:
            tqdm.write(f"    {bua_name}: no buildings in bbox")
        return gpd.GeoDataFrame()

    # Then filter to actual boundary (slower but precise)
    buildings = buildings[buildings.intersects(boundary_geom)].copy()

    if len(buildings) == 0:
        if verbose:
            tqdm.write(f"    {bua_name}: no buildings inside boundary")
        return gpd.GeoDataFrame()

    if verbose:
        tqdm.write(f"    {bua_name}: processing {len(buildings):,} buildings...")

    # Reset index for momepy (requires contiguous integer index)
    buildings = buildings.reset_index(drop=True)

    # Compute metrics
    buildings = compute_geometry_metrics(buildings)
    buildings = compute_shape_metrics(buildings)
    buildings = compute_shared_walls(buildings)

    if verbose:
        # Summary stats
        mean_ratio = buildings["shared_wall_ratio"].mean()
        detached_pct = (buildings["shared_wall_ratio"] == 0).mean() * 100
        tqdm.write(
            f"    {bua_name}: mean shared_wall_ratio={mean_ratio:.2f}, "
            f"detached={detached_pct:.0f}%"
        )

    return buildings


def main() -> None:
    """Main processing pipeline."""
    print("=" * 60)
    print("Building Morphology Processing")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Check for input data
    if not BUILDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Building heights not found: {BUILDINGS_PATH}\n"
            "Run process_lidar.py first."
        )

    if not BOUNDARIES_PATH.exists():
        raise FileNotFoundError(
            f"Boundaries not found: {BOUNDARIES_PATH}\n"
            "Run process_boundaries.py first."
        )

    # Load data
    print("\n[1/4] Loading data...")
    print(f"  Loading buildings from {BUILDINGS_PATH}...")
    all_buildings = gpd.read_file(BUILDINGS_PATH)
    print(f"  Loaded {len(all_buildings):,} buildings with heights")

    # Build spatial index for faster bbox queries
    all_buildings.sindex

    print(f"  Loading boundaries from {BOUNDARIES_PATH}...")
    boundaries = gpd.read_file(BOUNDARIES_PATH)

    # Sort by area (largest first) - same order as LiDAR processing
    boundaries["_area"] = boundaries.geometry.area
    boundaries = boundaries.sort_values("_area", ascending=False).reset_index(drop=True)
    boundaries = boundaries.drop(columns=["_area"])
    print(f"  {len(boundaries)} built-up areas (sorted largest to smallest)")

    # Check for cached results
    cached_files = list(CACHE_DIR.glob("*.gpkg"))
    cached_ids = {f.stem for f in cached_files}
    n_to_process = len(boundaries) - len(cached_ids & set(boundaries["BUA22CD"]))
    if cached_ids:
        print(f"  Found {len(cached_ids)} cached files, {n_to_process} remaining")
    else:
        print(f"  No cache found, processing all {len(boundaries)} boundaries")

    # Process each boundary
    print("\n[2/4] Computing morphology metrics...")
    n_processed = 0
    n_empty = 0

    # Filter to only boundaries that need processing
    boundaries_to_process = boundaries[~boundaries["BUA22CD"].isin(cached_ids)]
    n_cached = len(boundaries) - len(boundaries_to_process)

    for _, row in tqdm(
        boundaries_to_process.iterrows(),
        total=len(boundaries),
        initial=n_cached,
        desc="  Areas",
    ):
        bua_code = row.get("BUA22CD", "unknown")
        cache_path = CACHE_DIR / f"{bua_code}.gpkg"

        try:
            result = process_boundary(row, all_buildings, verbose=True)

            # Always cache result (even if empty)
            if len(result) > 0:
                result.to_file(cache_path, driver="GPKG")
                n_processed += 1
            else:
                # Create empty GeoPackage for caching
                gpd.GeoDataFrame(
                    columns=[
                        "id",
                        "geometry",
                        "footprint_area_m2",
                        "perimeter_m",
                        "orientation",
                        "convexity",
                        "compactness",
                        "elongation",
                        "shared_wall_length_m",
                        "shared_wall_ratio",
                    ],
                    crs="EPSG:27700",
                ).to_file(cache_path, driver="GPKG")
                n_empty += 1

        except Exception as e:
            bua_name = row.get("BUA22NM", "Unknown")
            tqdm.write(f"    {bua_name}: ERROR - {e}")
            continue

    print(f"\n  Processed: {n_processed}, Empty: {n_empty}, Cached: {n_cached}")

    # Combine all cached results
    print("\n[3/4] Combining cached results...")
    all_cache_files = list(CACHE_DIR.glob("*.gpkg"))

    if not all_cache_files:
        print("  Warning: No morphology data computed")
        return

    all_results = []
    for f in tqdm(all_cache_files, desc="  Loading cache"):
        gdf = gpd.read_file(f)
        if len(gdf) > 0:
            all_results.append(gdf)

    if not all_results:
        print("  Warning: All cached files are empty")
        return

    combined = gpd.GeoDataFrame(
        pd.concat(all_results, ignore_index=True),
        crs="EPSG:27700",
    )

    # Deduplicate (buildings may appear in multiple BUAs if on boundary)
    n_before = len(combined)
    combined = combined.drop_duplicates(subset=["id"], keep="first")
    n_after = len(combined)
    if n_before != n_after:
        print(f"  Deduplicated: {n_before:,} → {n_after:,} buildings")

    # Save combined results
    print("\n[4/4] Saving results...")
    output_path = OUTPUT_DIR / "buildings_morphology.gpkg"
    combined.to_file(output_path, driver="GPKG")
    print(f"  Saved to {output_path}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
    print(f"\nTotal buildings: {len(combined):,}")

    print("\nGeometry statistics:")
    print(
        f"  Footprint area (m²): median={combined['footprint_area_m2'].median():.0f}, "
        f"mean={combined['footprint_area_m2'].mean():.0f}"
    )

    print("\nShape statistics:")
    print(
        f"  Orientation (°): median={combined['orientation'].median():.1f}, "
        f"mean={combined['orientation'].mean():.1f}"
    )
    print(
        f"  Convexity: median={combined['convexity'].median():.2f}, "
        f"mean={combined['convexity'].mean():.2f}"
    )
    print(
        f"  Compactness: median={combined['compactness'].median():.2f}, "
        f"mean={combined['compactness'].mean():.2f}"
    )
    print(
        f"  Elongation: median={combined['elongation'].median():.2f}, "
        f"mean={combined['elongation'].mean():.2f}"
    )

    print("\nShared wall statistics:")
    print(
        f"  Shared wall ratio: median={combined['shared_wall_ratio'].median():.2f}, "
        f"mean={combined['shared_wall_ratio'].mean():.2f}"
    )

    # Distribution of shared wall ratio
    detached = (combined["shared_wall_ratio"] == 0).sum()
    semi = ((combined["shared_wall_ratio"] > 0) & (combined["shared_wall_ratio"] < 0.3)).sum()
    terraced = (combined["shared_wall_ratio"] >= 0.3).sum()

    print(f"\n  Approx. detached (ratio=0): {detached:,} ({100*detached/len(combined):.1f}%)")
    print(f"  Approx. semi (0<ratio<0.3): {semi:,} ({100*semi/len(combined):.1f}%)")
    print(f"  Approx. terraced (ratio≥0.3): {terraced:,} ({100*terraced/len(combined):.1f}%)")


if __name__ == "__main__":
    main()
