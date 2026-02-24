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
        - longest_axis_length: Length of longest axis (m)
        - fractal_dimension: Perimeter complexity (1=simple, 2=space-filling)
        - rectangularity: Area / oriented envelope area (1=perfect rectangle)
        - square_compactness: Compactness relative to square
        - equivalent_rectangular_index: Closeness to perfect rectangle
        - shape_index: Complexity index (1=circle, higher=complex perimeter)
        - squareness: Mean deviation from 90° corners (0=all right angles)
        - courtyard_area: Internal courtyard area (m²)
        - courtyard_index: Courtyard area / total area
        - facade_ratio: Area / perimeter

    Adjacency (via momepy):
        - shared_wall_length_m: Total length of walls shared with other buildings
        - shared_wall_ratio: shared_wall_length / perimeter
          (0 = detached, ~0.5 = terraced)

    Spatial context (via momepy + libpysal):
        - neighbor_distance: Distance to nearest neighbor (m)
        - neighbors: Count of adjacent buildings (within tolerance)
        - mean_interbuilding_distance: Mean distance to neighbors (m)
        - building_adjacency: Ratio of adjacent to nearby buildings
        - perimeter_wall: Length of perimeter touching other buildings (m)

    Thermal envelope (from geometry + LiDAR height + shared walls):
        - volume_m3: footprint_area × height
        - external_wall_area_m2: perimeter × height × (1 - shared_wall_ratio)
        - envelope_area_m2: roof + floor + external walls (flat roof)
        - surface_to_volume: envelope_area / volume (lower = efficient)
        - form_factor: envelope_area / volume^(2/3) (dimensionless)
"""

import warnings

import geopandas as gpd
import momepy
import numpy as np
import pandas as pd
import shapely
from libpysal.graph import Graph
from tqdm import tqdm

# Configuration
from urban_energy.paths import TEMP_DIR

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
        - longest_axis_length: Length of longest axis (m)
        - fractal_dimension: Perimeter complexity (1-2)
        - rectangularity: Area / oriented envelope area (0-1)
        - square_compactness: Compactness relative to square
        - equivalent_rectangular_index: Closeness to perfect rectangle
        - shape_index: Complexity index (1=circle, higher=complex)
        - squareness: Mean deviation from 90° corners
        - courtyard_area: Internal courtyard area (m²)
        - courtyard_index: Courtyard area / total area
        - facade_ratio: Area / perimeter
    """
    buildings = buildings.copy()

    shape_cols = [
        "orientation",
        "convexity",
        "compactness",
        "elongation",
        "longest_axis_length",
        "fractal_dimension",
        "rectangularity",
        "square_compactness",
        "equivalent_rectangular_index",
        "shape_index",
        "squareness",
        "courtyard_area",
        "courtyard_index",
        "facade_ratio",
    ]

    if len(buildings) == 0:
        for col in shape_cols:
            buildings[col] = []
        return buildings

    # Suppress harmless RuntimeWarnings from shapely/momepy internals:
    # - oriented_envelope: near-degenerate geometries that pass the area filter
    # - arccos: squareness corner angles with cosine slightly outside [-1, 1]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*oriented_envelope.*",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*arccos.*",
            category=RuntimeWarning,
        )

        # --- Existing shape metrics ---

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

        # --- New shape metrics ---

        # Longest axis length (m) — absolute size of longest dimension
        try:
            buildings["longest_axis_length"] = momepy.longest_axis_length(
                buildings
            ).values
        except Exception as e:
            tqdm.write(f"      Warning: momepy.longest_axis_length failed: {e}")
            buildings["longest_axis_length"] = np.nan

        # Fractal dimension: perimeter complexity (1 = simple, 2 = space-filling)
        try:
            buildings["fractal_dimension"] = momepy.fractal_dimension(buildings).values
        except Exception as e:
            tqdm.write(f"      Warning: momepy.fractal_dimension failed: {e}")
            buildings["fractal_dimension"] = np.nan

        # Rectangularity: area / oriented envelope area (1 = perfect rectangle)
        try:
            buildings["rectangularity"] = momepy.rectangularity(buildings).values
        except Exception as e:
            tqdm.write(f"      Warning: momepy.rectangularity failed: {e}")
            buildings["rectangularity"] = np.nan

        # Square compactness: compactness relative to a square
        try:
            buildings["square_compactness"] = momepy.square_compactness(
                buildings
            ).values
        except Exception as e:
            tqdm.write(f"      Warning: momepy.square_compactness failed: {e}")
            buildings["square_compactness"] = np.nan

        # Equivalent rectangular index: closeness to a perfect rectangle
        try:
            buildings["equivalent_rectangular_index"] = (
                momepy.equivalent_rectangular_index(buildings).values
            )
        except Exception as e:
            tqdm.write(
                f"      Warning: momepy.equivalent_rectangular_index failed: {e}"
            )
            buildings["equivalent_rectangular_index"] = np.nan

        # Shape index: complexity (1 = circle, higher = more complex perimeter)
        try:
            buildings["shape_index"] = momepy.shape_index(buildings).values
        except Exception as e:
            tqdm.write(f"      Warning: momepy.shape_index failed: {e}")
            buildings["shape_index"] = np.nan

        # Squareness: mean deviation of corners from 90° (0 = all right angles)
        try:
            buildings["squareness"] = momepy.squareness(buildings).values
        except Exception as e:
            tqdm.write(f"      Warning: momepy.squareness failed: {e}")
            buildings["squareness"] = np.nan

        # Courtyard area: internal courtyard area in m² (0 = no courtyard)
        try:
            cy_area = momepy.courtyard_area(buildings)
            buildings["courtyard_area"] = cy_area.values
        except Exception as e:
            tqdm.write(f"      Warning: momepy.courtyard_area failed: {e}")
            buildings["courtyard_area"] = 0.0
            cy_area = None

        # Courtyard index: courtyard area / total area (0 = solid, >0 = has courtyard)
        try:
            buildings["courtyard_index"] = momepy.courtyard_index(
                buildings, courtyard_area=cy_area
            ).values
        except Exception as e:
            tqdm.write(f"      Warning: momepy.courtyard_index failed: {e}")
            buildings["courtyard_index"] = 0.0

        # Facade ratio: area / perimeter (inverse of perimeter complexity)
        try:
            buildings["facade_ratio"] = momepy.facade_ratio(buildings).values
        except Exception as e:
            tqdm.write(f"      Warning: momepy.facade_ratio failed: {e}")
            buildings["facade_ratio"] = np.nan

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


def compute_spatial_context(
    buildings: gpd.GeoDataFrame, tolerance: float = 1.5
) -> gpd.GeoDataFrame:
    """
    Compute spatial context metrics using momepy + libpysal Graph.

    Captures how buildings relate to their neighbours: isolation (sprawl)
    vs clustering (compact). Uses a fuzzy contiguity graph with an absolute
    buffer (metres) for adjacency — bridging OS Open Map Local cartographic
    gaps — and k-nearest-neighbours for the wider neighbourhood context.

    Parameters
    ----------
    buildings : gpd.GeoDataFrame
        Buildings with geometry and contiguous integer index.
    tolerance : float
        Buffer distance in metres for adjacency graph. Default 1.5m
        bridges OS cartographic gaps (~0.5–1m between adjacent buildings).

    Returns
    -------
    gpd.GeoDataFrame
        Buildings with spatial context metrics added:
        - neighbor_distance: Distance to nearest neighbor (m)
        - neighbors: Count of adjacent buildings (within tolerance)
        - mean_interbuilding_distance: Mean distance to neighbors (m)
        - building_adjacency: Ratio of adjacent to nearby buildings
        - perimeter_wall: Length of perimeter touching other buildings (m)
    """
    buildings = buildings.copy()

    spatial_cols = [
        "neighbor_distance",
        "neighbors",
        "mean_interbuilding_distance",
        "building_adjacency",
        "perimeter_wall",
    ]

    if len(buildings) < 2:
        for col in spatial_cols:
            buildings[col] = 0.0 if len(buildings) > 0 else []
        return buildings

    # Build adjacency graph (fuzzy contiguity with buffer in metres)
    # Uses buffer= (absolute distance) not tolerance= (percentage of bbox),
    # bridging OS Open Map Local cartographic gaps between adjacent buildings.
    try:
        adjacency_graph = Graph.build_fuzzy_contiguity(buildings, buffer=tolerance)
    except Exception as e:
        tqdm.write(f"      Warning: adjacency graph failed: {e}")
        for col in spatial_cols:
            buildings[col] = np.nan
        return buildings

    # Build neighbourhood graph (k=5 nearest neighbours, from centroids)
    try:
        centroids = buildings.copy()
        centroids["geometry"] = buildings.geometry.centroid
        neighborhood_graph = Graph.build_knn(centroids, k=min(5, len(buildings) - 1))
    except Exception as e:
        tqdm.write(f"      Warning: knn graph failed: {e}")
        for col in spatial_cols:
            buildings[col] = np.nan
        return buildings

    # Neighbor distance: distance to nearest neighbour (m)
    try:
        buildings["neighbor_distance"] = momepy.neighbor_distance(
            buildings, neighborhood_graph
        ).values
    except Exception as e:
        tqdm.write(f"      Warning: momepy.neighbor_distance failed: {e}")
        buildings["neighbor_distance"] = np.nan

    # Neighbors: count of adjacent buildings (within tolerance buffer)
    try:
        buildings["neighbors"] = momepy.neighbors(buildings, adjacency_graph).values
    except Exception as e:
        tqdm.write(f"      Warning: momepy.neighbors failed: {e}")
        buildings["neighbors"] = 0

    # Mean interbuilding distance: average distance to neighbourhood (m)
    # Suppress scalar divide warning for isolated buildings (0 adjacency neighbours)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*scalar divide.*",
                category=RuntimeWarning,
            )
            buildings["mean_interbuilding_distance"] = (
                momepy.mean_interbuilding_distance(
                    buildings, adjacency_graph, neighborhood_graph
                ).values
            )
    except Exception as e:
        tqdm.write(f"      Warning: momepy.mean_interbuilding_distance failed: {e}")
        buildings["mean_interbuilding_distance"] = np.nan

    # Building adjacency: ratio of adjacent to nearby buildings (0-1)
    try:
        buildings["building_adjacency"] = momepy.building_adjacency(
            adjacency_graph, neighborhood_graph
        ).values
    except Exception as e:
        tqdm.write(f"      Warning: momepy.building_adjacency failed: {e}")
        buildings["building_adjacency"] = np.nan

    # Perimeter wall: length of perimeter touching/near other buildings (m)
    # Let momepy infer contiguity internally — avoids graph-type mismatch
    try:
        buildings["perimeter_wall"] = momepy.perimeter_wall(buildings).values
    except Exception as e:
        tqdm.write(f"      Warning: momepy.perimeter_wall failed: {e}")
        buildings["perimeter_wall"] = np.nan

    return buildings


def compute_thermal_metrics(
    buildings: gpd.GeoDataFrame,
    height_col: str = "height_median",
) -> gpd.GeoDataFrame:
    """
    Compute thermal envelope metrics from geometry, height, and shared walls.

    Derives surface-to-volume ratio and form factor, which capture thermal
    envelope efficiency (Rode et al., 2014). Requires height data from LiDAR
    and shared wall ratio from momepy.

    Parameters
    ----------
    buildings : gpd.GeoDataFrame
        Buildings with footprint_area_m2, perimeter_m, shared_wall_ratio,
        and a height column.
    height_col : str
        Name of the height column to use. Default ``"height_median"``.

    Returns
    -------
    gpd.GeoDataFrame
        Buildings with thermal metrics added:
        - volume_m3: footprint × height
        - external_wall_area_m2: perimeter × height × (1 - shared_wall_ratio)
        - envelope_area_m2: roof + floor + external walls (flat roof assumption)
        - surface_to_volume: envelope_area / volume (lower = more efficient)
        - form_factor: envelope_area / volume^(2/3) (dimensionless)
    """
    buildings = buildings.copy()

    if len(buildings) == 0 or height_col not in buildings.columns:
        for col in [
            "volume_m3",
            "external_wall_area_m2",
            "envelope_area_m2",
            "surface_to_volume",
            "form_factor",
        ]:
            buildings[col] = np.nan if len(buildings) > 0 else []
        return buildings

    height = pd.to_numeric(buildings[height_col], errors="coerce")

    # Volume
    buildings["volume_m3"] = buildings["footprint_area_m2"] * height

    # External wall area (accounting for shared walls)
    buildings["external_wall_area_m2"] = (
        buildings["perimeter_m"] * height * (1 - buildings["shared_wall_ratio"])
    )

    # Envelope area = roof + floor + external walls (flat roof assumption)
    buildings["envelope_area_m2"] = (
        buildings["footprint_area_m2"]  # roof
        + buildings["footprint_area_m2"]  # floor
        + buildings["external_wall_area_m2"]  # walls
    )

    # Surface-to-volume ratio (lower = more thermally efficient)
    MIN_VOLUME_M3 = 10.0  # ~2m × 2m × 2.5m minimum reasonable building
    MAX_S2V = 5.0  # cap at physically plausible maximum
    MAX_FF = 30.0  # cap form factor (95th percentile ~11)

    valid_volume = buildings["volume_m3"] >= MIN_VOLUME_M3

    buildings["surface_to_volume"] = np.where(
        valid_volume,
        np.clip(
            buildings["envelope_area_m2"] / buildings["volume_m3"],
            0,
            MAX_S2V,
        ),
        np.nan,
    )

    # Form factor: envelope_area / volume^(2/3) — dimensionless, comparable across sizes
    # A perfect cube ≈ 6.0; higher values = less thermally efficient
    buildings["form_factor"] = np.where(
        valid_volume,
        np.clip(
            buildings["envelope_area_m2"] / np.power(buildings["volume_m3"], 2 / 3),
            0,
            MAX_FF,
        ),
        np.nan,
    )

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

    # Drop sub-building artefacts (post boxes, phone boxes, etc.)
    MIN_BUILDING_AREA_M2 = 10.0
    n_before = len(buildings)
    buildings = buildings[buildings.geometry.area >= MIN_BUILDING_AREA_M2].copy()
    n_dropped = n_before - len(buildings)
    if n_dropped > 0 and verbose:
        tqdm.write(
            f"    {bua_name}: dropped {n_dropped} "
            f"sub-{MIN_BUILDING_AREA_M2}m² artefacts"
        )

    if len(buildings) == 0:
        if verbose:
            tqdm.write(f"    {bua_name}: no buildings after filtering")
        return gpd.GeoDataFrame()

    # Validate CRS is projected (metric distances/areas)
    if buildings.crs and buildings.crs.is_geographic:
        msg = f"Buildings CRS must be projected, got {buildings.crs}"
        raise ValueError(msg)

    # Fix invalid geometries
    invalid = ~buildings.geometry.is_valid
    if invalid.any():
        n_invalid = invalid.sum()
        buildings.loc[invalid, "geometry"] = shapely.make_valid(
            buildings.loc[invalid].geometry.values
        )
        if verbose:
            tqdm.write(f"    {bua_name}: fixed {n_invalid} invalid geometries")

    if verbose:
        tqdm.write(f"    {bua_name}: processing {len(buildings):,} buildings...")

    # Reset index for momepy (requires contiguous integer index)
    buildings = buildings.reset_index(drop=True)

    # Compute metrics
    buildings = compute_geometry_metrics(buildings)
    buildings = compute_shape_metrics(buildings)
    buildings = compute_shared_walls(buildings)
    buildings = compute_spatial_context(buildings)

    # Compute thermal envelope metrics (requires height + shared walls)
    height_col = None
    for col in ["height_median", "height_mean", "height"]:
        if col in buildings.columns:
            height_col = col
            break

    if height_col is not None:
        buildings = compute_thermal_metrics(buildings, height_col=height_col)
    else:
        if verbose:
            tqdm.write(f"    {bua_name}: no height column — skipping thermal metrics")

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
    """
    Main processing pipeline.

    Usage:
        uv run python processing/process_morphology.py              # all BUAs
        uv run python processing/process_morphology.py E63011231    # one BUA
        uv run python processing/process_morphology.py E63011231 E63007907 E63012524
    """
    import sys

    print("=" * 60)
    print("Building Morphology Processing")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Check for input data
    if not BUILDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Building heights not found: {BUILDINGS_PATH}\nRun process_lidar.py first."
        )

    if not BOUNDARIES_PATH.exists():
        raise FileNotFoundError(
            f"Boundaries not found: {BOUNDARIES_PATH}\nRun process_boundaries.py first."
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

    # Filter to requested BUA codes if provided
    args = sys.argv[1:]
    if args:
        boundaries = boundaries[boundaries["BUA22CD"].isin(args)].copy()
        missing = set(args) - set(boundaries["BUA22CD"])
        if missing:
            print(f"  WARNING: BUA codes not found: {missing}")
        if len(boundaries) == 0:
            print("  No matching boundaries found.")
            return
        print(f"  Filtered to {len(boundaries)} requested BUA(s)")

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
                        # Geometry
                        "footprint_area_m2",
                        "perimeter_m",
                        # Shape
                        "orientation",
                        "convexity",
                        "compactness",
                        "elongation",
                        "longest_axis_length",
                        "fractal_dimension",
                        "rectangularity",
                        "square_compactness",
                        "equivalent_rectangular_index",
                        "shape_index",
                        "squareness",
                        "courtyard_area",
                        "courtyard_index",
                        "facade_ratio",
                        # Adjacency
                        "shared_wall_length_m",
                        "shared_wall_ratio",
                        # Spatial context
                        "neighbor_distance",
                        "neighbors",
                        "mean_interbuilding_distance",
                        "building_adjacency",
                        "perimeter_wall",
                        # Thermal envelope
                        "volume_m3",
                        "external_wall_area_m2",
                        "envelope_area_m2",
                        "surface_to_volume",
                        "form_factor",
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
    semi = (
        (combined["shared_wall_ratio"] > 0) & (combined["shared_wall_ratio"] < 0.3)
    ).sum()
    terraced = (combined["shared_wall_ratio"] >= 0.3).sum()

    n = len(combined)
    print(f"\n  Approx. detached (ratio=0): {detached:,} ({100 * detached / n:.1f}%)")
    print(f"  Approx. semi (0<ratio<0.3): {semi:,} ({100 * semi / n:.1f}%)")
    print(f"  Approx. terraced (ratio≥0.3): {terraced:,} ({100 * terraced / n:.1f}%)")

    if "surface_to_volume" in combined.columns:
        valid_stv = combined["surface_to_volume"].notna().sum()
        print(f"\nThermal envelope statistics ({valid_stv:,} with valid volume):")
        print(
            f"  S/V ratio: median={combined['surface_to_volume'].median():.3f}, "
            f"mean={combined['surface_to_volume'].mean():.3f}"
        )
        print(
            f"  Form factor: median={combined['form_factor'].median():.2f}, "
            f"mean={combined['form_factor'].mean():.2f}"
        )


if __name__ == "__main__":
    main()
