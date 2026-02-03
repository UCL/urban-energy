"""
Process LiDAR data to derive building heights (streaming approach).

For each built-up area:
1. Download required DSM/DTM tiles via WCS (async, parallel)
2. Mosaic and compute nDSM in memory
3. Extract building heights via zonal statistics
4. Delete downloaded tiles, move to next area

Input:
    - temp/os_open_local/opmplc_gb.gpkg (building layer)
    - temp/boundaries/built_up_areas.gpkg (study area boundaries)

Output:
    - temp/lidar/building_heights.gpkg (building polygons with height stats)
        Columns: id, geometry, height_min, height_max, height_mean,
                 height_median, height_std, height_pixel_count
    - temp/lidar/cache/{BUA22CD}.gpkg (per-boundary cache for resumable runs)
"""

import asyncio
import math
from pathlib import Path

import aiofiles
import aiohttp
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterstats import zonal_stats
from shapely import box
from tqdm import tqdm

# Configuration
BASE_DIR = Path(__file__).parent.parent
TEMP_DIR = BASE_DIR / "temp"
BUILDINGS_PATH = TEMP_DIR / "os_open_local" / "opmplc_gb.gpkg"
BOUNDARIES_PATH = TEMP_DIR / "boundaries" / "built_up_areas.gpkg"
OUTPUT_DIR = TEMP_DIR / "lidar"
CACHE_DIR = OUTPUT_DIR / "cache"
TILE_DIR = OUTPUT_DIR / "tiles"  # Temporary tile storage

# Tile size in metres (5km to match EA standard tiles)
TILE_SIZE = 5000

# WCS configuration
_DSM_DATASET = "9ba4d5ac-d596-445a-9056-dae3ddec0178"
_DTM_DATASET = "13787b9a-26a4-4775-8523-806d13af58fc"
_WCS_BASE = "https://environment.data.gov.uk/geoservices/datasets"

WCS_CONFIG = {
    "dsm": {
        "endpoint": f"{_WCS_BASE}/{_DSM_DATASET}/wcs",
        "coverage_id": f"{_DSM_DATASET}__Lidar_Composite_Elevation_LZ_DSM_1m",
    },
    "dtm": {
        "endpoint": f"{_WCS_BASE}/{_DTM_DATASET}/wcs",
        "coverage_id": f"{_DTM_DATASET}__Lidar_Composite_Elevation_DTM_1m",
    },
}

# Request settings
REQUEST_TIMEOUT = 120  # seconds
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2  # seconds between retries
MAX_CONCURRENT_DOWNLOADS = 8  # Limit concurrent requests
OUTPUT_RESOLUTION = 2  # metres (1 = native 1m, 2 = resampled to 2m)


def generate_tiles_for_bounds(
    bounds: tuple[float, float, float, float],
    tile_size: float = TILE_SIZE,
) -> list[tuple[str, float, float, float, float]]:
    """
    Generate tile grid covering a bounding box.

    Parameters
    ----------
    bounds : tuple
        Bounding box (minx, miny, maxx, maxy).
    tile_size : float
        Tile size in metres.

    Returns
    -------
    list[tuple]
        List of (tile_name, minx, miny, maxx, maxy) tuples.
    """
    minx, miny, maxx, maxy = bounds

    # Snap to tile grid
    grid_minx = math.floor(minx / tile_size) * tile_size
    grid_miny = math.floor(miny / tile_size) * tile_size
    grid_maxx = math.ceil(maxx / tile_size) * tile_size
    grid_maxy = math.ceil(maxy / tile_size) * tile_size

    tiles = []
    y = grid_miny
    while y < grid_maxy:
        x = grid_minx
        while x < grid_maxx:
            tile_x = int(x / 1000)
            tile_y = int(y / 1000)
            tile_name = f"E{tile_x:03d}_N{tile_y:03d}"
            tiles.append((tile_name, x, y, x + tile_size, y + tile_size))
            x += tile_size
        y += tile_size

    return tiles


def filter_tiles_by_geometry(
    tiles: list[tuple[str, float, float, float, float]],
    geometry,
) -> list[tuple[str, float, float, float, float]]:
    """
    Filter tiles to only those intersecting a geometry.

    Parameters
    ----------
    tiles : list
        List of (tile_name, minx, miny, maxx, maxy) tuples.
    geometry : shapely geometry
        Geometry to test intersection against.

    Returns
    -------
    list
        Filtered list of tiles.
    """
    filtered = []
    for tile_name, minx, miny, maxx, maxy in tiles:
        tile_box = box(minx, miny, maxx, maxy)
        if tile_box.intersects(geometry):
            filtered.append((tile_name, minx, miny, maxx, maxy))
    return filtered


async def download_wcs_tile_async(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    endpoint: str,
    coverage_id: str,
    bounds: tuple[float, float, float, float],
    output_path: Path,
) -> Path | None:
    """
    Download a single tile via WCS GetCoverage request (async).

    Parameters
    ----------
    session : aiohttp.ClientSession
        HTTP session.
    semaphore : asyncio.Semaphore
        Concurrency limiter.
    endpoint : str
        WCS endpoint URL.
    coverage_id : str
        Coverage identifier.
    bounds : tuple
        Tile bounds (minx, miny, maxx, maxy) in EPSG:27700.
    output_path : Path
        Path to save the GeoTIFF.

    Returns
    -------
    Path | None
        Path if download succeeded, None otherwise.
    """
    # Skip if already exists
    if output_path.exists():
        return output_path

    minx, miny, maxx, maxy = bounds

    params = {
        "service": "WCS",
        "version": "2.0.1",
        "request": "GetCoverage",
        "CoverageId": coverage_id,
        "format": "image/tiff",
        "subset": [f"E({minx},{maxx})", f"N({miny},{maxy})"],
    }

    # Add scale factor if not native 1m resolution
    if OUTPUT_RESOLUTION > 1:
        scale = 1.0 / OUTPUT_RESOLUTION
        params["SCALEFACTOR"] = scale

    async with semaphore:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
                async with session.get(
                    endpoint, params=params, timeout=timeout
                ) as response:
                    content_type = response.headers.get("content-type", "")
                    if "xml" in content_type.lower():
                        return None  # WCS error or no data

                    if response.status == 200:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        content = await response.read()
                        async with aiofiles.open(output_path, "wb") as f:
                            await f.write(content)
                        return output_path

                    if response.status >= 500:
                        await asyncio.sleep(RETRY_DELAY)
                        continue

                    return None

            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                return None

    return None


async def download_tiles_for_boundary_async(
    tiles: list[tuple[str, float, float, float, float]],
    tile_dir: Path,
) -> tuple[list[Path], list[Path]]:
    """
    Download DSM and DTM tiles for a boundary (async, parallel).

    Parameters
    ----------
    tiles : list
        List of (tile_name, minx, miny, maxx, maxy) tuples.
    tile_dir : Path
        Directory to save tiles.

    Returns
    -------
    tuple[list[Path], list[Path]]
        Lists of downloaded DSM and DTM tile paths.
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

    async with aiohttp.ClientSession() as session:
        tasks = []

        for tile_name, minx, miny, maxx, maxy in tiles:
            bounds = (minx, miny, maxx, maxy)

            # DSM download task
            dsm_path = tile_dir / "dsm" / f"{tile_name}.tif"
            tasks.append(
                download_wcs_tile_async(
                    session,
                    semaphore,
                    WCS_CONFIG["dsm"]["endpoint"],
                    WCS_CONFIG["dsm"]["coverage_id"],
                    bounds,
                    dsm_path,
                )
            )

            # DTM download task
            dtm_path = tile_dir / "dtm" / f"{tile_name}.tif"
            tasks.append(
                download_wcs_tile_async(
                    session,
                    semaphore,
                    WCS_CONFIG["dtm"]["endpoint"],
                    WCS_CONFIG["dtm"]["coverage_id"],
                    bounds,
                    dtm_path,
                )
            )

        results = await asyncio.gather(*tasks)

    # Split results into DSM and DTM (alternating in tasks list)
    # CRITICAL: Only include tiles where BOTH DSM and DTM succeeded
    # to ensure arrays have matching extents for nDSM computation
    dsm_paths = []
    dtm_paths = []
    for i in range(0, len(results), 2):
        dsm_result = results[i]
        dtm_result = results[i + 1] if i + 1 < len(results) else None
        if dsm_result is not None and dtm_result is not None:
            dsm_paths.append(dsm_result)
            dtm_paths.append(dtm_result)

    return dsm_paths, dtm_paths


def delete_tiles(tile_paths: list[Path]) -> None:
    """Delete tile files from disk."""
    for path in tile_paths:
        if path.exists():
            path.unlink()


def mosaic_tiles(tile_paths: list[Path]) -> tuple[np.ndarray, dict]:
    """
    Mosaic multiple tiles into a single array.

    Parameters
    ----------
    tile_paths : list[Path]
        Paths to raster tiles.

    Returns
    -------
    tuple[np.ndarray, dict]
        Mosaicked array and rasterio profile.
    """
    if len(tile_paths) == 1:
        with rasterio.open(tile_paths[0]) as src:
            data = src.read(1).astype(np.float32)
            profile = src.profile.copy()
            # Replace source nodata with NaN
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
            profile["nodata"] = np.nan
            return data, profile

    datasets = [rasterio.open(p) for p in tile_paths]

    try:
        # Pass explicit nodata to avoid float32 representation issues
        mosaic_arr, transform = merge(datasets, nodata=np.nan)
        mosaic_arr = mosaic_arr[0].astype(np.float32)

        profile = datasets[0].profile.copy()
        profile.update(
            width=mosaic_arr.shape[1],
            height=mosaic_arr.shape[0],
            transform=transform,
            nodata=np.nan,
        )

        return mosaic_arr, profile

    finally:
        for ds in datasets:
            ds.close()


def compute_ndsm_in_memory(
    dsm_tiles: list[Path],
    dtm_tiles: list[Path],
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Compute nDSM from DSM and DTM tiles.

    Parameters
    ----------
    dsm_tiles : list[Path]
        Paths to DSM tiles.
    dtm_tiles : list[Path]
        Paths to DTM tiles.
    verbose : bool
        Print diagnostic information about raster values.

    Returns
    -------
    tuple[np.ndarray, dict]
        nDSM array and profile.

    Raises
    ------
    ValueError
        If DSM and DTM arrays have mismatched shapes.
    """
    # Debug: Check first tile directly before mosaic
    if verbose and dsm_tiles:
        with rasterio.open(dsm_tiles[0]) as src:
            data = src.read(1)
            tqdm.write(f"      First DSM tile: shape={data.shape}, dtype={data.dtype}")
            tqdm.write(f"      First DSM tile: min={data.min()}, max={data.max()}")
            tqdm.write(f"      First DSM tile profile nodata={src.nodata}")

    dsm, dsm_profile = mosaic_tiles(dsm_tiles)
    dtm, _ = mosaic_tiles(dtm_tiles)

    if dsm.shape != dtm.shape:
        raise ValueError(
            f"DSM and DTM shape mismatch: DSM {dsm.shape} vs DTM {dtm.shape}. "
            f"Tile count: DSM={len(dsm_tiles)}, DTM={len(dtm_tiles)}"
        )

    # Debug: Check raw raster values
    if verbose:
        tqdm.write(f"      DSM shape={dsm.shape}, dtype={dsm.dtype}")
        tqdm.write(f"      DSM ALL values: min={np.nanmin(dsm)}, max={np.nanmax(dsm)}")
        tqdm.write(f"      DTM ALL values: min={np.nanmin(dtm)}, max={np.nanmax(dtm)}")

    # Compute nDSM - NaN propagates naturally
    valid_mask = ~np.isnan(dsm) & ~np.isnan(dtm)
    ndsm = dsm - dtm

    # Debug: Check nDSM before clipping
    if verbose:
        ndsm_valid = ndsm[valid_mask]
        if len(ndsm_valid) > 0:
            tqdm.write(
                f"      nDSM (pre-clip): min={ndsm_valid.min():.1f}, "
                f"max={ndsm_valid.max():.1f}, mean={ndsm_valid.mean():.1f}"
            )

    # Clip negative values to 0 (below-ground artifacts)
    ndsm = np.where(ndsm < 0, 0, ndsm)

    # Debug: Check nDSM after clipping
    if verbose:
        ndsm_final = ndsm[~np.isnan(ndsm)]
        if len(ndsm_final) > 0:
            tqdm.write(
                f"      nDSM (final): min={ndsm_final.min():.1f}, "
                f"max={ndsm_final.max():.1f}, mean={ndsm_final.mean():.1f}"
            )

    profile = dsm_profile.copy()
    profile.update(dtype=np.float32, nodata=np.nan)

    return ndsm, profile


def compute_heights_from_array(
    buildings: gpd.GeoDataFrame,
    ndsm: np.ndarray,
    profile: dict,
) -> gpd.GeoDataFrame:
    """
    Compute building heights using zonal stats on in-memory array.

    Parameters
    ----------
    buildings : gpd.GeoDataFrame
        Building polygons.
    ndsm : np.ndarray
        nDSM array.
    profile : dict
        Rasterio profile with transform.

    Returns
    -------
    gpd.GeoDataFrame
        Buildings with height statistics and geometry.
    """
    if len(buildings) == 0:
        return gpd.GeoDataFrame()

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(ndsm, 1)

        with memfile.open() as dataset:
            stats = zonal_stats(
                buildings.geometry,
                dataset.read(1),
                affine=dataset.transform,
                stats=["min", "max", "mean", "median", "std", "count"],
                nodata=np.nan,
            )

    stats_df = pd.DataFrame(stats)

    result = gpd.GeoDataFrame(
        {
            "id": buildings["id"].values,
            "geometry": buildings.geometry.values,
            "height_min": stats_df["min"],
            "height_max": stats_df["max"],
            "height_mean": stats_df["mean"],
            "height_median": stats_df["median"],
            "height_std": stats_df["std"],
            "height_pixel_count": stats_df["count"],
        },
        crs=buildings.crs,
    )

    return result


def process_boundary(
    boundary_row: pd.Series,
    buildings_path: Path,
    tile_dir: Path,
    verbose: bool = False,
) -> gpd.GeoDataFrame:
    """
    Process a single built-up area boundary (download, process, cleanup).

    Parameters
    ----------
    boundary_row : pd.Series
        Row from boundaries GeoDataFrame.
    buildings_path : Path
        Path to buildings GeoPackage.
    tile_dir : Path
        Temporary directory for tiles.
    verbose : bool
        Print progress details.

    Returns
    -------
    gpd.GeoDataFrame
        Buildings with height statistics. Heights are null if no LiDAR coverage.
        Empty if no buildings in boundary.
    """
    bua_name = boundary_row.get("BUA22NM", "Unknown")
    boundary_geom = boundary_row.geometry
    bounds = boundary_geom.bounds  # (minx, miny, maxx, maxy)

    # Check for buildings FIRST (before downloading tiles)
    buildings = gpd.read_file(
        buildings_path,
        layer="building",
        bbox=bounds,
    )

    if len(buildings) == 0:
        if verbose:
            tqdm.write(f"    {bua_name}: no buildings in bbox")
        return gpd.GeoDataFrame()  # Empty but cacheable

    # Filter to buildings inside boundary
    buildings = buildings[buildings.intersects(boundary_geom)]

    if len(buildings) == 0:
        if verbose:
            tqdm.write(f"    {bua_name}: no buildings inside boundary")
        return gpd.GeoDataFrame()  # Empty but cacheable

    # Generate and filter tiles for this boundary
    all_tiles = generate_tiles_for_bounds(bounds)
    tiles = filter_tiles_by_geometry(all_tiles, boundary_geom)

    # Helper to create buildings GeoDataFrame with null heights
    def buildings_without_heights() -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {
                "id": buildings["id"].values,
                "geometry": buildings.geometry.values,
                "height_min": None,
                "height_max": None,
                "height_mean": None,
                "height_median": None,
                "height_std": None,
                "height_pixel_count": 0,
            },
            crs=buildings.crs,
        )

    if not tiles:
        if verbose:
            tqdm.write(f"    {bua_name}: {len(buildings):,} bldgs, no LiDAR tiles")
        return buildings_without_heights()

    if verbose:
        tqdm.write(f"    {bua_name}: {len(buildings):,} bldgs, {len(tiles)} tiles...")

    # Download tiles (async)
    dsm_paths, dtm_paths = asyncio.run(
        download_tiles_for_boundary_async(tiles, tile_dir)
    )

    if not dsm_paths or not dtm_paths:
        if verbose:
            tqdm.write(
                f"    {bua_name}: no LiDAR coverage "
                f"(DSM: {len(dsm_paths)}, DTM: {len(dtm_paths)} tiles)"
            )
        return buildings_without_heights()

    if verbose:
        tqdm.write(
            f"    {bua_name}: downloaded {len(dsm_paths)}/{len(tiles)} tile pairs"
        )

    try:
        # Compute nDSM in memory
        ndsm, profile = compute_ndsm_in_memory(dsm_paths, dtm_paths, verbose=verbose)

        # Compute zonal stats
        heights = compute_heights_from_array(buildings, ndsm, profile)

        if verbose:
            tqdm.write(f"    {bua_name}: {len(heights):,} buildings with heights")

        return heights

    finally:
        # Clean up tiles
        delete_tiles(dsm_paths)
        delete_tiles(dtm_paths)


def main() -> None:
    """Main processing pipeline."""
    print("=" * 60)
    print("LiDAR Building Height Processing (Async Streaming)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TILE_DIR.mkdir(parents=True, exist_ok=True)

    # Check for buildings
    if not BUILDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Buildings not found: {BUILDINGS_PATH}\nDownload OS Open Map Local first."
        )

    # Load boundaries and sort by area (largest first)
    print("\n[1/3] Loading built-up area boundaries...")
    boundaries = gpd.read_file(BOUNDARIES_PATH)
    boundaries["_area"] = boundaries.geometry.area
    boundaries = boundaries.sort_values("_area", ascending=False).reset_index(drop=True)
    boundaries = boundaries.drop(columns=["_area"])
    print(f"  {len(boundaries)} built-up areas (sorted largest to smallest)")

    # Check for cached results
    cached_files = list(CACHE_DIR.glob("*.gpkg"))
    cached_ids = {f.stem for f in cached_files}
    print(f"  {len(cached_ids)} boundaries already cached")

    # Process each boundary
    print("\n[2/3] Processing boundaries (async download -> process -> cleanup)...")
    print(f"  Max concurrent downloads: {MAX_CONCURRENT_DOWNLOADS}")
    n_skipped = 0
    n_processed = 0
    n_empty = 0

    for _, row in tqdm(boundaries.iterrows(), total=len(boundaries), desc="  Areas"):
        bua_code = row.get("BUA22CD", "unknown")
        cache_path = CACHE_DIR / f"{bua_code}.gpkg"

        # Skip if already cached
        if bua_code in cached_ids:
            n_skipped += 1
            continue

        try:
            heights = process_boundary(row, BUILDINGS_PATH, TILE_DIR, verbose=True)
            # Always cache result (even if empty)
            if len(heights) > 0:
                heights.to_file(cache_path, driver="GPKG")
            else:
                # Create empty GeoPackage for caching
                gpd.GeoDataFrame(
                    columns=[
                        "id",
                        "geometry",
                        "height_min",
                        "height_max",
                        "height_mean",
                        "height_median",
                        "height_std",
                        "height_pixel_count",
                    ],
                    crs="EPSG:27700",
                ).to_file(cache_path, driver="GPKG")
            if len(heights) > 0:
                n_processed += 1
            else:
                n_empty += 1
        except Exception as e:
            bua_name = row.get("BUA22NM", "Unknown")
            tqdm.write(f"    {bua_name}: ERROR - {e}")
            continue

    print(f"\n  Processed: {n_processed}, Empty: {n_empty}, Cached: {n_skipped}")

    # Combine all cached results
    print("\n[3/3] Combining cached results...")
    all_cache_files = list(CACHE_DIR.glob("*.gpkg"))

    if not all_cache_files:
        print("  Warning: No building heights computed")
        return

    all_heights = [gpd.read_file(f) for f in all_cache_files]
    combined = gpd.GeoDataFrame(
        pd.concat(all_heights, ignore_index=True),
        crs="EPSG:27700",
    )

    # Deduplicate (buildings may appear in multiple BUAs if on boundary)
    combined = combined.sort_values("height_pixel_count", ascending=False)
    combined = combined.drop_duplicates(subset=["id"], keep="first")

    # Save results as GeoPackage
    output_path = OUTPUT_DIR / "building_heights.gpkg"
    combined.to_file(output_path, driver="GPKG")
    print(f"\n  Saved to {output_path}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
    print(f"\nBuildings processed: {len(combined):,}")

    valid_heights = combined["height_max"].dropna()
    print(f"Buildings with valid heights: {len(valid_heights):,}")

    if len(valid_heights) > 0:
        print("\nHeight distribution (max):")
        print(f"  Min:    {valid_heights.min():.1f} m")
        print(f"  Median: {valid_heights.median():.1f} m")
        print(f"  Mean:   {valid_heights.mean():.1f} m")
        print(f"  Max:    {valid_heights.max():.1f} m")


if __name__ == "__main__":
    main()
