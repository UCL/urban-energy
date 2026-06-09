"""
Shared plumbing for the national OA pipeline.

These paths, column constants, and the Stage 1 morphology cache-loader were
previously imported by ``pipeline_oa.py`` from ``archive/pipeline_lsoa.py`` —
a live dependency on an otherwise-frozen module. They are extracted here so the
live pipeline imports only from live modules and ``archive/`` is genuinely
archive.

The morphology inputs (``buildings``/``morphology_cache``) are part of the
deferred LiDAR/morphology path: ``run_stage1_morphology`` returns ``None`` when
no cache is present, and the downstream stages skip the (optional) morphology
columns. See REPRODUCTION.md for what is in the load-bearing rebuild.
"""

import geopandas as gpd
import pandas as pd

from urban_energy.paths import DATA_DIR, latest_uprn_gpkg

# Input paths (shared by pipeline_oa.py). Entries for deferred/optional sources
# (buildings, morphology_cache, scaling) are retained but their consumers guard
# on ``.exists()``, so an absent file is skipped rather than fatal.
PATHS = {
    "boundaries": DATA_DIR / "boundaries" / "built_up_areas.gpkg",
    "buildings": DATA_DIR / "lidar" / "building_heights.gpkg",
    "morphology_cache": DATA_DIR / "morphology" / "cache",
    "census": DATA_DIR / "statistics" / "census_oa_joined.gpkg",
    "epc": DATA_DIR / "epc" / "epc_domestic_spatial.parquet",
    "uprn": (
        latest_uprn_gpkg()
        or DATA_DIR / "osopenuprn_202601_gpkg" / "osopenuprn_202601.gpkg"
    ),
    "roads": DATA_DIR / "oproad_gpkg_gb" / "Data" / "oproad_gb.gpkg",
    "fsa": DATA_DIR / "fsa" / "fsa_establishments.gpkg",
    "greenspace": DATA_DIR / "opgrsp_gpkg_gb" / "Data" / "opgrsp_gb.gpkg",
    "transport": DATA_DIR / "transport" / "naptan_england.gpkg",
    "schools": DATA_DIR / "education" / "gias_schools.gpkg",
    "health": DATA_DIR / "health" / "nhs_facilities.gpkg",
}

# Building-physics columns aggregated to OA (deferred LiDAR/morphology path).
_MORPH_SUM_COLS = ["footprint_area_m2", "volume_m3", "envelope_area_m2"]
_MORPH_MEAN_COLS = ["surface_to_volume", "height_mean", "form_factor"]

# Census column names needed for OA area/population derivation.
_TS001_POP = "ts001_Residence type: Lives in a household; measures: Value"
_TS006_DENSITY = (
    "ts006_Population Density: Persons per square kilometre; measures: Value"
)

# Maps EPC CONSTRUCTION_AGE_BAND strings → midpoint year (sole source of
# building age; see the consumption audit — load-bearing for the Form model,
# the OLS controls, and the Atlas build-year slider).
ERA_MAP: dict[str, int] = {
    "England and Wales: before 1900": 1880,
    "England and Wales: 1900-1929": 1915,
    "England and Wales: 1930-1949": 1940,
    "England and Wales: 1950-1966": 1958,
    "England and Wales: 1967-1975": 1971,
    "England and Wales: 1976-1982": 1979,
    "England and Wales: 1983-1990": 1987,
    "England and Wales: 1991-1995": 1993,
    "England and Wales: 1996-2002": 1999,
    "England and Wales: 2003-2006": 2005,
    "England and Wales: 2007 onwards": 2010,
    "England and Wales: 2007-2011": 2009,
    "England and Wales: 2012 onwards": 2016,
    "England and Wales: 2012-2021": 2016,
    "England and Wales: 2022 onwards": 2023,
}


def run_stage1_morphology(
    boundaries: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame | None:
    """
    Stage 1: Building Morphology.

    Load cached morphology results for the given boundaries. Returns ``None``
    if any boundary lacks a morphology cache (the deferred LiDAR/morphology
    path), in which case downstream stages skip the optional morphology
    columns and still produce a complete OA dataset.

    Parameters
    ----------
    boundaries : geopandas.GeoDataFrame
        BUA boundaries with ``BUA22CD`` and ``BUA22NM`` columns.

    Returns
    -------
    geopandas.GeoDataFrame or None
        Per-building morphology for the boundaries, or ``None`` if no cache.
    """
    print()
    print("=" * 60)
    print("STAGE 1: BUILDING MORPHOLOGY")
    print("=" * 60)

    cache_dir = PATHS["morphology_cache"]
    results = []

    for _, boundary in boundaries.iterrows():
        bua_code = boundary["BUA22CD"]
        bua_name = boundary["BUA22NM"]
        cache_file = cache_dir / f"{bua_code}.gpkg"

        if cache_file.exists():
            gdf = gpd.read_file(cache_file)
            print(f"  {bua_code} ({bua_name}): {len(gdf)} buildings from cache")
            if len(gdf) > 0:
                results.append(gdf)
        else:
            print(f"  {bua_code} ({bua_name}): No cache found - needs processing")
            return None

    if not results:
        print("  No buildings found in boundaries!")
        return None

    buildings = pd.concat(results, ignore_index=True)
    buildings = gpd.GeoDataFrame(buildings, crs=results[0].crs)

    # Validate required columns
    required_cols = [
        "footprint_area_m2",
        "perimeter_m",
        "orientation",
        "convexity",
        "compactness",
        "elongation",
        "shared_wall_length_m",
        "shared_wall_ratio",
    ]
    thermal_cols = [
        "volume_m3",
        "external_wall_area_m2",
        "envelope_area_m2",
        "surface_to_volume",
        "form_factor",
    ]

    missing_cols = [c for c in required_cols if c not in buildings.columns]
    if missing_cols:
        print(f"  WARNING: Missing morphology columns: {missing_cols}")
    else:
        print("  ✓ All morphology columns present")

    missing_thermal = [c for c in thermal_cols if c not in buildings.columns]
    if missing_thermal:
        print(f"  WARNING: Missing thermal columns: {missing_thermal}")
        print("    (Re-run process_morphology.py to compute these)")
    else:
        valid_stv = buildings["surface_to_volume"].notna().sum()
        mean_stv = buildings["surface_to_volume"].mean()
        print(
            f"  ✓ Thermal metrics present: "
            f"{valid_stv:,} with valid S/V (mean={mean_stv:.3f})"
        )

    print(f"\n  Total buildings: {len(buildings)}")

    return buildings
