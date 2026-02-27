"""
LSOA-level processing pipeline for urban energy analysis.

Three stages:
  Stage 1: Building morphology (from cached LiDAR + momepy metrics)
  Stage 2: Network analysis (cityseer centrality + accessibility)
  Stage 3: LSOA aggregation (transient UPRN joins → aggregate to LSOA polygons)

Output structure:
    processing/{city}/lsoa_integrated.gpkg   — per-city LSOA polygons
    processing/combined/lsoa_integrated.gpkg — all cities merged

Usage:
    uv run python processing/pipeline_lsoa.py                  # all cities
    uv run python processing/pipeline_lsoa.py manchester       # one city
    uv run python processing/pipeline_lsoa.py york cambridge   # subset
"""

import gc
import sys
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.spatial import cKDTree  # type: ignore[unresolved-import]

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

from urban_energy.paths import TEMP_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = TEMP_DIR / "processing"

# Study cities: BUA22CD -> short name
CITIES: dict[str, str] = {
    # Large cities (original 8)
    "E63008401": "manchester",
    "E63012168": "bristol",
    "E63010901": "milton_keynes",
    "E63007706": "york",
    "E63010556": "cambridge",
    # Smaller towns (typological contrast)
    "E63011231": "stevenage",  # new town (sprawl control)
    "E63007907": "burnley",  # northern mill town (dense terraced)
    "E63012524": "canterbury",  # historic compact city
    # Expansion — large provincial cities for statistical power
    "E63010038": "birmingham",
    "E63007883": "leeds",
    "E63008489": "sheffield",
    "E63008477": "liverpool",
    "E63007169": "newcastle",
    "E63009088": "nottingham",
    "E63009743": "leicester",
    # Southern/coastal for regional balance
    "E63014082": "plymouth",
    "E63013524": "southampton",
    "E63013666": "brighton",
}

# Input paths
PATHS = {
    "boundaries": TEMP_DIR / "boundaries" / "built_up_areas.gpkg",
    "buildings": TEMP_DIR / "lidar" / "building_heights.gpkg",
    "morphology_cache": TEMP_DIR / "morphology" / "cache",
    "census": TEMP_DIR / "statistics" / "census_oa_joined.gpkg",
    "epc": TEMP_DIR / "epc" / "epc_domestic_spatial.parquet",
    "uprn": TEMP_DIR / "osopenuprn_202601_gpkg" / "osopenuprn_202601.gpkg",
    "roads": TEMP_DIR / "oproad_gpkg_gb" / "Data" / "oproad_gb.gpkg",
    "fsa": TEMP_DIR / "fsa" / "fsa_establishments.gpkg",
    "greenspace": TEMP_DIR / "opgrsp_gpkg_gb" / "Data" / "opgrsp_gb.gpkg",
    "transport": TEMP_DIR / "transport" / "naptan_england.gpkg",
    "energy_stats": TEMP_DIR / "statistics" / "lsoa_energy_consumption.parquet",
    "scaling": TEMP_DIR / "statistics" / "lsoa_scaling.parquet",
    "schools": TEMP_DIR / "education" / "gias_schools.gpkg",
    "health": TEMP_DIR / "health" / "nhs_facilities.gpkg",
}

# Building physics columns needed for LSOA aggregation
_MORPH_SUM_COLS = ["footprint_area_m2", "volume_m3", "envelope_area_m2"]
_MORPH_MEAN_COLS = ["surface_to_volume", "height_mean", "form_factor"]

# Census column names needed for OA area derivation
_TS001_POP = "ts001_Residence type: Lives in a household; measures: Value"
_TS006_DENSITY = (
    "ts006_Population Density: Persons per square kilometre; measures: Value"
)

# Maps EPC CONSTRUCTION_AGE_BAND strings → midpoint year
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


# ---------------------------------------------------------------------------
# Stage 1: Building Morphology
# ---------------------------------------------------------------------------


def check_inputs() -> dict[str, bool]:
    """Check that all required input files exist."""
    print("=" * 60)
    print("CHECKING INPUTS")
    print("=" * 60)

    status = {}
    for name, path in PATHS.items():
        exists = path.exists()
        status[name] = exists
        icon = "✓" if exists else "✗"
        print(f"  {icon} {name}: {path}")

    print()
    missing = [k for k, v in status.items() if not v]
    if missing:
        print(f"Missing inputs: {missing}")
    else:
        print("All inputs found.")

    return status


def load_boundary(bua_code: str) -> gpd.GeoDataFrame:
    """Load a single BUA boundary geometry."""
    boundaries = gpd.read_file(PATHS["boundaries"])
    boundary = boundaries[boundaries["BUA22CD"] == bua_code].copy()

    if len(boundary) == 0:
        msg = f"BUA code {bua_code} not found in boundaries"
        raise ValueError(msg)

    row = boundary.iloc[0]
    area_km2 = row.geometry.area / 1e6
    print(f"  Boundary: {row['BUA22CD']} ({row['BUA22NM']}): {area_km2:.2f} km²")

    return boundary


def run_stage1_morphology(boundaries: gpd.GeoDataFrame) -> gpd.GeoDataFrame | None:
    """
    Stage 1: Building Morphology.

    Load cached morphology results for boundaries.
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
        print("  No buildings found in test boundaries!")
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


# ---------------------------------------------------------------------------
# Stage 2: Network Analysis
# ---------------------------------------------------------------------------


def run_stage2_network(
    boundaries: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame | None = None,
    test_mode: bool = True,
) -> gpd.GeoDataFrame | None:
    """
    Stage 2: Network Analysis.

    Compute network centrality, building statistics, and accessibility metrics.

    Parameters
    ----------
    boundaries : gpd.GeoDataFrame
        Study area boundaries.
    buildings : gpd.GeoDataFrame | None
        Buildings with morphology metrics for statistical aggregation.
    test_mode : bool
        If True, use reduced buffer and centrality distances for faster testing.
    """
    from cityseer.metrics import layers, networks
    from cityseer.tools import graphs, io

    print()
    print("=" * 60)
    print("STAGE 2: NETWORK ANALYSIS")
    print("=" * 60)

    # Analysis parameters - distances in meters (based on ~80m/min walking speed)
    # 5min=400m, 10min=800m, 20min=1600m, 40min=3200m, 60min=4800m, 120min=9600m
    if test_mode:
        CENTRALITY_DISTANCES = [800, 1600]
        ACCESSIBILITY_DISTANCES = [400, 800, 1600]
        print("  [TEST MODE: reduced distances for faster processing]")
    else:
        CENTRALITY_DISTANCES = [800, 1600, 3200, 4800, 9600]
        ACCESSIBILITY_DISTANCES = [400, 800, 1600, 4800]

    max_distance = max(max(CENTRALITY_DISTANCES), max(ACCESSIBILITY_DISTANCES))
    buffer_m = int(max_distance * 1.1)
    print(f"  Buffer: {buffer_m}m (max distance + 10% margin)")

    combined_bounds = boundaries.union_all()
    buffered_bounds = combined_bounds.buffer(buffer_m)

    # Load network from OS Open Roads
    print("\n  Loading network from OS Open Roads...")
    bbox = buffered_bounds.bounds
    nx_graph = io.nx_from_open_roads(
        PATHS["roads"],
        target_bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
    )
    print(
        f"    Network: {nx_graph.number_of_nodes()} nodes, "
        f"{nx_graph.number_of_edges()} edges"
    )

    # Prepare network for analysis
    print("  Preparing network...")
    nx_graph = graphs.nx_remove_filler_nodes(nx_graph)
    nx_graph = graphs.nx_remove_dangling_nodes(nx_graph)
    print(
        f"    After cleanup: {nx_graph.number_of_nodes()} nodes, "
        f"{nx_graph.number_of_edges()} edges"
    )

    # Mark nodes live=True (within study boundary) or live=False (buffer only).
    # Cityseer computes metrics only for live nodes; buffer nodes provide network
    # context for accurate catchment computation at the boundary edge.
    from shapely.geometry import Point as _Point

    for node_key, node_data in nx_graph.nodes(data=True):
        x, y = node_data["x"], node_data["y"]
        node_data["live"] = combined_bounds.contains(_Point(x, y))

    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(nx_graph)
    n_live = nodes_gdf["live"].sum()
    print(
        f"    Network structure built: {len(nodes_gdf)} nodes "
        f"({n_live} live / {len(nodes_gdf) - n_live} buffer)"
    )

    # Compute centrality
    print(f"\n  Computing centrality (distances={CENTRALITY_DISTANCES})...")
    nodes_gdf = networks.node_centrality_shortest_adaptive(
        network_structure,
        nodes_gdf,
        distances=CENTRALITY_DISTANCES,
        compute_closeness=True,
        compute_betweenness=True,
    )
    centrality_cols = [c for c in nodes_gdf.columns if "beta" in c or "harmonic" in c]
    print(f"    Centrality columns: {len(centrality_cols)}")

    # Compute building statistics over the network at accessibility distances
    STATS_DISTANCES = ACCESSIBILITY_DISTANCES
    ASSUMED_STOREY_HEIGHT_M = 3.0
    if buildings is not None and len(buildings) > 0:
        print(f"\n  Computing building stats (distances={STATS_DISTANCES})...")
        buildings_pts = buildings.copy()
        buildings_pts["geometry"] = buildings_pts.geometry.centroid
        buildings_in_buffer = buildings_pts[
            buildings_pts.intersects(buffered_bounds)
        ].copy()
        print(f"    Buildings in buffer: {len(buildings_in_buffer)}")

        height_cols = [
            c for c in buildings_in_buffer.columns if c.startswith("height_")
        ]

        # Convert all numeric columns (may be Arrow-backed strings)
        numeric_cols = ["footprint_area_m2", "shared_wall_ratio"] + height_cols
        for col in numeric_cols:
            if col in buildings_in_buffer.columns:
                buildings_in_buffer[col] = pd.to_numeric(
                    buildings_in_buffer[col], errors="coerce"
                )

        # Ensure volume is present
        if (
            "volume_m3" not in buildings_in_buffer.columns
            and "footprint_area_m2" in buildings_in_buffer.columns
            and "height_median" in buildings_in_buffer.columns
        ):
            buildings_in_buffer["volume_m3"] = (
                buildings_in_buffer["footprint_area_m2"]
                * buildings_in_buffer["height_median"]
            )

        # Estimated gross floor area for FAR
        if (
            "footprint_area_m2" in buildings_in_buffer.columns
            and "height_median" in buildings_in_buffer.columns
        ):
            height_m = pd.to_numeric(
                buildings_in_buffer["height_median"], errors="coerce"
            )
            buildings_in_buffer["estimated_floors"] = np.clip(
                np.floor(height_m / ASSUMED_STOREY_HEIGHT_M), 1, None
            )
            buildings_in_buffer["gross_floor_area_m2"] = (
                buildings_in_buffer["footprint_area_m2"]
                * buildings_in_buffer["estimated_floors"]
            )
            mean_floors = buildings_in_buffer["estimated_floors"].mean()
            print(f"    Estimated floors: mean={mean_floors:.1f}")

        # Select numeric columns to aggregate
        stats_columns = []
        if "footprint_area_m2" in buildings_in_buffer.columns:
            stats_columns.append("footprint_area_m2")
        if "volume_m3" in buildings_in_buffer.columns:
            stats_columns.append("volume_m3")
        if "gross_floor_area_m2" in buildings_in_buffer.columns:
            stats_columns.append("gross_floor_area_m2")
        stats_columns.extend(height_cols)

        if stats_columns:
            print(f"    Aggregating columns: {stats_columns}")
            nodes_gdf, _data_gdf = layers.compute_stats(
                buildings_in_buffer,
                stats_column_labels=stats_columns,
                nodes_gdf=nodes_gdf,
                network_structure=network_structure,
                distances=STATS_DISTANCES,
            )
            stats_result_cols = [
                c for c in nodes_gdf.columns if any(sc in c for sc in stats_columns)
            ]
            print(f"    ✓ Building stats: {len(stats_result_cols)} columns")
        else:
            print("    No numeric columns found for aggregation")

        # Derive FAR and BCR from aggregated stats
        print("\n  Deriving FAR and BCR from catchment stats...")
        for dist in STATS_DISTANCES:
            catchment_area = np.pi * dist**2
            gfa_col = f"cc_gross_floor_area_m2_{dist}_sum"
            if gfa_col in nodes_gdf.columns:
                nodes_gdf[f"far_{dist}"] = nodes_gdf[gfa_col] / catchment_area
                mean_far = nodes_gdf[f"far_{dist}"].mean()
                print(f"    ✓ far_{dist}: mean={mean_far:.3f}")
            fp_col = f"cc_footprint_area_m2_{dist}_sum"
            if fp_col in nodes_gdf.columns:
                nodes_gdf[f"bcr_{dist}"] = nodes_gdf[fp_col] / catchment_area
                mean_bcr = nodes_gdf[f"bcr_{dist}"].mean()
                print(f"    ✓ bcr_{dist}: mean={mean_bcr:.3f}")
    else:
        print("\n  Skipping building statistics (no buildings provided)")

    # Load land use data for accessibility
    print("\n  Loading land use data...")

    # FSA establishments
    fsa = gpd.read_file(PATHS["fsa"], bbox=buffered_bounds.bounds)
    fsa = fsa.to_crs(boundaries.crs)
    fsa_in_buffer = fsa[fsa.intersects(buffered_bounds)].copy()
    fsa_category_map = {
        "Restaurant/Cafe/Canteen": "fsa_restaurant",
        "Takeaway/sandwich shop": "fsa_takeaway",
        "Pub/bar/nightclub": "fsa_pub",
    }
    fsa_in_buffer["landuse"] = (
        fsa_in_buffer["business_type"].map(fsa_category_map).fillna("fsa_other")
    )
    print(f"    FSA: {len(fsa_in_buffer)} establishments")
    for cat, count in fsa_in_buffer["landuse"].value_counts().items():
        print(f"      - {cat}: {count}")

    # Greenspace
    greenspace = gpd.read_file(
        PATHS["greenspace"],
        layer="greenspace_site",
        bbox=buffered_bounds.bounds,
    )
    greenspace = greenspace.to_crs(boundaries.crs)
    greenspace = greenspace.copy()
    greenspace["geometry"] = greenspace.geometry.centroid
    greenspace["landuse"] = "greenspace"
    print(f"    Greenspace: {len(greenspace)} sites")

    # Transport
    transport = gpd.read_file(PATHS["transport"], bbox=buffered_bounds.bounds)
    transport = transport.to_crs(boundaries.crs)
    transport_in_buffer = transport[transport.intersects(buffered_bounds)].copy()
    bus_types = ["BCT", "BCS", "BCE", "BCQ", "BST"]
    rail_types = ["RSE", "RLY", "PLT", "MET"]
    bus_stops = transport_in_buffer[
        transport_in_buffer["stop_type"].isin(bus_types)
    ].copy()
    rail_stops = transport_in_buffer[
        transport_in_buffer["stop_type"].isin(rail_types)
    ].copy()
    bus_stops["landuse"] = "bus"
    rail_stops["landuse"] = "rail"
    print(f"    Bus stops: {len(bus_stops)}, Rail stations: {len(rail_stops)}")

    # Compute accessibility metrics
    print(f"\n  Computing accessibility (distances={ACCESSIBILITY_DISTANCES})...")
    fsa_keys = list(fsa_in_buffer["landuse"].unique())
    nodes_gdf, _data_gdf = layers.compute_accessibilities(
        fsa_in_buffer,
        landuse_column_label="landuse",
        accessibility_keys=fsa_keys,
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=ACCESSIBILITY_DISTANCES,
    )
    print(f"    ✓ FSA accessibility computed ({len(fsa_keys)} categories)")

    nodes_gdf, _data_gdf = layers.compute_accessibilities(
        greenspace,
        landuse_column_label="landuse",
        accessibility_keys=["greenspace"],
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=ACCESSIBILITY_DISTANCES,
    )
    print("    ✓ Greenspace accessibility computed")

    transport_combined = pd.concat([bus_stops, rail_stops], ignore_index=True)
    if len(transport_combined) > 0:
        transport_keys = list(transport_combined["landuse"].unique())
        nodes_gdf, _data_gdf = layers.compute_accessibilities(
            transport_combined,
            landuse_column_label="landuse",
            accessibility_keys=transport_keys,
            nodes_gdf=nodes_gdf,
            network_structure=network_structure,
            distances=ACCESSIBILITY_DISTANCES,
        )
        print(f"    ✓ Transport accessibility computed ({transport_keys})")

    # Schools (GIAS)
    schools_path = PATHS.get("schools")
    if schools_path and schools_path.exists():
        schools = gpd.read_file(schools_path, bbox=buffered_bounds.bounds)
        schools = schools.to_crs(boundaries.crs)
        schools_in_buffer = schools[schools.intersects(buffered_bounds)].copy()
        schools_in_buffer["landuse"] = "school"
        print(f"    Schools: {len(schools_in_buffer)} establishments")
        if len(schools_in_buffer) > 0:
            nodes_gdf, _data_gdf = layers.compute_accessibilities(
                schools_in_buffer,
                landuse_column_label="landuse",
                accessibility_keys=["school"],
                nodes_gdf=nodes_gdf,
                network_structure=network_structure,
                distances=ACCESSIBILITY_DISTANCES,
            )
            print("    ✓ Schools accessibility computed")
    else:
        print("    Schools data not available — skipping")

    # Health facilities (NHS ODS)
    health_path = PATHS.get("health")
    if health_path and health_path.exists():
        health = gpd.read_file(health_path, bbox=buffered_bounds.bounds)
        health = health.to_crs(boundaries.crs)
        health_in_buffer = health[health.intersects(buffered_bounds)].copy()
        # Map facility types to landuse labels
        type_map = {
            "hospitals": "hospital",
            "gp_practices": "gp_practice",
            "pharmacies": "pharmacy",
        }
        health_in_buffer["landuse"] = (
            health_in_buffer["facility_type"].map(type_map).fillna("health_other")
        )
        health_keys = list(health_in_buffer["landuse"].unique())
        print(f"    Health facilities: {len(health_in_buffer)} ({health_keys})")
        if len(health_in_buffer) > 0:
            nodes_gdf, _data_gdf = layers.compute_accessibilities(
                health_in_buffer,
                landuse_column_label="landuse",
                accessibility_keys=health_keys,
                nodes_gdf=nodes_gdf,
                network_structure=network_structure,
                distances=ACCESSIBILITY_DISTANCES,
            )
            print(f"    ✓ Health accessibility computed ({health_keys})")
    else:
        print("    Health data not available — skipping")

    # Drop redundant columns to keep output manageable
    all_cc = [c for c in nodes_gdf.columns if c.startswith("cc_")]
    keep_cc: set[str] = set()
    for c in all_cc:
        for prefix in (
            "cc_harmonic_",
            "cc_betweenness_",
            "cc_density_",
            "cc_beta_",
            "cc_cycles_",
        ):
            if c.startswith(prefix):
                keep_cc.add(c)
    for c in all_cc:
        if c.endswith("_wt") or c.endswith("_nw") or "_nearest_" in c:
            keep_cc.add(c)
    for c in all_cc:
        if c.startswith("cc_gross_floor_area_m2_sum_"):
            keep_cc.add(c)
        if c.startswith("cc_footprint_area_m2_sum_"):
            keep_cc.add(c)
        if c.startswith("cc_footprint_area_m2_count_"):
            keep_cc.add(c)
    drop_cc = [c for c in all_cc if c not in keep_cc]
    if drop_cc:
        nodes_gdf = nodes_gdf.drop(columns=drop_cc)
        print(
            f"\n  Trimmed: kept {len(keep_cc)} of {len(all_cc)} cc_ columns"
            f" (dropped {len(drop_cc)} redundant stats)"
        )

    # Discard buffer nodes — keep only live nodes (those within the study boundary)
    live_nodes = nodes_gdf[nodes_gdf["live"]].copy()

    cc_cols = [c for c in live_nodes.columns if c.startswith("cc_")]
    if not cc_cols:
        raise RuntimeError(
            f"Stage 2 produced no cc_ columns. Columns: {list(live_nodes.columns)}"
        )
    print(f"  Cityseer metric columns: {len(cc_cols)}")
    print(f"  Live nodes saved: {len(live_nodes)}")

    return live_nodes


# ---------------------------------------------------------------------------
# Stage 3: LSOA Aggregation
# ---------------------------------------------------------------------------


def run_stage3_lsoa_aggregation(
    boundaries: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame | None,
    nodes: gpd.GeoDataFrame | None,
    city_name: str,
) -> gpd.GeoDataFrame | None:
    """
    Stage 3: Aggregate all data to LSOA level.

    UPRNs are the transient join key — used to link morphology, census, EPC,
    and network data — then immediately discarded after aggregation.

    Parameters
    ----------
    boundaries : gpd.GeoDataFrame
        Study area boundaries.
    buildings : gpd.GeoDataFrame or None
        Buildings with morphology metrics from Stage 1.
    nodes : gpd.GeoDataFrame or None
        Network nodes with cityseer metrics from Stage 2.
    city_name : str
        Short name for the city (added as a column).

    Returns
    -------
    gpd.GeoDataFrame or None
        LSOA-level dataset with polygon geometry, or None if no data.
    """
    print()
    print("=" * 60)
    print("STAGE 3: LSOA AGGREGATION")
    print("=" * 60)

    combined_bounds = boundaries.union_all()

    # ------------------------------------------------------------------
    # 1. Load UPRNs (transient — never saved)
    # ------------------------------------------------------------------
    print("  Loading UPRNs...")
    uprn = gpd.read_file(PATHS["uprn"], bbox=combined_bounds.bounds)
    uprn = uprn.to_crs(boundaries.crs)
    uprn_gdf = uprn[uprn.intersects(combined_bounds)].copy()
    uprn_gdf = uprn_gdf.reset_index(drop=True)
    del uprn
    print(f"    {len(uprn_gdf):,} UPRNs in boundary")

    if len(uprn_gdf) == 0:
        print("  WARNING: No UPRNs found in boundary")
        return None

    uprn_col = "UPRN" if "UPRN" in uprn_gdf.columns else "uprn"

    # ------------------------------------------------------------------
    # 2. Spatial join: UPRN → Building (morphology)
    # ------------------------------------------------------------------
    print("\n  1. Joining UPRNs to buildings...")
    if buildings is not None:
        morph_cols = [
            c for c in _MORPH_SUM_COLS + _MORPH_MEAN_COLS if c in buildings.columns
        ]
        if "height_mean" not in buildings.columns:
            for alt in ["height_median", "height_max"]:
                if alt in buildings.columns:
                    morph_cols.append(alt)
                    print(f"    Note: using {alt} (height_mean not found)")
                    break

        buildings_for_join = buildings[["geometry"] + morph_cols].copy()
        uprn_gdf = gpd.sjoin(
            uprn_gdf,
            buildings_for_join,
            how="left",
            predicate="within",
        ).drop(columns=["index_right"], errors="ignore")

        matched = uprn_gdf[morph_cols[0]].notna().sum() if morph_cols else 0
        print(f"    {matched:,}/{len(uprn_gdf):,} UPRNs matched to buildings")
    else:
        print("    Buildings not available — skipping")

    # ------------------------------------------------------------------
    # 3. Spatial join: UPRN → Census OA
    # ------------------------------------------------------------------
    print("\n  2. Joining UPRNs to Census Output Areas...")
    census = gpd.read_file(PATHS["census"], bbox=combined_bounds.bounds)
    census = census.to_crs(boundaries.crs)

    census_ts_cols = [c for c in census.columns if c.startswith("ts0")]
    census_id_cols = [c for c in ["OA21CD", "LSOA21CD"] if c in census.columns]
    census_for_join = census[["geometry"] + census_id_cols + census_ts_cols].copy()

    uprn_gdf = gpd.sjoin(
        uprn_gdf,
        census_for_join,
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")

    # Deduplicate: sjoin can produce duplicates at OA boundaries
    uprn_gdf = uprn_gdf.drop_duplicates(subset=[uprn_col], keep="first")

    n_with_lsoa = uprn_gdf["LSOA21CD"].notna().sum()
    print(f"    {n_with_lsoa:,}/{len(uprn_gdf):,} UPRNs matched to Census OAs")

    if n_with_lsoa == 0:
        print("  WARNING: No UPRNs matched to Census OAs")
        return None

    # ------------------------------------------------------------------
    # 3b. LSOA geometry: dissolve OA polygons
    # ------------------------------------------------------------------
    print("\n  3. Dissolving OA polygons to LSOA boundaries...")
    lsoa_geom = (
        census[["LSOA21CD", "geometry"]]
        .dissolve(by="LSOA21CD")
        .reset_index()[["LSOA21CD", "geometry"]]
    )
    print(f"    {len(lsoa_geom):,} LSOA polygons")
    del census
    gc.collect()

    # ------------------------------------------------------------------
    # 4. Direct join: UPRN → EPC (narrow read)
    # ------------------------------------------------------------------
    print("\n  4. Joining UPRNs to EPC records...")
    has_epc_coverage = False
    has_build_year = False

    epc_path = PATHS.get("epc")
    if epc_path and epc_path.exists():
        epc_schema = pq.read_schema(epc_path)
        uprn_col_epc = "UPRN" if "UPRN" in epc_schema.names else "uprn"

        epc_want = [
            uprn_col_epc,
            "INSPECTION_DATE",
            "PROPERTY_TYPE",
            "CONSTRUCTION_AGE_BAND",
        ]
        epc_read_cols = [c for c in epc_want if c in epc_schema.names]

        uprn_keys = set(uprn_gdf[uprn_col].dropna().astype(int))
        uprn_arr = pa.array(list(uprn_keys), type=pa.int64())
        epc_filter = pc.is_in(pc.field(uprn_col_epc), value_set=uprn_arr)  # type: ignore[unresolved-attribute]
        epc_table = pq.read_table(epc_path, columns=epc_read_cols, filters=epc_filter)
        epc = epc_table.to_pandas()
        del epc_table
        print(f"    {len(epc):,} EPC records matching city UPRNs")

        if "INSPECTION_DATE" in epc.columns:
            epc = epc.sort_values("INSPECTION_DATE", ascending=False).drop_duplicates(
                subset=[uprn_col_epc], keep="first"
            )

        epc_join_cols = [
            c for c in ["PROPERTY_TYPE", "CONSTRUCTION_AGE_BAND"] if c in epc.columns
        ]
        epc_for_join = epc[[uprn_col_epc] + epc_join_cols].rename(
            columns={uprn_col_epc: uprn_col}
        )
        uprn_gdf = uprn_gdf.merge(epc_for_join, on=uprn_col, how="left")

        matched = (
            uprn_gdf["PROPERTY_TYPE"].notna().sum()
            if "PROPERTY_TYPE" in uprn_gdf.columns
            else 0
        )
        print(f"    {matched:,}/{len(uprn_gdf):,} UPRNs matched to EPC")
        has_epc_coverage = "PROPERTY_TYPE" in uprn_gdf.columns
        has_build_year = "CONSTRUCTION_AGE_BAND" in uprn_gdf.columns
        del epc
        gc.collect()
    else:
        print("    EPC data not available — skipping")

    # ------------------------------------------------------------------
    # 5. Nearest: UPRN → Network node (cKDTree)
    # ------------------------------------------------------------------
    print("\n  5. Joining UPRNs to nearest network nodes...")
    if nodes is not None and len(nodes) > 0:
        exclude = {
            "geometry",
            "geom",
            "x",
            "y",
            "index",
            "ns_node_idx",
            "live",
            "weight",
        }
        network_cols = [
            c
            for c in nodes.columns
            if c not in exclude
            and (c.startswith("cc_") or c.startswith("far_") or c.startswith("bcr_"))
        ]

        node_coords = np.column_stack([nodes.geometry.x, nodes.geometry.y])
        uprn_coords = np.column_stack([uprn_gdf.geometry.x, uprn_gdf.geometry.y])
        tree = cKDTree(node_coords)
        distances, indices = tree.query(uprn_coords, k=1)

        nearest_data = nodes.iloc[indices][network_cols].reset_index(drop=True)
        uprn_gdf = pd.concat([uprn_gdf.reset_index(drop=True), nearest_data], axis=1)
        uprn_gdf = gpd.GeoDataFrame(uprn_gdf, geometry="geometry", crs=boundaries.crs)

        print(f"    {len(uprn_gdf):,} UPRNs linked to nearest node")
        print(f"    Network columns: {len(network_cols)}")
        print(f"    Mean distance: {distances.mean():.1f}m")
    else:
        print("    Network nodes not available — skipping")

    # ==================================================================
    # AGGREGATION TO LSOA
    # ==================================================================
    print("\n  " + "=" * 56)
    print("  AGGREGATING TO LSOA")
    print("  " + "=" * 56)

    uprn_gdf = uprn_gdf[uprn_gdf["LSOA21CD"].notna()].copy()

    for col in _MORPH_SUM_COLS + _MORPH_MEAN_COLS:
        if col in uprn_gdf.columns:
            uprn_gdf[col] = pd.to_numeric(uprn_gdf[col], errors="coerce")

    # Phase 1: Census deduplication (OA → LSOA)
    print("\n  Phase 1: Census deduplication...")
    census_cols = [c for c in uprn_gdf.columns if c.startswith("ts0")]
    oa_census = uprn_gdf.groupby("OA21CD")[census_cols].first()
    oa_to_lsoa = uprn_gdf.groupby("OA21CD")["LSOA21CD"].first()
    oa_census["LSOA21CD"] = oa_to_lsoa

    if _TS006_DENSITY in oa_census.columns and _TS001_POP in oa_census.columns:
        oa_pop = pd.to_numeric(oa_census[_TS001_POP], errors="coerce")
        oa_dens = pd.to_numeric(oa_census[_TS006_DENSITY], errors="coerce")
        oa_census["_oa_area_km2"] = oa_pop / oa_dens.replace(0, np.nan)

    lsoa_census = oa_census.groupby("LSOA21CD").sum()
    n_oas = len(oa_census)
    n_lsoas_census = len(lsoa_census)
    print(f"    {n_oas:,} OAs → {n_lsoas_census:,} LSOAs")

    # Phase 2: UPRN → LSOA aggregation
    print("\n  Phase 2: UPRN aggregation...")

    n_uprns = uprn_gdf.groupby("LSOA21CD").size().reset_index(name="n_uprns")

    sum_cols = [c for c in _MORPH_SUM_COLS if c in uprn_gdf.columns]
    mean_cols = [c for c in _MORPH_MEAN_COLS if c in uprn_gdf.columns]

    agg_dict: dict[str, str] = {}
    for c in sum_cols:
        agg_dict[c] = "sum"
    for c in mean_cols:
        agg_dict[c] = "mean"

    cc_cols = [c for c in uprn_gdf.columns if c.startswith(("cc_", "far_", "bcr_"))]
    for c in cc_cols:
        agg_dict[c] = "mean"

    lsoa_agg = uprn_gdf.groupby("LSOA21CD").agg(agg_dict).reset_index()
    print(f"    {len(lsoa_agg):,} LSOAs from UPRN aggregation")

    # EPC-derived aggregations
    epc_pieces: list[pd.DataFrame] = []

    if has_epc_coverage:
        print("\n  EPC coverage...")
        n_epc = (
            uprn_gdf.groupby("LSOA21CD")["PROPERTY_TYPE"]
            .apply(lambda s: s.notna().sum())
            .reset_index(name="n_epc")
        )
        epc_pieces.append(n_epc)

    if has_build_year:
        print("  Building era...")
        age = uprn_gdf[["LSOA21CD", "CONSTRUCTION_AGE_BAND"]].copy()
        age = age[age["CONSTRUCTION_AGE_BAND"].notna()].copy()
        age["_year"] = age["CONSTRUCTION_AGE_BAND"].map(ERA_MAP)
        unmapped = age["_year"].isna()
        age.loc[unmapped, "_year"] = pd.to_numeric(
            age.loc[unmapped, "CONSTRUCTION_AGE_BAND"], errors="coerce"
        )
        age = age[age["_year"].notna()]
        if len(age) > 0:
            median_year = (
                age.groupby("LSOA21CD")["_year"]
                .median()
                .reset_index(name="median_build_year")
            )
            epc_pieces.append(median_year)

    # Assemble LSOA DataFrame
    print("\n  Assembling LSOA dataset...")

    lsoa = lsoa_geom.copy()
    lsoa = lsoa.merge(n_uprns, on="LSOA21CD", how="inner")
    lsoa = lsoa.merge(lsoa_agg, on="LSOA21CD", how="left")
    lsoa = lsoa.merge(lsoa_census.reset_index(), on="LSOA21CD", how="left")

    for piece in epc_pieces:
        lsoa = lsoa.merge(piece, on="LSOA21CD", how="left")

    if "n_epc" in lsoa.columns:
        lsoa["epc_coverage"] = lsoa["n_epc"] / lsoa["n_uprns"]

    # Aggregate S/V ratio — sum(envelope) / sum(volume)
    if "envelope_area_m2" in lsoa.columns and "volume_m3" in lsoa.columns:
        lsoa["lsoa_sv"] = lsoa["envelope_area_m2"] / lsoa["volume_m3"].replace(
            0, np.nan
        )

    # Metered energy — direct join on LSOA21CD
    energy_path = PATHS.get("energy_stats")
    if energy_path and energy_path.exists():
        print("  Joining metered energy...")
        energy_df = pd.read_parquet(energy_path)
        energy_cols = [c for c in energy_df.columns if c != "LSOA21CD"]
        lsoa = lsoa.merge(energy_df, on="LSOA21CD", how="left")
        matched = lsoa[energy_cols[0]].notna().sum() if energy_cols else 0
        print(f"    {matched:,}/{len(lsoa):,} LSOAs matched to metered energy")

    # Scaling data (GVA) — direct join on LSOA code
    scaling_path = PATHS.get("scaling")
    if scaling_path and scaling_path.exists():
        print("  Joining scaling data (GVA)...")
        scaling_df = pd.read_parquet(scaling_path)
        scaling_df = scaling_df.rename(columns={"LSOA_CODE": "LSOA21CD"})
        lsoa = lsoa.merge(scaling_df, on="LSOA21CD", how="left")
        matched = (
            lsoa["lsoa_gva_millions"].notna().sum()
            if "lsoa_gva_millions" in lsoa.columns
            else 0
        )
        print(f"    {matched:,}/{len(lsoa):,} LSOAs matched to GVA")

    lsoa["city"] = city_name
    lsoa = gpd.GeoDataFrame(lsoa, geometry="geometry", crs=boundaries.crs)

    del uprn_gdf
    gc.collect()

    print(f"\n  LSOA output: {len(lsoa):,} LSOAs, {len(lsoa.columns)} columns")
    return lsoa


# ---------------------------------------------------------------------------
# Save and combine
# ---------------------------------------------------------------------------


def save_city_lsoa(city_name: str, lsoa: gpd.GeoDataFrame | None) -> None:
    """Save LSOA GeoPackage for a single city."""
    if lsoa is None:
        return

    city_dir = OUTPUT_DIR / city_name
    city_dir.mkdir(parents=True, exist_ok=True)

    # Drop any extra geometry columns from joins
    geom_cols = [
        c
        for c in lsoa.columns
        if c != "geometry" and (lsoa[c].dtype == "geometry" or c == "geom")
    ]
    if geom_cols:
        lsoa = lsoa.drop(columns=geom_cols)  # type: ignore[invalid-assignment]

    path = city_dir / "lsoa_integrated.gpkg"
    assert lsoa is not None
    lsoa.to_file(path, driver="GPKG")
    print(f"  Saved: {path}")


def combine_lsoa_cities(city_names: list[str]) -> None:
    """Combine per-city LSOA datasets into a single GeoPackage."""
    print()
    print("=" * 60)
    print("COMBINING LSOA OUTPUTS")
    print("=" * 60)

    frames: list[gpd.GeoDataFrame] = []
    for name in city_names:
        path = OUTPUT_DIR / name / "lsoa_integrated.gpkg"
        if not path.exists():
            print(f"  SKIP: {name} (no output yet)")
            continue
        gdf = gpd.read_file(path)
        if "city" not in gdf.columns:
            gdf["city"] = name
        frames.append(gdf)
        print(f"  {name}: {len(gdf):,} LSOAs")

    if not frames:
        print("  No city data to combine.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=frames[0].crs)

    combined_dir = OUTPUT_DIR / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    out_path = combined_dir / "lsoa_integrated.gpkg"
    combined.to_file(out_path, driver="GPKG")
    print(f"\n  Combined: {len(combined):,} LSOAs → {out_path}")


# ---------------------------------------------------------------------------
# Per-city orchestration
# ---------------------------------------------------------------------------


def process_city(
    bua_code: str,
    city_name: str,
    *,
    test_mode: bool = False,
) -> None:
    """
    Run the full LSOA pipeline for a single city.

    Parameters
    ----------
    bua_code : str
        BUA22CD code for the city boundary.
    city_name : str
        Short name used for output directory and city column.
    test_mode : bool
        If True, use reduced centrality distances for faster testing.
    """
    print()
    print("=" * 60)
    print(f"PROCESSING (LSOA): {city_name} ({bua_code})")
    print("=" * 60)

    # Skip if already complete
    lsoa_path = OUTPUT_DIR / city_name / "lsoa_integrated.gpkg"
    if lsoa_path.exists():
        _probe = gpd.read_file(lsoa_path, rows=1)
        has_cc = any(c.startswith("cc_") for c in _probe.columns)
        del _probe
        if has_cc:
            print(f"  Already processed (with cityseer metrics): {lsoa_path}")
            print("  Delete to reprocess.")
            return
        print("  Incomplete output (no cityseer metrics) — reprocessing...")
        lsoa_path.unlink()

    boundary = load_boundary(bua_code)

    # Stage 1: Morphology
    buildings = run_stage1_morphology(boundary)

    # Stage 2: Network Analysis — use cache if available
    nodes_cache = OUTPUT_DIR / city_name / "network_segments.gpkg"
    nodes = None
    if nodes_cache.exists():
        _probe = gpd.read_file(nodes_cache, rows=1)
        has_cc = any(c.startswith("cc_") for c in _probe.columns)
        del _probe
        if has_cc:
            print(f"\n  Loading cached network: {nodes_cache}")
            nodes = gpd.read_file(nodes_cache)
            print(f"  {len(nodes):,} nodes from cache")
    if nodes is None:
        nodes = run_stage2_network(boundary, buildings=buildings, test_mode=test_mode)
        if nodes is not None:
            city_dir = OUTPUT_DIR / city_name
            city_dir.mkdir(parents=True, exist_ok=True)
            nodes.to_file(nodes_cache, driver="GPKG")
            print(f"  Saved network cache: {nodes_cache}")

    # Stage 3: LSOA Aggregation
    lsoa = run_stage3_lsoa_aggregation(boundary, buildings, nodes, city_name)

    # Save
    save_city_lsoa(city_name, lsoa)

    if lsoa is not None:
        print(f"\n  {city_name}: {len(lsoa):,} LSOAs, {len(lsoa.columns)} columns")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the LSOA pipeline for all (or selected) cities."""
    args = sys.argv[1:]
    if args:
        selected = {code: name for code, name in CITIES.items() if name in args}
        if not selected:
            print(f"Unknown cities: {args}")
            print(f"Available: {list(CITIES.values())}")
            sys.exit(1)
    else:
        selected = CITIES

    print()
    print("=" * 60)
    print("URBAN ENERGY LSOA PIPELINE")
    print("=" * 60)
    print(f"Cities: {list(selected.values())}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    check_inputs()

    for bua_code, city_name in selected.items():
        process_city(bua_code, city_name, test_mode=False)
        gc.collect()

    combine_lsoa_cities(list(CITIES.values()))


if __name__ == "__main__":
    main()
