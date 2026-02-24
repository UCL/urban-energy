"""
Processing pipeline for urban energy analysis.

Runs all three processing stages on Built-Up Area boundaries:
  Stage 1: Building morphology (from cached LiDAR + momepy metrics)
  Stage 2: Network analysis (cityseer centrality + accessibility)
  Stage 3: UPRN integration (joins morphology, census, EPC, network)

Each city is processed independently (separate network, separate output),
then combined into a single dataset with a city identifier column.

Usage:
    uv run python processing/test_pipeline.py                  # all cities
    uv run python processing/test_pipeline.py manchester       # one city
    uv run python processing/test_pipeline.py york cambridge   # subset
"""

import gc
import sys
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

# Configuration
from urban_energy.paths import TEMP_DIR

OUTPUT_DIR = TEMP_DIR / "processing"

# Study cities: BUA22CD -> short name
# Each city is processed independently through all 3 stages.
CITIES: dict[str, str] = {
    # Large cities
    "E63008401": "manchester",
    "E63012168": "bristol",
    "E63010901": "milton_keynes",
    "E63007706": "york",
    "E63010556": "cambridge",
    # Smaller towns (typological contrast)
    "E63011231": "stevenage",  # new town (sprawl control)
    "E63007907": "burnley",  # northern mill town (dense terraced)
    "E63012524": "canterbury",  # historic compact city
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
}


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

    Check if morphology is already cached for test boundaries.
    If not, compute it.
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
            # TODO: Run morphology processing for this boundary
            return None

    if not results:
        print("  No buildings found in test boundaries!")
        return None

    # Combine all buildings
    buildings = pd.concat(results, ignore_index=True)
    buildings = gpd.GeoDataFrame(buildings, crs=results[0].crs)

    # Validate required columns (including thermal metrics from process_morphology.py)
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

    Buffer requirements (full mode):
    - Centrality: 10, 20, 40, 60, 120 minutes (max 120 min)
    - Accessibility: 5, 10, 20, 60 minutes (max 60 min)
    - At 1.33 m/s, 120 min = ~9600m buffer needed
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
        CENTRALITY_DISTANCES = [800, 1600]  # Reduced for faster testing
        ACCESSIBILITY_DISTANCES = [400, 800, 1600]
        print("  [TEST MODE: reduced distances for faster processing]")
    else:
        CENTRALITY_DISTANCES = [800, 1600, 3200, 4800, 9600]
        ACCESSIBILITY_DISTANCES = [400, 800, 1600, 4800]

    # Buffer must cover max analysis distance
    max_distance = max(max(CENTRALITY_DISTANCES), max(ACCESSIBILITY_DISTANCES))
    buffer_m = int(max_distance * 1.1)  # 10% margin
    print(f"  Buffer: {buffer_m}m (max distance + 10% margin)")

    # Get combined boundary for clipping
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

    # Convert to cityseer network structure
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(
        nx_graph
    )
    print(f"    Network structure built: {len(nodes_gdf)} nodes")

    # Compute centrality
    print(f"\n  Computing centrality (distances={CENTRALITY_DISTANCES})...")
    nodes_gdf = networks.node_centrality_shortest_adaptive(
        network_structure,
        nodes_gdf,
        distances=CENTRALITY_DISTANCES,
        compute_closeness=True,
        compute_betweenness=True,
    )
    centrality_cols = [
        c for c in nodes_gdf.columns if "beta" in c or "harmonic" in c
    ]
    print(f"    Centrality columns: {len(centrality_cols)}")

    # Compute building statistics over the network at accessibility distances
    STATS_DISTANCES = ACCESSIBILITY_DISTANCES
    ASSUMED_STOREY_HEIGHT_M = 3.0
    if buildings is not None and len(buildings) > 0:
        print(f"\n  Computing building stats (distances={STATS_DISTANCES})...")
        # Use building centroids for network assignment
        buildings_pts = buildings.copy()
        buildings_pts["geometry"] = buildings_pts.geometry.centroid
        buildings_in_buffer = buildings_pts[
            buildings_pts.intersects(buffered_bounds)
        ].copy()
        print(f"    Buildings in buffer: {len(buildings_in_buffer)}")

        # Find all height columns
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

        # Ensure volume is present (from process_morphology.py or compute here)
        if (
            "volume_m3" not in buildings_in_buffer.columns
            and "footprint_area_m2" in buildings_in_buffer.columns
            and "height_median" in buildings_in_buffer.columns
        ):
            buildings_in_buffer["volume_m3"] = (
                buildings_in_buffer["footprint_area_m2"]
                * buildings_in_buffer["height_median"]
            )

        # --- Building type classification from shared_wall_ratio ---
        if "shared_wall_ratio" in buildings_in_buffer.columns:
            swr = buildings_in_buffer["shared_wall_ratio"]
            buildings_in_buffer["is_detached"] = (swr == 0).astype(float)
            buildings_in_buffer["is_semi"] = (
                (swr > 0) & (swr < 0.3)
            ).astype(float)
            buildings_in_buffer["is_terraced"] = (swr >= 0.3).astype(float)
            n_det = int(buildings_in_buffer["is_detached"].sum())
            n_semi = int(buildings_in_buffer["is_semi"].sum())
            n_ter = int(buildings_in_buffer["is_terraced"].sum())
            print(
                f"    Building types (morphology): "
                f"detached={n_det:,}, semi={n_semi:,}, terraced={n_ter:,}"
            )

        # --- Estimated gross floor area for FAR ---
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
        # Building type indicators (mean within catchment = proportion)
        for type_col in ["is_detached", "is_semi", "is_terraced"]:
            if type_col in buildings_in_buffer.columns:
                stats_columns.append(type_col)
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
                c
                for c in nodes_gdf.columns
                if any(sc in c for sc in stats_columns)
            ]
            print(f"    ✓ Building stats: {len(stats_result_cols)} columns")
        else:
            print("    No numeric columns found for aggregation")

        # --- Derive FAR and BCR from aggregated stats ---
        print("\n  Deriving FAR and BCR from catchment stats...")
        for dist in STATS_DISTANCES:
            catchment_area = np.pi * dist**2
            gfa_col = f"cc_gross_floor_area_m2_{dist}_sum"
            if gfa_col in nodes_gdf.columns:
                nodes_gdf[f"far_{dist}"] = (
                    nodes_gdf[gfa_col] / catchment_area
                )
                mean_far = nodes_gdf[f"far_{dist}"].mean()
                print(f"    ✓ far_{dist}: mean={mean_far:.3f}")
            fp_col = f"cc_footprint_area_m2_{dist}_sum"
            if fp_col in nodes_gdf.columns:
                nodes_gdf[f"bcr_{dist}"] = (
                    nodes_gdf[fp_col] / catchment_area
                )
                mean_bcr = nodes_gdf[f"bcr_{dist}"].mean()
                print(f"    ✓ bcr_{dist}: mean={mean_bcr:.3f}")
    else:
        print("\n  Skipping building statistics (no buildings provided)")

    # Load land use data for accessibility
    print("\n  Loading land use data...")

    # FSA establishments - categorize by business type
    fsa = gpd.read_file(PATHS["fsa"], bbox=buffered_bounds.bounds)
    fsa = fsa.to_crs(boundaries.crs)
    fsa_in_buffer = fsa[fsa.intersects(buffered_bounds)].copy()
    # Map business types to analysis categories
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

    # Greenspace - use centroids
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
    # FSA accessibility - by category in one call
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

    # Greenspace accessibility
    nodes_gdf, _data_gdf = layers.compute_accessibilities(
        greenspace,
        landuse_column_label="landuse",
        accessibility_keys=["greenspace"],
        nodes_gdf=nodes_gdf,
        network_structure=network_structure,
        distances=ACCESSIBILITY_DISTANCES,
    )
    print("    ✓ Greenspace accessibility computed")

    # Transport accessibility - bus and rail in one call
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

    # --- Drop redundant columns to keep output manageable ---
    # Building stats produce 12 metrics × 6 agg types × 4 distances × 2 weights
    # = 576 columns. We only need a curated subset for analysis.
    all_cc = [c for c in nodes_gdf.columns if c.startswith("cc_")]
    keep_cc: set[str] = set()
    # Centrality (all distances): harmonic, betweenness, density, beta, cycles
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
    # Accessibility: weighted (_wt) and nearest only — drop unweighted (_nw)
    for c in all_cc:
        if c.endswith("_wt") or "_nearest_" in c:
            keep_cc.add(c)
    # Building stats: keep only FAR/BCR (derived) and mean height at 800
    for c in all_cc:
        if c.startswith("cc_gross_floor_area_m2_sum_"):
            keep_cc.add(c)  # needed for FAR
        if c.startswith("cc_footprint_area_m2_sum_"):
            keep_cc.add(c)  # needed for BCR
        if c.startswith("cc_footprint_area_m2_count_"):
            keep_cc.add(c)  # building density
    # FAR/BCR columns (already derived, not cc_ prefixed)
    drop_cc = [c for c in all_cc if c not in keep_cc]
    if drop_cc:
        nodes_gdf = nodes_gdf.drop(columns=drop_cc)
        print(f"\n  Trimmed: kept {len(keep_cc)} of {len(all_cc)} cc_ columns"
              f" (dropped {len(drop_cc)} redundant stats)")

    # Filter to nodes within study area (not just buffer)
    nodes_in_bounds = nodes_gdf[nodes_gdf.intersects(combined_bounds)]

    # Validate: cityseer metrics must be present
    cc_cols = [c for c in nodes_in_bounds.columns if c.startswith("cc_")]
    if not cc_cols:
        raise RuntimeError(
            "Stage 2 produced no cc_ columns. "
            f"Columns: {list(nodes_in_bounds.columns)}"
        )
    print(f"  Cityseer metric columns: {len(cc_cols)}")
    print(f"  Nodes in study area: {len(nodes_in_bounds)}")

    return nodes_in_bounds


def run_stage3_uprn_integration(
    boundaries: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame | None,
    segments: gpd.GeoDataFrame | None,
) -> gpd.GeoDataFrame | None:
    """
    Stage 3: UPRN Integration.

    Link all attributes to UPRNs as the atomic unit of analysis.

    Performs:
    - Spatial join: UPRN → Building (point-in-polygon for morphology)
    - Spatial join: UPRN → Census OA (point-in-polygon for demographics)
    - Direct join: UPRN → EPC (by UPRN field for energy performance)
    - Nearest: UPRN → Street segment (for network metrics)
    """
    print()
    print("=" * 60)
    print("STAGE 3: UPRN INTEGRATION")
    print("=" * 60)

    # Get combined boundary
    combined_bounds = boundaries.union_all()

    # Load UPRNs
    print("  Loading UPRNs...")
    uprn = gpd.read_file(
        PATHS["uprn"],
        bbox=combined_bounds.bounds,
    )
    uprn = uprn.to_crs(boundaries.crs)
    uprn_gdf = uprn[uprn.intersects(combined_bounds)].copy()
    print(f"    {len(uprn_gdf)} UPRNs in test boundaries")

    if len(uprn_gdf) == 0:
        print("  WARNING: No UPRNs found in test boundaries")
        return None

    # 1. Spatial join: UPRN → Building (morphology)
    print("\n  1. Joining UPRNs to buildings...")
    if buildings is not None:
        # Select morphology columns to transfer
        morph_cols = [
            "footprint_area_m2",
            "perimeter_m",
            "orientation",
            "convexity",
            "compactness",
            "elongation",
            "shared_wall_length_m",
            "shared_wall_ratio",
            # Thermal efficiency metrics
            "surface_to_volume",
            "form_factor",
            "external_wall_area_m2",
            "envelope_area_m2",
            "volume_m3",
        ]
        # Include height columns if present
        height_cols = [c for c in buildings.columns if c.startswith("height_")]
        cols_to_join = [c for c in morph_cols + height_cols if c in buildings.columns]

        # Perform spatial join
        buildings_for_join = buildings[["geometry"] + cols_to_join].copy()
        uprn_gdf = gpd.sjoin(
            uprn_gdf,
            buildings_for_join,
            how="left",
            predicate="within",
        )
        # Drop index column from sjoin
        if "index_right" in uprn_gdf.columns:
            uprn_gdf = uprn_gdf.drop(columns=["index_right"])

        matched = uprn_gdf[cols_to_join[0]].notna().sum() if cols_to_join else 0
        print(f"    ✓ {matched}/{len(uprn_gdf)} UPRNs matched to buildings")
    else:
        print("    ✗ Buildings not available - skipping")

    # 2. Spatial join: UPRN → Census OA (demographics)
    print("\n  2. Joining UPRNs to Census Output Areas...")
    census = gpd.read_file(PATHS["census"], bbox=combined_bounds.bounds)
    census = census.to_crs(boundaries.crs)

    # Select census columns (exclude geometry and join columns)
    census_cols = [
        c for c in census.columns if c not in ["geometry", "index_right", "index_left"]
    ]
    census_for_join = census[["geometry"] + census_cols].copy()

    uprn_gdf = gpd.sjoin(
        uprn_gdf,
        census_for_join,
        how="left",
        predicate="within",
    )
    if "index_right" in uprn_gdf.columns:
        uprn_gdf = uprn_gdf.drop(columns=["index_right"])

    # Count matches using first census column
    if census_cols:
        matched = uprn_gdf[census_cols[0]].notna().sum()
        print(f"    ✓ {matched}/{len(uprn_gdf)} UPRNs matched to Census OAs")
        print(f"    Census columns added: {len(census_cols)}")

    # 2b. Tabular join: UPRN → LSOA metered energy consumption (DESNZ)
    energy_stats_path = PATHS.get("energy_stats")
    if (
        energy_stats_path
        and energy_stats_path.exists()
        and "LSOA21CD" in uprn_gdf.columns
    ):
        print("\n  2b. Joining LSOA metered energy consumption (DESNZ)...")
        energy_df = pd.read_parquet(energy_stats_path)
        energy_cols = [c for c in energy_df.columns if c != "LSOA21CD"]
        uprn_gdf = uprn_gdf.merge(energy_df, on="LSOA21CD", how="left")
        if energy_cols:
            matched = uprn_gdf[energy_cols[0]].notna().sum()
            print(f"    ✓ {matched}/{len(uprn_gdf)} UPRNs matched to LSOA energy")
            print(f"    Energy columns added: {len(energy_cols)}")
    else:
        print("\n  2b. LSOA energy data not available - skipping")

    # 3. Direct join: UPRN → EPC (energy performance)
    print("\n  3. Joining UPRNs to EPC records...")
    uprn_col_uprn = "UPRN" if "UPRN" in uprn_gdf.columns else "uprn"
    # Use PyArrow to read only matching UPRNs (avoids loading 1.7GB into memory)
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    epc_schema = pq.read_schema(PATHS["epc"])
    uprn_col_epc = "UPRN" if "UPRN" in epc_schema.names else "uprn"
    uprn_keys = set(uprn_gdf[uprn_col_uprn].dropna().astype(int))
    # Exclude geometry/spatial columns from the read
    epc_read_cols = [
        c
        for c in epc_schema.names
        if c.lower() not in ["geometry", "lat", "lon", "latitude", "longitude"]
    ]
    uprn_arr = pa.array(list(uprn_keys), type=pa.int64())
    epc_filter = pc.is_in(pc.field(uprn_col_epc), value_set=uprn_arr)
    epc_table = pq.read_table(
        PATHS["epc"], columns=epc_read_cols, filters=epc_filter
    )
    n_total = pq.read_metadata(PATHS["epc"]).num_rows
    epc = epc_table.to_pandas()
    del epc_table
    print(f"    {n_total:,} total EPC records, {len(epc):,} matching city UPRNs")

    if uprn_col_uprn in uprn_gdf.columns and uprn_col_epc in epc.columns:
        # Select EPC columns to join (exclude UPRN itself and geometry-related)
        epc_cols = [
            c
            for c in epc.columns
            if c.lower()
            not in ["uprn", "geometry", "lat", "lon", "latitude", "longitude"]
        ]

        # Merge on UPRN
        epc_for_join = epc[[uprn_col_epc] + epc_cols].copy()
        epc_for_join = epc_for_join.rename(columns={uprn_col_epc: uprn_col_uprn})

        # Take most recent EPC per UPRN if duplicates exist
        if "INSPECTION_DATE" in epc_for_join.columns:
            epc_for_join = epc_for_join.sort_values("INSPECTION_DATE", ascending=False)
            epc_for_join = epc_for_join.drop_duplicates(
                subset=[uprn_col_uprn], keep="first"
            )

        before_count = len(uprn_gdf)
        uprn_gdf = uprn_gdf.merge(
            epc_for_join,
            on=uprn_col_uprn,
            how="left",
        )
        matched = uprn_gdf[epc_cols[0]].notna().sum() if epc_cols else 0
        print(f"    ✓ {matched}/{before_count} UPRNs matched to EPC records")
        print(f"    EPC columns added: {len(epc_cols)}")
    else:
        print("    ✗ UPRN column not found - skipping")

    # 4. Nearest: UPRN → Street segment (network metrics)
    print("\n  4. Joining UPRNs to nearest street segments...")
    if segments is not None and len(segments) > 0:
        # Get segment centroids for nearest neighbour search
        segment_points = segments.geometry.centroid
        segment_coords = list(zip(segment_points.x, segment_points.y))

        # Get UPRN coordinates
        uprn_coords = list(zip(uprn_gdf.geometry.x, uprn_gdf.geometry.y))

        # Build KDTree for efficient nearest neighbour search
        tree = cKDTree(segment_coords)
        distances, indices = tree.query(uprn_coords, k=1)

        # Get metric columns (exclude geometry, coords, raw node fields)
        exclude = {
            "geometry", "geom", "x", "y",
            "index", "ns_node_idx", "live", "weight",
        }
        network_cols = [c for c in segments.columns if c not in exclude]

        # Transfer network metrics from nearest segment
        # (all at once to avoid DataFrame fragmentation)
        nearest_data = segments.iloc[indices][network_cols].reset_index(drop=True)
        nearest_data["segment_distance_m"] = distances
        uprn_gdf = pd.concat(
            [uprn_gdf.reset_index(drop=True), nearest_data], axis=1
        )
        uprn_gdf = gpd.GeoDataFrame(uprn_gdf, geometry="geometry", crs=boundaries.crs)

        print(f"    ✓ All {len(uprn_gdf)} UPRNs linked to nearest segment")
        print(f"    Network columns added: {len(network_cols)}")
        print(f"    Mean distance to segment: {distances.mean():.1f}m")
    else:
        print("    ✗ Network segments not available - skipping")

    # Summary
    n_cols = len(uprn_gdf.columns)
    print(f"\n  Final UPRN dataset: {len(uprn_gdf)} records, {n_cols} columns")

    return uprn_gdf


def print_summary(
    buildings: gpd.GeoDataFrame | None,
    segments: gpd.GeoDataFrame | None,
    uprns: gpd.GeoDataFrame | None,
) -> None:
    """Print a summary of the test pipeline results."""
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    stages = [
        ("Stage 1: Morphology", buildings),
        ("Stage 2: Network", segments),
        ("Stage 3: UPRN Integration", uprns),
    ]

    for name, result in stages:
        if result is not None:
            print(f"  ✓ {name}: {len(result)} records")
        else:
            print(f"  ✗ {name}: Not complete")


def save_city_outputs(
    city_name: str,
    buildings: gpd.GeoDataFrame | None,
    segments: gpd.GeoDataFrame | None,
    uprns: gpd.GeoDataFrame | None,
) -> None:
    """Save pipeline outputs for a single city."""
    city_dir = OUTPUT_DIR / city_name
    city_dir.mkdir(parents=True, exist_ok=True)

    if buildings is not None:
        path = city_dir / "buildings_morphology.gpkg"
        buildings.to_file(path, driver="GPKG")
        print(f"  Saved: {path}")

    if segments is not None:
        path = city_dir / "network_segments.gpkg"
        segments.to_file(path, driver="GPKG")
        print(f"  Saved: {path}")

    if uprns is not None:
        if not isinstance(uprns, gpd.GeoDataFrame):
            uprns = gpd.GeoDataFrame(uprns, geometry="geometry")

        # Drop extra geometry columns (from joins)
        geom_cols = [
            c
            for c in uprns.columns
            if c != "geometry" and (uprns[c].dtype == "geometry" or c == "geom")
        ]
        if geom_cols:
            uprns = uprns.drop(columns=geom_cols)

        path = city_dir / "uprn_integrated.gpkg"
        uprns.to_file(path, driver="GPKG")
        print(f"  Saved: {path}")


def combine_cities(city_names: list[str]) -> None:
    """
    Combine per-city UPRN datasets into a single file with a city column.

    Writes incrementally (one city at a time) to avoid loading all cities
    into memory simultaneously. Also writes to the legacy ``test/`` path
    for backward compatibility with existing analysis scripts.
    """
    print()
    print("=" * 60)
    print("COMBINING CITIES")
    print("=" * 60)

    combined_dir = OUTPUT_DIR / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_path = combined_dir / "uprn_integrated.gpkg"

    # Remove existing file so we start fresh
    if combined_path.exists():
        combined_path.unlink()

    total = 0
    file_created = False

    for name in city_names:
        path = OUTPUT_DIR / name / "uprn_integrated.gpkg"
        if not path.exists():
            print(f"  SKIP: {name} (no output yet)")
            continue

        gdf = gpd.read_file(path)
        gdf["city"] = name
        print(f"  {name}: {len(gdf):,} UPRNs")

        if not file_created:
            gdf.to_file(combined_path, driver="GPKG")
            file_created = True
        else:
            gdf.to_file(combined_path, driver="GPKG", mode="a")

        total += len(gdf)
        del gdf
        gc.collect()

    if total == 0:
        print("  No city data to combine.")
        return

    print(f"\n  Combined: {total:,} UPRNs -> {combined_path}")

    # Backward compatibility: copy to test/ for existing analysis scripts
    import shutil

    test_dir = OUTPUT_DIR / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    compat_path = test_dir / "uprn_integrated.gpkg"
    shutil.copy2(combined_path, compat_path)
    print(f"  Compat:   {compat_path}")


def process_city(
    bua_code: str,
    city_name: str,
    *,
    test_mode: bool = False,
) -> None:
    """
    Run the full pipeline for a single city.

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
    print(f"PROCESSING: {city_name} ({bua_code})")
    print("=" * 60)

    # Check if already processed — but verify output quality
    uprn_path = OUTPUT_DIR / city_name / "uprn_integrated.gpkg"
    if uprn_path.exists():
        # Quick check: does it have cityseer columns?
        # (fiona truncates schema at ~200 cols, use geopandas instead)
        _probe = gpd.read_file(uprn_path, rows=1)
        has_cc = any(c.startswith("cc_") for c in _probe.columns)
        del _probe
        if has_cc:
            print(f"  Already processed (with cityseer metrics): {uprn_path}")
            print("  Delete to reprocess.")
            return
        else:
            print(f"  Incomplete output (no cityseer metrics): {uprn_path}")
            print("  Deleting stale outputs and reprocessing...")
            uprn_path.unlink()
            seg_path = OUTPUT_DIR / city_name / "network_segments.gpkg"
            if seg_path.exists():
                seg_path.unlink()

    boundary = load_boundary(bua_code)

    # Stage 1: Morphology
    buildings = run_stage1_morphology(boundary)

    # Stage 2: Network Analysis
    segments = run_stage2_network(boundary, buildings=buildings, test_mode=test_mode)

    # Stage 3: UPRN Integration
    uprns = run_stage3_uprn_integration(boundary, buildings, segments)

    # Save per-city outputs
    save_city_outputs(city_name, buildings, segments, uprns)

    # Summary
    print_summary(buildings, segments, uprns)


def main() -> None:
    """Run the pipeline for all (or selected) cities."""
    # Parse CLI args: city names to process (default: all)
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
    print("URBAN ENERGY PROCESSING PIPELINE")
    print("=" * 60)
    print(f"Cities: {list(selected.values())}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    check_inputs()

    # Process each city independently
    for bua_code, city_name in selected.items():
        process_city(bua_code, city_name, test_mode=False)
        gc.collect()

    # Combine all available city outputs
    combine_cities(list(CITIES.values()))


if __name__ == "__main__":
    main()
