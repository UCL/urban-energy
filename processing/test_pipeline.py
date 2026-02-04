"""
Test pipeline for validating the full processing workflow on small boundaries.

Runs all three processing stages on 1-2 small Built-Up Areas to validate
the pipeline before running at scale.

Usage:
    uv run python processing/test_pipeline.py

Test boundaries:
    - E63009036 (Keele): 399 buildings, 8 FSA, 20 transport stops
    - E63007158 (Woolsington): 265 buildings, 11 FSA, 19 transport stops
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree

# Configuration
BASE_DIR = Path(__file__).parent.parent
TEMP_DIR = BASE_DIR / "temp"
TEST_OUTPUT_DIR = TEMP_DIR / "processing" / "test"

# Test boundaries - small BUAs with good data coverage
TEST_BOUNDARIES = [
    "E63008401",  # Manchester
]

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


def load_test_boundaries() -> gpd.GeoDataFrame:
    """Load the test boundary geometries."""
    boundaries = gpd.read_file(PATHS["boundaries"])
    test_bounds = boundaries[boundaries["BUA22CD"].isin(TEST_BOUNDARIES)].copy()

    print(f"\nTest boundaries loaded: {len(test_bounds)}")
    for _, row in test_bounds.iterrows():
        area_km2 = row.geometry.area / 1e6
        print(f"  - {row['BUA22CD']} ({row['BUA22NM']}): {area_km2:.2f} km²")

    return test_bounds


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

    missing_cols = [c for c in required_cols if c not in buildings.columns]
    if missing_cols:
        print(f"  WARNING: Missing columns: {missing_cols}")
    else:
        print("  ✓ All morphology columns present")

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
    try:
        bbox = buffered_bounds.bounds
        nx_graph = io.nx_from_open_roads(
            PATHS["roads"],
            target_bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
        )
        print(
            f"    Network: {nx_graph.number_of_nodes()} nodes, "
            f"{nx_graph.number_of_edges()} edges"
        )
    except Exception as e:
        print(f"    ERROR loading network: {e}")
        return None

    # Prepare network for analysis
    print("  Preparing network...")
    try:
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
    except Exception as e:
        print(f"    ERROR preparing network: {e}")
        return None

    # Compute centrality
    print(f"\n  Computing centrality (distances={CENTRALITY_DISTANCES})...")
    try:
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
    except Exception as e:
        print(f"    ERROR computing centrality: {e}")

    # Compute building statistics over the network (400m only for local context)
    STATS_DISTANCES = [400]
    if buildings is not None and len(buildings) > 0:
        print(f"\n  Computing building stats (distances={STATS_DISTANCES})...")
        try:
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
            numeric_cols = ["footprint_area_m2"] + height_cols
            for col in numeric_cols:
                if col in buildings_in_buffer.columns:
                    buildings_in_buffer[col] = pd.to_numeric(
                        buildings_in_buffer[col], errors="coerce"
                    )

            # Compute building volume (footprint × height)
            if (
                "footprint_area_m2" in buildings_in_buffer.columns
                and "height_median" in buildings_in_buffer.columns
            ):
                buildings_in_buffer["volume_m3"] = (
                    buildings_in_buffer["footprint_area_m2"]
                    * buildings_in_buffer["height_median"]
                )

            # Select numeric columns to aggregate
            stats_columns = []
            if "footprint_area_m2" in buildings_in_buffer.columns:
                stats_columns.append("footprint_area_m2")
            if "volume_m3" in buildings_in_buffer.columns:
                stats_columns.append("volume_m3")
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
        except Exception as e:
            print(f"    ERROR computing building stats: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("\n  Skipping building statistics (no buildings provided)")

    # Load land use data for accessibility
    print("\n  Loading land use data...")

    # FSA establishments - categorize by business type
    fsa = gpd.read_file(PATHS["fsa"])
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
    transport = gpd.read_file(PATHS["transport"])
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
    try:
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

    except Exception as e:
        print(f"    ERROR computing accessibility: {e}")
        import traceback

        traceback.print_exc()

    # Filter to nodes within study area (not just buffer)
    nodes_in_bounds = nodes_gdf[nodes_gdf.intersects(combined_bounds)]
    print(
        f"\n  Nodes in study area: {len(nodes_in_bounds)} (of {len(nodes_gdf)} total)"
    )

    # Summary of columns
    all_cols = list(nodes_gdf.columns)
    metric_cols = [c for c in all_cols if c not in ["geometry", "x", "y"]]
    print(f"  Metric columns: {metric_cols[:8]}...")

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
    census = gpd.read_file(PATHS["census"])
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

    # 3. Direct join: UPRN → EPC (energy performance)
    print("\n  3. Joining UPRNs to EPC records...")
    epc = pd.read_parquet(PATHS["epc"])
    print(f"    {len(epc)} total EPC records")

    # Find UPRN column in both datasets
    uprn_col_uprn = "UPRN" if "UPRN" in uprn_gdf.columns else "uprn"
    uprn_col_epc = "UPRN" if "UPRN" in epc.columns else "uprn"

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

        # Get network metric columns (exclude geometry, x, y)
        network_cols = [c for c in segments.columns if c not in ["geometry", "x", "y"]]

        # Transfer network metrics from nearest segment
        for col in network_cols:
            uprn_gdf[col] = segments.iloc[indices][col].values

        # Add distance to nearest segment
        uprn_gdf["segment_distance_m"] = distances

        print(f"    ✓ All {len(uprn_gdf)} UPRNs linked to nearest segment")
        print(f"    Network columns added: {len(network_cols)}")
        print(f"    Mean distance to segment: {distances.mean():.1f}m")
    else:
        print("    ✗ Network segments not available - skipping")

    # Summary
    print(
        f"\n  Final UPRN dataset: {len(uprn_gdf)} records, {len(uprn_gdf.columns)} columns"
    )

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


def save_outputs(
    buildings: gpd.GeoDataFrame | None,
    segments: gpd.GeoDataFrame | None,
    uprns: gpd.GeoDataFrame | None,
) -> None:
    """Save pipeline outputs to geopackages."""
    print()
    print("=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)

    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if buildings is not None:
        path = TEST_OUTPUT_DIR / "buildings_morphology.gpkg"
        buildings.to_file(path, driver="GPKG")
        print(f"  ✓ Buildings: {path}")

    if segments is not None:
        path = TEST_OUTPUT_DIR / "network_segments.gpkg"
        segments.to_file(path, driver="GPKG")
        print(f"  ✓ Network segments: {path}")

    if uprns is not None:
        # Ensure it's a GeoDataFrame before saving
        if not isinstance(uprns, gpd.GeoDataFrame):
            uprns = gpd.GeoDataFrame(uprns, geometry="geometry")

        # Drop extra geometry columns (from joins) - keep only the UPRN point geometry
        # Also drop "geom" column from cityseer network output
        geom_cols = [
            c
            for c in uprns.columns
            if c != "geometry"
            and (
                uprns[c].dtype == "geometry" or c == "geom"  # cityseer network column
            )
        ]
        if geom_cols:
            uprns = uprns.drop(columns=geom_cols)

        path = TEST_OUTPUT_DIR / "uprn_integrated.gpkg"
        uprns.to_file(path, driver="GPKG")
        print(f"  ✓ UPRNs: {path}")


def main() -> None:
    """Run the test pipeline."""
    print()
    print("=" * 60)
    print("URBAN ENERGY TEST PIPELINE")
    print("=" * 60)
    print(f"Test boundaries: {TEST_BOUNDARIES}")
    print(f"Output directory: {TEST_OUTPUT_DIR}")
    print()

    # Create output directory
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check inputs
    input_status = check_inputs()

    # Load test boundaries
    boundaries = load_test_boundaries()

    # Stage 1: Morphology
    buildings = run_stage1_morphology(boundaries)

    # Stage 2: Network Analysis (with building statistics)
    segments = run_stage2_network(boundaries, buildings=buildings, test_mode=True)

    # Stage 3: UPRN Integration
    uprns = run_stage3_uprn_integration(boundaries, buildings, segments)

    # Save outputs
    save_outputs(buildings, segments, uprns)

    # Summary
    print_summary(buildings, segments, uprns)


if __name__ == "__main__":
    main()
