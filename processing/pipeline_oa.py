"""
National OA-level processing pipeline for urban energy analysis.

Processes ALL English Built-Up Areas using the CityNetwork API (cityseer 4.25+).

Three stages per BUA:
  Stage 1: Building morphology (from cached LiDAR + momepy metrics)
  Stage 2: Network analysis (CityNetwork: centrality + accessibility)
  Stage 3: OA aggregation (transient UPRN joins → aggregate to OA polygons)

Output structure:
    processing/{bua_name}/oa_integrated.gpkg   — per-BUA OA polygons
    processing/combined/oa_integrated.gpkg     — all BUAs merged

Usage:
    uv run python processing/pipeline_oa.py                  # all BUAs
    uv run python processing/pipeline_oa.py cambridge        # by name
    uv run python processing/pipeline_oa.py E63010556        # by code
"""

import gc
import re
import sys
import traceback
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.spatial import cKDTree  # type: ignore[unresolved-import]

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

from archive.pipeline_lsoa import (  # noqa: E402
    _MORPH_MEAN_COLS,
    _MORPH_SUM_COLS,
    _TS001_POP,
    _TS006_DENSITY,
    ERA_MAP,
    PATHS,
    run_stage1_morphology,
)

from urban_energy.paths import DATA_DIR, PROCESSING_DIR  # noqa: E402

OUTPUT_DIR = PROCESSING_DIR

# OA-specific paths
OA_PATHS = {
    "energy_stats_oa": DATA_DIR / "statistics" / "oa_energy_consumption.parquet",
    "imd": DATA_DIR / "statistics" / "lsoa_imd2025.parquet",
    "vehicles": DATA_DIR / "statistics" / "lsoa_vehicles.parquet",
}

# Analysis distances (meters, based on ~80m/min walking speed)
CENTRALITY_DISTANCES = [800, 1600, 3200, 4800, 9600]
ACCESSIBILITY_DISTANCES = [400, 800, 1600, 4800]
ASSUMED_STOREY_HEIGHT_M = 3.0


# ---------------------------------------------------------------------------
# BUA loading
# ---------------------------------------------------------------------------


def _sanitise_name(name: str) -> str:
    """Convert a BUA name to a filesystem-safe directory name."""
    name = name.lower().strip()
    name = re.sub(r"\s*\([^)]*\)\s*", "", name)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def load_all_buas() -> dict[str, str]:
    """
    Load all English BUAs from the boundaries file.

    Returns
    -------
    dict[str, str]
        Mapping from BUA22CD to sanitised name.
    """
    bua = gpd.read_file(PATHS["boundaries"])
    # Sort largest first — heavy cities processed early for better ETA
    bua["_area"] = bua.geometry.area
    bua = bua.sort_values("_area", ascending=False)
    return {
        row["BUA22CD"]: _sanitise_name(row["BUA22NM"])
        for _, row in bua.iterrows()
    }


def load_boundary(bua_code: str) -> gpd.GeoDataFrame:
    """Load a single BUA boundary geometry."""
    boundaries = gpd.read_file(PATHS["boundaries"])
    boundary = boundaries[boundaries["BUA22CD"] == bua_code].copy()
    if len(boundary) == 0:
        raise ValueError(f"BUA code {bua_code} not found")
    row = boundary.iloc[0]
    area_km2 = row.geometry.area / 1e6
    print(f"  Boundary: {row['BUA22CD']} ({row['BUA22NM']}): {area_km2:.2f} km2")
    return boundary


# ---------------------------------------------------------------------------
# Stage 2: Network Analysis (CityNetwork API)
# ---------------------------------------------------------------------------


def run_stage2_network(
    boundaries: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame | None:
    """
    Stage 2: Network analysis using CityNetwork API.

    Loads OS Open Roads directly as a GeoDataFrame within a buffered bounding
    box, constructs a CityNetwork, and computes centrality, building stats,
    and land-use accessibility. The boundary polygon marks live nodes; buffer
    nodes provide network context but are discarded in the output.

    Parameters
    ----------
    boundaries : gpd.GeoDataFrame
        Study area boundaries (BUA polygon).
    buildings : gpd.GeoDataFrame or None
        Buildings with morphology metrics for statistical aggregation.

    Returns
    -------
    gpd.GeoDataFrame or None
        Live network nodes with centrality and accessibility columns.
    """
    from cityseer.network import CityNetwork

    print()
    print("=" * 60)
    print("STAGE 2: NETWORK ANALYSIS (CityNetwork)")
    print("=" * 60)

    max_distance = max(max(CENTRALITY_DISTANCES), max(ACCESSIBILITY_DISTANCES))
    buffer_m = int(max_distance * 1.25)
    print(f"  Buffer: {buffer_m}m")

    combined_bounds = boundaries.union_all()
    buffered_bounds = combined_bounds.buffer(buffer_m)

    # ------------------------------------------------------------------
    # Load roads directly as GeoDataFrame → CityNetwork
    # ------------------------------------------------------------------
    print("\n  Loading OS Open Roads (road_link)...")
    bbox = buffered_bounds.bounds
    roads_gdf = gpd.read_file(
        PATHS["roads"],
        layer="road_link",
        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
    )
    print(f"    {len(roads_gdf):,} road links in buffer")

    if len(roads_gdf) == 0:
        print("  WARNING: No roads found — skipping")
        return None

    # Build CityNetwork directly from GeoDataFrame
    # boundary= sets live/dead nodes (live = within BUA polygon)
    print("  Building CityNetwork from GeoDataFrame...")
    cn = CityNetwork.from_geopandas(roads_gdf, boundary=combined_bounds)
    print(f"    Nodes: {cn.node_count}")

    if cn.node_count < 3:
        print("  WARNING: Too few nodes — skipping")
        return None

    # ------------------------------------------------------------------
    # Centrality
    # ------------------------------------------------------------------
    print(f"\n  Computing centrality (distances={CENTRALITY_DISTANCES})...")
    cn = cn.centrality_shortest(
        distances=CENTRALITY_DISTANCES,
        compute_closeness=True,
        compute_betweenness=True,
        sample=True,
    )

    # ------------------------------------------------------------------
    # Building statistics
    # ------------------------------------------------------------------
    if buildings is not None and len(buildings) > 0:
        print(f"\n  Computing building stats (distances={ACCESSIBILITY_DISTANCES})...")
        buildings_pts = buildings.copy()
        buildings_pts["geometry"] = buildings_pts.geometry.centroid
        buildings_in_buffer = buildings_pts[
            buildings_pts.intersects(buffered_bounds)
        ].copy()
        print(f"    Buildings in buffer: {len(buildings_in_buffer)}")

        height_cols = [
            c for c in buildings_in_buffer.columns if c.startswith("height_")
        ]
        numeric_cols = ["footprint_area_m2", "shared_wall_ratio"] + height_cols
        for col in numeric_cols:
            if col in buildings_in_buffer.columns:
                buildings_in_buffer[col] = pd.to_numeric(
                    buildings_in_buffer[col], errors="coerce"
                )

        # Ensure volume
        if (
            "volume_m3" not in buildings_in_buffer.columns
            and "footprint_area_m2" in buildings_in_buffer.columns
            and "height_median" in buildings_in_buffer.columns
        ):
            buildings_in_buffer["volume_m3"] = (
                buildings_in_buffer["footprint_area_m2"]
                * buildings_in_buffer["height_median"]
            )

        # Gross floor area
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

        stats_columns = []
        for c in ["footprint_area_m2", "volume_m3", "gross_floor_area_m2"]:
            if c in buildings_in_buffer.columns:
                stats_columns.append(c)
        stats_columns.extend(height_cols)

        if stats_columns and len(buildings_in_buffer) > 0:
            cn, _ = cn.compute_stats(
                buildings_in_buffer,
                stats_column_labels=stats_columns,
                distances=ACCESSIBILITY_DISTANCES,
            )
            print(f"    Building stats: {len(stats_columns)} columns")

    # ------------------------------------------------------------------
    # Land-use accessibility (single combined call)
    # ------------------------------------------------------------------
    print("\n  Loading land uses...")
    landuse_parts: list[gpd.GeoDataFrame] = []

    # FSA
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
    landuse_parts.append(fsa_in_buffer[["geometry", "landuse"]])
    print(f"    FSA: {len(fsa_in_buffer)}")

    # Greenspace
    greenspace = gpd.read_file(
        PATHS["greenspace"],
        layer="greenspace_site",
        bbox=buffered_bounds.bounds,
    )
    greenspace = greenspace.to_crs(boundaries.crs).copy()
    greenspace["geometry"] = greenspace.geometry.centroid
    greenspace["landuse"] = "greenspace"
    landuse_parts.append(greenspace[["geometry", "landuse"]])
    print(f"    Greenspace: {len(greenspace)}")

    # Transport
    transport = gpd.read_file(PATHS["transport"], bbox=buffered_bounds.bounds)
    transport = transport.to_crs(boundaries.crs)
    transport_buf = transport[transport.intersects(buffered_bounds)].copy()
    bus_types = ["BCT", "BCS", "BCE", "BCQ", "BST"]
    rail_types = ["RSE", "RLY", "PLT", "MET"]
    if len(transport_buf) > 0:
        bus_mask = transport_buf["stop_type"].isin(bus_types)
        rail_mask = transport_buf["stop_type"].isin(rail_types)
        transport_buf.loc[bus_mask, "landuse"] = "bus"
        transport_buf.loc[rail_mask, "landuse"] = "rail"
        transport_tagged = transport_buf[transport_buf["landuse"].notna()]
        landuse_parts.append(transport_tagged[["geometry", "landuse"]])
        print(f"    Transport: bus={bus_mask.sum()}, rail={rail_mask.sum()}")
    else:
        print("    Transport: bus=0, rail=0")

    # Schools
    schools_path = PATHS.get("schools")
    if schools_path and schools_path.exists():
        schools = gpd.read_file(schools_path, bbox=buffered_bounds.bounds)
        schools = schools.to_crs(boundaries.crs)
        schools_buf = schools[schools.intersects(buffered_bounds)].copy()
        schools_buf["landuse"] = "school"
        landuse_parts.append(schools_buf[["geometry", "landuse"]])
        print(f"    Schools: {len(schools_buf)}")

    # Health
    health_path = PATHS.get("health")
    if health_path and health_path.exists():
        health = gpd.read_file(health_path, bbox=buffered_bounds.bounds)
        health = health.to_crs(boundaries.crs)
        health_buf = health[health.intersects(buffered_bounds)].copy()
        type_map = {
            "hospitals": "hospital",
            "gp_practices": "gp_practice",
            "pharmacies": "pharmacy",
        }
        health_buf["landuse"] = (
            health_buf["facility_type"].map(type_map).fillna("health_other")
        )
        landuse_parts.append(health_buf[["geometry", "landuse"]])
        print(f"    Health: {len(health_buf)}")

    # Single combined accessibility call
    if landuse_parts:
        all_landuses = gpd.GeoDataFrame(
            pd.concat(landuse_parts, ignore_index=True),
            crs=boundaries.crs,
        )
        all_keys = sorted(all_landuses["landuse"].unique().tolist())
        print(
            f"\n  Computing accessibility "
            f"({len(all_landuses):,} points, {len(all_keys)} keys)..."
        )
        cn, _ = cn.compute_accessibilities(
            all_landuses,
            landuse_column_label="landuse",
            accessibility_keys=all_keys,
            distances=ACCESSIBILITY_DISTANCES,
        )
        print(f"    Keys: {all_keys}")

    # ------------------------------------------------------------------
    # Extract live nodes only
    # ------------------------------------------------------------------
    nodes_gdf = cn.to_geopandas()

    # Derive FAR and BCR
    for dist in ACCESSIBILITY_DISTANCES:
        catchment_area = np.pi * dist**2
        for prefix, metric in [
            ("far", "gross_floor_area_m2"),
            ("bcr", "footprint_area_m2"),
        ]:
            sum_col = f"cc_{metric}_sum_{dist}_nw"
            if sum_col in nodes_gdf.columns:
                nodes_gdf[f"{prefix}_{dist}"] = nodes_gdf[sum_col] / catchment_area

    if "live" in nodes_gdf.columns:
        live_nodes = nodes_gdf[nodes_gdf["live"]].copy()
    else:
        live_nodes = nodes_gdf.copy()

    cc_cols = [c for c in live_nodes.columns if c.startswith("cc_")]
    print(f"\n  Result: {len(live_nodes)} live nodes, {len(cc_cols)} cc_ columns")

    return live_nodes


# ---------------------------------------------------------------------------
# Stage 3: OA Aggregation
# ---------------------------------------------------------------------------


def run_stage3_oa_aggregation(
    boundaries: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame | None,
    nodes: gpd.GeoDataFrame | None,
    city_name: str,
) -> gpd.GeoDataFrame | None:
    """
    Stage 3: Aggregate all data to Output Area (OA) level.

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
        OA-level dataset with polygon geometry, or None if no data.
    """
    print()
    print("=" * 60)
    print("STAGE 3: OA AGGREGATION")
    print("=" * 60)

    combined_bounds = boundaries.union_all()

    # 1. Load UPRNs
    print("  Loading UPRNs...")
    uprn = gpd.read_file(PATHS["uprn"], bbox=combined_bounds.bounds)
    uprn = uprn.to_crs(boundaries.crs)
    uprn_gdf = uprn[uprn.intersects(combined_bounds)].copy().reset_index(drop=True)
    del uprn
    print(f"    {len(uprn_gdf):,} UPRNs in boundary")

    if len(uprn_gdf) == 0:
        print("  WARNING: No UPRNs found")
        return None

    uprn_col = "UPRN" if "UPRN" in uprn_gdf.columns else "uprn"

    # 2. UPRN → Building
    if buildings is not None:
        morph_cols = [
            c for c in _MORPH_SUM_COLS + _MORPH_MEAN_COLS if c in buildings.columns
        ]
        if "height_mean" not in buildings.columns:
            for alt in ["height_median", "height_max"]:
                if alt in buildings.columns:
                    morph_cols.append(alt)
                    break
        if morph_cols:
            buildings_for_join = buildings[["geometry"] + morph_cols].copy()
            uprn_gdf = gpd.sjoin(
                uprn_gdf, buildings_for_join, how="left", predicate="within"
            ).drop(columns=["index_right"], errors="ignore")

    # 3. UPRN → Census OA
    census = gpd.read_file(PATHS["census"], bbox=combined_bounds.bounds)
    census = census.to_crs(boundaries.crs)
    census_ts_cols = [c for c in census.columns if c.startswith("ts0")]
    census_id_cols = [c for c in ["OA21CD", "LSOA21CD"] if c in census.columns]
    census_for_join = census[["geometry"] + census_id_cols + census_ts_cols].copy()
    uprn_gdf = gpd.sjoin(
        uprn_gdf, census_for_join, how="left", predicate="within"
    ).drop(columns=["index_right"], errors="ignore")
    uprn_gdf = uprn_gdf.drop_duplicates(subset=[uprn_col], keep="first")

    n_with_oa = uprn_gdf["OA21CD"].notna().sum()
    if n_with_oa == 0:
        print("  WARNING: No UPRNs matched Census OAs")
        return None
    print(f"    {n_with_oa:,} UPRNs with OA assignment")

    oa_geom = (
        census[["OA21CD", "LSOA21CD", "geometry"]]
        .drop_duplicates(subset=["OA21CD"])
        .reset_index(drop=True)
    )
    del census
    gc.collect()

    # 4. UPRN → EPC
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
        epc = pq.read_table(
            epc_path, columns=epc_read_cols, filters=epc_filter
        ).to_pandas()
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
        has_epc_coverage = "PROPERTY_TYPE" in uprn_gdf.columns
        has_build_year = "CONSTRUCTION_AGE_BAND" in uprn_gdf.columns
        del epc
        gc.collect()

    # 5. UPRN → Network node (cKDTree)
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
        node_centroids = nodes.geometry.centroid
        node_coords = np.column_stack([node_centroids.x, node_centroids.y])
        uprn_coords = np.column_stack([uprn_gdf.geometry.x, uprn_gdf.geometry.y])
        tree = cKDTree(node_coords)
        dists, indices = tree.query(uprn_coords, k=1)
        nearest_data = nodes.iloc[indices][network_cols].reset_index(drop=True)
        uprn_gdf = pd.concat([uprn_gdf.reset_index(drop=True), nearest_data], axis=1)
        uprn_gdf = gpd.GeoDataFrame(uprn_gdf, geometry="geometry", crs=boundaries.crs)
        print(
            f"    Network join: {len(network_cols)} cols, mean dist {dists.mean():.0f}m"
        )

    # ==================================================================
    # AGGREGATE TO OA
    # ==================================================================
    uprn_gdf = uprn_gdf[uprn_gdf["OA21CD"].notna()].copy()
    for col in _MORPH_SUM_COLS + _MORPH_MEAN_COLS:
        if col in uprn_gdf.columns:
            uprn_gdf[col] = pd.to_numeric(uprn_gdf[col], errors="coerce")

    # Census dedup
    census_cols = [c for c in uprn_gdf.columns if c.startswith("ts0")]
    oa_census = uprn_gdf.groupby("OA21CD")[census_cols].first()
    if _TS006_DENSITY in oa_census.columns and _TS001_POP in oa_census.columns:
        oa_pop = pd.to_numeric(oa_census[_TS001_POP], errors="coerce")
        oa_dens = pd.to_numeric(oa_census[_TS006_DENSITY], errors="coerce")
        oa_census["_oa_area_km2"] = oa_pop / oa_dens.replace(0, np.nan)

    # UPRN aggregation
    n_uprns = uprn_gdf.groupby("OA21CD").size().reset_index(name="n_uprns")
    agg_dict: dict[str, str] = {}
    for c in [c for c in _MORPH_SUM_COLS if c in uprn_gdf.columns]:
        agg_dict[c] = "sum"
    for c in [c for c in _MORPH_MEAN_COLS if c in uprn_gdf.columns]:
        agg_dict[c] = "mean"
    for c in [c for c in uprn_gdf.columns if c.startswith(("cc_", "far_", "bcr_"))]:
        agg_dict[c] = "mean"
    oa_agg = uprn_gdf.groupby("OA21CD").agg(agg_dict).reset_index()

    # EPC
    epc_pieces: list[pd.DataFrame] = []
    if has_epc_coverage:
        n_epc = (
            uprn_gdf.groupby("OA21CD")["PROPERTY_TYPE"]
            .apply(lambda s: s.notna().sum())
            .reset_index(name="n_epc")
        )
        epc_pieces.append(n_epc)
    if has_build_year:
        age = uprn_gdf[["OA21CD", "CONSTRUCTION_AGE_BAND"]].copy()
        age = age[age["CONSTRUCTION_AGE_BAND"].notna()].copy()
        age["_year"] = age["CONSTRUCTION_AGE_BAND"].map(ERA_MAP)
        unmapped = age["_year"].isna()
        age.loc[unmapped, "_year"] = pd.to_numeric(
            age.loc[unmapped, "CONSTRUCTION_AGE_BAND"], errors="coerce"
        )
        age = age[age["_year"].notna()]
        if len(age) > 0:
            epc_pieces.append(
                age.groupby("OA21CD")["_year"]
                .median()
                .reset_index(name="median_build_year")
            )

    # Assemble
    oa = oa_geom.copy()
    oa = oa.merge(n_uprns, on="OA21CD", how="inner")
    oa = oa.merge(oa_agg, on="OA21CD", how="left")
    oa = oa.merge(oa_census.reset_index(), on="OA21CD", how="left")
    for piece in epc_pieces:
        oa = oa.merge(piece, on="OA21CD", how="left")
    if "n_epc" in oa.columns:
        oa["epc_coverage"] = oa["n_epc"] / oa["n_uprns"]
    if "envelope_area_m2" in oa.columns and "volume_m3" in oa.columns:
        oa["oa_sv"] = oa["envelope_area_m2"] / oa["volume_m3"].replace(0, np.nan)

    # National dataset joins
    for path_key, key_col in [
        ("energy_stats_oa", "OA21CD"),
        ("imd", "LSOA21CD"),
        ("vehicles", "LSOA21CD"),
    ]:
        data_path = OA_PATHS.get(path_key)
        if data_path and data_path.exists():
            df = pd.read_parquet(data_path)
            if path_key == "energy_stats_oa":
                df = df[[c for c in df.columns if c not in ("LSOA21CD",)]]
            elif path_key == "imd":
                df = df[
                    ["LSOA21CD"]
                    + [c for c in df.columns if c.startswith("imd_") and "pop" not in c]
                ]
            oa = oa.merge(df, on=key_col, how="left")

    scaling_path = PATHS.get("scaling")
    if scaling_path and scaling_path.exists():
        scaling_df = pd.read_parquet(scaling_path).rename(
            columns={"LSOA_CODE": "LSOA21CD"}
        )
        oa = oa.merge(scaling_df, on="LSOA21CD", how="left")

    oa["city"] = city_name
    oa = gpd.GeoDataFrame(oa, geometry="geometry", crs=boundaries.crs)

    del uprn_gdf
    gc.collect()

    print(f"  OA output: {len(oa):,} OAs, {len(oa.columns)} columns")
    return oa


# ---------------------------------------------------------------------------
# Save and combine
# ---------------------------------------------------------------------------


def save_city_oa(city_name: str, oa: gpd.GeoDataFrame | None) -> None:
    """Save OA GeoPackage for a single city."""
    if oa is None:
        return
    city_dir = OUTPUT_DIR / city_name
    city_dir.mkdir(parents=True, exist_ok=True)
    geom_cols = [
        c
        for c in oa.columns
        if c != "geometry" and (oa[c].dtype == "geometry" or c == "geom")
    ]
    if geom_cols:
        oa = oa.drop(columns=geom_cols)
    path = city_dir / "oa_integrated.gpkg"
    assert oa is not None
    oa.to_file(path, driver="GPKG")
    print(f"  Saved: {path}")


def combine_oa_cities(city_names: list[str]) -> None:
    """Combine per-city OA datasets into a single GeoPackage."""
    print()
    print("=" * 60)
    print("COMBINING OA OUTPUTS")
    print("=" * 60)

    frames: list[gpd.GeoDataFrame] = []
    for name in city_names:
        path = OUTPUT_DIR / name / "oa_integrated.gpkg"
        if not path.exists():
            continue
        gdf = gpd.read_file(path)
        if "city" not in gdf.columns:
            gdf["city"] = name
        frames.append(gdf)

    if not frames:
        print("  No city data to combine.")
        return

    print(f"  Combining {len(frames)} BUAs...")
    combined = pd.concat(frames, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=frames[0].crs)

    combined_dir = OUTPUT_DIR / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    out_path = combined_dir / "oa_integrated.gpkg"
    combined.to_file(out_path, driver="GPKG")
    print(f"  Combined: {len(combined):,} OAs from {len(frames)} BUAs -> {out_path}")


# ---------------------------------------------------------------------------
# Per-city orchestration
# ---------------------------------------------------------------------------


def process_city(bua_code: str, city_name: str) -> None:
    """Run the full OA pipeline for a single BUA."""
    print()
    print("=" * 60)
    print(f"PROCESSING (OA): {city_name} ({bua_code})")
    print("=" * 60)

    # Skip if already complete
    oa_path = OUTPUT_DIR / city_name / "oa_integrated.gpkg"
    if oa_path.exists():
        _probe = gpd.read_file(oa_path, rows=1)
        has_cc = any(c.startswith("cc_") for c in _probe.columns)
        del _probe
        if has_cc:
            print(f"  Already processed: {oa_path}")
            return
        oa_path.unlink()

    boundary = load_boundary(bua_code)

    # Stage 1: Morphology
    buildings = run_stage1_morphology(boundary)

    # Stage 2: Network — use cache if available
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
        nodes = run_stage2_network(boundary, buildings=buildings)
        if nodes is not None:
            city_dir = OUTPUT_DIR / city_name
            city_dir.mkdir(parents=True, exist_ok=True)
            nodes.to_file(nodes_cache, driver="GPKG")
            print(f"  Saved network cache: {nodes_cache}")

    # Stage 3: OA Aggregation
    oa = run_stage3_oa_aggregation(boundary, buildings, nodes, city_name)
    save_city_oa(city_name, oa)

    if oa is not None:
        print(f"\n  {city_name}: {len(oa):,} OAs, {len(oa.columns)} columns")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the OA pipeline for all (or selected) BUAs."""
    all_buas = load_all_buas()

    args = sys.argv[1:]
    if args:
        selected = {}
        for code, name in all_buas.items():
            if name in args or code in args:
                selected[code] = name
        # Partial match fallback
        if not selected:
            for code, name in all_buas.items():
                for arg in args:
                    if arg.lower() in name:
                        selected[code] = name
        if not selected:
            print(f"No BUAs matched: {args}")
            sys.exit(1)
    else:
        selected = all_buas

    print()
    print("=" * 60)
    print("URBAN ENERGY OA PIPELINE (NATIONAL)")
    print("=" * 60)
    print(f"BUAs to process: {len(selected)}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    n_errors = 0
    for i, (bua_code, city_name) in enumerate(selected.items()):
        print(f"\n[{i + 1}/{len(selected)}] {city_name} ({bua_code})")
        try:
            process_city(bua_code, city_name)
        except Exception:
            print(f"  ERROR processing {city_name}:")
            traceback.print_exc()
            n_errors += 1
        gc.collect()

    if n_errors > 0:
        print(f"\n  {n_errors} BUAs failed")

    combine_oa_cities(list(selected.values()))


if __name__ == "__main__":
    main()
