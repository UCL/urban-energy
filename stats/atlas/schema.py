"""
Atlas output schema — single source of truth for what gets exported.

When B1 (CO2/£) lands, add fuel-split fields to OA_PROPERTIES and populate
the `units` block in summary.py. When B4 (scenarios) lands, populate the
`scenarios` block. The frontend reads SCHEMA_VERSION to detect breaking
changes.
"""

from pathlib import Path

from urban_energy.paths import DATA_DIR, PROJECT_DIR

SCHEMA_VERSION = "0.1"

# Frontend assets directory — written GeoJSON + summary.json land here
OUTPUT_DIR = PROJECT_DIR / "stats" / "nepi_static" / "atlas"

# Cache for computed national NEPI dataframe (parquet, ~50 MB)
NEPI_CACHE_PATH = DATA_DIR / "stats" / "nepi_national.parquet"

# Per-OA properties exported to GeoJSON.
# Kept minimal: the frontend computes derived values (CO2, £, scenario projections)
# by combining these with the conversion factors in summary.json.
OA_PROPERTIES: list[str] = [
    # Identity
    "OA21CD",
    "city",
    # Banding (computed on national distribution)
    "nepi_band",
    "nepi_total_kwh",
    # Three NEPI surfaces, all kWh/hh/yr
    "nepi_form_kwh",
    "nepi_mobility_kwh",
    "nepi_access_kwh",
    # Form surface fuel split — needed for per-OA CO2/£ conversion under
    # grid decarbonisation scenarios.
    "oa_elec_mean_kwh",
    "oa_gas_mean_kwh",
    # Mobility surface fleet composition — per-OA share of registered cars
    # that are battery-electric (DVLA via LSOA → OA). Lets the Mobility
    # surface ride grid decarbonisation per-OA, not as a national blend.
    "bev_share",
    # Right-panel context
    "dominant_type",
    "local_coverage",
    # XGBoost feature inputs — bundled into pmtiles so the in-browser model
    # can predict each OA's NEPI from these 9 features and explain the
    # prediction via per-feature contributions ("what's driving this rating").
    "people_per_ha",
    "pct_detached",
    "pct_semi",
    "pct_terraced",
    "pct_flat",
    "cc_bus_800_wt",
    "cc_rail_800_wt",
    "median_build_year",
]

# Coordinate precision for GeoJSON output. 5 decimals ≈ 1 m at UK latitudes —
# more than enough for OA-polygon display at any zoom level.
GEOJSON_COORDINATE_PRECISION: int = 5

# Vector tile zoom range. 8 is "city overview" (whole BUA in one tile);
# 14 is street-level detail. Below 8 OA polygons collapse to specks; above
# 14 we'd be over-rendering. tippecanoe simplifies geometry per zoom.
TIPPECANOE_MIN_ZOOM: int = 8
TIPPECANOE_MAX_ZOOM: int = 14

# Vector tile layer name (referenced by the frontend's `source-layer`).
TILE_LAYER_NAME: str = "oas"
