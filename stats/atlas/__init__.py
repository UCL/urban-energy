"""
NEPI Atlas data pipeline.

Produces the artefacts consumed by the static MapLibre frontend in
[stats/nepi_static/](../nepi_static/):

    {bua}_oas.geojson        OA polygons + per-OA NEPI properties
    summary.json             band thresholds, distributions, units, scenarios

The pipeline has three stages, each independently testable:

    compute   compute_national_nepi()      → cached parquet of all-England NEPI
    geometry  load_bua_geometry(bua)       → deduplicated OA polygons in 4326
    summary   build_summary(...)           → JSON with extension points for
                                             units (B1) and scenarios (B4)

The CLI entry point is `stats/export_atlas_data.py`.
"""

from .compute import compute_national_nepi
from .export import export_bua, export_region
from .geometry import attach_region_geometry, load_oa_geometry_in_polygon
from .region import REGIONS, BUARegion, LADRegion, NationalRegion, Region, get_region
from .schema import OA_PROPERTIES, OUTPUT_DIR, SCHEMA_VERSION
from .summary import build_summary

__all__ = [
    "BUARegion",
    "LADRegion",
    "NationalRegion",
    "OA_PROPERTIES",
    "OUTPUT_DIR",
    "REGIONS",
    "Region",
    "SCHEMA_VERSION",
    "attach_region_geometry",
    "build_summary",
    "compute_national_nepi",
    "export_bua",
    "export_region",
    "get_region",
    "load_oa_geometry_in_polygon",
]
