"""
Load OA geometry for an Atlas region.

The integrated GeoPackage at
`$PROCESSING_DIR/combined/oa_integrated.gpkg` contains duplicate OA21CD
rows for some boundary OAs (an OA centroid that falls within two adjacent
BUA polygons). We dedupe defensively here.

Region selection is delegated to `region.Region`: any object that exposes
a `boundary_27700()` polygon is supported.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry

from urban_energy.paths import PROCESSING_DIR

from .region import Region

GPKG_PATH = PROCESSING_DIR / "combined" / "oa_integrated.gpkg"


def load_oa_geometry_in_polygon(
    boundary: BaseGeometry | None, target_crs: int = 4326
) -> gpd.GeoDataFrame:
    """
    Load OAs whose centroid falls within `boundary` (or all OAs if None).

    Uses the polygon's bounding box for an initial bbox-filter (cheap, via
    GeoPackage spatial index) before testing centroid containment. When
    `boundary` is None, returns every unique OA in the source GeoPackage —
    used for the national export.

    Geometry is reprojected to `target_crs` for output.
    """
    if boundary is None:
        gdf = gpd.read_file(GPKG_PATH, columns=["OA21CD", "geometry"])
    else:
        minx, miny, maxx, maxy = boundary.bounds
        gdf = gpd.read_file(
            GPKG_PATH,
            columns=["OA21CD", "geometry"],
            bbox=(minx, miny, maxx, maxy),
        )
    n_before = len(gdf)
    gdf = gdf.drop_duplicates(subset=["OA21CD"]).copy()
    if len(gdf) < n_before:
        print(
            f"  load_oa_geometry: dropped {n_before - len(gdf)} duplicate "
            f"OA21CD rows (cross-BUA boundary OAs)"
        )

    if boundary is not None:
        # Centroid containment — narrows from bbox to true polygon membership
        centroids = gdf.geometry.centroid
        mask = centroids.within(boundary)
        gdf = gdf[mask].copy()

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf.to_crs(target_crs)


def attach_region_geometry(
    region: Region,
    properties: pd.DataFrame,
    target_crs: int = 4326,
) -> gpd.GeoDataFrame:
    """
    Spatial-filter OAs by region, attach per-OA properties, return GeoDataFrame.

    Parameters
    ----------
    region : Region
        Region defining the spatial filter (boundary polygon).
    properties : pd.DataFrame
        Per-OA properties, must include OA21CD. Inner-joined on OA21CD.
    target_crs : int
        EPSG code for output.
    """
    boundary = region.boundary_27700()
    geom = load_oa_geometry_in_polygon(boundary, target_crs=target_crs)
    properties = properties.drop_duplicates(subset=["OA21CD"])
    return geom.merge(properties, on="OA21CD", how="inner")
