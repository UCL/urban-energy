"""
Region definitions for the Atlas exporter.

A `Region` is anything that can produce a boundary polygon in EPSG:27700.
The exporter spatial-intersects OA centroids with that boundary to determine
which OAs belong. Two concrete kinds of region are provided:

    - BUARegion: a single OS Built-Up Area, identified by BUA22CD.
    - LADRegion: a union of one or more Local Authority Districts (e.g.
      Greater Manchester is the union of 10 LADs).

New regions slot in by adding an entry to `REGIONS`. The CLI looks up by
slug; the frontend uses the slug as the GeoJSON filename stem.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import ClassVar

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from urban_energy.paths import DATA_DIR

BDLINE_PATH = DATA_DIR / "bdline_gpkg_gb" / "Data" / "bdline_gb.gpkg"
BUA_PATH = DATA_DIR / "boundaries" / "built_up_areas.gpkg"


@dataclass(frozen=True)
class Region:
    """Base class — subclasses implement `boundary_27700`."""

    slug: str  # filename-safe identifier; used for output paths
    label: str  # display name for the frontend
    kind: ClassVar[str] = "abstract"

    def boundary_27700(self) -> BaseGeometry:
        raise NotImplementedError


@dataclass(frozen=True)
class BUARegion(Region):
    bua_code: str = ""
    kind: ClassVar[str] = "bua"

    def boundary_27700(self) -> BaseGeometry:
        gdf = gpd.read_file(BUA_PATH, where=f"BUA22CD = '{self.bua_code}'")
        if len(gdf) == 0:
            raise ValueError(f"BUA22CD {self.bua_code!r} not found in {BUA_PATH}")
        return gdf.geometry.union_all()


@dataclass(frozen=True)
class NationalRegion(Region):
    """All OAs in England — no spatial filter applied."""

    kind: ClassVar[str] = "national"

    def boundary_27700(self) -> BaseGeometry | None:
        # Sentinel: None means "no filter" downstream. Loading 177k OA
        # geometries unfiltered is faster than testing centroid-in-polygon
        # against a 350-LAD union.
        return None


@dataclass(frozen=True)
class LADRegion(Region):
    """Region defined as the union of one or more LAD polygons."""

    lad_codes: tuple[str, ...] = ()
    kind: ClassVar[str] = "lad"

    def boundary_27700(self) -> BaseGeometry:
        codes_sql = ",".join(f"'{c}'" for c in self.lad_codes)
        gdf = gpd.read_file(
            BDLINE_PATH,
            layer="district_borough_unitary",
            where=f"Census_Code IN ({codes_sql})",
        )
        if len(gdf) != len(self.lad_codes):
            found = set(gdf["Census_Code"])
            missing = set(self.lad_codes) - found
            raise ValueError(f"LAD codes not found: {missing}")
        return gdf.geometry.union_all()


# Greater Manchester Combined Authority — 10 metropolitan boroughs.
GM_LAD_CODES = (
    "E08000001",  # Bolton
    "E08000002",  # Bury
    "E08000003",  # Manchester
    "E08000004",  # Oldham
    "E08000005",  # Rochdale
    "E08000006",  # Salford
    "E08000007",  # Stockport
    "E08000008",  # Tameside
    "E08000009",  # Trafford
    "E08000010",  # Wigan
)


REGIONS: dict[str, Region] = {
    "manchester": BUARegion(
        slug="manchester",
        label="Manchester (BUA)",
        bua_code="E63008401",
    ),
    "greater_manchester": LADRegion(
        slug="greater_manchester",
        label="Greater Manchester",
        lad_codes=GM_LAD_CODES,
    ),
    "england": NationalRegion(
        slug="england",
        label="England",
    ),
}


@cache
def get_region(slug: str) -> Region:
    if slug not in REGIONS:
        avail = ", ".join(REGIONS)
        raise KeyError(f"Unknown region {slug!r}. Available: {avail}")
    return REGIONS[slug]
