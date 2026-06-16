"""
Straight-line access per Output Area.

For each OA centroid, count the everyday destinations within ``radius`` and the
distance to the nearest of each, using a KD-tree over the point layers in
``$DATA_DIR`` (NHS, GIAS, FSA food + grocery, OS greenspace, NaPTAN, and Census
workplace jobs). Straight-line distance is the deliberate simplification:
transparent, reproducible in seconds, and a conservative proxy for walkable access
(it can only over-credit, never under-credit — so the flat→detached gradient it
reports is a floor).

The result is cached at ``$DATA_DIR/statistics/oa_access.parquet`` and consumed by
``oa_data.load_and_aggregate``. Rebuild with ``uv run python stats/oa_access.py``.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from urban_energy.paths import DATA_DIR

#: Walkable / short-trip catchment for the count measure.
RADIUS_M = 1600

_CENSUS = DATA_DIR / "statistics" / "census_oa_joined.gpkg"
_CACHE = DATA_DIR / "statistics" / "oa_access.parquet"
_JOBS = DATA_DIR / "employment" / "workplace_jobs.gpkg"

_BUS = {"BCT", "BCS", "BCE", "BCQ", "BST"}
_RAIL = {"RSE", "RLY", "PLT", "MET"}
_FOOD = {"Restaurant/Cafe/Canteen", "Takeaway/sandwich shop", "Pub/bar/nightclub"}
_GROCERY = {"Retailers - supermarkets/hypermarkets", "Retailers - other"}

#: service -> (gpkg, filter column or None, allowed values or None, layer or None)
_SERVICES: dict[str, tuple] = {
    "gp": (
        DATA_DIR / "health" / "nhs_facilities.gpkg",
        "facility_type",
        {"gp_practices"},
        None,
    ),
    "pharmacy": (
        DATA_DIR / "health" / "nhs_facilities.gpkg",
        "facility_type",
        {"pharmacies"},
        None,
    ),
    "hospital": (
        DATA_DIR / "health" / "nhs_facilities.gpkg",
        "facility_type",
        {"hospitals"},
        None,
    ),
    "school": (DATA_DIR / "education" / "gias_schools.gpkg", None, None, None),
    "food": (
        DATA_DIR / "fsa" / "fsa_establishments.gpkg",
        "business_type",
        _FOOD,
        None,
    ),
    "grocery": (
        DATA_DIR / "fsa" / "fsa_establishments.gpkg",
        "business_type",
        _GROCERY,
        None,
    ),
    "greenspace": (
        DATA_DIR / "opgrsp_gpkg_gb" / "Data" / "opgrsp_gb.gpkg",
        None,
        None,
        "greenspace_site",
    ),
    "bus": (DATA_DIR / "transport" / "naptan_england.gpkg", "stop_type", _BUS, None),
    "rail": (DATA_DIR / "transport" / "naptan_england.gpkg", "stop_type", _RAIL, None),
}

#: Everyday services counted into the access measure (excludes jobs, kept separate).
SERVICES = list(_SERVICES)


def _points(path, col, vals, layer) -> np.ndarray:
    """Load a service layer → (n, 2) array of EPSG:27700 point coords."""
    g = gpd.read_file(
        path, columns=[col] if col else None, **({"layer": layer} if layer else {})
    ).to_crs(27700)
    if col:
        g = g[g[col].isin(vals)]
    p = g.geometry.representative_point()
    return np.c_[p.x.to_numpy(), p.y.to_numpy()]


def compute_access(
    centroids: gpd.GeoDataFrame, radius_m: int = RADIUS_M
) -> pd.DataFrame:
    """
    Per-OA straight-line access: ``{svc}_n`` (count within radius), ``{svc}_near``
    (metres to nearest), and ``jobs_n`` (workplace population reachable, weighted).

    ``centroids`` is one point per OA (any CRS) with an ``OA21CD`` column.
    """
    cen = centroids.to_crs(27700)
    oa_xy = np.c_[cen.geometry.x.to_numpy(), cen.geometry.y.to_numpy()]
    out = pd.DataFrame({"OA21CD": cen["OA21CD"].to_numpy()})

    for name, spec in _SERVICES.items():
        tree = KDTree(_points(*spec))
        out[f"{name}_n"] = tree.query_ball_point(oa_xy, radius_m, return_length=True)
        out[f"{name}_near"] = tree.query(oa_xy, k=1)[0]

    jobs = gpd.read_file(_JOBS).to_crs(27700)
    w = jobs["jobs"].to_numpy(dtype=float)
    nbr = KDTree(np.c_[jobs.geometry.x, jobs.geometry.y]).query_ball_point(
        oa_xy, radius_m
    )
    out["jobs_n"] = [float(w[ix].sum()) for ix in nbr]
    return out.set_index("OA21CD")


def access_table(rebuild: bool = False) -> pd.DataFrame:
    """Cached per-OA access table (built from census OA centroids on first use)."""
    if _CACHE.exists() and not rebuild:
        return pd.read_parquet(_CACHE)
    cen = gpd.read_file(_CENSUS, columns=["OA21CD"])
    cen["geometry"] = cen.geometry.representative_point()
    acc = compute_access(cen)
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    acc.to_parquet(_CACHE)
    return acc


def main() -> None:
    """(Re)build the access cache and print a quick by-service summary."""
    acc = access_table(rebuild=True)
    print(f"  built {_CACHE}  ({len(acc):,} OAs)")
    for s in SERVICES:
        print(f"    {s:<11s} median within {RADIUS_M} m: {acc[f'{s}_n'].median():.0f}")
    print(
        f"    {'jobs':<11s} median within {RADIUS_M} m: {acc['jobs_n'].median():,.0f}"
    )


if __name__ == "__main__":
    main()
