"""
National NETWORK access per Output Area — cityseer over OS Open Roads, built ONCE.

The whole England road network is built into a single cityseer Rust structure (never
rebuilt). Every OA is snapped to its nearest node; those nodes are set live and the
**full count-vs-distance curve** is computed in one pass — amenities reachable at
every ladder distance from **1,600 m to the maximum (25,600 m)**. From that one curve
per OA we read three numbers:

1. **walkable catchment** — amenities within a walk (the 1,600 m point);
2. **like-for-like drivable** — amenities within the *same* fixed distance for every
   OA (pure density/connectivity, no catchment scaling);
3. **catchment-scaled drivable** — each OA interpolated at its own car-trip catchment
   (NTS mileage ÷ trips) — the ~2.9×/kWh rate.

Accessibilities-only (no centrality). Output `statistics/oa_network_access.parquet`
with `trip_m`, `net_amen` (own catchment), the full `net_total_{d}` curve, and the
per-service walkable counts `net_{svc}_1600`.

    uv run python stats/oa_network_access.py
    uv run python stats/oa_network_access.py --bbox XMIN YMIN XMAX YMAX  # region test
"""

from __future__ import annotations

import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from cityseer.network import CityNetwork
from oa_access import _SERVICES, DEST, _points, oa_centroids
from oa_data import load_and_aggregate
from scipy.spatial import KDTree

from urban_energy.paths import DATA_DIR

ROADS = DATA_DIR / "oproad_gpkg_gb" / "Data" / "oproad_gb.gpkg"
CACHE = DATA_DIR / "statistics" / "oa_network_access.parquet"

STEP = 1600  # ladder resolution (m)
WALK = 1600  # walkable band / ladder start
MAX_M = 25600  # ladder maximum (≈ p97 of OA catchments)
TRIPS_PER_YEAR = 370  # NTS car/van-driver trips per person per year
_M_PER_MILE = 1609.34
LADDER = list(range(WALK, MAX_M + 1, STEP))  # 1600, 3200, … 25600


def _amenities(bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Everyday-destination points within ``bbox``, tagged by land-use (27700)."""
    parts = []
    for svc in DEST:
        xy = _points(*_SERVICES[svc])
        m = (
            (xy[:, 0] >= bbox[0])
            & (xy[:, 0] <= bbox[2])
            & (xy[:, 1] >= bbox[1])
            & (xy[:, 1] <= bbox[3])
        )
        xy = xy[m]
        parts.append(
            gpd.GeoDataFrame(
                {"landuse": [svc] * len(xy)},
                geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]),
                crs=27700,
            )
        )
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=27700)


def _oa_catchments(bbox: tuple | None = None) -> pd.DataFrame:
    """OA21CD + x/y + catchment ``trip_m`` (NTS mileage ÷ trips)."""
    df = load_and_aggregate()
    xy = oa_centroids(df["OA21CD"])
    trip = (
        pd.to_numeric(df["car_miles_per_person"], errors="coerce")
        / TRIPS_PER_YEAR
        * _M_PER_MILE
    ).clip(WALK, MAX_M)
    oa = pd.DataFrame(
        {
            "OA21CD": df["OA21CD"].to_numpy(),
            "x": xy[:, 0],
            "y": xy[:, 1],
            "trip_m": trip.to_numpy(),
        }
    ).dropna(subset=["x", "y", "trip_m"])
    if bbox is not None:
        oa = oa[
            (oa.x >= bbox[0])
            & (oa.x <= bbox[2])
            & (oa.y >= bbox[1])
            & (oa.y <= bbox[3])
        ]
    return oa.reset_index(drop=True)


def main(bbox: tuple | None = None) -> pd.DataFrame:
    """Build the network once; one pass over the full ladder for all OA nodes."""
    oa = _oa_catchments(bbox)
    rb = (
        bbox
        if bbox is not None
        else (
            oa.x.min() - MAX_M,
            oa.y.min() - MAX_M,
            oa.x.max() + MAX_M,
            oa.y.max() + MAX_M,
        )
    )

    t0 = time.time()
    roads = gpd.read_file(ROADS, layer="road_link", bbox=rb)
    cn = CityNetwork.from_geopandas(roads)
    ns = cn._network_structure
    nodes = cn.nodes_gdf
    nidx = nodes["ns_node_idx"].to_numpy()
    print(
        f"  built ONCE: {cn.node_count:,} nodes · {len(roads):,} links · "
        f"{time.time() - t0:.0f}s",
        flush=True,
    )
    amen = _amenities(rb)

    # snap OAs to nodes; set only those live; one full-ladder pass
    oa_node = KDTree(nodes[["x", "y"]].to_numpy()).query(oa[["x", "y"]].to_numpy())[1]
    for i in nidx:
        ns.set_node_live(int(i), False)
    for i in nidx[np.unique(oa_node)]:
        ns.set_node_live(int(i), True)

    t1 = time.time()
    cn, _ = cn.compute_accessibilities(
        amen, landuse_column_label="landuse", accessibility_keys=DEST, distances=LADDER
    )
    print(f"  accessibility (1,600→{MAX_M} m): {time.time() - t1:.0f}s", flush=True)

    sub = cn.nodes_gdf.iloc[oa_node]
    out = pd.DataFrame(
        {"OA21CD": oa["OA21CD"].to_numpy(), "trip_m": oa["trip_m"].to_numpy()}
    )
    for d in LADDER:  # full per-OA amenity-vs-distance curve
        out[f"net_total_{d}"] = sum(
            sub[f"cc_{s}_{d}"].to_numpy() for s in DEST if f"cc_{s}_{d}" in sub
        )
    for s in DEST:  # per-service walkable counts (richness table)
        out[f"net_{s}_1600"] = (
            sub[f"cc_{s}_1600"].to_numpy() if f"cc_{s}_1600" in sub else 0.0
        )

    # catchment-scaled: interpolate each OA's total curve at its own trip distance
    xp = np.asarray(LADDER, float)
    fp = out[[f"net_total_{d}" for d in LADDER]].to_numpy(float)
    t = np.clip(out["trip_m"].to_numpy(), xp[0], xp[-1])
    j = np.clip(np.searchsorted(xp, t, side="right") - 1, 0, len(xp) - 2)
    w = (t - xp[j]) / (xp[j + 1] - xp[j])
    out["net_amen"] = fp[np.arange(len(fp)), j] + w * (
        fp[np.arange(len(fp)), j + 1] - fp[np.arange(len(fp)), j]
    )
    out = out.set_index("OA21CD")

    if bbox is None:
        out.to_parquet(CACHE)
        print(f"  wrote {CACHE}  ({len(out):,} OAs · {time.time() - t1:.0f}s)")
    else:
        print(f"  {len(out):,} OAs · loop {time.time() - t1:.0f}s")
        print(
            out[["trip_m", "net_total_1600", "net_amen", f"net_total_{MAX_M}"]]
            .describe()
            .round(0)
            .to_string()
        )
    return out


if __name__ == "__main__":
    _bbox = None
    if "--bbox" in sys.argv:
        _i = sys.argv.index("--bbox")
        _bbox = tuple(float(v) for v in sys.argv[_i + 1 : _i + 5])
    main(_bbox)
