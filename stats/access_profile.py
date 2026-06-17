"""
Access profile — three numbers from the network access curve (``oa_network_access``).

All three are network distance over OS Open Roads, on the same ruler (so the walkable
set is a true subset of the drivable):

  [1] WALKABLE CATCHMENT — amenities within a 1,600 m walk: per-service counts + the
      share with ZERO. The richness of the doorstep, reached without travel energy.
  [2] LIKE-FOR-LIKE DRIVABLE — amenities within the SAME fixed distance for every OA,
      flat vs detached at each ladder rung: pure density/connectivity, no catchment
      scaling. Shows the flat's lead at short range narrowing as distance grows.
  [3] DRIVABLE RATE — each OA at its OWN car-trip catchment (NTS mileage ÷ trips) ÷ its
      car-travel energy: the access-per-kWh rate (~2.9×). Same amenities as [1]/[2] at a
      larger radius — paid for in travel energy.

Jobs are reported alongside the amenities in all three sections (a weighted SUM of
reachable jobs rather than a unit count), so the same flat→detached comparison can be
read for employment access.

Run:
    uv run python stats/oa_network_access.py   # build the cache first
    uv run python stats/access_profile.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from oa_access import DEST
from oa_data import load_and_aggregate

from urban_energy.paths import DATA_DIR

NET_CACHE = DATA_DIR / "statistics" / "oa_network_access.parquet"
TYPES = ["Flat", "Terraced", "Semi", "Detached"]
LABELS = {
    "gp": "GP",
    "pharmacy": "Pharmacy",
    "hospital": "Hospital",
    "school": "School",
    "food": "Food",
    "grocery": "Grocery",
    "greenspace": "Greenspace",
}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _med(df: pd.DataFrame, col: str) -> dict[str, float]:
    return {
        t: float(_num(df.loc[df["dominant_type"] == t, col]).median()) for t in TYPES
    }


def _ratio(m: dict[str, float]) -> float:
    return m["Flat"] / m["Detached"] if m["Detached"] else float("nan")


def main() -> None:
    """Print the three access numbers from the network curve cache."""
    df = load_and_aggregate().reset_index(drop=True)
    if not NET_CACHE.exists():
        print(
            f"\n  [network cache not found ({NET_CACHE.name}) — build it first:\n"
            "     uv run python stats/oa_network_access.py]"
        )
        return
    net = pd.read_parquet(NET_CACHE)
    d = df.merge(net, left_on="OA21CD", right_index=True, how="left")
    d["transport"] = _num(d["transport_kwh_per_hh_total_est"])
    ladder = sorted(
        int(c.rsplit("_", 1)[1]) for c in net.columns if c.startswith("net_total_")
    )

    # ---- [1] WALKABLE CATCHMENT (network, within 1,600 m) ----
    print("\n  [1] WALKABLE — network count within 1,600 m, by type (the doorstep)")
    print(f"  {'service':<11s}{'Flat':>8s}{'Det':>8s}{'%Det=0':>9s}")
    for svc in DEST:
        m = _med(d, f"net_{svc}_1600")
        zdet = (
            _num(d.loc[d["dominant_type"] == "Detached", f"net_{svc}_1600"]).fillna(0)
            == 0
        ).mean() * 100
        print(
            f"  {LABELS[svc]:<11s}{m['Flat']:>8.0f}{m['Detached']:>8.0f}{zdet:>8.0f}%"
        )
    d["walk_basket"] = sum((_num(d[f"net_{s}_1600"]) > 0).astype(int) for s in DEST)
    mb = {t: float(d.loc[d["dominant_type"] == t, "walk_basket"].mean()) for t in TYPES}
    print(
        "  walkable basket (of 7 on foot, mean):  "
        + "   ".join(f"{t} {mb[t]:.1f}" for t in TYPES)
    )
    mj = _med(d, "net_jobs_1600")
    print(
        "  jobs within 1,600 m (median):  "
        + "  ".join(f"{t} {mj[t]:,.0f}" for t in TYPES)
        + f"   (Flat:Det {_ratio(mj):.1f}x)"
    )

    # ---- [1b] STRUCTURAL INTENSITY (network + population, within 1,600 m) ----
    print("\n  [1b] STRUCTURAL INTENSITY — by type (compactness → complexity)")
    print(f"  {'':16s}" + "".join(f"{t:>10s}" for t in TYPES) + f"{'Flat:Det':>10s}")
    for label, col, fmt in [
        ("closeness", "net_closeness_1600", "{:>10.2f}"),
        ("node density", "net_density_1600", "{:>10.0f}"),
        ("pop /ha", "pop_density", "{:>10.1f}"),
    ]:
        m = _med(d, col)
        print(
            f"  {label:<16s}"
            + "".join(fmt.format(m[t]) for t in TYPES)
            + f"{_ratio(m):>8.1f}x"
        )

    # ---- [2] LIKE-FOR-LIKE DRIVABLE (same network distance for both) ----
    print("\n  [2] LIKE-FOR-LIKE — amenities within the SAME network distance, by type")
    print(
        f"  {'dist (m)':<10s}"
        + "".join(f"{t:>10s}" for t in TYPES)
        + f"{'Flat:Det':>10s}"
    )
    for dist in ladder:
        m = _med(d, f"net_total_{dist}")
        print(
            f"  {dist:<10d}"
            + "".join(f"{m[t]:>10.0f}" for t in TYPES)
            + f"{_ratio(m):>8.1f}x"
        )

    # ---- [2b] LIKE-FOR-LIKE JOBS (same network distance, weighted sum) ----
    print(
        "\n  [2b] LIKE-FOR-LIKE JOBS — jobs reachable within the SAME network distance"
    )
    print(
        f"  {'dist (m)':<10s}"
        + "".join(f"{t:>12s}" for t in TYPES)
        + f"{'Flat:Det':>10s}"
    )
    for dist in ladder:
        m = _med(d, f"net_jobs_{dist}")
        print(
            f"  {dist:<10d}"
            + "".join(f"{m[t]:>12,.0f}" for t in TYPES)
            + f"{_ratio(m):>8.1f}x"
        )

    # ---- [3] DRIVABLE RATE (each OA at its own catchment) ----
    d["trip_km"] = _num(d["trip_m"]) / 1000
    d["amenities"] = _num(d["net_amen"])
    d["rate"] = d["amenities"] / d["transport"].replace(0, np.nan)
    print("\n  [3] DRIVABLE RATE — amenities per kWh, at each OA's own catchment")
    print(f"  {'':14s}" + "".join(f"{t:>10s}" for t in TYPES) + f"{'Flat:Det':>10s}")
    for label, col in [
        ("trip dist (km)", "trip_km"),
        ("amenities", "amenities"),
        ("travel kWh", "transport"),
        ("access / kWh", "rate"),
    ]:
        m = _med(d, col)
        print(
            f"  {label:<14s}"
            + "".join(f"{m[t]:>10.1f}" for t in TYPES)
            + f"{_ratio(m):>8.1f}x"
        )
    # jobs at the same own-catchment radius (weighted sum, then per kWh)
    d["jobs_catch"] = _num(d["net_jobs_catch"])
    d["jobs_rate"] = d["jobs_catch"] / d["transport"].replace(0, np.nan)
    for label, col, fmt in [
        ("jobs (catchment)", "jobs_catch", "{:>10,.0f}"),
        ("jobs / kWh", "jobs_rate", "{:>10,.0f}"),
    ]:
        m = _med(d, col)
        print(
            f"  {label:<14s}"
            + "".join(fmt.format(m[t]) for t in TYPES)
            + f"{_ratio(m):>8.1f}x"
        )


if __name__ == "__main__":
    main()
