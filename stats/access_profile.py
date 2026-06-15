"""
Access profile — energy spent vs access gained, by service.

Three perspectives, by dominant dwelling type, with access measured as the COUNT
of each everyday service reachable within a ``CATCHMENT_M``-metre catchment.
Counts capture richness/choice, and — unlike nearest distance — can honestly
report **zero** (no service within the catchment), which is the starkest
statement of the deficit:

1. **Count within the catchment** (median) — the access, and its richness.
2. **% of neighbourhoods with ZERO** within the catchment — the deficit.
3. **Count per unit energy** (Flat = 100) — what each kWh of household energy buys.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from urban_energy.paths import PROCESSING_DIR

#: Local catchment radius (m). 1,600 m ≈ a 20-min walk / short cycle.
CATCHMENT_M = 1600

#: Everyday services and their cityseer count-column stem.
SERVICES: list[tuple[str, str]] = [
    ("GP", "cc_gp_practice"),
    ("Pharmacy", "cc_pharmacy"),
    ("Hospital", "cc_hospital"),
    ("School", "cc_school"),
    ("Food", "cc_fsa_restaurant"),
    ("Greenspace", "cc_greenspace"),
    ("Bus", "cc_bus"),
    ("Rail", "cc_rail"),
]
TYPES = ["Flat", "Terraced", "Semi", "Detached"]


def _with_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Merge the per-service within-catchment count columns + total energy."""
    cols = [f"{base}_{CATCHMENT_M}_nw" for _, base in SERVICES]
    ex = gpd.read_file(
        PROCESSING_DIR / "combined" / "oa_integrated.gpkg",
        columns=["OA21CD", *cols],
        ignore_geometry=True,
    ).drop_duplicates("OA21CD")
    df = df.merge(ex, on="OA21CD", how="left")
    df["energy"] = df["building_kwh_per_hh"] + pd.to_numeric(
        df["transport_kwh_per_hh_total_est"], errors="coerce"
    )
    return df


def _series(df: pd.DataFrame, dtype: str, col: str) -> pd.Series:
    return pd.to_numeric(df.loc[df["dominant_type"] == dtype, col], errors="coerce")


def main() -> None:
    """Print the three access perspectives."""
    from oa_data import load_and_aggregate

    df = _with_counts(load_and_aggregate())
    hdr = f"  {'':<12s}" + "".join(f"{t:>9s}" for t in TYPES)

    print(f"\n  1. COUNT within a {CATCHMENT_M} m catchment (median)")
    print(hdr + f"{'Det:Flat':>9s}")
    for nm, base in SERVICES:
        c = f"{base}_{CATCHMENT_M}_nw"
        m = {t: _series(df, t, c).median() for t in TYPES}
        r = m["Detached"] / m["Flat"] if m["Flat"] else float("nan")
        print(f"  {nm:<12s}" + "".join(f"{m[t]:>9.0f}" for t in TYPES) + f"{r:>8.2f}x")

    print(f"\n  2. % of neighbourhoods with ZERO within {CATCHMENT_M} m")
    print(hdr)
    for nm, base in SERVICES:
        c = f"{base}_{CATCHMENT_M}_nw"
        z = {t: (_series(df, t, c).fillna(0) == 0).mean() * 100 for t in TYPES}
        print(f"  {nm:<12s}" + "".join(f"{z[t]:>8.0f}%" for t in TYPES))

    print(f"\n  3. COUNT per unit ENERGY within {CATCHMENT_M} m (Flat = 100)")
    print(hdr)
    e = {t: df.loc[df["dominant_type"] == t, "energy"].mean() for t in TYPES}
    for nm, base in SERVICES:
        c = f"{base}_{CATCHMENT_M}_nw"
        ape = {t: _series(df, t, c).mean() / e[t] for t in TYPES}
        base_ape = ape["Flat"] or float("nan")
        idx = {t: ape[t] / base_ape * 100 for t in TYPES}
        print(f"  {nm:<12s}" + "".join(f"{idx[t]:>9.0f}" for t in TYPES))

    # Headline: how many times more access per kWh a flat gives than a detached.
    print("\n  HEADLINE — × access per kWh (a flat vs a detached home)")
    xs: list[float] = []
    for nm, base in SERVICES:
        c = f"{base}_{CATCHMENT_M}_nw"
        det = _series(df, "Detached", c).mean() / e["Detached"]
        flt = _series(df, "Flat", c).mean() / e["Flat"]
        x = flt / det if det else float("nan")
        xs.append(x)
        print(f"    {nm:<12s}{x:>6.1f}x")
    overall = float(np.exp(np.nanmean(np.log(xs))))
    print(f"    {'OVERALL':<12s}{overall:>6.1f}x  (geometric mean)")


if __name__ == "__main__":
    main()
