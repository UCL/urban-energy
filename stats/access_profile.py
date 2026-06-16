"""
Access profile — energy spent vs access gained, by service.

Straight-line access counts (within 1,600 m, from ``oa_access.py``) are already on
the frame; this reports them three ways, by dominant dwelling type:

1. **Count within the catchment** (median) — the access, and its richness.
2. **% of neighbourhoods with ZERO** within the catchment — the deficit.
3. **Count per unit energy** (Flat = 100) — what each kWh of household energy buys.

Employment (workplace jobs reachable) is reported separately as the single largest
travel driver.

Run:
    uv run python stats/access_profile.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from oa_data import load_and_aggregate

#: Everyday services (label → access-column stem), counted within 1,600 m.
SERVICES: list[tuple[str, str]] = [
    ("GP", "gp"),
    ("Pharmacy", "pharmacy"),
    ("Hospital", "hospital"),
    ("School", "school"),
    ("Food", "food"),
    ("Grocery", "grocery"),
    ("Greenspace", "greenspace"),
    ("Bus", "bus"),
    ("Rail", "rail"),
]
TYPES = ["Flat", "Terraced", "Semi", "Detached"]


def _series(df: pd.DataFrame, dtype: str, col: str) -> pd.Series:
    return pd.to_numeric(df.loc[df["dominant_type"] == dtype, col], errors="coerce")


def main() -> None:
    """Print the three access perspectives + the employment headline."""
    df = load_and_aggregate()
    df["energy"] = df["building_kwh_per_hh"] + pd.to_numeric(
        df["transport_kwh_per_hh_total_est"], errors="coerce"
    )
    hdr = f"  {'':<12s}" + "".join(f"{t:>9s}" for t in TYPES)

    print("\n  1. COUNT within a 1,600 m catchment (median)")
    print(hdr + f"{'Det:Flat':>9s}")
    for nm, stem in SERVICES:
        c = f"{stem}_n"
        m = {t: _series(df, t, c).median() for t in TYPES}
        r = m["Detached"] / m["Flat"] if m["Flat"] else float("nan")
        print(f"  {nm:<12s}" + "".join(f"{m[t]:>9.0f}" for t in TYPES) + f"{r:>8.2f}x")

    print("\n  2. % of neighbourhoods with ZERO within 1,600 m")
    print(hdr)
    for nm, stem in SERVICES:
        c = f"{stem}_n"
        z = {t: (_series(df, t, c).fillna(0) == 0).mean() * 100 for t in TYPES}
        print(f"  {nm:<12s}" + "".join(f"{z[t]:>8.0f}%" for t in TYPES))

    print("\n  3. COUNT per unit ENERGY within 1,600 m (Flat = 100)")
    print(hdr)
    e = {t: df.loc[df["dominant_type"] == t, "energy"].mean() for t in TYPES}
    for nm, stem in SERVICES:
        c = f"{stem}_n"
        ape = {t: _series(df, t, c).mean() / e[t] for t in TYPES}
        base = ape["Flat"] or float("nan")
        idx = {t: ape[t] / base * 100 for t in TYPES}
        print(f"  {nm:<12s}" + "".join(f"{idx[t]:>9.0f}" for t in TYPES))

    print("\n  HEADLINE — × access per kWh (a flat vs a detached home)")
    xs: list[float] = []
    for nm, stem in SERVICES:
        c = f"{stem}_n"
        det = _series(df, "Detached", c).mean() / e["Detached"]
        flt = _series(df, "Flat", c).mean() / e["Flat"]
        x = flt / det if det else float("nan")
        xs.append(x)
        print(f"    {nm:<12s}{x:>6.1f}x")
    overall = float(np.exp(np.nanmean(np.log(xs))))
    print(f"    {'OVERALL':<12s}{overall:>6.1f}x  (geometric mean)")

    print("\n  EMPLOYMENT — workplace jobs reachable within 1,600 m")
    jm = {t: _series(df, t, "jobs_n").median() for t in TYPES}
    jpe = {t: _series(df, t, "jobs_n").mean() / e[t] for t in TYPES}
    print("  " + "".join(f"{t}: {jm[t]:,.0f}   " for t in TYPES))
    print(
        f"  per kWh, a flat reaches {jpe['Flat'] / jpe['Detached']:.1f}× "
        "the jobs of a detached home"
    )


if __name__ == "__main__":
    main()
