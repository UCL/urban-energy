"""
Lock-in penalty — how much of the sprawl energy gap survives perfect optimisation.

Defines a "perfectly optimised" scenario and recomputes the energy axis:

* **Best-practice fabric** — metered heat × the EPC potential/current ratio
  (modelled fabric-upgrade headroom; type-specific via the OA-median ratio). The
  ratio, not the level, is used so the measured baseline is preserved (avoids the
  EPC modelled-vs-metered performance gap).
* **Full electrification** — travel energy at the EV fleet intensity (miles
  unchanged; that is the point — technology cuts kWh/mile, never the miles).

The residual Flat→Detached energy gradient is the **lock-in penalty**. The access
axis is unchanged by construction — no technology moves destinations closer.

Output cache: $DATA_DIR/statistics/oa_epc_retrofit_ratio.parquet
"""

from __future__ import annotations

import pandas as pd
from travel_energy import KWH_PER_MILE_EV, fleet_intensity_kwh_per_mile

from urban_energy.paths import DATA_DIR

_RATIO_PATH = DATA_DIR / "statistics" / "oa_epc_retrofit_ratio.parquet"


def oa_retrofit_ratio() -> pd.DataFrame:
    """OA-median EPC potential/current energy ratio (fabric-upgrade headroom)."""
    if _RATIO_PATH.exists():
        return pd.read_parquet(_RATIO_PATH)
    epc = pd.read_parquet(
        DATA_DIR / "epc" / "epc_domestic_spatial.parquet",
        columns=[
            "POSTCODE",
            "ENERGY_CONSUMPTION_CURRENT",
            "ENERGY_CONSUMPTION_POTENTIAL",
        ],
    )
    epc["POSTCODE"] = epc["POSTCODE"].astype(str).str.upper().str.strip()
    cur = pd.to_numeric(epc["ENERGY_CONSUMPTION_CURRENT"], errors="coerce")
    pot = pd.to_numeric(epc["ENERGY_CONSUMPTION_POTENTIAL"], errors="coerce")
    ok = cur.between(20, 1000) & pot.between(10, 1000)
    epc = epc[ok].copy()
    epc["ratio"] = (pot[ok] / cur[ok]).clip(0.2, 1.0)
    lk = pd.read_parquet(
        DATA_DIR / "statistics" / "postcode_oa_lookup.parquet",
        columns=["Postcode", "OA21CD"],
    )
    lk["Postcode"] = lk["Postcode"].astype(str).str.upper().str.strip()
    m = epc.merge(lk, left_on="POSTCODE", right_on="Postcode", how="inner")
    oa = (
        m.groupby("OA21CD")["ratio"].median().reset_index()
        .rename(columns={"ratio": "epc_retrofit_ratio"})
    )
    _RATIO_PATH.parent.mkdir(parents=True, exist_ok=True)
    oa.to_parquet(_RATIO_PATH, index=False)
    return oa


def main() -> None:
    """Compute and print the lock-in: current vs perfectly-optimised energy gap."""
    import sys

    sys.argv = ["x"]
    from proof_of_concept_oa import load_and_aggregate

    df = load_and_aggregate().merge(oa_retrofit_ratio(), on="OA21CD", how="left")

    heat = df["building_kwh_per_hh"]
    travel = pd.to_numeric(df["transport_kwh_per_hh_total_est"], errors="coerce")
    ratio = df["epc_retrofit_ratio"].fillna(df["epc_retrofit_ratio"].median())
    cur_int = fleet_intensity_kwh_per_mile(df)

    df["heat"], df["travel"], df["total"] = heat, travel, heat + travel
    df["heat_opt"] = heat * ratio
    df["travel_opt"] = travel * (KWH_PER_MILE_EV / cur_int)
    df["total_opt"] = df["heat_opt"] + df["travel_opt"]

    types = ["Flat", "Terraced", "Semi", "Detached"]
    med = {t: df[df["dominant_type"] == t][
        ["heat", "travel", "total", "heat_opt", "travel_opt", "total_opt"]
    ].median() for t in types}

    print("\n  CURRENT vs PERFECTLY-OPTIMISED (best fabric + full EV), kWh/hh/yr")
    print(f"  {'type':<10s}{'heat':>7s}{'trav':>7s}{'TOT':>8s} | "
          f"{'heatO':>7s}{'travO':>7s}{'TOTo':>8s}")
    for t in types:
        m = med[t]
        print(f"  {t:<10s}{m.heat:>7.0f}{m.travel:>7.0f}{m.total:>8.0f} | "
              f"{m.heat_opt:>7.0f}{m.travel_opt:>7.0f}{m.total_opt:>8.0f}")

    f, d = med["Flat"], med["Detached"]
    g_now = d.total / f.total
    g_opt = d.total_opt / f.total_opt
    print(
        f"\n  Flat→Detached gradient: current {g_now:.2f}x → optimised {g_opt:.2f}x"
    )
    excess_now = d.total - f.total
    excess_opt = d.total_opt - f.total_opt
    print(f"  Sprawl excess (Det − Flat): current {excess_now:,.0f}  →  "
          f"optimised {excess_opt:,.0f} kWh ({excess_opt / excess_now:.0%} survives)")
    print("  Access deficit (e.g. GP 633→1,530 m): UNCHANGED — tech-immune.")


if __name__ == "__main__":
    main()
