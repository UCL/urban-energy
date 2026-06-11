"""
Lock-in penalty — how much of the sprawl energy gap survives perfect optimisation.

A "perfectly optimised" scenario, recomputing the energy axis:

* **Best-practice fabric** — modelled best-fabric building energy = EPC POTENTIAL
  intensity (kWh/m²/yr) × dwelling floor area. Using *intensity × size* (rather
  than scaling metered energy by a potential/current ratio) preserves the
  irreducible **size** effect: a best-fabric detached home is still bigger, so it
  still uses more heat. Insulation fixes per-m² efficiency, not floor area.
* **Full electrification** — travel at the EV fleet intensity; the *miles* are
  unchanged (technology cuts kWh/mile, never the miles).

The residual Flat→Detached gradient is the **lock-in**, and it splits across
*both* halves — bigger homes (heat) and longer trips (travel) — because
technology optimises per-unit efficiency but not the structural quantities. The
access axis is unchanged by construction (no technology moves destinations
closer).

Output cache: $DATA_DIR/statistics/oa_epc_potential.parquet
"""

from __future__ import annotations

import pandas as pd
from travel_energy import KWH_PER_MILE_EV, fleet_intensity_kwh_per_mile

from urban_energy.paths import DATA_DIR

_POT_PATH = DATA_DIR / "statistics" / "oa_epc_potential.parquet"


def oa_epc_potential() -> pd.DataFrame:
    """OA-median EPC best-fabric (POTENTIAL) energy intensity, kWh/m²/yr."""
    if _POT_PATH.exists():
        return pd.read_parquet(_POT_PATH)
    epc = pd.read_parquet(
        DATA_DIR / "epc" / "epc_domestic_spatial.parquet",
        columns=["POSTCODE", "ENERGY_CONSUMPTION_POTENTIAL"],
    )
    epc["POSTCODE"] = epc["POSTCODE"].astype(str).str.upper().str.strip()
    pot = pd.to_numeric(epc["ENERGY_CONSUMPTION_POTENTIAL"], errors="coerce")
    ok = pot.between(10, 1000)
    epc = epc[ok].assign(pot=pot[ok])
    lk = pd.read_parquet(
        DATA_DIR / "statistics" / "postcode_oa_lookup.parquet",
        columns=["Postcode", "OA21CD"],
    )
    lk["Postcode"] = lk["Postcode"].astype(str).str.upper().str.strip()
    oa = (
        epc.merge(lk, left_on="POSTCODE", right_on="Postcode", how="inner")
        .groupby("OA21CD")["pot"].median().reset_index()
        .rename(columns={"pot": "epc_potential_kwh_m2"})
    )
    _POT_PATH.parent.mkdir(parents=True, exist_ok=True)
    oa.to_parquet(_POT_PATH, index=False)
    return oa


def main() -> None:
    """Print the lock-in: current vs perfectly-optimised energy gap, by type."""
    import sys

    sys.argv = ["x"]
    from proof_of_concept_oa import load_and_aggregate

    df = load_and_aggregate().merge(oa_epc_potential(), on="OA21CD", how="left")
    fa = pd.to_numeric(df["oa_median_floor_area_m2"], errors="coerce")
    cur_int = fleet_intensity_kwh_per_mile(df)

    df["heat"] = df["building_kwh_per_hh"]
    df["travel"] = pd.to_numeric(df["transport_kwh_per_hh_total_est"], errors="coerce")
    df["heat_opt"] = df["epc_potential_kwh_m2"] * fa  # best-fabric × size
    df["travel_opt"] = df["travel"] * (KWH_PER_MILE_EV / cur_int)  # full EV

    types = ["Flat", "Terraced", "Semi", "Detached"]
    med = {
        t: df[df["dominant_type"] == t][["heat", "travel", "heat_opt", "travel_opt"]]
        .median()
        for t in types
    }

    print("\n  CURRENT vs OPTIMISED (best fabric × size + full EV), kWh/hh/yr")
    print(f"  {'type':<10s}{'heat':>7s}{'trav':>7s}{'TOT':>8s} | "
          f"{'heatO':>7s}{'travO':>7s}{'TOTo':>8s}")
    for t in types:
        m = med[t]
        tn, to = m.heat + m.travel, m.heat_opt + m.travel_opt
        print(f"  {t:<10s}{m.heat:>7.0f}{m.travel:>7.0f}{tn:>8.0f} | "
              f"{m.heat_opt:>7.0f}{m.travel_opt:>7.0f}{to:>8.0f}")

    f, d = med["Flat"], med["Detached"]
    g_now = (d.heat + d.travel) / (f.heat + f.travel)
    g_opt = (d.heat_opt + d.travel_opt) / (f.heat_opt + f.travel_opt)
    ex_now = (d.heat + d.travel) - (f.heat + f.travel)
    ex_opt = (d.heat_opt + d.travel_opt) - (f.heat_opt + f.travel_opt)
    print(f"\n  Flat→Detached gradient: current {g_now:.2f}x → optimised {g_opt:.2f}x")
    print(f"  Sprawl excess (Det−Flat): {ex_now:,.0f} → {ex_opt:,.0f} kWh "
          f"({ex_opt / ex_now:.0%} survives)")
    print(f"  Residual split: heat/size {d.heat_opt - f.heat_opt:,.0f} | "
          f"travel/miles {d.travel_opt - f.travel_opt:,.0f}")
    print("  Access deficit (GP 633→1,530 m): UNCHANGED — tech-immune.")


if __name__ == "__main__":
    main()
