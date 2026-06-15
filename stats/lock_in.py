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

Both EPC inputs (floor area + best-fabric intensity) come from
``data/aggregate_epc_oa.py`` via the loader. Run:

    uv run python stats/lock_in.py
"""

from __future__ import annotations

import pandas as pd
from oa_data import load_and_aggregate
from travel_energy import KWH_PER_MILE_EV, fleet_intensity_kwh_per_mile


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def main() -> None:
    """Print the lock-in: current vs perfectly-optimised energy gap, by type."""
    df = load_and_aggregate()
    fa = _num(df["oa_median_floor_area_m2"])
    cur_int = fleet_intensity_kwh_per_mile(df)

    df["heat"] = df["building_kwh_per_hh"]
    df["travel"] = _num(df["transport_kwh_per_hh_total_est"])
    df["heat_opt"] = _num(df["epc_potential_kwh_m2"]) * fa  # best-fabric × size
    df["travel_opt"] = df["travel"] * (KWH_PER_MILE_EV / cur_int)  # full EV
    # Gradients use the median of the per-OA TOTAL (consistent with the energy
    # axis in argument.md §2), not the sum of per-component medians.
    df["total"] = df["heat"] + df["travel"]
    df["total_opt"] = df["heat_opt"] + df["travel_opt"]

    types = ["Flat", "Terraced", "Semi", "Detached"]
    cols = ["heat", "travel", "total", "heat_opt", "travel_opt", "total_opt"]
    med = {t: df[df["dominant_type"] == t][cols].median() for t in types}

    print("\n  CURRENT vs OPTIMISED (best fabric × size + full EV), kWh/hh/yr")
    print(f"  {'type':<10s}{'heat':>7s}{'trav':>7s}{'TOT':>8s} | "
          f"{'heatO':>7s}{'travO':>7s}{'TOTo':>8s}")
    for t in types:
        m = med[t]
        print(f"  {t:<10s}{m.heat:>7.0f}{m.travel:>7.0f}{m.total:>8.0f} | "
              f"{m.heat_opt:>7.0f}{m.travel_opt:>7.0f}{m.total_opt:>8.0f}")

    f, d = med["Flat"], med["Detached"]
    g_now = d.total / f.total
    g_opt = d.total_opt / f.total_opt
    ex_now = d.total - f.total
    ex_opt = d.total_opt - f.total_opt
    print(f"\n  Flat→Detached gradient: current {g_now:.2f}x → optimised {g_opt:.2f}x")
    print(f"  Sprawl excess (Det−Flat): {ex_now:,.0f} → {ex_opt:,.0f} kWh "
          f"({ex_opt / ex_now:.0%} survives)")
    print(f"  Residual split: heat/size {d.heat_opt - f.heat_opt:,.0f} | "
          f"travel/miles {d.travel_opt - f.travel_opt:,.0f}")
    print("  Access deficit (GP 633→1,530 m): UNCHANGED — tech-immune.")


if __name__ == "__main__":
    main()
