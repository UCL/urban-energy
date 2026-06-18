"""
Lock-in penalty — how much of the sprawl energy gap survives perfect optimisation.

A "perfectly optimised" scenario, recomputing the energy axis:

* **Best-practice fabric** — metered **gas** (space heat + hot water) scaled by the
  EPC fabric-improvement ratio (``ENERGY_CONSUMPTION_POTENTIAL`` /
  ``ENERGY_CONSUMPTION_CURRENT``, both EPC-modelled so the performance gap cancels
  in the ratio); metered **electricity** (appliances, lighting) is left unchanged,
  since insulation does not touch it. Anchoring to the metered bill keeps the
  scenario on the same scale as the headline energy axis. (An earlier version
  multiplied EPC *potential intensity × floor area* — a modelled quantity that
  exceeds metered current consumption, so "insulation" perversely *raised* heat
  and over-stated how far the gap closes.)
* **Full electrification** — travel at the EV fleet intensity; the *miles* are
  unchanged (technology cuts kWh/mile, never the miles).

The residual Flat→Detached gradient is the **lock-in**, reported as the
compositional (method-D) ratio per household and per person to match
``paper/summary.md``. The access axis is unchanged by construction (no technology
moves destinations closer).

Run:
    uv run python stats/lock_in.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from form_size_decomposition import (
    _SHARE_FRACS,
    _comp_ols,
    _compositional_frame,
    _hdd_cols,
    _imd_income_col,
    _tenure_cols,
)
from oa_data import load_and_aggregate
from travel_energy import KWH_PER_MILE_EV, fleet_intensity_kwh_per_mile


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _fabric_factor(df: pd.DataFrame) -> pd.Series:
    """EPC fabric-improvement ratio (potential/current intensity), clipped to (0, 1]."""
    pot = _num(df["epc_potential_kwh_m2"])
    cur = _num(df["epc_current_kwh_m2"])
    return (pot / cur).clip(lower=0.1, upper=1.0)


def _d_ratio(cf: pd.DataFrame, y_col: str, confounds: list[str]) -> float:
    """Compositional Detached:Flat ratio (exp of the share-coefficient gap)."""
    m = _comp_ols(cf, y_col, _SHARE_FRACS + confounds, "total_hh")
    if m is None:
        return float("nan")
    return float(np.exp(m.params["s_detached"] - m.params["s_flat"]))


def main() -> None:
    """Print the lock-in: current vs perfectly-optimised energy gap (method D)."""
    df = load_and_aggregate()
    hh = _num(df["total_hh"])

    def _per_hh(mean_col: str, meter_col: str) -> pd.Series:
        return (_num(df[mean_col]) * _num(df[meter_col])).fillna(0) / hh

    gas = _per_hh("oa_gas_mean_kwh", "oa_gas_num_meters")
    elec = _per_hh("oa_elec_mean_kwh", "oa_elec_num_meters")
    heat = _num(df["building_kwh_per_hh"])
    travel = _num(df["transport_kwh_per_hh_total_est"])
    size = _num(df["avg_hh_size"])
    factor = _fabric_factor(df)

    df["heat_opt"] = gas * factor + elec  # fabric improves gas; electricity unchanged
    df["travel_opt"] = travel * (KWH_PER_MILE_EV / fleet_intensity_kwh_per_mile(df))
    df["total_now"] = heat + travel
    df["total_opt"] = df["heat_opt"] + df["travel_opt"]
    df["totpp_now"] = df["total_now"] / size
    df["totpp_opt"] = df["total_opt"] / size

    cf = _compositional_frame(df)
    conf = (
        ["median_build_year"] + _imd_income_col(cf) + _tenure_cols(cf) + _hdd_cols(cf)
    )
    for col in ["total_now", "total_opt", "totpp_now", "totpp_opt"]:
        cf[f"_log_{col}"] = np.log(_num(cf[col]).clip(lower=1))

    now_hh = _d_ratio(cf, "_log_total_now", conf)
    opt_hh = _d_ratio(cf, "_log_total_opt", conf)
    now_pp = _d_ratio(cf, "_log_totpp_now", conf)
    opt_pp = _d_ratio(cf, "_log_totpp_opt", conf)

    print(
        f"\n  Fabric-improvement factor (EPC potential/current, median): "
        f"{factor.median():.2f}"
    )
    print("\n  Flat→Detached TOTAL energy gap (compositional, method D):")
    print(f"    per household:  now {now_hh:.2f}×  →  optimised {opt_hh:.2f}×")
    print(f"    per person:     now {now_pp:.2f}×  →  optimised {opt_pp:.2f}×")
    survives = (opt_hh - 1) / (now_hh - 1) if now_hh > 1 else float("nan")
    print(
        f"\n  {survives:.0%} of the per-household excess survives best insulation "
        "+ a full EV fleet."
    )
    print("  Access deficit (on foot ~24×): UNCHANGED — tech-immune.")


if __name__ == "__main__":
    main()
