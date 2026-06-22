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
compositional (method-D) per-dwelling ratio — both as-lived and at equal family
size (household size held as a free regressor, never as a per-person denominator).
The access axis is unchanged by construction (no technology moves destinations
closer).

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
    _deprivation_cols,
    _hdd_cols,
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
    factor = _fabric_factor(df)

    df["heat_opt"] = gas * factor + elec  # fabric improves gas; electricity unchanged
    df["travel_opt"] = travel * (KWH_PER_MILE_EV / fleet_intensity_kwh_per_mile(df))
    df["total_now"] = heat + travel
    df["total_opt"] = df["heat_opt"] + df["travel_opt"]

    cf = _compositional_frame(df)
    cf["log_hh_size"] = np.log(_num(cf["avg_hh_size"]).clip(lower=1))
    conf = (
        ["median_build_year"] + _deprivation_cols(cf) + _tenure_cols(cf) + _hdd_cols(cf)
    )
    # Family-size-controlled variant: hold log household size as a FREE regressor
    # rather than dividing energy by people (which would impose an occupancy
    # elasticity of 1 the data reject). Per-person ratios are deliberately not
    # reported here — they are a descriptive lens only (see form_size_decomposition).
    conf_fam = conf + ["log_hh_size"]
    for col in ["total_now", "total_opt"]:
        cf[f"_log_{col}"] = np.log(_num(cf[col]).clip(lower=1))

    now_hh = _d_ratio(cf, "_log_total_now", conf)
    opt_hh = _d_ratio(cf, "_log_total_opt", conf)
    now_fam = _d_ratio(cf, "_log_total_now", conf_fam)
    opt_fam = _d_ratio(cf, "_log_total_opt", conf_fam)

    print(
        f"\n  Fabric-improvement factor (EPC potential/current, median): "
        f"{factor.median():.2f}"
    )
    print("\n  Flat→Detached TOTAL energy gap (per dwelling, compositional, method D):")
    print(f"    as-lived:              now {now_hh:.2f}×  →  optimised {opt_hh:.2f}×")
    print(f"    at equal family size:  now {now_fam:.2f}×  →  optimised {opt_fam:.2f}×")
    # Surviving share on the model-native log scale: log(optimised) / log(now).
    # (An earlier version used the (ratio−1) excess scale, which understated it.)
    survives = np.log(opt_hh) / np.log(now_hh) if now_hh > 1 else float("nan")
    print(
        f"\n  {survives:.0%} of the per-dwelling gap (log scale) survives best "
        "insulation + a full EV fleet."
    )
    print("  Access deficit (on foot ~24×): UNCHANGED — tech-immune.")


if __name__ == "__main__":
    main()
