"""
Decarbonisation scenarios — how the flat→detached energy gap moves under realistic
technology rollouts, and how much of it (and all of the access gap) survives.

Generalises ``lock_in.py`` from a single "perfectly optimised" end-state to a
ladder of scenarios, each recomputing the per-dwelling energy axis and re-fitting
the same compositional (method-D) Detached:Flat ratio. Two atomic per-home
transformations are applied at a scenario-specific uptake fraction:

* **Heat retrofit** (fabric + heat pump) — a converting home's metered **gas**
  (space heat + hot water) is first cut by its EPC fabric-improvement ratio
  (potential/current intensity, both EPC-modelled so the performance gap cancels),
  then delivered by a heat pump: the insulated heat demand is met with electricity
  at ``BOILER_EFF / COP`` of the gas figure (a heat pump draws less *delivered*
  energy than a boiler for the same useful heat). Metered **electricity**
  (appliances, lighting) is unchanged.
* **EV** — a converting car's travel energy is recomputed at the electric fleet's
  energy per mile; the *miles* are unchanged (technology cuts kWh/mile, not miles).

Partial uptake blends the transformed and untransformed per-household energy
linearly (``u`` of homes on the new technology, ``1 − u`` unchanged), which for
the EV lever is exactly a fleet-intensity blend.

Uptake fractions for the milestone scenario are the Climate Change Committee's
Seventh Carbon Budget (Feb 2025) Balanced Pathway at 2040: about half of homes on
a heat pump and about three-quarters of cars electric.

The access axis is unchanged in every scenario by construction — no fabric,
heat pump or drivetrain moves a destination closer.

Run:
    uv run python stats/scenarios.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from access_profile import _comp_poisson
from form_size_decomposition import (
    _SHARE_FRACS,
    _comp_ols,
    _compositional_frame,
    _deprivation_cols,
    _hdd_cols,
    _imd_income_col,
    _tenure_cols,
)
from inference import CLUSTER_COL, NAN_CI, fmt_ci, log_contrast_ci
from oa_data import load_and_aggregate
from travel_energy import KWH_PER_MILE_EV, fleet_intensity_kwh_per_mile

from urban_energy.paths import DATA_DIR

# --- Technology assumptions (adjustable; a COP sensitivity is reported below) ---
BOILER_EFF = 0.90  # gas boiler seasonal efficiency
COP = 2.8  # heat-pump seasonal coefficient of performance

# --- CCC Seventh Carbon Budget, Balanced Pathway, 2040 stock shares ---
HEAT_UPTAKE_2040 = 0.50  # ~half of homes heated by a heat pump
EV_UPTAKE_2040 = 0.75  # ~three-quarters of cars electric

# Fabric and heat pumps are separate levers (they act differently on the gap:
# fabric closes it via per-area headroom, a heat pump is a near-uniform delivered-
# energy cut that unmasks the travel gap), so they are reported in isolation.
# (label, heat transform, heat uptake, EV uptake). The isolated single-lever rows
# are shown at full deployment (100%) to expose each measure's ceiling effect; the
# pathway rows use the CCC Balanced Pathway 2040 stock shares.
SCENARIOS: list[tuple[str, str, float, float]] = [
    ("S0 status quo", "none", 0.0, 0.0),
    ("Fabric only (100%)", "fabric", 1.0, 0.0),
    ("Heat pumps only (100%)", "hp", 1.0, 0.0),
    ("EVs only (100%)", "none", 0.0, 1.0),
    ("CCC Balanced Pathway 2040", "fabric+hp", HEAT_UPTAKE_2040, EV_UPTAKE_2040),
    ("Full rollout (100%)", "fabric+hp", 1.0, 1.0),
]


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _fabric_factor(df: pd.DataFrame) -> pd.Series:
    """EPC fabric-improvement ratio (potential/current intensity), clipped to (0, 1]."""
    pot = _num(df["epc_potential_kwh_m2"])
    cur = _num(df["epc_current_kwh_m2"])
    return (pot / cur).clip(lower=0.1, upper=1.0)


def scenario_energy(
    df: pd.DataFrame,
    gas: pd.Series,
    elec: pd.Series,
    travel: pd.Series,
    heat_transform: str,
    u_heat: float,
    u_ev: float,
    cop: float = COP,
) -> tuple[pd.Series, pd.Series]:
    """Per-household heat and travel energy under a scenario's levers.

    The heat side applies one transform to the converting homes' metered gas:
    ``"fabric"`` (insulate: gas × EPC fabric factor), ``"hp"`` (heat pump: fuel
    switch, gas × boiler_eff/COP), ``"fabric+hp"`` (insulate, then a heat pump
    meets the reduced demand), or ``"none"``. Metered electricity is unchanged.
    The travel side switches the converting fraction of cars to the electric
    fleet intensity. Uptake blends the transformed and untransformed energy.

    Parameters
    ----------
    df : pandas.DataFrame
        OA frame (for the EPC fabric factor and fleet intensity).
    gas, elec, travel : pandas.Series
        Baseline per-household metered gas, metered electricity, and NTS-anchored
        car-travel energy (kWh/dwelling/yr).
    heat_transform : str
        One of ``"none"``, ``"fabric"``, ``"hp"``, ``"fabric+hp"``.
    u_heat, u_ev : float
        Uptake fractions in [0, 1] for the heat transform and the EV switch.
    cop : float, default ``COP``
        Heat-pump seasonal coefficient of performance (varied in the sensitivity).

    Returns
    -------
    tuple of pandas.Series
        ``(heat_per_hh, travel_per_hh)`` under the scenario.
    """
    factor = _fabric_factor(df)
    hp = BOILER_EFF / cop  # heat-pump delivered-energy factor vs a gas boiler
    treated = {
        "none": gas,
        "fabric": gas * factor,
        "hp": gas * hp,
        "fabric+hp": gas * factor * hp,
    }[heat_transform]
    heat = u_heat * treated + (1.0 - u_heat) * gas + elec

    travel_ev = travel * (KWH_PER_MILE_EV / fleet_intensity_kwh_per_mile(df))
    travel_s = u_ev * travel_ev + (1.0 - u_ev) * travel
    return heat, travel_s


def _ratios(
    df: pd.DataFrame,
    gas: pd.Series,
    elec: pd.Series,
    travel: pd.Series,
    heat_transform: str,
    u_heat: float,
    u_ev: float,
    cop: float = COP,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Compositional Detached:Flat total-energy ratio: as-lived and equal-family."""
    heat, travel_s = scenario_energy(
        df, gas, elec, travel, heat_transform, u_heat, u_ev, cop
    )
    cf = _compositional_frame(df)
    cf["_log_total"] = np.log((heat + travel_s).clip(lower=1).to_numpy())
    cf["log_hh_size"] = np.log(_num(cf["avg_hh_size"]).clip(lower=1))
    conf = (
        ["median_build_year"] + _deprivation_cols(cf) + _tenure_cols(cf) + _hdd_cols(cf)
    )
    as_lived = _d_ratio_ci(cf, "_log_total", conf)
    equal_fam = _d_ratio_ci(cf, "_log_total", conf + ["log_hh_size"])
    return as_lived, equal_fam


def _d_ratio_ci(
    cf: pd.DataFrame, y_col: str, confounds: list[str]
) -> tuple[float, float, float, float]:
    """Compositional Detached:Flat ratio with a LAD-clustered 95% CI."""
    m = _comp_ols(
        cf, y_col, _SHARE_FRACS + confounds, "total_hh", cluster_col=CLUSTER_COL
    )
    if m is None:
        return NAN_CI
    return log_contrast_ci(m, "s_detached", "s_flat")


def _access_deficit(cf: pd.DataFrame) -> str:
    """On-foot amenity Flat:Detached gap (unchanged across scenarios)."""
    net_path = DATA_DIR / "statistics" / "oa_network_access.parquet"
    if not net_path.exists():
        return "network cache absent; see access_profile"
    net = pd.read_parquet(net_path, columns=["net_total_1600"])
    cfa = cf.merge(net, left_on="OA21CD", right_index=True, how="left", validate="m:1")
    ma = _comp_poisson(
        cfa,
        "net_total_1600",
        _SHARE_FRACS + _imd_income_col(cfa),
        "total_hh",
        cluster_col=CLUSTER_COL,
    )
    if ma is None:
        return "unavailable"
    return fmt_ci(log_contrast_ci(ma, "s_flat", "s_detached"))


def main() -> None:
    """Print the decarbonisation-scenario ladder for the total energy gap."""
    df = load_and_aggregate()

    # One common complete-case sample for every scenario: the fabric factor is
    # NaN where EPC data are missing, so without this restriction the fabric
    # scenarios would fit on a smaller sample than S0 and the survives-share
    # would compare fits across samples.
    n_all = len(df)
    df = df[
        _fabric_factor(df).notna() & _num(df["transport_kwh_per_hh_total_est"]).notna()
    ].copy()
    print(
        f"\n  Common scenario sample: {len(df):,} of {n_all:,} OAs "
        "(complete EPC fabric factor + travel; every scenario fits on these rows)."
    )

    hh = _num(df["total_hh"])
    gas = (_num(df["oa_gas_mean_kwh"]) * _num(df["oa_gas_num_meters"])).fillna(0) / hh
    elec = (_num(df["oa_elec_mean_kwh"]) * _num(df["oa_elec_num_meters"])).fillna(
        0
    ) / hh
    travel = _num(df["transport_kwh_per_hh_total_est"])

    print(
        f"\n  Heat-pump COP {COP}, boiler efficiency {BOILER_EFF} "
        f"(heat-pump delivered-energy factor {BOILER_EFF / COP:.2f} vs a gas boiler)."
    )
    print("  Pathway uptakes: CCC 7CB Balanced Pathway 2040 (heat pumps 50%, EVs 75%).")
    print("\n  Flat→Detached TOTAL energy gap (per dwelling, compositional, method D):")
    print(f"\n  {'scenario':<30s}{'as-lived [95% CI]':>24s}{'equal family size':>24s}")
    print("  " + "-" * 76)
    ratios: dict[str, float] = {}
    for label, heat_transform, u_heat, u_ev in SCENARIOS:
        as_lived, equal_fam = _ratios(
            df, gas, elec, travel, heat_transform, u_heat, u_ev
        )
        ratios[label] = as_lived[0]
        print(f"  {label:<30s}{fmt_ci(as_lived):>24s}{fmt_ci(equal_fam):>24s}")

    # Transparency on each lever's direction: fabric closes the gap (detached homes
    # have more EPC headroom), a heat pump widens it (a near-uniform delivered-energy
    # cut that removes the low-gap heat component and unmasks the high-gap travel),
    # EVs close it most (they attack travel, where the gap is largest).
    r0 = ratios["S0 status quo"]
    for lever in ("Fabric only (100%)", "Heat pumps only (100%)", "EVs only (100%)"):
        r = ratios[lever]
        closed = 1 - np.log(r) / np.log(r0) if r0 > 1 and r > 1 else 0.0
        verb = "closes" if closed > 0 else "widens"
        print(
            f"\n  {lever}: {r:.2f}× {verb} the status-quo gap by "
            f"{abs(closed):.0%} (log scale)."
        )

    # Fraction of the status-quo log-gap that survives under the realistic pathway
    # and the full rollout, derived from the ladder point estimates (the per-scenario
    # CIs above carry the sampling uncertainty; the composite ratio is unstable under
    # resampling, so no separate interval is quoted).
    for label in ("CCC Balanced Pathway 2040", "Full rollout (100%)"):
        r = ratios[label]
        survives = np.log(r) / np.log(r0) if r0 > 1 and r > 1 else float("nan")
        print(
            f"\n  {label}: {survives:.0%} of the status-quo per-dwelling energy gap "
            f"survives\n  (log scale); {1 - survives:.0%} closed."
        )

    cf = _compositional_frame(df)
    print(
        f"\n  Access deficit on foot {_access_deficit(cf)}: UNCHANGED in every "
        "scenario — no\n  technology moves destinations closer."
    )

    # COP sensitivity on the milestone (Balanced Pathway) scenario.
    print("\n  COP sensitivity — Balanced Pathway 2040 total Det:Flat (as-lived):")
    for cop in (2.4, 2.8, 3.2):
        r, _ = _ratios(
            df, gas, elec, travel, "fabric+hp", HEAT_UPTAKE_2040, EV_UPTAKE_2040, cop
        )
        print(f"    COP {cop}:  {fmt_ci(r)}")


if __name__ == "__main__":
    main()
