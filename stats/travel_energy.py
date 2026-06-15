"""
Total household car-travel energy (kWh/household/year) by constrained
disaggregation of measured National Travel Survey mileage.

The previous Mobility figure captured only the journey to work — roughly
one-sixth of a household's car travel — so it under-counted travel energy ~6×.
There is no measured *local* vehicle mileage in open data (the residence-linked
MOT product is access-restricted; the only all-trip small-area OD is
commercial). The best open answer is therefore to **anchor to a measured
coarse-unit total and reverse-project it onto Output Areas** (a
constrained / maximum-entropy disaggregation):

* **Constraint (measured, residence-based):** NTS9904 — average car/van-driver
  distance per person by 2021 Rural-Urban Classification of residence
  (`nts_mileage_by_ruc.parquet`). Survey-measured by where people *live*, so it
  carries the real urban→rural driving gradient (~2,500 → ~5,200 mi/person)
  without through-traffic bias.
* **Allocator (measured, per-OA):** cars-per-person (Census TS045 ÷ household
  size) and commute distance (Census TS058) redistribute mileage *within* each
  rural-urban class.
* **Conservation:** within each class the population-weighted mean of the
  allocated per-person mileage equals the NTS class figure exactly, so the
  measured marginal is preserved while each OA varies by its local signal. The
  rural-urban gradient and the local car-ownership signal therefore combine
  with no double-count.
* **Energy:** × fleet intensity (DVLA `bev_share`) → kWh per household.

Inputs (built by `data/download_nts_mileage.py` and `data/download_ons_ruc.py`):
    - $DATA_DIR/statistics/nts_mileage_by_ruc.parquet
    - $DATA_DIR/statistics/oa21_ruc21.parquet

Sources / constants
-------------------
* NTS9904, DfT National Travel Survey 2024 (OGL) — the class mileage marginals.
* ONS 2021 Rural-Urban Classification of OAs (OGL) — the OA→class lookup.
* Vehicle-km energy intensities from DfT/ECUK: ICE car ≈ 0.58 kWh/vkm, battery
  electric ≈ 0.20 kWh/vkm (converted to per-mile below).
* Commute-distance elasticity of within-class mileage = 0.30 (the one
  transparent allocation assumption; commute is a minority of total mileage but
  correlates with it). Reported with a sensitivity in the analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from urban_energy.paths import DATA_DIR

# Energy intensity per vehicle-MILE (NTS distances are in miles).
_KM_PER_MILE = 1.60934
KWH_PER_MILE_ICE: float = 0.58 * _KM_PER_MILE  # ≈ 0.93
KWH_PER_MILE_EV: float = 0.20 * _KM_PER_MILE  # ≈ 0.32

#: Elasticity of within-class per-person mileage to local commute distance.
COMMUTE_DIST_ELASTICITY: float = 0.30

#: Census TS058 distance-to-work bands → representative midpoint (km).
TS058_BAND_MIDPOINTS_KM: dict[str, float] = {
    "ts058_Distance travelled to work: Less than 2km": 1.0,
    "ts058_Distance travelled to work: 2km to less than 5km": 3.5,
    "ts058_Distance travelled to work: 5km to less than 10km": 7.5,
    "ts058_Distance travelled to work: 10km to less than 20km": 15.0,
    "ts058_Distance travelled to work: 20km to less than 30km": 25.0,
    "ts058_Distance travelled to work: 30km to less than 40km": 35.0,
    "ts058_Distance travelled to work: 40km to less than 60km": 50.0,
    "ts058_Distance travelled to work: 60km and over": 75.0,
}

_NTS_PATH = DATA_DIR / "statistics" / "nts_mileage_by_ruc.parquet"
_RUC_PATH = DATA_DIR / "statistics" / "oa21_ruc21.parquet"


def mean_commute_km(lsoa: pd.DataFrame) -> pd.Series:
    """Mean one-way commute distance per OA from Census TS058 band midpoints."""
    num = sum(
        pd.to_numeric(lsoa[col], errors="coerce").fillna(0) * mid
        for col, mid in TS058_BAND_MIDPOINTS_KM.items()
    )
    den = sum(
        pd.to_numeric(lsoa[col], errors="coerce").fillna(0)
        for col in TS058_BAND_MIDPOINTS_KM
    )
    return num / den.replace(0, np.nan)


def fleet_intensity_kwh_per_mile(lsoa: pd.DataFrame) -> pd.Series:
    """Per-OA car energy intensity (kWh/vehicle-mile), weighted by BEV share."""
    if "bev_share" not in lsoa.columns:
        return pd.Series(KWH_PER_MILE_ICE, index=lsoa.index)
    bev = pd.to_numeric(lsoa["bev_share"], errors="coerce").fillna(0).clip(0, 1)
    return (1 - bev) * KWH_PER_MILE_ICE + bev * KWH_PER_MILE_EV


def _join_ruc_mileage(lsoa: pd.DataFrame) -> pd.DataFrame:
    """Merge each OA's 2021 rural-urban class and its NTS class mileage."""
    ruc = pd.read_parquet(_RUC_PATH)[["OA21CD", "RUC21NM"]]
    nts = pd.read_parquet(_NTS_PATH)[["ruc21_name", "car_miles_per_person"]]
    # Class names differ only in capitalisation between the two sources.
    ruc["_key"] = ruc["RUC21NM"].str.lower().str.strip()
    nts["_key"] = nts["ruc21_name"].str.lower().str.strip()
    ruc = ruc.merge(nts[["_key", "car_miles_per_person"]], on="_key", how="left")
    out = lsoa.merge(
        ruc[["OA21CD", "RUC21NM", "car_miles_per_person"]], on="OA21CD", how="left"
    )
    return out.rename(columns={"car_miles_per_person": "ruc_class_miles_pp"})


def compute_travel_energy(lsoa: pd.DataFrame) -> pd.DataFrame:
    """
    Add disaggregated total car-travel energy (kWh/hh/yr) and its components.

    For each OA the per-person car mileage is the NTS class marginal scaled by a
    local allocator (cars-per-person × commute-distance^elasticity), normalised
    so the population-weighted class mean is preserved. Energy follows from
    household size and fleet intensity.

    Returns
    -------
    pandas.DataFrame
        ``lsoa`` plus ``RUC21NM``, ``commute_km``, ``car_miles_per_person``,
        ``travel_kwh_per_hh_car`` (and intermediates).
    """
    df = _join_ruc_mileage(lsoa)
    df["commute_km"] = mean_commute_km(df)

    hh_size = pd.to_numeric(df["avg_hh_size"], errors="coerce")
    cars_pp = pd.to_numeric(df["cars_per_hh"], errors="coerce") / hh_size
    pop = pd.to_numeric(df["total_people"], errors="coerce")

    # Local allocator: car ownership per person, lifted mildly by commute length.
    commute_factor = (df["commute_km"] / df["commute_km"].median()).clip(lower=0.1) ** (
        COMMUTE_DIST_ELASTICITY
    )
    w = cars_pp.clip(lower=0) * commute_factor.fillna(1.0)

    # Population-weighted mean of the allocator within each rural-urban class.
    valid = w.notna() & pop.notna() & (pop > 0) & df["ruc_class_miles_pp"].notna()
    tmp = pd.DataFrame(
        {
            "ruc": df["RUC21NM"],
            "wp": np.where(valid, w * pop, np.nan),
            "pv": np.where(valid, pop, np.nan),
        }
    )
    wbar = tmp.groupby("ruc")["wp"].transform("sum") / tmp.groupby("ruc")[
        "pv"
    ].transform("sum")

    # Allocated per-person mileage: preserves the class marginal exactly.
    df["car_miles_per_person"] = np.where(
        valid & (wbar > 0),
        df["ruc_class_miles_pp"] * w / wbar,
        df["ruc_class_miles_pp"],  # neutral fallback = class average
    )
    df["travel_kwh_per_km"] = fleet_intensity_kwh_per_mile(df)
    df["travel_kwh_per_hh_car"] = (
        df["car_miles_per_person"] * hh_size * df["travel_kwh_per_km"]
    )
    return df


def _demo() -> None:
    """Print the disaggregation: marginal check + by-type gradient."""
    from oa_data import load_and_aggregate

    df = compute_travel_energy(load_and_aggregate())

    print("\n  Marginal check (pop-weighted class mean == NTS figure):")
    for cls, g in df.groupby("RUC21NM"):
        pop = pd.to_numeric(g["total_people"], errors="coerce")
        got = np.average(
            g["car_miles_per_person"].fillna(0), weights=pop.fillna(0) + 1e-9
        )
        nts = g["ruc_class_miles_pp"].iloc[0]
        print(f"    {cls:<46s} got {got:>6.0f}  nts {nts:>6.0f}")

    print(f"\n  {'type':<10s}{'cars/hh':>8s}{'car kWh':>10s}{'heat':>9s}{'TOTAL':>9s}")
    for t in ["Flat", "Terraced", "Semi", "Detached"]:
        s = df[df["dominant_type"] == t]
        car = s["travel_kwh_per_hh_car"].median()
        heat = s["building_kwh_per_hh"].median()
        cph = pd.to_numeric(s["cars_per_hh"], errors="coerce").median()
        print(f"  {t:<10s}{cph:>8.2f}{car:>10.0f}{heat:>9.0f}{heat + car:>9.0f}")


if __name__ == "__main__":
    _demo()
