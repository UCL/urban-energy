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
* Vehicle-km energy intensities: ICE car ≈ 0.58 kWh/vkm, battery electric ≈ 0.20
  kWh/vkm (converted to per-mile below). Provenance: 0.58 is fleet-average
  combustion consumption of ~6.5 l/100 km at 8.9 kWh per litre of petrol; 0.20 is
  the real-world consumption of current BEVs (~3.5–4.5 miles/kWh). Both are
  consistent with the DfT ECUK road-transport and TAG appraisal tables; the
  manuscript states the derivation (Methods, "The energy axis").
* Commute-distance elasticity of within-class mileage = 0.30 (commute is a
  minority of total mileage but correlates with it). Reported with a
  sensitivity (0.15 / 0.30 / 0.45) in ``sensitivity_report`` below.
* Ownership elasticity of within-class mileage = 1.0: the allocator scales
  miles proportionally with cars per person. NTS evidence is that additional
  household cars add *less* than proportional mileage, so proportionality if
  anything over-allocates to high-ownership (detached) areas. Because the class
  marginals are held fixed, this assumption moves only the within-class spread;
  ``sensitivity_report`` sweeps it (1.0 / 0.8 / 0.6 / 0.0, where 0.0 is the
  measured between-class floor) and reports the flat-to-detached travel gap at
  each value.
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

#: Elasticity of within-class per-person mileage to cars per person. 1.0 is the
#: proportional headline allocator; swept in ``sensitivity_report``.
OWNERSHIP_ELASTICITY: float = 1.0

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

#: Columns produced by :func:`compute_travel_energy`. Dropped on re-entry so the
#: function is idempotent when called on a frame it has already processed (the
#: loader calls it once, and the sensitivity report re-runs it several times).
_TRAVEL_COLS = [
    "RUC21NM",
    "ruc_class_miles_pp",
    "commute_km",
    "car_miles_per_person",
    "travel_kwh_per_mile",
    "travel_kwh_per_hh_car",
    "travel_is_fallback",
]


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
    # NTS has one row per rural-urban class; the RUC lookup one row per OA.
    ruc = ruc.merge(
        nts[["_key", "car_miles_per_person"]], on="_key", how="left", validate="m:1"
    )
    out = lsoa.merge(
        ruc[["OA21CD", "RUC21NM", "car_miles_per_person"]],
        on="OA21CD",
        how="left",
        validate="m:1",
    )
    return out.rename(columns={"car_miles_per_person": "ruc_class_miles_pp"})


def compute_travel_energy(
    lsoa: pd.DataFrame,
    elasticity: float = COMMUTE_DIST_ELASTICITY,
    ownership_elasticity: float = OWNERSHIP_ELASTICITY,
) -> pd.DataFrame:
    """
    Add disaggregated total car-travel energy (kWh/hh/yr) and its components.

    For each OA the per-person car mileage is the NTS class marginal scaled by a
    local allocator (cars-per-person^ownership_elasticity ×
    commute-distance^elasticity), normalised so the population-weighted class
    mean is preserved. Energy follows from household size and fleet intensity.

    Parameters
    ----------
    lsoa : pandas.DataFrame
        OA frame with ``cars_per_hh``, ``avg_hh_size``, ``total_people`` and the
        TS058 commute bands.
    elasticity : float, default :data:`COMMUTE_DIST_ELASTICITY`
        Commute-distance elasticity of the allocator, exposed so
        :func:`sensitivity_report` can vary it without mutating the module
        constant.
    ownership_elasticity : float, default :data:`OWNERSHIP_ELASTICITY`
        Exponent on cars per person in the allocator. 1.0 spreads a class's
        miles proportionally with ownership; 0.0 removes the ownership signal,
        so every OA in a class receives the class average scaled only by the
        commute factor (the measured between-class floor).

    Returns
    -------
    pandas.DataFrame
        ``lsoa`` plus ``RUC21NM``, ``commute_km``, ``car_miles_per_person``,
        ``travel_kwh_per_hh_car`` and a boolean ``travel_is_fallback`` marking OAs
        that received the flat class average rather than a local estimate.

    Notes
    -----
    The normalising median in ``commute_factor`` cancels in the ``w / wbar`` ratio,
    so the choice of median (global here) does not affect any allocated mileage.
    Fallback OAs (invalid allocator but valid class mileage) receive the exact
    class average, so the population-weighted class marginal is preserved over all
    OAs; only OAs failing the rural-urban-class join get ``NaN`` mileage and drop
    out downstream. :func:`sensitivity_report` prints both counts.
    """
    # Idempotent: drop any columns from a previous pass so a re-run does not
    # collide on the rural-urban-class merge.
    lsoa = lsoa.drop(columns=[c for c in _TRAVEL_COLS if c in lsoa.columns])
    df = _join_ruc_mileage(lsoa)
    df["commute_km"] = mean_commute_km(df)

    hh_size = pd.to_numeric(df["avg_hh_size"], errors="coerce")
    cars_pp = pd.to_numeric(df["cars_per_hh"], errors="coerce") / hh_size
    pop = pd.to_numeric(df["total_people"], errors="coerce")

    # Local allocator: car ownership per person, lifted mildly by commute length.
    commute_factor = (df["commute_km"] / df["commute_km"].median()).clip(lower=0.1) ** (
        elasticity
    )
    w = cars_pp.clip(lower=0) ** ownership_elasticity * commute_factor.fillna(1.0)

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
    allocated = valid & (wbar > 0)
    df["travel_is_fallback"] = df["ruc_class_miles_pp"].notna() & ~allocated
    df["car_miles_per_person"] = np.where(
        allocated,
        df["ruc_class_miles_pp"] * w / wbar,
        df["ruc_class_miles_pp"],  # neutral fallback = class average
    )
    df["travel_kwh_per_mile"] = fleet_intensity_kwh_per_mile(df)
    df["travel_kwh_per_hh_car"] = (
        df["car_miles_per_person"] * hh_size * df["travel_kwh_per_mile"]
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

    # Car uses the canonical pipeline column (transport_kwh_per_hh_total_est, whose
    # class normaliser is taken over all OAs), not the sample-restricted recompute,
    # so Table 2 rests on the same travel figure as the headline rate. TOTAL is the
    # per-OA median of heat + travel, not the sum of the two column medians (medians
    # are not additive).
    print(f"\n  {'type':<10s}{'cars/hh':>8s}{'car kWh':>10s}{'heat':>9s}{'TOTAL':>9s}")
    print("  (TOTAL is the per-OA median of heat+travel, not the sum of the columns)")
    med: dict[str, dict[str, float]] = {"car": {}, "heat": {}, "tot": {}}
    for t in ["Flat", "Terraced", "Semi", "Detached"]:
        s = df[df["dominant_type"] == t]
        car = pd.to_numeric(s["transport_kwh_per_hh_total_est"], errors="coerce")
        heat = pd.to_numeric(s["building_kwh_per_hh"], errors="coerce")
        cph = pd.to_numeric(s["cars_per_hh"], errors="coerce").median()
        cmed, hmed, tot = car.median(), heat.median(), (heat + car).median()
        med["car"][t], med["heat"][t], med["tot"][t] = cmed, hmed, tot
        print(f"  {t:<10s}{cph:>8.2f}{cmed:>10.0f}{hmed:>9.0f}{tot:>9.0f}")

    # The energy panel of the manuscript's two-axis table; the compositional
    # ratio column comes from ledger macros recorded by the model scripts.
    import ledger

    def _cells(kind: str) -> str:
        return "".join(
            f" & {med[kind][t]:,.0f}" for t in ["Flat", "Terraced", "Semi", "Detached"]
        )

    ledger.table(
        "axesenergy",
        "\\multicolumn{6}{l}{\\textit{Energy (kWh per dwelling per year)}} \\\\\n"
        f"Home (metered gas + electricity){_cells('heat')} & "
        "\\nepiheatGap$\\times$ \\\\\n"
        f"Car travel (NTS-anchored){_cells('car')} & \\nepitravelGap$\\times$ \\\\\n"
        f"Total{_cells('tot')} & \\nepitotalGap$\\times$ \\\\\n",
    )
    ledger.record(
        travelIce=ledger.pt(KWH_PER_MILE_ICE),
        travelEv=ledger.pt(KWH_PER_MILE_EV),
    )


def sensitivity_report() -> None:
    """Coverage counts and allocator-assumption sensitivities of the travel gradient.

    Two allocator assumptions are swept: the commute-distance elasticity (the
    small within-class term) and the ownership elasticity (the large one). Both
    only redistribute miles *within* a rural-urban class — the measured NTS class
    marginals are preserved at every setting — so each sweep reports how far the
    flat-to-detached travel contrast moves while the totals stay anchored.
    """
    # Imported lazily: form_size_decomposition imports oa_data, which imports
    # this module, so a top-level import would be circular.
    from form_size_decomposition import (
        _SHARE_FRACS,
        _comp_ols,
        _compositional_frame,
        _deprivation_cols,
        _hdd_cols,
        _tenure_cols,
    )
    from inference import CLUSTER_COL, fmt_ci, log_contrast_ci
    from oa_data import load_and_aggregate

    def _comp_travel_gap(d: pd.DataFrame) -> tuple[float, float, float, float] | None:
        """Compositional pure-type Det:Flat travel gap (LAD-clustered CI)."""
        cf = _compositional_frame(d)
        conf = (
            ["median_build_year"]
            + _deprivation_cols(cf)
            + _tenure_cols(cf)
            + _hdd_cols(cf)
        )
        cf["_log_travel"] = np.log(
            pd.to_numeric(cf["travel_kwh_per_hh_car"], errors="coerce").clip(lower=1)
        )
        m = _comp_ols(
            cf, "_log_travel", _SHARE_FRACS + conf, "total_hh", cluster_col=CLUSTER_COL
        )
        if m is None:
            return None
        return log_contrast_ci(m, "s_detached", "s_flat")

    def _dom_medians(d: pd.DataFrame) -> tuple[float, float]:
        med = {
            t: pd.to_numeric(
                d.loc[d["dominant_type"] == t, "travel_kwh_per_hh_car"], errors="coerce"
            ).median()
            for t in ("Flat", "Detached")
        }
        ratio = med["Detached"] / med["Flat"] if med["Flat"] else float("nan")
        return med["Flat"], ratio

    base = load_and_aggregate()

    d0 = compute_travel_energy(base)
    n_fallback = int(d0["travel_is_fallback"].sum())
    n_rucfail = int(d0["ruc_class_miles_pp"].isna().sum())
    print("\n" + "=" * 70)
    print("TRAVEL-ENERGY SENSITIVITY")
    print("=" * 70)
    print(
        f"\n  Allocator coverage of {len(d0):,} OAs: {n_fallback:,} fallback "
        f"(flat class average, marginal preserved), {n_rucfail:,} rural-urban-class "
        "join failures (NaN mileage, dropped downstream)."
    )
    import ledger

    print("\n  Commute-elasticity sensitivity — Det:Flat car kWh/hh:")
    commute_ratios: list[float] = []
    for e in (0.15, 0.30, 0.45):
        d = compute_travel_energy(base, elasticity=e)
        flat_med, ratio = _dom_medians(d)
        commute_ratios.append(ratio)
        print(
            f"    elasticity {e:.2f}:  dominant-type median {ratio:.2f}×  "
            f"(Flat median {flat_med:,.0f} kWh)"
        )
    ledger.record(
        commuteSweepLo=ledger.pt(min(commute_ratios)),
        commuteSweepHi=ledger.pt(max(commute_ratios)),
    )
    print(
        "\n  Ownership-elasticity sensitivity — within-class allocator "
        "cars_pp**alpha:\n"
        "  (alpha 1.0 = proportional headline; NTS per-car mileage declines with\n"
        "   household car count, so alpha < 1 is the realistic direction; alpha 0\n"
        "   removes the ownership signal entirely, leaving the measured NTS\n"
        "   between-class gradient — the floor of the contrast. Class marginals\n"
        "   are preserved exactly at every alpha.)"
    )
    alpha_keys = {0.8: "alphaEightGap", 0.6: "alphaSixGap", 0.0: "alphaZeroGap"}
    for a in (1.0, 0.8, 0.6, 0.0):
        d = compute_travel_energy(base, ownership_elasticity=a)
        _, ratio = _dom_medians(d)
        comp = _comp_travel_gap(d)
        if comp is not None and a in alpha_keys:
            ledger.record(
                **{
                    alpha_keys[a]: ledger.pt(comp[0]),
                    alpha_keys[a] + "CI": ledger.ci(comp[1], comp[2]),
                }
            )
        print(
            f"    alpha {a:.1f}:  dominant-type median {ratio:.2f}×   "
            f"compositional {fmt_ci(comp) if comp is not None else 'n/a'}"
        )
    print(
        "\n  (TRIPS_PER_YEAR=370 sets the access catchment radius in "
        "oa_network_access.py; a full sweep needs the network cache rebuilt and is "
        "documented there, not run here.)"
    )


if __name__ == "__main__":
    _demo()
    sensitivity_report()
