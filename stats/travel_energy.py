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

The headline axis is car travel only. ``public_transport_report`` bounds that
scope: bus, rail and Underground mileage from the same NTS table, disaggregated
by the same constrained method with the Census TS061 commute share as allocator,
added to the axis at an assumed per-passenger-mile intensity and again at the
car-intensity ceiling. It reports to Extended Data, not to the headline.

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

#: Census TS061 public-transport commute categories. Their share of an OA's
#: commuters is the within-class allocator of the public-transport sensitivity,
#: standing in the same place as cars per person in the car allocator.
TS061_PT_COLS: list[str] = [
    "ts061_Method of travel to workplace: Underground, metro, light rail, tram",
    "ts061_Method of travel to workplace: Train",
    "ts061_Method of travel to workplace: Bus, minibus or coach",
]
TS061_TOTAL_COL: str = (
    "ts061_Method of travel to workplace: Total: All usual residents aged 16 years "
    "and over in employment the week before the census"
)

#: Floor on the public-transport commute share. An OA where nobody commutes by
#: public transport still makes some bus and rail journeys, so a zero share would
#: allocate it no mileage at all; the floor keeps every OA in the distribution.
PT_SHARE_FLOOR: float = 0.01

#: Public-transport energy per passenger-mile, expressed as a fraction of the
#: petrol-car intensity so that no external emission factor enters the sweep.
#:
#: ``PT_INTENSITY_ASSUMED`` is the working assumption. Rail and Underground carry
#: about three-quarters of public-transport mileage in NTS9904 and run well below
#: a car per passenger-mile; buses run near a car. Weighting the two by that
#: mileage mix lands near a third, and 0.35 rounds it upward. The value is an
#: assumption, not a measurement: energy per passenger-mile depends on vehicle
#: load factors that no Output-Area source records.
#:
#: ``PT_INTENSITY_CEILING`` charges every public-transport passenger-mile as
#: though it had been driven alone in a petrol car. No bus, tram or train
#: approaches it, so it is an upper bound requiring no external factor at all.
PT_INTENSITY_ASSUMED: float = 0.35
PT_INTENSITY_CEILING: float = 1.0

_NTS_PATH = DATA_DIR / "statistics" / "nts_mileage_by_ruc.parquet"
_RUC_PATH = DATA_DIR / "statistics" / "oa21_ruc21.parquet"

#: Columns produced by :func:`compute_travel_energy`. Dropped on re-entry so the
#: function is idempotent when called on a frame it has already processed (the
#: loader calls it once, and the sensitivity report re-runs it several times).
_TRAVEL_COLS = [
    "RUC21NM",
    "ruc_class_miles_pp",
    "ruc_class_pt_miles_pp",
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
    nts_cols = ["ruc21_name", "car_miles_per_person", "pt_miles_per_person"]
    nts = pd.read_parquet(_NTS_PATH)[nts_cols]
    # Class names differ only in capitalisation between the two sources.
    ruc["_key"] = ruc["RUC21NM"].str.lower().str.strip()
    nts["_key"] = nts["ruc21_name"].str.lower().str.strip()
    # NTS has one row per rural-urban class; the RUC lookup one row per OA.
    mileage_cols = ["car_miles_per_person", "pt_miles_per_person"]
    ruc = ruc.merge(nts[["_key", *mileage_cols]], on="_key", how="left", validate="m:1")
    out = lsoa.merge(
        ruc[["OA21CD", "RUC21NM", *mileage_cols]],
        on="OA21CD",
        how="left",
        validate="m:1",
    )
    return out.rename(
        columns={
            "car_miles_per_person": "ruc_class_miles_pp",
            "pt_miles_per_person": "ruc_class_pt_miles_pp",
        }
    )


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


#: Columns produced by :func:`compute_pt_energy`, dropped on re-entry so the
#: intensity sweep can call it repeatedly on the same frame.
_PT_COLS = ["pt_share_commute", "pt_miles_per_person", "pt_kwh_per_hh"]


def pt_commute_share(df: pd.DataFrame) -> pd.Series:
    """
    Share of an OA's commuters travelling to work by public transport (TS061).

    Parameters
    ----------
    df : pandas.DataFrame
        OA frame carrying the Census TS061 method-of-travel-to-work columns.

    Returns
    -------
    pandas.Series
        Bus, train, metro and tram commuters over all commuters. ``NaN`` where
        the table records no commuters at all.
    """
    num = sum(
        pd.to_numeric(df[col], errors="coerce").fillna(0.0) for col in TS061_PT_COLS
    )
    den = pd.to_numeric(df[TS061_TOTAL_COL], errors="coerce")
    return num / den.replace(0, np.nan)


def compute_pt_energy(
    df: pd.DataFrame,
    intensity_ratio: float = 1.0,
    allocate: bool = True,
) -> pd.DataFrame:
    """
    Add public-transport passenger mileage and its energy (kWh per household).

    The construction mirrors :func:`compute_travel_energy` exactly: the measured
    NTS class marginal for bus, rail and Underground mileage is redistributed
    within each rural-urban class by a local allocator, normalised so the
    population-weighted class mean is preserved. Only the allocator differs, the
    public-transport commute share standing where cars per person stands for car
    travel.

    This does not enter the headline energy axis. It exists to bound how far the
    reported form gradient could move if public-transport energy were charged to
    residents (:func:`public_transport_report`).

    Parameters
    ----------
    df : pandas.DataFrame
        OA frame that has been through :func:`compute_travel_energy`, so it
        carries ``RUC21NM`` and ``ruc_class_pt_miles_pp``, plus the TS061 columns.
    intensity_ratio : float, default 1.0
        Public-transport energy per passenger-mile as a fraction of the petrol-car
        intensity :data:`KWH_PER_MILE_ICE`. 1.0 charges every passenger-mile as
        though it had been driven alone in a petrol car, an upper bound no mode
        approaches.
    allocate : bool, default True
        Whether to redistribute the class mileage by the commute-share allocator.
        ``False`` gives every OA in a class the class average, the neutral variant
        that carries the measured between-class gradient and nothing else.

    Returns
    -------
    pandas.DataFrame
        ``df`` plus ``pt_share_commute``, ``pt_miles_per_person`` and
        ``pt_kwh_per_hh``.
    """
    df = df.drop(columns=[c for c in _PT_COLS if c in df.columns])
    hh_size = pd.to_numeric(df["avg_hh_size"], errors="coerce")
    pop = pd.to_numeric(df["total_people"], errors="coerce")
    class_pt = pd.to_numeric(df["ruc_class_pt_miles_pp"], errors="coerce")

    share = pt_commute_share(df)
    df["pt_share_commute"] = share
    w = share.clip(lower=PT_SHARE_FLOOR) if allocate else pd.Series(1.0, index=df.index)

    valid = w.notna() & pop.notna() & (pop > 0) & class_pt.notna()
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

    allocated = valid & (wbar > 0)
    df["pt_miles_per_person"] = np.where(allocated, class_pt * w / wbar, class_pt)
    df["pt_kwh_per_hh"] = (
        df["pt_miles_per_person"] * hh_size * intensity_ratio * KWH_PER_MILE_ICE
    )
    return df


def public_transport_report() -> None:
    """
    Bound the effect of charging public-transport energy to residents.

    The headline energy axis counts car travel only. Public transport is measured
    by the same National Travel Survey table, so the same constrained
    disaggregation can be run for it; what has no measured counterpart is the
    energy per passenger-mile, which depends on vehicle load factors that vary by
    mode, place and time of day.

    Three rows are reported: the headline axis, the axis with public transport at
    the assumed intensity, and the axis with every passenger-mile charged at the
    petrol-car intensity. The last needs no external factor, so it bounds the
    question outright. Both public-transport rows use the TS061 commute-share
    allocator, which concentrates the mileage in the compact areas that use it and
    is therefore the variant least favourable to the reported gap; the
    class-average allocator is reported alongside for the table caption.
    """
    import ledger
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

    def _cell(c: tuple[float, float, float, float]) -> str:
        """One table cell: point estimate and its clustered interval."""
        return f"{ledger.pt(c[0])}$\\times$ {ledger.ci(c[1], c[2])}"

    base = load_and_aggregate()
    cf = _compositional_frame(base)
    confounds = (
        ["median_build_year"] + _deprivation_cols(cf) + _tenure_cols(cf) + _hdd_cols(cf)
    )
    heat = pd.to_numeric(cf["building_kwh_per_hh"], errors="coerce")
    car = pd.to_numeric(cf["transport_kwh_per_hh_total_est"], errors="coerce")

    keep = [*_SHARE_FRACS, *confounds, "total_hh", "building_kwh_per_hh"]
    cf = cf[heat.notna() & car.notna()].dropna(subset=keep).copy()
    heat = pd.to_numeric(cf["building_kwh_per_hh"], errors="coerce")
    car = pd.to_numeric(cf["transport_kwh_per_hh_total_est"], errors="coerce")

    print("\n" + "=" * 70)
    print("PUBLIC-TRANSPORT SENSITIVITY (not in the headline energy axis)")
    print("=" * 70)
    print(f"\n  Common sample: N = {len(cf):,} OAs")

    nts = pd.read_parquet(_NTS_PATH)
    print("\n  Measured NTS9904 class marginals (miles per person per year):")
    print(f"    {'class':<48s} {'car':>7s} {'public transport':>17s} {'rail %':>7s}")
    for _, r in nts.iterrows():
        share = r["pt_rail_miles_per_person"] / r["pt_miles_per_person"]
        print(
            f"    {r['ruc21_name']:<48s} {r['car_miles_per_person']:>7.0f} "
            f"{r['pt_miles_per_person']:>17.0f} {share:>6.0%}"
        )

    # Population-weighted rail share of public-transport mileage: the mix that
    # justifies the assumed intensity (rail runs well below a car per
    # passenger-mile, buses near it).
    pop_by_class = (
        pd.to_numeric(base["total_people"], errors="coerce")
        .groupby(base["RUC21NM"].str.lower().str.strip())
        .sum()
    )
    key = nts["ruc21_name"].str.lower().str.strip()
    wt = key.map(pop_by_class).fillna(0.0)
    rail_share = float(
        (nts["pt_rail_miles_per_person"] * wt).sum()
        / (nts["pt_miles_per_person"] * wt).sum()
    )
    print(f"\n  Rail and Underground are {rail_share:.0%} of public-transport mileage")

    def _gap(series: pd.Series) -> tuple[float, float, float, float]:
        """Compositional pure-type Det:Flat ratio for one energy definition."""
        cf["_y"] = np.log(series.clip(lower=1))
        m = _comp_ols(
            cf, "_y", _SHARE_FRACS + confounds, "total_hh", cluster_col=CLUSTER_COL
        )
        if m is None:
            msg = "compositional fit failed on the public-transport sample"
            raise RuntimeError(msg)
        return log_contrast_ci(m, "s_detached", "s_flat")

    base_travel = _gap(car)
    base_total = _gap(heat + car)
    print("\n  Baseline on this sample (car travel only):")
    print(f"    travel Det:Flat {fmt_ci(base_travel)}")
    print(f"    total  Det:Flat {fmt_ci(base_total)}")

    def _variant(
        ratio: float, allocate: bool
    ) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
        """Travel and total Det:Flat ratios with public transport added."""
        d = compute_pt_energy(cf, intensity_ratio=ratio, allocate=allocate)
        pt_kwh = pd.to_numeric(d["pt_kwh_per_hh"], errors="coerce").fillna(0.0)
        return _gap(car + pt_kwh), _gap(heat + car + pt_kwh)

    assumed_pct = f"{PT_INTENSITY_ASSUMED:.2f}"
    labels = (
        "Excluded (headline axis)",
        f"Included, at {assumed_pct}$\\times$ car intensity",
        "Included, at car intensity (ceiling)",
    )
    a_trav, a_tot = _variant(PT_INTENSITY_ASSUMED, allocate=True)
    c_trav, c_tot = _variant(PT_INTENSITY_CEILING, allocate=True)

    rows = [
        f"{labels[0]} & {_cell(base_travel)} & {_cell(base_total)} \\\\",
        f"{labels[1]} & {_cell(a_trav)} & {_cell(a_tot)} \\\\",
        f"{labels[2]} & {_cell(c_trav)} & {_cell(c_tot)} \\\\",
    ]

    print("\n  Commute-share allocator (the variant least favourable to the gap):")
    print(f"    {'public-transport energy':<38s}{'travel':>22s}{'total':>22s}")
    for label, trav, tot in (
        ("excluded (headline axis)", base_travel, base_total),
        (f"included, {assumed_pct}× car intensity", a_trav, a_tot),
        ("included, at car intensity", c_trav, c_tot),
    ):
        print(f"    {label:<38s}{fmt_ci(trav):>22s}{fmt_ci(tot):>22s}")

    # Class-average allocator: the between-class gradient with no within-class
    # signal. Recorded for the table caption rather than given its own rows.
    _, ca_tot = _variant(PT_INTENSITY_ASSUMED, allocate=False)
    _, cc_tot = _variant(PT_INTENSITY_CEILING, allocate=False)
    print(
        f"\n  Class-average allocator, total: {ledger.pt(ca_tot[0])}× at "
        f"{assumed_pct}× car intensity, {ledger.pt(cc_tot[0])}× at car intensity"
    )

    ledger.record(
        ptBaseTotal=ledger.pt(base_total[0]),
        ptSampleN=f"{len(cf):,}",
        ptIntensityAssumed=assumed_pct,
        ptRailShare=f"{rail_share * 100:.0f}",
        ptAssumedTravel=ledger.pt(a_trav[0]),
        ptAssumedTravelCI=ledger.ci(a_trav[1], a_trav[2]),
        ptAssumedTotal=ledger.pt(a_tot[0]),
        ptAssumedTotalCI=ledger.ci(a_tot[1], a_tot[2]),
        ptCeilingTravel=ledger.pt(c_trav[0]),
        ptCeilingTravelCI=ledger.ci(c_trav[1], c_trav[2]),
        ptCeilingTotal=ledger.pt(c_tot[0]),
        ptCeilingTotalCI=ledger.ci(c_tot[1], c_tot[2]),
        ptClassAssumedTotal=ledger.pt(ca_tot[0]),
        ptClassCeilingTotal=ledger.pt(cc_tot[0]),
    )
    ledger.table("publictransport", "\n".join(rows) + "\n")
    print(
        "\n  Every row preserves the measured NTS class marginals for both modes;\n"
        "  only the intensity assumption varies."
    )


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

    def _med_ratio(kind: str) -> str:
        # Detached:flat quotient of the dominant-type median columns — the
        # observed counterpart to the compositional pure-type ratio.
        return f"{med[kind]['Detached'] / med[kind]['Flat']:.2f}$\\times$"

    ledger.table(
        "axesenergy",
        "\\multicolumn{8}{l}{\\textit{Energy (kWh per dwelling per year)}} \\\\\n"
        f"Home energy{_cells('heat')} & {_med_ratio('heat')} & "
        "\\nepiheatGap$\\times$ & \\nepiheatFamGap$\\times$ \\\\\n"
        f"Car travel{_cells('car')} & {_med_ratio('car')} & "
        "\\nepitravelGap$\\times$ & \\nepitravelFamGap$\\times$ \\\\\n"
        f"Total{_cells('tot')} & {_med_ratio('tot')} & \\nepitotalGap$\\times$ & "
        "\\nepifamNowGap$\\times$ \\\\\n",
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
    public_transport_report()
