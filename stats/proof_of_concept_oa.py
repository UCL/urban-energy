"""
OA-Level Accessibility Analysis: Three Energy Surfaces.

Output Area (OA) version of the proof-of-concept analysis. Identical
analytical framework to the LSOA version (proof_of_concept_lsoa.py) but
at ~5x finer spatial resolution (~130 households per OA vs ~700 per LSOA).

Key differences from LSOA version:
- Unit of analysis: OA (~130 households) instead of LSOA (~700)
- Energy source: postcode-aggregated DESNZ metered data (via OA lookup)
- Census data: native at OA level (no OA→LSOA aggregation needed)
- Deprivation: Census TS011 is OA-native; IMD 2025 joined via LSOA FK
- Filter thresholds: relaxed for smaller units

Usage:
    uv run python stats/proof_of_concept_oa.py
    uv run python stats/proof_of_concept_oa.py manchester york
"""

import json
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats

from urban_energy.paths import TEMP_DIR

DATA_PATH = TEMP_DIR / "processing" / "combined" / "oa_integrated.gpkg"
RESULTS_PATH = TEMP_DIR / "stats" / "results" / "poc_oa.json"

# ---------------------------------------------------------------------------
# Census column names
# ---------------------------------------------------------------------------
_TS001_POP = "ts001_Residence type: Lives in a household; measures: Value"
_TS017_TOTAL = "ts017_Household size: Total: All household spaces; measures: Value"
_TS017_ZERO = "ts017_Household size: 0 people in household; measures: Value"
_TS011_NOT_DEP = (
    "ts011_Household deprivation: "
    "Household is not deprived in any dimension; measures: Value"
)
_TS011_TOTAL = "ts011_Household deprivation: Total: All households; measures: Value"

# Commute distance bands — midpoint km (travelling commuters only)
_COMMUTE_DISTANCE_BANDS: dict[str, float] = {
    "ts058_Distance travelled to work: Less than 2km": 1.0,
    "ts058_Distance travelled to work: 2km to less than 5km": 3.5,
    "ts058_Distance travelled to work: 5km to less than 10km": 7.5,
    "ts058_Distance travelled to work: 10km to less than 20km": 15.0,
    "ts058_Distance travelled to work: 20km to less than 30km": 25.0,
    "ts058_Distance travelled to work: 30km to less than 40km": 35.0,
    "ts058_Distance travelled to work: 40km to less than 60km": 50.0,
    "ts058_Distance travelled to work: 60km and over": 80.0,
}
_TS058_HOME = "ts058_Distance travelled to work: Works mainly from home"
_TS058_OFFSHORE = (
    "ts058_Distance travelled to work: "
    "Works mainly at an offshore installation, in no fixed place, or outside the UK"
)
_COMMUTE_TOTAL = (
    "ts058_Distance travelled to work: "
    "Total: All usual residents aged 16 years and over "
    "in employment the week before the census"
)
_TS061_CAR = "ts061_Method of travel to workplace: Driving a car or van"
_TS061_PASSENGER = "ts061_Method of travel to workplace: Passenger in a car or van"
_TS061_TAXI = "ts061_Method of travel to workplace: Taxi"
_TS061_MOTORCYCLE = "ts061_Method of travel to workplace: Motorcycle, scooter or moped"
_TS061_BUS = "ts061_Method of travel to workplace: Bus, minibus or coach"
_TS061_TRAIN = "ts061_Method of travel to workplace: Train"
_TS061_METRO = (
    "ts061_Method of travel to workplace: Underground, metro, light rail, tram"
)
_TS061_TOTAL = (
    "ts061_Method of travel to workplace: "
    "Total: All usual residents aged 16 years and over "
    "in employment the week before the census"
)
_TS061_WALK = "ts061_Method of travel to workplace: On foot"
_TS061_CYCLE = "ts061_Method of travel to workplace: Bicycle"
_TS061_HOME = "ts061_Method of travel to workplace: Work mainly at or from home"
_TS061_OTHER = "ts061_Method of travel to workplace: Other method of travel to work"
_TS006_DENSITY = (
    "ts006_Population Density: Persons per square kilometre; measures: Value"
)

# Car ownership (ts045) — for total transport energy
_TS045_TOTAL = "ts045_Number of cars or vans: Total: All households"
_TS045_NONE = "ts045_Number of cars or vans: No cars or vans in household"
_TS045_ONE = "ts045_Number of cars or vans: 1 car or van in household"
_TS045_TWO = "ts045_Number of cars or vans: 2 cars or vans in household"
_TS045_THREE = "ts045_Number of cars or vans: 3 or more cars or vans in household"

# Accommodation type (ts044) — Census-derived, complete coverage
_TS044_TOTAL = "ts044_Accommodation type: Total: All households"
_TS044_DETACHED = "ts044_Accommodation type: Detached"
_TS044_SEMI = "ts044_Accommodation type: Semi-detached"
_TS044_TERRACED = "ts044_Accommodation type: Terraced"
_TS044_FLAT = "ts044_Accommodation type: In a purpose-built block of flats or tenement"
_TS044_COMMERCIAL = (
    "ts044_Accommodation type: In a commercial building, "
    "for example, in an office building, hotel or over a shop"
)

# Mode energy intensities for commute-energy decomposition (kWh/passenger-km):
# Best available national passenger-energy intensities (ECUK 2025):
# - Road passenger: 34.3 ktoe / billion passenger-km (2023)
# - Rail passenger: 15.3 ktoe / billion passenger-km (2024)
# Conversion: 1 ktoe = 11.63 GWh, so kWh/pkm = ktoe_per_billion_pkm * 0.01163
_KWH_PER_PKM_ROAD = 34.3 * 0.01163  # ≈ 0.399
_KWH_PER_PKM_RAIL = 15.3 * 0.01163  # ≈ 0.178
# NTS 2024 distance ratio (England): total miles / commuting miles per person.
# Used only for a secondary "overall travel" scenario estimate.
_NTS_TOTAL_TO_COMMUTE_DISTANCE_FACTOR = 6082 / 1007  # ≈ 6.04

# ---------------------------------------------------------------------------
# Cityseer accessibility — only the columns we use
# ---------------------------------------------------------------------------
_CC_ACCESSIBILITY = [
    "cc_fsa_restaurant_800_wt",
    "cc_fsa_pub_800_wt",
    "cc_fsa_takeaway_800_wt",
    "cc_fsa_other_800_wt",
    "cc_bus_800_wt",
    "cc_rail_800_wt",
    "cc_greenspace_800_wt",
]
_CC_CENTRALITY = [
    "cc_harmonic_800",
    "cc_density_800",
]

# Metered energy — postcode-aggregated to OA
_OA_ENERGY = ["oa_total_mean_kwh"]

# Building physics (LiDAR + OS footprints) — optional robustness check.
# Aggregated from all buildings (incl. non-domestic); not used as primary
# predictors. ts044 accommodation type is the domestic-only form variable.
_BUILDING_COLS = [
    "footprint_area_m2",
    "volume_m3",
    "envelope_area_m2",
    "surface_to_volume",
    "height_mean",
    "form_factor",
]

# Census columns required for the analysis
_CENSUS_REQUIRED = [
    _TS001_POP,
    _TS017_TOTAL,
    _TS017_ZERO,
    _TS011_NOT_DEP,
    _TS011_TOTAL,
    _COMMUTE_TOTAL,
    _TS058_HOME,
    _TS058_OFFSHORE,
    _TS061_CAR,
    _TS061_PASSENGER,
    _TS061_TAXI,
    _TS061_MOTORCYCLE,
    _TS061_BUS,
    _TS061_TRAIN,
    _TS061_METRO,
    _TS061_TOTAL,
    _TS061_WALK,
    _TS061_CYCLE,
    _TS061_HOME,
    _TS061_OTHER,
    _TS006_DENSITY,
    _TS045_TOTAL,
    _TS045_NONE,
    _TS045_ONE,
    _TS045_TWO,
    _TS045_THREE,
    _TS044_TOTAL,
    _TS044_DETACHED,
    _TS044_SEMI,
    _TS044_TERRACED,
    _TS044_FLAT,
    _TS044_COMMERCIAL,
    *_COMMUTE_DISTANCE_BANDS.keys(),
]

# Every column the script requires — missing any of these is a hard error
_REQUIRED = (
    ["OA21CD"] + _OA_ENERGY + _CC_ACCESSIBILITY + _CC_CENTRALITY + _CENSUS_REQUIRED
)

# Optional columns — warn if missing, do not error
_OPTIONAL = _BUILDING_COLS


def _sigstars(p: float) -> str:
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _validate_columns(available: set[str]) -> None:
    """
    Check that all required columns exist in the dataset.

    Raises
    ------
    ValueError
        If any required columns are missing.
    """
    missing = [c for c in _REQUIRED if c not in available]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )
    opt_missing = [c for c in _OPTIONAL if c not in available]
    if opt_missing:
        print(
            f"  NOTE: {len(opt_missing)} optional (LiDAR) columns missing "
            f"— S/V robustness check will be skipped"
        )


# ---------------------------------------------------------------------------
# 1. Load and aggregate to OA
# ---------------------------------------------------------------------------


def load_and_aggregate(cities: list[str] | None = None) -> pd.DataFrame:
    """
    Load pre-aggregated OA data from pipeline_oa.py output.

    The pipeline has already performed:
    - Census deduplication (OA-native, single-step)
    - Building physics aggregation (sum/mean per OA)
    - Cityseer metric averaging across UPRNs
    - EPC coverage and building era derivation
    - Postcode-aggregated metered energy join
    - IMD 2025 and DVLA vehicles join (via LSOA foreign key)
    - Aggregate S/V ratio computation

    This function reads the OA GeoPackage and computes analytical
    derived variables (transport energy, deprivation, housing type, etc.).

    Parameters
    ----------
    cities : list[str] or None
        Filter to specific cities, or None for all.

    Returns
    -------
    pd.DataFrame
        OA-level dataset with derived analytical variables.

    Raises
    ------
    ValueError
        If required columns are missing from the dataset.
    """
    print("=" * 70)
    print("LOADING OA DATA")
    print("=" * 70)

    # Validate columns
    _probe = gpd.read_file(DATA_PATH, rows=1)
    available = set(_probe.columns)
    del _probe
    _validate_columns(available)

    lsoa = gpd.read_file(DATA_PATH)
    print(f"  Loaded {len(lsoa):,} OAs ({len(lsoa.columns)} columns)")

    if cities and "city" in lsoa.columns:
        lsoa = lsoa[lsoa["city"].isin(cities)].copy()
        print(f"  Filtered to {cities}: {len(lsoa):,}")

    lsoa = lsoa[lsoa["OA21CD"].notna()].copy()

    # Coerce numerics
    for col in _BUILDING_COLS:
        if col in lsoa.columns:
            lsoa[col] = pd.to_numeric(lsoa[col], errors="coerce")

    # Drop geometry — analysis is non-spatial from here
    if "geometry" in lsoa.columns:
        lsoa = pd.DataFrame(lsoa.drop(columns=["geometry"]))

    # Building era categories (from pipeline's median_build_year)
    if "median_build_year" in lsoa.columns:
        bins = [0, 1945, 1982, 2100]
        labels = ["pre-1945", "1945-1982", "post-1982"]
        valid_yr = lsoa["median_build_year"].notna()
        lsoa.loc[valid_yr, "building_era"] = pd.cut(
            lsoa.loc[valid_yr, "median_build_year"],
            bins=bins,
            labels=labels,
        )

    # ===================================================================
    # Derived variables
    # ===================================================================

    # --- Aggregate S/V (robustness check — includes non-domestic buildings) ---
    # Pipeline pre-computes lsoa_sv; recompute as fallback
    if "oa_sv" not in lsoa.columns and "envelope_area_m2" in lsoa.columns:
        lsoa["oa_sv"] = lsoa["envelope_area_m2"] / lsoa["volume_m3"].replace(0, np.nan)
    if "surface_to_volume" in lsoa.columns:
        lsoa["mean_building_sv"] = lsoa["surface_to_volume"]

    # --- Population and households ---
    lsoa["total_people"] = pd.to_numeric(lsoa[_TS001_POP], errors="coerce")
    lsoa["total_hh"] = pd.to_numeric(
        lsoa[_TS017_TOTAL], errors="coerce"
    ) - pd.to_numeric(lsoa[_TS017_ZERO], errors="coerce")

    # ===================================================================
    # THE COST SIDE: energy per household
    # ===================================================================

    # Building energy (metered, postcode-aggregated to OA)
    lsoa["building_kwh_per_hh"] = pd.to_numeric(
        lsoa["oa_total_mean_kwh"], errors="coerce"
    )

    # --- Transport energy from Census ts058 + ts061 ---
    # Private vs public commute energy from mode counts (ts061),
    # with distance from ts058 bands (excluding WFH/offshore).
    # Energy conversion uses national passenger intensities:
    #   road modes: _KWH_PER_PKM_ROAD
    #   rail/metro: _KWH_PER_PKM_RAIL

    # ts058: total employed, home-working,
    # offshore/no-fixed-place, and travelling commuters
    total_commuters = pd.to_numeric(lsoa[_COMMUTE_TOTAL], errors="coerce")
    home_commuters_ts058 = pd.to_numeric(lsoa[_TS058_HOME], errors="coerce")
    offshore_commuters = pd.to_numeric(lsoa[_TS058_OFFSHORE], errors="coerce")
    travelling_commuters = total_commuters - home_commuters_ts058 - offshore_commuters
    travelling_commuters = travelling_commuters.clip(lower=0)

    weighted_km = sum(
        km * pd.to_numeric(lsoa[col], errors="coerce")
        for col, km in _COMMUTE_DISTANCE_BANDS.items()
    )
    # Backward-compatible average distance (includes WFH/offshore in denominator).
    lsoa["avg_commute_km"] = weighted_km / total_commuters.replace(0, np.nan)
    # More exact travelling-commuter average distance for mode-energy decomposition.
    lsoa["avg_commute_km_travelling"] = weighted_km / travelling_commuters.replace(
        0, np.nan
    )
    lsoa["work_from_home_share"] = home_commuters_ts058 / total_commuters.replace(
        0, np.nan
    )
    lsoa["offshore_or_no_fixed_share"] = offshore_commuters / total_commuters.replace(
        0, np.nan
    )

    # ts061: mode shares and counts
    car_commuters = pd.to_numeric(lsoa[_TS061_CAR], errors="coerce")
    passenger_commuters = pd.to_numeric(lsoa[_TS061_PASSENGER], errors="coerce")
    taxi_commuters = pd.to_numeric(lsoa[_TS061_TAXI], errors="coerce")
    motorcycle_commuters = pd.to_numeric(lsoa[_TS061_MOTORCYCLE], errors="coerce")
    bus_commuters = pd.to_numeric(lsoa[_TS061_BUS], errors="coerce")
    train_commuters = pd.to_numeric(lsoa[_TS061_TRAIN], errors="coerce")
    metro_commuters = pd.to_numeric(lsoa[_TS061_METRO], errors="coerce")
    home_commuters_ts061 = pd.to_numeric(lsoa[_TS061_HOME], errors="coerce")
    other_commuters = pd.to_numeric(lsoa[_TS061_OTHER], errors="coerce")
    walk_commuters = pd.to_numeric(lsoa[_TS061_WALK], errors="coerce")
    cycle_commuters = pd.to_numeric(lsoa[_TS061_CYCLE], errors="coerce")

    total_w = pd.to_numeric(lsoa[_TS061_TOTAL], errors="coerce")
    lsoa["car_commute_share"] = car_commuters / total_w.replace(0, np.nan)
    lsoa["walk_share"] = walk_commuters / total_w.replace(0, np.nan)
    lsoa["cycle_share"] = cycle_commuters / total_w.replace(0, np.nan)
    lsoa["active_share"] = lsoa["walk_share"].fillna(0) + lsoa["cycle_share"].fillna(0)
    lsoa["bus_share"] = bus_commuters / total_w.replace(0, np.nan)
    lsoa["rail_share"] = (train_commuters + metro_commuters) / total_w.replace(
        0, np.nan
    )
    lsoa["work_from_home_share_ts061"] = home_commuters_ts061 / total_w.replace(
        0, np.nan
    )

    private_commuters = (
        car_commuters + passenger_commuters + taxi_commuters + motorcycle_commuters
    )
    public_commuters = bus_commuters + train_commuters + metro_commuters
    lsoa["private_commute_share"] = private_commuters / total_w.replace(0, np.nan)
    lsoa["public_commute_share"] = public_commuters / total_w.replace(0, np.nan)
    lsoa["other_mode_share"] = other_commuters / total_w.replace(0, np.nan)

    # Convert commute energy to kWh per household.
    total_hh_for_transport = pd.to_numeric(lsoa[_TS045_TOTAL], errors="coerce").replace(
        0, np.nan
    )

    # Mode-specific commute energy decomposition (kWh/hh),
    # from ts061 shares and ts058 distance.
    mode_distance_annual = lsoa["avg_commute_km_travelling"] * 2 * 220

    private_commute_kwh_annual = (
        mode_distance_annual * private_commuters * _KWH_PER_PKM_ROAD
    )
    public_commute_kwh_annual = (
        mode_distance_annual * bus_commuters * _KWH_PER_PKM_ROAD
        + mode_distance_annual * train_commuters * _KWH_PER_PKM_RAIL
        + mode_distance_annual * metro_commuters * _KWH_PER_PKM_RAIL
    )
    lsoa["private_transport_kwh_per_hh_est"] = (
        private_commute_kwh_annual / total_hh_for_transport
    )
    lsoa["public_transport_kwh_per_hh_est"] = (
        public_commute_kwh_annual / total_hh_for_transport
    )
    lsoa["motorised_commute_kwh_per_hh_est"] = (
        lsoa["private_transport_kwh_per_hh_est"]
        + lsoa["public_transport_kwh_per_hh_est"]
    )
    lsoa["transport_kwh_per_hh_est"] = lsoa["motorised_commute_kwh_per_hh_est"]
    lsoa["transport_kwh_per_hh_total_est"] = (
        lsoa["transport_kwh_per_hh_est"] * _NTS_TOTAL_TO_COMMUTE_DISTANCE_FACTOR
    )

    # Car ownership — kept as a secondary metric
    one_car = pd.to_numeric(lsoa[_TS045_ONE], errors="coerce")
    two_car = pd.to_numeric(lsoa[_TS045_TWO], errors="coerce")
    three_car = pd.to_numeric(lsoa[_TS045_THREE], errors="coerce")
    car_hh_total = pd.to_numeric(lsoa[_TS045_TOTAL], errors="coerce")
    lsoa["cars_per_hh"] = (
        one_car + 2 * two_car + 3 * three_car
    ) / car_hh_total.replace(0, np.nan)

    # Total energy cost per household
    lsoa["total_kwh_per_hh"] = lsoa["building_kwh_per_hh"] + lsoa[
        "transport_kwh_per_hh_est"
    ].fillna(0)
    lsoa["total_kwh_per_hh_total_est"] = lsoa["building_kwh_per_hh"] + lsoa[
        "transport_kwh_per_hh_total_est"
    ].fillna(0)

    lsoa["log_total_kwh_per_hh"] = np.log(lsoa["total_kwh_per_hh"].clip(lower=1))
    lsoa["log_building_kwh_per_hh"] = np.log(lsoa["building_kwh_per_hh"].clip(lower=1))

    # --- Occupation density (Census-derived, no building-type confound) ---
    lsoa["avg_hh_size"] = lsoa["total_people"] / lsoa["total_hh"].replace(0, np.nan)

    # --- Population density (from Census OA areas) ---
    if "_oa_area_km2" in lsoa.columns:
        lsoa_area_km2 = lsoa["_oa_area_km2"]
        lsoa["people_per_ha"] = lsoa["total_people"] / (lsoa_area_km2 * 100).replace(
            0, np.nan
        )
        lsoa.drop(columns=["_oa_area_km2"], inplace=True)

    # --- Energy per person (building only, metered) ---
    lsoa_total_bldg = lsoa["building_kwh_per_hh"] * lsoa["total_hh"]
    lsoa["kwh_per_person"] = lsoa_total_bldg / lsoa["total_people"].replace(0, np.nan)
    # Total (building + transport) per person
    lsoa_total_all = lsoa["total_kwh_per_hh"] * lsoa["total_hh"]
    lsoa["total_kwh_per_person"] = lsoa_total_all / lsoa["total_people"].replace(
        0, np.nan
    )

    # NOTE: volume_per_person and kwh_per_m3 are not computed. Building
    # volume is scaled by EPC coverage (which varies spatially), while
    # Census population and metered energy have complete coverage. Mixing
    # inconsistently-scaled volume with consistently-scaled denominators
    # produces biased cross-area comparisons. S/V is unaffected because
    # the EPC scaling cancels in the ratio (both numerator and denominator
    # are scaled identically).

    # --- Deprivation ---
    not_dep = pd.to_numeric(lsoa[_TS011_NOT_DEP], errors="coerce")
    total_dep = pd.to_numeric(lsoa[_TS011_TOTAL], errors="coerce")
    lsoa["pct_not_deprived"] = not_dep / total_dep.replace(0, np.nan) * 100
    valid_dep = lsoa["pct_not_deprived"].notna()
    if valid_dep.sum() > 10:
        lsoa.loc[valid_dep, "deprivation_quintile"] = pd.qcut(
            lsoa.loc[valid_dep, "pct_not_deprived"],
            q=5,
            labels=["Q1 most", "Q2", "Q3", "Q4", "Q5 least"],
        )

    # --- GVA per household (ONS Small Area GVA) ---
    if "lsoa_gva_millions" in lsoa.columns:
        lsoa["gva_per_hh"] = (lsoa["lsoa_gva_millions"] * 1_000_000) / lsoa[
            "total_hh"
        ].replace(0, np.nan)

    # --- Accommodation type (Census ts044, complete coverage) ---
    ts044_total = pd.to_numeric(lsoa[_TS044_TOTAL], errors="coerce")
    ts044_denom = ts044_total.replace(0, np.nan)
    lsoa["pct_detached"] = (
        pd.to_numeric(lsoa[_TS044_DETACHED], errors="coerce") / ts044_denom * 100
    )
    lsoa["pct_semi"] = (
        pd.to_numeric(lsoa[_TS044_SEMI], errors="coerce") / ts044_denom * 100
    )
    lsoa["pct_terraced"] = (
        pd.to_numeric(lsoa[_TS044_TERRACED], errors="coerce") / ts044_denom * 100
    )
    lsoa["pct_flat"] = (
        pd.to_numeric(lsoa[_TS044_FLAT], errors="coerce") / ts044_denom * 100
    )
    lsoa["pct_in_commercial"] = (
        pd.to_numeric(lsoa[_TS044_COMMERCIAL], errors="coerce") / ts044_denom * 100
    )

    # --- Dominant housing type (Census ts044, complete coverage) ---
    # Assign each OA the accommodation type with the highest percentage.
    # Ordered categorical: compact → sprawl.
    _type_map = {
        "pct_flat": "Flat",
        "pct_terraced": "Terraced",
        "pct_semi": "Semi",
        "pct_detached": "Detached",
    }
    _type_order = ["Flat", "Terraced", "Semi", "Detached"]
    type_pcts = lsoa[list(_type_map.keys())].fillna(0)
    lsoa["dominant_type"] = pd.Categorical(
        type_pcts.idxmax(axis=1).map(_type_map),
        categories=_type_order,
        ordered=True,
    )

    # --- Filter (relaxed for smaller OA units) ---
    valid = (
        (lsoa["total_people"] > 10)
        & (lsoa["n_uprns"] >= 5)
        & lsoa["building_kwh_per_hh"].notna()
        & (lsoa["building_kwh_per_hh"] > 0)
    )
    lsoa = lsoa[valid].copy()

    n_cities = lsoa["city"].nunique() if "city" in lsoa.columns else 1
    print(f"\n  {len(lsoa):,} OAs across {n_cities} cities")
    print(f"  UPRNs/OA: median={lsoa['n_uprns'].median():.0f}")
    if "oa_sv" in lsoa.columns:
        sv = lsoa["oa_sv"].dropna()
        print(
            f"\n  LiDAR S/V (robustness): median={sv.median():.3f} "
            f"(mean={sv.mean():.3f}, N={len(sv):,})"
        )
    if "height_mean" in lsoa.columns:
        print(f"    Height: median={lsoa['height_mean'].dropna().median():.1f}m")
    print("\n  Cost (kWh/household):")
    print(f"    Building: median={lsoa['building_kwh_per_hh'].median():.0f}")
    t = lsoa["transport_kwh_per_hh_est"].dropna()
    print(
        f"    Transport (est.): median={t.median():,.0f}  "
        f"(cars/hh={lsoa['cars_per_hh'].median():.2f})"
    )
    t_total = lsoa["transport_kwh_per_hh_total_est"].dropna()
    print(
        "    Transport (overall scenario, est.): "
        f"median={t_total.median():,.0f}  "
        f"(factor={_NTS_TOTAL_TO_COMMUTE_DISTANCE_FACTOR:.2f}x)"
    )
    if "private_transport_kwh_per_hh_est" in lsoa.columns:
        print(
            "    Commute mode energy (est., kWh/hh): "
            f"private={lsoa['private_transport_kwh_per_hh_est'].median():,.0f}, "
            f"public={lsoa['public_transport_kwh_per_hh_est'].median():,.0f}"
        )
    print(f"    Total: median={lsoa['total_kwh_per_hh'].median():,.0f}")
    print(
        f"    Total (overall scenario): "
        f"median={lsoa['total_kwh_per_hh_total_est'].median():,.0f}"
    )
    if "median_build_year" in lsoa.columns:
        yr = lsoa["median_build_year"].dropna()
        print(f"\n  Stock: median build year={yr.median():.0f}")
    if "building_era" in lsoa.columns:
        era_counts = lsoa["building_era"].value_counts().sort_index()
        for era, n in era_counts.items():
            print(f"    {era}: {n:,} OAs")
    if "epc_coverage" in lsoa.columns:
        print(f"\n  EPC coverage: median={lsoa['epc_coverage'].median():.1%}")
    print("\n  Accommodation (Census ts044):")
    for label, col in [
        ("Detached", "pct_detached"),
        ("Semi", "pct_semi"),
        ("Terraced", "pct_terraced"),
        ("Flat", "pct_flat"),
    ]:
        if col in lsoa.columns:
            print(f"    {label}: median={lsoa[col].median():.0f}%")
    if "dominant_type" in lsoa.columns:
        print("  Dominant type:")
        for dtype in _type_order:
            n = (lsoa["dominant_type"] == dtype).sum()
            print(f"    {dtype}: {n:,} OAs")

    return lsoa


# ---------------------------------------------------------------------------
# 2. The return side: accessibility
# ---------------------------------------------------------------------------


def build_accessibility(lsoa: pd.DataFrame) -> pd.DataFrame:
    """
    Build two simple accessibility metrics from cityseer columns.

    1. Street frontage: ``cc_density_800`` — reachable street network
       node density within 800m (pedestrian catchment).
    2. FSA count: sum of gravity-weighted FSA establishment counts
       (restaurant + pub + takeaway + other) within 800m.

    Parameters
    ----------
    lsoa : pd.DataFrame
        OA-level data with _CC_ACCESSIBILITY and _CC_CENTRALITY columns.

    Returns
    -------
    pd.DataFrame
        Updated with ``street_frontage``, ``fsa_count``, and
        ``accessibility`` (sum of both, z-scored).
    """
    print("\n" + "=" * 70)
    print("THE RETURN SIDE: What does the energy buy?")
    print("=" * 70)

    # 1. Street frontage — cityseer density centrality at 800m
    if "cc_density_800" not in lsoa.columns:
        raise ValueError("Missing cc_density_800 (street frontage)")
    lsoa["street_frontage"] = pd.to_numeric(lsoa["cc_density_800"], errors="coerce")
    sf_med = lsoa["street_frontage"].median()
    print(f"\n  Street frontage (cc_density_800): median = {sf_med:.1f}")

    # 2. FSA count — sum of all gravity-weighted FSA categories at 800m
    fsa_cols = [c for c in _CC_ACCESSIBILITY if c.startswith("cc_fsa_")]
    missing = [c for c in fsa_cols if c not in lsoa.columns]
    if missing:
        raise ValueError(f"Missing FSA columns: {missing}")
    fsa_vals = lsoa[fsa_cols].apply(pd.to_numeric, errors="coerce")
    lsoa["fsa_count"] = fsa_vals.sum(axis=1)
    print(f"  FSA establishments (800m wt): median = {lsoa['fsa_count'].median():.1f}")
    for c in fsa_cols:
        short = c.replace("cc_fsa_", "").replace("_800_wt", "")
        print(f"    {short:<20s} median = {lsoa[c].median():.1f}")

    # Combined accessibility: z-score both and sum (equal weight)
    sf_z = (lsoa["street_frontage"] - lsoa["street_frontage"].mean()) / lsoa[
        "street_frontage"
    ].std()
    fsa_z = (lsoa["fsa_count"] - lsoa["fsa_count"].mean()) / lsoa["fsa_count"].std()
    lsoa["accessibility"] = sf_z + fsa_z
    acc_med = lsoa["accessibility"].median()
    print(f"  Combined accessibility (z-scored sum): median = {acc_med:.2f}")

    if "cc_harmonic_800" in lsoa.columns:
        print(f"  cc_harmonic_800: median={lsoa['cc_harmonic_800'].median():.3f}")

    return lsoa


# ---------------------------------------------------------------------------
# 3. Walkable destinations per kWh
# ---------------------------------------------------------------------------


def compute_access_per_kwh(lsoa: pd.DataFrame) -> pd.DataFrame:
    """
    Derive walkable-destinations-per-kWh ratio and density quartiles.

    Parameters
    ----------
    lsoa : pd.DataFrame
        OA data with accessibility and energy.

    Returns
    -------
    pd.DataFrame
        Updated with ``access_per_kwh``, ``log_access_per_kwh``,
        and ``density_quartile``.
    """
    print("\n" + "=" * 70)
    print("WALKABLE DESTINATIONS PER kWh")
    print("=" * 70)

    if "accessibility" not in lsoa.columns:
        raise ValueError("accessibility not found — run build_accessibility() first")

    # Shift accessibility positive for meaningful ratios
    acc = lsoa["accessibility"]
    acc_pos = acc - acc.min() + 1
    lsoa["access_per_kwh"] = acc_pos / lsoa["total_kwh_per_hh"]
    lsoa["log_access_per_kwh"] = np.log(lsoa["access_per_kwh"].clip(lower=1e-10))
    lsoa["kwh_per_access"] = lsoa["total_kwh_per_hh"] / acc_pos

    # Population density quartiles
    lsoa["density_quartile"] = pd.qcut(
        lsoa["people_per_ha"],
        4,
        labels=["Q4 sparse", "Q3", "Q2", "Q1 dense"],
    )

    # Summary by dominant housing type (Census ts044)
    types = ["Flat", "Terraced", "Semi", "Detached"]
    print(
        f"\n  {'Type':<14s} {'%det':>6s} {'%flat':>6s} "
        f"{'kWh/hh':>8s} {'Streets':>8s} {'FSA':>6s} "
        f"{'Access':>7s} {'Acc/kWh':>8s} {'N':>5s}"
    )
    print(f"  {'-' * 75}")

    for t in types:
        sub = lsoa[lsoa["dominant_type"] == t]
        if len(sub) == 0:
            continue
        print(
            f"  {t:<14s} {sub['pct_detached'].median():>5.0f}% "
            f"{sub['pct_flat'].median():>5.0f}% "
            f"{sub['total_kwh_per_hh'].median():>8.0f} "
            f"{sub['street_frontage'].median():>8.1f} "
            f"{sub['fsa_count'].median():>6.1f} "
            f"{sub['accessibility'].median():>7.2f} "
            f"{sub['access_per_kwh'].median():>8.5f} "
            f"{len(sub):>5d}"
        )

    flat = lsoa[lsoa["dominant_type"] == "Flat"]
    det = lsoa[lsoa["dominant_type"] == "Detached"]
    if len(flat) > 0 and len(det) > 0:
        ratio = flat["access_per_kwh"].median() / det["access_per_kwh"].median()
        print(
            f"\n  Flat-dominant: {ratio:.2f}x more accessibility per kWh than detached"
        )
        sf_f = flat["street_frontage"].median()
        sf_d = det["street_frontage"].median()
        fsa_f = flat["fsa_count"].median()
        fsa_d = det["fsa_count"].median()
        print(f"    Streets: flat {sf_f:.0f} vs detached {sf_d:.0f}")
        print(f"    FSA: flat {fsa_f:.1f} vs detached {fsa_d:.1f}")

    return lsoa


# ---------------------------------------------------------------------------
# 4. Systematic summary: three energy surfaces
# ---------------------------------------------------------------------------


def print_systematic_summary(lsoa: pd.DataFrame) -> None:
    """
    Decompose the urban energy landscape into three surfaces.

    Stratified by Census ts044 dominant housing type (domestic-only,
    complete coverage). LiDAR S/V shown for validation where available.
    """
    print(f"\n{'=' * 70}")
    print("THREE ENERGY SURFACES BY DOMINANT HOUSING TYPE (Census ts044)")
    print("=" * 70)

    if "dominant_type" not in lsoa.columns:
        return

    types = ["Flat", "Terraced", "Semi", "Detached"]
    w = 12  # column width

    # (section, label, column, format)
    metrics: list[tuple[str, str, str, str]] = [
        ("FORM", "% detached", "pct_detached", ".0f"),
        ("FORM", "% semi", "pct_semi", ".0f"),
        ("FORM", "% terraced", "pct_terraced", ".0f"),
        ("FORM", "% flat", "pct_flat", ".0f"),
        ("FORM (LiDAR)", "S/V ratio", "oa_sv", ".3f"),
        ("FORM (LiDAR)", "Height (m)", "height_mean", ".1f"),
        ("STOCK", "Build year", "median_build_year", ".0f"),
        ("DENSITY", "People/ha", "people_per_ha", ".0f"),
        ("DENSITY", "People/hh", "avg_hh_size", ".2f"),
        ("1:THERMAL", "kWh/hh (bldg)", "building_kwh_per_hh", ",.0f"),
        ("1:THERMAL", "kWh/person (bldg)", "kwh_per_person", ",.0f"),
        ("2:MOBILITY", "Cars/hh", "cars_per_hh", ".2f"),
        ("2:MOBILITY", "kWh/hh (trans)", "transport_kwh_per_hh_est", ",.0f"),
        (
            "2:MOBILITY",
            "kWh/hh (trans total est)",
            "transport_kwh_per_hh_total_est",
            ",.0f",
        ),
        ("2:MOBILITY", "kWh/hh (total)", "total_kwh_per_hh", ",.0f"),
        ("2:MOBILITY", "kWh/hh (total est)", "total_kwh_per_hh_total_est", ",.0f"),
        ("2:MOBILITY", "kWh/person+trans", "total_kwh_per_person", ",.0f"),
        ("3:ACCESS", "Street frontage", "street_frontage", ".0f"),
        ("3:ACCESS", "FSA count", "fsa_count", ".1f"),
        ("3:ACCESS", "Accessibility", "accessibility", ".2f"),
        ("3:ACCESS", "Access/kWh", "access_per_kwh", ".5f"),
    ]

    # Header
    hdr = f"  {'':22s}"
    for t in types:
        hdr += f" {t:>{w}s}"
    hdr += f" {'Flat/Det':>8s}"
    print(f"\n{hdr}")
    print(f"  {'-' * (22 + len(types) * (w + 1) + 9)}")

    # Count row
    line = f"    {'N OAs':<20s}"
    for t in types:
        n = (lsoa["dominant_type"] == t).sum()
        line += f" {n:>{w},d}"
    print(line)
    print()

    prev_section = ""
    for section, label, col, fmt in metrics:
        if col not in lsoa.columns:
            continue
        if section != prev_section:
            if prev_section:
                print()
            print(f"  {section}")
            prev_section = section

        vals = [lsoa.loc[lsoa["dominant_type"] == t, col].median() for t in types]
        line = f"    {label:<20s}"
        for v in vals:
            line += f" {v:>{w}{fmt}}"
        v_flat, v_det = vals[0], vals[3]
        if v_det != 0 and not np.isnan(v_flat) and not np.isnan(v_det):
            line += f" {v_flat / v_det:>7.2f}x"
        print(line)

    # Narrative: three surfaces decomposition
    flat = lsoa[lsoa["dominant_type"] == "Flat"]
    det = lsoa[lsoa["dominant_type"] == "Detached"]
    if len(flat) == 0 or len(det) == 0:
        return

    print("\n  Three surfaces (Flat-dominant / Detached-dominant):")

    r_bldg = flat["building_kwh_per_hh"].median()
    r_bldg /= det["building_kwh_per_hh"].median()
    r_pcap = flat["kwh_per_person"].median()
    r_pcap /= det["kwh_per_person"].median()
    print("    1. THERMAL SURFACE")
    print(f"       kWh/hh (bldg)    {r_bldg:.2f}x  -- modest gradient")
    print(
        f"       kWh/person (bldg) {r_pcap:.2f}x  -- "
        "smaller households partly wash out S/V gain"
    )

    r_cars = flat["cars_per_hh"].median()
    r_cars /= det["cars_per_hh"].median()
    r_trans = flat["transport_kwh_per_hh_est"].median()
    r_trans /= det["transport_kwh_per_hh_est"].median()
    r_total = flat["total_kwh_per_hh"].median()
    r_total /= det["total_kwh_per_hh"].median()
    print("    2. MOBILITY SURFACE")
    print(f"       Cars/hh           {r_cars:.2f}x  -- fewer cars in flat areas")
    print(f"       kWh/hh (trans)    {r_trans:.2f}x  -- THE dominant cost gradient")
    print(f"       kWh/hh (total)    {r_total:.2f}x  -- transport flips the sign")

    sf_col = "street_frontage"
    sf_f = flat[sf_col].median() if sf_col in flat.columns else np.nan
    sf_d = det[sf_col].median() if sf_col in det.columns else np.nan
    fsa_f = flat["fsa_count"].median() if "fsa_count" in flat.columns else np.nan
    fsa_d = det["fsa_count"].median() if "fsa_count" in det.columns else np.nan
    print("    3. ACCESSIBILITY SURFACE")
    print(f"       Street frontage   {sf_f:.0f} vs {sf_d:.0f}")
    print(f"       FSA count         {fsa_f:.1f} vs {fsa_d:.1f}")
    if "access_per_kwh" in lsoa.columns:
        r = flat["access_per_kwh"].median()
        r /= det["access_per_kwh"].median()
        print(f"       Access/kWh        {r:.2f}x  -- more city per kWh")


# ---------------------------------------------------------------------------
# 5. Energy cost decomposition
# ---------------------------------------------------------------------------


def print_energy_decomposition(lsoa: pd.DataFrame) -> None:
    """Show building vs transport energy by dominant housing type."""
    print(f"\n{'=' * 70}")
    print("ENERGY COST DECOMPOSITION")
    print("=" * 70)

    valid = lsoa["transport_kwh_per_hh_est"].notna()
    sub = lsoa[valid]

    print(f"\n  {'Metric':<25s} {'Median':>8s} {'Mean':>8s}")
    print(f"  {'-' * 44}")
    for label, col in [
        ("Building kWh/hh", "building_kwh_per_hh"),
        ("Transport kWh/hh (est)", "transport_kwh_per_hh_est"),
        ("Transport kWh/hh (overall est)", "transport_kwh_per_hh_total_est"),
        ("Total kWh/hh", "total_kwh_per_hh"),
        ("Total kWh/hh (overall est)", "total_kwh_per_hh_total_est"),
    ]:
        s = sub[col]
        print(f"  {label:<25s} {s.median():>8.0f} {s.mean():>8.0f}")
    t_share = sub["transport_kwh_per_hh_est"] / sub["total_kwh_per_hh"]
    print(f"  {'Transport share':<25s} {t_share.median():>7.0%} {t_share.mean():>7.0%}")
    t_total_share = (
        sub["transport_kwh_per_hh_total_est"] / sub["total_kwh_per_hh_total_est"]
    )
    print(
        f"  {'Transport share (overall)':<25s} "
        f"{t_total_share.median():>7.0%} {t_total_share.mean():>7.0%}"
    )

    if "dominant_type" not in sub.columns:
        return

    print("\n  By dominant housing type:")
    print(
        f"  {'Type':<14s} {'Build':>7s} {'Trans':>7s} {'Total':>7s} "
        f"{'T%':>5s} {'Car%':>5s} {'Commute':>8s} {'Active':>7s}"
    )
    print(f"  {'-' * 62}")
    for dtype in ["Flat", "Terraced", "Semi", "Detached"]:
        s = sub[sub["dominant_type"] == dtype]
        if len(s) == 0:
            continue
        b = s["building_kwh_per_hh"].median()
        t = s["transport_kwh_per_hh_est"].median()
        tot = s["total_kwh_per_hh"].median()
        tpct = (s["transport_kwh_per_hh_est"] / s["total_kwh_per_hh"]).median()
        t_total = s["transport_kwh_per_hh_total_est"].median()
        tot_total = s["total_kwh_per_hh_total_est"].median()
        tpct_total = (
            s["transport_kwh_per_hh_total_est"] / s["total_kwh_per_hh_total_est"]
        ).median()
        car = s["car_commute_share"].median()
        km = s["avg_commute_km"].median()
        act = s["active_share"].median()
        print(
            f"  {dtype:<14s} {b:>7.0f} {t:>7.0f} {tot:>7.0f} "
            f"{tpct:>4.0%} {car:>4.0%} {km:>7.1f}km {act:>5.0%}"
        )
        print(
            f"  {'':<14s} {'':>7s} {t_total:>7.0f} {tot_total:>7.0f} "
            f"{tpct_total:>4.0%} {'':>4s} {'':>8s} {'':>7s}  (overall scenario)"
        )


# ---------------------------------------------------------------------------
# 5. Regression: what predicts walkable destinations per kWh?
# ---------------------------------------------------------------------------


def _run_ols(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    label: str,
    cluster_col: str | None = None,
) -> sm.regression.linear_model.RegressionResultsWrapper | None:
    """
    Run OLS with HC1 robust SEs.

    Parameters
    ----------
    cluster_col : str or None
        If provided, also fits with cluster-robust SEs and stores
        the clustered result as ``model._clustered_fit``.
    """
    extra_cols = [cluster_col] if cluster_col and cluster_col not in [y_col] + x_cols else []
    cols = [y_col] + x_cols + extra_cols
    sub = df[cols].dropna()
    if len(sub) < len(x_cols) + 10:
        print(f"  {label}: insufficient data (N={len(sub)})")
        return None
    y = sub[y_col]
    X = sm.add_constant(sub[x_cols])
    result = sm.OLS(y, X).fit(cov_type="HC1")
    # Also fit with BUA-clustered SEs if requested
    if cluster_col and cluster_col in sub.columns:
        groups = sub[cluster_col]
        try:
            clustered = sm.OLS(y, X).fit(
                cov_type="cluster",
                cov_kwds={"groups": groups},
            )
            result._clustered_fit = clustered  # type: ignore[attr-defined]
        except Exception:
            pass
    return result


def _print_model(
    m: sm.regression.linear_model.RegressionResultsWrapper | None,
    label: str,
    prev_r2: float | None = None,
) -> float | None:
    """Print one model's summary line. Return its R²."""
    if m is None:
        print(f"  {label}: SKIPPED")
        return prev_r2
    delta = ""
    if prev_r2 is not None:
        delta = f"  ΔR² = {m.rsquared - prev_r2:+.4f}"
    print(
        f"  {label}:  R² = {m.rsquared:.4f}  "
        f"adj = {m.rsquared_adj:.4f}  N = {int(m.nobs):,}{delta}"
    )
    return m.rsquared


def _print_coefficients(
    m: sm.regression.linear_model.RegressionResultsWrapper,
) -> None:
    """Print coefficient table."""
    print(f"\n  {'Variable':<25s} {'β':>8s} {'SE':>8s} {'t':>8s} {'p':>10s}")
    print(f"  {'-' * 62}")
    for var in m.params.index:
        if var == "const":
            continue
        print(
            f"  {var:<25s} {m.params[var]:>8.4f} {m.bse[var]:>8.4f} "
            f"{m.tvalues[var]:>8.2f} {m.pvalues[var]:>8.1e} "
            f"{_sigstars(m.pvalues[var])}"
        )


def run_regressions(lsoa: pd.DataFrame) -> None:
    """
    Regression analysis with progressive control sets.

    Three framings:
    A. DV = log(building energy per hh) — thermal surface
    B. DV = accessibility — access surface
    C. DV = log(access per kWh) — composite (decomposition check)

    Control sets build progressively:
    M1: form only (pct_detached, pct_flat, pct_terraced; semi = reference)
    M2: + log density, household size
    M3: + deprivation, building age
    M4: + city fixed effects
    M5: + cars_per_hh (mediator — shown separately)
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    print(f"\n{'=' * 70}")
    print("REGRESSION ANALYSIS")
    print("=" * 70)

    # --- Prepare variables ---
    # Log density (right-skewed)
    lsoa["log_people_per_ha"] = np.log(lsoa["people_per_ha"].clip(lower=0.1))

    # Type shares: detached, flat, terraced (semi = reference)
    form = ["pct_detached", "pct_flat", "pct_terraced"]
    density = ["log_people_per_ha", "avg_hh_size"]
    controls = ["pct_not_deprived"]
    if "median_build_year" in lsoa.columns:
        controls.append("median_build_year")

    # IMD income domain if available
    imd_col = [c for c in lsoa.columns if "imd_income" in c.lower() and "score" in c.lower()]
    if imd_col:
        controls.append(imd_col[0])

    # City fixed effects
    city_dummies: list[str] = []
    if "city" in lsoa.columns and lsoa["city"].nunique() > 1:
        dummies = pd.get_dummies(lsoa["city"], prefix="city", drop_first=True)
        dummies = dummies.astype(int)
        city_dummies = list(dummies.columns)
        lsoa = pd.concat([lsoa, dummies], axis=1)

    substantive = form + density + controls
    print(f"\n  Substantive controls: {substantive}")
    print(f"  City FE: {len(city_dummies)} dummies")

    # --- Helper to print model with VIF ---
    def _print_full_model(
        m: sm.regression.linear_model.RegressionResultsWrapper | None,
        label: str,
        show_vif: bool = False,
    ) -> None:
        if m is None:
            print(f"  {label}: SKIPPED")
            return
        print(
            f"\n  {label}:  R²={m.rsquared:.4f}  "
            f"adj={m.rsquared_adj:.4f}  N={int(m.nobs):,}"
        )
        print(f"  {'Variable':<25s} {'β':>8s} {'SE':>8s} {'t':>8s} {'p':>10s}")
        print(f"  {'-' * 62}")
        for var in m.params.index:
            if var == "const" or var.startswith("city_"):
                continue
            print(
                f"  {var:<25s} {m.params[var]:>8.4f} "
                f"{m.bse[var]:>8.4f} "
                f"{m.tvalues[var]:>8.2f} "
                f"{m.pvalues[var]:>8.1e} "
                f"{_sigstars(m.pvalues[var])}"
            )
        if city_dummies:
            n_sig = sum(
                1 for v in city_dummies
                if v in m.pvalues and m.pvalues[v] < 0.05
            )
            print(f"  ({len(city_dummies)} city dummies, {n_sig} sig at p<0.05)")

        # BUA-clustered SE comparison
        clustered = getattr(m, "_clustered_fit", None)
        if clustered is not None:
            print(f"\n  BUA-clustered SEs (vs HC1):")
            print(
                f"  {'Variable':<25s} {'β':>8s} "
                f"{'SE(HC1)':>8s} {'SE(clust)':>10s} "
                f"{'t(clust)':>9s} {'p(clust)':>10s}"
            )
            print(f"  {'-' * 76}")
            for var in m.params.index:
                if var == "const" or var.startswith("city_"):
                    continue
                se_hc1 = m.bse[var]
                se_cl = clustered.bse[var]
                t_cl = clustered.tvalues[var]
                p_cl = clustered.pvalues[var]
                print(
                    f"  {var:<25s} {m.params[var]:>8.4f} "
                    f"{se_hc1:>8.4f} {se_cl:>10.4f} "
                    f"{t_cl:>9.2f} {p_cl:>8.1e} "
                    f"{_sigstars(p_cl)}"
                )

        if show_vif:
            # VIF for substantive variables only
            x_cols_sub = [v for v in m.params.index
                          if v != "const" and not v.startswith("city_")]
            cols = [m.model.exog_names.index(v) for v in x_cols_sub
                    if v in m.model.exog_names]
            if cols:
                print(f"\n  VIF:")
                for idx in cols:
                    vname = m.model.exog_names[idx]
                    try:
                        vif = variance_inflation_factor(m.model.exog, idx)
                        flag = " *** HIGH" if vif > 5 else ""
                        print(f"    {vname:<25s} {vif:>6.1f}{flag}")
                    except Exception:
                        pass

    # =================================================================
    # A. DV = log(building energy per hh) — what predicts thermal demand?
    # =================================================================
    print(f"\n  {'=' * 60}")
    print("  A. DV = log(building kWh/hh)")
    print(f"  {'=' * 60}")

    a1 = _run_ols(lsoa, "log_building_kwh_per_hh", form, "A1")
    _print_model(a1, "A1 Form only")
    a2 = _run_ols(lsoa, "log_building_kwh_per_hh", form + density, "A2")
    _print_model(a2, "A2 +Density+HH size")
    a3 = _run_ols(lsoa, "log_building_kwh_per_hh", form + density + controls, "A3")
    _print_model(a3, "A3 +Controls")
    a4 = None
    if city_dummies:
        a4 = _run_ols(
            lsoa, "log_building_kwh_per_hh",
            form + density + controls + city_dummies, "A4",
            cluster_col="city",
        )
        _print_model(a4, "A4 +City FE")
    best_a = a4 or a3 or a2 or a1
    _print_full_model(best_a, "A: Full model (building energy)", show_vif=True)

    # =================================================================
    # B. DV = accessibility — what predicts local access?
    # =================================================================
    print(f"\n  {'=' * 60}")
    print("  B. DV = accessibility (z-scored)")
    print(f"  {'=' * 60}")

    b1 = _run_ols(lsoa, "accessibility", form, "B1")
    _print_model(b1, "B1 Form only")
    b2 = _run_ols(lsoa, "accessibility", form + density, "B2")
    _print_model(b2, "B2 +Density+HH size")
    b3 = _run_ols(lsoa, "accessibility", form + density + controls, "B3")
    _print_model(b3, "B3 +Controls")
    b4 = None
    if city_dummies:
        b4 = _run_ols(
            lsoa, "accessibility",
            form + density + controls + city_dummies, "B4",
            cluster_col="city",
        )
        _print_model(b4, "B4 +City FE")
    best_b = b4 or b3 or b2 or b1
    _print_full_model(best_b, "B: Full model (accessibility)", show_vif=True)

    # =================================================================
    # C. DV = log(access/kWh) — composite (decomposition check)
    # =================================================================
    print(f"\n  {'=' * 60}")
    print("  C. DV = log(access per kWh) — composite ratio")
    print("  NOTE: This DV conflates energy and access channels.")
    print("  Models A and B above decompose the two channels separately.")
    print(f"  {'=' * 60}")

    c_vars = form + density + controls
    if city_dummies:
        c_vars = c_vars + city_dummies
    c1 = _run_ols(lsoa, "log_access_per_kwh", c_vars, "C1",
                  cluster_col="city")
    _print_full_model(c1, "C: Composite model (access/kWh)")

    # =================================================================
    # D. Mediator check: adding cars_per_hh
    # =================================================================
    print(f"\n  {'=' * 60}")
    print("  D. Mediator check: cars_per_hh")
    print("  Cars/hh is likely a mediator (morphology -> car ownership -> energy)")
    print("  not a confounder. Including it shows how much gradient it absorbs.")
    print(f"  {'=' * 60}")

    mediator_vars = form + density + controls + ["cars_per_hh"]
    if city_dummies:
        mediator_vars = mediator_vars + city_dummies
    d1 = _run_ols(lsoa, "log_building_kwh_per_hh", mediator_vars, "D1",
                  cluster_col="city")
    _print_full_model(d1, "D: Building energy + cars/hh (mediator)")


# ---------------------------------------------------------------------------
# 6. Deprivation control
# ---------------------------------------------------------------------------


def run_deprivation_control(lsoa: pd.DataFrame) -> None:
    """Check accessibility-per-kWh within each deprivation quintile."""
    print(f"\n{'=' * 70}")
    print("DEPRIVATION CONTROL")
    print("=" * 70)

    if "deprivation_quintile" not in lsoa.columns:
        print("  No deprivation quintiles (too few valid observations)")
        return

    form = ["pct_detached", "pct_flat"]

    quintiles = ["Q1 most", "Q2", "Q3", "Q4", "Q5 least"]
    print(
        f"\n  {'Quintile':<12s} {'N':>5s} {'kWh/hh':>8s} {'Acc/kWh':>8s} "
        f"{'det_β':>8s} {'p':>10s}"
    )
    print(f"  {'-' * 56}")

    n_sig = 0
    for q in quintiles:
        sub = lsoa[lsoa["deprivation_quintile"] == q].copy()
        if len(sub) < 20:
            continue
        kwh = sub["total_kwh_per_hh"].median()
        cond = sub["access_per_kwh"].median()

        m = _run_ols(sub, "log_access_per_kwh", form, f"dep-{q}")
        det_b = m.params.get("pct_detached", np.nan) if m else np.nan
        det_p = m.pvalues.get("pct_detached", np.nan) if m else np.nan
        if not np.isnan(det_p) and det_p < 0.05:
            n_sig += 1
        print(
            f"  {q:<12s} {len(sub):>5d} {kwh:>8.0f} {cond:>8.5f} "
            f"{det_b:>8.4f} {det_p:>8.1e} {_sigstars(det_p)}"
        )

    print(f"\n  pct_detached → access/kWh significant in {n_sig}/5 quintiles")
    if n_sig >= 3:
        print("  Effect holds across wealth levels")


# ---------------------------------------------------------------------------
# 7. Per-city
# ---------------------------------------------------------------------------


def run_per_city(lsoa: pd.DataFrame) -> None:
    """Accessibility comparison across cities."""
    print(f"\n{'=' * 70}")
    print("PER-CITY BREAKDOWN")
    print("=" * 70)

    if "city" not in lsoa.columns:
        return

    form = ["pct_detached", "pct_flat"]
    cities = sorted(lsoa["city"].unique())

    print(
        f"\n  {'City':<16s} {'N':>5s} {'kWh/hh':>8s} {'%det':>5s} "
        f"{'Acc/kWh':>8s} {'F/D':>6s} {'det_β':>8s} {'p':>10s}"
    )
    print(f"  {'-' * 72}")

    for city in cities:
        sub = lsoa[lsoa["city"] == city].copy()
        kwh = sub["total_kwh_per_hh"].median()
        pdet = sub["pct_detached"].median()
        cond = sub["access_per_kwh"].median()

        # Flat/Detached accessibility ratio within city
        ratio = np.nan
        flat = sub[sub["dominant_type"] == "Flat"]
        det = sub[sub["dominant_type"] == "Detached"]
        if len(flat) > 0 and len(det) > 0:
            c_f = flat["access_per_kwh"].median()
            c_d = det["access_per_kwh"].median()
            if c_d > 0:
                ratio = c_f / c_d

        m = _run_ols(sub, "log_access_per_kwh", form, city)
        det_b = m.params.get("pct_detached", np.nan) if m else np.nan
        det_p = m.pvalues.get("pct_detached", np.nan) if m else np.nan

        print(
            f"  {city:<16s} {len(sub):>5d} {kwh:>8.0f} {pdet:>4.0f}% "
            f"{cond:>8.5f} {ratio:>5.1f}x {det_b:>8.4f} "
            f"{det_p:>8.1e} {_sigstars(det_p)}"
        )


# ---------------------------------------------------------------------------
# 8. GVA within-city stratification
# ---------------------------------------------------------------------------


def run_gva_stratified(lsoa: pd.DataFrame) -> None:
    """
    GVA per household stratified by dwelling type within cities.

    The unstratified comparison is confounded by city size: flat-dominated
    OAs concentrate in London and other large cities where GVA is higher
    regardless of dwelling type. This computes within-city medians and a
    pooled within-city ratio.
    """
    print(f"\n{'=' * 70}")
    print("GVA PER HOUSEHOLD — WITHIN-CITY STRATIFICATION")
    print("=" * 70)

    if "gva_per_hh" not in lsoa.columns or "city" not in lsoa.columns:
        print("  Skipped: gva_per_hh or city column missing")
        return

    gva = lsoa[lsoa["gva_per_hh"].notna() & lsoa["dominant_type"].notna()].copy()
    print(f"  N = {len(gva):,} OAs with GVA data")

    types = ["Flat", "Terraced", "Semi", "Detached"]

    # Unstratified (for comparison)
    print("\n  UNSTRATIFIED (confounded by city composition):")
    print(f"    {'Type':<12s} {'GVA/hh':>10s} {'N':>6s}")
    for t in types:
        sub = gva[gva["dominant_type"] == t]
        if len(sub) > 0:
            print(f"    {t:<12s} £{sub['gva_per_hh'].median():>8,.0f} {len(sub):>6,d}")

    # Within-city: for each city, compute median GVA/hh by type
    print("\n  WITHIN-CITY (median GVA/hh by dominant type, per city):")
    print(
        f"    {'City':<16s} {'Flat':>10s} {'Terraced':>10s} "
        f"{'Semi':>10s} {'Detached':>10s} {'F/D':>6s}"
    )
    print(f"    {'-' * 64}")

    within_ratios = []
    for city in sorted(gva["city"].unique()):
        csub = gva[gva["city"] == city]
        vals = {}
        for t in types:
            tsub = csub[csub["dominant_type"] == t]
            vals[t] = tsub["gva_per_hh"].median() if len(tsub) >= 5 else np.nan

        line = f"    {city:<16s}"
        for t in types:
            v = vals[t]
            line += f" {'—':>10s}" if np.isnan(v) else f" £{v:>8,.0f}"

        if not np.isnan(vals["Flat"]) and not np.isnan(vals["Detached"]):
            ratio = vals["Flat"] / vals["Detached"]
            within_ratios.append(ratio)
            line += f" {ratio:>5.2f}x"
        print(line)

    if within_ratios:
        print("\n  Within-city Flat/Detached GVA ratio:")
        print(f"    Median across cities: {np.median(within_ratios):.2f}x")
        print(f"    Range: {min(within_ratios):.2f}x – {max(within_ratios):.2f}x")
        print(f"    Cities with both types: {len(within_ratios)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 8a. Spatial autocorrelation diagnostics (Moran's I)
# ---------------------------------------------------------------------------


def run_spatial_diagnostics(max_n: int = 50_000) -> None:
    """
    Compute Global Moran's I for key variables using Queen contiguity.

    Loads geometry from the GeoPackage (retained for this diagnostic only).
    Uses a random subsample if N > max_n for computational feasibility.

    Parameters
    ----------
    max_n : int
        Maximum OAs for spatial weights computation.
    """
    print(f"\n{'=' * 70}")
    print("SPATIAL AUTOCORRELATION DIAGNOSTICS (Moran's I)")
    print("=" * 70)

    try:
        import libpysal
        from esda.moran import Moran
    except ImportError:
        print("  esda/libpysal not available — skipping spatial diagnostics")
        return

    gdf = gpd.read_file(DATA_PATH)
    # Basic filters matching load_and_aggregate
    gdf = gdf[gdf["OA21CD"].notna()].copy()
    gdf["oa_total_mean_kwh"] = pd.to_numeric(
        gdf["oa_total_mean_kwh"], errors="coerce"
    )
    gdf = gdf[gdf["oa_total_mean_kwh"].notna() & (gdf["oa_total_mean_kwh"] > 0)]
    gdf = gdf[gdf["geometry"].notna() & ~gdf["geometry"].is_empty]

    print(f"  {len(gdf):,} OAs with valid geometry")

    if len(gdf) > max_n:
        print(f"  Subsampling to {max_n:,} for computational feasibility")
        gdf = gdf.sample(n=max_n, random_state=42).copy()

    print("  Building Queen contiguity weights...")
    try:
        w = libpysal.weights.Queen.from_dataframe(gdf)
    except Exception:
        print("  Queen weights failed — trying KNN(k=8)")
        w = libpysal.weights.KNN.from_dataframe(gdf, k=8)

    w.transform = "r"
    n_islands = sum(1 for v in w.cardinalities.values() if v == 0)
    print(f"  Weights: {w.n} OAs, {n_islands} islands, "
          f"mean neighbours={w.mean_neighbors:.1f}")

    # Remove islands
    if n_islands > 0:
        non_island = [i for i, v in w.cardinalities.items() if v > 0]
        gdf = gdf.iloc[non_island].copy()
        w = libpysal.weights.Queen.from_dataframe(gdf)
        w.transform = "r"
        print(f"  After removing islands: {w.n} OAs")

    variables = [
        ("oa_total_mean_kwh", "Building energy (kWh/hh)"),
        ("cc_density_800", "Network density (800m)"),
    ]

    # Add accessibility composite if columns exist
    fsa_cols = [c for c in gdf.columns if c.startswith("cc_fsa_") and c.endswith("_800_wt")]
    if fsa_cols:
        gdf["_fsa_sum"] = gdf[fsa_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        variables.append(("_fsa_sum", "FSA count (800m)"))

    print(
        f"\n  {'Variable':<30s} {'Moran I':>9s} {'E[I]':>9s} "
        f"{'z':>8s} {'p':>10s}"
    )
    print(f"  {'-' * 72}")

    for col, label in variables:
        if col not in gdf.columns:
            continue
        vals = pd.to_numeric(gdf[col], errors="coerce")
        valid = vals.notna()
        if valid.sum() < 100:
            continue
        y = vals[valid].values
        # Need aligned weights
        try:
            mi = Moran(y, w)
            print(
                f"  {label:<30s} {mi.I:>9.4f} {mi.EI:>9.4f} "
                f"{mi.z_sim:>8.2f} {mi.p_sim:>8.1e} "
                f"{_sigstars(mi.p_sim)}"
            )
        except Exception as e:
            print(f"  {label:<30s} ERROR: {e}")

    print(
        "\n  NOTE: Significant Moran's I indicates spatial autocorrelation."
        "\n  OLS standard errors are anti-conservative; effective N < observed N."
        "\n  Spatial lag/error models are recommended for formal inference."
    )


# ---------------------------------------------------------------------------
# 8b. Bootstrap confidence intervals on key ratios
# ---------------------------------------------------------------------------


def run_bootstrap_cis(
    lsoa: pd.DataFrame,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, dict]:
    """
    Bootstrap 95% BCa confidence intervals for Flat/Detached median ratios.

    Parameters
    ----------
    lsoa : pd.DataFrame
        OA data with dominant_type and energy/access columns.
    n_boot : int
        Number of bootstrap resamples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Mapping metric_name -> {ratio, ci_lo, ci_hi}.
    """
    print(f"\n{'=' * 70}")
    print(f"BOOTSTRAP CONFIDENCE INTERVALS ({n_boot:,} resamples)")
    print("=" * 70)

    rng = np.random.default_rng(seed)
    flat = lsoa[lsoa["dominant_type"] == "Flat"]
    det = lsoa[lsoa["dominant_type"] == "Detached"]

    if len(flat) < 30 or len(det) < 30:
        print("  Insufficient data for bootstrap")
        return {}

    metrics = [
        ("building_kwh_per_hh", "Building kWh/hh"),
        ("total_kwh_per_hh", "Total kWh/hh (commute)"),
        ("total_kwh_per_hh_total_est", "Total kWh/hh (overall)"),
        ("transport_kwh_per_hh_total_est", "Transport kWh/hh (overall)"),
        ("kwh_per_access", "kWh per access unit"),
        ("cars_per_hh", "Cars/hh"),
    ]

    results: dict[str, dict] = {}
    print(
        f"\n  {'Metric':<30s} {'Ratio':>7s} {'95% CI':>16s} "
        f"{'Flat med':>10s} {'Det med':>10s}"
    )
    print(f"  {'-' * 78}")

    for col, label in metrics:
        if col not in flat.columns:
            continue
        f_vals = flat[col].dropna().values
        d_vals = det[col].dropna().values
        if len(f_vals) < 30 or len(d_vals) < 30:
            continue

        obs_ratio = float(np.median(f_vals) / np.median(d_vals))

        boot_ratios = np.empty(n_boot)
        for i in range(n_boot):
            f_sample = rng.choice(f_vals, size=len(f_vals), replace=True)
            d_sample = rng.choice(d_vals, size=len(d_vals), replace=True)
            d_med = np.median(d_sample)
            boot_ratios[i] = np.median(f_sample) / d_med if d_med != 0 else np.nan

        boot_ratios = boot_ratios[~np.isnan(boot_ratios)]
        ci_lo = float(np.percentile(boot_ratios, 2.5))
        ci_hi = float(np.percentile(boot_ratios, 97.5))

        results[col] = {
            "ratio": round(obs_ratio, 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
        }

        print(
            f"  {label:<30s} {obs_ratio:>7.3f} [{ci_lo:.3f}, {ci_hi:.3f}] "
            f"{np.median(f_vals):>10,.0f} {np.median(d_vals):>10,.0f}"
        )

    return results


# ---------------------------------------------------------------------------
# 8c. Plurality share sensitivity
# ---------------------------------------------------------------------------


def run_plurality_sensitivity(lsoa: pd.DataFrame) -> None:
    """
    Test sensitivity of key ratios to stricter dominant-type thresholds.

    The default classification uses plurality share (no minimum). This
    tests thresholds at 40%, 50%, and 60% to assess how much the gradient
    depends on mixed OAs near the classification boundary.

    Parameters
    ----------
    lsoa : pd.DataFrame
        OA data with pct_detached, pct_semi, pct_terraced, pct_flat columns.
    """
    print(f"\n{'=' * 70}")
    print("SENSITIVITY: PLURALITY SHARE THRESHOLDS")
    print("=" * 70)

    _type_map = {
        "pct_flat": "Flat",
        "pct_terraced": "Terraced",
        "pct_semi": "Semi",
        "pct_detached": "Detached",
    }
    type_pcts = lsoa[list(_type_map.keys())].fillna(0)
    max_share = type_pcts.max(axis=1)
    dominant = type_pcts.idxmax(axis=1).map(_type_map)

    thresholds = [0, 40, 50, 60]
    print(
        f"\n  {'Threshold':<12s} {'N total':>8s} {'N Flat':>8s} "
        f"{'N Det':>8s} {'Bldg ratio':>11s} {'Total ratio':>12s} "
        f"{'kWh/Acc ratio':>14s}"
    )
    print(f"  {'-' * 80}")

    for thresh in thresholds:
        mask = max_share >= thresh
        sub = lsoa[mask].copy()
        sub["dominant_type_strict"] = pd.Categorical(
            dominant[mask],
            categories=["Flat", "Terraced", "Semi", "Detached"],
            ordered=True,
        )

        flat = sub[sub["dominant_type_strict"] == "Flat"]
        det = sub[sub["dominant_type_strict"] == "Detached"]

        if len(flat) < 10 or len(det) < 10:
            print(f"  {thresh}%{' (plurality)' if thresh == 0 else '':<12s} "
                  f"insufficient data")
            continue

        r_bldg = flat["building_kwh_per_hh"].median() / det[
            "building_kwh_per_hh"
        ].median()
        r_total = flat["total_kwh_per_hh_total_est"].median() / det[
            "total_kwh_per_hh_total_est"
        ].median()
        r_acc = np.nan
        if "kwh_per_access" in flat.columns:
            f_acc = flat["kwh_per_access"].median()
            d_acc = det["kwh_per_access"].median()
            if d_acc > 0:
                r_acc = f_acc / d_acc

        label = f"{thresh}%" if thresh > 0 else "plurality"
        print(
            f"  {label:<12s} {len(sub):>8,d} {len(flat):>8,d} "
            f"{len(det):>8,d} {r_bldg:>11.3f} {r_total:>12.3f} "
            f"{r_acc:>14.3f}"
        )


# ---------------------------------------------------------------------------
# 8d. NTS scalar sensitivity
# ---------------------------------------------------------------------------


def run_nts_sensitivity(lsoa: pd.DataFrame) -> None:
    """
    Test sensitivity of total energy gradient to the NTS distance scalar.

    The default scalar is 6.04x (NTS 2024 total/commute distance ratio).
    This tests the gradient at 1x (commute only), 4x, 6x, 8x, and 10x.

    Parameters
    ----------
    lsoa : pd.DataFrame
        OA data with transport_kwh_per_hh_est and building_kwh_per_hh.
    """
    print(f"\n{'=' * 70}")
    print("SENSITIVITY: NTS TOTAL-TO-COMMUTE DISTANCE SCALAR")
    print("=" * 70)

    flat = lsoa[lsoa["dominant_type"] == "Flat"]
    det = lsoa[lsoa["dominant_type"] == "Detached"]

    if len(flat) < 10 or len(det) < 10:
        print("  Insufficient data")
        return

    scalars = [1.0, 3.0, 4.0, 5.0, 6.04, 7.0, 8.0, 10.0]
    print(
        f"\n  {'Scalar':>8s} {'Flat total':>12s} {'Det total':>12s} "
        f"{'Ratio':>8s} {'Trans share (Det)':>18s}"
    )
    print(f"  {'-' * 64}")

    for s in scalars:
        f_bldg = flat["building_kwh_per_hh"].median()
        f_trans = flat["transport_kwh_per_hh_est"].median() * s
        d_bldg = det["building_kwh_per_hh"].median()
        d_trans = det["transport_kwh_per_hh_est"].median() * s

        f_total = f_bldg + f_trans
        d_total = d_bldg + d_trans
        ratio = f_total / d_total if d_total > 0 else np.nan
        d_share = d_trans / d_total if d_total > 0 else np.nan

        marker = " *" if abs(s - _NTS_TOTAL_TO_COMMUTE_DISTANCE_FACTOR) < 0.1 else ""
        print(
            f"  {s:>7.1f}x {f_total:>12,.0f} {d_total:>12,.0f} "
            f"{ratio:>8.3f} {d_share:>17.1%}{marker}"
        )

    print("\n  * = baseline NTS 2024 scalar")


# ---------------------------------------------------------------------------
# 8e. Edge effect diagnostic
# ---------------------------------------------------------------------------


def run_edge_diagnostic(lsoa: pd.DataFrame) -> None:
    """
    Compare accessibility metrics for OAs near BUA boundaries vs interior.

    Uses cc_density_800 as a proxy: OAs with very low network density
    relative to their type are likely at the BUA edge where the road
    network is truncated.

    Parameters
    ----------
    lsoa : pd.DataFrame
        OA data with cc_density_800 and dominant_type.
    """
    print(f"\n{'=' * 70}")
    print("EDGE EFFECT DIAGNOSTIC")
    print("=" * 70)

    if "cc_density_800" not in lsoa.columns:
        print("  cc_density_800 not available")
        return

    # Within each dominant type, flag bottom 10% of cc_density as "edge"
    lsoa = lsoa.copy()
    lsoa["_edge_flag"] = False
    for dtype in ["Flat", "Terraced", "Semi", "Detached"]:
        mask = lsoa["dominant_type"] == dtype
        if mask.sum() < 20:
            continue
        threshold = lsoa.loc[mask, "cc_density_800"].quantile(0.10)
        lsoa.loc[mask & (lsoa["cc_density_800"] <= threshold), "_edge_flag"] = True

    n_edge = lsoa["_edge_flag"].sum()
    n_interior = (~lsoa["_edge_flag"]).sum()
    print(f"\n  Edge OAs (bottom 10% cc_density within type): {n_edge:,}")
    print(f"  Interior OAs: {n_interior:,}")

    print(
        f"\n  {'Metric':<30s} {'Interior':>10s} {'Edge':>10s} {'Diff %':>8s}"
    )
    print(f"  {'-' * 62}")

    for col, label in [
        ("building_kwh_per_hh", "Building kWh/hh"),
        ("total_kwh_per_hh_total_est", "Total kWh/hh (overall)"),
        ("accessibility", "Accessibility (z-score)"),
        ("fsa_count", "FSA count (800m)"),
        ("cc_density_800", "Network density (800m)"),
    ]:
        if col not in lsoa.columns:
            continue
        interior = lsoa.loc[~lsoa["_edge_flag"], col].median()
        edge = lsoa.loc[lsoa["_edge_flag"], col].median()
        pct = ((edge - interior) / abs(interior) * 100) if interior != 0 else np.nan
        print(f"  {label:<30s} {interior:>10.1f} {edge:>10.1f} {pct:>+7.1f}%")

    # Key question: does excluding edge OAs change the Flat/Det gradient?
    interior_only = lsoa[~lsoa["_edge_flag"]]
    flat_i = interior_only[interior_only["dominant_type"] == "Flat"]
    det_i = interior_only[interior_only["dominant_type"] == "Detached"]

    if len(flat_i) > 10 and len(det_i) > 10:
        r_all = lsoa[lsoa["dominant_type"] == "Flat"][
            "building_kwh_per_hh"
        ].median() / lsoa[lsoa["dominant_type"] == "Detached"][
            "building_kwh_per_hh"
        ].median()
        r_interior = flat_i["building_kwh_per_hh"].median() / det_i[
            "building_kwh_per_hh"
        ].median()
        print(
            f"\n  Building kWh/hh Flat/Det ratio:"
            f"\n    All OAs:      {r_all:.3f}"
            f"\n    Interior only: {r_interior:.3f}"
            f"\n    Change:       {(r_interior - r_all) / r_all:+.1%}"
        )


# ---------------------------------------------------------------------------
# 9. Collect results
# ---------------------------------------------------------------------------


def _collect_results(lsoa: pd.DataFrame) -> dict:
    """
    Collect all key metrics into a JSON-serialisable dict.

    This is the single source of truth for numbers cited in the paper.
    """
    types = ["Flat", "Terraced", "Semi", "Detached"]
    results: dict = {"n_lsoas": len(lsoa)}
    if "city" in lsoa.columns:
        results["n_cities"] = int(lsoa["city"].nunique())

    # --- Per dwelling type medians ---
    by_type: dict = {}
    for t in types:
        sub = lsoa[lsoa["dominant_type"] == t]
        if len(sub) == 0:
            continue
        entry: dict = {"n": len(sub)}
        for col, key in [
            ("building_kwh_per_hh", "building_kwh_per_hh"),
            ("kwh_per_person", "kwh_per_person"),
            ("transport_kwh_per_hh_est", "transport_kwh_per_hh_est"),
            ("transport_kwh_per_hh_total_est", "transport_kwh_per_hh_total_est"),
            ("total_kwh_per_hh", "total_kwh_per_hh"),
            ("total_kwh_per_hh_total_est", "total_kwh_per_hh_total_est"),
            ("total_kwh_per_person", "total_kwh_per_person"),
            ("cars_per_hh", "cars_per_hh"),
            ("car_commute_share", "car_commute_share"),
            ("avg_hh_size", "avg_hh_size"),
            ("people_per_ha", "people_per_ha"),
            ("street_frontage", "street_frontage"),
            ("fsa_count", "fsa_count"),
            ("accessibility", "accessibility"),
            ("access_per_kwh", "access_per_kwh"),
            ("kwh_per_access", "kwh_per_access"),
        ]:
            if col in sub.columns:
                v = sub[col].median()
                entry[key] = round(float(v), 4) if not np.isnan(v) else None
        # Unweighted accessibility counts
        for col in _CC_ACCESSIBILITY + _CC_CENTRALITY:
            if col in sub.columns:
                v = sub[col].median()
                short = col.replace("cc_", "")
                entry[short] = round(float(v), 2) if not np.isnan(v) else None
        by_type[t] = entry
    results["by_type"] = by_type

    # --- Flat/Detached ratios ---
    flat = lsoa[lsoa["dominant_type"] == "Flat"]
    det = lsoa[lsoa["dominant_type"] == "Detached"]
    if len(flat) > 0 and len(det) > 0:
        ratios: dict = {}
        for col, key in [
            ("building_kwh_per_hh", "building_kwh_per_hh"),
            ("total_kwh_per_hh", "total_kwh_per_hh"),
            ("transport_kwh_per_hh_est", "transport_kwh_per_hh_est"),
            ("total_kwh_per_hh_total_est", "total_kwh_per_hh_total_est"),
            ("transport_kwh_per_hh_total_est", "transport_kwh_per_hh_total_est"),
            ("access_per_kwh", "access_per_kwh"),
            ("kwh_per_access", "kwh_per_access"),
            ("cars_per_hh", "cars_per_hh"),
        ]:
            if col in flat.columns:
                f_v = flat[col].median()
                d_v = det[col].median()
                if d_v != 0 and not np.isnan(f_v) and not np.isnan(d_v):
                    ratios[key] = round(float(f_v / d_v), 4)
        results["flat_detached_ratios"] = ratios

    # --- GVA within-city ---
    if "gva_per_hh" in lsoa.columns and "city" in lsoa.columns:
        gva = lsoa[lsoa["gva_per_hh"].notna() & lsoa["dominant_type"].notna()]
        gva_results: dict = {"n_lsoas_with_gva": len(gva)}

        # Unstratified
        unstrat: dict = {}
        for t in types:
            sub = gva[gva["dominant_type"] == t]
            if len(sub) > 0:
                unstrat[t] = {
                    "median_gva_per_hh": round(float(sub["gva_per_hh"].median()), 0),
                    "n": len(sub),
                }
        gva_results["unstratified"] = unstrat

        # Within-city
        within_city: dict = {}
        within_ratios: list[float] = []
        for city in sorted(gva["city"].unique()):
            csub = gva[gva["city"] == city]
            city_entry: dict = {}
            for t in types:
                tsub = csub[csub["dominant_type"] == t]
                if len(tsub) >= 5:
                    city_entry[t] = {
                        "median_gva_per_hh": round(
                            float(tsub["gva_per_hh"].median()), 0
                        ),
                        "n": len(tsub),
                    }
            if "Flat" in city_entry and "Detached" in city_entry:
                ratio = (
                    city_entry["Flat"]["median_gva_per_hh"]
                    / city_entry["Detached"]["median_gva_per_hh"]
                )
                city_entry["flat_detached_ratio"] = round(ratio, 4)
                within_ratios.append(ratio)
            within_city[city] = city_entry
        gva_results["within_city"] = within_city
        if within_ratios:
            gva_results["within_city_summary"] = {
                "median_ratio": round(float(np.median(within_ratios)), 4),
                "min_ratio": round(float(min(within_ratios)), 4),
                "max_ratio": round(float(max(within_ratios)), 4),
                "n_cities": len(within_ratios),
            }
        results["gva"] = gva_results

    return results


def main(cities: list[str] | None = None) -> None:
    """Run the OA accessibility analysis."""
    print("=" * 70)
    print("OA ACCESSIBILITY ANALYSIS: THREE ENERGY SURFACES")
    print("Per unit energy spent, how much city do you get?")
    print("=" * 70)

    lsoa = load_and_aggregate(cities)
    lsoa = build_accessibility(lsoa)
    lsoa = compute_access_per_kwh(lsoa)
    print_systematic_summary(lsoa)
    print_energy_decomposition(lsoa)
    run_regressions(lsoa)
    run_deprivation_control(lsoa)
    run_per_city(lsoa)
    run_gva_stratified(lsoa)

    # --- Robustness and sensitivity ---
    bootstrap_results = run_bootstrap_cis(lsoa)
    run_plurality_sensitivity(lsoa)
    run_nts_sensitivity(lsoa)
    run_edge_diagnostic(lsoa)
    run_spatial_diagnostics()

    # Save machine-readable results
    results = _collect_results(lsoa)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {RESULTS_PATH}")

    print(f"\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    _cities = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(cities=_cities or None)
