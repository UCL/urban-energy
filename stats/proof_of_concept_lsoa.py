"""
LSOA-Level Accessibility Analysis: Three Energy Surfaces.

A building's morphological type is not just a thermal envelope — it is a
commitment to a pattern of living. This script decomposes the urban energy
landscape into three surfaces:

    1. THE THERMAL SURFACE (building envelope)
       Heat through walls and roofs. Metered building energy increases
       modestly from compact to sprawl (~13k → ~17k kWh/hh). The S/V
       advantage of compact form is real but partly absorbed by smaller
       household sizes: per capita, flats are higher than terraced/semi.

    2. THE MOBILITY SURFACE (transport cost)
       How far you drive. Transport energy estimated from Census-reported
       commute distances (ts058) and car mode share (ts061), scaled to
       total car travel via NTS commute-to-total ratio (22%). Transport
       energy roughly doubles from compact to sprawl (~2.3k → ~4.5k
       kWh/hh). Combined with building energy, sprawl costs ~1.4x more.

    3. THE ACCESSIBILITY SURFACE (the return)
       What you can reach on foot without spending energy. Two metrics:
       (a) reachable street frontage (network density within 800m), and
       (b) FSA establishment count (gravity-weighted, 800m). Compact
       form delivers dramatically more walkable city per kWh.

The primary form variable is Census accommodation type (ts044), which
provides domestic-only dwelling counts by type (detached, semi-detached,
terraced, flat) with complete coverage. Stratification uses dominant
housing type per LSOA; regressions use pct_detached and pct_flat as
continuous predictors capturing the sprawl–compact spectrum.

LiDAR-derived S/V ratios are retained as a robustness check. Because
building morphology is aggregated from all structures (including
non-domestic), S/V is not used as a primary predictor. Where available,
it validates that the ts044 type gradient aligns with physical form.

All variables aggregate to LSOA (~1,500 people), where metered energy is
native and building counts are large enough for stable averages.

Usage:
    uv run python stats/proof_of_concept_lsoa.py
    uv run python stats/proof_of_concept_lsoa.py manchester york
"""

import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm

from urban_energy.paths import TEMP_DIR

DATA_PATH = TEMP_DIR / "processing" / "combined" / "lsoa_integrated.gpkg"

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

# Commute distance bands — midpoint km
_COMMUTE_BANDS: dict[str, float] = {
    "ts058_Distance travelled to work: Less than 2km": 1.0,
    "ts058_Distance travelled to work: 2km to less than 5km": 3.5,
    "ts058_Distance travelled to work: 5km to less than 10km": 7.5,
    "ts058_Distance travelled to work: 10km to less than 20km": 15.0,
    "ts058_Distance travelled to work: 20km to less than 30km": 25.0,
    "ts058_Distance travelled to work: 30km to less than 40km": 35.0,
    "ts058_Distance travelled to work: 40km to less than 60km": 50.0,
    "ts058_Distance travelled to work: 60km and over": 80.0,
    "ts058_Distance travelled to work: Works mainly from home": 0.0,
}
_COMMUTE_TOTAL = (
    "ts058_Distance travelled to work: "
    "Total: All usual residents aged 16 years and over "
    "in employment the week before the census"
)
_TS061_CAR = "ts061_Method of travel to workplace: Driving a car or van"
_TS061_TOTAL = (
    "ts061_Method of travel to workplace: "
    "Total: All usual residents aged 16 years and over "
    "in employment the week before the census"
)
_TS061_WALK = "ts061_Method of travel to workplace: On foot"
_TS061_CYCLE = "ts061_Method of travel to workplace: Bicycle"
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

# Metered energy — native LSOA
_LSOA_ENERGY = ["lsoa_total_mean_kwh"]

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
    _TS061_CAR,
    _TS061_TOTAL,
    _TS061_WALK,
    _TS061_CYCLE,
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
    *_COMMUTE_BANDS.keys(),
]

# Every column the script requires — missing any of these is a hard error
_REQUIRED = (
    ["LSOA21CD"] + _LSOA_ENERGY + _CC_ACCESSIBILITY + _CC_CENTRALITY + _CENSUS_REQUIRED
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
# 1. Load and aggregate to LSOA
# ---------------------------------------------------------------------------


def load_and_aggregate(cities: list[str] | None = None) -> pd.DataFrame:
    """
    Load pre-aggregated LSOA data from pipeline_lsoa.py output.

    The pipeline has already performed:
    - Census deduplication (OA-level first, then sum to LSOA)
    - Building physics aggregation (sum/mean per LSOA)
    - Cityseer metric averaging across UPRNs
    - EPC coverage and building era derivation
    - Metered energy join
    - Aggregate S/V ratio computation

    This function reads the LSOA GeoPackage and computes analytical
    derived variables (transport energy, deprivation, housing type, etc.).

    Parameters
    ----------
    cities : list[str] or None
        Filter to specific cities, or None for all.

    Returns
    -------
    pd.DataFrame
        LSOA-level dataset with derived analytical variables.

    Raises
    ------
    ValueError
        If required columns are missing from the dataset.
    """
    print("=" * 70)
    print("LOADING LSOA DATA")
    print("=" * 70)

    # Validate columns
    _probe = gpd.read_file(DATA_PATH, rows=1)
    available = set(_probe.columns)
    del _probe
    _validate_columns(available)

    lsoa = gpd.read_file(DATA_PATH)
    print(f"  Loaded {len(lsoa):,} LSOAs ({len(lsoa.columns)} columns)")

    if cities and "city" in lsoa.columns:
        lsoa = lsoa[lsoa["city"].isin(cities)].copy()
        print(f"  Filtered to {cities}: {len(lsoa):,}")

    lsoa = lsoa[lsoa["LSOA21CD"].notna()].copy()

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
    if "lsoa_sv" not in lsoa.columns and "envelope_area_m2" in lsoa.columns:
        lsoa["lsoa_sv"] = lsoa["envelope_area_m2"] / lsoa["volume_m3"].replace(
            0, np.nan
        )
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

    # Building energy (metered, native LSOA)
    lsoa["building_kwh_per_hh"] = pd.to_numeric(
        lsoa["lsoa_total_mean_kwh"], errors="coerce"
    )

    # --- Transport energy from Census ts058 + ts061 ---
    # Step 1: Commute-km by car from Census-reported distances and mode shares.
    #   ts058 gives distance-band counts → weighted average commute km per person.
    #   ts061 gives car-commute count → total car-commute-km per LSOA.
    # Step 2: Scale commute to total car travel.
    #   NTS 2019 Table NTS0409: commuting accounts for ~22% of total car-km.
    #   Scaling factor = 1/0.22 ≈ 4.5 (covers shopping, school, leisure, etc.)
    # Step 3: Convert km to kWh.
    #   0.73 kWh/km (petrol avg: ~8 L/100km × 9.1 kWh/L).
    kwh_per_km_car = 0.73
    commute_share_of_total_car_km = 0.22  # NTS 2019 Table NTS0409

    # ts058: total commuters and weighted commute distance
    total_commuters = pd.to_numeric(lsoa[_COMMUTE_TOTAL], errors="coerce")
    weighted_km = sum(
        km * pd.to_numeric(lsoa[col], errors="coerce")
        for col, km in _COMMUTE_BANDS.items()
    )
    lsoa["avg_commute_km"] = weighted_km / total_commuters.replace(0, np.nan)

    # ts061: car mode share and car-commuter count
    car_commuters = pd.to_numeric(lsoa[_TS061_CAR], errors="coerce")
    total_w = pd.to_numeric(lsoa[_TS061_TOTAL], errors="coerce")
    lsoa["car_commute_share"] = car_commuters / total_w.replace(0, np.nan)
    lsoa["walk_share"] = pd.to_numeric(
        lsoa[_TS061_WALK], errors="coerce"
    ) / total_w.replace(0, np.nan)
    lsoa["cycle_share"] = pd.to_numeric(
        lsoa[_TS061_CYCLE], errors="coerce"
    ) / total_w.replace(0, np.nan)
    lsoa["active_share"] = lsoa["walk_share"].fillna(0) + lsoa["cycle_share"].fillna(0)

    # Car commute-km: avg_commute_km × car_commuters × 2 (return) × 220 workdays
    # ts058 distances are one-way, Census-week basis. Annualise with 220 workdays.
    car_commute_km_annual = (
        lsoa["avg_commute_km"]
        * car_commuters
        * 2  # return trip
        * 220  # workdays per year
    )
    # Scale commute to total car travel (NTS: commuting ≈ 22% of total car-km)
    total_car_km_annual = car_commute_km_annual / commute_share_of_total_car_km
    # Convert to kWh per household
    total_hh_for_transport = pd.to_numeric(lsoa[_TS045_TOTAL], errors="coerce").replace(
        0, np.nan
    )
    lsoa["transport_kwh_per_hh"] = (
        total_car_km_annual * kwh_per_km_car
    ) / total_hh_for_transport

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
        "transport_kwh_per_hh"
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
    # Assign each LSOA the accommodation type with the highest percentage.
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

    # --- Filter ---
    valid = (
        (lsoa["total_people"] > 50)
        & (lsoa["n_uprns"] >= 20)
        & lsoa["building_kwh_per_hh"].notna()
        & (lsoa["building_kwh_per_hh"] > 0)
    )
    lsoa = lsoa[valid].copy()

    n_cities = lsoa["city"].nunique() if "city" in lsoa.columns else 1
    print(f"\n  {len(lsoa):,} LSOAs across {n_cities} cities")
    print(f"  UPRNs/LSOA: median={lsoa['n_uprns'].median():.0f}")
    if "lsoa_sv" in lsoa.columns:
        sv = lsoa["lsoa_sv"].dropna()
        print(
            f"\n  LiDAR S/V (robustness): median={sv.median():.3f}  "
            f"(mean={sv.mean():.3f}, N={len(sv):,})"
        )
    if "height_mean" in lsoa.columns:
        print(f"    Height: median={lsoa['height_mean'].dropna().median():.1f}m")
    print("\n  Cost (kWh/household):")
    print(f"    Building: median={lsoa['building_kwh_per_hh'].median():.0f}")
    t = lsoa["transport_kwh_per_hh"].dropna()
    print(
        f"    Transport: median={t.median():,.0f}  "
        f"(cars/hh={lsoa['cars_per_hh'].median():.2f})"
    )
    print(f"    Total: median={lsoa['total_kwh_per_hh'].median():,.0f}")
    if "median_build_year" in lsoa.columns:
        yr = lsoa["median_build_year"].dropna()
        print(f"\n  Stock: median build year={yr.median():.0f}")
    if "building_era" in lsoa.columns:
        era_counts = lsoa["building_era"].value_counts().sort_index()
        for era, n in era_counts.items():
            print(f"    {era}: {n:,} LSOAs")
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
            print(f"    {dtype}: {n:,} LSOAs")

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
        LSOA-level data with _CC_ACCESSIBILITY and _CC_CENTRALITY columns.

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
        LSOA data with accessibility and energy.

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
        ("FORM (LiDAR)", "S/V ratio", "lsoa_sv", ".3f"),
        ("FORM (LiDAR)", "Height (m)", "height_mean", ".1f"),
        ("STOCK", "Build year", "median_build_year", ".0f"),
        ("DENSITY", "People/ha", "people_per_ha", ".0f"),
        ("DENSITY", "People/hh", "avg_hh_size", ".2f"),
        ("1:THERMAL", "kWh/hh (bldg)", "building_kwh_per_hh", ",.0f"),
        ("1:THERMAL", "kWh/person (bldg)", "kwh_per_person", ",.0f"),
        ("2:MOBILITY", "Cars/hh", "cars_per_hh", ".2f"),
        ("2:MOBILITY", "kWh/hh (trans)", "transport_kwh_per_hh", ",.0f"),
        ("2:MOBILITY", "kWh/hh (total)", "total_kwh_per_hh", ",.0f"),
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
    line = f"    {'N LSOAs':<20s}"
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
    r_trans = flat["transport_kwh_per_hh"].median()
    r_trans /= det["transport_kwh_per_hh"].median()
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

    valid = lsoa["transport_kwh_per_hh"].notna()
    sub = lsoa[valid]

    print(f"\n  {'Metric':<25s} {'Median':>8s} {'Mean':>8s}")
    print(f"  {'-' * 44}")
    for label, col in [
        ("Building kWh/hh", "building_kwh_per_hh"),
        ("Transport kWh/hh", "transport_kwh_per_hh"),
        ("Total kWh/hh", "total_kwh_per_hh"),
    ]:
        s = sub[col]
        print(f"  {label:<25s} {s.median():>8.0f} {s.mean():>8.0f}")
    t_share = sub["transport_kwh_per_hh"] / sub["total_kwh_per_hh"]
    print(f"  {'Transport share':<25s} {t_share.median():>7.0%} {t_share.mean():>7.0%}")

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
        t = s["transport_kwh_per_hh"].median()
        tot = s["total_kwh_per_hh"].median()
        tpct = (s["transport_kwh_per_hh"] / s["total_kwh_per_hh"]).median()
        car = s["car_commute_share"].median()
        km = s["avg_commute_km"].median()
        act = s["active_share"].median()
        print(
            f"  {dtype:<14s} {b:>7.0f} {t:>7.0f} {tot:>7.0f} "
            f"{tpct:>4.0%} {car:>4.0%} {km:>7.1f}km {act:>5.0%}"
        )


# ---------------------------------------------------------------------------
# 5. Regression: what predicts walkable destinations per kWh?
# ---------------------------------------------------------------------------


def _run_ols(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    label: str,
) -> sm.regression.linear_model.RegressionResultsWrapper | None:
    """Run OLS with HC3 robust SEs."""
    cols = [y_col] + x_cols
    sub = df[cols].dropna()
    if len(sub) < len(x_cols) + 10:
        print(f"  {label}: insufficient data (N={len(sub)})")
        return None
    y = sub[y_col]
    X = sm.add_constant(sub[x_cols])
    return sm.OLS(y, X).fit(cov_type="HC3")


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
    Two complementary regression framings.

    A. DV = access per kWh (return/cost)
       What neighbourhood form delivers more city per kWh?

    B. DV = accessibility (what you get)
       Controlling for energy spend, does form predict access?
    """
    print(f"\n{'=' * 70}")
    print("REGRESSION: WHAT DRIVES ACCESSIBILITY?")
    print("=" * 70)

    # ts044 dwelling type shares: domestic-only, complete coverage
    form = ["pct_detached", "pct_flat"]
    occupation = ["people_per_ha"]
    controls = ["pct_not_deprived"]

    # City fixed effects — absorb city-level confounds (climate, stock age, etc.)
    city_dummies: list[str] = []
    if "city" in lsoa.columns and lsoa["city"].nunique() > 1:
        dummies = pd.get_dummies(lsoa["city"], prefix="city", drop_first=True)
        # Ensure boolean columns become numeric
        dummies = dummies.astype(int)
        city_dummies = list(dummies.columns)
        lsoa = pd.concat([lsoa, dummies], axis=1)

    # --- A: DV = log(walkable destinations per kWh) ---
    print("\n  --- A. What predicts walkable destinations per kWh? ---")
    print("  DV = log(walkable destinations / kWh per household)\n")

    a1 = _run_ols(lsoa, "log_access_per_kwh", form, "A1 Form")
    r2 = _print_model(a1, "A1 Form")

    a2 = _run_ols(lsoa, "log_access_per_kwh", form + occupation, "A2 +Occupation")
    r2 = _print_model(a2, "A2 +Occupation", r2)

    a3 = _run_ols(
        lsoa,
        "log_access_per_kwh",
        form + occupation + controls,
        "A3 +Deprivation",
    )
    r2 = _print_model(a3, "A3 +Deprivation", r2)

    if city_dummies:
        a4 = _run_ols(
            lsoa,
            "log_access_per_kwh",
            form + occupation + controls + city_dummies,
            "A4 +City FE",
        )
        r2 = _print_model(a4, "A4 +City FE", r2)
        best_a = a4 or a3 or a2 or a1
    else:
        best_a = a3 or a2 or a1

    if best_a:
        # Print only substantive coefficients (skip city dummies)
        print(f"\n  {'Variable':<25s} {'β':>8s} {'SE':>8s} {'t':>8s} {'p':>10s}")
        print(f"  {'-' * 62}")
        for var in best_a.params.index:
            if var == "const" or var.startswith("city_"):
                continue
            print(
                f"  {var:<25s} {best_a.params[var]:>8.4f} "
                f"{best_a.bse[var]:>8.4f} "
                f"{best_a.tvalues[var]:>8.2f} "
                f"{best_a.pvalues[var]:>8.1e} "
                f"{_sigstars(best_a.pvalues[var])}"
            )
        if city_dummies:
            n_city_sig = sum(
                1
                for v in city_dummies
                if v in best_a.pvalues and best_a.pvalues[v] < 0.05
            )
            print(
                f"  ({len(city_dummies)} city dummies, "
                f"{n_city_sig} significant at p<0.05)"
            )

    # --- B: DV = accessibility, energy as predictor ---
    print("\n\n  --- B. For a given energy spend, does form deliver more access? ---")
    print("  DV = accessibility (street frontage + FSA z-scored)\n")

    b1 = _run_ols(lsoa, "accessibility", ["log_total_kwh_per_hh"], "B1 Energy")
    r2 = _print_model(b1, "B1 Energy alone")

    b2 = _run_ols(
        lsoa,
        "accessibility",
        ["log_total_kwh_per_hh"] + form,
        "B2 +Form",
    )
    r2 = _print_model(b2, "B2 +Form", r2)

    b3 = _run_ols(
        lsoa,
        "accessibility",
        ["log_total_kwh_per_hh"] + form + occupation + controls,
        "B3 +Occupation+Deprivation",
    )
    r2 = _print_model(b3, "B3 +Occ+Dep", r2)

    if city_dummies:
        b4 = _run_ols(
            lsoa,
            "accessibility",
            ["log_total_kwh_per_hh"] + form + occupation + controls + city_dummies,
            "B4 +City FE",
        )
        r2 = _print_model(b4, "B4 +City FE", r2)
        best_b = b4 or b3 or b2 or b1
    else:
        best_b = b3 or b2 or b1

    if best_b:
        print(f"\n  {'Variable':<25s} {'β':>8s} {'SE':>8s} {'t':>8s} {'p':>10s}")
        print(f"  {'-' * 62}")
        for var in best_b.params.index:
            if var == "const" or var.startswith("city_"):
                continue
            print(
                f"  {var:<25s} {best_b.params[var]:>8.4f} "
                f"{best_b.bse[var]:>8.4f} "
                f"{best_b.tvalues[var]:>8.2f} "
                f"{best_b.pvalues[var]:>8.1e} "
                f"{_sigstars(best_b.pvalues[var])}"
            )
        if city_dummies:
            n_city_sig = sum(
                1
                for v in city_dummies
                if v in best_b.pvalues and best_b.pvalues[v] < 0.05
            )
            print(
                f"  ({len(city_dummies)} city dummies, "
                f"{n_city_sig} significant at p<0.05)"
            )


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
# Main
# ---------------------------------------------------------------------------


def main(cities: list[str] | None = None) -> None:
    """Run the LSOA accessibility analysis."""
    print("=" * 70)
    print("LSOA ACCESSIBILITY ANALYSIS: THREE ENERGY SURFACES")
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

    print(f"\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    _cities = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(cities=_cities or None)
