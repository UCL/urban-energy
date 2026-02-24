"""
LSOA-Level Conduit Analysis.

Cities are conduits that capture energy and recycle it through layers of
human interaction (Jacobs, 2000). The measure of urban efficiency is not
how much energy a neighbourhood consumes, but how many transactions,
connections, and transformations that energy enables before it dissipates.

This script asks two questions:

    1. THE COST: How much energy does it take to house and move a household
       in this neighbourhood? (metered building energy + transport from
       commute patterns)

    2. THE RETURN: What does that energy buy? How many amenities, transit
       stops, green spaces are reachable on foot? How connected is the
       street network?

    conduit_efficiency = return / cost

The test: does compact urban form (lower aggregate S/V, taller buildings,
denser fabric) deliver more city per kWh?

All variables aggregate to LSOA (~1,500 people), where metered energy is
native and building counts are large enough for stable physics averages.

Compound metrics (S/V) are computed as aggregate ratios:
    lsoa_sv = sum(envelope_area) / sum(volume)
not as mean(per-building S/V). The aggregate is physically correct — it
represents the total heat-loss surface per unit of heated space for the
neighbourhood.

Usage:
    uv run python stats/proof_of_concept_lsoa.py
    uv run python stats/proof_of_concept_lsoa.py manchester york
"""

import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from urban_energy.paths import TEMP_DIR

DATA_PATH = TEMP_DIR / "processing" / "test" / "uprn_integrated.gpkg"

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
    "ts006_Population Density: "
    "Persons per square kilometre; measures: Value"
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

# Building physics (all from LiDAR + OS footprints, no EPC data)
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
    *_COMMUTE_BANDS.keys(),
]

# Every column the script requires — missing any of these is a hard error
_REQUIRED = (
    ["LSOA21CD", "OA21CD"]
    + _BUILDING_COLS
    + _LSOA_ENERGY
    + _CC_ACCESSIBILITY
    + _CC_CENTRALITY
    + _CENSUS_REQUIRED
)


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


# ---------------------------------------------------------------------------
# 1. Load and aggregate to LSOA
# ---------------------------------------------------------------------------


def load_and_aggregate(cities: list[str] | None = None) -> pd.DataFrame:
    """
    Load UPRN data, aggregate to LSOA with proper Census deduplication.

    Compound metrics (S/V) are computed as aggregate ratios at LSOA level:
    lsoa_sv = sum(envelope) / sum(volume), not mean(per-building S/V).

    Parameters
    ----------
    cities : list[str] or None
        Filter to specific cities, or None for all.

    Returns
    -------
    pd.DataFrame
        LSOA-level dataset.

    Raises
    ------
    ValueError
        If required columns are missing from the dataset.
    """
    print("=" * 70)
    print("LOADING & AGGREGATING TO LSOA")
    print("=" * 70)

    # Probe available columns and validate
    _probe = gpd.read_file(DATA_PATH, rows=1)
    available = set(_probe.columns)
    del _probe
    _validate_columns(available)

    # Build column list — only what we use
    load_cols = ["city"] + _REQUIRED
    load_cols = [c for c in load_cols if c in available]

    col_list = ", ".join(f'"{c}"' for c in load_cols)
    sql = f"SELECT {col_list} FROM uprn_integrated"
    df = gpd.read_file(DATA_PATH, sql=sql)
    print(f"  Loaded {len(df):,} UPRNs ({len(load_cols)} columns)")

    if cities and "city" in df.columns:
        df = df[df["city"].isin(cities)].copy()
        print(f"  Filtered to {cities}: {len(df):,}")

    df = df[df["LSOA21CD"].notna()].copy()

    # Coerce numerics
    for col in _BUILDING_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ===================================================================
    # Phase 1: Census deduplication (OA-level → sum to LSOA)
    # ===================================================================
    census_cols = [c for c in df.columns if c.startswith("ts0")]
    oa_census = df.groupby("OA21CD")[census_cols].first()
    oa_to_lsoa = df.groupby("OA21CD")["LSOA21CD"].first()
    oa_census["LSOA21CD"] = oa_to_lsoa

    # Derive OA area from pop density before summing
    # area_km2 = population / (persons per km²)
    if _TS006_DENSITY in oa_census.columns and _TS001_POP in oa_census.columns:
        oa_pop = pd.to_numeric(oa_census[_TS001_POP], errors="coerce")
        oa_dens = pd.to_numeric(
            oa_census[_TS006_DENSITY], errors="coerce"
        )
        oa_census["_oa_area_km2"] = oa_pop / oa_dens.replace(0, np.nan)

    lsoa_census = oa_census.groupby("LSOA21CD").sum()
    print(f"  Census: {len(oa_census):,} OAs → {len(lsoa_census):,} LSOAs")

    # ===================================================================
    # Phase 2: UPRN → LSOA aggregation
    # ===================================================================
    # S/V = sum(envelope) / sum(volume) — UPRN duplication cancels in the
    # ratio. Means (height, form_factor) weight by UPRN count, giving
    # buildings with more dwellings proportionally more influence.
    # All buildings (residential + commercial) are included: they all
    # shape the neighbourhood's built form and pedestrian environment.
    building_agg: dict[str, str] = {
        # SUM — totals for aggregate ratios
        "footprint_area_m2": "sum",
        "volume_m3": "sum",
        "envelope_area_m2": "sum",
        # MEAN — average building characteristics
        "surface_to_volume": "mean",
        "height_mean": "mean",
        "form_factor": "mean",
    }
    if "city" in df.columns:
        building_agg["city"] = "first"

    # Cityseer — MEAN accessibility across UPRNs
    for col in _CC_ACCESSIBILITY + _CC_CENTRALITY:
        building_agg[col] = "mean"

    # LSOA metered energy — FIRST (already LSOA-level)
    for col in _LSOA_ENERGY:
        building_agg[col] = "first"

    sizes = df.groupby("LSOA21CD").size().reset_index(name="n_uprns")
    lsoa = df.groupby("LSOA21CD").agg(building_agg).reset_index()
    lsoa = lsoa.merge(sizes, on="LSOA21CD")
    lsoa = lsoa.merge(lsoa_census, on="LSOA21CD", how="left")
    lsoa = lsoa.copy()  # defragment

    # ===================================================================
    # Derived variables
    # ===================================================================

    # --- Aggregate S/V (the primary physics metric) ---
    lsoa["lsoa_sv"] = lsoa["envelope_area_m2"] / lsoa["volume_m3"].replace(0, np.nan)
    # Also keep mean per-building S/V for comparison
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

    # Transport energy from Census commute patterns
    total_commuters = pd.to_numeric(lsoa[_COMMUTE_TOTAL], errors="coerce")
    weighted_km = sum(
        km * pd.to_numeric(lsoa[col], errors="coerce")
        for col, km in _COMMUTE_BANDS.items()
    )
    lsoa["avg_commute_km"] = weighted_km / total_commuters.replace(0, np.nan)

    # Car commute share
    car = pd.to_numeric(lsoa[_TS061_CAR], errors="coerce")
    total_w = pd.to_numeric(lsoa[_TS061_TOTAL], errors="coerce")
    lsoa["car_commute_share"] = car / total_w.replace(0, np.nan)

    # Walk/cycle shares
    lsoa["walk_share"] = pd.to_numeric(
        lsoa[_TS061_WALK], errors="coerce"
    ) / total_w.replace(0, np.nan)
    lsoa["cycle_share"] = pd.to_numeric(
        lsoa[_TS061_CYCLE], errors="coerce"
    ) / total_w.replace(0, np.nan)
    lsoa["active_share"] = lsoa["walk_share"].fillna(0) + lsoa["cycle_share"].fillna(0)

    # Transport kWh/hh: commute_km * 2 (return) * 230 days * car% * kWh/km
    kwh_per_km_car = 0.17 / 0.233  # ~0.73
    workers_per_hh = total_commuters / lsoa["total_hh"].replace(0, np.nan)
    annual_commute_km = lsoa["avg_commute_km"] * 2 * 230
    lsoa["transport_kwh_per_hh"] = (
        annual_commute_km
        * lsoa["car_commute_share"]
        * kwh_per_km_car
        * workers_per_hh.clip(upper=2)
    )

    # Total energy cost per household
    lsoa["total_kwh_per_hh"] = lsoa["building_kwh_per_hh"] + lsoa[
        "transport_kwh_per_hh"
    ].fillna(0)

    lsoa["log_total_kwh_per_hh"] = np.log(lsoa["total_kwh_per_hh"].clip(lower=1))
    lsoa["log_building_kwh_per_hh"] = np.log(lsoa["building_kwh_per_hh"].clip(lower=1))

    # --- Occupation density (Census-derived, no building-type confound) ---
    lsoa["avg_hh_size"] = (
        lsoa["total_people"] / lsoa["total_hh"].replace(0, np.nan)
    )

    # --- Population density (from Census OA areas) ---
    if "_oa_area_km2" in lsoa.columns:
        lsoa_area_km2 = lsoa["_oa_area_km2"]
        lsoa["people_per_ha"] = (
            lsoa["total_people"]
            / (lsoa_area_km2 * 100).replace(0, np.nan)
        )
        lsoa.drop(columns=["_oa_area_km2"], inplace=True)

    # --- Energy per person (building only, metered) ---
    lsoa_total_bldg = lsoa["building_kwh_per_hh"] * lsoa["total_hh"]
    lsoa["kwh_per_person"] = (
        lsoa_total_bldg
        / lsoa["total_people"].replace(0, np.nan)
    )
    # Total (building + transport) per person
    lsoa_total_all = lsoa["total_kwh_per_hh"] * lsoa["total_hh"]
    lsoa["total_kwh_per_person"] = (
        lsoa_total_all
        / lsoa["total_people"].replace(0, np.nan)
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

    # --- Filter ---
    valid = (
        (lsoa["total_people"] > 50)
        & (lsoa["n_uprns"] >= 20)
        & lsoa["building_kwh_per_hh"].notna()
        & (lsoa["building_kwh_per_hh"] > 0)
        & lsoa["lsoa_sv"].notna()
    )
    lsoa = lsoa[valid].copy()

    n_cities = lsoa["city"].nunique() if "city" in lsoa.columns else 1
    print(f"\n  {len(lsoa):,} LSOAs across {n_cities} cities")
    print(f"  UPRNs/LSOA: median={lsoa['n_uprns'].median():.0f}")
    print("\n  Physics:")
    print(
        f"    Aggregate S/V: median={lsoa['lsoa_sv'].median():.3f}  "
        f"(mean={lsoa['lsoa_sv'].mean():.3f})"
    )
    print(
        f"    Mean bldg S/V: median={lsoa['mean_building_sv'].median():.3f}  "
        f"(mean={lsoa['mean_building_sv'].mean():.3f})"
    )
    print(f"    Height: median={lsoa['height_mean'].median():.1f}m")
    print("\n  Cost (kWh/household):")
    print(f"    Building: median={lsoa['building_kwh_per_hh'].median():.0f}")
    t = lsoa["transport_kwh_per_hh"].dropna()
    print(f"    Transport: median={t.median():.0f}")
    print(f"    Total: median={lsoa['total_kwh_per_hh'].median():.0f}")

    return lsoa


# ---------------------------------------------------------------------------
# 2. The return side: accessibility
# ---------------------------------------------------------------------------


def build_accessibility(lsoa: pd.DataFrame) -> pd.DataFrame:
    """
    Build accessibility metrics from cityseer columns.

    Parameters
    ----------
    lsoa : pd.DataFrame
        LSOA-level data with _CC_ACCESSIBILITY columns.

    Returns
    -------
    pd.DataFrame
        Updated with accessibility_pc1 and individual metrics.

    Raises
    ------
    ValueError
        If accessibility columns are missing.
    """
    print("\n" + "=" * 70)
    print("THE RETURN SIDE: What does the energy buy?")
    print("=" * 70)

    acc_cols = _CC_ACCESSIBILITY
    missing = [c for c in acc_cols if c not in lsoa.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} accessibility columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    print(f"\n  Amenities reachable within 800m walk ({len(acc_cols)} metrics):")
    for c in acc_cols:
        short = c.replace("cc_", "").replace("_800_wt", "")
        med = lsoa[c].median()
        print(f"    {short:<25s} median = {med:.1f}")

    # PCA
    valid_mask = lsoa[acc_cols].notna().all(axis=1)
    n_components = min(3, len(acc_cols))
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)
    X = scaler.fit_transform(lsoa.loc[valid_mask, acc_cols].values)
    components = pca.fit_transform(X)

    for i in range(n_components):
        lsoa[f"accessibility_pc{i + 1}"] = np.nan
        lsoa.loc[valid_mask, f"accessibility_pc{i + 1}"] = components[:, i]

    total_var = pca.explained_variance_ratio_.sum()
    print(f"\n  PCA ({n_components} components, {total_var:.0%} variance):")
    for i, var in enumerate(pca.explained_variance_ratio_):
        loadings = pca.components_[i]
        top_idx = np.argsort(np.abs(loadings))[::-1][:3]
        top = ", ".join(
            f"{acc_cols[j].replace('cc_', '').replace('_800_wt', '')}"
            f"({loadings[j]:+.2f})"
            for j in top_idx
        )
        print(f"    PC{i + 1}: {var:.0%}  [{top}]")

    # Network centrality
    for c in _CC_CENTRALITY:
        if c in lsoa.columns:
            print(f"  {c}: median={lsoa[c].median():.3f}")

    return lsoa


# ---------------------------------------------------------------------------
# 3. The conduit: return per unit cost
# ---------------------------------------------------------------------------


def compute_conduit(lsoa: pd.DataFrame) -> pd.DataFrame:
    """
    Compute conduit efficiency: accessibility per kWh.

    Parameters
    ----------
    lsoa : pd.DataFrame
        LSOA data with accessibility and energy.

    Returns
    -------
    pd.DataFrame
        Updated with conduit metrics.

    Raises
    ------
    ValueError
        If accessibility_pc1 is missing.
    """
    print("\n" + "=" * 70)
    print("THE CONDUIT: How much city per kWh?")
    print("=" * 70)

    if "accessibility_pc1" not in lsoa.columns:
        raise ValueError(
            "accessibility_pc1 not found — run build_accessibility() first"
        )

    # Shift PC1 positive for meaningful ratios
    pc1 = lsoa["accessibility_pc1"]
    pc1_pos = pc1 - pc1.min() + 1
    lsoa["conduit_total"] = pc1_pos / lsoa["total_kwh_per_hh"]
    lsoa["conduit_building"] = pc1_pos / lsoa["building_kwh_per_hh"]
    lsoa["log_conduit_total"] = np.log(lsoa["conduit_total"].clip(lower=1e-10))

    # Individual amenity conduit ratios
    for c in [
        "cc_fsa_restaurant_800_wt",
        "cc_fsa_pub_800_wt",
        "cc_bus_800_wt",
        "cc_rail_800_wt",
        "cc_greenspace_800_wt",
    ]:
        if c in lsoa.columns:
            short = c.replace("cc_", "").replace("_800_wt", "")
            val = pd.to_numeric(lsoa[c], errors="coerce").clip(lower=0) + 1
            lsoa[f"{short}_per_kwh"] = val / lsoa["total_kwh_per_hh"]

    # Compactness quartiles (aggregate S/V)
    lsoa["sv_quartile"] = pd.qcut(
        lsoa["lsoa_sv"],
        4,
        labels=["Q1 compact", "Q2", "Q3", "Q4 sprawl"],
    )

    print(
        f"\n  {'Quartile':<14s} {'S/V':>6s} {'Height':>7s} "
        f"{'kWh/hh':>8s} {'Access':>7s} {'Conduit':>8s} {'N':>5s}"
    )
    print(f"  {'-' * 60}")

    for q in ["Q1 compact", "Q2", "Q3", "Q4 sprawl"]:
        sub = lsoa[lsoa["sv_quartile"] == q]
        print(
            f"  {q:<14s} {sub['lsoa_sv'].median():>6.3f} "
            f"{sub['height_mean'].median():>6.1f}m "
            f"{sub['total_kwh_per_hh'].median():>8.0f} "
            f"{sub['accessibility_pc1'].median():>7.2f} "
            f"{sub['conduit_total'].median():>8.5f} "
            f"{len(sub):>5d}"
        )

    q1 = lsoa[lsoa["sv_quartile"] == "Q1 compact"]
    q4 = lsoa[lsoa["sv_quartile"] == "Q4 sprawl"]
    if len(q1) > 0 and len(q4) > 0:
        ratio = q1["conduit_total"].median() / q4["conduit_total"].median()
        print(f"\n  Compact delivers {ratio:.2f}x more city per kWh than sprawl")

        # Break down: how much is cost, how much is return?
        cost_ratio = q4["total_kwh_per_hh"].median() / q1["total_kwh_per_hh"].median()
        acc_q1 = q1["accessibility_pc1"].median()
        acc_q4 = q4["accessibility_pc1"].median()
        print(f"    Cost advantage: sprawl uses {cost_ratio:.2f}x more kWh/hh")
        print(f"    Return advantage: compact PC1 = {acc_q1:.2f} vs {acc_q4:.2f}")

    return lsoa


# ---------------------------------------------------------------------------
# 4. Systematic summary: the compounding normalisations
# ---------------------------------------------------------------------------


def print_systematic_summary(lsoa: pd.DataFrame) -> None:
    """
    Show how the efficiency gap widens with each normalisation layer.

    This is the centerpiece table: same data, different denominators,
    and the Q1/Q4 ratio reveals the compounding.
    """
    print(f"\n{'=' * 70}")
    print("SYSTEMATIC SUMMARY BY COMPACTNESS QUARTILE")
    print("=" * 70)

    if "sv_quartile" not in lsoa.columns:
        return

    qs = ["Q1 compact", "Q2", "Q3", "Q4 sprawl"]
    w = 12  # column width

    # (section, label, column, format)
    metrics: list[tuple[str, str, str, str]] = [
        ("FORM", "S/V ratio", "lsoa_sv", ".3f"),
        ("FORM", "Height (m)", "height_mean", ".1f"),
        ("DENSITY", "People/ha", "people_per_ha", ".0f"),
        ("DENSITY", "People/hh", "avg_hh_size", ".2f"),
        ("ENERGY", "kWh/person", "kwh_per_person", ",.0f"),
        ("ENERGY", "kWh/hh (bldg)", "building_kwh_per_hh", ",.0f"),
        ("ENERGY", "kWh/hh (trans)", "transport_kwh_per_hh", ",.0f"),
        ("ENERGY", "kWh/hh (total)", "total_kwh_per_hh", ",.0f"),
        ("ENERGY", "kWh/person+trans", "total_kwh_per_person", ",.0f"),
        ("RETURN", "Accessibility", "accessibility_pc1", ".2f"),
        ("CONDUIT", "Access/kWh_hh", "conduit_total", ".6f"),
    ]

    # Header
    hdr = f"  {'':22s}"
    for q in qs:
        hdr += f" {q:>{w}s}"
    hdr += f" {'Q1/Q4':>7s}"
    print(f"\n{hdr}")
    print(f"  {'-' * (22 + len(qs) * (w + 1) + 8)}")

    prev_section = ""
    for section, label, col, fmt in metrics:
        if col not in lsoa.columns:
            continue
        if section != prev_section:
            if prev_section:
                print()
            print(f"  {section}")
            prev_section = section

        vals = [
            lsoa.loc[lsoa["sv_quartile"] == q, col].median()
            for q in qs
        ]
        line = f"    {label:<20s}"
        for v in vals:
            line += f" {v:>{w}{fmt}}"
        v1, v4 = vals[0], vals[3]
        if v4 != 0 and not np.isnan(v1) and not np.isnan(v4):
            line += f" {v1 / v4:>6.2f}x"
        print(line)

    # Narrative: the compounding
    q1 = lsoa[lsoa["sv_quartile"] == "Q1 compact"]
    q4 = lsoa[lsoa["sv_quartile"] == "Q4 sprawl"]
    if len(q1) == 0 or len(q4) == 0:
        return

    print("\n  The compounding effect (Q1 compact / Q4 sprawl):")
    steps = [
        ("kWh/hh (bldg)", "building_kwh_per_hh", "building energy cost"),
        ("kWh/hh (total)", "total_kwh_per_hh", "+ transport"),
        ("kWh/person", "total_kwh_per_person", "+ occupation density"),
    ]
    for label, col, gloss in steps:
        if col not in lsoa.columns:
            continue
        r = q1[col].median() / q4[col].median()
        print(f"    {label:<20s} {r:.2f}x  ({gloss})")

    if "conduit_total" in lsoa.columns:
        r = q1["conduit_total"].median() / q4["conduit_total"].median()
        print(f"    {'Conduit':<20s} {r:.2f}x  (+ accessibility return)")



# ---------------------------------------------------------------------------
# 5. Energy cost decomposition
# ---------------------------------------------------------------------------


def print_energy_decomposition(lsoa: pd.DataFrame) -> None:
    """Show building vs transport energy by compactness."""
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

    if "sv_quartile" not in lsoa.columns:
        return

    print("\n  By compactness:")
    print(
        f"  {'Quartile':<14s} {'Build':>7s} {'Trans':>7s} {'Total':>7s} "
        f"{'T%':>5s} {'Car%':>5s} {'Commute':>8s} {'Active':>7s}"
    )
    print(f"  {'-' * 62}")
    for q in ["Q1 compact", "Q2", "Q3", "Q4 sprawl"]:
        s = sub[sub["sv_quartile"] == q]
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
            f"  {q:<14s} {b:>7.0f} {t:>7.0f} {tot:>7.0f} "
            f"{tpct:>4.0%} {car:>4.0%} {km:>7.1f}km {act:>5.0%}"
        )


# ---------------------------------------------------------------------------
# 5. Regression: what predicts conduit efficiency?
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

    A. DV = conduit efficiency (return/cost)
       What neighbourhood form delivers more city per kWh?

    B. DV = accessibility (what you get)
       Controlling for energy spend, does form predict access?
    """
    print(f"\n{'=' * 70}")
    print("REGRESSION: WHAT DRIVES CONDUIT EFFICIENCY?")
    print("=" * 70)

    physics = ["lsoa_sv", "height_mean"]
    occupation = ["people_per_ha"]
    controls = ["pct_not_deprived"]

    # --- A: DV = log(conduit efficiency) ---
    print("\n  --- A. What makes a neighbourhood an efficient conduit? ---")
    print("  DV = log(accessibility / kWh per household)\n")

    a1 = _run_ols(lsoa, "log_conduit_total", physics, "A1 Compactness")
    r2 = _print_model(a1, "A1 Compactness")

    a2 = _run_ols(lsoa, "log_conduit_total", physics + occupation, "A2 +Occupation")
    r2 = _print_model(a2, "A2 +Occupation", r2)

    a3 = _run_ols(
        lsoa,
        "log_conduit_total",
        physics + occupation + controls,
        "A3 +Deprivation",
    )
    r2 = _print_model(a3, "A3 +Deprivation", r2)

    best_a = a3 or a2 or a1
    if best_a:
        _print_coefficients(best_a)

    # --- B: DV = accessibility, energy as predictor ---
    print("\n\n  --- B. For a given energy spend, does form deliver more access? ---")
    print("  DV = accessibility_pc1\n")

    b1 = _run_ols(lsoa, "accessibility_pc1", ["log_total_kwh_per_hh"], "B1 Energy")
    r2 = _print_model(b1, "B1 Energy alone")

    b2 = _run_ols(
        lsoa,
        "accessibility_pc1",
        ["log_total_kwh_per_hh"] + physics,
        "B2 +Compactness",
    )
    r2 = _print_model(b2, "B2 +Compactness", r2)

    b3 = _run_ols(
        lsoa,
        "accessibility_pc1",
        ["log_total_kwh_per_hh"] + physics + occupation + controls,
        "B3 +Occupation+Deprivation",
    )
    r2 = _print_model(b3, "B3 +Occ+Dep", r2)

    best_b = b3 or b2 or b1
    if best_b:
        _print_coefficients(best_b)


# ---------------------------------------------------------------------------
# 6. Deprivation control
# ---------------------------------------------------------------------------


def run_deprivation_control(lsoa: pd.DataFrame) -> None:
    """Check conduit efficiency within each deprivation quintile."""
    print(f"\n{'=' * 70}")
    print("DEPRIVATION CONTROL")
    print("=" * 70)

    if "deprivation_quintile" not in lsoa.columns:
        print("  No deprivation quintiles (too few valid observations)")
        return

    physics = ["lsoa_sv", "height_mean"]

    quintiles = ["Q1 most", "Q2", "Q3", "Q4", "Q5 least"]
    print(
        f"\n  {'Quintile':<12s} {'N':>5s} {'kWh/hh':>8s} {'Conduit':>8s} "
        f"{'sv_β':>8s} {'p':>10s}"
    )
    print(f"  {'-' * 56}")

    n_sig = 0
    for q in quintiles:
        sub = lsoa[lsoa["deprivation_quintile"] == q].copy()
        if len(sub) < 20:
            continue
        kwh = sub["total_kwh_per_hh"].median()
        cond = sub["conduit_total"].median()

        m = _run_ols(sub, "log_conduit_total", physics, f"dep-{q}")
        sv_b = m.params.get("lsoa_sv", np.nan) if m else np.nan
        sv_p = m.pvalues.get("lsoa_sv", np.nan) if m else np.nan
        if not np.isnan(sv_p) and sv_p < 0.05:
            n_sig += 1
        print(
            f"  {q:<12s} {len(sub):>5d} {kwh:>8.0f} {cond:>8.5f} "
            f"{sv_b:>8.4f} {sv_p:>8.1e} {_sigstars(sv_p)}"
        )

    print(f"\n  S/V → conduit efficiency significant in {n_sig}/5 quintiles")
    if n_sig >= 3:
        print("  Effect holds across wealth levels")


# ---------------------------------------------------------------------------
# 7. Per-city
# ---------------------------------------------------------------------------


def run_per_city(lsoa: pd.DataFrame) -> None:
    """Conduit comparison across cities."""
    print(f"\n{'=' * 70}")
    print("PER-CITY BREAKDOWN")
    print("=" * 70)

    if "city" not in lsoa.columns:
        return

    physics = ["lsoa_sv", "height_mean"]
    cities = sorted(lsoa["city"].unique())

    print(
        f"\n  {'City':<16s} {'N':>5s} {'kWh/hh':>8s} {'S/V':>6s} "
        f"{'Conduit':>8s} {'Q1/Q4':>6s} {'sv_β':>8s} {'p':>10s}"
    )
    print(f"  {'-' * 72}")

    for city in cities:
        sub = lsoa[lsoa["city"] == city].copy()
        kwh = sub["total_kwh_per_hh"].median()
        sv = sub["lsoa_sv"].median()
        cond = sub["conduit_total"].median()

        # Q1/Q4 ratio within city
        ratio = np.nan
        if "sv_quartile" in sub.columns:
            q1 = sub[sub["sv_quartile"] == "Q1 compact"]
            q4 = sub[sub["sv_quartile"] == "Q4 sprawl"]
            if len(q1) > 0 and len(q4) > 0:
                c1 = q1["conduit_total"].median()
                c4 = q4["conduit_total"].median()
                if c4 > 0:
                    ratio = c1 / c4

        m = _run_ols(sub, "log_conduit_total", physics, city)
        sv_b = m.params.get("lsoa_sv", np.nan) if m else np.nan
        sv_p = m.pvalues.get("lsoa_sv", np.nan) if m else np.nan

        print(
            f"  {city:<16s} {len(sub):>5d} {kwh:>8.0f} {sv:>6.3f} "
            f"{cond:>8.5f} {ratio:>5.1f}x {sv_b:>8.4f} "
            f"{sv_p:>8.1e} {_sigstars(sv_p)}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cities: list[str] | None = None) -> None:
    """Run the LSOA conduit analysis."""
    print("=" * 70)
    print("LSOA CONDUIT ANALYSIS")
    print("Per unit energy spent, how much city do you get?")
    print("=" * 70)

    lsoa = load_and_aggregate(cities)
    lsoa = build_accessibility(lsoa)
    lsoa = compute_conduit(lsoa)
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
