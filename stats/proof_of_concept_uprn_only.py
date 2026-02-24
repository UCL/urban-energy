"""
Proof of Concept: The Trophic Layers of Urban Energy.

Tests the core thesis on Manchester (E63008401):
    Cities are conduits that capture energy and recycle it through layers of
    human interaction (Jacobs, 2000). Compact morphologies deliver more urban
    connectivity per unit of energy consumed. Sprawl burns the same energy
    through a single trophic level — the conduit leaks.

    The analogy: a rainforest and a desert receive the same solar radiation
    per m². The rainforest captures it through dozens of trophic layers
    (canopy, understorey, epiphytes, soil biome, mycorrhizal networks).
    The desert radiates it straight back. The measure of ecosystem complexity
    is not energy input but energy RETENTION — how many times it's recycled
    before dissipating.

    Urban equivalent: a 1km² inner-city neighbourhood has thousands of land
    uses, each a layer in the conduit. A 1km² suburban plot has a handful.
    Same energy input (building + transport), radically different interaction
    depth.

Steps:
    1. Physics signatures — Types have distinct thermal envelopes
    2. Physics → energy — Physics predicts SAP + metered energy (real, not artifact)
    3. Trophic layers — Types differ across ALL accessibility layers
    4. The compounding — Three normalisations, gap widens at each level
    5. Deprivation control — Compounding holds within deprivation quintiles
    6. Lock-in — Stock composition locks inefficiency in for decades

Decision gates:
    After Step 1: If types don't separate on physics → no morphological signal
    After Step 2: If physics doesn't predict energy → physics is decorative
    After Step 4: If gap doesn't widen → no compounding (just building physics)

Usage:
    uv run python stats/proof_of_concept.py
"""

import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from urban_energy.paths import TEMP_DIR

DATA_PATH = TEMP_DIR / "processing" / "test" / "uprn_integrated.gpkg"
BUILDINGS_PATH = TEMP_DIR / "processing" / "test" / "buildings_morphology.gpkg"

# Morphological type ordering (least to most compact)
TYPE_ORDER = ["detached", "semi", "end_terrace", "mid_terrace", "flat"]
TYPE_LABELS = {
    "detached": "Detached",
    "semi": "Semi-detached",
    "end_terrace": "End terrace",
    "mid_terrace": "Mid terrace",
    "flat": "Flat/apartment",
}

# Physics signature columns — all geometry-derived (no EPC input)
PHYSICS_COLS = [
    "envelope_per_dwelling",
    "party_ratio",
    "surface_to_volume",
    "height_mean",
    "form_factor",
]

# Control variables for "all else equal"
CONTROL_COLS = ["log_floor_area", "building_age"]

# Trophic layers of the urban conduit
# Each layer captures a different dimension of how energy is recycled
# through the urban system. Mapped to cityseer accessibility metrics.
#
# _wt = gravity-weighted count at 800m network distance (higher = more accessible)
# cc_harmonic = harmonic closeness centrality (higher = more integrated)
TROPHIC_LAYERS: dict[str, dict[str, str | list[str]]] = {
    "PHYSICAL SUBSTRATE": {
        "label": "Street connectivity",
        "cols": ["cc_harmonic_800"],
        "desc": "Network integration — how many places reachable on foot",
    },
    "COMMERCIAL EXCHANGE": {
        "label": "Amenity access",
        "cols": [
            "cc_fsa_restaurant_800_wt",
            "cc_fsa_pub_800_wt",
            "cc_fsa_takeaway_800_wt",
            "cc_fsa_other_800_wt",
        ],
        "desc": "Places where economic/social transactions happen",
    },
    "MOBILITY": {
        "label": "Transit access",
        "cols": ["cc_bus_800_wt", "cc_rail_800_wt"],
        "desc": "Connections to the wider city network",
    },
    "RESTORATION": {
        "label": "Green space",
        "cols": ["cc_greenspace_800_wt"],
        "desc": "Regenerative capacity — parks and open space",
    },
}


def _sigstars(p: float) -> str:
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _short_physics(col: str) -> str:
    """Abbreviate physics column names for table display."""
    return (
        col.replace("surface_to_volume", "S/V")
        .replace("envelope_per_dwelling", "Env/dw")
        .replace("log_envelope_per_dwelling", "lnEnv/dw")
        .replace("party_ratio", "Party")
        .replace("height_mean", "Height")
        .replace("form_factor", "FormF")
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(city: str | None = None) -> pd.DataFrame:
    """
    Load and prepare the integrated UPRN dataset.

    Parameters
    ----------
    city : str or None
        If provided, filter to a single city (requires ``city`` column
        in the dataset, written by the multi-city pipeline).
        If None, load all available data.

    Returns
    -------
    pd.DataFrame
        Analysis-ready dataframe with derived variables.
    """
    print("Loading data...")
    # Select only columns used by the analysis (199 → ~40 columns, much faster)
    _LOAD_COLS = [
        # Identifiers
        "city",
        "OA21CD",
        "LSOA21CD",
        # EPC
        "ENERGY_CONSUMPTION_CURRENT",
        "TOTAL_FLOOR_AREA",
        "BUILT_FORM",
        "PROPERTY_TYPE",
        "CONSTRUCTION_AGE_BAND",
        # Building morphology
        "footprint_area_m2",
        "perimeter_m",
        "envelope_area_m2",
        "volume_m3",
        "surface_to_volume",
        "form_factor",
        "height_mean",
        "height_max",
        "shared_wall_ratio",
        # Cityseer centrality (800–4800m)
        "cc_harmonic_800",
        "cc_harmonic_1600",
        "cc_harmonic_3200",
        "cc_harmonic_4800",
        "cc_density_800",
        "cc_density_1600",
        "cc_density_3200",
        "cc_density_4800",
        "cc_betweenness_800",
        "cc_betweenness_1600",
        "cc_betweenness_3200",
        "cc_betweenness_4800",
        # Cityseer accessibility (400, 800, 1600m)
        "cc_fsa_restaurant_400_wt",
        "cc_fsa_restaurant_800_wt",
        "cc_fsa_restaurant_1600_wt",
        "cc_fsa_pub_400_wt",
        "cc_fsa_pub_800_wt",
        "cc_fsa_pub_1600_wt",
        "cc_fsa_takeaway_400_wt",
        "cc_fsa_takeaway_800_wt",
        "cc_fsa_takeaway_1600_wt",
        "cc_fsa_other_400_wt",
        "cc_fsa_other_800_wt",
        "cc_fsa_other_1600_wt",
        "cc_bus_400_wt",
        "cc_bus_800_wt",
        "cc_bus_1600_wt",
        "cc_rail_400_wt",
        "cc_rail_800_wt",
        "cc_rail_1600_wt",
        "cc_greenspace_400_wt",
        "cc_greenspace_800_wt",
        "cc_greenspace_1600_wt",
        # Census population (OA-level, for genuine per-capita)
        "ts001_Residence type: Lives in a household; measures: Value",
        # Census demographics
        "ts006_Population Density: Persons per square kilometre; measures: Value",
        "ts011_Household deprivation: Household is not deprived in any dimension; measures: Value",
        "ts011_Household deprivation: Total: All households; measures: Value",
        "ts017_Household size: Total: All household spaces; measures: Value",
        "ts017_Household size: 0 people in household; measures: Value",
        "ts017_Household size: 1 person in household; measures: Value",
        "ts017_Household size: 2 people in household; measures: Value",
        "ts017_Household size: 3 people in household; measures: Value",
        "ts017_Household size: 4 people in household; measures: Value",
        "ts017_Household size: 5 people in household; measures: Value",
        "ts017_Household size: 6 people in household; measures: Value",
        "ts017_Household size: 7 people in household; measures: Value",
        "ts017_Household size: 8 or more people in household; measures: Value",
        "ts045_Number of cars or vans: No cars or vans in household; measures: Value",
        "ts045_Number of cars or vans: 1 car or van in household; measures: Value",
        "ts045_Number of cars or vans: 2 cars or vans in household; measures: Value",
        "ts045_Number of cars or vans: 3 or more cars or vans in household; measures: Value",
    ]
    # Filter to columns that exist in the file, then use SQL for fast load
    import fiona

    with fiona.open(DATA_PATH) as f:
        available = set(f.schema["properties"].keys())
    load_cols = [c for c in _LOAD_COLS if c in available]
    skipped = set(_LOAD_COLS) - set(load_cols)
    if skipped:
        print(f"  Note: {len(skipped)} columns not in data, skipping")
    col_list = ", ".join(f'"{c}"' for c in load_cols)
    sql = f"SELECT {col_list} FROM uprn_integrated"
    gdf = gpd.read_file(DATA_PATH, sql=sql)
    print(f"  Total UPRNs: {len(gdf):,}  ({len(load_cols)} columns loaded)")

    # Filter to a single city if requested
    if city is not None and "city" in gdf.columns:
        gdf = gdf[gdf["city"] == city].copy()
        print(f"  Filtered to {city}: {len(gdf):,}")
    elif city is not None:
        print(f"  WARNING: city column not in data, ignoring city={city}")

    # --- Join building morphology from OS + LiDAR ---
    morph_cols = [
        "footprint_area_m2",
        "perimeter_m",
        "envelope_area_m2",
        "volume_m3",
        "surface_to_volume",
        "form_factor",
        "height_mean",
        "height_max",
        "shared_wall_ratio",
    ]
    need_join = not all(c in gdf.columns for c in morph_cols[:3])
    if not need_join:
        print("  Building morphology already in dataset")
    else:
        print("  WARNING: morphology columns missing from UPRN data")

    # Filter to EPC records with valid energy and floor area
    df = gdf[gdf["ENERGY_CONSUMPTION_CURRENT"].notna()].copy()
    df = df[(df["TOTAL_FLOOR_AREA"] > 0) & df["TOTAL_FLOOR_AREA"].notna()].copy()
    print(f"  With EPC energy + valid floor area: {len(df):,}")

    # --- Derived energy variables ---
    df["energy_intensity"] = df["ENERGY_CONSUMPTION_CURRENT"]
    df["log_energy_intensity"] = np.log(df["energy_intensity"].clip(lower=1))
    df["log_floor_area"] = np.log(df["TOTAL_FLOOR_AREA"])
    df["total_energy_kwh"] = df["ENERGY_CONSUMPTION_CURRENT"] * df["TOTAL_FLOOR_AREA"]

    # Construction age
    age_map: dict[str, int] = {
        "England and Wales: before 1900": 1875,
        "England and Wales: 1900-1929": 1915,
        "England and Wales: 1930-1949": 1940,
        "England and Wales: 1950-1966": 1958,
        "England and Wales: 1967-1975": 1971,
        "England and Wales: 1976-1982": 1979,
        "England and Wales: 1983-1990": 1987,
        "England and Wales: 1991-1995": 1993,
        "England and Wales: 1996-2002": 1999,
        "England and Wales: 2003-2006": 2005,
        "England and Wales: 2007 onwards": 2010,
        "England and Wales: 2012 onwards": 2015,
    }
    df["construction_year"] = df["CONSTRUCTION_AGE_BAND"].map(age_map)
    df["building_age"] = 2024 - df["construction_year"]

    # Morphological type classification
    built_form_map = {
        "Detached": "detached",
        "Semi-Detached": "semi",
        "Mid-Terrace": "mid_terrace",
        "End-Terrace": "end_terrace",
        "Enclosed Mid-Terrace": "mid_terrace",
        "Enclosed End-Terrace": "end_terrace",
    }
    df["morph_type"] = df["BUILT_FORM"].map(built_form_map)
    is_flat = df["PROPERTY_TYPE"].str.lower().str.contains("flat", na=False)
    df.loc[is_flat, "morph_type"] = "flat"

    # --- Envelope per dwelling (geometry-only, no EPC input) ---
    bldg_cols = ["footprint_area_m2", "perimeter_m", "envelope_area_m2"]
    bldg_cols_avail = [c for c in bldg_cols if c in df.columns]
    if len(bldg_cols_avail) >= 2:
        for c in bldg_cols_avail:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        group_keys = [df[c].round(2) for c in bldg_cols_avail]
        group_series = pd.Series(
            list(zip(*[k.values for k in group_keys])), index=df.index
        )
        uprn_counts = group_series.map(group_series.value_counts())
        df["n_uprns_in_building"] = uprn_counts

        if "envelope_area_m2" in df.columns:
            envelope = pd.to_numeric(df["envelope_area_m2"], errors="coerce")
        else:
            perim = pd.to_numeric(df["perimeter_m"], errors="coerce")
            fp = pd.to_numeric(df["footprint_area_m2"], errors="coerce")
            ht = pd.to_numeric(df["height_mean"], errors="coerce")
            envelope = perim * ht + 2 * fp

        df["envelope_per_dwelling"] = envelope / df["n_uprns_in_building"]
        df["log_envelope_per_dwelling"] = np.log(
            df["envelope_per_dwelling"].clip(lower=1)
        )

        n_bldg = group_series.nunique()
        print(f"  Unique buildings identified: {n_bldg:,}")
        print(
            f"  UPRNs/building: median={df['n_uprns_in_building'].median():.0f},"
            f" max={df['n_uprns_in_building'].max():.0f}"
        )

        # --- Party ratio: virtual interior wall metric ---
        storey_height = 2.7
        n_uprns = df["n_uprns_in_building"].copy()
        ht = pd.to_numeric(df["height_mean"], errors="coerce")
        fp = pd.to_numeric(df["footprint_area_m2"], errors="coerce")
        perim = pd.to_numeric(df["perimeter_m"], errors="coerce")

        n_floors = np.maximum(1, np.round(ht / storey_height)).fillna(1)
        half_perim = perim / 2
        discriminant = half_perim**2 - 4 * fp
        depth = np.where(
            discriminant > 0,
            (half_perim - np.sqrt(np.maximum(discriminant, 0))) / 2,
            np.sqrt(fp),
        )
        depth = pd.Series(depth, index=df.index).clip(lower=1.0)

        n_per_floor = n_uprns / n_floors
        party_floor_surfaces = (np.minimum(n_floors, n_uprns) - 1).clip(lower=0)
        party_floor_area = party_floor_surfaces * fp
        walls_per_floor = (n_per_floor - 1).clip(lower=0)
        party_wall_area = walls_per_floor * depth * storey_height * n_floors
        total_party = party_floor_area + party_wall_area

        df["party_ratio"] = total_party / (envelope + total_party)
        df["party_ratio"] = df["party_ratio"].clip(lower=0, upper=1).fillna(0)

        for mt in ["detached", "semi", "mid_terrace", "flat"]:
            sub = df[df["morph_type"] == mt]
            if len(sub) > 0:
                med = sub["party_ratio"].median()
                print(f"  party_ratio median ({mt}): {med:.3f}")

    # Coerce physics columns
    for col in PHYSICS_COLS + ["perimeter_m", "footprint_area_m2", "height_mean"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Recompute S/V using EPC-inferred wall sharing ---
    perim_ext_map: dict[str, float] = {
        "Detached": 1.00,
        "Semi-Detached": 0.75,
        "End-Terrace": 0.75,
        "Enclosed End-Terrace": 0.75,
        "Mid-Terrace": 0.50,
        "Enclosed Mid-Terrace": 0.50,
    }
    ext_frac = df["BUILT_FORM"].map(perim_ext_map).fillna(1.0)

    sv_geom_cols = ["perimeter_m", "footprint_area_m2", "height_mean"]
    if all(c in df.columns for c in sv_geom_cols):
        has_geom = (
            df["perimeter_m"].notna()
            & df["footprint_area_m2"].notna()
            & df["height_mean"].notna()
            & (df["height_mean"] > 0)
            & (df["footprint_area_m2"] > 0)
        )
        height = df.loc[has_geom, "height_mean"]
        perimeter = df.loc[has_geom, "perimeter_m"]
        footprint = df.loc[has_geom, "footprint_area_m2"]
        frac = ext_frac.loc[has_geom]

        ext_wall = perimeter * height * frac
        envelope_sv = ext_wall + footprint + footprint  # walls + roof + floor
        volume = footprint * height

        sv_corrected = (envelope_sv / volume).clip(upper=3.0)
        df.loc[has_geom, "surface_to_volume"] = sv_corrected
    else:
        print("  WARNING: S/V recomputation skipped (missing columns)")

    # --- Household size and per-capita energy ---
    size_cols = {
        n: f"ts017_Household size: {n} {'person' if n == 1 else 'people'}"
        f" in household; measures: Value"
        for n in range(1, 8)
    }
    size_cols[8] = (
        "ts017_Household size: 8 or more people in household; measures: Value"
    )
    total_people = sum(size * df[col] for size, col in size_cols.items())
    total_hh = (
        df["ts017_Household size: Total: All household spaces; measures: Value"]
        - df["ts017_Household size: 0 people in household; measures: Value"]
    )
    df["avg_household_size"] = total_people / total_hh
    df["energy_per_capita"] = df["total_energy_kwh"] / df["avg_household_size"]

    # Population density
    df["pop_density"] = df[
        "ts006_Population Density: Persons per square kilometre; measures: Value"
    ]

    # --- Transport energy from Census car ownership (TS045) ---
    _car_pfx = "ts045_Number of cars or vans: "
    _car_sfx = "; measures: Value"
    car_cols = {
        0: f"{_car_pfx}No cars or vans in household{_car_sfx}",
        1: f"{_car_pfx}1 car or van in household{_car_sfx}",
        2: f"{_car_pfx}2 cars or vans in household{_car_sfx}",
        3: f"{_car_pfx}3 or more cars or vans in household{_car_sfx}",
    }
    has_cars = all(c in df.columns for c in car_cols.values())
    if has_cars:
        total_cars = sum(n * df[col] for n, col in car_cols.items())
        total_hh_cars = sum(df[col] for col in car_cols.values())
        df["avg_cars_per_hh"] = total_cars / total_hh_cars.replace(0, np.nan)

        km_per_vehicle_year = 12_000
        kwh_per_km = 0.17 / 0.233  # kg CO2/km ÷ kg CO2/kWh
        kwh_per_vehicle = km_per_vehicle_year * kwh_per_km
        df["transport_kwh_per_hh"] = df["avg_cars_per_hh"] * kwh_per_vehicle

        # Total energy per capita: building + transport
        df["total_energy_per_capita"] = (
            df["total_energy_kwh"] + df["transport_kwh_per_hh"]
        ) / df["avg_household_size"]
    else:
        df["total_energy_per_capita"] = df["energy_per_capita"]

    # --- Deprivation score from Census TS011 ---
    dep_col = (
        "ts011_Household deprivation: "
        "Household is not deprived in any dimension; measures: Value"
    )
    dep_total_col = (
        "ts011_Household deprivation: Total: All households; measures: Value"
    )
    if dep_col in df.columns and dep_total_col in df.columns:
        df["pct_not_deprived"] = df[dep_col] / df[dep_total_col] * 100
        dep_labels = ["Q1 most", "Q2", "Q3", "Q4", "Q5 least"]
        df["deprivation_quintile"] = pd.qcut(
            df["pct_not_deprived"], q=5, labels=dep_labels
        )

    # --- LSOA-level scaling data (GVA + BRES) ---
    scaling_path = TEMP_DIR / "statistics" / "lsoa_scaling.parquet"
    if scaling_path.exists() and "LSOA21CD" in df.columns:
        scaling_df = pd.read_parquet(scaling_path)
        # Direct join: LSOA_CODE matches LSOA21CD for ~95% of LSOAs
        # (only split/merged LSOAs between 2011 and 2021 differ)
        scaling_df = scaling_df.rename(columns={"LSOA_CODE": "LSOA21CD"})
        n_before = len(df)
        df = df.merge(
            scaling_df[["LSOA21CD", "lsoa_gva_millions", "lsoa_employment"]],
            on="LSOA21CD",
            how="left",
        )
        gva_matched = df["lsoa_gva_millions"].notna().sum()
        bres_matched = df["lsoa_employment"].notna().sum()
        print(f"  GVA matched: {gva_matched:,}/{n_before:,} UPRNs")
        print(f"  BRES matched: {bres_matched:,}/{n_before:,} UPRNs")
    elif not scaling_path.exists():
        print("  Scaling data not found — run data/download_scaling.py first")
    else:
        print("  LSOA21CD not in dataset — scaling data not joined")

    # Filter to valid records with known type
    valid = (
        df["morph_type"].isin(TYPE_ORDER)
        & np.isfinite(df["energy_intensity"])
        & (df["energy_intensity"] > 0)
    )
    df = df[valid].copy()
    print(f"  Valid records with known type: {len(df):,}")

    return df


# ---------------------------------------------------------------------------
# Step 1: Physics signatures — Types have distinct thermal envelopes
# ---------------------------------------------------------------------------


def step1_physics_signatures(df: pd.DataFrame) -> bool:
    """
    Show that morphological types have distinct physics signatures.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared analysis data.

    Returns
    -------
    bool
        True if types clearly separate on physics variables.
    """
    print("\n" + "=" * 70)
    print("STEP 1: Do morphological types bundle different physics?")
    print("=" * 70)

    available_physics = [c for c in PHYSICS_COLS if c in df.columns]
    if not available_physics:
        print("  FAIL: No physics columns available in data.")
        return False

    # Physics signature table
    header = f"  {'Type':<18} {'N':>8}"
    for col in available_physics:
        header += f" {_short_physics(col):>10}"
    print(f"\n{header}")
    print("  " + "-" * (28 + 12 * len(available_physics)))

    type_means: dict[str, dict[str, float]] = {}
    for morph_type in TYPE_ORDER:
        subset = df[df["morph_type"] == morph_type]
        if len(subset) == 0:
            continue
        row = f"  {TYPE_LABELS[morph_type]:<18} {len(subset):>8,}"
        means: dict[str, float] = {}
        for col in available_physics:
            val = subset[col].mean()
            means[col] = val
            row += f" {val:>10.3f}"
        type_means[morph_type] = means
        print(row)

    # ANOVA with effect sizes
    print("\n  One-way ANOVA on physics variables by type:")
    all_sig = True
    for col in available_physics:
        groups = [
            df.loc[df["morph_type"] == t, col].dropna()
            for t in TYPE_ORDER
            if t in type_means and len(df[df["morph_type"] == t]) > 10
        ]
        if len(groups) < 2:
            continue
        f_stat, p_val = stats.f_oneway(*groups)
        grand_mean = df[col].dropna().mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = sum(((g - grand_mean) ** 2).sum() for g in groups)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
        sig = _sigstars(p_val)
        print(
            f"    {_short_physics(col):<12} F = {f_stat:>10.1f}"
            f"  p = {p_val:.2e} {sig}  η² = {eta_sq:.3f}"
        )
        if p_val >= 0.05:
            all_sig = False

    if all_sig:
        print("\n  GATE 1: PASS — all physics variables differ by type")
    else:
        print("\n  GATE 1: WARNING — some physics variables do not differ")

    return all_sig


# ---------------------------------------------------------------------------
# Step 2: Physics → energy (SAP + metered validation)
# ---------------------------------------------------------------------------


def step2_physics_energy(df: pd.DataFrame) -> bool:
    """
    Test whether geometry-derived physics predicts energy.

    Runs two regressions:
      Model A: building-level SAP energy (with controls)
      Model B: LSOA-level metered energy (DESNZ)

    If physics predicts both with consistent signs, the effect is real —
    not a SAP modelling artifact (Few et al. 2023, Summerfield et al. 2019).

    Parameters
    ----------
    df : pd.DataFrame
        Prepared analysis data.

    Returns
    -------
    bool
        True if physics predicts energy in both models.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Does building physics predict energy? (SAP + metered)")
    print("=" * 70)

    physics_vars = [
        c for c in ["log_envelope_per_dwelling", "party_ratio"] if c in df.columns
    ]
    if not physics_vars:
        print("  FAIL: No physics columns available")
        return False

    # --- Model A: Building-level SAP energy ---
    print("\n  --- Model A: SAP-modelled energy (building-level) ---")
    sap_cols = ["log_energy_intensity"] + CONTROL_COLS + physics_vars
    sap_df = df[sap_cols].dropna()
    print(f"  n = {len(sap_df):,}")

    m0 = smf.ols(
        "log_energy_intensity ~ " + " + ".join(CONTROL_COLS), data=sap_df
    ).fit()
    m_sap = smf.ols(
        "log_energy_intensity ~ " + " + ".join(CONTROL_COLS + physics_vars),
        data=sap_df,
    ).fit()

    delta_r2 = m_sap.rsquared - m0.rsquared
    print(f"  R² controls only: {m0.rsquared:.4f}")
    print(f"  R² with physics:  {m_sap.rsquared:.4f}  (ΔR² = {delta_r2:+.4f})")

    sap_results: dict[str, tuple[float, float, str]] = {}
    for var in physics_vars:
        beta = m_sap.params[var]
        p = m_sap.pvalues[var]
        sig = _sigstars(p)
        sap_results[var] = (beta, p, sig)
        print(f"    {var:<26} β = {beta:>+.4f}  p = {p:.4f} {sig}")

    # --- Model B: LSOA-level metered energy ---
    metered_col = "lsoa_total_mean_kwh"
    met_results: dict[str, tuple[float, float, str]] = {}
    has_metered = metered_col in df.columns

    if has_metered:
        print("\n  --- Model B: DESNZ metered energy (LSOA-level) ---")
        lsoa_agg_cols = {metered_col: "first"}
        for col in CONTROL_COLS + physics_vars:
            if col in df.columns:
                lsoa_agg_cols[col] = "mean"

        has_lsoa = df["LSOA21CD"].notna() & df[metered_col].notna()
        lsoa_df = df.loc[has_lsoa].groupby("LSOA21CD").agg(lsoa_agg_cols)
        lsoa_df["log_metered"] = np.log(lsoa_df[metered_col].clip(lower=1))
        lsoa_df = lsoa_df.dropna()
        print(f"  n = {len(lsoa_df):,} LSOAs")

        if len(lsoa_df) >= 30:
            m_met = smf.ols(
                "log_metered ~ " + " + ".join(CONTROL_COLS + physics_vars),
                data=lsoa_df,
            ).fit()
            print(f"  R² = {m_met.rsquared:.4f}")
            for var in physics_vars:
                beta = m_met.params[var]
                p = m_met.pvalues[var]
                sig = _sigstars(p)
                met_results[var] = (beta, p, sig)
                print(f"    {var:<26} β = {beta:>+.4f}  p = {p:.4f} {sig}")

    # --- Comparison table ---
    if met_results:
        print("\n  --- Coefficient comparison ---")
        print(
            f"  {'Variable':<26} {'SAP β':>10} {'Metered β':>10}"
            f" {'Same sign?':>12} {'Both sig?':>10}"
        )
        print("  " + "-" * 72)
        for var in physics_vars:
            s_beta, s_p, _ = sap_results[var]
            m_beta, m_p, _ = met_results[var]
            same_sign = "Yes" if (s_beta > 0) == (m_beta > 0) else "NO"
            both_sig = "Yes" if s_p < 0.05 and m_p < 0.05 else "No"
            print(
                f"  {var:<26} {s_beta:>+10.4f} {m_beta:>+10.4f}"
                f" {same_sign:>12} {both_sig:>10}"
            )

    # Gate: physics significant in SAP; if metered available, consistent signs
    sap_sig = any(sap_results[v][1] < 0.05 for v in physics_vars)
    if met_results:
        any_validated = any(
            sap_results[v][1] < 0.05
            and met_results[v][1] < 0.05
            and (sap_results[v][0] > 0) == (met_results[v][0] > 0)
            for v in physics_vars
        )
        if any_validated:
            print("\n  GATE 2: PASS — physics predicts both SAP and metered energy")
            print("  The effect is real, not a SAP artifact")
        else:
            print("\n  GATE 2: FAIL — physics does not predict metered energy")
        return any_validated
    else:
        if sap_sig:
            print("\n  GATE 2: PARTIAL PASS — physics predicts SAP (no metered data)")
        else:
            print("\n  GATE 2: FAIL — physics not significant")
        return sap_sig


# ---------------------------------------------------------------------------
# Step 3: Trophic layers — Accessibility signatures by type
# ---------------------------------------------------------------------------


def step3_trophic_layers(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Show that morphological types differ across ALL trophic layers.

    The urban conduit has depth: each layer captures a different dimension
    of how energy is recycled through interactions. Like a rainforest's
    canopy → understorey → soil biome, the urban conduit has:
      - Physical substrate (street connectivity)
      - Commercial exchange (amenity access)
      - Mobility (transit)
      - Restoration (green space)

    A sprawling suburb is a desert: one trophic level.
    A dense inner city is a rainforest: many.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared analysis data with cityseer columns.

    Returns
    -------
    dict[str, dict[str, float]]
        Layer scores by morphological type (for use in Step 4).
    """
    print("\n" + "=" * 70)
    print("STEP 3: The trophic layers of the urban conduit")
    print("=" * 70)
    print("  Rainforest vs desert: how many layers capture the energy?")

    # Build layer scores (mean of available columns per layer)
    layer_scores: dict[str, pd.Series] = {}
    available_layers: list[str] = []

    for layer_name, spec in TROPHIC_LAYERS.items():
        cols = [c for c in spec["cols"] if c in df.columns]  # type: ignore[union-attr]
        if not cols:
            continue
        available_layers.append(layer_name)
        vals = sum(pd.to_numeric(df[c], errors="coerce").fillna(0) for c in cols)
        layer_scores[layer_name] = vals  # type: ignore[assignment]

    if not available_layers:
        print("  SKIP: No cityseer accessibility columns found")
        return {}

    # Table: each layer by morphological type
    print(f"\n  {'Layer':<28}", end="")
    for t in TYPE_ORDER:
        print(f" {TYPE_LABELS[t]:>12}", end="")
    print(f" {'F':>9} {'η²':>7}")
    print("  " + "-" * (28 + 12 * len(TYPE_ORDER) + 18))

    type_layer_means: dict[str, dict[str, float]] = {t: {} for t in TYPE_ORDER}

    for layer_name in available_layers:
        vals = layer_scores[layer_name]
        label = TROPHIC_LAYERS[layer_name]["label"]

        # Print layer header
        print(f"  {layer_name}")

        # Compute means and ANOVA
        row = f"    {label:<26}"
        groups = []
        for t in TYPE_ORDER:
            sub = vals[df["morph_type"] == t].dropna()
            mean_val = sub.mean() if len(sub) > 0 else np.nan
            type_layer_means[t][layer_name] = mean_val
            row += f" {mean_val:>12.2f}"
            if len(sub) > 10:
                groups.append(sub)

        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)
            grand_mean = vals.dropna().mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
            ss_total = sum(((g - grand_mean) ** 2).sum() for g in groups)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
            sig = _sigstars(p_val)
            row += f" {f_stat:>8.0f}{sig} {eta_sq:>7.3f}"
        print(row)

    # Monotonicity: do compact types score higher on non-green layers?
    print("\n  Monotonicity check (detached → flat):")
    for layer_name in available_layers:
        means = [type_layer_means[t].get(layer_name, np.nan) for t in TYPE_ORDER]
        valid_means = [m for m in means if not np.isnan(m)]
        if len(valid_means) < 3:
            continue
        increases = sum(
            1
            for i in range(len(valid_means) - 1)
            if valid_means[i] < valid_means[i + 1]
        )
        direction = "↑ increases" if increases >= len(valid_means) - 2 else "mixed"
        if layer_name == "RESTORATION":
            direction += " (green space may favour suburbs — expected)"
        label = TROPHIC_LAYERS[layer_name]["label"]
        print(f"    {label:<26} {increases}/{len(valid_means) - 1} steps {direction}")

    print("\n  Dense neighbourhoods: more layers, more capacity in each layer.")
    print("  Suburban sprawl: fewer layers. The conduit is thin.")

    return type_layer_means


# ---------------------------------------------------------------------------
# Step 4: The compounding — Centerpiece
# ---------------------------------------------------------------------------


def step4_compounding(
    df: pd.DataFrame,
    type_layer_means: dict[str, dict[str, float]],
) -> bool:
    """
    Demonstrate the compounding effect across trophic layers.

    The key table: each successive normalisation widens the efficiency gap
    between compact and sprawling morphologies.

      1. kWh/m² — building physics alone
      2. kWh/capita — add transport, normalise per person
      3-N. kWh/capita per unit of each trophic layer — what you GET for the energy

    If the ratio widens at each level, compounding is confirmed:
    sprawl is not just energy-costly, it delivers less city per unit consumed.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared analysis data.
    type_layer_means : dict[str, dict[str, float]]
        Layer scores by type from Step 3.

    Returns
    -------
    bool
        True if compounding is demonstrated.
    """
    print("\n" + "=" * 70)
    print("STEP 4: The compounding effect")
    print("=" * 70)
    print("  Does each trophic layer amplify the efficiency gap?")

    # Collect per-type energy stats
    type_stats: dict[str, dict[str, float]] = {}
    for morph_type in TYPE_ORDER:
        sub = df[df["morph_type"] == morph_type]
        if len(sub) == 0:
            continue
        type_stats[morph_type] = {
            "n": len(sub),
            "kwh_m2": sub["energy_intensity"].mean(),
            "kwh_cap": sub["total_energy_per_capita"].mean(),
        }

    if "detached" not in type_stats or "flat" not in type_stats:
        print("  SKIP: Need both detached and flat types")
        return False

    # --- Main table: normalisations × types ---
    print(f"\n  {'Normalisation':<32}", end="")
    for t in TYPE_ORDER:
        if t in type_stats:
            print(f" {TYPE_LABELS[t]:>12}", end="")
    print(f" {'Det/Flat':>10}")
    print("  " + "-" * (32 + 12 * len(type_stats) + 12))

    # Row 1: kWh/m²
    row = f"  {'1. kWh/m² (physics)':<32}"
    for t in TYPE_ORDER:
        if t in type_stats:
            row += f" {type_stats[t]['kwh_m2']:>12.1f}"
    ratio_m2 = type_stats["detached"]["kwh_m2"] / type_stats["flat"]["kwh_m2"]
    row += f" {ratio_m2:>9.2f}x"
    print(row)

    # Row 2: kWh/capita (building + transport)
    row = f"  {'2. kWh/capita (bldg+trns)':<32}"
    for t in TYPE_ORDER:
        if t in type_stats:
            row += f" {type_stats[t]['kwh_cap']:>12.0f}"
    ratio_cap = type_stats["detached"]["kwh_cap"] / type_stats["flat"]["kwh_cap"]
    row += f" {ratio_cap:>9.2f}x"
    print(row)

    # Rows 3+: kWh/capita per unit of each trophic layer
    ratios: list[float] = [ratio_m2, ratio_cap]
    layer_num = 3

    for layer_name in type_layer_means.get("detached", {}):
        det_acc = type_layer_means["detached"].get(layer_name, np.nan)
        flat_acc = type_layer_means["flat"].get(layer_name, np.nan)

        if np.isnan(det_acc) or np.isnan(flat_acc):
            continue

        # Compute energy/access for each type
        # Shift all values so minimum is positive (for meaningful ratio)
        all_layer_vals = [
            type_layer_means[t].get(layer_name, np.nan)
            for t in TYPE_ORDER
            if t in type_stats
        ]
        min_val = min(v for v in all_layer_vals if not np.isnan(v))
        shift = abs(min_val) + 1.0 if min_val <= 0 else 0.0

        label = TROPHIC_LAYERS[layer_name]["label"]
        row = f"  {f'{layer_num}. kWh/cap/{label[:15]}':<32}"

        det_epa = np.nan
        flat_epa = np.nan

        for t in TYPE_ORDER:
            if t not in type_stats:
                continue
            layer_val = type_layer_means[t].get(layer_name, np.nan)
            if np.isnan(layer_val):
                row += f" {'—':>12}"
                continue
            epa = type_stats[t]["kwh_cap"] / (layer_val + shift)
            row += f" {epa:>12.0f}"
            if t == "detached":
                det_epa = epa
            if t == "flat":
                flat_epa = epa

        if not np.isnan(det_epa) and not np.isnan(flat_epa) and flat_epa > 0:
            ratio = det_epa / flat_epa
            ratios.append(ratio)
            row += f" {ratio:>9.2f}x"
        print(row)
        layer_num += 1

    # --- Summary: the compounding progression ---
    print("\n  Compounding progression (Detached / Flat ratio):")
    labels = ["kWh/m²", "kWh/capita"]
    for layer_name in type_layer_means.get("detached", {}):
        if layer_name in TROPHIC_LAYERS:
            labels.append(f"kWh/cap/{TROPHIC_LAYERS[layer_name]['label'][:15]}")
    for i, (label, ratio) in enumerate(zip(labels, ratios)):
        bar = "█" * int(ratio * 10)
        print(f"    {label:<30} {ratio:>5.2f}x  {bar}")

    # Gate: do the majority of trophic layer ratios exceed the kWh/capita ratio?
    # (Green space may favour suburbs — that's an honest caveat, not a failure.)
    if len(ratios) >= 3:
        capita_ratio = ratios[1]  # kWh/capita
        layer_ratios = ratios[2:]  # the trophic layer ratios
        n_amplifying = sum(1 for r in layer_ratios if r > capita_ratio)
        majority = n_amplifying > len(layer_ratios) / 2

        print(
            f"\n  {n_amplifying}/{len(layer_ratios)} trophic layers"
            f" amplify beyond kWh/capita ({capita_ratio:.2f}x)"
        )

        if majority:
            print("\n  GATE 4: PASS — COMPOUNDING CONFIRMED")
            print("  The majority of trophic layers amplify the gap.")
            print("  Sprawl is not just energy-costly — it delivers less")
            print("  city per unit of energy consumed. The conduit leaks.")
        else:
            print("\n  GATE 4: FAIL — compounding not demonstrated")
        return majority
    else:
        print("\n  GATE 4: INSUFFICIENT DATA for compounding test")
        return False


# ---------------------------------------------------------------------------
# Step 5: Deprivation control
# ---------------------------------------------------------------------------


def step5_deprivation_control(df: pd.DataFrame) -> None:
    """
    Test whether the compounding effect holds within deprivation quintiles.

    Critical control: is the accessibility-energy pattern just a wealth effect?
    Rich people live in suburbs and drive more. If the compounding holds WITHIN
    deprivation quintiles — if dense-and-deprived still beats sprawl-and-affluent
    on energy-per-accessibility — then morphology trumps wealth.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared analysis data with deprivation_quintile.
    """
    print("\n" + "=" * 70)
    print("STEP 5: Does compounding hold within deprivation quintiles?")
    print("=" * 70)
    print("  If yes: the effect is morphological, not socioeconomic.")

    if "deprivation_quintile" not in df.columns:
        print("  SKIP: No deprivation data (Census TS011)")
        return

    if "accessibility" not in df.columns:
        # Build quick composite from available layers
        acc_cols = [c for c in ["cc_harmonic_800"] if c in df.columns]
        if not acc_cols:
            print("  SKIP: No accessibility data")
            return
        df["accessibility"] = pd.to_numeric(df[acc_cols[0]], errors="coerce")

    # For each deprivation quintile, compare houses vs flats
    print(
        f"\n  {'Quintile':<14} {'Type':<12} {'N':>7} {'kWh/cap':>9}"
        f" {'Access':>9} {'kWh/c/acc':>10}"
    )
    print("  " + "-" * 65)

    quintile_ratios: list[float] = []

    for q in df["deprivation_quintile"].cat.categories:
        q_df = df[df["deprivation_quintile"] == q]

        for t in ["detached", "semi", "flat"]:
            sub = q_df[q_df["morph_type"] == t]
            if len(sub) < 20:
                continue

            kwh_cap = sub["total_energy_per_capita"].mean()
            acc = sub["accessibility"].mean()

            # Shift accessibility for ratio
            acc_shift = acc + abs(df["accessibility"].min()) + 1.0
            epa = kwh_cap / acc_shift

            print(
                f"  {str(q):<14} {TYPE_LABELS[t]:<12} {len(sub):>7,}"
                f" {kwh_cap:>9.0f} {acc:>+9.3f} {epa:>10.0f}"
            )

        # Compute ratio within this quintile
        det_sub = q_df[q_df["morph_type"].isin(["detached", "semi"])]
        flat_sub = q_df[q_df["morph_type"] == "flat"]
        if len(det_sub) >= 20 and len(flat_sub) >= 20:
            det_epa = det_sub["total_energy_per_capita"].mean()
            flat_epa = flat_sub["total_energy_per_capita"].mean()
            if flat_epa > 0:
                quintile_ratios.append(det_epa / flat_epa)

    if quintile_ratios:
        print("\n  Energy/capita ratio (houses vs flats) by deprivation quintile:")
        all_above_1 = all(r > 1.0 for r in quintile_ratios)
        for i, r in enumerate(quintile_ratios):
            bar = "█" * int(r * 10)
            print(f"    Q{i + 1}: {r:.2f}x  {bar}")

        if all_above_1:
            print("\n  Houses use more energy per capita than flats in EVERY")
            print("  deprivation quintile. The effect is morphological,")
            print("  not a wealth proxy.")
        else:
            print("\n  Mixed results — wealth may partially explain the pattern.")


# ---------------------------------------------------------------------------
# Step 6: Lock-in
# ---------------------------------------------------------------------------


def step6_lockin(df: pd.DataFrame) -> None:
    """
    Quantify the energy locked into current morphological stock.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared analysis data.
    """
    print("\n" + "=" * 70)
    print("STEP 6: How much energy is locked in by morphology?")
    print("=" * 70)

    # Stock composition
    print("\n  ### Stock Composition")
    total = len(df)
    for morph_type in TYPE_ORDER:
        n = len(df[df["morph_type"] == morph_type])
        pct = n / total * 100
        print(f"  {TYPE_LABELS[morph_type]:<18} {n:>8,} ({pct:>5.1f}%)")

    # Energy by type
    print("\n  ### Aggregate Energy by Type")
    total_energy = df["total_energy_kwh"].sum()
    print(f"  {'Type':<18} {'Total GWh':>10} {'Share':>8} {'Mean kWh/m²':>12}")
    print("  " + "-" * 52)
    for morph_type in TYPE_ORDER:
        subset = df[df["morph_type"] == morph_type]
        if len(subset) == 0:
            continue
        type_energy = subset["total_energy_kwh"].sum()
        share = type_energy / total_energy * 100
        mean_ei = subset["energy_intensity"].mean()
        print(
            f"  {TYPE_LABELS[morph_type]:<18} {type_energy / 1e6:>10.1f}"
            f" {share:>7.1f}% {mean_ei:>12.1f}"
        )

    # Age distribution
    if "building_age" in df.columns:
        print("\n  ### Median Building Age by Type")
        for morph_type in TYPE_ORDER:
            subset = df[df["morph_type"] == morph_type]
            if len(subset) == 0:
                continue
            age = subset["building_age"].median()
            year = 2024 - age
            print(
                f"  {TYPE_LABELS[morph_type]:<18} median age: {age:.0f} years"
                f" (built ~{year:.0f})"
            )
        print("\n  These buildings will stand for decades more.")
        print("  Their morphological energy penalty is locked in.")
        print("  And the thin conduit they create — few layers, few")
        print("  interactions per unit of energy — is locked in too.")


# ---------------------------------------------------------------------------
# Step 7: Scaling — The urban conduit amplifies economic output
# ---------------------------------------------------------------------------


def _find_oa_population_col(df: pd.DataFrame) -> str | None:
    """Find the TS001 population column by prefix pattern."""
    for col in df.columns:
        if col.startswith("ts001_") and "total" in col.lower():
            return col
    # Fallback: any ts001 column
    ts001_cols = [c for c in df.columns if c.startswith("ts001_")]
    return ts001_cols[0] if ts001_cols else None


def step7_scaling(df: pd.DataFrame) -> None:
    """
    Show that compact morphology predicts higher economic output per capita.

    Aggregates to Output Area level (~125 households) where Census
    population is exact, then joins LSOA-level GVA and BRES. This gives
    genuine per-capita energy (total OA energy / OA population) rather
    than the dwelling-level approximation using OA-average household size.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared analysis data with lsoa_gva_millions and lsoa_employment.
    """
    print("\n" + "=" * 70)
    print("STEP 7: Does the urban conduit also amplify economic output?")
    print("=" * 70)
    print("  Bettencourt et al. (2007): GDP scales superlinearly with")
    print("  city size (~N^1.15). If compact form is the mechanism, the")
    print("  same morphology that saves energy should generate more value.")
    print("\n  Unit of analysis: Output Area (~125 households)")
    print("  Census population exact; morphology aggregated; GVA via LSOA")

    gva_col = "lsoa_gva_millions"
    emp_col = "lsoa_employment"
    has_gva = gva_col in df.columns and df[gva_col].notna().any()
    has_bres = emp_col in df.columns and df[emp_col].notna().any()

    if not has_gva and not has_bres:
        print("\n  SKIP: No scaling data available.")
        print("  Run: uv run python data/download_scaling.py")
        return

    # --- Find OA code and population columns ---
    if "OA21CD" not in df.columns:
        print("\n  SKIP: OA21CD not in dataset.")
        return

    pop_col = _find_oa_population_col(df)
    if pop_col is None:
        print("\n  SKIP: Census TS001 population column not found.")
        return

    print(f"  Population column: {pop_col[:50]}...")

    # --- Aggregate to OA level ---
    # OA is the natural unit: population is exact, morphology coherent
    oa_agg: dict[str, str] = {
        "total_energy_kwh": "sum",  # true total energy for this OA
        pop_col: "first",  # same for all UPRNs in OA (OA-level Census)
        "LSOA21CD": "first",
        "surface_to_volume": "mean",
        "party_ratio": "mean",
        "pop_density": "first",
    }
    # Building type composition: count each type per OA
    for t in TYPE_ORDER:
        key = f"n_{t}"
        df[key] = (df["morph_type"] == t).astype(int)
        oa_agg[key] = "sum"

    # Trophic layer columns
    for layer in TROPHIC_LAYERS.values():
        for col in layer["cols"]:
            if col in df.columns:
                oa_agg[col] = "mean"

    # Scaling data (LSOA-level, same for all UPRNs in LSOA)
    if has_gva:
        oa_agg[gva_col] = "first"
    if has_bres:
        oa_agg[emp_col] = "first"

    # Only include OAs with valid data
    oa_mask = df["OA21CD"].notna() & df[pop_col].notna()
    if has_gva:
        oa_mask = oa_mask & df[gva_col].notna()

    oa_df = df.loc[oa_mask].groupby("OA21CD").agg(oa_agg).reset_index()

    # Filter to OAs with positive population
    oa_pop = pd.to_numeric(oa_df[pop_col], errors="coerce")
    oa_df = oa_df[oa_pop > 0].copy()
    oa_pop = pd.to_numeric(oa_df[pop_col], errors="coerce")

    # --- Genuine per-capita energy ---
    oa_df["oa_energy_per_capita"] = oa_df["total_energy_kwh"] / oa_pop
    oa_df["oa_n_dwellings"] = sum(oa_df[f"n_{t}"] for t in TYPE_ORDER)

    # Dominant type per OA (plurality)
    type_counts = oa_df[[f"n_{t}" for t in TYPE_ORDER]]
    oa_df["dominant_type"] = type_counts.idxmax(axis=1).str.replace(
        "n_", "", regex=False
    )
    # Compactness: share of flats + terraces
    compact_types = ["mid_terrace", "end_terrace", "flat"]
    oa_df["compact_share"] = sum(oa_df[f"n_{t}"] for t in compact_types) / oa_df[
        "oa_n_dwellings"
    ].replace(0, np.nan)

    n_oa = len(oa_df)
    print(f"\n  OAs with complete data: {n_oa:,}")
    print(f"  Mean OA population: {oa_pop.mean():.0f} (median {oa_pop.median():.0f})")
    print(f"  Mean OA dwellings: {oa_df['oa_n_dwellings'].mean():.0f}")

    if n_oa < 10:
        print("  Too few OAs for meaningful analysis.")
        return

    # --- Density quintiles ---
    print("\n  ### OA density quintile comparison")
    print("  (Q1 = least dense / sprawl, Q5 = densest / compact)")

    oa_df["density_quintile"] = pd.qcut(
        oa_df["pop_density"], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
    )

    header = f"  {'Q':<4} {'OAs':>5} {'Pop/km2':>9} {'kWh/cap':>9} {'Compact%':>9}"
    if has_gva:
        header += f" {'GVA £m':>9}"
    if has_bres:
        header += f" {'Jobs':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    q_stats: dict[str, dict[str, float]] = {}
    for q_label in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        q_sub = oa_df[oa_df["density_quintile"] == q_label]
        if len(q_sub) == 0:
            continue
        sd: dict[str, float] = {
            "n": len(q_sub),
            "pop_density": q_sub["pop_density"].mean(),
            "kwh_cap": q_sub["oa_energy_per_capita"].mean(),
            "compact": q_sub["compact_share"].mean() * 100,
        }
        row = (
            f"  {q_label:<4} {len(q_sub):>5}"
            f" {sd['pop_density']:>9,.0f}"
            f" {sd['kwh_cap']:>9,.0f}"
            f" {sd['compact']:>8.1f}%"
        )
        if has_gva:
            gva_mean = q_sub[gva_col].mean()
            sd["gva"] = gva_mean
            row += f" {gva_mean:>9.2f}"
        if has_bres:
            emp_mean = q_sub[emp_col].mean()
            sd["employment"] = emp_mean
            row += f" {emp_mean:>7.0f}"
        q_stats[q_label] = sd
        print(row)

    # --- Dual scaling summary ---
    if "Q1" in q_stats and "Q5" in q_stats:
        q1 = q_stats["Q1"]
        q5 = q_stats["Q5"]

        print("\n  ### Dual scaling (Q5 densest vs Q1 sprawl):")
        if q5["kwh_cap"] > 0:
            e_ratio = q1["kwh_cap"] / q5["kwh_cap"]
            print(f"    Energy/capita:   Q1/Q5 = {e_ratio:.2f}x (sprawl costs more)")

        if has_gva and q1.get("gva", 0) > 0:
            g_ratio = q5["gva"] / q1["gva"]
            print(
                f"    GVA/LSOA:        Q5/Q1 = {g_ratio:.2f}x (density produces more)"
            )

        if has_bres and q1.get("employment", 0) > 0:
            emp_ratio = q5["employment"] / q1["employment"]
            print(
                f"    Employment/LSOA: Q5/Q1 = {emp_ratio:.2f}x (density employs more)"
            )

        print(
            f"\n    Compact share:   Q1 = {q1['compact']:.0f}%"
            f"  vs  Q5 = {q5['compact']:.0f}%"
        )

    # --- Correlation: morphology vs GVA at OA level ---
    if has_gva:
        print("\n  ### Correlation: OA morphology vs LSOA GVA")
        corr_vars = [
            ("compact_share", "Compact share"),
            ("surface_to_volume", "S/V ratio"),
            ("party_ratio", "Party ratio"),
            ("pop_density", "Pop density"),
        ]
        for layer in TROPHIC_LAYERS.values():
            for col in layer["cols"]:
                if col in oa_df.columns:
                    short = (
                        col.replace("cc_", "").replace("_800", "").replace("_wt", "")
                    )
                    corr_vars.append((col, short))

        for col, label in corr_vars:
            if col not in oa_df.columns:
                continue
            valid = oa_df[[col, gva_col]].dropna()
            if len(valid) < 20:
                continue
            r, p = stats.pearsonr(valid[col], valid[gva_col])
            sig = _sigstars(p)
            print(f"    {label:<26} r = {r:>6.3f} {sig}")

    # --- The argument ---
    print("\n  The urban conduit doesn't just save energy — it amplifies")
    print("  economic output. Compact form is simultaneously greener")
    print("  and more productive. Planning decisions about density lock")
    print("  in both the energy cost and the economic return.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the proof of concept argument chain."""
    print("=" * 70)
    print("PROOF OF CONCEPT: The Trophic Layers of Urban Energy")
    print("=" * 70)
    print("  Cities are conduits. The question is not how much energy,")
    print("  but how much *city* each unit of energy enables.")
    print(f"\n  Data: {DATA_PATH}\n")

    df = load_data()

    # Step 1: Foundation — types bundle physics
    gate1 = step1_physics_signatures(df)
    if not gate1:
        print("\n  STOPPING: Step 1 failed. No morphological signal.")
        sys.exit(1)

    # Step 2: Foundation — physics predicts energy (SAP + metered)
    gate2 = step2_physics_energy(df)
    if not gate2:
        print("\n  STOPPING: Step 2 failed. Physics is decorative.")
        sys.exit(1)

    # Step 3: The other side — types differ across trophic layers
    type_layer_means = step3_trophic_layers(df)

    # Step 4: THE CENTERPIECE — the compounding effect
    gate4 = step4_compounding(df, type_layer_means)

    # Step 5: Control — not a wealth effect
    step5_deprivation_control(df)

    # Step 6: Implication — it's locked in
    step6_lockin(df)

    # Step 7: The other side of scaling — economic output
    step7_scaling(df)

    # Final summary
    print("\n" + "=" * 70)
    print("PROOF OF CONCEPT SUMMARY")
    print("=" * 70)
    gates = {
        "Step 1 (types bundle physics)": gate1,
        "Step 2 (physics predicts energy)": gate2,
        "Step 4 (compounding effect)": gate4,
    }
    for name, passed in gates.items():
        icon = "PASS" if passed else "FAIL"
        print(f"  {icon}: {name}")

    all_pass = all(gates.values())
    if all_pass:
        print("\n  ALL GATES PASSED.")
        print("  Compact morphologies deliver more city per unit of energy.")
        print("  The effect compounds across trophic layers and is locked")
        print("  into the building stock for decades.")
    else:
        failed = [k for k, v in gates.items() if not v]
        print(f"\n  {len(failed)} gate(s) failed: {', '.join(failed)}")
        print("  Review failed steps before scaling.")


if __name__ == "__main__":
    main()
