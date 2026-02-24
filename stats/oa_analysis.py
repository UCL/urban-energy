"""
OA-Level Hierarchical Regression: Building Physics → Conduit Test.

Drops EPC dwelling categories entirely and works with continuous building
physics derived from LiDAR + OS footprints (surface_to_volume, shared_wall_ratio,
height_mean, etc.).

Flow: per-building physics → aggregate to OA → join Census population + LSOA
metered energy → hierarchical regression.

Each regression layer tests a distinct causal mechanism:
    1. Building physics → heat loss per m² (thermal envelope efficiency)
    2. Occupation density → floor space per person (per-capita conversion)
    3. Accessibility → what the energy buys (the conduit test)

The R² increment from layer 2 → layer 3 is the conduit test: does accessibility
explain additional variance in per-capita energy beyond physics and occupation?

Usage:
    uv run python stats/oa_analysis.py                    # all cities
    uv run python stats/oa_analysis.py canterbury          # single city
    uv run python stats/oa_analysis.py manchester york     # specific cities
"""

import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from urban_energy.paths import TEMP_DIR

DATA_PATH = TEMP_DIR / "processing" / "test" / "uprn_integrated.gpkg"

# Building-level columns to load (no EPC categories — continuous only)
_BUILDING_COLS = [
    # Identifiers
    "city",
    "OA21CD",
    "LSOA21CD",
    # Morphology (from buildings_morphology.gpkg, joined in pipeline)
    "footprint_area_m2",
    "perimeter_m",
    "volume_m3",
    "envelope_area_m2",
    "surface_to_volume",
    "form_factor",
    "height_mean",
    "shared_wall_ratio",
    "neighbors",
    "neighbor_distance",
    # Floor area and energy (from EPC — numbers only, no categories)
    "TOTAL_FLOOR_AREA",
    "ENERGY_CONSUMPTION_CURRENT",
]

# Census column names (OA-level)
_TS001_POP = "ts001_Residence type: Lives in a household; measures: Value"
_TS006_DENSITY = (
    "ts006_Population Density: Persons per square kilometre; measures: Value"
)
_TS011_NOT_DEP = (
    "ts011_Household deprivation: "
    "Household is not deprived in any dimension; measures: Value"
)
_TS011_TOTAL = (
    "ts011_Household deprivation: Total: All households; measures: Value"
)
_TS017_TOTAL = "ts017_Household size: Total: All household spaces; measures: Value"
_TS017_ZERO = "ts017_Household size: 0 people in household; measures: Value"

# Household size band columns {n_people: column_name}
_TS017_BANDS: dict[int, str] = {
    n: f"ts017_Household size: {n} {'person' if n == 1 else 'people'}"
    f" in household; measures: Value"
    for n in range(1, 8)
}
_TS017_BANDS[8] = (
    "ts017_Household size: 8 or more people in household; measures: Value"
)

# Car ownership columns
_CAR_PFX = "ts045_Number of cars or vans: "
_TS045_CARS: dict[int, str] = {
    0: f"{_CAR_PFX}No cars or vans in household",
    1: f"{_CAR_PFX}1 car or van in household",
    2: f"{_CAR_PFX}2 cars or vans in household",
    3: f"{_CAR_PFX}3 or more cars or vans in household",
}

# Physics predictors (continuous, geometry-derived)
PHYSICS_VARS = ["surface_to_volume", "shared_wall_ratio", "height_mean", "form_factor"]


def _sigstars(p: float) -> str:
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# 1. Load per-building data
# ---------------------------------------------------------------------------


def load_buildings(cities: list[str] | None = None) -> pd.DataFrame:
    """
    Load UPRN-level data with targeted building physics columns.

    Parameters
    ----------
    cities : list[str] or None
        If provided, filter to these cities. If None, load all.

    Returns
    -------
    pd.DataFrame
        Per-building data with continuous physics, energy, and census columns.
    """
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    # Probe available columns (avoids fiona truncation bug)
    _probe = gpd.read_file(DATA_PATH, rows=1)
    available = set(_probe.columns)
    del _probe

    # Build column list: explicit + all cc_*, ts*, lsoa_* found
    want = list(_BUILDING_COLS)
    want.extend([_TS001_POP, _TS006_DENSITY, _TS011_NOT_DEP, _TS011_TOTAL])
    want.extend([_TS017_TOTAL, _TS017_ZERO])
    want.extend(_TS017_BANDS.values())
    want.extend(_TS045_CARS.values())

    # Add all cc_* and lsoa_* columns from the data
    for col in sorted(available):
        if col.startswith("cc_") or col.startswith("lsoa_"):
            if col not in want:
                want.append(col)

    load_cols = [c for c in want if c in available]
    skipped = set(want) - set(load_cols)
    if skipped:
        print(f"  Note: {len(skipped)} requested columns not in data")
        for c in sorted(skipped):
            if c.startswith("cc_") or c.startswith("ts0"):
                print(f"    MISSING: {c}")

    col_list = ", ".join(f'"{c}"' for c in load_cols)
    sql = f"SELECT {col_list} FROM uprn_integrated"
    df = gpd.read_file(DATA_PATH, sql=sql)
    print(f"  Loaded {len(df):,} UPRNs ({len(load_cols)} columns)")

    # Filter to specific cities
    if cities and "city" in df.columns:
        df = df[df["city"].isin(cities)].copy()
        print(f"  Filtered to {cities}: {len(df):,}")

    # Filter to valid energy + floor area
    df = df[
        df["ENERGY_CONSUMPTION_CURRENT"].notna()
        & df["TOTAL_FLOOR_AREA"].notna()
        & (df["TOTAL_FLOOR_AREA"] > 0)
        & (df["ENERGY_CONSUMPTION_CURRENT"] > 0)
    ].copy()
    print(f"  With valid EPC energy + floor area: {len(df):,}")

    # Coerce numeric columns
    for col in PHYSICS_VARS + [
        "footprint_area_m2",
        "perimeter_m",
        "volume_m3",
        "envelope_area_m2",
        "TOTAL_FLOOR_AREA",
        "ENERGY_CONSUMPTION_CURRENT",
        "neighbors",
        "neighbor_distance",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derive per-building energy
    df["total_energy_kwh"] = (
        df["ENERGY_CONSUMPTION_CURRENT"] * df["TOTAL_FLOOR_AREA"]
    )

    # Check cityseer coverage
    cc_cols = [c for c in df.columns if c.startswith("cc_")]
    if cc_cols:
        cc_coverage = df[cc_cols[0]].notna().mean()
        print(f"  Cityseer coverage: {cc_coverage:.1%} ({len(cc_cols)} cc_ columns)")
    else:
        print("  WARNING: No cityseer columns found")

    return df


# ---------------------------------------------------------------------------
# 2. Aggregate to OA
# ---------------------------------------------------------------------------


def aggregate_to_oa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-building data to Output Area level.

    Parameters
    ----------
    df : pd.DataFrame
        Per-building data from ``load_buildings()``.

    Returns
    -------
    pd.DataFrame
        OA-level dataset with one row per Output Area.
    """
    if "OA21CD" not in df.columns:
        raise ValueError("OA21CD column required for OA aggregation")

    print("\nAggregating to Output Area level...")

    # --- Aggregation rules ---
    agg: dict[str, str] = {
        # Totals (SUM — genuine OA stock)
        "total_energy_kwh": "sum",
        "TOTAL_FLOOR_AREA": "sum",
        "footprint_area_m2": "sum",
        "volume_m3": "sum",
        "envelope_area_m2": "sum",
        # Building physics (MEAN — average character)
        "surface_to_volume": "mean",
        "form_factor": "mean",
        "height_mean": "mean",
        "shared_wall_ratio": "mean",
        "neighbors": "mean",
        "neighbor_distance": "mean",
        # Identifiers
        "LSOA21CD": "first",
    }

    if "city" in df.columns:
        agg["city"] = "first"

    # Cityseer columns (MEAN — average accessibility)
    for col in df.columns:
        if col.startswith("cc_"):
            agg[col] = "mean"

    # Census columns (FIRST — already OA-level)
    for col in df.columns:
        if col.startswith("ts0") and col not in agg:
            agg[col] = "first"

    # LSOA metered energy (FIRST)
    for col in df.columns:
        if col.startswith("lsoa_") and col not in agg:
            agg[col] = "first"

    # Only include columns that exist
    agg = {k: v for k, v in agg.items() if k in df.columns}

    sizes = df.groupby("OA21CD").size().reset_index(name="n_buildings")
    oa_df = df.groupby("OA21CD").agg(agg).reset_index()
    oa_df = oa_df.merge(sizes, on="OA21CD")
    oa_df = oa_df.copy()  # defragment

    # --- Derive Census variables ---
    # Population
    if _TS001_POP in oa_df.columns:
        oa_df["oa_total_people"] = pd.to_numeric(
            oa_df[_TS001_POP], errors="coerce"
        )
    else:
        # Fall back to household size bands
        total_people = sum(
            size * pd.to_numeric(oa_df[col], errors="coerce")
            for size, col in _TS017_BANDS.items()
            if col in oa_df.columns
        )
        oa_df["oa_total_people"] = total_people

    # Total occupied households
    if _TS017_TOTAL in oa_df.columns and _TS017_ZERO in oa_df.columns:
        oa_df["oa_total_hh"] = pd.to_numeric(
            oa_df[_TS017_TOTAL], errors="coerce"
        ) - pd.to_numeric(oa_df[_TS017_ZERO], errors="coerce")
    else:
        oa_df["oa_total_hh"] = np.nan

    # Average household size
    if _TS017_TOTAL in oa_df.columns:
        total_people = sum(
            size * pd.to_numeric(oa_df[col], errors="coerce")
            for size, col in _TS017_BANDS.items()
            if col in oa_df.columns
        )
        oa_df["avg_household_size"] = total_people / oa_df["oa_total_hh"]

    # Filter: need valid population
    oa_df = oa_df[
        (oa_df["oa_total_people"] > 10) & (oa_df["n_buildings"] >= 5)
    ].copy()

    # --- Derived energy variables ---
    oa_df["oa_energy_intensity"] = (
        oa_df["total_energy_kwh"]
        / oa_df["TOTAL_FLOOR_AREA"].replace(0, np.nan)
    )
    oa_df["log_energy_intensity"] = np.log(
        oa_df["oa_energy_intensity"].clip(lower=0.1)
    )
    oa_df["energy_per_capita"] = (
        oa_df["total_energy_kwh"] / oa_df["oa_total_people"]
    )
    oa_df["log_energy_per_capita"] = np.log(
        oa_df["energy_per_capita"].clip(lower=0.1)
    )
    oa_df["floor_area_per_capita"] = (
        oa_df["TOTAL_FLOOR_AREA"] / oa_df["oa_total_people"]
    )
    oa_df["log_n_buildings"] = np.log(oa_df["n_buildings"].clip(lower=1))

    # Transport energy
    has_cars = all(c in oa_df.columns for c in _TS045_CARS.values())
    if has_cars:
        total_cars = sum(
            n * pd.to_numeric(oa_df[col], errors="coerce")
            for n, col in _TS045_CARS.items()
        )
        total_hh_cars = sum(
            pd.to_numeric(oa_df[col], errors="coerce")
            for col in _TS045_CARS.values()
        )
        oa_df["avg_cars_per_hh"] = total_cars / total_hh_cars.replace(0, np.nan)
        kwh_per_vehicle = 12_000 * (0.17 / 0.233)
        oa_df["transport_kwh_per_hh"] = oa_df["avg_cars_per_hh"] * kwh_per_vehicle
        oa_df["total_energy_per_capita"] = (
            oa_df["total_energy_kwh"]
            + oa_df["transport_kwh_per_hh"] * oa_df["oa_total_hh"]
        ) / oa_df["oa_total_people"]
    else:
        oa_df["total_energy_per_capita"] = oa_df["energy_per_capita"]

    # Deprivation
    if _TS011_NOT_DEP in oa_df.columns and _TS011_TOTAL in oa_df.columns:
        oa_df["pct_not_deprived"] = (
            pd.to_numeric(oa_df[_TS011_NOT_DEP], errors="coerce")
            / pd.to_numeric(oa_df[_TS011_TOTAL], errors="coerce")
            * 100
        )
        valid_dep = oa_df["pct_not_deprived"].notna()
        if valid_dep.sum() > 10:
            dep_labels = ["Q1 most", "Q2", "Q3", "Q4", "Q5 least"]
            oa_df.loc[valid_dep, "deprivation_quintile"] = pd.qcut(
                oa_df.loc[valid_dep, "pct_not_deprived"],
                q=5,
                labels=dep_labels,
            )

    # Population density
    if _TS006_DENSITY in oa_df.columns:
        oa_df["pop_density"] = pd.to_numeric(
            oa_df[_TS006_DENSITY], errors="coerce"
        )

    n_cities = oa_df["city"].nunique() if "city" in oa_df.columns else 1
    print(f"  {len(oa_df):,} Output Areas across {n_cities} cities")
    print(
        f"  Buildings/OA: median={oa_df['n_buildings'].median():.0f}, "
        f"mean={oa_df['n_buildings'].mean():.0f}"
    )
    print(
        f"  Energy/capita: median={oa_df['energy_per_capita'].median():.0f} kWh, "
        f"mean={oa_df['energy_per_capita'].mean():.0f} kWh"
    )

    return oa_df


# ---------------------------------------------------------------------------
# 3. Accessibility PCA
# ---------------------------------------------------------------------------


def build_accessibility_pca(
    oa_df: pd.DataFrame,
) -> tuple[pd.DataFrame, PCA, list[str]]:
    """
    Reduce cityseer accessibility columns to principal components.

    Parameters
    ----------
    oa_df : pd.DataFrame
        OA-level data with cc_*_wt columns.

    Returns
    -------
    tuple[pd.DataFrame, PCA, list[str]]
        Updated DataFrame, fitted PCA object, and list of input column names.
    """
    print("\n" + "=" * 70)
    print("ACCESSIBILITY PCA")
    print("=" * 70)

    # Select 800m gravity-weighted accessibility columns (genuine land-use/transit)
    # Exclude building stat aggregations (footprint, height, volume, etc.)
    _building_stat_prefixes = (
        "cc_footprint_", "cc_gross_floor_", "cc_height_", "cc_volume_",
        "cc_is_detached", "cc_is_semi", "cc_is_terraced",
        "cc_height_pixel_count",
    )
    acc_cols = sorted(
        c for c in oa_df.columns
        if c.endswith("_800_wt")
        and c.startswith("cc_")
        and not any(c.startswith(p) for p in _building_stat_prefixes)
    )
    if not acc_cols:
        print("  No _800_wt columns found — skipping PCA")
        return oa_df, None, []

    print(f"  Input columns ({len(acc_cols)}):")
    for c in acc_cols:
        short = c.replace("cc_", "").replace("_800_wt", "")
        print(f"    {short}")

    # Standardise and fit PCA
    valid_mask = oa_df[acc_cols].notna().all(axis=1)
    n_valid = valid_mask.sum()
    print(f"  OAs with complete accessibility data: {n_valid:,}/{len(oa_df):,}")

    n_components = min(3, len(acc_cols))
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    X = oa_df.loc[valid_mask, acc_cols].values
    X_scaled = scaler.fit_transform(X)
    components = pca.fit_transform(X_scaled)

    for i in range(n_components):
        col_name = f"accessibility_pc{i + 1}"
        oa_df[col_name] = np.nan
        oa_df.loc[valid_mask, col_name] = components[:, i]

    # Report variance explained and top loadings
    print(f"\n  Variance explained:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"    PC{i + 1}: {var:.1%}")
        # Top 3 loadings
        loadings = pca.components_[i]
        top_idx = np.argsort(np.abs(loadings))[::-1][:3]
        for j in top_idx:
            short = acc_cols[j].replace("cc_", "").replace("_800_wt", "")
            print(f"      {short}: {loadings[j]:+.3f}")
    print(f"    Total: {pca.explained_variance_ratio_.sum():.1%}")

    return oa_df, pca, acc_cols


# ---------------------------------------------------------------------------
# 4. Hierarchical regression
# ---------------------------------------------------------------------------


def _run_ols(
    oa_df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    label: str,
) -> sm.regression.linear_model.RegressionResultsWrapper | None:
    """
    Run OLS with HC3 robust standard errors.

    Parameters
    ----------
    oa_df : pd.DataFrame
        OA-level data.
    y_col : str
        Dependent variable column.
    x_cols : list[str]
        Predictor columns.
    label : str
        Model label for printing.

    Returns
    -------
    RegressionResultsWrapper or None
        Fitted model, or None if insufficient data.
    """
    cols = [y_col] + x_cols
    sub = oa_df[cols].dropna()
    if len(sub) < len(x_cols) + 10:
        print(f"  {label}: insufficient data (N={len(sub)})")
        return None

    y = sub[y_col]
    X = sm.add_constant(sub[x_cols])
    model = sm.OLS(y, X).fit(cov_type="HC3")
    return model


def run_hierarchical_regression(
    oa_df: pd.DataFrame,
    dv: str = "log_energy_per_capita",
    label_prefix: str = "",
) -> dict[str, sm.regression.linear_model.RegressionResultsWrapper | None]:
    """
    Run the four-model hierarchical regression sequence.

    Parameters
    ----------
    oa_df : pd.DataFrame
        OA-level data with physics, occupation, and accessibility variables.
    dv : str
        Dependent variable for M1–M4.
    label_prefix : str
        Prefix for model labels (e.g. city name).

    Returns
    -------
    dict mapping model name to fitted result.
    """
    pfx = f"{label_prefix} " if label_prefix else ""

    # Check which physics vars are available
    physics = [v for v in PHYSICS_VARS if v in oa_df.columns]
    if not physics:
        print(f"  {pfx}No physics variables available")
        return {}

    # M0: Physics → energy/m² (the building envelope test)
    m0 = _run_ols(oa_df, "log_energy_intensity", physics, f"{pfx}M0")

    # M1: Physics → energy/capita (shifting to per-capita)
    m1 = _run_ols(oa_df, dv, physics, f"{pfx}M1")

    # M2: + Occupation density
    occupation_vars = []
    for v in ["floor_area_per_capita", "avg_household_size", "log_n_buildings"]:
        if v in oa_df.columns and oa_df[v].notna().sum() > 100:
            occupation_vars.append(v)
    m2 = _run_ols(oa_df, dv, physics + occupation_vars, f"{pfx}M2")

    # M3: + Accessibility (the conduit test)
    acc_vars = []
    for v in ["accessibility_pc1", "cc_harmonic_800", "cc_density_800"]:
        if v in oa_df.columns and oa_df[v].notna().sum() > 100:
            acc_vars.append(v)
    m3 = _run_ols(oa_df, dv, physics + occupation_vars + acc_vars, f"{pfx}M3")

    # M4: + Interaction (compactness × accessibility)
    interaction_vars = []
    if (
        "shared_wall_ratio" in oa_df.columns
        and "accessibility_pc1" in oa_df.columns
    ):
        oa_df["swr_x_acc"] = oa_df["shared_wall_ratio"] * oa_df["accessibility_pc1"]
        interaction_vars = ["swr_x_acc"]
    m4 = _run_ols(
        oa_df,
        dv,
        physics + occupation_vars + acc_vars + interaction_vars,
        f"{pfx}M4",
    )

    return {"M0": m0, "M1": m1, "M2": m2, "M3": m3, "M4": m4}


def print_model_sequence(
    models: dict[str, sm.regression.linear_model.RegressionResultsWrapper | None],
    label: str = "",
) -> None:
    """
    Print the hierarchical model comparison table.

    Parameters
    ----------
    models : dict
        Model name → fitted result.
    label : str
        Section header.
    """
    header = f"MODEL SEQUENCE{f' — {label}' if label else ''}"
    print(f"\n{'=' * 70}")
    print(header)
    print("=" * 70)

    labels = {
        "M0": "Physics → kWh/m²",
        "M1": "Physics → kWh/capita",
        "M2": "+ Occupation",
        "M3": "+ Accessibility   ← conduit test",
        "M4": "+ Interaction",
    }

    prev_r2 = None
    for name in ["M0", "M1", "M2", "M3", "M4"]:
        m = models.get(name)
        if m is None:
            print(f"  {name} ({labels.get(name, '')}): SKIPPED")
            prev_r2 = None
            continue

        r2 = m.rsquared
        adj_r2 = m.rsquared_adj
        aic = m.aic
        n = int(m.nobs)

        delta = ""
        if prev_r2 is not None and name != "M0":
            dr2 = r2 - prev_r2
            delta = f"  ΔR² = {dr2:+.4f}"

        print(
            f"  {name} ({labels.get(name, '')}):  "
            f"R² = {r2:.4f}  adj = {adj_r2:.4f}  AIC = {aic:.0f}  N = {n:,}{delta}"
        )
        prev_r2 = r2

    # Print M3 coefficient table (the key model)
    m3 = models.get("M3")
    if m3 is not None:
        print(f"\n  --- Coefficients (M3) ---")
        print(f"  {'Variable':<30s} {'β':>8s} {'SE':>8s} {'t':>8s} {'p':>10s}")
        print(f"  {'-' * 66}")
        for var in m3.params.index:
            if var == "const":
                continue
            b = m3.params[var]
            se = m3.bse[var]
            t = m3.tvalues[var]
            p = m3.pvalues[var]
            print(
                f"  {var:<30s} {b:>8.4f} {se:>8.4f} {t:>8.2f} "
                f"{p:>8.1e} {_sigstars(p)}"
            )

    # F-test for accessibility increment (M2 → M3)
    m2 = models.get("M2")
    if m2 is not None and m3 is not None:
        dr2 = m3.rsquared - m2.rsquared
        df_num = m3.df_model - m2.df_model
        if df_num > 0:
            f_stat = (dr2 / df_num) / ((1 - m3.rsquared) / m3.df_resid)
            f_p = 1 - sp_stats.f.cdf(f_stat, df_num, m3.df_resid)
            print(
                f"\n  Accessibility increment: ΔR² = {dr2:.4f}, "
                f"F({df_num:.0f},{m3.df_resid:.0f}) = {f_stat:.2f}, "
                f"p = {f_p:.1e} {_sigstars(f_p)}"
            )
            if dr2 > 0 and f_p < 0.05:
                print("  ✓ CONDUIT TEST PASSES: accessibility adds explanatory power")
            else:
                print("  ✗ Conduit test fails: accessibility does not add power")


# ---------------------------------------------------------------------------
# 5. Variance decomposition
# ---------------------------------------------------------------------------


def print_variance_decomposition(
    models: dict[str, sm.regression.linear_model.RegressionResultsWrapper | None],
) -> None:
    """Print sequential R² decomposition."""
    print(f"\n{'=' * 70}")
    print("VARIANCE DECOMPOSITION")
    print("=" * 70)

    r2 = {}
    for name in ["M0", "M1", "M2", "M3", "M4"]:
        m = models.get(name)
        if m is not None:
            r2[name] = m.rsquared

    if "M1" not in r2:
        print("  Insufficient models for decomposition")
        return

    shares = {}
    shares["Physics (M1)"] = r2.get("M1", 0)
    if "M2" in r2:
        shares["Occupation (M2-M1)"] = r2["M2"] - r2["M1"]
    if "M3" in r2 and "M2" in r2:
        shares["Accessibility (M3-M2)"] = r2["M3"] - r2["M2"]
    if "M4" in r2 and "M3" in r2:
        shares["Interaction (M4-M3)"] = r2["M4"] - r2["M3"]

    total = sum(shares.values())
    print(f"\n  {'Component':<30s} {'ΔR²':>8s} {'Share':>8s}")
    print(f"  {'-' * 48}")
    for component, val in shares.items():
        pct = val / total * 100 if total > 0 else 0
        bar = "█" * max(1, int(pct / 2))
        print(f"  {component:<30s} {val:>8.4f} {pct:>6.1f}%  {bar}")
    print(f"  {'Total':<30s} {total:>8.4f} {'100.0':>6s}%")


# ---------------------------------------------------------------------------
# 6. Deprivation control
# ---------------------------------------------------------------------------


def run_deprivation_control(oa_df: pd.DataFrame) -> None:
    """Re-run M3 within each deprivation quintile."""
    print(f"\n{'=' * 70}")
    print("DEPRIVATION CONTROL")
    print("=" * 70)

    if "deprivation_quintile" not in oa_df.columns:
        print("  No deprivation data available")
        return

    # Check which accessibility vars are available
    acc_vars = []
    for v in ["accessibility_pc1", "cc_harmonic_800", "cc_density_800"]:
        if v in oa_df.columns and oa_df[v].notna().sum() > 100:
            acc_vars.append(v)

    occupation_vars = []
    for v in ["floor_area_per_capita", "avg_household_size", "log_n_buildings"]:
        if v in oa_df.columns and oa_df[v].notna().sum() > 100:
            occupation_vars.append(v)

    physics = [v for v in PHYSICS_VARS if v in oa_df.columns]
    all_x = physics + occupation_vars + acc_vars

    quintiles = ["Q1 most", "Q2", "Q3", "Q4", "Q5 least"]
    print(f"\n  {'Quintile':<12s} {'N':>6s} {'R²':>8s} {'acc_β':>8s} {'p':>10s}")
    print(f"  {'-' * 50}")

    n_sig = 0
    for q in quintiles:
        sub = oa_df[oa_df["deprivation_quintile"] == q].copy()
        if len(sub) < len(all_x) + 20:
            print(f"  {q:<12s} {len(sub):>6d}  insufficient data")
            continue

        m = _run_ols(sub, "log_energy_per_capita", all_x, f"dep-{q}")
        if m is None:
            continue

        acc_beta = np.nan
        acc_p = np.nan
        if "accessibility_pc1" in m.params.index:
            acc_beta = m.params["accessibility_pc1"]
            acc_p = m.pvalues["accessibility_pc1"]
            if acc_p < 0.05:
                n_sig += 1

        print(
            f"  {q:<12s} {int(m.nobs):>6d} {m.rsquared:>8.4f} "
            f"{acc_beta:>8.4f} {acc_p:>8.1e} {_sigstars(acc_p)}"
        )

    if n_sig >= 4:
        print(f"\n  ✓ Accessibility significant in {n_sig}/5 quintiles")
        print("    Effect is not confounded by deprivation")
    else:
        print(f"\n  Accessibility significant in only {n_sig}/5 quintiles")


# ---------------------------------------------------------------------------
# 7. Per-city breakdown
# ---------------------------------------------------------------------------


def run_per_city(oa_df: pd.DataFrame) -> None:
    """Run M0–M3 for each city separately."""
    print(f"\n{'=' * 70}")
    print("PER-CITY BREAKDOWN")
    print("=" * 70)

    if "city" not in oa_df.columns:
        print("  No city column — skipping")
        return

    cities = sorted(oa_df["city"].unique())
    print(
        f"\n  {'City':<16s} {'N':>6s} {'M0_R²':>8s} {'M3_R²':>8s} "
        f"{'acc_β':>8s} {'acc_p':>10s}"
    )
    print(f"  {'-' * 62}")

    for city in cities:
        sub = oa_df[oa_df["city"] == city].copy()
        models = run_hierarchical_regression(sub, label_prefix=city)

        m0_r2 = models["M0"].rsquared if models.get("M0") else np.nan
        m3 = models.get("M3")
        m3_r2 = m3.rsquared if m3 else np.nan
        acc_beta = np.nan
        acc_p = np.nan
        if m3 is not None and "accessibility_pc1" in m3.params.index:
            acc_beta = m3.params["accessibility_pc1"]
            acc_p = m3.pvalues["accessibility_pc1"]

        print(
            f"  {city:<16s} {len(sub):>6d} {m0_r2:>8.4f} {m3_r2:>8.4f} "
            f"{acc_beta:>8.4f} {acc_p:>8.1e} {_sigstars(acc_p)}"
        )


# ---------------------------------------------------------------------------
# 8. Metered energy validation
# ---------------------------------------------------------------------------


def run_metered_validation(oa_df: pd.DataFrame) -> None:
    """Repeat M0–M3 using LSOA metered energy instead of SAP."""
    print(f"\n{'=' * 70}")
    print("METERED ENERGY VALIDATION")
    print("=" * 70)

    if "lsoa_total_mean_kwh" not in oa_df.columns:
        print("  No metered energy data (lsoa_total_mean_kwh) — skipping")
        return

    # Create metered per-capita DV
    # lsoa_total_mean_kwh is mean consumption per meter in the LSOA
    # Approximate OA energy from LSOA mean × n_buildings (rough proxy)
    metered = pd.to_numeric(oa_df["lsoa_total_mean_kwh"], errors="coerce")
    valid = metered.notna() & (metered > 0)
    if valid.sum() < 100:
        print(f"  Only {valid.sum()} OAs with metered data — skipping")
        return

    # Use log of metered mean kWh as DV (LSOA-level, so ecological)
    oa_df["log_metered_mean"] = np.log(metered.clip(lower=0.1))

    print(f"  OAs with metered data: {valid.sum():,}")
    print(f"  Metered mean kWh: median={metered[valid].median():.0f}")

    models_sap = run_hierarchical_regression(oa_df, dv="log_energy_per_capita")
    models_met = run_hierarchical_regression(oa_df, dv="log_metered_mean")

    # Compare
    print(f"\n  {'Metric':<20s} {'R²':>8s} {'acc_β':>8s} {'acc_p':>10s}")
    print(f"  {'-' * 50}")

    for label, models in [("SAP (kWh/cap)", models_sap), ("Metered (mean)", models_met)]:
        m3 = models.get("M3")
        if m3 is None:
            print(f"  {label:<20s}  M3 failed")
            continue
        acc_beta = np.nan
        acc_p = np.nan
        if "accessibility_pc1" in m3.params.index:
            acc_beta = m3.params["accessibility_pc1"]
            acc_p = m3.pvalues["accessibility_pc1"]
        print(
            f"  {label:<20s} {m3.rsquared:>8.4f} {acc_beta:>8.4f} "
            f"{acc_p:>8.1e} {_sigstars(acc_p)}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cities: list[str] | None = None) -> None:
    """
    Run the full OA-level hierarchical regression analysis.

    Parameters
    ----------
    cities : list[str] or None
        Filter to specific cities, or None for all.
    """
    print("=" * 70)
    print("OA-LEVEL HIERARCHICAL REGRESSION")
    print("Building Physics → Occupation → Accessibility (Conduit Test)")
    print("=" * 70)

    # 1. Load per-building data
    df = load_buildings(cities)

    # 2. Aggregate to OA
    oa_df = aggregate_to_oa(df)
    del df  # free memory

    # 3. PCA on accessibility
    oa_df, pca, acc_cols = build_accessibility_pca(oa_df)

    # 4. Hierarchical regression
    models = run_hierarchical_regression(oa_df)
    print_model_sequence(models)

    # 5. Variance decomposition
    print_variance_decomposition(models)

    # 6. Deprivation control
    run_deprivation_control(oa_df)

    # 7. Per-city breakdown
    run_per_city(oa_df)

    # 8. Metered validation
    run_metered_validation(oa_df)

    print(f"\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    _cities = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(cities=_cities or None)
