"""
Replication of key findings from foundational urban form-energy studies.

Replicates three landmark results using observed EPC data across England:

1. Rode et al. (2014): S/V ratio and morphology → heat-energy demand
   - S/V ratio vs energy intensity relationship
   - FAR convergence at high density
   - Building typology energy performance hierarchy

2. Norman et al. (2006): Functional unit sensitivity
   - Per-capita vs per-m² normalisation by density quintile
   - Demonstrates how functional unit choice reverses conclusions

3. Mediation test: Density → building form → energy
   - Tests whether density has a residual effect after thermal physics controls
   - Progressive attenuation of the density coefficient

Usage:
    uv run python stats/07_replication_analysis.py
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from urban_energy.paths import TEMP_DIR

DATA_PATH = TEMP_DIR / "processing" / "test" / "uprn_integrated.gpkg"
OUTPUT_DIR = TEMP_DIR / "stats" / "results"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_and_prepare_data() -> pd.DataFrame:
    """
    Load UPRN-integrated data and compute analysis variables.

    Returns
    -------
    pd.DataFrame
        Prepared dataframe with energy metrics, morphology, and controls.
    """
    print("Loading data...")
    gdf = gpd.read_file(DATA_PATH)
    print(f"  Total UPRNs: {len(gdf):,}")

    # Filter to records with EPC energy data and valid floor area
    df = gdf[gdf["ENERGY_CONSUMPTION_CURRENT"].notna()].copy()
    valid_area = (df["TOTAL_FLOOR_AREA"] > 0) & df["TOTAL_FLOOR_AREA"].notna()
    df = df[valid_area].copy()
    print(f"  With EPC energy + valid floor area: {len(df):,}")

    # --- Energy metrics ---
    # ENERGY_CONSUMPTION_CURRENT is already kWh/m²/year (SAP energy intensity)
    df["energy_intensity"] = df["ENERGY_CONSUMPTION_CURRENT"]
    # Total energy for aggregate / per-capita calculations
    df["total_energy_kwh"] = df["ENERGY_CONSUMPTION_CURRENT"] * df["TOTAL_FLOOR_AREA"]

    # Per-capita energy via Census household size
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
    # Per-capita uses total kWh (intensity × area), not intensity
    df["energy_per_capita"] = df["total_energy_kwh"] / df["avg_household_size"]

    # --- Population density ---
    df["pop_density"] = df[
        "ts006_Population Density: Persons per square kilometre; measures: Value"
    ]

    # --- Building type ---
    df["is_flat"] = df["PROPERTY_TYPE"].str.lower().str.contains("flat", na=False)
    built_form_map = {
        "Detached": "detached",
        "Semi-Detached": "semi",
        "Mid-Terrace": "mid_terrace",
        "End-Terrace": "end_terrace",
        "Enclosed Mid-Terrace": "mid_terrace",
        "Enclosed End-Terrace": "end_terrace",
    }
    df["attached_type"] = df["BUILT_FORM"].map(built_form_map)
    df.loc[df["is_flat"], "attached_type"] = "flat"
    df["attached_type"] = df["attached_type"].fillna("other")

    # --- Construction age ---
    age_band_to_year: dict[str, int] = {
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
    df["construction_year"] = df["CONSTRUCTION_AGE_BAND"].map(age_band_to_year)
    df["building_age"] = 2024 - df["construction_year"]

    # --- Log transforms ---
    df["log_energy_intensity"] = np.log(df["energy_intensity"].clip(lower=1))
    df["log_energy_per_capita"] = np.log(df["energy_per_capita"].clip(lower=1))
    df["log_floor_area"] = np.log(df["TOTAL_FLOOR_AREA"])
    df["log_pop_density"] = np.log(df["pop_density"].clip(lower=1))

    # --- Building height ---
    if "height_mean" in df.columns:
        df["building_height"] = pd.to_numeric(df["height_mean"], errors="coerce")

    # --- Morphology (from process_morphology.py) ---
    for col in ["surface_to_volume", "form_factor", "shared_wall_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Filter valid records ---
    valid = (
        np.isfinite(df["energy_intensity"])
        & np.isfinite(df["energy_per_capita"])
        & (df["energy_intensity"] > 0)
        & (df["energy_per_capita"] > 0)
    )
    df = df[valid].copy()
    print(f"  Valid records: {len(df):,}")

    return df


# ---------------------------------------------------------------------------
# Replication 1: Rode et al. (2014)
# ---------------------------------------------------------------------------


def rode_sv_energy_relationship(df: pd.DataFrame) -> None:
    """
    Replicate Rode et al.'s S/V ratio → energy intensity relationship.

    Rode et al. reported:
    - S/V ≈ 0.15: energy intensity 35–80 kWh/m²/a
    - S/V ≈ 0.40: energy intensity 110–200 kWh/m²/a
    - Up to 6× variation from morphology alone

    We test this with observed EPC data across English buildings.
    """
    print("\n" + "=" * 70)
    print("REPLICATION 1: Rode et al. (2014)")
    print("S/V Ratio → Energy Intensity Relationship")
    print("=" * 70)

    if "surface_to_volume" not in df.columns:
        print("  WARNING: surface_to_volume column not available.")
        print("  Re-run process_morphology.py and test_pipeline.py first.")
        return

    sv_df = df[df["surface_to_volume"].notna()].copy()
    print(f"\n  Records with S/V data: {len(sv_df):,}")

    # --- Overall S/V vs energy intensity ---
    r, p = stats.pearsonr(sv_df["surface_to_volume"], sv_df["energy_intensity"])
    print(f"\n  Pearson r(S/V, energy_intensity): {r:+.4f} (p = {p:.2e})")

    r_log, p_log = stats.pearsonr(
        sv_df["surface_to_volume"], sv_df["log_energy_intensity"]
    )
    print(f"  Pearson r(S/V, log_energy_intensity): {r_log:+.4f} (p = {p_log:.2e})")

    # --- S/V bins matching Rode et al.'s reported ranges ---
    sv_bins = [0, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 1.0, 5.0]
    sv_df["sv_bin"] = pd.cut(sv_df["surface_to_volume"], bins=sv_bins)

    print("\n  ### Energy Intensity by S/V Bin")
    print(
        f"  {'S/V range':<16} {'N':>8} {'Mean':>8} {'Median':>8} {'P10':>8} {'P90':>8}"
    )
    print("  " + "-" * 64)

    for interval in sv_df["sv_bin"].cat.categories:
        subset = sv_df[sv_df["sv_bin"] == interval]
        if len(subset) < 10:
            continue
        ei = subset["energy_intensity"]
        print(
            f"  {str(interval):<16} {len(subset):>8,} {ei.mean():>8.1f}"
            f" {ei.median():>8.1f} {ei.quantile(0.1):>8.1f}"
            f" {ei.quantile(0.9):>8.1f}"
        )

    # --- Compare with Rode et al.'s specific benchmarks ---
    print("\n  ### Comparison with Rode et al. (2014) benchmarks")
    low_sv = sv_df[sv_df["surface_to_volume"].between(0.10, 0.20)]
    high_sv = sv_df[sv_df["surface_to_volume"].between(0.35, 0.45)]

    if len(low_sv) > 0:
        print(
            f"  S/V ≈ 0.15 (range 0.10-0.20): "
            f"N={len(low_sv):,}, "
            f"mean={low_sv['energy_intensity'].mean():.1f} kWh/m²"
        )
        print("    Rode et al. reported: 35–80 kWh/m²/a")
    if len(high_sv) > 0:
        print(
            f"  S/V ≈ 0.40 (range 0.35-0.45): "
            f"N={len(high_sv):,}, "
            f"mean={high_sv['energy_intensity'].mean():.1f} kWh/m²"
        )
        print("    Rode et al. reported: 110–200 kWh/m²/a")

    if len(low_sv) > 0 and len(high_sv) > 0:
        ratio = high_sv["energy_intensity"].mean() / low_sv["energy_intensity"].mean()
        print(f"\n  Ratio (high S/V / low S/V): {ratio:.2f}×")
        print("    Rode et al. reported: up to 6× variation")

    # --- Regression: S/V as predictor of energy intensity ---
    print("\n  ### OLS: S/V → energy intensity")
    reg_df = sv_df[
        ["log_energy_intensity", "surface_to_volume", "log_floor_area", "building_age"]
    ].dropna()

    m_sv_only = smf.ols("log_energy_intensity ~ surface_to_volume", data=reg_df).fit()
    print(
        f"  S/V only: β = {m_sv_only.params['surface_to_volume']:.4f}, "
        f"R² = {m_sv_only.rsquared:.4f}"
    )

    m_sv_controls = smf.ols(
        "log_energy_intensity ~ surface_to_volume + log_floor_area + building_age",
        data=reg_df,
    ).fit()
    print(
        f"  S/V + controls: β = {m_sv_controls.params['surface_to_volume']:.4f}, "
        f"R² = {m_sv_controls.rsquared:.4f}"
    )

    # --- Methodological caveats ---
    print("\n  ### Methodological Notes")
    print("  1. Our S/V uses a prismatic approximation (flat-roof, single")
    print("     LiDAR height), not 3D DEM. This likely COMPRESSES the")
    print("     true S/V range vs Rode et al.'s resolved geometry.")
    print("  2. Our DV is total SAP energy (heating + hot water + lighting),")
    print("     not heating-only theoretical demand. Absolute values are")
    print("     NOT directly comparable — focus on the GRADIENT.")
    print("  3. If the S/V-energy gradient holds despite measurement error")
    print("     and uncontrolled technology variation, this STRENGTHENS")
    print("     the conclusion: morphology predicts energy even in")
    print("     heterogeneous real-world conditions.")


def rode_height_proxy(df: pd.DataFrame) -> None:
    """
    Test Rode et al.'s finding that building height is a logarithmic proxy.

    They found diminishing returns: doubling height from 3m to 6m has a much
    larger effect than doubling from 30m to 60m.
    """
    print("\n" + "-" * 70)
    print("  Building Height as Logarithmic Proxy (Rode et al. 2014)")
    print("-" * 70)

    if "building_height" not in df.columns or df["building_height"].notna().sum() < 50:
        print("  WARNING: Insufficient building_height data.")
        return

    ht_df = df[["log_energy_intensity", "building_height"]].dropna()
    ht_df = ht_df[ht_df["building_height"] > 0].copy()
    ht_df["log_height"] = np.log(ht_df["building_height"])
    print(f"  Records with height data: {len(ht_df):,}")

    # Linear vs log comparison
    m_linear = smf.ols("log_energy_intensity ~ building_height", data=ht_df).fit()
    m_log = smf.ols("log_energy_intensity ~ log_height", data=ht_df).fit()

    print("\n  ### Linear vs Logarithmic Specification")
    print(f"  {'Spec':<12} {'β':>10} {'R²':>10} {'AIC':>12}")
    print("  " + "-" * 48)
    print(
        f"  {'Linear':<12}"
        f" {m_linear.params['building_height']:>+10.4f}"
        f" {m_linear.rsquared:>10.4f}"
        f" {m_linear.aic:>12.1f}"
    )
    print(
        f"  {'Log':<12}"
        f" {m_log.params['log_height']:>+10.4f}"
        f" {m_log.rsquared:>10.4f}"
        f" {m_log.aic:>12.1f}"
    )

    better = "Log" if m_log.aic < m_linear.aic else "Linear"
    print(f"\n  Better fit (lower AIC): {better}")
    print("  Rode et al. prediction: log specification should outperform")

    # Joint model with S/V if available
    if "surface_to_volume" in df.columns:
        joint_df = df[
            ["log_energy_intensity", "building_height", "surface_to_volume"]
        ].dropna()
        joint_df = joint_df[joint_df["building_height"] > 0].copy()
        joint_df["log_height"] = np.log(joint_df["building_height"])

        if len(joint_df) > 50:
            m_joint = smf.ols(
                "log_energy_intensity ~ surface_to_volume + log_height",
                data=joint_df,
            ).fit()
            print("\n  ### Joint Model: S/V + log(height)")
            print(
                f"  S/V: β = {m_joint.params['surface_to_volume']:+.4f}"
                f" (p = {m_joint.pvalues['surface_to_volume']:.4f})"
            )
            print(
                f"  log(height): β = {m_joint.params['log_height']:+.4f}"
                f" (p = {m_joint.pvalues['log_height']:.4f})"
            )
            print(f"  R² = {m_joint.rsquared:.4f}")


def rode_far_convergence(df: pd.DataFrame) -> None:
    """
    Test Rode et al.'s FAR convergence finding.

    They reported: at FAR > 4, all morphologies converge to 30–50 kWh/m²/a.
    At FAR ≈ 1, maximum variation (50–150 kWh/m²/a).
    """
    print("\n" + "-" * 70)
    print("  FAR Convergence Test (Rode et al. 2014)")
    print("-" * 70)

    # Find FAR columns at different distances
    far_cols = sorted(c for c in df.columns if c.startswith("far_"))
    if not far_cols:
        print("  WARNING: No FAR columns found. Run test_pipeline.py first.")
        return

    # 400m catchment (π×0.4²≈0.5 km²) is closest to Rode's 500m×500m grid
    preferred = ["far_400", "far_800", "far_1600"]
    primary_col = next((c for c in preferred if c in df.columns), far_cols[0])

    far_bins = [0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 50.0]

    # --- Multi-scale FAR tables ---
    for far_col in far_cols:
        far_df = df[df[far_col].notna() & (df[far_col] > 0)].copy()
        if len(far_df) < 50:
            continue
        far_df["far_bin"] = pd.cut(far_df[far_col], bins=far_bins)

        is_primary = far_col == primary_col
        label = " (PRIMARY — closest to Rode 500m)" if is_primary else ""
        print(f"\n  ### Energy Intensity by FAR Bin: {far_col}{label}")
        print(
            f"  {'FAR range':<16} {'N':>8} {'Mean':>8}"
            f" {'Median':>8} {'Std':>8} {'CV':>8}"
        )
        print("  " + "-" * 64)

        for interval in far_df["far_bin"].cat.categories:
            subset = far_df[far_df["far_bin"] == interval]
            if len(subset) < 10:
                continue
            ei = subset["energy_intensity"]
            cv = ei.std() / ei.mean() if ei.mean() > 0 else np.nan
            print(
                f"  {str(interval):<16} {len(subset):>8,}"
                f" {ei.mean():>8.1f} {ei.median():>8.1f}"
                f" {ei.std():>8.1f} {cv:>8.2f}"
            )

    # --- Quadratic regression on primary FAR ---
    print(f"\n  ### Quadratic Regression: {primary_col}")
    q_df = df[[primary_col, "log_energy_intensity"]].dropna()
    q_df = q_df[q_df[primary_col] > 0].copy()
    q_df["far"] = q_df[primary_col]
    q_df["far_sq"] = q_df["far"] ** 2

    if len(q_df) > 50:
        m_lin = smf.ols("log_energy_intensity ~ far", data=q_df).fit()
        m_quad = smf.ols("log_energy_intensity ~ far + far_sq", data=q_df).fit()
        print(
            f"  Linear:    β(FAR) = {m_lin.params['far']:+.4f},"
            f" R² = {m_lin.rsquared:.4f}"
        )
        print(
            f"  Quadratic: β(FAR) = {m_quad.params['far']:+.4f},"
            f" β(FAR²) = {m_quad.params['far_sq']:+.6f},"
            f" R² = {m_quad.rsquared:.4f}"
        )
        if m_quad.pvalues["far_sq"] < 0.05:
            print("  FAR² is significant — nonlinear relationship confirmed")
        else:
            print("  FAR² is not significant — linear fit sufficient")

    # --- Levene's test: variance at low FAR vs high FAR ---
    print("\n  ### Levene's Test: Variance at Low vs High FAR")
    low_far = df[df[primary_col].between(0.5, 1.5)]["energy_intensity"]
    high_far = df[df[primary_col] > 4.0]["energy_intensity"]
    low_far = low_far.dropna()
    high_far = high_far.dropna()

    if len(low_far) > 10 and len(high_far) > 10:
        lev_stat, lev_p = stats.levene(low_far, high_far)
        print(f"  FAR 0.5-1.5 (N={len(low_far):,}): std = {low_far.std():.1f}")
        print(f"  FAR > 4.0   (N={len(high_far):,}): std = {high_far.std():.1f}")
        print(f"  Levene's F = {lev_stat:.2f}, p = {lev_p:.4f}")
        if lev_p < 0.05:
            print("  Variance differs significantly between groups")
        else:
            print("  No significant variance difference")
    else:
        print("  Insufficient data in one or both FAR groups")

    print("\n  Rode et al. prediction: variation should DECREASE at high FAR")
    print("  (CV should shrink; Levene's should show lower variance)")


def rode_typology_hierarchy(df: pd.DataFrame) -> None:
    """
    Test Rode et al.'s building typology energy hierarchy.

    They found: detached → semi → terrace → apartment block,
    with detached being least efficient and compact blocks most efficient.
    """
    print("\n" + "-" * 70)
    print("  Building Typology Energy Hierarchy (Rode et al. 2014)")
    print("-" * 70)

    type_df = df[df["attached_type"] != "other"].copy()

    # Order from Rode et al.'s least to most efficient
    type_order = ["detached", "semi", "end_terrace", "mid_terrace", "flat"]
    type_labels = {
        "detached": "Detached",
        "semi": "Semi-detached",
        "end_terrace": "End terrace",
        "mid_terrace": "Mid terrace",
        "flat": "Flat",
    }

    print(
        f"\n  {'Type':<18} {'N':>8} {'Mean EI':>10} {'Median EI':>10} {'Mean S/V':>10}"
    )
    print("  " + "-" * 62)

    for btype in type_order:
        subset = type_df[type_df["attached_type"] == btype]
        if len(subset) == 0:
            continue
        ei = subset["energy_intensity"]
        sv_str = "N/A"
        if "surface_to_volume" in subset.columns:
            sv = subset["surface_to_volume"]
            if sv.notna().sum() > 0:
                sv_str = f"{sv.mean():.3f}"
        print(
            f"  {type_labels.get(btype, btype):<18} {len(subset):>8,}"
            f" {ei.mean():>10.1f} {ei.median():>10.1f} {sv_str:>10}"
        )

    # Test ordering with Jonckheere-Terpstra trend (approximated with
    # Spearman correlation against ordinal rank)
    rank_map = {t: i for i, t in enumerate(type_order)}
    ranked = type_df[type_df["attached_type"].isin(type_order)].copy()
    ranked["type_rank"] = ranked["attached_type"].map(rank_map)
    r, p = stats.spearmanr(ranked["type_rank"], ranked["energy_intensity"])
    print(f"\n  Trend test (Spearman): rho = {r:+.4f} (p = {p:.2e})")
    print("  Rode et al. prediction: detached highest, compact forms lowest")
    direction = "CONFIRMED" if r < 0 else "NOT CONFIRMED"
    print(f"  Result: {direction} (negative rho = decreasing energy with rank)")


# ---------------------------------------------------------------------------
# Replication 2: Norman et al. (2006)
# ---------------------------------------------------------------------------


def norman_functional_unit(df: pd.DataFrame) -> None:
    """
    Replicate Norman et al.'s functional unit sensitivity finding.

    They found:
    - Per capita: low-density 2.0–2.5× worse than high-density
    - Per m²: low-density only 1.0–1.5× worse (advantage largely disappears)
    - Building operations ~60-70% of lifecycle, transport ~20-30%, embodied ~10%

    We bin by population density quintile and show both metrics, plus transport.
    """
    print("\n" + "=" * 70)
    print("REPLICATION 2: Norman et al. (2006)")
    print("Functional Unit Sensitivity: Per Capita vs Per m²")
    print("=" * 70)

    norm_df = df[df["pop_density"].notna()].copy()

    # --- Transport energy from Census car ownership (TS045) ---
    # Same method as stats/03_transport_analysis.py
    km_per_vehicle_year = 12_000  # UK average annual mileage
    kwh_per_km_ice = 0.17 / 0.233  # kg CO2/km ÷ kg CO2/kWh
    kwh_per_vehicle = km_per_vehicle_year * kwh_per_km_ice

    _car_pfx = "ts045_Number of cars or vans: "
    _car_sfx = "; measures: Value"
    car_cols = {
        0: f"{_car_pfx}No cars or vans in household{_car_sfx}",
        1: f"{_car_pfx}1 car or van in household{_car_sfx}",
        2: f"{_car_pfx}2 cars or vans in household{_car_sfx}",
        3: f"{_car_pfx}3 or more cars or vans in household{_car_sfx}",
    }
    has_transport = all(c in norm_df.columns for c in car_cols.values())

    if has_transport:
        total_cars = sum(n_cars * norm_df[col] for n_cars, col in car_cols.items())
        total_hh_cars = sum(norm_df[col] for col in car_cols.values())
        norm_df["avg_cars_per_hh"] = total_cars / total_hh_cars.replace(0, np.nan)
        norm_df["transport_energy_kwh"] = norm_df["avg_cars_per_hh"] * kwh_per_vehicle
        norm_df["transport_per_capita"] = (
            norm_df["transport_energy_kwh"] / norm_df["avg_household_size"]
        )
        norm_df["combined_per_capita"] = (
            norm_df["energy_per_capita"] + norm_df["transport_per_capita"]
        )

    # Create density quintiles
    norm_df["density_quintile"] = pd.qcut(
        norm_df["pop_density"],
        q=5,
        labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"],
    )

    print(f"\n  Records: {len(norm_df):,}")
    print(
        f"  Density range: {norm_df['pop_density'].min():.0f}"
        f" – {norm_df['pop_density'].max():.0f} persons/km²"
    )

    # --- Table: metrics by density quintile ---
    print("\n  ### Energy by Density Quintile")
    print(
        f"  {'Quintile':<16} {'N':>8} {'Density':>10}"
        f" {'kWh/m²':>10} {'kWh/cap':>10}"
        f" {'Floor m²':>10}"
        + (" {'Trnsp/cap':>10} {'Total/cap':>10}" if has_transport else "")
    )
    print("  " + "-" * (70 + (22 if has_transport else 0)))

    quintile_stats: list[dict] = []
    for q in ["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"]:
        subset = norm_df[norm_df["density_quintile"] == q]
        row: dict = {
            "quintile": q,
            "n": len(subset),
            "density": subset["pop_density"].mean(),
            "intensity": subset["energy_intensity"].mean(),
            "per_capita": subset["energy_per_capita"].mean(),
            "floor_area": subset["TOTAL_FLOOR_AREA"].mean(),
        }
        line = (
            f"  {q:<16} {row['n']:>8,} {row['density']:>10.0f}"
            f" {row['intensity']:>10.1f} {row['per_capita']:>10.0f}"
            f" {row['floor_area']:>10.1f}"
        )
        if has_transport:
            tp = subset["transport_per_capita"].mean()
            cp = subset["combined_per_capita"].mean()
            row["transport_pc"] = tp
            row["combined_pc"] = cp
            line += f" {tp:>10.0f} {cp:>10.0f}"
        quintile_stats.append(row)
        print(line)

    # --- Ratios: Q1 / Q5 for each metric ---
    if len(quintile_stats) >= 5:
        q1 = quintile_stats[0]
        q5 = quintile_stats[4]

        ratio_intensity = q1["intensity"] / q5["intensity"]
        ratio_percapita = q1["per_capita"] / q5["per_capita"]
        ratio_floor = q1["floor_area"] / q5["floor_area"]

        print("\n  ### Low-density / High-density Ratios (Q1 / Q5)")
        print(f"  Building per m²:     {ratio_intensity:.2f}×")
        print(f"  Building per capita: {ratio_percapita:.2f}×")
        print(f"  Floor area:          {ratio_floor:.2f}×")

        if has_transport and "combined_pc" in q1:
            ratio_combined = q1["combined_pc"] / q5["combined_pc"]
            ratio_transport = q1["transport_pc"] / q5["transport_pc"]
            print(f"  Transport per cap:   {ratio_transport:.2f}×")
            print(f"  Combined per cap:    {ratio_combined:.2f}×")

        print("\n  Norman et al. reported:")
        print("    Per capita:  2.0–2.5× (building + transport + embodied)")
        print("    Per m²:      1.0–1.5×")
        print("    Building operations: 60-70% of lifecycle energy")
        print("    Transport: 20-30%; Embodied: ~10% (excluded here)")

    # --- Regression with both DVs ---
    print("\n  ### Same model, different functional unit")
    reg_cols = [
        "log_energy_intensity",
        "log_energy_per_capita",
        "log_floor_area",
        "building_age",
        "pop_density",
    ]
    reg_df = norm_df[reg_cols].dropna()

    # With floor area control
    formula_ctrl = "{dv} ~ log_floor_area + building_age + pop_density"
    m_int = smf.ols(formula_ctrl.format(dv="log_energy_intensity"), data=reg_df).fit()
    m_pc = smf.ols(formula_ctrl.format(dv="log_energy_per_capita"), data=reg_df).fit()

    # Without floor area control (shows the mechanism)
    formula_no_fa = "{dv} ~ building_age + pop_density"
    m_pc_nofa = smf.ols(
        formula_no_fa.format(dv="log_energy_per_capita"), data=reg_df
    ).fit()

    print(f"  {'Model':<30} {'β(density)':>12} {'R²':>8}")
    print("  " + "-" * 54)
    print(
        f"  {'Intensity + controls':<30}"
        f" {m_int.params['pop_density']:>+12.6f}"
        f" {m_int.rsquared:>8.4f}"
    )
    print(
        f"  {'Per capita + controls':<30}"
        f" {m_pc.params['pop_density']:>+12.6f}"
        f" {m_pc.rsquared:>8.4f}"
    )
    print(
        f"  {'Per capita NO floor area':<30}"
        f" {m_pc_nofa.params['pop_density']:>+12.6f}"
        f" {m_pc_nofa.rsquared:>8.4f}"
    )

    print("\n  Note: floor area absorbs the per-capita mechanism (larger")
    print("  homes in low-density → more energy per person). Removing it")
    print("  reveals the full density-percapita gradient.")

    print("\n  Norman et al. prediction: per-m² shows weaker density effect")
    if abs(m_int.params["pop_density"]) < abs(m_pc.params["pop_density"]):
        print("  Result: CONFIRMED — intensity shows smaller density coefficient")
    else:
        print("  Result: NOT CONFIRMED — intensity shows larger density coefficient")


# ---------------------------------------------------------------------------
# Replication 3: Density → form → energy mediation
# ---------------------------------------------------------------------------


def density_form_energy_mediation(df: pd.DataFrame) -> None:
    """
    Test whether density has a residual effect on energy after thermal controls.

    Progressive model building (using log density for interpretability):
    - M0: density only
    - M1: + building size and age
    - M2: + thermal physics (S/V, shared walls) — PRIMARY mediation result
    - M3: + building type (robustness check — risk of overcontrolling)

    Supplemented with Sobel test and bootstrap CI for the indirect effect.
    """
    print("\n" + "=" * 70)
    print("REPLICATION 3: Density → Building Form → Energy Mediation")
    print("=" * 70)

    # Prepare complete cases
    needed = [
        "log_energy_intensity",
        "log_pop_density",
        "log_floor_area",
        "building_age",
        "shared_wall_ratio",
    ]
    has_sv = "surface_to_volume" in df.columns
    if has_sv:
        needed.append("surface_to_volume")

    # Include LSOA for clustered SEs if available
    has_lsoa = "LSOA21CD" in df.columns
    extra_cols = ["attached_type"]
    if has_lsoa:
        extra_cols.append("LSOA21CD")

    med_df = df[needed + extra_cols].dropna().copy()
    # Dummy encode attached_type
    med_df = pd.get_dummies(med_df, columns=["attached_type"], drop_first=True)
    attached_cols = [c for c in med_df.columns if c.startswith("attached_type_")]

    print(f"\n  Complete cases: {len(med_df):,}")
    print("  Using log(pop_density) for interpretable coefficients")

    # --- Fit function with optional clustered SEs ---
    def _fit(formula: str):  # noqa: ANN202
        model = smf.ols(formula, data=med_df).fit()
        if has_lsoa:
            model = smf.ols(formula, data=med_df).fit(
                cov_type="cluster",
                cov_kwds={"groups": med_df["LSOA21CD"]},
            )
        return model

    density_var = "log_pop_density"

    # --- M0: Density only ---
    m0 = _fit(f"log_energy_intensity ~ {density_var}")

    # --- M1: + Size and age ---
    m1 = _fit(f"log_energy_intensity ~ {density_var} + log_floor_area + building_age")

    # --- M2: + Thermal physics (PRIMARY mediation result) ---
    sv_term = " + surface_to_volume" if has_sv else ""
    m2 = _fit(
        f"log_energy_intensity ~ {density_var}"
        f" + log_floor_area + building_age + shared_wall_ratio{sv_term}"
    )

    # --- M3: + Building type (robustness check) ---
    type_terms = " + ".join(attached_cols)
    m3 = _fit(
        f"log_energy_intensity ~ {density_var}"
        f" + log_floor_area + building_age"
        f" + shared_wall_ratio{sv_term} + {type_terms}"
    )

    # --- Summary table ---
    models = {
        "M0: Density only": m0,
        "M1: + Size/age": m1,
        "M2: + Thermal *": m2,
        "M3: + Type": m3,
    }

    se_note = " (LSOA-clustered SEs)" if has_lsoa else ""
    print(f"\n  ### Progressive Attenuation of Density Coefficient{se_note}")
    print(f"  {'Model':<22} {'β(logdens)':>12} {'p-value':>10} {'R²':>8} {'ΔR²':>8}")
    print("  " + "-" * 66)

    prev_r2 = 0.0
    for name, model in models.items():
        beta = model.params[density_var]
        pval = model.pvalues[density_var]
        r2 = model.rsquared
        if pval < 0.001:
            sig = "***"
        elif pval < 0.01:
            sig = "**"
        elif pval < 0.05:
            sig = "*"
        else:
            sig = "ns"
        print(
            f"  {name:<22} {beta:>+12.6f} {pval:>10.4f} {sig:<3}"
            f" {r2:>8.4f} {r2 - prev_r2:>+8.4f}"
        )
        prev_r2 = r2

    print("  (* = primary mediation result; M3 = robustness check)")

    # --- Percentage attenuation ---
    beta_m0 = m0.params[density_var]
    beta_m2 = m2.params[density_var]
    beta_m3 = m3.params[density_var]
    if abs(beta_m0) > 0:
        att_m2 = (1 - abs(beta_m2) / abs(beta_m0)) * 100
        att_m3 = (1 - abs(beta_m3) / abs(beta_m0)) * 100
        print(f"\n  Attenuation M0→M2 (thermal): {att_m2:.1f}%")
        print(f"  Attenuation M0→M3 (+ type):  {att_m3:.1f}%")

    if m2.pvalues[density_var] >= 0.05:
        print("  CONCLUSION: Density effect fully mediated by thermal physics")
        print("    → Supports Rode et al.: morphology drives the relationship")
    else:
        print("  CONCLUSION: Density retains residual effect after thermal controls")
        print("    → Partial mediation: building form explains some but not all")

    # --- Sobel test for density → S/V → energy (Step 6) ---
    if has_sv:
        print("\n  ### Sobel Test: Indirect Effect via S/V")
        # a path: density → S/V
        m_a = smf.ols(f"surface_to_volume ~ {density_var}", data=med_df).fit()
        a = m_a.params[density_var]
        se_a = m_a.bse[density_var]

        # b path: S/V → energy | density
        m_b = smf.ols(
            f"log_energy_intensity ~ {density_var} + surface_to_volume",
            data=med_df,
        ).fit()
        b = m_b.params["surface_to_volume"]
        se_b = m_b.bse["surface_to_volume"]

        # Sobel statistic
        se_indirect = np.sqrt(a**2 * se_b**2 + b**2 * se_a**2)
        indirect = a * b
        z_sobel = indirect / se_indirect
        p_sobel = 2 * (1 - stats.norm.cdf(abs(z_sobel)))

        print(f"  a (density→S/V):  {a:+.4f} (SE={se_a:.4f})")
        print(f"  b (S/V→energy):   {b:+.4f} (SE={se_b:.4f})")
        print(f"  Indirect (a×b):   {indirect:+.6f}")
        print(f"  Sobel z = {z_sobel:.3f}, p = {p_sobel:.4f}")

        # Bootstrap CI
        print("\n  ### Bootstrap CI for Indirect Effect (N=1000)")
        rng = np.random.default_rng(42)
        n = len(med_df)
        boot_indirect = np.empty(1000)
        for i in range(1000):
            idx = rng.choice(n, size=n, replace=True)
            boot_df = med_df.iloc[idx]
            try:
                ma = smf.ols(f"surface_to_volume ~ {density_var}", data=boot_df).fit()
                mb = smf.ols(
                    f"log_energy_intensity ~ {density_var} + surface_to_volume",
                    data=boot_df,
                ).fit()
                boot_indirect[i] = (
                    ma.params[density_var] * mb.params["surface_to_volume"]
                )
            except Exception:
                boot_indirect[i] = np.nan

        boot_clean = boot_indirect[np.isfinite(boot_indirect)]
        ci_lo = np.percentile(boot_clean, 2.5)
        ci_hi = np.percentile(boot_clean, 97.5)
        print(
            f"  Indirect effect: {np.mean(boot_clean):+.6f}"
            f" (95% CI: [{ci_lo:+.6f}, {ci_hi:+.6f}])"
        )
        excludes_zero = (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0)
        if excludes_zero:
            print("  CI excludes zero — mediation is significant")
        else:
            print("  CI includes zero — mediation is not significant")

    # --- VIF check for M3 overcontrolling (Step 7) ---
    print("\n  ### VIF Check: M3 Overcontrolling Risk")
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Build numeric design matrix for M3
    vif_vars = ["log_floor_area", "building_age", "shared_wall_ratio"]
    if has_sv:
        vif_vars.append("surface_to_volume")
    vif_vars.extend(attached_cols)

    vif_df = med_df[vif_vars].dropna()
    vif_results = []
    for i, col in enumerate(vif_vars):
        vif_val = variance_inflation_factor(vif_df.values, i)
        vif_results.append((col, vif_val))

    high_vif = False
    for col, vif_val in vif_results:
        flag = " ⚠ HIGH" if vif_val > 5 else ""
        if vif_val > 5:
            high_vif = True
        print(f"  {col:<30} VIF = {vif_val:.1f}{flag}")

    if high_vif:
        print("\n  WARNING: VIF > 5 detected. M2 (thermal physics) is the")
        print("  cleaner test; M3 risks absorbing collinear variance")
        print("  from type dummies that overlap with S/V and shared walls.")
    else:
        print("\n  VIF acceptable — no severe multicollinearity in M3")

    # --- Key coefficients in primary model (M2) ---
    print("\n  ### Key Coefficients in Primary Model (M2)")
    key_vars = ["shared_wall_ratio"]
    if has_sv:
        key_vars.insert(0, "surface_to_volume")

    for var in key_vars:
        beta = m2.params[var]
        pval = m2.pvalues[var]
        if pval < 0.001:
            sig = "***"
        elif pval < 0.01:
            sig = "**"
        elif pval < 0.05:
            sig = "*"
        else:
            sig = "ns"
        print(f"  {var}: β = {beta:+.4f} (p = {pval:.4f}) {sig}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all replication analyses."""
    print("=" * 70)
    print("REPLICATION ANALYSIS")
    print("Testing key findings from foundational urban form-energy studies")
    print("=" * 70)
    print(f"\nData: {DATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_data()

    # --- Rode et al. (2014) ---
    rode_sv_energy_relationship(df)
    rode_height_proxy(df)
    rode_far_convergence(df)
    rode_typology_hierarchy(df)

    # --- Norman et al. (2006) ---
    norman_functional_unit(df)

    # --- Mediation test ---
    density_form_energy_mediation(df)

    # --- Overall summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
  Three foundational findings tested against observed EPC data:

  1. Rode et al. (2014): S/V ratio → energy intensity
     - Does the S/V-energy relationship from simulated European archetypes
       hold in observed English building data?
     - Does FAR convergence occur at high density?
     - Is the typology hierarchy (detached > semi > terrace > flat) preserved?

  2. Norman et al. (2006): Functional unit sensitivity
     - Does the per-capita vs per-m² discrepancy replicate?
     - Is the low-density disadvantage exaggerated by per-capita normalisation?

  3. Density mediation: Does density operate through building form?
     - Does the density coefficient attenuate when thermal physics are
       controlled?
     - Is the residual density effect significant or fully mediated?
""")


if __name__ == "__main__":
    main()
