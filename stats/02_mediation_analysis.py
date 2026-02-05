"""
Mediation analysis for Hypothesis H2.

Tests whether the association between neighborhood density and building energy
is MEDIATED by building stock composition (% terraces, % flats, etc.).

Theoretical model:
    Density → Stock Composition → Building Energy
              (mediator)

If stock composition fully mediates the relationship, this explains WHY
compact development is associated with lower energy: it's about building
type selection, not independent "density effects."

Methods:
1. Baron-Kenny approach (classical)
2. Sobel test for indirect effect
3. Bootstrap confidence intervals for indirect effect

Usage:
    uv run python stats/02d_mediation_analysis.py
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "temp" / "processing" / "test" / "uprn_integrated.gpkg"
OUTPUT_DIR = BASE_DIR / "temp" / "stats"


def load_data() -> pd.DataFrame:
    """Load and prepare data for mediation analysis."""
    print("Loading data...")
    gdf = gpd.read_file(DATA_PATH)

    # Filter to UPRNs with EPC data
    df = gdf[gdf["CURRENT_ENERGY_EFFICIENCY"].notna()].copy()
    print(f"  Records with EPC: {len(df):,}")

    # Compute energy per capita (same as in main regression)
    size_cols = {
        1: "ts017_Household size: 1 person in household; measures: Value",
        2: "ts017_Household size: 2 people in household; measures: Value",
        3: "ts017_Household size: 3 people in household; measures: Value",
        4: "ts017_Household size: 4 people in household; measures: Value",
        5: "ts017_Household size: 5 people in household; measures: Value",
        6: "ts017_Household size: 6 people in household; measures: Value",
        7: "ts017_Household size: 7 people in household; measures: Value",
        8: "ts017_Household size: 8 or more people in household; measures: Value",
    }

    total_people = sum(size * df[col] for size, col in size_cols.items())
    total_households = df[
        "ts017_Household size: Total: All household spaces; measures: Value"
    ] - df["ts017_Household size: 0 people in household; measures: Value"]

    df["avg_household_size"] = total_people / total_households
    df["energy_per_capita"] = df["ENERGY_CONSUMPTION_CURRENT"] / df["avg_household_size"]
    df["log_energy_per_capita"] = np.log(df["energy_per_capita"].clip(lower=1))

    # Population density
    df["pop_density"] = df[
        "ts006_Population Density: Persons per square kilometre; measures: Value"
    ]

    # Building type indicators (mediators)
    df["is_flat"] = df["PROPERTY_TYPE"].str.lower().str.contains("flat", na=False).astype(int)
    df["is_terrace"] = df["BUILT_FORM"].str.contains("Terrace", na=False).astype(int)
    df["is_detached"] = (df["BUILT_FORM"] == "Detached").astype(int)
    df["is_semi"] = (df["BUILT_FORM"] == "Semi-Detached").astype(int)

    # Log floor area (control)
    df["log_floor_area"] = np.log(df["TOTAL_FLOOR_AREA"].clip(lower=1))

    # Building age
    age_band_to_year = {
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

    # Filter to complete cases
    key_vars = [
        "log_energy_per_capita",
        "pop_density",
        "is_flat",
        "is_terrace",
        "log_floor_area",
        "building_age",
    ]
    df = df.dropna(subset=key_vars)
    print(f"  Complete cases: {len(df):,}")

    return df


def standardize(series: pd.Series) -> pd.Series:
    """Standardize a series to mean=0, sd=1."""
    return (series - series.mean()) / series.std()


def baron_kenny_mediation(df: pd.DataFrame) -> dict:
    """
    Baron-Kenny (1986) approach to mediation analysis.

    Steps:
    1. Total effect: X → Y (c path)
    2. X → M (a path)
    3. M → Y controlling for X (b path)
    4. X → Y controlling for M (c' path = direct effect)

    Mediation exists if:
    - a is significant
    - b is significant
    - c' < c (direct effect smaller than total effect)

    Full mediation: c' ≈ 0
    Partial mediation: c' < c but c' ≠ 0
    """
    print("\n" + "=" * 70)
    print("BARON-KENNY MEDIATION ANALYSIS")
    print("=" * 70)
    print("\nModel: Density → Building Type Composition → Energy per Capita")
    print("\nX (predictor): pop_density (standardized)")
    print("M (mediator): is_terrace + is_flat (building type composition)")
    print("Y (outcome): log_energy_per_capita")
    print("Controls: log_floor_area, building_age")

    # Standardize variables for comparable coefficients
    df = df.copy()
    df["X"] = standardize(df["pop_density"])
    df["Y"] = standardize(df["log_energy_per_capita"])
    df["M_terrace"] = df["is_terrace"]
    df["M_flat"] = df["is_flat"]
    df["log_floor_area_z"] = standardize(df["log_floor_area"])
    df["building_age_z"] = standardize(df["building_age"])

    results = {}

    # Step 1: Total effect (c path)
    # Y ~ X + controls
    print("\n### Step 1: Total Effect (c path)")
    print("    Y ~ X + controls")
    model_c = smf.ols(
        "Y ~ X + log_floor_area_z + building_age_z",
        data=df,
    ).fit()
    c = model_c.params["X"]
    c_se = model_c.bse["X"]
    c_p = model_c.pvalues["X"]
    results["c_total"] = {"coef": c, "se": c_se, "p": c_p}
    print(f"    c (total effect): {c:.4f} (SE={c_se:.4f}, p={c_p:.4f})")

    # Step 2a: X → M_terrace (a1 path)
    print("\n### Step 2a: Density → Terrace (a1 path)")
    print("    M_terrace ~ X + controls")
    model_a1 = smf.ols(
        "M_terrace ~ X + log_floor_area_z + building_age_z",
        data=df,
    ).fit()
    a1 = model_a1.params["X"]
    a1_se = model_a1.bse["X"]
    a1_p = model_a1.pvalues["X"]
    results["a1_terrace"] = {"coef": a1, "se": a1_se, "p": a1_p}
    print(f"    a1: {a1:.4f} (SE={a1_se:.4f}, p={a1_p:.4f})")

    # Step 2b: X → M_flat (a2 path)
    print("\n### Step 2b: Density → Flat (a2 path)")
    print("    M_flat ~ X + controls")
    model_a2 = smf.ols(
        "M_flat ~ X + log_floor_area_z + building_age_z",
        data=df,
    ).fit()
    a2 = model_a2.params["X"]
    a2_se = model_a2.bse["X"]
    a2_p = model_a2.pvalues["X"]
    results["a2_flat"] = {"coef": a2, "se": a2_se, "p": a2_p}
    print(f"    a2: {a2:.4f} (SE={a2_se:.4f}, p={a2_p:.4f})")

    # Step 3: M → Y controlling for X (b paths) and direct effect (c' path)
    print("\n### Step 3: Building Type → Energy (b paths) + Direct Effect (c' path)")
    print("    Y ~ X + M_terrace + M_flat + controls")
    model_full = smf.ols(
        "Y ~ X + M_terrace + M_flat + log_floor_area_z + building_age_z",
        data=df,
    ).fit()

    c_prime = model_full.params["X"]
    c_prime_se = model_full.bse["X"]
    c_prime_p = model_full.pvalues["X"]
    results["c_prime_direct"] = {"coef": c_prime, "se": c_prime_se, "p": c_prime_p}
    print(f"    c' (direct effect): {c_prime:.4f} (SE={c_prime_se:.4f}, p={c_prime_p:.4f})")

    b1 = model_full.params["M_terrace"]
    b1_se = model_full.bse["M_terrace"]
    b1_p = model_full.pvalues["M_terrace"]
    results["b1_terrace"] = {"coef": b1, "se": b1_se, "p": b1_p}
    print(f"    b1 (terrace → energy): {b1:.4f} (SE={b1_se:.4f}, p={b1_p:.4f})")

    b2 = model_full.params["M_flat"]
    b2_se = model_full.bse["M_flat"]
    b2_p = model_full.pvalues["M_flat"]
    results["b2_flat"] = {"coef": b2, "se": b2_se, "p": b2_p}
    print(f"    b2 (flat → energy): {b2:.4f} (SE={b2_se:.4f}, p={b2_p:.4f})")

    # Compute indirect effects
    print("\n### Indirect Effects")
    indirect1 = a1 * b1
    indirect2 = a2 * b2
    total_indirect = indirect1 + indirect2

    results["indirect_terrace"] = indirect1
    results["indirect_flat"] = indirect2
    results["indirect_total"] = total_indirect

    print(f"    Indirect via terrace (a1 × b1): {indirect1:.4f}")
    print(f"    Indirect via flat (a2 × b2): {indirect2:.4f}")
    print(f"    Total indirect effect: {total_indirect:.4f}")
    print(f"    Direct effect (c'): {c_prime:.4f}")
    print(f"    Total effect (c): {c:.4f}")

    # Proportion mediated
    if abs(c) > 0.001:
        prop_mediated = total_indirect / c
        results["proportion_mediated"] = prop_mediated
        print(f"\n    Proportion mediated: {prop_mediated:.1%}")
    else:
        results["proportion_mediated"] = np.nan
        print("\n    Proportion mediated: undefined (total effect ≈ 0)")

    return results


def sobel_test(a: float, b: float, se_a: float, se_b: float) -> tuple[float, float]:
    """
    Sobel test for significance of indirect effect (a × b).

    Returns z-statistic and p-value.
    """
    se_indirect = np.sqrt(a**2 * se_b**2 + b**2 * se_a**2)
    z = (a * b) / se_indirect
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def bootstrap_indirect_effect(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict:
    """
    Bootstrap confidence interval for indirect effect.

    More robust than Sobel test, especially for non-normal sampling distributions.
    """
    print("\n" + "=" * 70)
    print(f"BOOTSTRAP CONFIDENCE INTERVALS (n={n_bootstrap})")
    print("=" * 70)

    rng = np.random.default_rng(random_state)
    n = len(df)

    # Standardize once
    df = df.copy()
    df["X"] = standardize(df["pop_density"])
    df["Y"] = standardize(df["log_energy_per_capita"])
    df["log_floor_area_z"] = standardize(df["log_floor_area"])
    df["building_age_z"] = standardize(df["building_age"])

    indirect_terrace = []
    indirect_flat = []
    indirect_total = []
    direct_effects = []

    print("\n  Bootstrapping...")
    for i in range(n_bootstrap):
        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{n_bootstrap}")

        # Resample with replacement
        idx = rng.choice(n, size=n, replace=True)
        boot_df = df.iloc[idx]

        try:
            # a paths
            model_a1 = smf.ols(
                "is_terrace ~ X + log_floor_area_z + building_age_z",
                data=boot_df,
            ).fit()
            a1 = model_a1.params["X"]

            model_a2 = smf.ols(
                "is_flat ~ X + log_floor_area_z + building_age_z",
                data=boot_df,
            ).fit()
            a2 = model_a2.params["X"]

            # b paths and c'
            model_full = smf.ols(
                "Y ~ X + is_terrace + is_flat + log_floor_area_z + building_age_z",
                data=boot_df,
            ).fit()
            b1 = model_full.params["is_terrace"]
            b2 = model_full.params["is_flat"]
            c_prime = model_full.params["X"]

            indirect_terrace.append(a1 * b1)
            indirect_flat.append(a2 * b2)
            indirect_total.append(a1 * b1 + a2 * b2)
            direct_effects.append(c_prime)

        except Exception:
            continue

    # Compute confidence intervals
    results = {}

    for name, values in [
        ("indirect_terrace", indirect_terrace),
        ("indirect_flat", indirect_flat),
        ("indirect_total", indirect_total),
        ("direct_effect", direct_effects),
    ]:
        values = np.array(values)
        ci_lower = np.percentile(values, 2.5)
        ci_upper = np.percentile(values, 97.5)
        mean_est = np.mean(values)

        results[name] = {
            "mean": mean_est,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significant": not (ci_lower <= 0 <= ci_upper),
        }

        sig_str = "SIGNIFICANT" if results[name]["significant"] else "not significant"
        print(f"\n  {name}:")
        print(f"    Mean: {mean_est:.4f}")
        print(f"    95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"    {sig_str} (CI excludes zero: {results[name]['significant']})")

    return results


def mediation_with_shared_walls(df: pd.DataFrame) -> dict:
    """
    Alternative mediation using shared_wall_ratio as mediator.

    This tests whether density operates through building FORM
    (party walls) rather than just building TYPE labels.
    """
    print("\n" + "=" * 70)
    print("MEDIATION WITH SHARED WALL RATIO")
    print("=" * 70)
    print("\nModel: Density → Shared Wall Ratio → Energy per Capita")

    if "shared_wall_ratio" not in df.columns:
        print("  shared_wall_ratio not available")
        return {}

    df = df.dropna(subset=["shared_wall_ratio"])
    print(f"  Records with shared_wall_ratio: {len(df):,}")

    # Standardize
    df = df.copy()
    df["X"] = standardize(df["pop_density"])
    df["Y"] = standardize(df["log_energy_per_capita"])
    df["M"] = standardize(df["shared_wall_ratio"])
    df["log_floor_area_z"] = standardize(df["log_floor_area"])
    df["building_age_z"] = standardize(df["building_age"])

    results = {}

    # Total effect
    print("\n### Total Effect")
    model_c = smf.ols("Y ~ X + log_floor_area_z + building_age_z", data=df).fit()
    c = model_c.params["X"]
    print(f"    c: {c:.4f} (p={model_c.pvalues['X']:.4f})")
    results["c_total"] = c

    # a path: X → M
    print("\n### Density → Shared Wall Ratio")
    model_a = smf.ols("M ~ X + log_floor_area_z + building_age_z", data=df).fit()
    a = model_a.params["X"]
    a_se = model_a.bse["X"]
    print(f"    a: {a:.4f} (p={model_a.pvalues['X']:.4f})")
    results["a"] = a

    # b path and c'
    print("\n### Shared Wall Ratio → Energy (controlling for density)")
    model_full = smf.ols(
        "Y ~ X + M + log_floor_area_z + building_age_z",
        data=df,
    ).fit()
    b = model_full.params["M"]
    b_se = model_full.bse["M"]
    c_prime = model_full.params["X"]

    print(f"    b: {b:.4f} (p={model_full.pvalues['M']:.4f})")
    print(f"    c': {c_prime:.4f} (p={model_full.pvalues['X']:.4f})")
    results["b"] = b
    results["c_prime"] = c_prime

    # Indirect effect
    indirect = a * b
    results["indirect"] = indirect

    print(f"\n### Summary")
    print(f"    Indirect effect (a × b): {indirect:.4f}")
    print(f"    Direct effect (c'): {c_prime:.4f}")
    print(f"    Total effect (c): {c:.4f}")

    if abs(c) > 0.001:
        prop_med = indirect / c
        print(f"    Proportion mediated: {prop_med:.1%}")
        results["proportion_mediated"] = prop_med

    # Sobel test
    z, p = sobel_test(a, b, a_se, b_se)
    print(f"\n    Sobel test: z={z:.3f}, p={p:.4f}")
    results["sobel_z"] = z
    results["sobel_p"] = p

    return results


def summarize_mediation_findings(
    bk_results: dict,
    bootstrap_results: dict,
    shared_wall_results: dict,
) -> None:
    """Print summary of mediation findings."""
    print("\n" + "=" * 70)
    print("MEDIATION ANALYSIS SUMMARY")
    print("=" * 70)

    print("""
## Hypothesis H2: Stock Composition Mediation

The hypothesis is that the density-energy association is MEDIATED by
building stock composition. If so:
- Dense areas have more flats and terraces (a path)
- Flats and terraces have lower energy demand (b path)
- The direct density effect (c') should be near zero after controlling for building type

## Key Findings
""")

    # Total effect
    c = bk_results.get("c_total", {}).get("coef", np.nan)
    print(f"1. Total effect of density on energy: β = {c:.4f}")
    if abs(c) < 0.05:
        print("   → Near-zero total effect (already known from previous analysis)")
    else:
        print(f"   → Modest total effect")

    # a paths
    a1 = bk_results.get("a1_terrace", {}).get("coef", np.nan)
    a2 = bk_results.get("a2_flat", {}).get("coef", np.nan)
    print(f"\n2. Does density predict building type?")
    print(f"   - Density → Terrace: β = {a1:.4f} (p = {bk_results.get('a1_terrace', {}).get('p', np.nan):.4f})")
    print(f"   - Density → Flat: β = {a2:.4f} (p = {bk_results.get('a2_flat', {}).get('p', np.nan):.4f})")

    # b paths
    b1 = bk_results.get("b1_terrace", {}).get("coef", np.nan)
    b2 = bk_results.get("b2_flat", {}).get("coef", np.nan)
    print(f"\n3. Does building type predict energy (controlling for density)?")
    print(f"   - Terrace → Energy: β = {b1:.4f} (p = {bk_results.get('b1_terrace', {}).get('p', np.nan):.4f})")
    print(f"   - Flat → Energy: β = {b2:.4f} (p = {bk_results.get('b2_flat', {}).get('p', np.nan):.4f})")

    # Indirect effects
    print(f"\n4. Indirect effects (mediated pathway):")
    print(f"   - Via terrace: {bk_results.get('indirect_terrace', np.nan):.4f}")
    print(f"   - Via flat: {bk_results.get('indirect_flat', np.nan):.4f}")
    print(f"   - Total indirect: {bk_results.get('indirect_total', np.nan):.4f}")

    # Bootstrap significance
    if bootstrap_results:
        total_bs = bootstrap_results.get("indirect_total", {})
        sig = "SIGNIFICANT" if total_bs.get("significant", False) else "not significant"
        print(f"\n5. Bootstrap test for indirect effect:")
        print(f"   - 95% CI: [{total_bs.get('ci_lower', np.nan):.4f}, {total_bs.get('ci_upper', np.nan):.4f}]")
        print(f"   - {sig}")

    # Interpretation
    print(f"\n## Interpretation")

    if abs(c) < 0.05:
        print("""
The total effect of density on energy is near zero, which limits the
scope for mediation analysis. However, this itself is informative:

1. If a × b ≈ 0 because both a and b are non-zero but opposite signs,
   this suggests density has OFFSETTING effects through different building types.

2. If a ≈ 0, density doesn't strongly predict building type composition
   in this sample (may be due to study area selection or confounding).

3. If b ≈ 0, building type doesn't strongly predict energy after
   controlling for other factors (e.g., floor area, age).
""")
    else:
        prop_med = bk_results.get("proportion_mediated", np.nan)
        if not np.isnan(prop_med):
            if prop_med > 0.8:
                print(f"""
Strong evidence for FULL MEDIATION: {prop_med:.0%} of the density effect
operates through building stock composition. The direct effect of density
is near zero, meaning density affects energy ONLY by selecting building types.
""")
            elif prop_med > 0.3:
                print(f"""
Evidence for PARTIAL MEDIATION: {prop_med:.0%} of the density effect
operates through building stock composition. There remains a direct
density effect beyond building type selection.
""")
            else:
                print(f"""
Limited mediation: Only {prop_med:.0%} operates through building stock.
The density-energy association is largely NOT explained by building type.
""")


def main() -> None:
    """Run mediation analysis for H2."""
    print("=" * 70)
    print("MEDIATION ANALYSIS: Density → Stock Composition → Energy")
    print("=" * 70)
    print("\nHypothesis H2: The association between neighborhood density and")
    print("building energy is mediated by building stock composition.")
    print(f"\nData: {DATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data()

    # Baron-Kenny mediation
    bk_results = baron_kenny_mediation(df)

    # Bootstrap confidence intervals
    bootstrap_results = bootstrap_indirect_effect(df, n_bootstrap=1000)

    # Alternative: mediation via shared_wall_ratio
    shared_wall_results = mediation_with_shared_walls(df)

    # Summary
    summarize_mediation_findings(bk_results, bootstrap_results, shared_wall_results)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("- Update WORKING_LOG.md with findings")
    print("- Run stratified analysis (02f_stratified_analysis.py)")
    print("- Compare mediation results across building subgroups")


if __name__ == "__main__":
    main()
