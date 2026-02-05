"""
Stratified analysis for Hypotheses H4 and H5.

H4: Age-Density Confounding
    The apparent lack of density-energy association is explained by age
    confounding: central (dense) areas have older buildings with worse fabric.

    Test: Within construction era strata, does density show clearer association?

H5: Building Type Heterogeneity
    The density-energy relationship differs by building type.

    Test: Run separate models for houses vs flats.

Additional analyses:
- Age-residualized density
- Propensity score matching (optional)

Usage:
    uv run python stats/02f_stratified_analysis.py
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
    """Load and prepare data for stratified analysis."""
    print("Loading data...")
    gdf = gpd.read_file(DATA_PATH)

    # Filter to UPRNs with EPC data
    df = gdf[gdf["CURRENT_ENERGY_EFFICIENCY"].notna()].copy()
    print(f"  Records with EPC: {len(df):,}")

    # Compute energy per capita
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
    df["log_floor_area"] = np.log(df["TOTAL_FLOOR_AREA"].clip(lower=1))

    # Population density
    df["pop_density"] = df[
        "ts006_Population Density: Persons per square kilometre; measures: Value"
    ]

    # Network centrality (if available)
    if "cc_harmonic_800" in df.columns:
        df["centrality"] = df["cc_harmonic_800"]

    # Building age from construction age band
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

    # Create construction era categories
    def categorize_era(year):
        if pd.isna(year):
            return np.nan
        elif year < 1919:
            return "Pre-1919 (Victorian/Edwardian)"
        elif year < 1945:
            return "1919-1944 (Interwar)"
        elif year < 1980:
            return "1945-1979 (Post-war)"
        else:
            return "1980+ (Modern)"

    df["construction_era"] = df["construction_year"].apply(categorize_era)

    # Building type categories
    df["is_flat"] = df["PROPERTY_TYPE"].str.lower().str.contains("flat", na=False)
    df["is_house"] = ~df["is_flat"]
    df["building_type"] = np.where(df["is_flat"], "Flat", "House")

    # Detailed building type
    df["detailed_type"] = df["PROPERTY_TYPE"].fillna("Unknown")

    # Filter to complete cases
    key_vars = [
        "log_energy_per_capita",
        "pop_density",
        "log_floor_area",
        "building_age",
        "construction_era",
        "building_type",
    ]
    df = df.dropna(subset=key_vars)
    print(f"  Complete cases: {len(df):,}")

    return df


def standardize(series: pd.Series) -> pd.Series:
    """Standardize to mean=0, sd=1."""
    return (series - series.mean()) / series.std()


def analyze_age_density_confounding(df: pd.DataFrame) -> dict:
    """
    H4: Test whether age-density confounding explains the null result.

    1. Quantify age-density correlation
    2. Run models within construction era strata
    3. Test age-residualized density
    """
    print("\n" + "=" * 70)
    print("H4: AGE-DENSITY CONFOUNDING ANALYSIS")
    print("=" * 70)

    results = {}

    # 1. Quantify confounding: correlation between age and density
    print("\n### 1. Age-Density Correlation (Confounding)")
    valid = df[["building_age", "pop_density"]].dropna()
    r, p = stats.pearsonr(valid["building_age"], valid["pop_density"])
    results["age_density_correlation"] = {"r": r, "p": p}

    print(f"    Correlation: r = {r:.3f} (p = {p:.4f})")
    if r > 0:
        print("    → Denser areas have OLDER buildings (positive confound)")
        print("    → This could mask a density benefit or create spurious association")
    else:
        print("    → Denser areas have NEWER buildings")

    # 2. Density-energy correlation within each construction era
    print("\n### 2. Density-Energy Association by Construction Era")
    print("-" * 70)
    print(f"{'Era':<35} {'N':>8} {'r':>8} {'p':>10} {'Interpretation':<20}")
    print("-" * 70)

    era_results = {}
    for era in df["construction_era"].dropna().unique():
        subset = df[df["construction_era"] == era]
        if len(subset) < 30:
            continue

        valid = subset[["log_energy_per_capita", "pop_density"]].dropna()
        r_era, p_era = stats.pearsonr(valid["log_energy_per_capita"], valid["pop_density"])

        interp = "positive" if r_era > 0.05 else "negative" if r_era < -0.05 else "near zero"
        print(f"  {era:<33} {len(subset):>8,} {r_era:>8.3f} {p_era:>10.4f} {interp:<20}")

        era_results[era] = {"n": len(subset), "r": r_era, "p": p_era}

    results["by_era"] = era_results

    # 3. Regression models within each era
    print("\n### 3. Regression Models by Construction Era")
    print("    Model: log_energy ~ pop_density + log_floor_area")
    print("-" * 70)
    print(f"{'Era':<35} {'N':>8} {'β(density)':>12} {'p':>10} {'R²':>8}")
    print("-" * 70)

    for era in sorted(df["construction_era"].dropna().unique()):
        subset = df[df["construction_era"] == era].dropna(
            subset=["log_energy_per_capita", "pop_density", "log_floor_area"]
        )
        if len(subset) < 50:
            continue

        try:
            model = smf.ols(
                "log_energy_per_capita ~ pop_density + log_floor_area",
                data=subset,
            ).fit()
            beta = model.params["pop_density"]
            p_val = model.pvalues["pop_density"]
            r2 = model.rsquared

            print(f"  {era:<33} {len(subset):>8,} {beta:>12.6f} {p_val:>10.4f} {r2:>8.3f}")

            era_results[era]["beta"] = beta
            era_results[era]["beta_p"] = p_val
            era_results[era]["r2"] = r2
        except Exception as e:
            print(f"  {era:<33} Error: {e}")

    print("-" * 70)

    # 4. Age-residualized density
    print("\n### 4. Age-Residualized Density")
    print("    Removing age effect from density to test 'pure' density association")

    # Regress density on age, get residuals
    valid = df[["pop_density", "building_age", "log_energy_per_capita", "log_floor_area"]].dropna()

    age_density_model = smf.ols("pop_density ~ building_age", data=valid).fit()
    valid["density_resid"] = age_density_model.resid

    # Correlation of residualized density with energy
    r_resid, p_resid = stats.pearsonr(valid["log_energy_per_capita"], valid["density_resid"])
    print(f"    Correlation (energy ~ age-residualized density): r = {r_resid:.4f} (p = {p_resid:.4f})")

    results["age_residualized"] = {"r": r_resid, "p": p_resid}

    # Model with age-residualized density
    model_resid = smf.ols(
        "log_energy_per_capita ~ density_resid + log_floor_area",
        data=valid,
    ).fit()
    print(f"    β(density_resid) = {model_resid.params['density_resid']:.6f} (p = {model_resid.pvalues['density_resid']:.4f})")

    results["age_residualized"]["beta"] = model_resid.params["density_resid"]
    results["age_residualized"]["beta_p"] = model_resid.pvalues["density_resid"]

    return results


def analyze_building_type_heterogeneity(df: pd.DataFrame) -> dict:
    """
    H5: Test whether density-energy relationship differs by building type.

    Houses: Shared wall variation (detached → semi → terrace)
    Flats: Already optimized for density (less variation expected)
    """
    print("\n" + "=" * 70)
    print("H5: BUILDING TYPE HETEROGENEITY ANALYSIS")
    print("=" * 70)

    results = {}

    # Sample sizes by type
    print("\n### Sample Sizes by Building Type")
    type_counts = df["building_type"].value_counts()
    for btype, count in type_counts.items():
        print(f"    {btype}: {count:,}")

    # 1. Correlation within each building type
    print("\n### 1. Density-Energy Correlation by Building Type")
    print("-" * 60)
    print(f"{'Type':<15} {'N':>10} {'r':>10} {'p':>12}")
    print("-" * 60)

    for btype in ["House", "Flat"]:
        subset = df[df["building_type"] == btype]
        valid = subset[["log_energy_per_capita", "pop_density"]].dropna()
        if len(valid) < 30:
            continue

        r, p = stats.pearsonr(valid["log_energy_per_capita"], valid["pop_density"])
        print(f"  {btype:<13} {len(valid):>10,} {r:>10.4f} {p:>12.4f}")
        results[f"{btype}_correlation"] = {"n": len(valid), "r": r, "p": p}

    print("-" * 60)

    # 2. Full regression models by type
    print("\n### 2. Regression Models by Building Type")
    print("    Model: log_energy ~ pop_density + log_floor_area + building_age")

    for btype in ["House", "Flat"]:
        subset = df[df["building_type"] == btype].dropna(
            subset=["log_energy_per_capita", "pop_density", "log_floor_area", "building_age"]
        )
        if len(subset) < 100:
            print(f"\n  {btype}: Insufficient data (n={len(subset)})")
            continue

        print(f"\n  {btype} (n={len(subset):,}):")
        model = smf.ols(
            "log_energy_per_capita ~ pop_density + log_floor_area + building_age",
            data=subset,
        ).fit()

        print(f"    R² = {model.rsquared:.4f}")
        print(f"    pop_density: β = {model.params['pop_density']:.6f} (p = {model.pvalues['pop_density']:.4f})")
        print(f"    log_floor_area: β = {model.params['log_floor_area']:.4f} (p = {model.pvalues['log_floor_area']:.4f})")
        print(f"    building_age: β = {model.params['building_age']:.5f} (p = {model.pvalues['building_age']:.4f})")

        results[f"{btype}_model"] = {
            "n": len(subset),
            "r2": model.rsquared,
            "beta_density": model.params["pop_density"],
            "p_density": model.pvalues["pop_density"],
        }

    # 3. Interaction test: building_type × density
    print("\n### 3. Interaction Test: Building Type × Density")

    df_model = df.dropna(
        subset=["log_energy_per_capita", "pop_density", "log_floor_area", "building_age", "is_flat"]
    ).copy()
    df_model["density_z"] = standardize(df_model["pop_density"])

    # Model without interaction
    model_main = smf.ols(
        "log_energy_per_capita ~ density_z + is_flat + log_floor_area + building_age",
        data=df_model,
    ).fit()

    # Model with interaction
    model_interact = smf.ols(
        "log_energy_per_capita ~ density_z * is_flat + log_floor_area + building_age",
        data=df_model,
    ).fit()

    print(f"\n  Model without interaction: R² = {model_main.rsquared:.4f}, AIC = {model_main.aic:.1f}")
    print(f"  Model with interaction: R² = {model_interact.rsquared:.4f}, AIC = {model_interact.aic:.1f}")

    interact_coef = model_interact.params.get("density_z:is_flat", np.nan)
    interact_p = model_interact.pvalues.get("density_z:is_flat", np.nan)

    print(f"\n  Interaction term (density × flat):")
    print(f"    β = {interact_coef:.4f} (p = {interact_p:.4f})")

    if interact_p < 0.05:
        print("    → SIGNIFICANT interaction: density effect differs for houses vs flats")
        if interact_coef > 0:
            print("    → Density has MORE positive (or less negative) effect for flats")
        else:
            print("    → Density has MORE negative effect for flats")
    else:
        print("    → No significant interaction: density effect similar across building types")

    results["interaction"] = {
        "coef": interact_coef,
        "p": interact_p,
        "significant": interact_p < 0.05,
    }

    # Likelihood ratio test
    lr_stat = 2 * (model_interact.llf - model_main.llf)
    lr_p = stats.chi2.sf(lr_stat, 1)
    print(f"\n  Likelihood ratio test: χ² = {lr_stat:.2f}, p = {lr_p:.4f}")
    results["lr_test"] = {"stat": lr_stat, "p": lr_p}

    return results


def analyze_detailed_building_types(df: pd.DataFrame) -> dict:
    """Analyze density-energy association for detailed building types."""
    print("\n" + "=" * 70)
    print("DETAILED BUILDING TYPE ANALYSIS")
    print("=" * 70)

    results = {}

    # Group similar types
    type_mapping = {
        "House": ["House", "Bungalow"],
        "Flat": ["Flat", "Maisonette"],
    }

    # Built form analysis (for houses only)
    print("\n### Density-Energy Association by Built Form (Houses Only)")
    houses = df[df["is_house"]].copy()

    if "BUILT_FORM" in houses.columns:
        print("-" * 70)
        print(f"{'Built Form':<25} {'N':>8} {'r':>8} {'β(density)':>12} {'p':>10}")
        print("-" * 70)

        for form in houses["BUILT_FORM"].dropna().unique():
            subset = houses[houses["BUILT_FORM"] == form].dropna(
                subset=["log_energy_per_capita", "pop_density", "log_floor_area", "building_age"]
            )
            if len(subset) < 50:
                continue

            # Correlation
            r, _ = stats.pearsonr(subset["log_energy_per_capita"], subset["pop_density"])

            # Regression
            try:
                model = smf.ols(
                    "log_energy_per_capita ~ pop_density + log_floor_area + building_age",
                    data=subset,
                ).fit()
                beta = model.params["pop_density"]
                p_val = model.pvalues["pop_density"]
                print(f"  {form:<23} {len(subset):>8,} {r:>8.4f} {beta:>12.6f} {p_val:>10.4f}")

                results[f"builtform_{form}"] = {
                    "n": len(subset),
                    "r": r,
                    "beta": beta,
                    "p": p_val,
                }
            except Exception:
                print(f"  {form:<23} {len(subset):>8,} {r:>8.4f} {'(model error)':<22}")

        print("-" * 70)

    return results


def summarize_stratified_findings(h4_results: dict, h5_results: dict) -> None:
    """Print summary of stratified analysis findings."""
    print("\n" + "=" * 70)
    print("STRATIFIED ANALYSIS SUMMARY")
    print("=" * 70)

    print("""
## H4: Age-Density Confounding

Question: Does the near-zero density-energy association arise because
dense areas have older buildings (which have worse fabric)?
""")

    # Age-density correlation
    age_dens = h4_results.get("age_density_correlation", {})
    r_ad = age_dens.get("r", np.nan)
    print(f"1. Age-density correlation: r = {r_ad:.3f}")
    if r_ad > 0.1:
        print("   → Dense areas DO have older buildings → confounding present")
    elif r_ad < -0.1:
        print("   → Dense areas have NEWER buildings → reverse pattern")
    else:
        print("   → Weak age-density relationship → confounding limited")

    # Age-residualized density
    resid = h4_results.get("age_residualized", {})
    r_resid = resid.get("r", np.nan)
    print(f"\n2. Age-residualized density correlation: r = {r_resid:.4f}")
    print("   (This is 'pure' density effect, with age variation removed)")
    if abs(r_resid) > abs(r_ad) + 0.05:
        print("   → Removing age confounding REVEALS a density effect")
    elif abs(r_resid) < 0.05:
        print("   → Even after removing age, density shows no association")

    # Era-stratified
    print("\n3. Within-era analysis:")
    by_era = h4_results.get("by_era", {})
    consistent_null = all(abs(era.get("r", 0)) < 0.1 for era in by_era.values())
    if consistent_null:
        print("   → Density-energy association near zero within ALL construction eras")
        print("   → Confirms: age confounding doesn't explain the null result")
    else:
        print("   → Density effect varies by construction era")
        print("   → Age confounding may partially explain overall null")

    print("""
## H5: Building Type Heterogeneity

Question: Does the density-energy relationship differ between houses and flats?
""")

    # Interaction test
    interact = h5_results.get("interaction", {})
    if interact.get("significant", False):
        coef = interact.get("coef", 0)
        print(f"1. Interaction test: SIGNIFICANT (β = {coef:.4f})")
        print("   → Density effect DIFFERS between houses and flats")
        if coef > 0:
            print("   → Flats show WEAKER negative (or more positive) density association")
        else:
            print("   → Flats show STRONGER negative density association")
    else:
        print(f"1. Interaction test: Not significant (p = {interact.get('p', np.nan):.3f})")
        print("   → Density effect SIMILAR for houses and flats")

    # By-type correlations
    house_r = h5_results.get("House_correlation", {}).get("r", np.nan)
    flat_r = h5_results.get("Flat_correlation", {}).get("r", np.nan)
    print(f"\n2. Correlations by type:")
    print(f"   - Houses: r = {house_r:.4f}")
    print(f"   - Flats: r = {flat_r:.4f}")

    print("""
## Interpretation for Research Framework
""")

    print("Key findings to update in RESEARCH_FRAMEWORK.md:")
    print(f"  - Age-density correlation: {r_ad:.3f}")
    print(f"  - Age-residualized density effect: r = {r_resid:.4f}")
    print(f"  - House-flat interaction significant: {interact.get('significant', 'unknown')}")


def main() -> None:
    """Run stratified analysis for H4 and H5."""
    print("=" * 70)
    print("STRATIFIED ANALYSIS: H4 (Age Confounding) & H5 (Type Heterogeneity)")
    print("=" * 70)
    print(f"\nData: {DATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data()

    # H4: Age-density confounding
    h4_results = analyze_age_density_confounding(df)

    # H5: Building type heterogeneity
    h5_results = analyze_building_type_heterogeneity(df)

    # Detailed built form analysis
    detailed_results = analyze_detailed_building_types(df)

    # Summary
    summarize_stratified_findings(h4_results, h5_results)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("- Update WORKING_LOG.md with findings")
    print("- Update RESEARCH_FRAMEWORK.md 'Current Understanding' section")
    print("- Consider additional stratifications if patterns emerge")


if __name__ == "__main__":
    main()
