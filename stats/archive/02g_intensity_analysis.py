"""
Energy Intensity Analysis: Comparing per-capita vs per-m² metrics.

This script tests whether the house/flat divergence is a per-capita artifact
by re-running key analyses with energy INTENSITY (kWh/m²) as the dependent variable.

Key questions:
1. Does the house/flat divergence persist with intensity as DV?
2. Does the modern building (+1980) positive association persist?
3. What building characteristics explain modern building energy patterns?

Usage:
    uv run python stats/02g_intensity_analysis.py
"""


import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# Configuration
from urban_energy.paths import TEMP_DIR

DATA_PATH = TEMP_DIR / "processing" / "test" / "uprn_integrated.gpkg"
OUTPUT_DIR = TEMP_DIR / "stats"


def load_data() -> pd.DataFrame:
    """Load and prepare data with both DV options."""
    print("Loading data...")
    gdf = gpd.read_file(DATA_PATH)

    # Filter to UPRNs with EPC data
    df = gdf[gdf["CURRENT_ENERGY_EFFICIENCY"].notna()].copy()
    print(f"  Records with EPC: {len(df):,}")

    # Filter valid floor area
    df = df[(df["TOTAL_FLOOR_AREA"] > 0) & df["TOTAL_FLOOR_AREA"].notna()].copy()

    # Compute BOTH dependent variables
    # 1. Energy per capita (affected by household size)
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

    # 2. Energy intensity (kWh per m² - true thermal efficiency)
    df["energy_intensity"] = df["ENERGY_CONSUMPTION_CURRENT"] / df["TOTAL_FLOOR_AREA"]

    # Log transforms
    df["log_energy_per_capita"] = np.log(df["energy_per_capita"].clip(lower=1))
    df["log_energy_intensity"] = np.log(df["energy_intensity"].clip(lower=1))
    df["log_floor_area"] = np.log(df["TOTAL_FLOOR_AREA"])

    # Population density
    df["pop_density"] = df[
        "ts006_Population Density: Persons per square kilometre; measures: Value"
    ]

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

    # Construction era
    def categorize_era(year):
        if pd.isna(year):
            return np.nan
        elif year < 1919:
            return "Pre-1919"
        elif year < 1945:
            return "1919-1944"
        elif year < 1980:
            return "1945-1979"
        else:
            return "1980+"

    df["construction_era"] = df["construction_year"].apply(categorize_era)

    # Building type
    df["is_flat"] = df["PROPERTY_TYPE"].str.lower().str.contains("flat", na=False)
    df["is_house"] = ~df["is_flat"]
    df["building_type"] = np.where(df["is_flat"], "Flat", "House")

    # Building height (if available)
    if "height_mean" in df.columns:
        df["building_height"] = pd.to_numeric(df["height_mean"], errors="coerce")
    elif "height_median" in df.columns:
        df["building_height"] = pd.to_numeric(df["height_median"], errors="coerce")

    # Filter complete cases
    key_vars = [
        "log_energy_per_capita",
        "log_energy_intensity",
        "pop_density",
        "log_floor_area",
        "building_age",
        "building_type",
    ]
    df = df.dropna(subset=key_vars)
    print(f"  Complete cases: {len(df):,}")

    return df


def compare_dv_correlations(df: pd.DataFrame) -> dict:
    """Compare density correlations with both DVs."""
    print("\n" + "=" * 70)
    print("COMPARING DEPENDENT VARIABLES: Per Capita vs Intensity")
    print("=" * 70)

    results = {}

    # Overall correlations
    print("\n### Overall Density-Energy Correlations")
    print("-" * 60)
    print(f"{'DV':<25} {'r':>10} {'p':>12} {'Interpretation':<20}")
    print("-" * 60)

    for dv, label in [
        ("log_energy_per_capita", "Energy per capita"),
        ("log_energy_intensity", "Energy intensity (kWh/m²)"),
    ]:
        r, p = stats.pearsonr(df[dv], df["pop_density"])
        interp = "positive" if r > 0.05 else "negative" if r < -0.05 else "near zero"
        print(f"  {label:<23} {r:>10.4f} {p:>12.4f} {interp:<20}")
        results[f"overall_{dv}"] = {"r": r, "p": p}

    print("-" * 60)

    # By building type
    print("\n### By Building Type")
    print("-" * 70)
    print(f"{'Type':<10} {'DV':<25} {'N':>8} {'r':>10} {'p':>12}")
    print("-" * 70)

    for btype in ["House", "Flat"]:
        subset = df[df["building_type"] == btype]
        for dv, label in [
            ("log_energy_per_capita", "Per capita"),
            ("log_energy_intensity", "Intensity (kWh/m²)"),
        ]:
            r, p = stats.pearsonr(subset[dv], subset["pop_density"])
            print(f"  {btype:<8} {label:<23} {len(subset):>8,} {r:>10.4f} {p:>12.4f}")
            results[f"{btype}_{dv}"] = {"n": len(subset), "r": r, "p": p}

    print("-" * 70)

    # By construction era
    print("\n### By Construction Era")
    print("-" * 80)
    print(f"{'Era':<15} {'DV':<25} {'N':>8} {'r':>10} {'p':>12}")
    print("-" * 80)

    for era in ["Pre-1919", "1919-1944", "1945-1979", "1980+"]:
        subset = df[df["construction_era"] == era]
        if len(subset) < 100:
            continue
        for dv, label in [
            ("log_energy_per_capita", "Per capita"),
            ("log_energy_intensity", "Intensity"),
        ]:
            r, p = stats.pearsonr(subset[dv], subset["pop_density"])
            print(f"  {era:<13} {label:<23} {len(subset):>8,} {r:>10.4f} {p:>12.4f}")
            results[f"{era}_{dv}"] = {"n": len(subset), "r": r, "p": p}

    print("-" * 80)

    return results


def regression_comparison(df: pd.DataFrame) -> dict:
    """Run parallel regressions with both DVs."""
    print("\n" + "=" * 70)
    print("REGRESSION COMPARISON: Per Capita vs Intensity")
    print("=" * 70)

    results = {}

    # Prepare data
    model_df = df.dropna(subset=[
        "log_energy_per_capita", "log_energy_intensity",
        "pop_density", "log_floor_area", "building_age", "is_flat"
    ]).copy()

    print(f"\nSample size: {len(model_df):,}")

    # Convert is_flat to int for regression
    model_df["is_flat_int"] = model_df["is_flat"].astype(int)

    # Model specifications
    base_controls = "log_floor_area + building_age + is_flat_int"

    for dv, label in [
        ("log_energy_per_capita", "Per Capita"),
        ("log_energy_intensity", "Intensity"),
    ]:
        print(f"\n### {label} Models")
        print("-" * 60)

        # Simple model: just density
        formula1 = f"{dv} ~ pop_density"
        model1 = smf.ols(formula1, data=model_df).fit()

        # Full model: density + controls
        formula2 = f"{dv} ~ pop_density + {base_controls}"
        model2 = smf.ols(formula2, data=model_df).fit()

        print(f"  Simple (density only):")
        print(f"    β(density) = {model1.params['pop_density']:.6f} (p = {model1.pvalues['pop_density']:.4f})")
        print(f"    R² = {model1.rsquared:.4f}")

        print(f"\n  With controls (floor area, age, flat):")
        print(f"    β(density) = {model2.params['pop_density']:.6f} (p = {model2.pvalues['pop_density']:.4f})")
        print(f"    β(is_flat) = {model2.params['is_flat_int']:.4f} (p = {model2.pvalues['is_flat_int']:.4f})")
        print(f"    R² = {model2.rsquared:.4f}")

        results[f"{dv}_simple"] = {
            "beta_density": model1.params["pop_density"],
            "p_density": model1.pvalues["pop_density"],
            "r2": model1.rsquared,
        }
        results[f"{dv}_full"] = {
            "beta_density": model2.params["pop_density"],
            "p_density": model2.pvalues["pop_density"],
            "beta_flat": model2.params["is_flat_int"],
            "r2": model2.rsquared,
        }

    return results


def investigate_modern_buildings(df: pd.DataFrame) -> dict:
    """Deep dive into modern (1980+) buildings."""
    print("\n" + "=" * 70)
    print("MODERN BUILDINGS INVESTIGATION (1980+)")
    print("=" * 70)

    modern = df[df["construction_era"] == "1980+"].copy()
    print(f"\nModern buildings: {len(modern):,}")

    results = {}

    # 1. Composition by building type
    print("\n### 1. Building Type Composition")
    type_counts = modern["building_type"].value_counts()
    for btype, count in type_counts.items():
        pct = count / len(modern) * 100
        print(f"  {btype}: {count:,} ({pct:.1f}%)")
    results["composition"] = type_counts.to_dict()

    # 2. Property type breakdown
    print("\n### 2. Detailed Property Types")
    prop_counts = modern["PROPERTY_TYPE"].value_counts().head(10)
    for ptype, count in prop_counts.items():
        pct = count / len(modern) * 100
        print(f"  {ptype}: {count:,} ({pct:.1f}%)")

    # 3. Height comparison
    if "building_height" in modern.columns:
        print("\n### 3. Building Height by Type")
        height_stats = modern.groupby("building_type")["building_height"].agg(
            ["count", "mean", "median", "std"]
        )
        print(height_stats.to_string())
        results["height_stats"] = height_stats.to_dict()

        # Height-energy correlation
        print("\n### 4. Height-Energy Correlation (Modern Buildings)")
        for dv, label in [
            ("log_energy_per_capita", "Per capita"),
            ("log_energy_intensity", "Intensity"),
        ]:
            valid = modern[[dv, "building_height"]].dropna()
            if len(valid) > 30:
                r, p = stats.pearsonr(valid[dv], valid["building_height"])
                print(f"  {label}: r = {r:.4f} (p = {p:.4f})")
                results[f"height_{dv}"] = {"r": r, "p": p}

    # 4. Density-energy by type within modern era
    print("\n### 5. Density-Energy by Type (Modern Only)")
    print("-" * 70)
    print(f"{'Type':<10} {'DV':<25} {'N':>8} {'r':>10} {'β(density)':>12}")
    print("-" * 70)

    for btype in ["House", "Flat"]:
        subset = modern[modern["building_type"] == btype].dropna(
            subset=["log_energy_intensity", "pop_density", "log_floor_area", "building_age"]
        )
        if len(subset) < 100:
            continue

        for dv, label in [
            ("log_energy_per_capita", "Per capita"),
            ("log_energy_intensity", "Intensity"),
        ]:
            r, _ = stats.pearsonr(subset[dv], subset["pop_density"])

            # Regression
            model = smf.ols(
                f"{dv} ~ pop_density + log_floor_area + building_age",
                data=subset,
            ).fit()
            beta = model.params["pop_density"]

            print(f"  {btype:<8} {label:<23} {len(subset):>8,} {r:>10.4f} {beta:>12.6f}")
            results[f"modern_{btype}_{dv}"] = {"n": len(subset), "r": r, "beta": beta}

    print("-" * 70)

    # 5. Floor area comparison
    print("\n### 6. Floor Area by Type (Modern vs Pre-1980)")
    print("-" * 60)
    print(f"{'Era':<15} {'Type':<10} {'Mean m²':>10} {'Median m²':>10}")
    print("-" * 60)

    for era in ["Pre-1919", "1980+"]:
        era_df = df[df["construction_era"] == era]
        for btype in ["House", "Flat"]:
            subset = era_df[era_df["building_type"] == btype]
            if len(subset) > 0:
                mean_area = subset["TOTAL_FLOOR_AREA"].mean()
                median_area = subset["TOTAL_FLOOR_AREA"].median()
                print(f"  {era:<13} {btype:<8} {mean_area:>10.1f} {median_area:>10.1f}")

    print("-" * 60)

    # 6. Household size comparison
    print("\n### 7. Household Size by Type (Modern vs Pre-1980)")
    print("-" * 60)
    print(f"{'Era':<15} {'Type':<10} {'Mean HH size':>12}")
    print("-" * 60)

    for era in ["Pre-1919", "1980+"]:
        era_df = df[df["construction_era"] == era]
        for btype in ["House", "Flat"]:
            subset = era_df[era_df["building_type"] == btype]
            if len(subset) > 0:
                mean_hh = subset["avg_household_size"].mean()
                print(f"  {era:<13} {btype:<8} {mean_hh:>12.2f}")

    print("-" * 60)

    return results


def summarize_findings(corr_results: dict, reg_results: dict, modern_results: dict) -> None:
    """Summarize key findings."""
    print("\n" + "=" * 70)
    print("SUMMARY: ENERGY INTENSITY vs PER CAPITA")
    print("=" * 70)

    print("""
## Key Question: Is the house/flat divergence a per-capita artifact?
""")

    # Extract key comparisons
    house_pc = corr_results.get("House_log_energy_per_capita", {}).get("r", np.nan)
    house_int = corr_results.get("House_log_energy_intensity", {}).get("r", np.nan)
    flat_pc = corr_results.get("Flat_log_energy_per_capita", {}).get("r", np.nan)
    flat_int = corr_results.get("Flat_log_energy_intensity", {}).get("r", np.nan)

    print("### Density-Energy Correlations by Building Type")
    print("-" * 50)
    print(f"{'Metric':<25} {'Houses':>12} {'Flats':>12}")
    print("-" * 50)
    print(f"  {'Per capita':<23} {house_pc:>12.4f} {flat_pc:>12.4f}")
    print(f"  {'Intensity (kWh/m²)':<23} {house_int:>12.4f} {flat_int:>12.4f}")
    print("-" * 50)

    print("\n### Interpretation")

    if flat_int < 0 or abs(flat_int) < abs(flat_pc):
        print("""
✓ The per-capita artifact CONFIRMED:
  - With intensity as DV, flat correlation is less positive (or negative)
  - The positive flat-density association is largely a household size effect
  - Flats have smaller households → inflated per-capita values
""")
    else:
        print("""
✗ Per-capita artifact NOT the full explanation:
  - Flat correlation remains positive even with intensity
  - There may be real thermal efficiency issues with dense flat developments
  - Could be related to building form (high-rise, glass facades)
""")

    # Modern buildings
    modern_house_int = modern_results.get("modern_House_log_energy_intensity", {}).get("r", np.nan)
    modern_flat_int = modern_results.get("modern_Flat_log_energy_intensity", {}).get("r", np.nan)

    print("\n### Modern Buildings (1980+)")
    print(f"  House density-intensity: r = {modern_house_int:.4f}")
    print(f"  Flat density-intensity: r = {modern_flat_int:.4f}")

    if modern_flat_int > 0.05:
        print("""
  → Modern flats still show positive density-intensity association
  → Possible explanations:
    - High-rise construction (more external wall per floor area)
    - Glass facades (poor thermal performance)
    - Central heating systems with less individual control
    - Different building regulations/standards
""")


def main() -> None:
    """Run energy intensity analysis."""
    print("=" * 70)
    print("ENERGY INTENSITY ANALYSIS")
    print("=" * 70)
    print("\nComparing 'energy per capita' vs 'energy intensity (kWh/m²)'")
    print(f"\nData: {DATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data()

    # Compare DV correlations
    corr_results = compare_dv_correlations(df)

    # Regression comparison
    reg_results = regression_comparison(df)

    # Modern buildings investigation
    modern_results = investigate_modern_buildings(df)

    # Summary
    summarize_findings(corr_results, reg_results, modern_results)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nUpdate WORKING_LOG.md and RESEARCH_FRAMEWORK.md with findings")


if __name__ == "__main__":
    main()
