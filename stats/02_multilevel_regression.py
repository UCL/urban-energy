"""
Multi-level regression analysis for energy per capita.

Tests the hypothesis: Higher urban density/compactness → Lower energy per capita

This script implements Phase 3 of the statistical workflow:
- Compute energy per capita using Census household size data
- Fit multi-level models with UPRN nested within LSOA
- Progressive model building (null → controls → morphology → network)
- Variance decomposition (ICC at each level)

Usage:
    uv run python stats/02_multilevel_regression.py
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "temp" / "processing" / "test" / "uprn_integrated.gpkg"
OUTPUT_DIR = BASE_DIR / "temp" / "stats"


def load_and_prepare_data() -> pd.DataFrame:
    """
    Load UPRN data and compute energy per capita.

    Uses Census household size distribution to estimate average
    occupants per dwelling at the OA level.
    """
    print("Loading data...")
    gdf = gpd.read_file(DATA_PATH)
    print(f"  Total UPRNs: {len(gdf):,}")

    # Filter to UPRNs with EPC data
    df = gdf[gdf["CURRENT_ENERGY_EFFICIENCY"].notna()].copy()
    print(f"  With EPC data: {len(df):,}")

    # Filter out records with invalid floor area (zero or negative)
    valid_area = (df["TOTAL_FLOOR_AREA"] > 0) & df["TOTAL_FLOOR_AREA"].notna()
    n_invalid = (~valid_area).sum()
    if n_invalid > 0:
        print(f"  Removed {n_invalid:,} records with invalid floor area")
        df = df[valid_area].copy()

    # Compute average household size per OA from Census
    # Weighted average: sum(size * count) / total_households
    print("\nComputing average household size per OA...")

    # Get household counts by size
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

    # Compute weighted sum of people
    total_people = sum(size * df[col] for size, col in size_cols.items())
    total_households = df[
        "ts017_Household size: Total: All household spaces; measures: Value"
    ] - df["ts017_Household size: 0 people in household; measures: Value"]

    df["avg_household_size"] = total_people / total_households
    print(f"  Mean household size: {df['avg_household_size'].mean():.2f}")

    # Compute energy per capita (annual kWh per person)
    df["energy_per_capita"] = df["ENERGY_CONSUMPTION_CURRENT"] / df["avg_household_size"]
    print(f"  Mean energy per capita: {df['energy_per_capita'].mean():.0f} kWh/person/year")

    # Also compute energy intensity for comparison
    df["energy_intensity_kwh_m2"] = (
        df["ENERGY_CONSUMPTION_CURRENT"] / df["TOTAL_FLOOR_AREA"]
    )

    # Filter out records with invalid energy values (inf/nan)
    valid_energy = (
        np.isfinite(df["energy_per_capita"])
        & np.isfinite(df["energy_intensity_kwh_m2"])
        & (df["energy_per_capita"] > 0)
        & (df["energy_intensity_kwh_m2"] > 0)
    )
    n_invalid = (~valid_energy).sum()
    if n_invalid > 0:
        print(f"  Removed {n_invalid:,} records with invalid energy values")
        df = df[valid_energy].copy()

    # Log transforms (often better for regression)
    df["log_energy_per_capita"] = np.log(df["energy_per_capita"])
    df["log_energy_intensity"] = np.log(df["energy_intensity_kwh_m2"])
    df["log_floor_area"] = np.log(df["TOTAL_FLOOR_AREA"])

    # Census derived variables
    total_hh = df["ts011_Household deprivation: Total: All households; measures: Value"]
    df["pct_deprived"] = (
        (
            df["ts011_Household deprivation: Household is deprived in one dimension; measures: Value"]
            + df["ts011_Household deprivation: Household is deprived in two dimensions; measures: Value"]
            + df["ts011_Household deprivation: Household is deprived in three dimensions; measures: Value"]
            + df["ts011_Household deprivation: Household is deprived in four dimensions; measures: Value"]
        )
        / total_hh
        * 100
    )

    tenure_total = df["ts054_Tenure of household: Total: All households"]
    df["pct_owner_occupied"] = (
        df["ts054_Tenure of household: Owned"] / tenure_total * 100
    )

    df["pop_density"] = df[
        "ts006_Population Density: Persons per square kilometre; measures: Value"
    ]

    # Travel to work mode (ts061) - compute percentages
    travel_total = df[
        "ts061_Method of travel to workplace: Total: All usual residents aged 16 years and over in employment the week before the census"
    ]
    df["pct_car_commute"] = (
        df["ts061_Method of travel to workplace: Driving a car or van"] / travel_total * 100
    )
    df["pct_active_travel"] = (
        (df["ts061_Method of travel to workplace: Bicycle"] +
         df["ts061_Method of travel to workplace: On foot"]) / travel_total * 100
    )
    df["pct_public_transport"] = (
        (df["ts061_Method of travel to workplace: Underground, metro, light rail, tram"] +
         df["ts061_Method of travel to workplace: Train"] +
         df["ts061_Method of travel to workplace: Bus, minibus or coach"]) / travel_total * 100
    )

    # Building height (convert to numeric if needed)
    if "height_mean" in df.columns:
        df["building_height"] = pd.to_numeric(df["height_mean"], errors="coerce")

    # Clean categorical variables for regression
    df["property_type"] = df["PROPERTY_TYPE"].fillna("Unknown")
    df["built_form"] = df["BUILT_FORM"].fillna("Unknown")

    # Standardise fuel type
    df["main_fuel"] = df["MAIN_FUEL"].apply(
        lambda x: "gas"
        if "gas" in str(x).lower()
        else ("electric" if "electric" in str(x).lower() else "other")
    )

    # Convert construction age band to numeric (midpoint year)
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

    def parse_age_band(x):
        if pd.isna(x) or x == "NO DATA!":
            return np.nan
        if x in age_band_to_year:
            return age_band_to_year[x]
        try:
            return int(x)
        except (ValueError, TypeError):
            return np.nan

    df["construction_year"] = df["CONSTRUCTION_AGE_BAND"].apply(parse_age_band)
    df["building_age"] = 2024 - df["construction_year"]
    print(f"  With building age: {df['building_age'].notna().sum():,}")

    # Filter to complete cases for key variables
    key_vars = [
        "energy_per_capita",
        "TOTAL_FLOOR_AREA",
        "property_type",
        "LSOA21CD",
        "pop_density",
    ]
    complete = df[key_vars].notna().all(axis=1)
    df = df[complete].copy()
    print(f"\n  Complete cases: {len(df):,}")

    return df


def descriptive_stats(df: pd.DataFrame) -> None:
    """Print descriptive statistics for key variables."""
    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)

    print("\n### Energy Variables")
    energy_vars = ["ENERGY_CONSUMPTION_CURRENT", "energy_per_capita", "energy_intensity_kwh_m2"]
    print(df[energy_vars].describe().T.to_string())

    print("\n### Energy per capita by property type")
    summary = df.groupby("property_type")["energy_per_capita"].agg(
        ["count", "mean", "std", "median"]
    )
    print(summary.sort_values("mean", ascending=False).to_string())

    print("\n### Energy per capita by built form")
    summary = df.groupby("built_form")["energy_per_capita"].agg(
        ["count", "mean", "std", "median"]
    )
    print(summary.sort_values("mean", ascending=False).to_string())


def correlation_analysis(df: pd.DataFrame) -> None:
    """Compute correlations with energy per capita."""
    print("\n" + "=" * 70)
    print("CORRELATIONS WITH ENERGY PER CAPITA")
    print("=" * 70)

    target = "energy_per_capita"

    variables = [
        # Building
        ("TOTAL_FLOOR_AREA", "Floor area (m²)"),
        ("avg_household_size", "Avg household size"),
        ("building_age", "Building age (years)"),
        # Morphology
        ("footprint_area_m2", "Building footprint"),
        ("compactness", "Building compactness"),
        ("convexity", "Building convexity"),
        ("shared_wall_ratio", "Shared wall ratio"),
        # Network centrality
        ("cc_harmonic_800", "Network integration (800m)"),
        ("cc_betweenness_800", "Betweenness (800m)"),
        # Accessibility
        ("cc_fsa_restaurant_800_nw", "Restaurant access (800m)"),
        ("cc_bus_800_nw", "Bus stop access (800m)"),
        ("cc_greenspace_800_nw", "Green space access (800m)"),
        # Census
        ("pop_density", "Population density"),
        ("pct_owner_occupied", "% Owner occupied"),
        ("pct_deprived", "% Deprived"),
        # Travel to work
        ("pct_car_commute", "% Car commute"),
        ("pct_active_travel", "% Active travel (walk/cycle)"),
        ("pct_public_transport", "% Public transport"),
        # Building height
        ("building_height", "Building height (m)"),
    ]

    print(f"\nCorrelations with {target}:")
    for var, label in variables:
        if var in df.columns:
            valid = df[[target, var]].dropna()
            if len(valid) > 10:
                r, p = stats.pearsonr(valid[target], valid[var])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {label:35} r = {r:+.3f} {sig}")


def fit_ols_models(df: pd.DataFrame) -> dict:
    """
    Fit progressive OLS models to understand variable contributions.

    Model 1: Building controls only
    Model 2: + Morphology
    Model 3: + Network/accessibility
    Model 4: + Census demographics
    """
    print("\n" + "=" * 70)
    print("OLS REGRESSION MODELS (Progressive)")
    print("=" * 70)

    # Prepare data - need complete cases for all variables
    model_vars = [
        "log_energy_per_capita",
        "log_floor_area",
        "property_type",
        "building_age",
        "compactness",
        "shared_wall_ratio",
        "cc_harmonic_800",
        "cc_bus_800_nw",
        "pop_density",
        "pct_owner_occupied",
    ]

    available = [v for v in model_vars if v in df.columns]
    model_df = df[available + ["LSOA21CD"]].dropna().copy()
    print(f"\nComplete cases for regression: {len(model_df)}")

    # Create dummy variables for property type
    model_df = pd.get_dummies(model_df, columns=["property_type"], drop_first=True)

    results = {}

    # Model 1: Building controls (including age)
    print("\n### Model 1: Building Controls (size, type, age)")
    formula1 = "log_energy_per_capita ~ log_floor_area"
    # Add property type dummies if they exist
    prop_cols = [c for c in model_df.columns if c.startswith("property_type_")]
    if prop_cols:
        formula1 += " + " + " + ".join(prop_cols)
    # Add building age
    if "building_age" in model_df.columns:
        formula1 += " + building_age"

    try:
        model1 = smf.ols(formula1, data=model_df).fit()
        results["M1_controls"] = model1
        print(f"  R² = {model1.rsquared:.3f}, Adj R² = {model1.rsquared_adj:.3f}")
        print(f"  log_floor_area: β = {model1.params.get('log_floor_area', 'N/A'):.3f}")
        if "building_age" in model1.params:
            print(f"  building_age: β = {model1.params.get('building_age', 'N/A'):.4f}")
    except Exception as e:
        print(f"  Error: {e}")

    # Model 2: + Morphology
    print("\n### Model 2: + Morphology")
    if "compactness" in model_df.columns and "shared_wall_ratio" in model_df.columns:
        formula2 = formula1 + " + compactness + shared_wall_ratio"
        try:
            model2 = smf.ols(formula2, data=model_df).fit()
            results["M2_morphology"] = model2
            print(f"  R² = {model2.rsquared:.3f}, Adj R² = {model2.rsquared_adj:.3f}")
            print(f"  compactness: β = {model2.params.get('compactness', 'N/A'):.3f}")
            print(f"  shared_wall_ratio: β = {model2.params.get('shared_wall_ratio', 'N/A'):.3f}")
            print(f"  ΔR² from M1: {model2.rsquared - model1.rsquared:.3f}")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("  Morphology variables not available")

    # Model 3: + Network
    print("\n### Model 3: + Network/Accessibility")
    if "cc_harmonic_800" in model_df.columns:
        formula3 = formula2 + " + cc_harmonic_800 + cc_bus_800_nw"
        try:
            model3 = smf.ols(formula3, data=model_df).fit()
            results["M3_network"] = model3
            print(f"  R² = {model3.rsquared:.3f}, Adj R² = {model3.rsquared_adj:.3f}")
            print(f"  cc_harmonic_800: β = {model3.params.get('cc_harmonic_800', 'N/A'):.3f}")
            print(f"  cc_bus_800_nw: β = {model3.params.get('cc_bus_800_nw', 'N/A'):.3f}")
            print(f"  ΔR² from M2: {model3.rsquared - model2.rsquared:.3f}")
        except Exception as e:
            print(f"  Error: {e}")

    # Model 4: + Census
    print("\n### Model 4: + Census Demographics")
    if "pop_density" in model_df.columns:
        formula4 = formula3 + " + pop_density + pct_owner_occupied"
        try:
            model4 = smf.ols(formula4, data=model_df).fit()
            results["M4_full"] = model4
            print(f"  R² = {model4.rsquared:.3f}, Adj R² = {model4.rsquared_adj:.3f}")
            print(f"  pop_density: β = {model4.params.get('pop_density', 'N/A'):.6f}")
            print(f"  pct_owner_occupied: β = {model4.params.get('pct_owner_occupied', 'N/A'):.3f}")
            print(f"  ΔR² from M3: {model4.rsquared - model3.rsquared:.3f}")
        except Exception as e:
            print(f"  Error: {e}")

    return results


def fit_mixed_effects_model(df: pd.DataFrame) -> None:
    """
    Fit multi-level model with random intercepts for LSOA.

    This accounts for the nested structure: UPRNs within LSOAs.
    """
    print("\n" + "=" * 70)
    print("MIXED EFFECTS MODEL (Random Intercepts)")
    print("=" * 70)

    # Prepare data
    model_vars = [
        "log_energy_per_capita",
        "log_floor_area",
        "property_type",
        "compactness",
        "shared_wall_ratio",
        "cc_harmonic_800",
        "pop_density",
        "LSOA21CD",
    ]

    available = [v for v in model_vars if v in df.columns]
    model_df = df[available].dropna().copy()

    # Need at least 2 observations per LSOA
    lsoa_counts = model_df["LSOA21CD"].value_counts()
    valid_lsoas = lsoa_counts[lsoa_counts >= 2].index
    model_df = model_df[model_df["LSOA21CD"].isin(valid_lsoas)]

    print(f"\nObservations: {len(model_df)}")
    print(f"LSOAs: {model_df['LSOA21CD'].nunique()}")

    if len(model_df) < 50:
        print("  Insufficient data for mixed effects model")
        return

    # Create dummies
    model_df = pd.get_dummies(model_df, columns=["property_type"], drop_first=True)

    # Build formula
    fixed_effects = ["log_floor_area"]
    prop_cols = [c for c in model_df.columns if c.startswith("property_type_")]
    fixed_effects.extend(prop_cols)

    if "compactness" in model_df.columns:
        fixed_effects.append("compactness")
    if "shared_wall_ratio" in model_df.columns:
        fixed_effects.append("shared_wall_ratio")
    if "cc_harmonic_800" in model_df.columns:
        fixed_effects.append("cc_harmonic_800")
    if "pop_density" in model_df.columns:
        fixed_effects.append("pop_density")

    formula = "log_energy_per_capita ~ " + " + ".join(fixed_effects)
    print(f"\nFormula: {formula}")

    try:
        # Fit mixed model with LSOA random intercept
        model = smf.mixedlm(
            formula,
            data=model_df,
            groups=model_df["LSOA21CD"],
        ).fit(method="powell")

        print("\n### Fixed Effects")
        print(model.summary().tables[1].to_string())

        # Compute ICC (Intraclass Correlation Coefficient)
        var_random = float(model.cov_re.iloc[0, 0])
        var_residual = model.scale
        icc = var_random / (var_random + var_residual)

        print(f"\n### Variance Components")
        print(f"  Between-LSOA variance: {var_random:.4f}")
        print(f"  Within-LSOA variance:  {var_residual:.4f}")
        print(f"  ICC: {icc:.3f} ({icc*100:.1f}% of variance is between LSOAs)")

    except Exception as e:
        print(f"  Error fitting mixed model: {e}")
        import traceback
        traceback.print_exc()


def test_density_hypothesis(df: pd.DataFrame) -> None:
    """
    Directly test the hypothesis: Does density reduce energy per capita?

    Compare energy per capita across density terciles.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS TEST: Density → Energy per Capita")
    print("=" * 70)

    # Create density terciles
    df["density_tercile"] = pd.qcut(
        df["pop_density"],
        q=3,
        labels=["Low", "Medium", "High"],
    )

    print("\n### Energy per Capita by Population Density Tercile")
    summary = df.groupby("density_tercile")["energy_per_capita"].agg(
        ["count", "mean", "std", "median"]
    )
    print(summary.to_string())

    # ANOVA test
    groups = [g["energy_per_capita"].values for _, g in df.groupby("density_tercile")]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"\n  ANOVA: F = {f_stat:.2f}, p = {p_val:.4f}")

    # Effect size (eta-squared)
    grand_mean = df["energy_per_capita"].mean()
    ss_between = sum(
        len(g) * (g["energy_per_capita"].mean() - grand_mean) ** 2
        for _, g in df.groupby("density_tercile")
    )
    ss_total = ((df["energy_per_capita"] - grand_mean) ** 2).sum()
    eta_squared = ss_between / ss_total
    print(f"  Effect size (η²): {eta_squared:.3f}")

    # Same for network centrality
    if "cc_harmonic_800" in df.columns:
        df["centrality_tercile"] = pd.qcut(
            df["cc_harmonic_800"],
            q=3,
            labels=["Low", "Medium", "High"],
        )

        print("\n### Energy per Capita by Network Centrality Tercile")
        summary = df.groupby("centrality_tercile")["energy_per_capita"].agg(
            ["count", "mean", "std", "median"]
        )
        print(summary.to_string())

        groups = [g["energy_per_capita"].values for _, g in df.groupby("centrality_tercile")]
        f_stat, p_val = stats.f_oneway(*groups)
        print(f"\n  ANOVA: F = {f_stat:.2f}, p = {p_val:.4f}")


def main() -> None:
    """Run multi-level regression analysis."""
    print("=" * 70)
    print("MULTI-LEVEL REGRESSION: Energy per Capita")
    print("=" * 70)
    print("\nHypothesis: Higher density/compactness → Lower energy per capita")
    print(f"\nData: {DATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    df = load_and_prepare_data()

    # Descriptive statistics
    descriptive_stats(df)

    # Correlations
    correlation_analysis(df)

    # Test hypothesis directly
    test_density_hypothesis(df)

    # OLS models (progressive)
    ols_results = fit_ols_models(df)

    # Mixed effects model
    fit_mixed_effects_model(df)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key findings to investigate:

1. ENERGY PER CAPITA vs ENERGY INTENSITY
   - Energy intensity (kWh/m²) may not capture the per-capita hypothesis
   - Smaller dwellings have higher intensity but may have lower per-capita use

2. DENSITY EFFECT
   - Does higher population density correlate with LOWER energy per capita?
   - This would support the compact city hypothesis

3. BUILDING TYPE CONFOUNDING
   - Flats are more common in dense areas
   - Flats have smaller floor areas → potentially lower per-capita energy
   - Need to control for building type to isolate density effect

4. NETWORK CENTRALITY
   - More central/accessible locations may have lower car use
   - But this may not directly affect building energy

5. NEXT STEPS
   - Run SHAP analysis to detect non-linear effects
   - Check if results hold for houses-only subset
   - Consider spatial autocorrelation
""")


if __name__ == "__main__":
    main()
