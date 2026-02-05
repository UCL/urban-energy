"""
Multi-level regression analysis for energy per capita.

Investigates the ASSOCIATION between urban density/compactness and energy per capita.

IMPORTANT NOTES ON INTERPRETATION:
- This is an OBSERVATIONAL study - we can identify associations, not causal effects
- The dependent variable (EPC energy) is SAP-MODELLED "potential" energy demand
  under standardized occupancy assumptions, NOT actual metered consumption
- Results describe building fabric efficiency under standard conditions,
  not behavioral differences in energy use

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

    # Create categorical attached_type from BUILT_FORM (more nuanced than continuous shared_wall_ratio)
    # This captures the distinct thermal behavior of different building configurations
    built_form_map = {
        "Detached": "detached",
        "Semi-Detached": "semi",
        "Mid-Terrace": "mid_terrace",
        "End-Terrace": "end_terrace",
        "Enclosed Mid-Terrace": "mid_terrace",
        "Enclosed End-Terrace": "end_terrace",
    }
    df["attached_type"] = df["BUILT_FORM"].map(built_form_map)
    # For flats, infer from property type
    is_flat = df["PROPERTY_TYPE"].str.lower().str.contains("flat", na=False)
    df.loc[is_flat, "attached_type"] = "flat"
    # Fill remaining with 'other'
    df["attached_type"] = df["attached_type"].fillna("other")
    print("  Attached type distribution:")
    print(df["attached_type"].value_counts().to_string())

    # EPC Fabric Efficiency Controls (1-5 scale, 5 = best)
    # These are critical controls for building thermal performance
    fabric_cols = {
        "WALLS_ENERGY_EFF": "walls_efficiency",
        "WINDOWS_ENERGY_EFF": "windows_efficiency",
        "MAINHEAT_ENERGY_EFF": "heating_efficiency",
        "ROOF_ENERGY_EFF": "roof_efficiency",
        "FLOOR_ENERGY_EFF": "floor_efficiency",
        "HOT_WATER_ENERGY_EFF": "hot_water_efficiency",
    }
    for epc_col, new_col in fabric_cols.items():
        if epc_col in df.columns:
            # Convert text ratings to numeric (Very Poor=1, Poor=2, Average=3, Good=4, Very Good=5)
            rating_map = {
                "Very Poor": 1, "Poor": 2, "Average": 3, "Good": 4, "Very Good": 5,
                "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
            }
            df[new_col] = df[epc_col].map(rating_map)
            n_valid = df[new_col].notna().sum()
            if n_valid > 0:
                print(f"  {new_col}: {n_valid:,} valid ({df[new_col].mean():.2f} mean)")
            else:
                # Try numeric conversion as fallback
                df[new_col] = pd.to_numeric(df[epc_col], errors="coerce")

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

    Model 1: Building controls (size, type, age, fabric efficiency)
    Model 2: + Morphology (compactness, shared walls)
    Model 3: + Network/accessibility
    Model 4: + Census demographics

    NOTE: These models identify ASSOCIATIONS, not causal effects.
    The dependent variable is SAP-modelled "potential" energy, not actual consumption.
    """
    print("\n" + "=" * 70)
    print("OLS REGRESSION MODELS (Progressive)")
    print("=" * 70)
    print("\nNOTE: Results show ASSOCIATIONS with potential energy demand")
    print("      (SAP-modelled), not causal effects on actual consumption.")

    # Prepare data - need complete cases for all variables
    model_vars = [
        "log_energy_per_capita",
        "log_floor_area",
        "property_type",
        "building_age",
        # EPC Fabric efficiency controls (critical for building physics)
        "walls_efficiency",
        "windows_efficiency",
        "heating_efficiency",
        # Morphology
        "compactness",
        "shared_wall_ratio",
        # Network
        "cc_harmonic_800",
        "cc_bus_800_nw",
        # Census
        "pop_density",
        "pct_owner_occupied",
    ]

    available = [v for v in model_vars if v in df.columns]
    model_df = df[available + ["LSOA21CD"]].dropna().copy()
    print(f"\nComplete cases for regression: {len(model_df)}")

    # Create dummy variables for property type
    model_df = pd.get_dummies(model_df, columns=["property_type"], drop_first=True)

    results = {}

    # Model 1: Building controls (including age and fabric efficiency)
    print("\n### Model 1: Building Controls (size, type, age, fabric efficiency)")
    formula1 = "log_energy_per_capita ~ log_floor_area"
    # Add property type dummies if they exist
    prop_cols = [c for c in model_df.columns if c.startswith("property_type_")]
    if prop_cols:
        formula1 += " + " + " + ".join(prop_cols)
    # Add building age
    if "building_age" in model_df.columns:
        formula1 += " + building_age"
    # Add fabric efficiency controls (critical for building physics)
    fabric_vars = ["walls_efficiency", "windows_efficiency", "heating_efficiency"]
    for var in fabric_vars:
        if var in model_df.columns:
            formula1 += f" + {var}"

    try:
        model1 = smf.ols(formula1, data=model_df).fit()
        results["M1_controls"] = model1
        print(f"  R² = {model1.rsquared:.3f}, Adj R² = {model1.rsquared_adj:.3f}")
        print(f"  log_floor_area: β = {model1.params.get('log_floor_area', 'N/A'):.3f}")
        if "building_age" in model1.params:
            print(f"  building_age: β = {model1.params.get('building_age', 'N/A'):.4f}")
        # Report fabric efficiency coefficients
        for var in fabric_vars:
            if var in model1.params:
                print(f"  {var}: β = {model1.params.get(var, 'N/A'):.3f}")
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


def compare_shared_wall_specifications(df: pd.DataFrame) -> None:
    """
    Compare continuous (shared_wall_ratio) vs categorical (attached_type) specification.

    The expert review suggested that treating all shared walls equally (continuous)
    may miss important distinctions between building types (detached, semi, terrace).
    """
    print("\n" + "=" * 70)
    print("SHARED WALL SPECIFICATION COMPARISON")
    print("=" * 70)
    print("\nComparing continuous (shared_wall_ratio) vs categorical (attached_type)")

    # Prepare data
    model_vars = [
        "log_energy_per_capita",
        "log_floor_area",
        "property_type",
        "shared_wall_ratio",
        "attached_type",
        "building_age",
        "LSOA21CD",
    ]

    available = [v for v in model_vars if v in df.columns]
    model_df = df[available].dropna().copy()

    print(f"\nComplete cases: {len(model_df)}")

    # Descriptive: mean energy by attached type
    print("\n### Mean Energy by Attached Type")
    print("-" * 50)
    summary = model_df.groupby("attached_type")["log_energy_per_capita"].agg(
        ["count", "mean", "std"]
    ).sort_values("mean", ascending=False)
    summary.columns = ["N", "Mean log(energy)", "SD"]
    print(summary.to_string())

    # Create dummies
    model_df_cont = pd.get_dummies(model_df, columns=["property_type"], drop_first=True)
    model_df_cat = pd.get_dummies(model_df, columns=["property_type", "attached_type"], drop_first=True)

    # Build formulas
    base_vars = ["log_floor_area", "building_age"]
    prop_cols = [c for c in model_df_cont.columns if c.startswith("property_type_")]
    base_vars.extend(prop_cols)

    # Model A: Continuous shared_wall_ratio
    formula_cont = "log_energy_per_capita ~ " + " + ".join(base_vars) + " + shared_wall_ratio"

    # Model B: Categorical attached_type
    attached_cols = [c for c in model_df_cat.columns if c.startswith("attached_type_")]
    formula_cat = "log_energy_per_capita ~ " + " + ".join(base_vars) + " + " + " + ".join(attached_cols)

    try:
        print("\n### Model A: Continuous (shared_wall_ratio)")
        model_cont = smf.ols(formula_cont, data=model_df_cont).fit()
        print(f"  R² = {model_cont.rsquared:.4f}")
        print(f"  AIC = {model_cont.aic:.1f}")
        print(f"  BIC = {model_cont.bic:.1f}")
        print(f"  shared_wall_ratio: β = {model_cont.params['shared_wall_ratio']:.4f} (p = {model_cont.pvalues['shared_wall_ratio']:.4f})")

        print("\n### Model B: Categorical (attached_type)")
        model_cat = smf.ols(formula_cat, data=model_df_cat).fit()
        print(f"  R² = {model_cat.rsquared:.4f}")
        print(f"  AIC = {model_cat.aic:.1f}")
        print(f"  BIC = {model_cat.bic:.1f}")
        print("\n  Attached type coefficients (reference = detached):")
        for col in attached_cols:
            coef = model_cat.params.get(col, np.nan)
            pval = model_cat.pvalues.get(col, np.nan)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            label = col.replace("attached_type_", "")
            print(f"    {label}: β = {coef:.4f} (p = {pval:.4f}) {sig}")

        # Compare
        print("\n### Comparison")
        print("-" * 50)
        delta_r2 = model_cat.rsquared - model_cont.rsquared
        delta_aic = model_cat.aic - model_cont.aic
        delta_bic = model_cat.bic - model_cont.bic

        print(f"  ΔR² (cat - cont):  {delta_r2:+.4f}")
        print(f"  ΔAIC (cat - cont): {delta_aic:+.1f}")
        print(f"  ΔBIC (cat - cont): {delta_bic:+.1f}")

        if delta_aic < -2:
            print("\n  Conclusion: Categorical specification is BETTER (lower AIC)")
        elif delta_aic > 2:
            print("\n  Conclusion: Continuous specification is BETTER (lower AIC)")
        else:
            print("\n  Conclusion: Models are SIMILAR (|ΔAIC| < 2)")

        print("\n  Note: Categorical allows different effects for mid- vs end-terrace,")
        print("        semi-detached vs flats, etc. - important for thermal physics.")

    except Exception as e:
        print(f"  Error: {e}")


def formal_model_selection(results: dict) -> None:
    """
    Perform formal model selection using AIC, BIC, and likelihood ratio tests.

    Compares nested models to determine which specification is preferred.
    """
    from scipy.stats import chi2

    print("\n" + "=" * 70)
    print("FORMAL MODEL SELECTION")
    print("=" * 70)

    if len(results) < 2:
        print("  Insufficient models for comparison")
        return

    # Extract model names and objects in order
    model_order = ["M1_controls", "M2_morphology", "M3_network", "M4_full"]
    available_models = [(name, results[name]) for name in model_order if name in results]

    if len(available_models) < 2:
        print("  Insufficient models for comparison")
        return

    # Print AIC/BIC comparison table
    print("\n### Information Criteria Comparison")
    print("-" * 70)
    print(f"  {'Model':<20} {'AIC':>12} {'BIC':>12} {'R²':>10} {'Adj R²':>10}")
    print("-" * 70)

    for name, model in available_models:
        print(f"  {name:<20} {model.aic:>12.1f} {model.bic:>12.1f} {model.rsquared:>10.3f} {model.rsquared_adj:>10.3f}")

    print("-" * 70)

    # Find best model by AIC and BIC
    best_aic = min(available_models, key=lambda x: x[1].aic)
    best_bic = min(available_models, key=lambda x: x[1].bic)

    print(f"\n  Best by AIC: {best_aic[0]} (AIC = {best_aic[1].aic:.1f})")
    print(f"  Best by BIC: {best_bic[0]} (BIC = {best_bic[1].bic:.1f})")

    if best_aic[0] != best_bic[0]:
        print("  Note: AIC and BIC disagree - AIC favors complexity, BIC favors parsimony")

    # Likelihood ratio tests for nested models
    print("\n### Likelihood Ratio Tests (Nested Models)")
    print("-" * 70)
    print(f"  {'Comparison':<30} {'LR Stat':>10} {'df':>6} {'p-value':>12} {'Significant':>12}")
    print("-" * 70)

    for i in range(len(available_models) - 1):
        name1, model1 = available_models[i]
        name2, model2 = available_models[i + 1]

        # LR = 2 * (LL_full - LL_reduced)
        lr_stat = 2 * (model2.llf - model1.llf)
        df_diff = model2.df_model - model1.df_model

        if df_diff > 0:
            p_value = chi2.sf(lr_stat, df_diff)
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

            comparison = f"{name1} vs {name2}"
            print(f"  {comparison:<30} {lr_stat:>10.2f} {df_diff:>6} {p_value:>12.4f} {sig:>12}")

    print("-" * 70)

    # Interpretation
    print("\n### Interpretation")
    print("""
  Model Selection Guidelines:
  - Lower AIC/BIC = better fit (penalized for complexity)
  - AIC tends to select more complex models
  - BIC tends to select simpler models (stronger complexity penalty)
  - LR test p < 0.05 suggests additional predictors significantly improve fit

  Recommendations:
  - If AIC and BIC agree: use that model
  - If they disagree: consider the research question
    - Prediction: prefer AIC (better out-of-sample)
    - Explanation: prefer BIC (more parsimonious)
  - Always check if added variables are substantively meaningful
""")


def fit_mixed_effects_model(df: pd.DataFrame) -> dict:
    """
    Fit multi-level model with random intercepts for LSOA.

    This accounts for the nested structure: UPRNs within LSOAs.

    Optimization strategy:
    - Try BFGS first (generally faster and more reliable)
    - Fall back to Powell if BFGS fails
    - Compare ML vs REML estimation
    """
    print("\n" + "=" * 70)
    print("MIXED EFFECTS MODEL (Random Intercepts)")
    print("=" * 70)

    # Prepare data
    model_vars = [
        "log_energy_per_capita",
        "log_floor_area",
        "property_type",
        "attached_type",  # Categorical attached type (refined from shared_wall_ratio)
        "compactness",
        "shared_wall_ratio",
        "cc_harmonic_800",
        "pop_density",
        "pct_deprived",  # For cross-level interactions
        "building_age",
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
        return {}

    # Create dummies for categorical variables
    if "property_type" in model_df.columns:
        model_df = pd.get_dummies(model_df, columns=["property_type"], drop_first=True)
    if "attached_type" in model_df.columns:
        model_df = pd.get_dummies(model_df, columns=["attached_type"], drop_first=True)

    # Build formula with refined attached_type instead of continuous shared_wall_ratio
    fixed_effects = ["log_floor_area"]
    prop_cols = [c for c in model_df.columns if c.startswith("property_type_")]
    fixed_effects.extend(prop_cols)

    # Add attached_type dummies (refined shared wall specification)
    attached_cols = [c for c in model_df.columns if c.startswith("attached_type_")]
    if attached_cols:
        fixed_effects.extend(attached_cols)
    elif "shared_wall_ratio" in model_df.columns:
        # Fall back to continuous if categorical not available
        fixed_effects.append("shared_wall_ratio")

    if "compactness" in model_df.columns:
        fixed_effects.append("compactness")
    if "cc_harmonic_800" in model_df.columns:
        fixed_effects.append("cc_harmonic_800")
    if "pop_density" in model_df.columns:
        fixed_effects.append("pop_density")

    formula = "log_energy_per_capita ~ " + " + ".join(fixed_effects)
    print(f"\nFormula: {formula}")

    results = {}

    def fit_with_fallback(formula, data, groups, reml=True, maxiter=500):
        """Try BFGS first, fall back to Powell if it fails."""
        method_name = "REML" if reml else "ML"
        model = smf.mixedlm(formula, data=data, groups=groups)

        # Try BFGS first (generally faster and more reliable)
        try:
            result = model.fit(method="bfgs", maxiter=maxiter, reml=reml)
            print(f"  {method_name}: Converged with BFGS")
            return result, "bfgs"
        except Exception as e:
            print(f"  {method_name}: BFGS failed ({e}), trying Powell...")

        # Fall back to Powell
        try:
            result = model.fit(method="powell", maxiter=maxiter, reml=reml)
            print(f"  {method_name}: Converged with Powell")
            return result, "powell"
        except Exception as e:
            print(f"  {method_name}: Powell also failed ({e})")
            return None, None

    # Fit with REML (default, better for variance component estimation)
    print("\n### Fitting with REML estimation...")
    reml_result, reml_method = fit_with_fallback(
        formula, model_df, model_df["LSOA21CD"], reml=True
    )

    # Fit with ML (needed for model comparison via likelihood ratio tests)
    print("\n### Fitting with ML estimation...")
    ml_result, ml_method = fit_with_fallback(
        formula, model_df, model_df["LSOA21CD"], reml=False
    )

    # Report results
    if reml_result is not None:
        results["reml"] = reml_result
        print("\n" + "-" * 70)
        print("REML RESULTS (preferred for inference)")
        print("-" * 70)
        print("\n### Fixed Effects")
        print(reml_result.summary().tables[1].to_string())

        # Compute ICC (Intraclass Correlation Coefficient)
        var_random = float(reml_result.cov_re.iloc[0, 0])
        var_residual = reml_result.scale
        icc = var_random / (var_random + var_residual)

        print("\n### Variance Components")
        print(f"  Between-LSOA variance: {var_random:.4f}")
        print(f"  Within-LSOA variance:  {var_residual:.4f}")
        print(f"  ICC: {icc:.3f} ({icc*100:.1f}% of variance is between LSOAs)")

        # Convergence diagnostics
        print("\n### Convergence Diagnostics")
        print(f"  Method: {reml_method.upper()}")
        print(f"  Converged: {reml_result.converged}")
        print(f"  Log-likelihood: {reml_result.llf:.2f}")

    # Compare ML vs REML
    if ml_result is not None and reml_result is not None:
        results["ml"] = ml_result
        print("\n" + "-" * 70)
        print("ML vs REML COMPARISON")
        print("-" * 70)
        print(f"  ML log-likelihood:   {ml_result.llf:.2f}")
        print(f"  REML log-likelihood: {reml_result.llf:.2f}")
        print("\n  Note: REML provides unbiased variance estimates but")
        print("        ML is needed for likelihood ratio tests between models")

    return results


def fit_cross_level_interactions(df: pd.DataFrame) -> dict:
    """
    Test cross-level interactions between building and neighborhood characteristics.

    Morphology effects may vary by neighborhood characteristics:
    - shared_wall_ratio × pct_deprived (party walls matter more in fuel poverty?)
    - compactness × pop_density (shape matters more in dense areas?)
    - building_age × pct_owner_occupied (age penalty varies by tenure mix?)

    Returns dict of model results for comparison.
    """
    print("\n" + "=" * 70)
    print("CROSS-LEVEL INTERACTIONS")
    print("=" * 70)
    print("\nTesting whether building-level effects vary by neighborhood context")

    # Prepare data
    model_vars = [
        "log_energy_per_capita",
        "log_floor_area",
        "property_type",
        "shared_wall_ratio",
        "compactness",
        "building_age",
        "pop_density",
        "pct_deprived",
        "pct_owner_occupied",
        "LSOA21CD",
    ]

    available = [v for v in model_vars if v in df.columns]
    model_df = df[available].dropna().copy()

    # Filter to LSOAs with sufficient observations
    lsoa_counts = model_df["LSOA21CD"].value_counts()
    valid_lsoas = lsoa_counts[lsoa_counts >= 5].index
    model_df = model_df[model_df["LSOA21CD"].isin(valid_lsoas)]

    print(f"\nObservations: {len(model_df)}")
    print(f"LSOAs: {model_df['LSOA21CD'].nunique()}")

    if len(model_df) < 100:
        print("  Insufficient data for interaction analysis")
        return {}

    # Create dummies
    model_df = pd.get_dummies(model_df, columns=["property_type"], drop_first=True)

    # Standardize continuous variables for interaction terms (easier interpretation)
    for var in ["shared_wall_ratio", "compactness", "building_age", "pop_density", "pct_deprived", "pct_owner_occupied"]:
        if var in model_df.columns:
            model_df[f"{var}_z"] = (model_df[var] - model_df[var].mean()) / model_df[var].std()

    results = {}

    # Base formula
    base_fixed = ["log_floor_area"]
    prop_cols = [c for c in model_df.columns if c.startswith("property_type_")]
    base_fixed.extend(prop_cols)
    base_fixed.extend(["shared_wall_ratio_z", "compactness_z", "building_age_z", "pop_density_z", "pct_deprived_z"])

    base_formula = "log_energy_per_capita ~ " + " + ".join(base_fixed)

    # Interaction 1: shared_wall_ratio × pct_deprived
    # Hypothesis: Party walls may matter more for households in fuel poverty
    print("\n### Interaction 1: Shared Walls × Deprivation")
    try:
        formula1 = base_formula + " + shared_wall_ratio_z:pct_deprived_z"
        model1 = smf.mixedlm(formula1, data=model_df, groups=model_df["LSOA21CD"])
        result1 = model1.fit(method="bfgs", maxiter=300, reml=True)

        interaction_coef = result1.params.get("shared_wall_ratio_z:pct_deprived_z", np.nan)
        interaction_pval = result1.pvalues.get("shared_wall_ratio_z:pct_deprived_z", np.nan)
        sig = "***" if interaction_pval < 0.001 else "**" if interaction_pval < 0.01 else "*" if interaction_pval < 0.05 else "ns"

        print(f"  Coefficient: {interaction_coef:.4f} (p = {interaction_pval:.4f}) {sig}")
        if interaction_coef < 0:
            print("  Interpretation: Shared walls associated with LARGER energy reduction in deprived areas")
        else:
            print("  Interpretation: Shared walls associated with SMALLER energy reduction in deprived areas")

        results["shared_wall_x_deprived"] = result1
    except Exception as e:
        print(f"  Error: {e}")

    # Interaction 2: compactness × pop_density
    # Hypothesis: Building shape may matter more in dense areas
    print("\n### Interaction 2: Compactness × Population Density")
    try:
        formula2 = base_formula + " + compactness_z:pop_density_z"
        model2 = smf.mixedlm(formula2, data=model_df, groups=model_df["LSOA21CD"])
        result2 = model2.fit(method="bfgs", maxiter=300, reml=True)

        interaction_coef = result2.params.get("compactness_z:pop_density_z", np.nan)
        interaction_pval = result2.pvalues.get("compactness_z:pop_density_z", np.nan)
        sig = "***" if interaction_pval < 0.001 else "**" if interaction_pval < 0.01 else "*" if interaction_pval < 0.05 else "ns"

        print(f"  Coefficient: {interaction_coef:.4f} (p = {interaction_pval:.4f}) {sig}")
        if interaction_coef < 0:
            print("  Interpretation: Compact buildings associated with LARGER energy reduction in dense areas")
        else:
            print("  Interpretation: Compact buildings associated with SMALLER energy reduction in dense areas")

        results["compactness_x_density"] = result2
    except Exception as e:
        print(f"  Error: {e}")

    # Interaction 3: building_age × pct_owner_occupied
    # Hypothesis: Age penalty may vary by tenure mix (owner-occupiers may retrofit more)
    print("\n### Interaction 3: Building Age × Owner-Occupation")
    try:
        formula3 = base_formula + " + building_age_z:pct_owner_occupied_z"
        model3 = smf.mixedlm(formula3, data=model_df, groups=model_df["LSOA21CD"])
        result3 = model3.fit(method="bfgs", maxiter=300, reml=True)

        interaction_coef = result3.params.get("building_age_z:pct_owner_occupied_z", np.nan)
        interaction_pval = result3.pvalues.get("building_age_z:pct_owner_occupied_z", np.nan)
        sig = "***" if interaction_pval < 0.001 else "**" if interaction_pval < 0.01 else "*" if interaction_pval < 0.05 else "ns"

        print(f"  Coefficient: {interaction_coef:.4f} (p = {interaction_pval:.4f}) {sig}")
        if interaction_coef < 0:
            print("  Interpretation: Age-energy association is WEAKER in owner-occupied areas (more retrofitting?)")
        else:
            print("  Interpretation: Age-energy association is STRONGER in owner-occupied areas")

        results["age_x_owner_occupied"] = result3
    except Exception as e:
        print(f"  Error: {e}")

    # Summary
    print("\n### Cross-Level Interaction Summary")
    print("-" * 70)
    print(f"  {'Interaction':<40} {'Coef':>10} {'p-value':>12} {'Sig':>6}")
    print("-" * 70)

    interactions = [
        ("Shared Walls × Deprivation", "shared_wall_x_deprived", "shared_wall_ratio_z:pct_deprived_z"),
        ("Compactness × Density", "compactness_x_density", "compactness_z:pop_density_z"),
        ("Building Age × Owner-Occupation", "age_x_owner_occupied", "building_age_z:pct_owner_occupied_z"),
    ]

    for label, key, param_name in interactions:
        if key in results:
            coef = results[key].params.get(param_name, np.nan)
            pval = results[key].pvalues.get(param_name, np.nan)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            print(f"  {label:<40} {coef:>10.4f} {pval:>12.4f} {sig:>6}")

    print("-" * 70)
    print("\n  Note: Standardized coefficients (mean=0, sd=1) for easier comparison")
    print("  Significant interactions suggest effects vary by neighborhood context")

    return results


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

    # Formal model selection (AIC/BIC, LR tests)
    formal_model_selection(ols_results)

    # Compare continuous vs categorical shared wall specification
    compare_shared_wall_specifications(df)

    # Mixed effects model (with BFGS optimization, ML vs REML comparison)
    fit_mixed_effects_model(df)

    # Cross-level interactions (morphology × neighborhood)
    fit_cross_level_interactions(df)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
IMPORTANT METHODOLOGICAL NOTES:

1. INTERPRETATION OF RESULTS
   - This analysis identifies ASSOCIATIONS, not causal effects
   - The dependent variable is SAP-MODELLED "potential" energy demand
   - SAP uses standardized occupancy assumptions (21°C, standard schedules)
   - Results reflect building FABRIC efficiency, not actual behavior

2. WHAT THE DEPENDENT VARIABLE CAPTURES
   - Building envelope performance (walls, windows, roof)
   - Heating system efficiency
   - Property size and configuration
   - Does NOT capture: behavioral differences, actual thermostat settings,
     occupancy patterns, fuel poverty (under-heating), lifestyle choices

3. DENSITY ASSOCIATION (not effect)
   - Higher density is ASSOCIATED with different building types
   - Flats common in dense areas → smaller, shared walls
   - After controlling for building type and age, density association is weak
   - This does NOT mean density "doesn't matter" for actual energy use

4. KEY CONFOUNDERS
   - Building age dominates: older buildings have worse fabric
   - Central areas have older building stock → apparent centrality "effect"
   - Fabric efficiency (walls, windows, heating) explains much variance

5. LIMITATIONS
   - Observational design: cannot establish causation
   - EPC selection bias: only transacted properties (see 00_selection_bias_analysis.py)
   - Spatial autocorrelation: residuals may cluster (see 02b_spatial_regression.py)
   - Single study area: results may not generalize

6. NEXT STEPS
   - Run SHAP analysis for non-linear effects
   - Test spatial autocorrelation in residuals
   - Subset analysis (houses only, post-2000 only)
   - Sensitivity analysis with different specifications
""")


if __name__ == "__main__":
    main()
