"""
Sensitivity analysis for energy-morphology associations.

This script tests the robustness of findings to:
1. Specification choices (log vs level, different DVs)
2. Sample restrictions (houses only, post-2000 only, gas-heated only)
3. Unmeasured confounding (E-value analysis)
4. Coefficient stability across model specifications

Usage:
    uv run python stats/02c_sensitivity_analysis.py
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


def load_and_prepare_data() -> pd.DataFrame:
    """Load UPRN data and prepare for analysis."""
    print("Loading data...")
    gdf = gpd.read_file(DATA_PATH)
    print(f"  Total UPRNs: {len(gdf):,}")

    # Filter to UPRNs with EPC data
    df = gdf[gdf["CURRENT_ENERGY_EFFICIENCY"].notna()].copy()
    print(f"  With EPC data: {len(df):,}")

    # Filter out records with invalid floor area
    valid_area = (df["TOTAL_FLOOR_AREA"] > 0) & df["TOTAL_FLOOR_AREA"].notna()
    df = df[valid_area].copy()

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
    # Energy metrics — ECC is already kWh/m²/year
    df["energy_intensity"] = df["ENERGY_CONSUMPTION_CURRENT"]
    df["total_energy_kwh"] = df["ENERGY_CONSUMPTION_CURRENT"] * df["TOTAL_FLOOR_AREA"]
    df["energy_per_capita"] = df["total_energy_kwh"] / df["avg_household_size"]

    # Filter valid energy values
    valid_energy = (
        np.isfinite(df["energy_per_capita"]) &
        np.isfinite(df["energy_intensity"]) &
        (df["energy_per_capita"] > 0) &
        (df["energy_intensity"] > 0)
    )
    df = df[valid_energy].copy()

    # Log transforms
    df["log_energy_per_capita"] = np.log(df["energy_per_capita"])
    df["log_energy_intensity"] = np.log(df["energy_intensity"])
    df["log_floor_area"] = np.log(df["TOTAL_FLOOR_AREA"])

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

    # Census variables
    df["pop_density"] = df[
        "ts006_Population Density: Persons per square kilometre; measures: Value"
    ]

    # Standardise fuel type
    df["main_fuel"] = df["MAIN_FUEL"].apply(
        lambda x: "gas" if "gas" in str(x).lower()
        else ("electric" if "electric" in str(x).lower() else "other")
    )

    # Property type
    df["property_type"] = df["PROPERTY_TYPE"].fillna("Unknown")

    print(f"  Analysis sample: {len(df):,}")
    return df


def test_log_transformation(df: pd.DataFrame) -> dict:
    """
    Validate log transformation assumptions.

    Checks:
    1. No zeros or negatives in energy variable
    2. Compares residual distributions (log vs level)
    3. Tests for heteroscedasticity
    """
    print("\n" + "=" * 70)
    print("LOG TRANSFORMATION VALIDATION")
    print("=" * 70)

    results = {}

    # Check for zeros/negatives
    n_zero = (df["energy_per_capita"] <= 0).sum()
    n_negative = (df["energy_per_capita"] < 0).sum()
    print(f"\n  Zeros in energy_per_capita: {n_zero}")
    print(f"  Negatives in energy_per_capita: {n_negative}")

    if n_zero > 0 or n_negative > 0:
        print("  WARNING: Log transformation may be invalid!")
        results["log_valid"] = False
    else:
        print("  ✓ Log transformation is valid (no zeros/negatives)")
        results["log_valid"] = True

    # Compare distributions
    print("\n  Distribution comparison:")
    print(f"    Energy per capita (level):")
    print(f"      Mean: {df['energy_per_capita'].mean():.1f}")
    print(f"      Median: {df['energy_per_capita'].median():.1f}")
    print(f"      Skewness: {df['energy_per_capita'].skew():.2f}")

    print(f"    Log energy per capita:")
    print(f"      Mean: {df['log_energy_per_capita'].mean():.2f}")
    print(f"      Median: {df['log_energy_per_capita'].median():.2f}")
    print(f"      Skewness: {df['log_energy_per_capita'].skew():.2f}")

    results["level_skewness"] = df["energy_per_capita"].skew()
    results["log_skewness"] = df["log_energy_per_capita"].skew()

    # Fit models and compare residual distributions
    model_vars = ["log_floor_area", "building_age", "pop_density"]
    available = [v for v in model_vars if v in df.columns]
    model_df = df[["energy_per_capita", "log_energy_per_capita"] + available].dropna()

    if len(model_df) > 100:
        formula = "energy_per_capita ~ " + " + ".join(available)
        formula_log = "log_energy_per_capita ~ " + " + ".join(available)

        model_level = smf.ols(formula, data=model_df).fit()
        model_log = smf.ols(formula_log, data=model_df).fit()

        # Shapiro-Wilk test on residuals (sample if large)
        sample_size = min(5000, len(model_df))
        resid_level_sample = np.random.choice(model_level.resid, sample_size, replace=False)
        resid_log_sample = np.random.choice(model_log.resid, sample_size, replace=False)

        _, p_level = stats.shapiro(resid_level_sample)
        _, p_log = stats.shapiro(resid_log_sample)

        print(f"\n  Residual normality (Shapiro-Wilk, n={sample_size}):")
        print(f"    Level model: p = {p_level:.4f}")
        print(f"    Log model: p = {p_log:.4f}")

        results["resid_normality_level_p"] = p_level
        results["resid_normality_log_p"] = p_log

        # R² comparison
        print(f"\n  Model fit comparison:")
        print(f"    Level model R²: {model_level.rsquared:.3f}")
        print(f"    Log model R²: {model_log.rsquared:.3f}")

        results["r2_level"] = model_level.rsquared
        results["r2_log"] = model_log.rsquared

    # Recommendation
    print("\n  Recommendation:")
    if abs(results.get("log_skewness", 0)) < abs(results.get("level_skewness", 0)):
        print("    → Log transformation reduces skewness - RECOMMENDED")
    else:
        print("    → Level specification may be adequate")

    return results


def compare_specifications(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare coefficient estimates across different DV specifications.

    Tests:
    1. Log energy per capita (current)
    2. Level energy per capita
    3. Log energy intensity (kWh/m²)
    4. SAP score (1-100)
    """
    print("\n" + "=" * 70)
    print("SPECIFICATION COMPARISON")
    print("=" * 70)

    # Prepare model data
    model_vars = [
        "log_floor_area", "building_age", "compactness",
        "shared_wall_ratio", "cc_harmonic_800", "pop_density",
    ]
    available = [v for v in model_vars if v in df.columns]

    # DVs to test
    dvs = {
        "log_energy_per_capita": "Log Energy/Capita",
        "energy_per_capita": "Energy/Capita (level)",
        "log_energy_intensity": "Log Energy/m²",
        "CURRENT_ENERGY_EFFICIENCY": "SAP Score (1-100)",
    }

    results = []

    for dv, dv_label in dvs.items():
        if dv not in df.columns:
            continue

        model_df = df[[dv] + available].dropna()
        if len(model_df) < 100:
            continue

        formula = f"{dv} ~ " + " + ".join(available)

        try:
            model = smf.ols(formula, data=model_df).fit()

            print(f"\n### {dv_label} (n={len(model_df):,})")
            print(f"    R² = {model.rsquared:.3f}")

            for var in available:
                coef = model.params.get(var, np.nan)
                se = model.bse.get(var, np.nan)
                p = model.pvalues.get(var, np.nan)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

                results.append({
                    "dv": dv_label,
                    "variable": var,
                    "coefficient": coef,
                    "std_error": se,
                    "p_value": p,
                    "significant": sig,
                })

                print(f"    {var:25} β = {coef:+.4f} {sig}")

        except Exception as e:
            print(f"    Error fitting {dv_label}: {e}")

    return pd.DataFrame(results)


def subset_analyses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test robustness across sample subsets.

    Subsets:
    1. Houses only (exclude flats)
    2. Post-2000 builds only (modern insulation)
    3. Gas-heated only (most common fuel)
    4. Owner-occupied areas (tenure control)
    """
    print("\n" + "=" * 70)
    print("SUBSET ANALYSES")
    print("=" * 70)

    # Define subsets
    subsets = {
        "Full sample": df,
        "Houses only": df[df["property_type"] == "House"],
        "Flats only": df[df["property_type"] == "Flat"],
        "Post-2000 builds": df[df["construction_year"] >= 2000],
        "Pre-1950 builds": df[df["construction_year"] < 1950],
        "Gas-heated only": df[df["main_fuel"] == "gas"],
    }

    # Model specification
    model_vars = ["log_floor_area", "building_age", "compactness", "shared_wall_ratio", "pop_density"]
    available = [v for v in model_vars if v in df.columns]
    formula = "log_energy_per_capita ~ " + " + ".join(available)

    results = []
    key_vars = ["building_age", "compactness", "shared_wall_ratio", "pop_density"]

    print(f"\nFormula: {formula}")
    print("\nCoefficient comparison across subsets:")
    print("-" * 80)
    header = f"{'Subset':<20}"
    for var in key_vars:
        header += f" {var[:12]:>12}"
    header += f" {'n':>8} {'R²':>6}"
    print(header)
    print("-" * 80)

    for subset_name, subset_df in subsets.items():
        model_df = subset_df[["log_energy_per_capita"] + available].dropna()

        if len(model_df) < 50:
            print(f"{subset_name:<20} Insufficient observations ({len(model_df)})")
            continue

        try:
            model = smf.ols(formula, data=model_df).fit()

            row = f"{subset_name:<20}"
            for var in key_vars:
                if var in model.params:
                    coef = model.params[var]
                    p = model.pvalues[var]
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    row += f" {coef:>+10.4f}{sig:<2}"

                    results.append({
                        "subset": subset_name,
                        "variable": var,
                        "coefficient": coef,
                        "p_value": p,
                        "n": len(model_df),
                        "r2": model.rsquared,
                    })
                else:
                    row += f" {'N/A':>12}"

            row += f" {len(model_df):>8,} {model.rsquared:>6.3f}"
            print(row)

        except Exception as e:
            print(f"{subset_name:<20} Error: {e}")

    print("-" * 80)

    return pd.DataFrame(results)


def coefficient_stability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assess coefficient stability as controls are added.

    Large changes suggest confounding sensitivity.
    """
    print("\n" + "=" * 70)
    print("COEFFICIENT STABILITY ANALYSIS")
    print("=" * 70)
    print("\nHow do coefficients change as controls are added?")
    print("Large changes suggest sensitivity to confounding.\n")

    # Prepare data
    model_vars = [
        "log_energy_per_capita", "log_floor_area", "building_age",
        "compactness", "shared_wall_ratio", "cc_harmonic_800", "pop_density",
    ]
    available = [v for v in model_vars if v in df.columns]
    model_df = df[available].dropna()

    print(f"Complete cases: {len(model_df):,}")

    # Progressive models
    models = [
        ("M1: Size only", ["log_floor_area"]),
        ("M2: + Age", ["log_floor_area", "building_age"]),
        ("M3: + Morphology", ["log_floor_area", "building_age", "compactness", "shared_wall_ratio"]),
        ("M4: + Network", ["log_floor_area", "building_age", "compactness", "shared_wall_ratio", "cc_harmonic_800"]),
        ("M5: + Density", ["log_floor_area", "building_age", "compactness", "shared_wall_ratio", "cc_harmonic_800", "pop_density"]),
    ]

    results = []
    track_vars = ["building_age", "compactness", "shared_wall_ratio", "cc_harmonic_800", "pop_density"]

    print("\nCoefficient trajectories:")
    print("-" * 90)
    header = f"{'Model':<20}"
    for var in track_vars:
        header += f" {var[:12]:>12}"
    header += f" {'R²':>8}"
    print(header)
    print("-" * 90)

    for model_name, predictors in models:
        avail_predictors = [p for p in predictors if p in model_df.columns]
        if not avail_predictors:
            continue

        formula = "log_energy_per_capita ~ " + " + ".join(avail_predictors)

        try:
            model = smf.ols(formula, data=model_df).fit()

            row = f"{model_name:<20}"
            for var in track_vars:
                if var in model.params:
                    coef = model.params[var]
                    row += f" {coef:>+12.4f}"
                    results.append({
                        "model": model_name,
                        "variable": var,
                        "coefficient": coef,
                        "r2": model.rsquared,
                    })
                else:
                    row += f" {'--':>12}"

            row += f" {model.rsquared:>8.3f}"
            print(row)

        except Exception as e:
            print(f"{model_name:<20} Error: {e}")

    print("-" * 90)

    # Calculate coefficient changes
    print("\n### Coefficient Changes (from first appearance to final model)")
    results_df = pd.DataFrame(results)
    for var in track_vars:
        var_data = results_df[results_df["variable"] == var]
        if len(var_data) >= 2:
            first_coef = var_data.iloc[0]["coefficient"]
            last_coef = var_data.iloc[-1]["coefficient"]
            change = last_coef - first_coef
            pct_change = (change / abs(first_coef)) * 100 if first_coef != 0 else np.nan

            print(f"  {var:25} {first_coef:+.4f} → {last_coef:+.4f} (Δ = {change:+.4f}, {pct_change:+.1f}%)")

            if abs(pct_change) > 20:
                print(f"    ⚠ Substantial change (>{20}%) - coefficient is sensitive to controls")

    return results_df


def compute_e_value(point_estimate: float, ci_lower: float, ci_upper: float) -> dict:
    """
    Compute E-value for unmeasured confounding sensitivity.

    The E-value is the minimum strength of association that an unmeasured
    confounder would need with both the treatment and outcome to fully
    explain away the observed association.

    Parameters
    ----------
    point_estimate : float
        Risk ratio (or approximation from standardized coefficient)
    ci_lower : float
        Lower bound of 95% CI
    ci_upper : float
        Upper bound of 95% CI

    Returns
    -------
    dict
        E-value for point estimate and confidence interval bound
    """
    # E-value formula for risk ratios
    def e_value(rr):
        if rr >= 1:
            return rr + np.sqrt(rr * (rr - 1))
        else:
            # For protective effects, use 1/RR
            rr_inv = 1 / rr
            return rr_inv + np.sqrt(rr_inv * (rr_inv - 1))

    e_point = e_value(point_estimate)

    # E-value for CI bound closest to null
    if point_estimate >= 1:
        e_ci = e_value(ci_lower) if ci_lower > 1 else 1.0
    else:
        e_ci = e_value(ci_upper) if ci_upper < 1 else 1.0

    return {
        "e_value_point": e_point,
        "e_value_ci": e_ci,
    }


def e_value_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute E-values for key associations.

    E-values quantify how strong unmeasured confounding would need to be
    to explain away the observed association.
    """
    print("\n" + "=" * 70)
    print("E-VALUE ANALYSIS (Unmeasured Confounding Sensitivity)")
    print("=" * 70)
    print("""
The E-value represents the minimum strength of association that an
unmeasured confounder would need with BOTH the exposure AND the outcome
to fully explain away the observed association.

Interpretation:
- E-value = 1.0: Association could be explained by any confounder
- E-value = 2.0: Confounder would need RR ≥ 2 with both exposure and outcome
- E-value ≥ 3.0: Robust to moderate unmeasured confounding
""")

    # Fit model to get standardized coefficients
    model_vars = [
        "log_energy_per_capita", "log_floor_area", "building_age",
        "compactness", "shared_wall_ratio", "cc_harmonic_800", "pop_density",
    ]
    available = [v for v in model_vars if v in df.columns]
    model_df = df[available].dropna()

    formula = "log_energy_per_capita ~ " + " + ".join([v for v in available if v != "log_energy_per_capita"])
    model = smf.ols(formula, data=model_df).fit()

    # Convert to approximate risk ratios
    # For log-linear models, exp(β) ≈ RR for small effects
    results = []
    key_vars = ["building_age", "compactness", "shared_wall_ratio", "pop_density"]

    print("\nE-values for key variables:")
    print("-" * 70)
    print(f"{'Variable':<25} {'β':>8} {'exp(β)':>8} {'E-value':>10} {'E-value CI':>10}")
    print("-" * 70)

    for var in key_vars:
        if var not in model.params:
            continue

        coef = model.params[var]
        se = model.bse[var]

        # Approximate RR
        rr = np.exp(coef)
        rr_lower = np.exp(coef - 1.96 * se)
        rr_upper = np.exp(coef + 1.96 * se)

        # Compute E-value
        e_vals = compute_e_value(rr, rr_lower, rr_upper)

        print(f"{var:<25} {coef:>+8.4f} {rr:>8.3f} {e_vals['e_value_point']:>10.2f} {e_vals['e_value_ci']:>10.2f}")

        results.append({
            "variable": var,
            "coefficient": coef,
            "exp_coef": rr,
            "e_value_point": e_vals["e_value_point"],
            "e_value_ci": e_vals["e_value_ci"],
        })

    print("-" * 70)

    # Interpretation
    print("\nInterpretation:")
    for r in results:
        var = r["variable"]
        e_val = r["e_value_point"]
        e_ci = r["e_value_ci"]

        if e_ci < 1.5:
            print(f"  {var}: SENSITIVE to unmeasured confounding (E-value CI < 1.5)")
        elif e_ci < 2.0:
            print(f"  {var}: Moderately robust (E-value CI = {e_ci:.2f})")
        else:
            print(f"  {var}: Robust to moderate confounding (E-value CI = {e_ci:.2f})")

    return pd.DataFrame(results)


def sensitivity_summary() -> None:
    """Print summary of sensitivity analysis."""
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 70)
    print("""
KEY TAKEAWAYS:

1. LOG TRANSFORMATION
   - Check for zeros/negatives before using log
   - Compare residual distributions
   - Log often reduces skewness in energy data

2. SPECIFICATION ROBUSTNESS
   - Compare results across different DVs
   - Similar patterns suggest robust findings
   - Divergent patterns require explanation

3. SUBSET STABILITY
   - Compare coefficients across subgroups
   - Stable coefficients suggest generalizability
   - Changing coefficients may indicate effect heterogeneity

4. COEFFICIENT STABILITY
   - Track how coefficients change with controls
   - Large changes (>20%) suggest confounding
   - Stable coefficients more credible

5. E-VALUES
   - Quantify robustness to unmeasured confounding
   - E-value ≥ 2 suggests moderate robustness
   - E-value near 1 suggests high sensitivity

LIMITATIONS:
- E-values assume single unmeasured confounder
- Subset analyses reduce power
- All sensitivity tests assume linear effects
""")


def main() -> None:
    """Run sensitivity analysis."""
    print("=" * 70)
    print("SENSITIVITY ANALYSIS")
    print("=" * 70)
    print("\nTesting robustness of energy-morphology associations")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_and_prepare_data()

    # Log transformation validation
    log_results = test_log_transformation(df)

    # Specification comparison
    spec_results = compare_specifications(df)
    if not spec_results.empty:
        spec_results.to_csv(OUTPUT_DIR / "specification_comparison.csv", index=False)

    # Subset analyses
    subset_results = subset_analyses(df)
    if not subset_results.empty:
        subset_results.to_csv(OUTPUT_DIR / "subset_analyses.csv", index=False)

    # Coefficient stability
    stability_results = coefficient_stability(df)
    if not stability_results.empty:
        stability_results.to_csv(OUTPUT_DIR / "coefficient_stability.csv", index=False)

    # E-value analysis
    e_value_results = e_value_analysis(df)
    if not e_value_results.empty:
        e_value_results.to_csv(OUTPUT_DIR / "e_values.csv", index=False)

    # Summary
    sensitivity_summary()

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
