"""
Selection bias analysis for EPC dataset.

EPC data only covers properties that have been sold or rented, creating
non-random missingness. This script:
1. Calculates EPC coverage rates by building characteristics
2. Implements inverse probability weighting (IPW)
3. Compares weighted vs unweighted correlation estimates

Usage:
    uv run python stats/00_selection_bias_analysis.py
"""


import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Configuration
from urban_energy.paths import TEMP_DIR

UPRN_PATH = TEMP_DIR / "processing" / "test" / "uprn_integrated.gpkg"
OUTPUT_DIR = TEMP_DIR / "stats"


def load_data() -> gpd.GeoDataFrame:
    """Load UPRN integrated dataset."""
    print("Loading data...")
    gdf = gpd.read_file(UPRN_PATH)
    print(f"  Total UPRNs: {len(gdf):,}")
    return gdf


def calculate_coverage_rates(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Calculate EPC coverage rates by building characteristics.

    Returns a DataFrame with coverage statistics for reporting.
    """
    print("\n" + "=" * 70)
    print("EPC COVERAGE ANALYSIS")
    print("=" * 70)

    # Create EPC indicator
    gdf["has_epc"] = gdf["CURRENT_ENERGY_EFFICIENCY"].notna()
    overall_coverage = gdf["has_epc"].mean()
    print(f"\nOverall EPC coverage: {overall_coverage:.1%}")

    results = []

    # Coverage by property type (from morphology/building data)
    if "PROPERTY_TYPE" in gdf.columns:
        print("\n### Coverage by Property Type")
        coverage = gdf.groupby("PROPERTY_TYPE")["has_epc"].agg(["sum", "count", "mean"])
        coverage.columns = ["with_epc", "total", "coverage_rate"]
        coverage = coverage.sort_values("coverage_rate", ascending=False)
        print(coverage.to_string())
        for prop_type, row in coverage.iterrows():
            results.append({
                "category": "Property Type",
                "value": prop_type,
                "with_epc": row["with_epc"],
                "total": row["total"],
                "coverage_rate": row["coverage_rate"],
            })

    # Coverage by built form
    if "BUILT_FORM" in gdf.columns:
        print("\n### Coverage by Built Form")
        coverage = gdf.groupby("BUILT_FORM")["has_epc"].agg(["sum", "count", "mean"])
        coverage.columns = ["with_epc", "total", "coverage_rate"]
        coverage = coverage.sort_values("coverage_rate", ascending=False)
        print(coverage.to_string())
        for built_form, row in coverage.iterrows():
            results.append({
                "category": "Built Form",
                "value": built_form,
                "with_epc": row["with_epc"],
                "total": row["total"],
                "coverage_rate": row["coverage_rate"],
            })

    # Coverage by construction age band
    if "CONSTRUCTION_AGE_BAND" in gdf.columns:
        print("\n### Coverage by Construction Age Band")
        # Filter to non-null age bands
        age_data = gdf[gdf["CONSTRUCTION_AGE_BAND"].notna() & (gdf["CONSTRUCTION_AGE_BAND"] != "NO DATA!")]
        coverage = age_data.groupby("CONSTRUCTION_AGE_BAND")["has_epc"].agg(["sum", "count", "mean"])
        coverage.columns = ["with_epc", "total", "coverage_rate"]
        coverage = coverage.sort_values("coverage_rate", ascending=False)
        print(coverage.to_string())
        for age_band, row in coverage.iterrows():
            results.append({
                "category": "Construction Age",
                "value": age_band,
                "with_epc": row["with_epc"],
                "total": row["total"],
                "coverage_rate": row["coverage_rate"],
            })

    # Coverage by LSOA deprivation (Census)
    deprivation_col = "ts011_Household deprivation: Total: All households; measures: Value"
    if deprivation_col in gdf.columns:
        print("\n### Coverage by LSOA Deprivation Level")
        total_hh = gdf[deprivation_col]
        deprived_cols = [
            "ts011_Household deprivation: Household is deprived in one dimension; measures: Value",
            "ts011_Household deprivation: Household is deprived in two dimensions; measures: Value",
            "ts011_Household deprivation: Household is deprived in three dimensions; measures: Value",
            "ts011_Household deprivation: Household is deprived in four dimensions; measures: Value",
        ]
        total_deprived = sum(gdf[col] for col in deprived_cols if col in gdf.columns)
        gdf["pct_deprived"] = (total_deprived / total_hh * 100).fillna(0)

        # Create deprivation quintiles
        gdf["deprivation_quintile"] = pd.qcut(
            gdf["pct_deprived"],
            q=5,
            labels=["Q1 (Least)", "Q2", "Q3", "Q4", "Q5 (Most)"],
            duplicates="drop",
        )
        coverage = gdf.groupby("deprivation_quintile", observed=True)["has_epc"].agg(["sum", "count", "mean"])
        coverage.columns = ["with_epc", "total", "coverage_rate"]
        print(coverage.to_string())
        for quintile, row in coverage.iterrows():
            results.append({
                "category": "Deprivation Quintile",
                "value": str(quintile),
                "with_epc": row["with_epc"],
                "total": row["total"],
                "coverage_rate": row["coverage_rate"],
            })

    # Coverage by population density tercile
    pop_density_col = "ts006_Population Density: Persons per square kilometre; measures: Value"
    if pop_density_col in gdf.columns:
        print("\n### Coverage by Population Density")
        gdf["density_tercile"] = pd.qcut(
            gdf[pop_density_col],
            q=3,
            labels=["Low", "Medium", "High"],
            duplicates="drop",
        )
        coverage = gdf.groupby("density_tercile", observed=True)["has_epc"].agg(["sum", "count", "mean"])
        coverage.columns = ["with_epc", "total", "coverage_rate"]
        print(coverage.to_string())
        for tercile, row in coverage.iterrows():
            results.append({
                "category": "Population Density",
                "value": str(tercile),
                "with_epc": row["with_epc"],
                "total": row["total"],
                "coverage_rate": row["coverage_rate"],
            })

    return pd.DataFrame(results)


def compute_propensity_scores(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute propensity scores for having EPC data.

    Uses logistic regression to model P(has_epc | building_characteristics).

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Full dataset with EPC indicator.

    Returns
    -------
    gpd.GeoDataFrame
        Dataset with propensity scores and IPW weights added.
    """
    print("\n" + "=" * 70)
    print("PROPENSITY SCORE ESTIMATION")
    print("=" * 70)

    gdf = gdf.copy()
    gdf["has_epc"] = gdf["CURRENT_ENERGY_EFFICIENCY"].notna().astype(int)

    # Select predictors for propensity model
    # Use variables available for ALL properties (not just those with EPC)
    predictors = []

    # Morphology variables (from building footprints)
    morph_vars = ["footprint_area_m2", "compactness", "convexity", "shared_wall_ratio"]
    for var in morph_vars:
        if var in gdf.columns:
            predictors.append(var)

    # Building height (from LiDAR)
    if "height_mean" in gdf.columns:
        gdf["building_height"] = pd.to_numeric(gdf["height_mean"], errors="coerce")
        predictors.append("building_height")

    # Census variables (available at OA level for all UPRNs)
    census_vars = []
    pop_density_col = "ts006_Population Density: Persons per square kilometre; measures: Value"
    if pop_density_col in gdf.columns:
        gdf["pop_density"] = gdf[pop_density_col]
        census_vars.append("pop_density")

    tenure_col = "ts054_Tenure of household: Total: All households"
    owned_col = "ts054_Tenure of household: Owned"
    if tenure_col in gdf.columns and owned_col in gdf.columns:
        gdf["pct_owner_occupied"] = gdf[owned_col] / gdf[tenure_col] * 100
        census_vars.append("pct_owner_occupied")

    predictors.extend(census_vars)

    print(f"\nPropensity model predictors: {predictors}")

    if len(predictors) < 2:
        print("  Insufficient predictors for propensity model")
        gdf["propensity_score"] = gdf["has_epc"].mean()
        gdf["ipw_weight"] = 1.0
        return gdf

    # Prepare data - complete cases only
    model_data = gdf[["has_epc"] + predictors].dropna()
    print(f"  Complete cases for propensity model: {len(model_data):,}")

    if len(model_data) < 100:
        print("  Insufficient complete cases for propensity model")
        gdf["propensity_score"] = gdf["has_epc"].mean()
        gdf["ipw_weight"] = 1.0
        return gdf

    X = model_data[predictors].values
    y = model_data["has_epc"].values

    # Standardize predictors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit logistic regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    # Report model performance
    accuracy = model.score(X_scaled, y)
    print(f"\n  Propensity model accuracy: {accuracy:.3f}")

    # Report coefficients
    print("\n  Propensity model coefficients:")
    for var, coef in zip(predictors, model.coef_[0]):
        print(f"    {var:30} β = {coef:+.3f}")

    # Predict propensity scores for complete cases
    propensity_scores = model.predict_proba(X_scaled)[:, 1]

    # Add to dataframe
    gdf["propensity_score"] = np.nan
    gdf.loc[model_data.index, "propensity_score"] = propensity_scores

    # For missing propensity scores, use marginal probability
    marginal_prob = gdf["has_epc"].mean()
    gdf["propensity_score"] = gdf["propensity_score"].fillna(marginal_prob)

    # Compute IPW weights (inverse of propensity score for treated units)
    # Clip to avoid extreme weights
    clipped_scores = np.clip(gdf["propensity_score"], 0.05, 0.95)
    gdf["ipw_weight"] = np.where(
        gdf["has_epc"] == 1,
        1 / clipped_scores,
        1 / (1 - clipped_scores),
    )

    # Normalize weights to sum to n
    gdf["ipw_weight"] = gdf["ipw_weight"] / gdf["ipw_weight"].mean()

    print(f"\n  Propensity score range: [{gdf['propensity_score'].min():.3f}, {gdf['propensity_score'].max():.3f}]")
    print(f"  IPW weight range: [{gdf['ipw_weight'].min():.2f}, {gdf['ipw_weight'].max():.2f}]")

    return gdf


def compare_weighted_unweighted(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Compare correlation estimates with and without IPW weighting.

    This shows whether selection bias affects the key findings.
    """
    print("\n" + "=" * 70)
    print("WEIGHTED vs UNWEIGHTED CORRELATION COMPARISON")
    print("=" * 70)

    # Filter to EPC sample
    epc_sample = gdf[gdf["has_epc"] == 1].copy()
    print(f"\nEPC sample size: {len(epc_sample):,}")

    # Prepare dependent variable
    if "ENERGY_CONSUMPTION_CURRENT" not in epc_sample.columns:
        print("  Energy consumption not available")
        return pd.DataFrame()

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
    if all(col in epc_sample.columns for col in size_cols.values()):
        total_people = sum(size * epc_sample[col] for size, col in size_cols.items())
        total_households = epc_sample[
            "ts017_Household size: Total: All household spaces; measures: Value"
        ] - epc_sample["ts017_Household size: 0 people in household; measures: Value"]
        epc_sample["avg_household_size"] = total_people / total_households
        epc_sample["energy_per_capita"] = (
            epc_sample["ENERGY_CONSUMPTION_CURRENT"] / epc_sample["avg_household_size"]
        )
    else:
        # Fallback
        epc_sample["energy_per_capita"] = epc_sample["ENERGY_CONSUMPTION_CURRENT"] / 2.4

    # Filter valid energy values
    valid_energy = (
        np.isfinite(epc_sample["energy_per_capita"]) &
        (epc_sample["energy_per_capita"] > 0)
    )
    epc_sample = epc_sample[valid_energy].copy()

    # Variables to test
    test_vars = [
        ("pop_density", "Population Density"),
        ("cc_harmonic_800", "Network Centrality (800m)"),
        ("compactness", "Building Compactness"),
        ("shared_wall_ratio", "Shared Wall Ratio"),
        ("building_height", "Building Height"),
    ]

    results = []
    target = "energy_per_capita"

    print(f"\nCorrelations with {target}:")
    print("-" * 70)
    print(f"{'Variable':<35} {'Unweighted r':>12} {'Weighted r':>12} {'Diff':>8}")
    print("-" * 70)

    for var, label in test_vars:
        if var not in epc_sample.columns:
            continue

        valid = epc_sample[[target, var, "ipw_weight"]].dropna()
        if len(valid) < 50:
            continue

        # Unweighted correlation
        r_unweighted, p_unweighted = stats.pearsonr(valid[target], valid[var])

        # Weighted correlation (using numpy cov with weights)
        weights = valid["ipw_weight"].values
        x = valid[target].values
        y_var = valid[var].values

        # Weighted means
        mean_x = np.average(x, weights=weights)
        mean_y = np.average(y_var, weights=weights)

        # Weighted covariance and variances
        cov_xy = np.average((x - mean_x) * (y_var - mean_y), weights=weights)
        var_x = np.average((x - mean_x) ** 2, weights=weights)
        var_y = np.average((y_var - mean_y) ** 2, weights=weights)

        r_weighted = cov_xy / np.sqrt(var_x * var_y)
        diff = r_weighted - r_unweighted

        print(f"{label:<35} {r_unweighted:>+12.3f} {r_weighted:>+12.3f} {diff:>+8.3f}")

        results.append({
            "variable": var,
            "label": label,
            "r_unweighted": r_unweighted,
            "r_weighted": r_weighted,
            "difference": diff,
            "n": len(valid),
        })

    print("-" * 70)

    return pd.DataFrame(results)


def selection_bias_summary(coverage_df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
    """Print summary of selection bias analysis."""
    print("\n" + "=" * 70)
    print("SELECTION BIAS SUMMARY")
    print("=" * 70)

    print("""
KEY FINDINGS:

1. EPC COVERAGE PATTERNS
   - EPC data is not randomly missing
   - Coverage varies by property type, age, and area characteristics
   - This creates potential selection bias in energy-morphology relationships

2. PROPENSITY SCORE MODEL
   - Models probability of having EPC based on observable characteristics
   - Allows inverse probability weighting (IPW) to adjust for selection

3. IMPACT ON CORRELATIONS
""")

    if not comparison_df.empty:
        max_diff = comparison_df["difference"].abs().max()
        avg_diff = comparison_df["difference"].abs().mean()
        print(f"   - Maximum correlation change with IPW: {max_diff:.3f}")
        print(f"   - Average absolute change: {avg_diff:.3f}")

        if max_diff < 0.05:
            print("   - Selection bias appears MINIMAL for these variables")
        elif max_diff < 0.10:
            print("   - Selection bias has MODERATE impact on some estimates")
        else:
            print("   - Selection bias has SUBSTANTIAL impact - use IPW weights")

    print("""
4. RECOMMENDATIONS
   - Report both weighted and unweighted estimates
   - Acknowledge selection bias in limitations section
   - Consider sensitivity analyses with different weighting schemes

5. LIMITATIONS OF IPW
   - Only adjusts for OBSERVED confounders
   - Cannot account for unobserved selection factors
   - Extreme weights can increase variance
""")


def main() -> None:
    """Run selection bias analysis."""
    print("=" * 70)
    print("SELECTION BIAS ANALYSIS: EPC Data Coverage")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    gdf = load_data()

    # Calculate coverage rates
    coverage_df = calculate_coverage_rates(gdf)

    # Save coverage results
    if not coverage_df.empty:
        coverage_path = OUTPUT_DIR / "epc_coverage_rates.csv"
        coverage_df.to_csv(coverage_path, index=False)
        print(f"\nCoverage rates saved to: {coverage_path}")

    # Compute propensity scores and IPW weights
    gdf = compute_propensity_scores(gdf)

    # Compare weighted vs unweighted correlations
    comparison_df = compare_weighted_unweighted(gdf)

    # Save comparison results
    if not comparison_df.empty:
        comparison_path = OUTPUT_DIR / "ipw_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"IPW comparison saved to: {comparison_path}")

    # Summary
    selection_bias_summary(coverage_df, comparison_df)


if __name__ == "__main__":
    main()
