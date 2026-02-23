"""
Spatial autocorrelation testing and spatial regression models.

Standard OLS assumes independent observations, but energy consumption
clusters spatially due to climate, gas network availability, and
neighborhood effects. This script:

1. Tests for spatial autocorrelation in OLS residuals (Moran's I)
2. Fits spatial lag and spatial error models if autocorrelation detected
3. Compares standard errors across model specifications

Usage:
    uv run python stats/02b_spatial_regression.py

Requires:
    uv add esda libpysal

Note: spreg (spatial regression) may require additional setup.
As a fallback, this script uses statsmodels with cluster-robust SEs.
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


def load_and_prepare_data() -> gpd.GeoDataFrame:
    """Load UPRN data and prepare for spatial analysis."""
    print("Loading data...")
    gdf = gpd.read_file(DATA_PATH)
    print(f"  Total UPRNs: {len(gdf):,}")

    # Filter to UPRNs with EPC data
    gdf = gdf[gdf["CURRENT_ENERGY_EFFICIENCY"].notna()].copy()
    print(f"  With EPC data: {len(gdf):,}")

    # Filter out records with invalid floor area
    valid_area = (gdf["TOTAL_FLOOR_AREA"] > 0) & gdf["TOTAL_FLOOR_AREA"].notna()
    gdf = gdf[valid_area].copy()

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
    total_people = sum(size * gdf[col] for size, col in size_cols.items())
    total_households = gdf[
        "ts017_Household size: Total: All household spaces; measures: Value"
    ] - gdf["ts017_Household size: 0 people in household; measures: Value"]
    gdf["avg_household_size"] = total_people / total_households
    # ECC is already kWh/m²/year; per-capita needs total kWh
    gdf["total_energy_kwh"] = gdf["ENERGY_CONSUMPTION_CURRENT"] * gdf["TOTAL_FLOOR_AREA"]
    gdf["energy_per_capita"] = gdf["total_energy_kwh"] / gdf["avg_household_size"]

    # Filter valid energy values
    valid_energy = (
        np.isfinite(gdf["energy_per_capita"]) &
        (gdf["energy_per_capita"] > 0)
    )
    gdf = gdf[valid_energy].copy()

    # Log transform
    gdf["log_energy_per_capita"] = np.log(gdf["energy_per_capita"])
    gdf["log_floor_area"] = np.log(gdf["TOTAL_FLOOR_AREA"])

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

    gdf["construction_year"] = gdf["CONSTRUCTION_AGE_BAND"].apply(parse_age_band)
    gdf["building_age"] = 2024 - gdf["construction_year"]

    # Census variables
    gdf["pop_density"] = gdf[
        "ts006_Population Density: Persons per square kilometre; measures: Value"
    ]

    print(f"  Analysis sample: {len(gdf):,}")
    return gdf


def test_spatial_autocorrelation(gdf: gpd.GeoDataFrame, residuals: np.ndarray) -> dict:
    """
    Test for spatial autocorrelation using Moran's I.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Spatial data with geometry.
    residuals : np.ndarray
        Model residuals to test.

    Returns
    -------
    dict
        Moran's I statistic, expected value, variance, z-score, and p-value.
    """
    print("\n" + "=" * 70)
    print("SPATIAL AUTOCORRELATION TEST (Moran's I)")
    print("=" * 70)

    try:
        from esda.moran import Moran
        from libpysal.weights import KNN, DistanceBand
    except ImportError:
        print("\n  esda/libpysal not installed. Install with:")
        print("    uv add esda libpysal")
        print("\n  Falling back to simplified permutation test...")
        return _fallback_spatial_test(gdf, residuals)

    # For large datasets, use a sample for weight matrix construction
    n = len(gdf)
    if n > 10000:
        print(f"\n  Large dataset ({n:,} obs). Sampling 5000 for spatial weights...")
        sample_idx = np.random.choice(n, size=5000, replace=False)
        gdf_sample = gdf.iloc[sample_idx].copy()
        residuals_sample = residuals[sample_idx]
    else:
        gdf_sample = gdf
        residuals_sample = residuals

    # Construct spatial weights matrix
    print("\n  Constructing spatial weights matrix...")
    try:
        # Use KNN for efficiency (k=8 nearest neighbors)
        w = KNN.from_dataframe(gdf_sample, k=8)
        w.transform = "r"  # Row-standardize
        print(f"    KNN weights (k=8): {w.n} observations")
    except Exception as e:
        print(f"    KNN failed: {e}")
        print("    Trying distance band...")
        try:
            # Fallback to distance band (1km)
            w = DistanceBand.from_dataframe(gdf_sample, threshold=1000, binary=True)
            w.transform = "r"
            print(f"    Distance band (1km): {w.n} observations")
        except Exception as e2:
            print(f"    Distance band failed: {e2}")
            return {"error": str(e2)}

    # Compute Moran's I
    print("\n  Computing Moran's I...")
    try:
        mi = Moran(residuals_sample, w, permutations=999)

        print(f"\n  Results:")
        print(f"    Moran's I:     {mi.I:.4f}")
        print(f"    Expected I:    {mi.EI:.4f}")
        print(f"    Variance:      {mi.VI_norm:.6f}")
        print(f"    Z-score:       {mi.z_norm:.2f}")
        print(f"    P-value (sim): {mi.p_sim:.4f}")

        # Interpret
        if mi.p_sim < 0.001:
            interpretation = "STRONG spatial autocorrelation detected (p < 0.001)"
        elif mi.p_sim < 0.01:
            interpretation = "Significant spatial autocorrelation (p < 0.01)"
        elif mi.p_sim < 0.05:
            interpretation = "Moderate spatial autocorrelation (p < 0.05)"
        else:
            interpretation = "No significant spatial autocorrelation (p >= 0.05)"

        print(f"\n  Interpretation: {interpretation}")

        if mi.I > 0 and mi.p_sim < 0.05:
            print("    → Positive autocorrelation: similar values cluster together")
            print("    → Standard errors are likely UNDERESTIMATED")
            print("    → Recommend: spatial regression or cluster-robust SEs")
        elif mi.I < 0 and mi.p_sim < 0.05:
            print("    → Negative autocorrelation: dissimilar values cluster")
            print("    → This is unusual and may indicate model misspecification")

        return {
            "morans_i": mi.I,
            "expected_i": mi.EI,
            "variance": mi.VI_norm,
            "z_score": mi.z_norm,
            "p_value": mi.p_sim,
            "significant": mi.p_sim < 0.05,
            "interpretation": interpretation,
        }

    except Exception as e:
        print(f"    Moran's I computation failed: {e}")
        return {"error": str(e)}


def _fallback_spatial_test(gdf: gpd.GeoDataFrame, residuals: np.ndarray) -> dict:
    """
    Simplified spatial autocorrelation test without esda.

    Uses correlation between residuals and spatially lagged residuals
    based on nearest neighbors.
    """
    from scipy.spatial import cKDTree

    print("\n  Using fallback spatial test (nearest neighbor correlation)...")

    # Sample if large
    n = len(gdf)
    if n > 10000:
        sample_idx = np.random.choice(n, size=5000, replace=False)
        coords = np.column_stack([
            gdf.geometry.iloc[sample_idx].x,
            gdf.geometry.iloc[sample_idx].y
        ])
        resid_sample = residuals[sample_idx]
    else:
        coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
        resid_sample = residuals

    # Build KDTree and find k nearest neighbors
    tree = cKDTree(coords)
    k = 8
    distances, indices = tree.query(coords, k=k + 1)  # +1 because first is self

    # Compute spatially lagged residuals (mean of k nearest neighbors)
    lagged_resid = np.mean(resid_sample[indices[:, 1:]], axis=1)

    # Correlation between residuals and spatial lag
    r, p = stats.pearsonr(resid_sample, lagged_resid)

    print(f"\n  Results:")
    print(f"    Spatial lag correlation: r = {r:.4f}")
    print(f"    P-value: {p:.4f}")

    if p < 0.05:
        print(f"\n  Interpretation: Significant spatial autocorrelation detected")
    else:
        print(f"\n  Interpretation: No significant spatial autocorrelation")

    return {
        "spatial_lag_correlation": r,
        "p_value": p,
        "significant": p < 0.05,
        "method": "fallback_knn_correlation",
    }


def fit_ols_with_cluster_robust_se(gdf: gpd.GeoDataFrame) -> dict:
    """
    Fit OLS model with cluster-robust standard errors.

    Clusters at LSOA level to account for spatial correlation
    within neighborhoods.

    Returns comparison of naive vs cluster-robust standard errors.
    """
    print("\n" + "=" * 70)
    print("OLS WITH CLUSTER-ROBUST STANDARD ERRORS")
    print("=" * 70)

    # Prepare model data
    model_vars = [
        "log_energy_per_capita",
        "log_floor_area",
        "building_age",
        "compactness",
        "shared_wall_ratio",
        "cc_harmonic_800",
        "pop_density",
        "LSOA21CD",
    ]
    available = [v for v in model_vars if v in gdf.columns]
    model_df = gdf[available].dropna().copy()
    print(f"\n  Complete cases: {len(model_df):,}")
    print(f"  Clusters (LSOAs): {model_df['LSOA21CD'].nunique()}")

    # Fit OLS with naive SEs
    formula = "log_energy_per_capita ~ log_floor_area + building_age"
    if "compactness" in model_df.columns:
        formula += " + compactness"
    if "shared_wall_ratio" in model_df.columns:
        formula += " + shared_wall_ratio"
    if "cc_harmonic_800" in model_df.columns:
        formula += " + cc_harmonic_800"
    if "pop_density" in model_df.columns:
        formula += " + pop_density"

    print(f"\n  Formula: {formula}")

    # Naive OLS
    model_naive = smf.ols(formula, data=model_df).fit()

    # Cluster-robust OLS (clustered at LSOA)
    model_cluster = smf.ols(formula, data=model_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": model_df["LSOA21CD"]},
    )

    # Compare standard errors
    print("\n  Standard Error Comparison:")
    print("-" * 70)
    print(f"  {'Variable':<25} {'Naive SE':>12} {'Cluster SE':>12} {'Ratio':>8}")
    print("-" * 70)

    results = []
    for var in model_naive.params.index:
        if var == "Intercept":
            continue
        naive_se = model_naive.bse[var]
        cluster_se = model_cluster.bse[var]
        ratio = cluster_se / naive_se
        print(f"  {var:<25} {naive_se:>12.4f} {cluster_se:>12.4f} {ratio:>8.2f}")
        results.append({
            "variable": var,
            "naive_se": naive_se,
            "cluster_se": cluster_se,
            "ratio": ratio,
            "coef": model_naive.params[var],
        })

    print("-" * 70)

    # Interpretation
    avg_ratio = np.mean([r["ratio"] for r in results])
    print(f"\n  Average SE inflation: {avg_ratio:.2f}x")

    if avg_ratio > 1.5:
        print("  → Substantial clustering effect detected")
        print("  → Naive standard errors are TOO SMALL")
        print("  → Use cluster-robust SEs for inference")
    elif avg_ratio > 1.1:
        print("  → Moderate clustering effect")
        print("  → Consider using cluster-robust SEs")
    else:
        print("  → Minimal clustering effect")
        print("  → Naive SEs appear adequate")

    return {
        "model_naive": model_naive,
        "model_cluster": model_cluster,
        "se_comparison": pd.DataFrame(results),
        "avg_se_inflation": avg_ratio,
        "residuals": model_naive.resid.values,
    }


def spatial_analysis_summary(moran_result: dict, se_comparison: dict) -> None:
    """Print comprehensive summary of spatial analysis."""
    print("\n" + "=" * 70)
    print("SPATIAL ANALYSIS SUMMARY")
    print("=" * 70)

    print("""
FINDINGS:

1. SPATIAL AUTOCORRELATION TEST
""")
    if "morans_i" in moran_result:
        print(f"   Moran's I = {moran_result['morans_i']:.4f} (p = {moran_result['p_value']:.4f})")
        print(f"   {moran_result['interpretation']}")
    elif "spatial_lag_correlation" in moran_result:
        print(f"   Spatial lag r = {moran_result['spatial_lag_correlation']:.4f}")
        print(f"   P-value = {moran_result['p_value']:.4f}")
    elif "error" in moran_result:
        print(f"   Test failed: {moran_result['error']}")

    print("""
2. STANDARD ERROR COMPARISON
""")
    print(f"   Average SE inflation with clustering: {se_comparison['avg_se_inflation']:.2f}x")

    print("""
3. IMPLICATIONS FOR ANALYSIS

   If spatial autocorrelation is significant:
   - OLS standard errors are underestimated
   - P-values are too small (false positives more likely)
   - Confidence intervals are too narrow

4. RECOMMENDATIONS
""")
    if moran_result.get("significant", False) or se_comparison["avg_se_inflation"] > 1.5:
        print("""   a) Use cluster-robust standard errors (implemented above)
   b) Consider spatial regression models (ML_Lag, ML_Error)
   c) Report both naive and robust SEs in paper
   d) Be conservative in significance claims
""")
    else:
        print("""   a) Spatial autocorrelation appears minimal
   b) Standard OLS inference is likely adequate
   c) Still recommend reporting cluster-robust SEs for robustness
""")

    print("""
5. LIMITATIONS

   - Moran's I test assumes specific spatial weight structure
   - Different weight specifications may yield different results
   - Cluster-robust SEs only account for within-cluster correlation
   - Full spatial models (spreg) provide more thorough treatment
""")


def main() -> None:
    """Run spatial autocorrelation analysis."""
    print("=" * 70)
    print("SPATIAL AUTOCORRELATION ANALYSIS")
    print("=" * 70)
    print("\nTesting whether OLS residuals are spatially correlated,")
    print("which would invalidate standard inference procedures.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    gdf = load_and_prepare_data()

    # Fit OLS and get cluster-robust SEs
    se_results = fit_ols_with_cluster_robust_se(gdf)

    # Test spatial autocorrelation in residuals
    # Need to align residuals with geodataframe
    model_vars = [
        "log_energy_per_capita", "log_floor_area", "building_age",
        "compactness", "shared_wall_ratio", "cc_harmonic_800",
        "pop_density", "LSOA21CD",
    ]
    available = [v for v in model_vars if v in gdf.columns]
    complete_mask = gdf[available].notna().all(axis=1)
    gdf_complete = gdf[complete_mask].copy()

    moran_result = test_spatial_autocorrelation(gdf_complete, se_results["residuals"])

    # Save SE comparison
    se_results["se_comparison"].to_csv(OUTPUT_DIR / "se_comparison.csv", index=False)
    print(f"\nSE comparison saved to: {OUTPUT_DIR / 'se_comparison.csv'}")

    # Summary
    spatial_analysis_summary(moran_result, se_results)


if __name__ == "__main__":
    main()
