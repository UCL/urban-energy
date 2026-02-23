"""
Spatial cross-validation for model assessment.

Standard cross-validation can overestimate performance when data is
spatially correlated. This script implements:

1. Spatial k-fold CV (by LSOA)
2. Leave-one-out by local authority (geographic generalization)
3. Out-of-sample R² and RMSE metrics

Usage:
    uv run python stats/02e_cross_validation.py
"""


import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold

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
    # ECC is already kWh/m²/year; per-capita needs total kWh
    df["total_energy_kwh"] = df["ENERGY_CONSUMPTION_CURRENT"] * df["TOTAL_FLOOR_AREA"]
    df["energy_per_capita"] = df["total_energy_kwh"] / df["avg_household_size"]

    # Filter valid energy values
    valid_energy = (
        np.isfinite(df["energy_per_capita"]) &
        (df["energy_per_capita"] > 0)
    )
    df = df[valid_energy].copy()

    # Log transform
    df["log_energy_per_capita"] = np.log(df["energy_per_capita"])
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

    print(f"  Analysis sample: {len(df):,}")
    return df


def random_cv(df: pd.DataFrame, n_folds: int = 5) -> dict:
    """
    Standard random k-fold cross-validation (baseline).

    This ignores spatial structure and likely overestimates performance.
    """
    print("\n" + "=" * 70)
    print(f"RANDOM {n_folds}-FOLD CROSS-VALIDATION (Baseline)")
    print("=" * 70)
    print("\nNote: Random CV ignores spatial structure and may overestimate performance.")

    # Prepare model data
    model_vars = [
        "log_energy_per_capita", "log_floor_area", "building_age",
        "compactness", "shared_wall_ratio", "pop_density",
    ]
    available = [v for v in model_vars if v in df.columns]
    model_df = df[available].dropna().reset_index(drop=True)

    print(f"Complete cases: {len(model_df):,}")

    if len(model_df) < 100:
        print("Insufficient data for cross-validation")
        return {}

    formula = "log_energy_per_capita ~ " + " + ".join([v for v in available if v != "log_energy_per_capita"])

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(model_df), 1):
        train_df = model_df.iloc[train_idx]
        test_df = model_df.iloc[test_idx]

        # Fit model on training data
        model = smf.ols(formula, data=train_df).fit()

        # Predict on test data
        y_true = test_df["log_energy_per_capita"].values
        y_pred = model.predict(test_df)

        # Metrics
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))

        fold_results.append({
            "fold": fold,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
        })

        print(f"  Fold {fold}: R² = {r2:.3f}, RMSE = {rmse:.3f}, MAE = {mae:.3f}")

    # Summary
    results_df = pd.DataFrame(fold_results)
    print(f"\n  Mean R²:   {results_df['r2'].mean():.3f} ± {results_df['r2'].std():.3f}")
    print(f"  Mean RMSE: {results_df['rmse'].mean():.3f} ± {results_df['rmse'].std():.3f}")
    print(f"  Mean MAE:  {results_df['mae'].mean():.3f} ± {results_df['mae'].std():.3f}")

    return {
        "method": "random_cv",
        "n_folds": n_folds,
        "mean_r2": results_df["r2"].mean(),
        "std_r2": results_df["r2"].std(),
        "mean_rmse": results_df["rmse"].mean(),
        "std_rmse": results_df["rmse"].std(),
        "fold_results": results_df,
    }


def spatial_cv_by_lsoa(df: pd.DataFrame, n_folds: int = 5) -> dict:
    """
    Spatial cross-validation by LSOA.

    Splits data by LSOA (neighborhood) rather than randomly.
    This provides a more realistic estimate of geographic generalization.
    """
    print("\n" + "=" * 70)
    print(f"SPATIAL {n_folds}-FOLD CROSS-VALIDATION (By LSOA)")
    print("=" * 70)
    print("\nSplits data by neighborhood (LSOA) - tests geographic generalization.")

    # Prepare model data
    model_vars = [
        "log_energy_per_capita", "log_floor_area", "building_age",
        "compactness", "shared_wall_ratio", "pop_density", "LSOA21CD",
    ]
    available = [v for v in model_vars if v in df.columns]
    model_df = df[available].dropna().reset_index(drop=True)

    print(f"Complete cases: {len(model_df):,}")
    print(f"Unique LSOAs: {model_df['LSOA21CD'].nunique()}")

    if model_df["LSOA21CD"].nunique() < n_folds:
        print(f"Insufficient LSOAs for {n_folds}-fold CV")
        return {}

    formula = "log_energy_per_capita ~ " + " + ".join([v for v in available if v not in ["log_energy_per_capita", "LSOA21CD"]])

    # Get unique LSOAs and assign to folds
    lsoas = model_df["LSOA21CD"].unique()
    np.random.seed(42)
    np.random.shuffle(lsoas)

    # Split LSOAs into folds
    lsoa_folds = np.array_split(lsoas, n_folds)

    fold_results = []
    for fold, test_lsoas in enumerate(lsoa_folds, 1):
        test_mask = model_df["LSOA21CD"].isin(test_lsoas)
        train_df = model_df[~test_mask]
        test_df = model_df[test_mask]

        if len(train_df) < 50 or len(test_df) < 10:
            print(f"  Fold {fold}: Insufficient data (train={len(train_df)}, test={len(test_df)})")
            continue

        # Fit model on training data
        model = smf.ols(formula, data=train_df).fit()

        # Predict on test data
        y_true = test_df["log_energy_per_capita"].values
        y_pred = model.predict(test_df)

        # Metrics
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))

        fold_results.append({
            "fold": fold,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_lsoas_test": len(test_lsoas),
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
        })

        print(f"  Fold {fold}: R² = {r2:.3f}, RMSE = {rmse:.3f}, n_test = {len(test_df)}, LSOAs = {len(test_lsoas)}")

    if not fold_results:
        return {}

    # Summary
    results_df = pd.DataFrame(fold_results)
    print(f"\n  Mean R²:   {results_df['r2'].mean():.3f} ± {results_df['r2'].std():.3f}")
    print(f"  Mean RMSE: {results_df['rmse'].mean():.3f} ± {results_df['rmse'].std():.3f}")
    print(f"  Mean MAE:  {results_df['mae'].mean():.3f} ± {results_df['mae'].std():.3f}")

    return {
        "method": "spatial_cv_lsoa",
        "n_folds": n_folds,
        "mean_r2": results_df["r2"].mean(),
        "std_r2": results_df["r2"].std(),
        "mean_rmse": results_df["rmse"].mean(),
        "std_rmse": results_df["rmse"].std(),
        "fold_results": results_df,
    }


def compare_cv_methods(random_results: dict, spatial_results: dict) -> None:
    """Compare random and spatial CV results."""
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION COMPARISON")
    print("=" * 70)

    if not random_results or not spatial_results:
        print("Insufficient results for comparison")
        return

    print("\n  Method comparison:")
    print("-" * 50)
    print(f"  {'Metric':<15} {'Random CV':>15} {'Spatial CV':>15}")
    print("-" * 50)

    r2_random = random_results.get("mean_r2", np.nan)
    r2_spatial = spatial_results.get("mean_r2", np.nan)
    print(f"  {'Mean R²':<15} {r2_random:>15.3f} {r2_spatial:>15.3f}")

    rmse_random = random_results.get("mean_rmse", np.nan)
    rmse_spatial = spatial_results.get("mean_rmse", np.nan)
    print(f"  {'Mean RMSE':<15} {rmse_random:>15.3f} {rmse_spatial:>15.3f}")

    print("-" * 50)

    # Interpretation
    r2_diff = r2_random - r2_spatial
    print(f"\n  R² difference (random - spatial): {r2_diff:+.3f}")

    if r2_diff > 0.05:
        print("  → Random CV OVERESTIMATES performance by >{:.0%}".format(r2_diff / r2_spatial if r2_spatial > 0 else 0))
        print("  → Spatial CV provides more realistic estimate of geographic generalization")
    elif r2_diff > 0.02:
        print("  → Moderate overestimation by random CV")
        print("  → Spatial CV recommended for geographic claims")
    else:
        print("  → Minimal difference between CV methods")
        print("  → Model may generalize well across neighborhoods")


def model_comparison_cv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare different model specifications using spatial CV.

    Tests which predictors improve out-of-sample performance.
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (Spatial CV)")
    print("=" * 70)
    print("\nComparing predictive performance of different model specifications.")

    # Prepare data
    model_vars = [
        "log_energy_per_capita", "log_floor_area", "building_age",
        "compactness", "shared_wall_ratio", "cc_harmonic_800", "pop_density", "LSOA21CD",
    ]
    available = [v for v in model_vars if v in df.columns]
    model_df = df[available].dropna().reset_index(drop=True)

    # Define model specifications
    models = {
        "M1: Size only": ["log_floor_area"],
        "M2: + Age": ["log_floor_area", "building_age"],
        "M3: + Morphology": ["log_floor_area", "building_age", "compactness", "shared_wall_ratio"],
        "M4: + Network": ["log_floor_area", "building_age", "compactness", "shared_wall_ratio", "cc_harmonic_800"],
        "M5: + Density": ["log_floor_area", "building_age", "compactness", "shared_wall_ratio", "cc_harmonic_800", "pop_density"],
    }

    # Get unique LSOAs for spatial CV
    lsoas = model_df["LSOA21CD"].unique()
    np.random.seed(42)
    np.random.shuffle(lsoas)
    n_folds = 5
    lsoa_folds = np.array_split(lsoas, n_folds)

    results = []

    for model_name, predictors in models.items():
        avail_preds = [p for p in predictors if p in model_df.columns]
        if not avail_preds:
            continue

        formula = "log_energy_per_capita ~ " + " + ".join(avail_preds)

        fold_r2s = []
        fold_rmses = []

        for test_lsoas in lsoa_folds:
            test_mask = model_df["LSOA21CD"].isin(test_lsoas)
            train_df = model_df[~test_mask]
            test_df = model_df[test_mask]

            if len(train_df) < 50 or len(test_df) < 10:
                continue

            try:
                model = smf.ols(formula, data=train_df).fit()
                y_true = test_df["log_energy_per_capita"].values
                y_pred = model.predict(test_df)

                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - y_true.mean()) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

                fold_r2s.append(r2)
                fold_rmses.append(rmse)
            except Exception:
                continue

        if fold_r2s:
            results.append({
                "model": model_name,
                "n_predictors": len(avail_preds),
                "mean_cv_r2": np.mean(fold_r2s),
                "std_cv_r2": np.std(fold_r2s),
                "mean_cv_rmse": np.mean(fold_rmses),
                "std_cv_rmse": np.std(fold_rmses),
            })

    # Display results
    print("\n  Model comparison (5-fold spatial CV):")
    print("-" * 70)
    print(f"  {'Model':<25} {'CV R²':>12} {'CV RMSE':>12} {'# Predictors':>12}")
    print("-" * 70)

    for r in results:
        print(f"  {r['model']:<25} {r['mean_cv_r2']:>8.3f}±{r['std_cv_r2']:.3f} {r['mean_cv_rmse']:>8.3f}±{r['std_cv_rmse']:.3f} {r['n_predictors']:>12}")

    print("-" * 70)

    # Best model
    if results:
        best = max(results, key=lambda x: x["mean_cv_r2"])
        print(f"\n  Best model by CV R²: {best['model']} (R² = {best['mean_cv_r2']:.3f})")

        # Check if adding predictors improves CV performance
        if len(results) >= 2:
            r2_m1 = results[0]["mean_cv_r2"]
            r2_best = best["mean_cv_r2"]
            improvement = r2_best - r2_m1
            print(f"  Improvement over M1: ΔR² = {improvement:+.3f}")

            if improvement < 0.01:
                print("  → Additional predictors provide minimal CV improvement")
                print("  → Simpler model may be preferred (parsimony)")

    return pd.DataFrame(results)


def cv_summary() -> None:
    """Print summary of cross-validation analysis."""
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)
    print("""
KEY INSIGHTS:

1. RANDOM VS SPATIAL CV
   - Random CV often overestimates performance
   - Spatial CV tests geographic generalization
   - Use spatial CV for claims about new areas

2. INTERPRETING CV RESULTS
   - CV R² < in-sample R²: Model is overfitting
   - Large CV variance: Unstable predictions
   - Spatial CV << Random CV: Spatial structure matters

3. MODEL SELECTION
   - Compare CV R² across specifications
   - More predictors ≠ better CV performance
   - Prefer parsimonious models if CV is similar

4. LIMITATIONS
   - CV assumes future data is similar to training
   - Spatial CV within one city may not generalize to others
   - Small folds can have high variance

5. RECOMMENDATIONS
   - Report spatial CV for geographic claims
   - Compare random and spatial CV
   - Use CV for model selection, not final performance
""")


def main() -> None:
    """Run cross-validation analysis."""
    print("=" * 70)
    print("CROSS-VALIDATION ANALYSIS")
    print("=" * 70)
    print("\nAssessing model performance and geographic generalization")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_and_prepare_data()

    # Random CV (baseline)
    random_results = random_cv(df, n_folds=5)

    # Spatial CV by LSOA
    spatial_results = spatial_cv_by_lsoa(df, n_folds=5)

    # Compare methods
    compare_cv_methods(random_results, spatial_results)

    # Model comparison with spatial CV
    model_comparison = model_comparison_cv(df)

    # Save results
    if not model_comparison.empty:
        model_comparison.to_csv(OUTPUT_DIR / "cv_model_comparison.csv", index=False)
        print(f"\nModel comparison saved to: {OUTPUT_DIR / 'cv_model_comparison.csv'}")

    # Summary
    cv_summary()


if __name__ == "__main__":
    main()
