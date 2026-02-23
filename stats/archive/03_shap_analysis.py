"""
SHAP analysis for energy per capita prediction.

Uses XGBoost to model energy per capita and SHAP values to understand
feature importance, non-linear effects, and interactions.

This script implements Phase 3 (complementary) of the statistical workflow:
- Train gradient boosting model
- Compute SHAP values for feature importance
- Generate dependence plots for non-linear effects
- Detect interactions between features

Usage:
    uv run python stats/03_shap_analysis.py
"""


import geopandas as gpd
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split

# Configuration
from urban_energy.paths import TEMP_DIR

DATA_PATH = TEMP_DIR / "processing" / "test" / "uprn_integrated.gpkg"
OUTPUT_DIR = TEMP_DIR / "stats"


def load_and_prepare_data() -> tuple[pd.DataFrame, list[str]]:
    """
    Load UPRN data and prepare features for modelling.

    Returns
    -------
    df : pd.DataFrame
        Analysis-ready dataframe
    feature_names : list[str]
        List of feature column names
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

    # Compute energy per capita
    df["energy_per_capita"] = df["ENERGY_CONSUMPTION_CURRENT"] / df["avg_household_size"]

    # Filter out records with invalid energy values (inf/nan/negative)
    valid_energy = (
        np.isfinite(df["energy_per_capita"])
        & (df["energy_per_capita"] > 0)
    )
    n_invalid = (~valid_energy).sum()
    if n_invalid > 0:
        print(f"  Removed {n_invalid:,} records with invalid energy values")
        df = df[valid_energy].copy()

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

    # Encode property type as numeric
    property_type_map = {"House": 0, "Bungalow": 1, "Flat": 2, "Maisonette": 3}
    df["property_type_num"] = df["PROPERTY_TYPE"].map(property_type_map).fillna(0)

    # Encode built form as numeric (ordered by attached-ness)
    built_form_map = {
        "Detached": 0,
        "Semi-Detached": 1,
        "End-Terrace": 2,
        "Enclosed End-Terrace": 2,
        "Mid-Terrace": 3,
    }
    df["built_form_num"] = df["BUILT_FORM"].map(built_form_map).fillna(1)

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

    # Define features
    features = [
        # Building characteristics
        "TOTAL_FLOOR_AREA",
        "property_type_num",
        "built_form_num",
        "building_age",
        "avg_household_size",
        # Morphology
        "footprint_area_m2",
        "compactness",
        "convexity",
        "shared_wall_ratio",
        "orientation",
        # Network centrality
        "cc_harmonic_800",
        "cc_betweenness_800",
        # Accessibility
        "cc_fsa_restaurant_800_nw",
        "cc_fsa_pub_800_nw",
        "cc_bus_800_nw",
        "cc_greenspace_800_nw",
        # Census demographics
        "pop_density",
        "pct_owner_occupied",
        "pct_deprived",
        # Travel to work
        "pct_car_commute",
        "pct_active_travel",
        "pct_public_transport",
        # Building height
        "building_height",
    ]

    # Filter to available features
    available_features = [f for f in features if f in df.columns]
    print(f"  Features available: {len(available_features)}/{len(features)}")

    # Filter to complete cases
    target = "energy_per_capita"
    model_cols = [target] + available_features
    df = df[model_cols].dropna()
    print(f"  Complete cases: {len(df):,}")

    return df, available_features


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[GradientBoostingRegressor, dict]:
    """
    Train Gradient Boosting model with cross-validation.

    Returns
    -------
    model : GradientBoostingRegressor
        Fitted model
    metrics : dict
        Cross-validation metrics
    """
    print("\n" + "=" * 70)
    print("TRAINING GRADIENT BOOSTING MODEL")
    print("=" * 70)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n  Train: {len(X_train)}, Test: {len(X_test)}")

    # Define model (sklearn's GradientBoostingRegressor)
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    print(f"\n  5-Fold CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Fit on full training data
    model.fit(X_train, y_train)

    # Evaluate on test set
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"  Train R²: {train_score:.3f}")
    print(f"  Test R²:  {test_score:.3f}")

    # Feature importance (gain-based)
    print("\n### Feature Importance (XGBoost Gain)")
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']:30} {row['importance']:.3f}")

    metrics = {
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std": cv_scores.std(),
        "train_r2": train_score,
        "test_r2": test_score,
    }

    return model, metrics


def compute_shap_values(
    model: GradientBoostingRegressor,
    X: pd.DataFrame,
) -> tuple[shap.Explainer, np.ndarray]:
    """
    Compute SHAP values for all observations.

    Returns
    -------
    explainer : shap.Explainer
        SHAP TreeExplainer
    shap_values : np.ndarray
        SHAP values matrix (n_samples, n_features)
    """
    print("\n" + "=" * 70)
    print("COMPUTING SHAP VALUES")
    print("=" * 70)

    # Use TreeExplainer for XGBoost (exact and fast)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    print(f"\n  SHAP values shape: {shap_values.shape}")

    return explainer, shap_values


def analyze_shap_importance(
    shap_values: np.ndarray,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze global feature importance from SHAP values.

    Returns mean |SHAP| for each feature.
    """
    print("\n" + "=" * 70)
    print("SHAP FEATURE IMPORTANCE")
    print("=" * 70)

    # Mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    print("\n### Global Feature Importance (mean |SHAP|)")
    print("  Higher = more important for predictions\n")

    for i, row in importance_df.iterrows():
        bar = "█" * int(row["mean_abs_shap"] * 2)
        print(f"  {row['feature']:30} {row['mean_abs_shap']:6.2f}  {bar}")

    return importance_df


def analyze_shap_directions(
    shap_values: np.ndarray,
    X: pd.DataFrame,
) -> None:
    """
    Analyze effect directions from SHAP values.

    For each feature, compute correlation between feature value and SHAP value
    to determine if higher values increase or decrease predictions.
    """
    print("\n" + "=" * 70)
    print("SHAP EFFECT DIRECTIONS")
    print("=" * 70)
    print("\n  Positive correlation: Higher feature → Higher energy per capita")
    print("  Negative correlation: Higher feature → Lower energy per capita\n")

    results = []
    for i, col in enumerate(X.columns):
        feature_values = X[col].values
        feature_shap = shap_values[:, i]

        # Correlation between feature and its SHAP values
        corr = np.corrcoef(feature_values, feature_shap)[0, 1]

        direction = "↑ increases" if corr > 0.1 else "↓ decreases" if corr < -0.1 else "~ neutral"
        results.append({
            "feature": col,
            "correlation": corr,
            "direction": direction,
        })

    results_df = pd.DataFrame(results).sort_values("correlation", key=abs, ascending=False)

    for _, row in results_df.iterrows():
        symbol = "+" if row["correlation"] > 0 else "-" if row["correlation"] < 0 else " "
        print(f"  {row['feature']:30} {symbol}{abs(row['correlation']):.2f}  {row['direction']}")


def analyze_density_effect(
    shap_values: np.ndarray,
    X: pd.DataFrame,
) -> None:
    """
    Detailed analysis of density/centrality effects on energy per capita.

    Tests whether the relationship is linear or shows non-linear patterns.
    """
    print("\n" + "=" * 70)
    print("DENSITY/CENTRALITY EFFECTS ON ENERGY")
    print("=" * 70)

    density_vars = [
        ("pop_density", "Population Density"),
        ("cc_harmonic_800", "Network Centrality"),
        ("cc_bus_800_nw", "Bus Accessibility"),
    ]

    for var, label in density_vars:
        if var not in X.columns:
            continue

        idx = list(X.columns).index(var)
        feature_values = X[var].values
        feature_shap = shap_values[:, idx]

        # Create terciles
        tercile_labels = ["Low", "Medium", "High"]
        terciles = pd.qcut(feature_values, q=3, labels=tercile_labels, duplicates="drop")

        print(f"\n### {label}")
        print(f"  Feature range: {feature_values.min():.2f} to {feature_values.max():.2f}")

        # SHAP by tercile
        for t in tercile_labels:
            mask = terciles == t
            if mask.sum() > 0:
                mean_shap = feature_shap[mask].mean()
                direction = "↑" if mean_shap > 0 else "↓"
                print(f"  {t:8} tercile: SHAP = {mean_shap:+.1f} kWh/capita {direction}")


def save_results(
    importance_df: pd.DataFrame,
    metrics: dict,
) -> None:
    """Save analysis results to files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save importance
    importance_df.to_csv(OUTPUT_DIR / "shap_importance.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'shap_importance.csv'}")

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'model_metrics.csv'}")


def main() -> None:
    """Run SHAP analysis."""
    print("=" * 70)
    print("SHAP ANALYSIS: Energy per Capita")
    print("=" * 70)
    print("\nObjective: Understand which features drive energy per capita")
    print("           and whether density/compactness reduces energy use")
    print(f"\nData: {DATA_PATH}")

    # Load and prepare data
    df, feature_names = load_and_prepare_data()

    # Split features and target
    X = df[feature_names]
    y = df["energy_per_capita"]

    print(f"\nTarget: energy_per_capita")
    print(f"  Mean: {y.mean():.1f} kWh/person/year")
    print(f"  Std:  {y.std():.1f}")

    # Train Gradient Boosting model
    model, metrics = train_model(X, y)

    # Compute SHAP values
    explainer, shap_values = compute_shap_values(model, X)

    # Analyze importance
    importance_df = analyze_shap_importance(shap_values, X)

    # Analyze directions
    analyze_shap_directions(shap_values, X)

    # Analyze density effects
    analyze_density_effect(shap_values, X)

    # Save results
    save_results(importance_df, metrics)

    # Summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print("""
1. MODEL PERFORMANCE
   - XGBoost can capture non-linear relationships
   - Check Test R² for generalization ability

2. FEATURE IMPORTANCE
   - Which features matter most for energy per capita?
   - Building size vs density vs accessibility?

3. EFFECT DIRECTIONS
   - Does higher density INCREASE or DECREASE energy?
   - This tests the compact city hypothesis

4. NON-LINEAR EFFECTS
   - Are effects consistent across the range?
   - Or do they plateau/reverse at extremes?

5. INTERPRETATION NOTES
   - SHAP values are in units of energy_per_capita (kWh/person/year)
   - Positive SHAP = feature pushes prediction UP (more energy)
   - Negative SHAP = feature pushes prediction DOWN (less energy)
""")


if __name__ == "__main__":
    main()
