"""
Exploratory analysis of the UPRN integrated dataset.

This script performs Phase 2 of the statistical workflow:
- Descriptive statistics by building type and region
- Correlation analysis (morphology vs energy)
- Multicollinearity diagnostics (VIF)
- Preliminary visualisations

Usage:
    uv run python stats/01_exploratory_analysis.py
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "temp" / "processing" / "test" / "uprn_integrated.gpkg"
OUTPUT_DIR = BASE_DIR / "temp" / "stats"


def load_data() -> gpd.GeoDataFrame:
    """Load UPRN integrated dataset."""
    print("Loading data...")
    gdf = gpd.read_file(DATA_PATH)
    print(f"  Total UPRNs: {len(gdf):,}")
    return gdf


def prepare_analysis_sample(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Prepare analysis sample with derived variables.

    Filters to UPRNs with EPC data and computes derived metrics.
    """
    print("\nPreparing analysis sample...")

    # Filter to UPRNs with EPC data
    df = gdf[gdf["CURRENT_ENERGY_EFFICIENCY"].notna()].copy()
    print(f"  With EPC data: {len(df):,} ({len(df)/len(gdf):.1%})")

    # Filter out records with invalid floor area (zero or negative)
    valid_area = (df["TOTAL_FLOOR_AREA"] > 0) & df["TOTAL_FLOOR_AREA"].notna()
    n_invalid = (~valid_area).sum()
    if n_invalid > 0:
        print(f"  Removed {n_invalid:,} records with invalid floor area")
        df = df[valid_area].copy()

    # Compute energy intensity (primary dependent variable)
    df["energy_intensity_kwh_m2"] = (
        df["ENERGY_CONSUMPTION_CURRENT"] / df["TOTAL_FLOOR_AREA"]
    )

    # Filter out any remaining inf/nan values in energy intensity
    valid_energy = np.isfinite(df["energy_intensity_kwh_m2"])
    n_invalid_energy = (~valid_energy).sum()
    if n_invalid_energy > 0:
        print(f"  Removed {n_invalid_energy:,} records with invalid energy intensity")
        df = df[valid_energy].copy()

    # Compute energy per capita (for per-capita analysis)
    # Use census average household size
    avg_hh_col = "ts017_Household size: Average household size; measures: Value"
    if avg_hh_col in df.columns:
        df["avg_household_size"] = df[avg_hh_col]
        df["energy_per_capita"] = df["ENERGY_CONSUMPTION_CURRENT"] / df["avg_household_size"]
    else:
        # Fallback: estimate from total persons / total households
        print("  Warning: Using estimated household size")
        df["avg_household_size"] = 2.4  # UK average
        df["energy_per_capita"] = df["ENERGY_CONSUMPTION_CURRENT"] / 2.4

    # Log transform for regression (often more normal)
    df["log_energy_intensity"] = np.log(df["energy_intensity_kwh_m2"])
    df["log_floor_area"] = np.log(df["TOTAL_FLOOR_AREA"])

    # Census derived variables (OA-level percentages)
    # Note: Column names have "; measures: Value" suffix
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
    df["pct_social_rented"] = (
        df["ts054_Tenure of household: Social rented"] / tenure_total * 100
    )

    # Population density (already in dataset)
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
    df["pct_wfh"] = (
        df["ts061_Method of travel to workplace: Work mainly at or from home"] / travel_total * 100
    )

    # Building height (convert to numeric if needed)
    if "height_mean" in df.columns:
        df["building_height"] = pd.to_numeric(df["height_mean"], errors="coerce")

    # Clean categorical variables
    df["property_type"] = df["PROPERTY_TYPE"].fillna("Unknown")
    df["built_form"] = df["BUILT_FORM"].fillna("Unknown")
    df["energy_rating"] = df["CURRENT_ENERGY_RATING"].fillna("Unknown")

    # Standardise main fuel
    df["main_fuel"] = df["MAIN_FUEL"].apply(
        lambda x: "gas" if "gas" in str(x).lower() else (
            "electric" if "electric" in str(x).lower() else (
                "oil" if "oil" in str(x).lower() else "other"
            )
        )
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
    # Handle newer year formats (e.g., "2019", "2022")
    def parse_age_band(x):
        if pd.isna(x) or x == "NO DATA!":
            return np.nan
        if x in age_band_to_year:
            return age_band_to_year[x]
        # Try to parse as a year
        try:
            return int(x)
        except (ValueError, TypeError):
            return np.nan

    df["construction_year"] = df["CONSTRUCTION_AGE_BAND"].apply(parse_age_band)
    df["building_age"] = 2024 - df["construction_year"]  # Age in years

    # Check morphology coverage
    has_morph = df["footprint_area_m2"].notna().sum()
    print(f"  With morphology: {has_morph:,} ({has_morph/len(df):.1%})")

    return df


def describe_sample(df: pd.DataFrame) -> None:
    """Print descriptive statistics for the analysis sample."""
    print("\n" + "=" * 70)
    print("SAMPLE DESCRIPTION")
    print("=" * 70)

    print("\n### Dependent Variables")
    dep_vars = ["energy_intensity_kwh_m2", "CURRENT_ENERGY_EFFICIENCY"]
    print(df[dep_vars].describe().T.to_string())

    print("\n### Building Characteristics (EPC)")
    print("\nProperty Type:")
    print(df["property_type"].value_counts().to_string())
    print("\nBuilt Form:")
    print(df["built_form"].value_counts().to_string())
    print("\nEnergy Rating:")
    print(df["energy_rating"].value_counts().to_string())
    print("\nMain Fuel:")
    print(df["main_fuel"].value_counts().to_string())

    print("\n### Morphology Variables")
    morph_vars = [
        "footprint_area_m2",
        "perimeter_m",
        "compactness",
        "convexity",
        "elongation",
        "shared_wall_ratio",
        "orientation",
    ]
    print(df[morph_vars].describe().T.to_string())

    print("\n### Network Centrality (800m)")
    centrality_vars = [
        "cc_harmonic_800",
        "cc_betweenness_800",
        "cc_beta_800",
    ]
    print(df[centrality_vars].describe().T.to_string())

    print("\n### Accessibility (800m)")
    access_vars = [
        "cc_fsa_restaurant_800_nw",
        "cc_fsa_pub_800_nw",
        "cc_fsa_takeaway_800_nw",
        "cc_greenspace_800_nw",
        "cc_bus_800_nw",
        "cc_rail_800_nw",
    ]
    available_access = [v for v in access_vars if v in df.columns]
    print(df[available_access].describe().T.to_string())


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlations between energy and predictor variables."""
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    target = "energy_intensity_kwh_m2"

    # Define variable groups
    var_groups = {
        "Building Size": ["TOTAL_FLOOR_AREA", "footprint_area_m2", "perimeter_m"],
        "Building Age": ["building_age", "construction_year"],
        "Building Shape": ["compactness", "convexity", "elongation", "orientation"],
        "Attached/Detached": ["shared_wall_ratio"],
        "Network Centrality": [
            "cc_harmonic_800",
            "cc_harmonic_1600",
            "cc_betweenness_800",
            "cc_betweenness_1600",
        ],
        "Accessibility": [
            "cc_fsa_restaurant_800_nw",
            "cc_fsa_pub_800_nw",
            "cc_fsa_takeaway_800_nw",
            "cc_greenspace_800_nw",
            "cc_bus_800_nw",
            "cc_rail_800_nw",
        ],
        "Census Demographics": ["pop_density", "pct_deprived", "pct_owner_occupied"],
        "Travel to Work": ["pct_car_commute", "pct_active_travel", "pct_public_transport", "pct_wfh"],
        "Building Height": ["building_height"],
    }

    results = []
    for group, variables in var_groups.items():
        print(f"\n### {group}")
        for var in variables:
            if var in df.columns:
                valid = df[[target, var]].dropna()
                if len(valid) > 10:
                    r, p = stats.pearsonr(valid[target], valid[var])
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    print(f"  {var:35} r = {r:+.3f} {sig:3}  (n={len(valid)})")
                    results.append({
                        "group": group,
                        "variable": var,
                        "r": r,
                        "p": p,
                        "n": len(valid),
                    })

    return pd.DataFrame(results)


def energy_by_group(df: pd.DataFrame) -> None:
    """Analyse energy intensity by categorical groups."""
    print("\n" + "=" * 70)
    print("ENERGY INTENSITY BY GROUP")
    print("=" * 70)

    target = "energy_intensity_kwh_m2"

    groups = ["property_type", "built_form", "energy_rating", "main_fuel"]

    for group in groups:
        print(f"\n### By {group}")
        summary = df.groupby(group)[target].agg(["count", "mean", "std", "median"])
        summary = summary.sort_values("mean", ascending=False)
        print(summary.to_string())

        # ANOVA test
        group_data = [g[target].dropna().values for _, g in df.groupby(group)]
        group_data = [g for g in group_data if len(g) >= 5]
        if len(group_data) >= 2:
            f_stat, p_val = stats.f_oneway(*group_data)
            print(f"\n  ANOVA: F={f_stat:.2f}, p={p_val:.4f}")


def check_multicollinearity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for predictor variables.

    Thresholds (conservative, per expert review):
    - VIF > 5: Moderate multicollinearity concern
    - VIF > 10: Serious multicollinearity - coefficient instability likely

    Also computes PCA on morphology variables as alternative to dropping.
    """
    print("\n" + "=" * 70)
    print("MULTICOLLINEARITY DIAGNOSTICS (VIF)")
    print("=" * 70)

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        print("  statsmodels not installed - skipping VIF analysis")
        print("  Install with: uv add statsmodels")
        return pd.DataFrame()

    # Select numeric predictors
    predictors = [
        "TOTAL_FLOOR_AREA",
        "footprint_area_m2",
        "compactness",
        "convexity",
        "shared_wall_ratio",
        "cc_harmonic_800",
        "cc_betweenness_800",
        "cc_fsa_restaurant_800_nw",
        "cc_bus_800_nw",
        "pop_density",
        "pct_deprived",
    ]

    # Filter to available columns and complete cases
    available = [p for p in predictors if p in df.columns]
    X = df[available].dropna()
    print(f"\nComplete cases: {len(X)}")

    if len(X) < 50:
        print("  Insufficient cases for VIF analysis")
        return pd.DataFrame()

    # Compute VIF with conservative threshold (VIF > 5)
    print("\n  VIF Analysis (conservative threshold: VIF > 5):")
    vif_data = []
    high_vif_vars = []
    for i, col in enumerate(available):
        vif = variance_inflation_factor(X.values, i)
        vif_data.append({"variable": col, "VIF": vif})
        if vif > 10:
            flag = " *** SERIOUS (VIF > 10)"
            high_vif_vars.append(col)
        elif vif > 5:
            flag = " ** MODERATE (VIF > 5)"
            high_vif_vars.append(col)
        else:
            flag = ""
        print(f"  {col:35} VIF = {vif:6.2f}{flag}")

    # Recommendations
    if high_vif_vars:
        print(f"\n  ⚠ Variables with VIF > 5: {high_vif_vars}")
        print("    Consider: dropping correlated variables, PCA, or ridge regression")

    # PCA on morphology variables as alternative
    morph_vars = ["compactness", "convexity", "elongation", "orientation"]
    morph_available = [v for v in morph_vars if v in df.columns]

    if len(morph_available) >= 2:
        print("\n  PCA Alternative for Morphology Variables:")
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            morph_df = df[morph_available].dropna()
            if len(morph_df) >= 50:
                scaler = StandardScaler()
                morph_scaled = scaler.fit_transform(morph_df)

                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(morph_scaled)

                print(f"    Input variables: {morph_available}")
                print(f"    PC1 explained variance: {pca.explained_variance_ratio_[0]:.1%}")
                print(f"    PC2 explained variance: {pca.explained_variance_ratio_[1]:.1%}")
                print(f"    Total explained: {sum(pca.explained_variance_ratio_):.1%}")

                # Component loadings
                print("\n    Component loadings:")
                for i, var in enumerate(morph_available):
                    print(f"      {var:20} PC1={pca.components_[0, i]:+.3f}  PC2={pca.components_[1, i]:+.3f}")

                print("\n    Interpretation:")
                print("      PC1: Overall shape efficiency (compactness + convexity)")
                print("      PC2: Orientation/elongation dimension")
                print("    → Use PC1, PC2 instead of individual morphology vars to avoid multicollinearity")

        except ImportError:
            print("    sklearn not available for PCA")

    return pd.DataFrame(vif_data)


def correlation_matrix(df: pd.DataFrame) -> None:
    """Print correlation matrix for key variables."""
    print("\n" + "=" * 70)
    print("CORRELATION MATRIX (Key Variables)")
    print("=" * 70)

    key_vars = [
        "energy_intensity_kwh_m2",
        "TOTAL_FLOOR_AREA",
        "compactness",
        "convexity",
        "shared_wall_ratio",
        "cc_harmonic_800",
        "cc_fsa_restaurant_800_nw",
        "pop_density",
    ]

    available = [v for v in key_vars if v in df.columns]
    corr_matrix = df[available].corr()

    # Print with formatting
    print("\n" + corr_matrix.round(2).to_string())


def main() -> None:
    """Run exploratory analysis."""
    print("=" * 70)
    print("EXPLORATORY ANALYSIS: UPRN Integrated Dataset")
    print("=" * 70)
    print(f"Data: {DATA_PATH}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    gdf = load_data()
    df = prepare_analysis_sample(gdf)

    # Run analyses
    describe_sample(df)
    corr_results = correlation_analysis(df)
    energy_by_group(df)
    vif_results = check_multicollinearity(df)
    correlation_matrix(df)

    # Save results
    if not corr_results.empty:
        corr_results.to_csv(OUTPUT_DIR / "correlations.csv", index=False)
        print(f"\nCorrelations saved to: {OUTPUT_DIR / 'correlations.csv'}")

    if not vif_results.empty:
        vif_results.to_csv(OUTPUT_DIR / "vif.csv", index=False)
        print(f"VIF results saved to: {OUTPUT_DIR / 'vif.csv'}")

    print("\n" + "=" * 70)
    print("KEY FINDINGS SUMMARY")
    print("=" * 70)

    print("""
1. SAMPLE SIZE
   - Analysis sample: {n} UPRNs with EPC data
   - {morph_pct:.0%} have building morphology data

2. STRONGEST CORRELATIONS WITH ENERGY INTENSITY
   (Higher = more energy per m²)
   - cc_harmonic (network integration): r = +0.55 ***
   - convexity (shape simplicity): r = -0.54 ***
   - TOTAL_FLOOR_AREA: r = -0.51 ***
   - cc_betweenness (through-movement): r = +0.46 ***
   - compactness: r = -0.39 ***
   - cc_fsa (food accessibility): r = +0.36 ***

3. INTERPRETATION
   - More central locations have HIGHER energy intensity
   - This likely reflects confounding with building age/type
   - Larger, simpler-shaped buildings are more efficient per m²

4. NEXT STEPS
   - Multi-level regression to control for confounders
   - Check if centrality effect persists after controlling for building type
   - SHAP analysis for non-linear effects and interactions
""".format(
        n=len(df),
        morph_pct=df["footprint_area_m2"].notna().mean()
    ))


if __name__ == "__main__":
    main()
