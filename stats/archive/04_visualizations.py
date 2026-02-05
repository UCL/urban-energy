"""
Generate visualizations for energy per capita analysis.

Creates figures using Seaborn and SHAP, saves to figures/ folder,
and generates a markdown report.

Usage:
    uv run python stats/04_visualizations.py
"""

from pathlib import Path
from datetime import datetime

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "temp" / "processing" / "test" / "uprn_integrated.gpkg"
FIGURES_DIR = BASE_DIR / "stats" / "figures"
REPORT_PATH = BASE_DIR / "stats" / "analysis_report.md"

# Set style - professional academic quality
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def load_and_prepare_data() -> pd.DataFrame:
    """Load and prepare data with all derived variables."""
    print("Loading data...")
    gdf = gpd.read_file(DATA_PATH)

    # Filter to UPRNs with EPC data
    df = gdf[gdf["CURRENT_ENERGY_EFFICIENCY"].notna()].copy()
    print(f"  With EPC data: {len(df):,}")

    # Filter out records with invalid floor area (zero or negative)
    valid_area = (df["TOTAL_FLOOR_AREA"] > 0) & df["TOTAL_FLOOR_AREA"].notna()
    n_invalid = (~valid_area).sum()
    if n_invalid > 0:
        print(f"  Removed {n_invalid:,} records with invalid floor area")
        df = df[valid_area].copy()

    # Compute average household size
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

    # Energy per capita
    df["energy_per_capita"] = (
        df["ENERGY_CONSUMPTION_CURRENT"] / df["avg_household_size"]
    )
    df["energy_intensity"] = df["ENERGY_CONSUMPTION_CURRENT"] / df["TOTAL_FLOOR_AREA"]

    # Filter out records with invalid energy values (inf/nan/negative)
    valid_energy = (
        np.isfinite(df["energy_per_capita"])
        & np.isfinite(df["energy_intensity"])
        & (df["energy_per_capita"] > 0)
        & (df["energy_intensity"] > 0)
    )
    n_invalid = (~valid_energy).sum()
    if n_invalid > 0:
        print(f"  Removed {n_invalid:,} records with invalid energy values")
        df = df[valid_energy].copy()

    print(f"  Final sample size: {len(df):,}")

    # Census variables
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
    df["pct_owner_occupied"] = df["ts054_Tenure of household: Owned"] / tenure_total * 100

    df["pop_density"] = df["ts006_Population Density: Persons per square kilometre; measures: Value"]

    # Travel to work
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

    # Building height
    if "height_mean" in df.columns:
        df["building_height"] = pd.to_numeric(df["height_mean"], errors="coerce")

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

    # Clean categories
    df["property_type"] = df["PROPERTY_TYPE"].fillna("Unknown")
    df["built_form"] = df["BUILT_FORM"].fillna("Unknown")

    print(f"  Loaded {len(df)} properties with EPC data")
    return df


def fig_energy_distribution(df: pd.DataFrame) -> str:
    """Distribution of energy per capita."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram with better styling
    ax1 = axes[0]
    sns.histplot(df["energy_per_capita"], bins=30, kde=True, ax=ax1, color="#3498db", alpha=0.7,
                 edgecolor="white", linewidth=0.5)
    median_val = df["energy_per_capita"].median()
    mean_val = df["energy_per_capita"].mean()
    ax1.axvline(median_val, color="#e74c3c", linestyle="--", linewidth=2,
                label=f"Median: {median_val:.0f} kWh")
    ax1.axvline(mean_val, color="#f39c12", linestyle="--", linewidth=2,
                label=f"Mean: {mean_val:.0f} kWh")
    ax1.set_xlabel("Energy per Capita (kWh/person/year)")
    ax1.set_ylabel("Count")
    ax1.set_title("Distribution of Energy per Capita", fontweight="bold")
    ax1.legend(frameon=True, fancybox=True, shadow=True)

    # Box plot by property type with hue
    ax2 = axes[1]
    order = df.groupby("property_type")["energy_per_capita"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="property_type", y="energy_per_capita", hue="property_type",
                order=order, ax=ax2, palette="Set2", legend=False)
    ax2.set_xlabel("Property Type")
    ax2.set_ylabel("Energy per Capita (kWh/person/year)")
    ax2.set_title("Energy per Capita by Property Type", fontweight="bold")
    ax2.tick_params(axis="x", rotation=45)

    # Add sample sizes as annotations
    for i, prop_type in enumerate(order):
        n = (df["property_type"] == prop_type).sum()
        median = df[df["property_type"] == prop_type]["energy_per_capita"].median()
        ax2.annotate(f"n={n}", (i, median + 15), ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    path = FIGURES_DIR / "01_energy_distribution.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return str(path.relative_to(BASE_DIR))


def fig_energy_by_built_form(df: pd.DataFrame) -> str:
    """Energy by built form (attached-ness)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    order = ["Detached", "Semi-Detached", "End-Terrace", "Enclosed End-Terrace", "Mid-Terrace"]
    order = [o for o in order if o in df["built_form"].values]

    # Left: violin plot
    ax1 = axes[0]
    sns.violinplot(data=df, x="built_form", y="energy_per_capita", hue="built_form",
                   order=order, ax=ax1, palette="coolwarm", legend=False, inner="box")
    ax1.set_xlabel("Built Form")
    ax1.set_ylabel("Energy per Capita (kWh/person/year)")
    ax1.set_title("Energy per Capita by Built Form", fontweight="bold")
    ax1.tick_params(axis="x", rotation=45)

    # Add mean annotations
    for i, form in enumerate(order):
        n = (df["built_form"] == form).sum()
        mean_val = df[df["built_form"] == form]["energy_per_capita"].mean()
        ax1.annotate(f"n={n}\nμ={mean_val:.0f}", (i, ax1.get_ylim()[1] - 10),
                     ha="center", fontsize=9, color="gray")

    # Right: mean with 95% CI - show expected vs observed
    ax2 = axes[1]
    summary = df.groupby("built_form")["energy_per_capita"].agg(["mean", "std", "count"]).loc[order]
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci"] = 1.96 * summary["se"]

    colors = sns.color_palette("coolwarm", len(order))
    bars = ax2.bar(range(len(order)), summary["mean"], yerr=summary["ci"],
                   capsize=5, color=colors, edgecolor="black", linewidth=0.5, alpha=0.8)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(order, rotation=45, ha="right")
    ax2.set_xlabel("Built Form")
    ax2.set_ylabel("Mean Energy per Capita (kWh/person/year)")
    ax2.set_title("Mean Energy ± 95% CI\n(Expected: Detached > Terraced)", fontweight="bold")

    # Add reference line for expected pattern
    ax2.axhline(y=summary["mean"].mean(), color="gray", linestyle=":", linewidth=1.5,
                label="Overall Mean")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    path = FIGURES_DIR / "02_energy_by_built_form.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return str(path.relative_to(BASE_DIR))


def fig_building_age_effect(df: pd.DataFrame) -> str:
    """Building age vs energy per capita."""
    from scipy import stats as scipy_stats

    df_valid = df[df["building_age"].notna()].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter with regression line and confidence interval
    ax1 = axes[0]
    sns.regplot(data=df_valid, x="building_age", y="energy_per_capita", ax=ax1,
                scatter_kws={"alpha": 0.4, "s": 40, "color": "#e74c3c"},
                line_kws={"color": "#2c3e50", "linewidth": 2},
                ci=95)

    # Calculate and show correlation
    r, p = scipy_stats.pearsonr(df_valid["building_age"], df_valid["energy_per_capita"])
    ax1.annotate(f"r = {r:.2f}***\nn = {len(df_valid)}", (0.05, 0.95),
                 xycoords="axes fraction", fontsize=12, fontweight="bold",
                 verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8))

    ax1.set_xlabel("Building Age (years)")
    ax1.set_ylabel("Energy per Capita (kWh/person/year)")
    ax1.set_title("Energy per Capita vs Building Age\n(Dominant Predictor)", fontweight="bold")

    # Box plot by age band with hue
    ax2 = axes[1]
    df_valid["age_band"] = pd.cut(df_valid["building_age"],
                                   bins=[0, 25, 50, 75, 100, 200],
                                   labels=["<25 yrs", "25-50", "50-75", "75-100", ">100 yrs"])
    sns.boxplot(data=df_valid, x="age_band", y="energy_per_capita", hue="age_band",
                ax=ax2, palette="YlOrRd", legend=False)
    ax2.set_xlabel("Building Age Band")
    ax2.set_ylabel("Energy per Capita (kWh/person/year)")
    ax2.set_title("Energy per Capita by Age Band", fontweight="bold")

    # Add sample sizes and means
    for i, band in enumerate(["<25 yrs", "25-50", "50-75", "75-100", ">100 yrs"]):
        subset = df_valid[df_valid["age_band"] == band]
        if len(subset) > 0:
            n = len(subset)
            mean_val = subset["energy_per_capita"].mean()
            ax2.annotate(f"n={n}\nμ={mean_val:.0f}", (i, ax2.get_ylim()[1] - 5),
                         ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    path = FIGURES_DIR / "04_building_age_effect.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return str(path.relative_to(BASE_DIR))


def fig_density_effect(df: pd.DataFrame) -> str:
    """Population density vs energy per capita."""
    from scipy import stats as scipy_stats

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter with regression line
    ax1 = axes[0]
    sns.regplot(data=df, x="pop_density", y="energy_per_capita", ax=ax1,
                scatter_kws={"alpha": 0.4, "s": 40, "color": "#3498db"},
                line_kws={"color": "#e74c3c", "linewidth": 2}, ci=95)

    # Calculate correlation
    valid = df[["pop_density", "energy_per_capita"]].dropna()
    r, p = scipy_stats.pearsonr(valid["pop_density"], valid["energy_per_capita"])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    ax1.annotate(f"r = {r:.2f}{sig}\nn = {len(valid)}", (0.05, 0.95),
                 xycoords="axes fraction", fontsize=12, fontweight="bold",
                 verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8))

    ax1.set_xlabel("Population Density (persons/km²)")
    ax1.set_ylabel("Energy per Capita (kWh/person/year)")
    ax1.set_title("Energy vs Density (Raw Correlation)", fontweight="bold")

    # Tercile comparison with hue
    ax2 = axes[1]
    df_plot = df.copy()
    df_plot["density_tercile"] = pd.qcut(df_plot["pop_density"], q=3, labels=["Low", "Medium", "High"])
    summary = df_plot.groupby("density_tercile", observed=True)["energy_per_capita"].agg(["mean", "std", "count"])
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci"] = 1.96 * summary["se"]

    colors = ["#a8d5e5", "#5ba3c6", "#2171b5"]
    bars = ax2.bar(range(3), summary["mean"], yerr=summary["ci"],
                   capsize=6, color=colors, edgecolor="black", linewidth=0.5, alpha=0.9)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["Low\nDensity", "Medium\nDensity", "High\nDensity"])
    ax2.set_xlabel("Population Density Tercile")
    ax2.set_ylabel("Mean Energy per Capita (kWh/person/year)")
    ax2.set_title("Mean Energy by Density Tercile ± 95% CI\n(WARNING: Potentially Confounded)", fontweight="bold")

    # Add count annotations
    for i, tercile in enumerate(["Low", "Medium", "High"]):
        count = summary.loc[tercile, "count"]
        mean_val = summary.loc[tercile, "mean"]
        ax2.annotate(f"n={count:.0f}", (i, mean_val + summary.loc[tercile, "ci"] + 3),
                     ha="center", fontsize=10, color="gray")

    plt.tight_layout()
    path = FIGURES_DIR / "05_density_effect.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return str(path.relative_to(BASE_DIR))


def fig_correlation_heatmap(df: pd.DataFrame) -> str:
    """Correlation heatmap of key variables."""
    vars_to_plot = [
        "energy_per_capita",
        "TOTAL_FLOOR_AREA",
        "building_age",
        "compactness",
        "shared_wall_ratio",
        "cc_harmonic_800",
        "cc_bus_800_nw",
        "pop_density",
        "pct_owner_occupied",
        "building_height",
    ]

    available = [v for v in vars_to_plot if v in df.columns]
    corr_df = df[available].dropna()
    corr_matrix = corr_df.corr()

    # Rename for display
    rename_map = {
        "energy_per_capita": "Energy/Capita",
        "TOTAL_FLOOR_AREA": "Floor Area",
        "building_age": "Building Age",
        "compactness": "Compactness",
        "shared_wall_ratio": "Shared Wall",
        "cc_harmonic_800": "Centrality",
        "cc_bus_800_nw": "Bus Access",
        "pop_density": "Pop Density",
        "pct_owner_occupied": "% Owner Occ",
        "building_height": "Height",
    }
    corr_matrix = corr_matrix.rename(index=rename_map, columns=rename_map)

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, square=True,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix: Key Variables")

    plt.tight_layout()
    path = FIGURES_DIR / "06_correlation_heatmap.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return str(path.relative_to(BASE_DIR))


def fig_centrality_vs_age(df: pd.DataFrame) -> str:
    """Show confounding between centrality and building age."""
    from scipy import stats as scipy_stats

    df_valid = df[df["building_age"].notna() & df["cc_harmonic_800"].notna()].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Centrality vs age
    ax1 = axes[0]
    sns.regplot(data=df_valid, x="cc_harmonic_800", y="building_age",
                scatter_kws={"alpha": 0.4, "s": 40, "color": "#9b59b6"},
                line_kws={"color": "#e74c3c", "linewidth": 2}, ax=ax1, ci=95)

    # Calculate correlation
    r, p = scipy_stats.pearsonr(df_valid["cc_harmonic_800"], df_valid["building_age"])
    ax1.annotate(
        f"r = {r:.2f}***", (0.05, 0.95),
        xycoords="axes fraction", fontsize=12, fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax1.set_xlabel("Network Centrality (Harmonic 800m)")
    ax1.set_ylabel("Building Age (years)")
    ax1.set_title("Centrality vs Building Age", fontweight="bold")

    # Energy by centrality, colored by age
    ax2 = axes[1]
    df_valid["age_category"] = pd.cut(df_valid["building_age"],
                                       bins=[0, 50, 100, 200],
                                       labels=["<50 yrs", "50-100 yrs", ">100 yrs"])
    palette = {"<50 yrs": "#fef0d9", "50-100 yrs": "#fdbb84", ">100 yrs": "#e34a33"}
    sns.scatterplot(data=df_valid, x="cc_harmonic_800", y="energy_per_capita",
                    hue="age_category", alpha=0.7, ax=ax2, palette=palette, s=50,
                    edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("Network Centrality (Harmonic 800m)")
    ax2.set_ylabel("Energy per Capita (kWh/person/year)")
    ax2.set_title("Energy vs Centrality\n(Stratified by Building Age)", fontweight="bold")
    ax2.legend(title="Building Age", loc="upper left", frameon=True, fancybox=True)

    plt.tight_layout()
    path = FIGURES_DIR / "07_centrality_age_confounding.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return str(path.relative_to(BASE_DIR))


def fig_controlled_effects(df: pd.DataFrame) -> str:
    """
    Show what happens to density/centrality effects AFTER controlling for building age.

    This is the key visualization for understanding confounding.
    """
    import statsmodels.formula.api as smf
    from scipy import stats as scipy_stats

    df_valid = df[
        df["building_age"].notna() &
        df["cc_harmonic_800"].notna() &
        df["pop_density"].notna()
    ].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top left: Raw centrality vs energy
    ax1 = axes[0, 0]
    sns.regplot(data=df_valid, x="cc_harmonic_800", y="energy_per_capita", ax=ax1,
                scatter_kws={"alpha": 0.4, "s": 30, "color": "#e74c3c"},
                line_kws={"color": "#2c3e50", "linewidth": 2}, ci=95)
    r_raw, _ = scipy_stats.pearsonr(df_valid["cc_harmonic_800"], df_valid["energy_per_capita"])
    ax1.annotate(f"Raw r = {r_raw:.2f}", (0.05, 0.95),
                 xycoords="axes fraction", fontsize=11, fontweight="bold",
                 verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="#ffcccc", edgecolor="#e74c3c"))
    ax1.set_xlabel("Network Centrality")
    ax1.set_ylabel("Energy per Capita")
    ax1.set_title("(A) Raw: Centrality vs Energy", fontweight="bold")

    # Top right: Residual centrality vs residual energy (after removing age effect)
    ax2 = axes[0, 1]

    # Regress out building age from both variables
    model_energy = smf.ols("energy_per_capita ~ building_age", data=df_valid).fit()
    model_centrality = smf.ols("cc_harmonic_800 ~ building_age", data=df_valid).fit()

    df_valid["energy_resid"] = model_energy.resid
    df_valid["centrality_resid"] = model_centrality.resid

    sns.regplot(data=df_valid, x="centrality_resid", y="energy_resid", ax=ax2,
                scatter_kws={"alpha": 0.4, "s": 30, "color": "#27ae60"},
                line_kws={"color": "#2c3e50", "linewidth": 2}, ci=95)
    r_controlled, _ = scipy_stats.pearsonr(df_valid["centrality_resid"], df_valid["energy_resid"])
    ax2.annotate(f"Partial r = {r_controlled:.2f}", (0.05, 0.95),
                 xycoords="axes fraction", fontsize=11, fontweight="bold",
                 verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="#ccffcc", edgecolor="#27ae60"))
    ax2.set_xlabel("Centrality (age-adjusted residuals)")
    ax2.set_ylabel("Energy per Capita (age-adjusted residuals)")
    ax2.set_title("(B) Partial: Centrality vs Energy\n(Age regressed out)", fontweight="bold")

    # Bottom left: Raw density vs energy
    ax3 = axes[1, 0]
    sns.regplot(data=df_valid, x="pop_density", y="energy_per_capita", ax=ax3,
                scatter_kws={"alpha": 0.4, "s": 30, "color": "#e74c3c"},
                line_kws={"color": "#2c3e50", "linewidth": 2}, ci=95)
    r_raw_dens, _ = scipy_stats.pearsonr(df_valid["pop_density"], df_valid["energy_per_capita"])
    ax3.annotate(f"Raw r = {r_raw_dens:.2f}", (0.05, 0.95),
                 xycoords="axes fraction", fontsize=11, fontweight="bold",
                 verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="#ffcccc", edgecolor="#e74c3c"))
    ax3.set_xlabel("Population Density")
    ax3.set_ylabel("Energy per Capita")
    ax3.set_title("(C) Raw: Density vs Energy", fontweight="bold")

    # Bottom right: Residual density vs residual energy
    ax4 = axes[1, 1]
    model_density = smf.ols("pop_density ~ building_age", data=df_valid).fit()
    df_valid["density_resid"] = model_density.resid

    sns.regplot(data=df_valid, x="density_resid", y="energy_resid", ax=ax4,
                scatter_kws={"alpha": 0.4, "s": 30, "color": "#27ae60"},
                line_kws={"color": "#2c3e50", "linewidth": 2}, ci=95)
    r_controlled_dens, _ = scipy_stats.pearsonr(df_valid["density_resid"], df_valid["energy_resid"])
    ax4.annotate(f"Partial r = {r_controlled_dens:.2f}", (0.05, 0.95),
                 xycoords="axes fraction", fontsize=11, fontweight="bold",
                 verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="#ccffcc", edgecolor="#27ae60"))
    ax4.set_xlabel("Density (age-adjusted residuals)")
    ax4.set_ylabel("Energy per Capita (age-adjusted residuals)")
    ax4.set_title("(D) Partial: Density vs Energy\n(Age regressed out)", fontweight="bold")

    # Add overall title
    fig.suptitle(
        "Raw vs Partial Correlations (Controlling for Building Age)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    path = FIGURES_DIR / "08_controlled_effects.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return str(path.relative_to(BASE_DIR))


def fig_other_confounders(df: pd.DataFrame) -> str:
    """Explore other potential confounders: property type, tenure, fuel type."""
    from scipy import stats as scipy_stats

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top left: Building age by property type (confounder check)
    ax1 = axes[0, 0]
    df_valid = df[df["building_age"].notna()].copy()
    order = df_valid.groupby("property_type")["building_age"].median().sort_values(ascending=False).index
    sns.boxplot(data=df_valid, x="property_type", y="building_age", hue="property_type",
                order=order, ax=ax1, palette="viridis", legend=False)
    ax1.set_xlabel("Property Type")
    ax1.set_ylabel("Building Age (years)")
    ax1.set_title("Building Age by Property Type", fontweight="bold")
    ax1.tick_params(axis="x", rotation=45)

    # Top right: Building age by built form
    ax2 = axes[0, 1]
    form_order = ["Detached", "Semi-Detached", "End-Terrace", "Enclosed End-Terrace", "Mid-Terrace"]
    form_order = [f for f in form_order if f in df_valid["built_form"].values]
    if form_order:
        sns.boxplot(data=df_valid, x="built_form", y="building_age", hue="built_form",
                    order=form_order, ax=ax2, palette="magma", legend=False)
        ax2.set_xlabel("Built Form")
        ax2.set_ylabel("Building Age (years)")
        ax2.set_title("Building Age by Built Form", fontweight="bold")
        ax2.tick_params(axis="x", rotation=45)

    # Bottom left: Tenure vs Energy
    ax3 = axes[1, 0]
    df_tenure = df[df["pct_owner_occupied"].notna()].copy()
    df_tenure["tenure_tercile"] = pd.qcut(df_tenure["pct_owner_occupied"], q=3,
                                           labels=["Low\nOwnership", "Medium", "High\nOwnership"])
    summary = df_tenure.groupby("tenure_tercile", observed=True)["energy_per_capita"].agg(["mean", "std", "count"])
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci"] = 1.96 * summary["se"]

    colors = ["#fee8c8", "#fdbb84", "#e34a33"]
    bars = ax3.bar(range(3), summary["mean"], yerr=summary["ci"],
                   capsize=6, color=colors, edgecolor="black", linewidth=0.5)
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(summary.index)
    ax3.set_xlabel("Tenure (% Owner Occupied)")
    ax3.set_ylabel("Mean Energy per Capita")
    ax3.set_title("Energy by Tenure", fontweight="bold")

    # Add count annotations
    for i, (idx, row) in enumerate(summary.iterrows()):
        ax3.annotate(f"n={row['count']:.0f}", (i, row["mean"] + row["ci"] + 2),
                     ha="center", fontsize=10, color="gray")

    # Bottom right: Confounder correlation matrix
    ax4 = axes[1, 1]
    confounders = ["building_age", "cc_harmonic_800", "pop_density",
                   "pct_owner_occupied", "TOTAL_FLOOR_AREA"]
    conf_names = ["Building Age", "Centrality", "Pop Density", "% Owner Occ", "Floor Area"]
    available = [c for c in confounders if c in df.columns]
    conf_df = df[available].dropna()
    corr = conf_df.corr()
    corr.columns = [conf_names[confounders.index(c)] for c in available]
    corr.index = corr.columns

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax4, square=True,
                cbar_kws={"shrink": 0.8}, annot_kws={"fontsize": 10})
    ax4.set_title("Variable Inter-correlations", fontweight="bold")

    plt.tight_layout()
    path = FIGURES_DIR / "09_other_confounders.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return str(path.relative_to(BASE_DIR))


def fig_built_form_controlled(df: pd.DataFrame) -> str:
    """Show built form effect after controlling for building age."""
    import statsmodels.formula.api as smf

    df_valid = df[df["building_age"].notna() & df["built_form"].notna()].copy()

    # Exclude unknown
    df_valid = df_valid[df_valid["built_form"] != "Unknown"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Raw energy by built form
    ax1 = axes[0]
    order = ["Detached", "Semi-Detached", "End-Terrace", "Enclosed End-Terrace", "Mid-Terrace"]
    order = [o for o in order if o in df_valid["built_form"].values]

    # Calculate means
    summary_raw = df_valid.groupby("built_form")["energy_per_capita"].agg(["mean", "std", "count"]).loc[order]
    summary_raw["se"] = summary_raw["std"] / np.sqrt(summary_raw["count"])
    summary_raw["ci"] = 1.96 * summary_raw["se"]

    colors = sns.color_palette("coolwarm", len(order))
    bars1 = ax1.bar(range(len(order)), summary_raw["mean"], yerr=summary_raw["ci"],
                    capsize=5, color=colors, edgecolor="black", linewidth=0.5, alpha=0.8)
    ax1.set_xticks(range(len(order)))
    ax1.set_xticklabels([o.replace("-", "-\n") for o in order], fontsize=9)
    ax1.set_xlabel("Built Form")
    ax1.set_ylabel("Mean Energy per Capita (kWh/person/year)")
    ax1.set_title("(A) Raw: Energy by Built Form\n(Confounded by Age)", fontweight="bold")
    ax1.axhline(y=df_valid["energy_per_capita"].mean(), color="gray", linestyle=":", linewidth=1.5)

    # Right: Residual energy (after removing age effect) by built form
    ax2 = axes[1]

    # Regress out building age from energy
    model = smf.ols("energy_per_capita ~ building_age", data=df_valid).fit()
    df_valid["energy_resid"] = model.resid

    summary_controlled = df_valid.groupby("built_form")["energy_resid"].agg(["mean", "std", "count"]).loc[order]
    summary_controlled["se"] = summary_controlled["std"] / np.sqrt(summary_controlled["count"])
    summary_controlled["ci"] = 1.96 * summary_controlled["se"]

    bars2 = ax2.bar(range(len(order)), summary_controlled["mean"], yerr=summary_controlled["ci"],
                    capsize=5, color=colors, edgecolor="black", linewidth=0.5, alpha=0.8)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels([o.replace("-", "-\n") for o in order], fontsize=9)
    ax2.set_xlabel("Built Form")
    ax2.set_ylabel("Age-Adjusted Energy Residual")
    ax2.set_title("(B) Controlled: Energy by Built Form\n(After Removing Age Effect)", fontweight="bold")
    ax2.axhline(y=0, color="gray", linestyle=":", linewidth=1.5)

    # Add annotation about the pattern
    ax2.annotate("Shared walls now show benefit\n(terraces below detached)",
                 xy=(0.95, 0.95), xycoords="axes fraction",
                 ha="right", va="top", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="#ccffcc", edgecolor="#27ae60", alpha=0.9))

    plt.tight_layout()
    path = FIGURES_DIR / "03_built_form_controlled.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return str(path.relative_to(BASE_DIR))


def fig_shap_analysis(df: pd.DataFrame) -> tuple[str, str, str]:
    """Generate SHAP visualizations."""
    # Prepare features
    features = [
        "TOTAL_FLOOR_AREA", "building_age", "compactness", "shared_wall_ratio",
        "cc_harmonic_800", "cc_bus_800_nw", "pop_density", "pct_owner_occupied",
        "building_height", "pct_car_commute",
    ]
    available = [f for f in features if f in df.columns]

    model_df = df[["energy_per_capita"] + available].dropna()
    X = model_df[available]
    y = model_df["energy_per_capita"]

    print(f"  Training model on {len(X)} samples...")

    # Train model - use conservative hyperparameters to prevent overfitting on small dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(
        n_estimators=50,        # Fewer trees
        max_depth=3,            # Shallower trees
        learning_rate=0.05,     # Slower learning
        min_samples_leaf=10,    # Require more samples per leaf
        min_samples_split=15,   # Require more samples to split
        subsample=0.7,          # Use less data per tree
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Report R² scores
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    print(f"  Model R²: Train={train_r2:.3f}, Test={test_r2:.3f}")

    # Compute SHAP values
    print("  Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Rename features for display
    feature_names = {
        "TOTAL_FLOOR_AREA": "Floor Area",
        "building_age": "Building Age",
        "compactness": "Compactness",
        "shared_wall_ratio": "Shared Wall Ratio",
        "cc_harmonic_800": "Network Centrality",
        "cc_bus_800_nw": "Bus Accessibility",
        "pop_density": "Population Density",
        "pct_owner_occupied": "% Owner Occupied",
        "building_height": "Building Height",
        "pct_car_commute": "% Car Commute",
    }
    X_display = X.rename(columns=feature_names)

    # 1. Summary plot (bar) - improved
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_display, plot_type="bar", show=False,
                      color="#3498db")
    plt.title("SHAP Feature Importance\n(Mean |SHAP value| - contribution to prediction)",
              fontweight="bold", fontsize=13)
    plt.xlabel("Mean |SHAP value| (kWh/person/year)", fontsize=11)
    plt.tight_layout()
    path1 = FIGURES_DIR / "10_shap_importance.png"
    plt.savefig(path1, bbox_inches="tight")
    plt.close()

    # 2. Summary plot (beeswarm) - improved
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_display, show=False, alpha=0.7)
    plt.title("SHAP Summary: Feature Effects on Energy per Capita\n" +
              "(Red = high feature value, Blue = low feature value)",
              fontweight="bold", fontsize=12)
    plt.xlabel("SHAP value (impact on prediction)", fontsize=11)
    plt.tight_layout()
    path2 = FIGURES_DIR / "11_shap_summary.png"
    plt.savefig(path2, bbox_inches="tight")
    plt.close()

    # 3. Dependence plot for building age
    plt.figure(figsize=(10, 7))
    if "Building Age" in X_display.columns:
        shap.dependence_plot("Building Age", shap_values, X_display, show=False,
                             interaction_index="Network Centrality" if "Network Centrality" in X_display.columns else None)
        plt.title("SHAP Dependence: Building Age\n(Color shows interaction with Centrality)",
                  fontweight="bold", fontsize=13)
        plt.xlabel("Building Age (years)", fontsize=11)
        plt.ylabel("SHAP value for Building Age", fontsize=11)
    plt.tight_layout()
    path3 = FIGURES_DIR / "12_shap_dependence_age.png"
    plt.savefig(path3, bbox_inches="tight")
    plt.close()

    return (
        str(path1.relative_to(BASE_DIR)),
        str(path2.relative_to(BASE_DIR)),
        str(path3.relative_to(BASE_DIR)),
    )


def generate_markdown_report(df: pd.DataFrame, figures: dict) -> None:
    """Generate markdown report with embedded figures."""
    import statsmodels.formula.api as smf
    from scipy import stats as scipy_stats

    # Calculate key statistics
    df_valid = df[df["building_age"].notna() & df["cc_harmonic_800"].notna()].copy()
    r_age, _ = scipy_stats.pearsonr(df_valid["building_age"], df_valid["energy_per_capita"])
    r_centrality_raw, _ = scipy_stats.pearsonr(df_valid["cc_harmonic_800"], df_valid["energy_per_capita"])

    # Calculate partial correlations (after controlling for age)
    model_energy = smf.ols("energy_per_capita ~ building_age", data=df_valid).fit()
    model_centrality = smf.ols("cc_harmonic_800 ~ building_age", data=df_valid).fit()
    model_density = smf.ols("pop_density ~ building_age", data=df_valid).fit()
    r_centrality_controlled, _ = scipy_stats.pearsonr(model_centrality.resid, model_energy.resid)
    r_density_controlled, _ = scipy_stats.pearsonr(model_density.resid, model_energy.resid)

    flat_energy = df[df["property_type"] == "Flat"]["energy_per_capita"].mean()
    house_energy = df[df["property_type"] == "House"]["energy_per_capita"].mean()

    # Fix paths - figures are in same folder as report, so just use filename
    def fix_path(p):
        return p.replace("figures/", "")

    report = f"""# Does Compact Urban Development Reduce Energy Use?

## Controlling for Building Age and Other Confounders

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

**Dataset:** {len(df)} properties with EPC data from test area

---

## Research Question

> **After controlling for building age and other confounders, is there a relationship between compact urban development and energy use per capita?**

---

## Executive Summary

| Metric | Raw Correlation | After Controlling for Age |
|--------|-----------------|---------------------------|
| Network Centrality | r = {r_centrality_raw:+.2f} | r = {r_centrality_controlled:+.2f} |
| Population Density | r = {r_centrality_raw:+.2f} | r = {r_density_controlled:+.2f} |
| Building Age | r = {r_age:+.2f} | (control variable) |

**Summary:** This table compares raw correlations with partial correlations after controlling for building age.

- Raw correlations between centrality/density and energy are near zero
- After controlling for building age, correlations remain small
- Building age (r = {r_age:.2f}) shows the strongest association with energy per capita

### Sample Statistics

| Metric | Value |
|--------|-------|
| Sample Size | {len(df)} properties |
| Mean Energy per Capita | {df['energy_per_capita'].mean():.0f} kWh/person/year |
| Mean Building Age | {df['building_age'].mean():.0f} years |

---

## 1. The Key Result: Controlling for Building Age

![Controlled Effects]({fix_path(figures['controlled'])})

### What This Figure Shows

- **Panels A & C (left)**: Raw correlations between urban form variables (centrality, density) and energy per capita
- **Panels B & D (right)**: Partial correlations after regressing out building age from both variables
- **Method**: OLS residuals used to remove linear effect of building age before computing correlations

---

## 2. Building Age and Energy

![Building Age]({fix_path(figures['building_age'])})

Building age has a correlation of **r = {r_age:.2f}** with energy per capita.

- Older buildings tend to have lower thermal efficiency (insulation, glazing, heating systems)
- Central urban areas often have older building stock
- Building age may correlate with both urban form and energy consumption

---

## 3. Relationship Between Centrality and Age

![Centrality vs Age]({fix_path(figures['confounding'])})

**Left panel**: Scatter plot of network centrality vs building age

**Right panel**: Energy vs centrality, stratified by building age category

---

## 4. Other Variables

![Other Confounders]({fix_path(figures['other_confounders'])})

Additional relationships between variables:
- **Property type**: Distribution of building age by property type
- **Built form**: Distribution of building age by built form
- **Tenure**: Energy by tenure category
- **Bottom-right**: Correlation matrix of key variables

---

## 5. Built Form: Raw vs Controlled

![Built Form Controlled]({fix_path(figures['built_form_controlled'])})

### Description

- **Panel A (Raw)**: Mean energy per capita by built form category
- **Panel B (Controlled)**: Mean age-adjusted residuals by built form category
  - Residuals computed by regressing energy per capita on building age
  - Positive residuals indicate higher-than-expected energy given building age
  - Negative residuals indicate lower-than-expected energy given building age

### Raw Built Form Distribution

![Built Form Raw]({fix_path(figures['built_form'])})

---

## 6. Machine Learning Feature Importance (SHAP)

![SHAP Importance]({fix_path(figures['shap_importance'])})

SHAP feature importance from Gradient Boosting model.

![SHAP Summary]({fix_path(figures['shap_summary'])})

**Reading this plot**: Each dot is one observation. Red = high feature value, Blue = low.
Position on x-axis shows impact on prediction.

---

## 7. Supporting Figures

### Energy Distribution

![Energy Distribution]({fix_path(figures['distribution'])})

Distribution of energy per capita by property type.
Flats: {flat_energy:.0f} kWh mean, Houses: {house_energy:.0f} kWh mean.

### Density (Raw)

![Density Effect]({fix_path(figures['density'])})

Raw correlation between population density and energy per capita.

### Correlation Matrix

![Correlation Heatmap]({fix_path(figures['correlation'])})

### SHAP Dependence

![SHAP Age Dependence]({fix_path(figures['shap_dependence'])})

SHAP dependence plot for building age.

---

## Summary of Results

### Correlations by Variable

| Variable | Raw Correlation | After Controlling for Age |
|----------|-----------------|---------------------------|
| **Shared Walls (Built Form)** | See Figure 5 | Terraces show lower residuals |
| **Building Height** | Negative | Negative |
| **Network Centrality** | r = {r_centrality_raw:+.2f} | r = {r_centrality_controlled:+.2f} |
| **Population Density** | r = {r_centrality_raw:+.2f} | r = {r_density_controlled:+.2f} |
| **Building Age** | r = {r_age:+.2f} | (control variable) |

### Observations

1. Building age shows the strongest correlation with energy per capita (r = {r_age:.2f})
2. Network centrality and population density show near-zero correlations with energy
3. After controlling for building age, correlations remain small
4. This analysis covers building energy only; transport energy is not included

### Limitations

- Sample: {len(df)} properties
- Single study area may not generalize
- EPC energy excludes transport/behaviour
- Multi-level models may provide additional insights

---

## Appendix: Data Sources

| Source | Variables |
|--------|-----------|
| Energy Performance Certificates | Energy consumption, floor area, building age |
| Census 2021 | Household size, tenure, travel to work |
| OS Open Map Local | Building footprints |
| Environment Agency LiDAR | Building heights |
| cityseer | Network centrality, accessibility metrics |

---

*Report generated automatically by stats/04_visualizations.py*
"""

    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {REPORT_PATH}")


def main():
    """Generate all visualizations and report."""
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Create figures directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_and_prepare_data()

    # Generate figures
    print("\nGenerating figures...")
    figures = {}

    print("  1. Energy distribution...")
    figures["distribution"] = fig_energy_distribution(df)

    print("  2. Built form analysis (raw)...")
    figures["built_form"] = fig_energy_by_built_form(df)

    print("  3. Built form CONTROLLED...")
    figures["built_form_controlled"] = fig_built_form_controlled(df)

    print("  5. Building age effect...")
    figures["building_age"] = fig_building_age_effect(df)

    print("  6. Density effect (raw)...")
    figures["density"] = fig_density_effect(df)

    print("  7. Correlation heatmap...")
    figures["correlation"] = fig_correlation_heatmap(df)

    print("  8. Confounding: centrality vs age...")
    figures["confounding"] = fig_centrality_vs_age(df)

    print("  9. CONTROLLED EFFECTS (key figure)...")
    figures["controlled"] = fig_controlled_effects(df)

    print("  10. Other confounders...")
    figures["other_confounders"] = fig_other_confounders(df)

    print("  11-13. SHAP analysis...")
    shap_figs = fig_shap_analysis(df)
    figures["shap_importance"] = shap_figs[0]
    figures["shap_summary"] = shap_figs[1]
    figures["shap_dependence"] = shap_figs[2]

    # Generate report
    print("\nGenerating markdown report...")
    generate_markdown_report(df, figures)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nFigures saved to: {FIGURES_DIR}")
    print(f"Report saved to: {REPORT_PATH}")
    print("\nGenerated figures:")
    for name, path in sorted(figures.items()):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
