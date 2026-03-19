"""
Empirical Access Penalty Model.

Instead of assuming trip rates and decay functions, this model estimates the
access energy penalty from the observed relationship between local service
coverage and Census-reported transport behaviour.

Approach:
    1. Fit: transport_energy ~ local_coverage + controls + city_FE
    2. For each OA, predict transport energy at actual coverage
    3. Predict transport energy at reference coverage (85th percentile = compact)
    4. Access penalty = predicted(actual) - predicted(reference)

This grounds the penalty in observed data (Census commute mode/distance,
car ownership) rather than assumed trip rates and decay parameters.

Usage:
    uv run python stats/access_penalty_model.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from proof_of_concept_oa import load_and_aggregate, build_accessibility
from nepi import compute_nepi, FIGURE_DIR as NEPI_FIGURE_DIR

FIGURE_DIR = Path(__file__).parent / "figures" / "nepi"


def build_coverage(lsoa: pd.DataFrame) -> pd.DataFrame:
    """
    Compute local service coverage from cityseer nearest-distance columns.

    Parameters
    ----------
    lsoa : pd.DataFrame
        OA data with cc_*_nearest_max_4800 columns.

    Returns
    -------
    pd.DataFrame
        Input augmented with per-service coverage columns and local_coverage.
    """
    services: dict[str, tuple[str, int]] = {
        "cc_fsa_restaurant_nearest_max_4800": ("food_restaurant", 800),
        "cc_fsa_takeaway_nearest_max_4800": ("food_takeaway", 800),
        "cc_fsa_pub_nearest_max_4800": ("food_pub", 800),
        "cc_gp_practice_nearest_max_4800": ("gp_practice", 1200),
        "cc_pharmacy_nearest_max_4800": ("pharmacy", 1000),
        "cc_school_nearest_max_4800": ("school", 1200),
        "cc_greenspace_nearest_max_4800": ("greenspace", 1000),
        "cc_bus_nearest_max_4800": ("bus_stop", 800),
        "cc_hospital_nearest_max_4800": ("hospital", 2000),
    }

    coverage_cols: list[str] = []
    for col, (name, threshold) in services.items():
        if col not in lsoa.columns:
            continue
        vals = pd.to_numeric(lsoa[col], errors="coerce")
        score = np.exp(-np.log(2) * (vals / threshold) ** 2)
        score = score.fillna(0)
        cov_col = f"cov_{name}"
        lsoa[cov_col] = score
        coverage_cols.append(cov_col)

    if coverage_cols:
        lsoa["local_coverage"] = lsoa[coverage_cols].mean(axis=1)

    return lsoa


def fit_penalty_models(lsoa: pd.DataFrame) -> dict:
    """
    Fit models linking local coverage to observed transport behaviour.

    Four DVs, all from Census:
        1. car_commute_share (TS061)
        2. active_share (TS061: walk + cycle)
        3. cars_per_hh (TS045)
        4. transport_kwh_per_hh_total_est (derived from TS058 × TS061)

    Controls: log population density, household size, deprivation,
    building age (where available), IMD income domain (where available).

    Returns
    -------
    dict
        Mapping DV name -> fitted statsmodels result.
    """
    print("=" * 70)
    print("EMPIRICAL ACCESS PENALTY MODEL")
    print("=" * 70)

    # Prepare controls
    lsoa["log_people_per_ha"] = np.log(lsoa["people_per_ha"].clip(lower=0.1))

    controls = ["log_people_per_ha", "avg_hh_size", "pct_not_deprived"]
    if "median_build_year" in lsoa.columns:
        valid_yr = lsoa["median_build_year"].notna()
        if valid_yr.sum() > len(lsoa) * 0.5:
            controls.append("median_build_year")

    # IMD income if available
    imd_cols = [
        c
        for c in lsoa.columns
        if "imd_income" in c.lower() and "score" in c.lower()
    ]
    if imd_cols:
        controls.append(imd_cols[0])

    # Skip city FE in penalty model to keep memory footprint manageable.
    # City FE with 2,844 BUA dummies creates a ~175k × 2,850 design matrix.
    # The penalty model is illustrative; formal inference uses the main regressions.
    x_cols = ["local_coverage"] + controls
    print(f"\n  Controls: {controls}")
    print(f"  City FE: none (penalty model uses controls only)")

    dvs = {
        "car_commute_share": "Car commute share",
        "active_share": "Walk+cycle share",
        "cars_per_hh": "Cars per household",
        "transport_kwh_per_hh_total_est": "Transport kWh/hh (overall est.)",
    }

    models: dict = {}
    print(
        f"\n  {'DV':<35s} {'β(coverage)':>12s} {'SE':>8s} "
        f"{'t':>8s} {'R²':>6s} {'N':>8s}"
    )
    print(f"  {'-' * 82}")

    for dv, label in dvs.items():
        if dv not in lsoa.columns:
            continue
        cols = [dv] + x_cols
        sub = lsoa[cols].dropna()
        if len(sub) < 1000:
            continue

        y = sub[dv]
        X = sm.add_constant(sub[x_cols])

        m = sm.OLS(y, X).fit(cov_type="HC1")

        models[dv] = m

        beta = m.params["local_coverage"]
        se = m.bse["local_coverage"]
        t = m.tvalues["local_coverage"]
        r2 = m.rsquared

        print(
            f"  {label:<35s} {beta:>12.4f} {se:>8.4f} "
            f"{t:>8.1f} {r2:>6.3f} {int(m.nobs):>8,}"
        )

    return models


def compute_empirical_penalty(
    lsoa: pd.DataFrame,
    models: dict,
    reference_coverage: float = 0.85,
) -> pd.DataFrame:
    """
    Compute the empirical access penalty for each OA.

    The penalty is the difference between predicted transport energy at the
    OA's actual coverage and at the reference coverage level (default: 85%,
    the median of flat-dominant OAs).

    Parameters
    ----------
    lsoa : pd.DataFrame
        OA data with local_coverage and all control variables.
    models : dict
        Fitted models from fit_penalty_models().
    reference_coverage : float
        Coverage level for the counterfactual (default 0.85).

    Returns
    -------
    pd.DataFrame
        Input augmented with empirical_penalty_kwh columns.
    """
    print(f"\n  Computing counterfactual penalties (reference coverage = {reference_coverage:.0%})")

    # Use the transport energy model
    transport_model = models.get("transport_kwh_per_hh_total_est")
    car_model = models.get("cars_per_hh")

    if transport_model is None:
        print("  No transport energy model available")
        return lsoa

    # Get the exogenous variable names and prepare data
    x_names = transport_model.model.exog_names
    sub_idx = lsoa.dropna(
        subset=[n for n in x_names if n != "const" and n in lsoa.columns]
    ).index

    # Predict at actual coverage
    X_actual = sm.add_constant(lsoa.loc[sub_idx, [n for n in x_names if n != "const"]])
    pred_actual = transport_model.predict(X_actual)

    # Predict at reference coverage
    X_ref = X_actual.copy()
    X_ref["local_coverage"] = reference_coverage
    pred_ref = transport_model.predict(X_ref)

    # Penalty = actual - reference (positive = OA uses more energy than compact reference)
    lsoa.loc[sub_idx, "empirical_penalty_kwh"] = pred_actual - pred_ref

    # Same for cars
    if car_model is not None:
        x_names_car = car_model.model.exog_names
        X_actual_car = sm.add_constant(
            lsoa.loc[sub_idx, [n for n in x_names_car if n != "const"]]
        )
        pred_cars_actual = car_model.predict(X_actual_car)
        X_ref_car = X_actual_car.copy()
        X_ref_car["local_coverage"] = reference_coverage
        pred_cars_ref = car_model.predict(X_ref_car)
        lsoa.loc[sub_idx, "empirical_excess_cars"] = pred_cars_actual - pred_cars_ref

    # Summary by type
    types = ["Flat", "Terraced", "Semi", "Detached"]
    valid = lsoa["empirical_penalty_kwh"].notna()

    print(
        f"\n  {'Type':<12s} {'Coverage':>9s} {'Pred transport':>15s} "
        f"{'At 85% ref':>11s} {'Penalty':>10s} {'Excess cars':>12s}"
    )
    print(f"  {'-' * 74}")

    for t in types:
        mask = (lsoa["dominant_type"] == t) & valid
        sub = lsoa[mask]
        if len(sub) == 0:
            continue

        cov = sub["local_coverage"].median()
        pred = pred_actual[mask].median() if hasattr(pred_actual, "loc") else np.nan
        ref = pred_ref[mask].median() if hasattr(pred_ref, "loc") else np.nan
        pen = sub["empirical_penalty_kwh"].median()
        excess = sub["empirical_excess_cars"].median() if "empirical_excess_cars" in sub.columns else np.nan

        print(
            f"  {t:<12s} {cov:>8.1%} {pred:>15,.0f} "
            f"{ref:>11,.0f} {pen:>+10,.0f} {excess:>+11.2f}"
        )

    flat_pen = lsoa.loc[(lsoa["dominant_type"] == "Flat") & valid, "empirical_penalty_kwh"].median()
    det_pen = lsoa.loc[(lsoa["dominant_type"] == "Detached") & valid, "empirical_penalty_kwh"].median()
    if flat_pen is not None and det_pen is not None:
        print(f"\n  Detached penalty: {det_pen:+,.0f} kWh/hh vs {flat_pen:+,.0f} for flats")
        print(f"  Gap: {det_pen - flat_pen:,.0f} kWh/hh")

    return lsoa


def fig_penalty_by_type(lsoa: pd.DataFrame) -> None:
    """Bar chart showing empirical access penalty by housing type."""
    types = ["Flat", "Terraced", "Semi", "Detached"]
    type_colors = {
        "Flat": "#2196F3",
        "Terraced": "#FF9800",
        "Semi": "#4CAF50",
        "Detached": "#E91E63",
    }

    penalties = []
    coverages = []
    for t in types:
        sub = lsoa[(lsoa["dominant_type"] == t) & lsoa["empirical_penalty_kwh"].notna()]
        penalties.append(sub["empirical_penalty_kwh"].median())
        coverages.append(sub["local_coverage"].median() * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Coverage
    bars = ax1.bar(types, coverages, color=[type_colors[t] for t in types],
                   edgecolor="white")
    ax1.set_ylabel("Local service coverage (%)", fontsize=10)
    ax1.set_title("A. Walkable service coverage", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 100)
    for bar, val in zip(bars, coverages):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=10)

    # Panel B: Empirical penalty
    bars = ax2.bar(types, penalties, color=[type_colors[t] for t in types],
                   edgecolor="white")
    ax2.set_ylabel("Access energy penalty (kWh/hh/yr)", fontsize=10)
    ax2.set_title("B. Empirical access penalty\n(vs 85% coverage reference)",
                  fontsize=12, fontweight="bold")
    ax2.axhline(y=0, color="#999", linewidth=0.8, linestyle="--")
    for bar, val in zip(bars, penalties):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (50 if val > 0 else -150),
                 f"{val:+,.0f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig_empirical_penalty.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved fig_empirical_penalty.png")


def fig_coverage_vs_transport(lsoa: pd.DataFrame) -> None:
    """Scatter/hex of coverage vs observed transport energy."""
    import seaborn as sns

    valid = (
        lsoa["local_coverage"].notna()
        & lsoa["transport_kwh_per_hh_total_est"].notna()
        & (lsoa["transport_kwh_per_hh_total_est"] > 0)
        & (lsoa["transport_kwh_per_hh_total_est"] < 30000)
    )
    sub = lsoa[valid].copy()

    fig, ax = plt.subplots(figsize=(8, 6))

    type_colors = {
        "Flat": "#2196F3",
        "Terraced": "#FF9800",
        "Semi": "#4CAF50",
        "Detached": "#E91E63",
    }

    for t in ["Detached", "Semi", "Terraced", "Flat"]:
        tsub = sub[sub["dominant_type"] == t]
        ax.scatter(
            tsub["local_coverage"] * 100,
            tsub["transport_kwh_per_hh_total_est"],
            s=1, alpha=0.05, color=type_colors[t], label=t,
        )
        # Add median crosshair
        mx = tsub["local_coverage"].median() * 100
        my = tsub["transport_kwh_per_hh_total_est"].median()
        ax.plot(mx, my, "o", color=type_colors[t], markersize=10,
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)

    ax.set_xlabel("Local service coverage (%)", fontsize=11)
    ax.set_ylabel("Transport energy (kWh/hh/yr, overall est.)", fontsize=11)
    ax.set_title("Local coverage predicts observed transport energy",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, markerscale=10)
    ax.set_xlim(0, 100)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig_coverage_vs_transport.png", bbox_inches="tight",
                dpi=200)
    plt.close()
    print(f"  Saved fig_coverage_vs_transport.png")


def main() -> None:
    """Run the empirical access penalty model."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    lsoa = load_and_aggregate()
    lsoa = build_accessibility(lsoa)
    lsoa = build_coverage(lsoa)

    # Fit models
    models = fit_penalty_models(lsoa)

    # Compute penalties
    lsoa = compute_empirical_penalty(lsoa, models)

    # Figures
    fig_penalty_by_type(lsoa)
    fig_coverage_vs_transport(lsoa)

    print(f"\n  All outputs saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
