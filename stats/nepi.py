"""
Neighbourhood Energy Performance Index (NEPI).

Scores each Output Area on three surfaces of energy performance, all
expressed in kWh/hh/yr:

    1. Form     — metered building energy (DESNZ gas + electricity)
    2. Mobility — estimated transport energy (Census commute × NTS scalar)
    3. Access   — energy penalty for poor walkable service coverage,
                  estimated from the empirical relationship between local
                  coverage and observed Census transport behaviour

The composite NEPI is the sum of all three surfaces (total neighbourhood
energy cost in kWh/hh/yr), banded A–G by national percentile position.
Using a common unit (kWh) eliminates the need for arbitrary surface
weighting: the surfaces weight themselves by their energy magnitude.

Usage:
    uv run python stats/nepi.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from proof_of_concept_oa import load_and_aggregate, build_accessibility

FIGURE_DIR = Path(__file__).parent / "figures" / "nepi"

# EPC-style bands assigned by percentile of composite kWh
# A = lowest energy (best), G = highest (worst)
# Thresholds are percentile boundaries applied to the national distribution
BAND_PERCENTILES = {
    "A": (0, 8),
    "B": (8, 19),
    "C": (19, 36),
    "D": (36, 60),
    "E": (60, 81),
    "F": (81, 95),
    "G": (95, 100),
}

BAND_COLORS = {
    "A": "#00845A",
    "B": "#2C9F29",
    "C": "#8CBF26",
    "D": "#FCD800",
    "E": "#F0AB00",
    "F": "#ED6F21",
    "G": "#E3242B",
}

# Service types and walking-distance thresholds (metres) for access surface
SERVICE_THRESHOLDS: dict[str, tuple[str, int]] = {
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


def compute_local_coverage(lsoa: pd.DataFrame) -> pd.DataFrame:
    """
    Compute local service coverage from cityseer nearest-distance columns.

    For each of 9 service types, coverage is computed as a Gaussian decay
    of network distance: score = exp(-ln(2) * (d / d_half)^2), where d_half
    is the service-specific walking threshold. NaN (no destination within
    4,800m) maps to zero coverage.

    Parameters
    ----------
    lsoa : pd.DataFrame
        OA data with cc_*_nearest_max_4800 columns.

    Returns
    -------
    pd.DataFrame
        Input augmented with per-service coverage columns and local_coverage.
    """
    coverage_cols: list[str] = []
    for col, (name, threshold) in SERVICE_THRESHOLDS.items():
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


def compute_access_penalty(lsoa: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the empirical access energy penalty for each OA.

    Fits an OLS model: transport_energy ~ local_coverage + controls,
    then computes the penalty as the difference between each OA's predicted
    transport energy at its actual coverage and at the compact reference
    level (85th percentile of coverage, ≈ flat-dominant median).

    Parameters
    ----------
    lsoa : pd.DataFrame
        OA data with local_coverage, transport energy, and control variables.

    Returns
    -------
    pd.DataFrame
        Input augmented with access_penalty_kwh column.
    """
    transport_col = "transport_kwh_per_hh_total_est"
    if transport_col not in lsoa.columns:
        lsoa["access_penalty_kwh"] = 0.0
        return lsoa

    lsoa["log_people_per_ha"] = np.log(lsoa["people_per_ha"].clip(lower=0.1))

    controls = ["log_people_per_ha", "avg_hh_size", "pct_not_deprived"]
    if "median_build_year" in lsoa.columns:
        if lsoa["median_build_year"].notna().sum() > len(lsoa) * 0.5:
            controls.append("median_build_year")
    imd_cols = [
        c for c in lsoa.columns
        if "imd_income" in c.lower() and "score" in c.lower()
    ]
    if imd_cols:
        controls.append(imd_cols[0])

    x_cols = ["local_coverage"] + controls
    cols_needed = [transport_col] + x_cols
    sub = lsoa[cols_needed].dropna()

    y = sub[transport_col]
    X = sm.add_constant(sub[x_cols])
    model = sm.OLS(y, X).fit(cov_type="HC1")

    # Reference coverage: 85% (median of flat-dominant OAs)
    ref_coverage = 0.85

    # Predict at actual coverage
    pred_actual = model.predict(X)

    # Predict at reference coverage
    X_ref = X.copy()
    X_ref["local_coverage"] = ref_coverage
    pred_ref = model.predict(X_ref)

    # Penalty = actual - reference (positive = worse than compact reference)
    penalty = pred_actual - pred_ref
    # Clip: minimum penalty is 0 (don't reward OAs that already exceed reference)
    penalty = penalty.clip(lower=0)

    lsoa["access_penalty_kwh"] = np.nan
    lsoa.loc[sub.index, "access_penalty_kwh"] = penalty

    # Print model summary
    beta = model.params["local_coverage"]
    t = model.tvalues["local_coverage"]
    r2 = model.rsquared
    print(f"\n  Access penalty model: β(coverage)={beta:.0f}, t={t:.1f}, R²={r2:.3f}, N={int(model.nobs):,}")

    return lsoa


def compute_nepi(lsoa: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Neighbourhood Energy Performance Index.

    All three surfaces are in kWh/hh/yr. The composite is their sum.
    Bands are assigned by national percentile position of the composite.

    Parameters
    ----------
    lsoa : pd.DataFrame
        OA-level data with building_kwh_per_hh, transport_kwh_per_hh_total_est,
        and cityseer nearest-distance columns.

    Returns
    -------
    pd.DataFrame
        Input augmented with NEPI columns.
    """
    print("=" * 70)
    print("NEIGHBOURHOOD ENERGY PERFORMANCE INDEX (NEPI)")
    print("=" * 70)

    # --- Compute local coverage ---
    lsoa = compute_local_coverage(lsoa)

    # Print coverage summary
    n_services = sum(1 for c in SERVICE_THRESHOLDS if c in lsoa.columns)
    print(f"\n  Access surface: {n_services} services, Gaussian decay")
    print(f"  Median local coverage: {lsoa['local_coverage'].median():.1%}")

    # --- Compute empirical access penalty ---
    lsoa = compute_access_penalty(lsoa)

    # --- Surface 1: Form (building kWh/hh) ---
    lsoa["nepi_form_kwh"] = lsoa["building_kwh_per_hh"]

    # --- Surface 2: Mobility (transport kWh/hh) ---
    transport_col = "transport_kwh_per_hh_total_est"
    if transport_col not in lsoa.columns:
        transport_col = "transport_kwh_per_hh_est"
    lsoa["nepi_mobility_kwh"] = lsoa[transport_col]

    # --- Surface 3: Access (penalty kWh/hh) ---
    lsoa["nepi_access_kwh"] = lsoa["access_penalty_kwh"].fillna(0)

    # --- Composite: sum of three surfaces ---
    lsoa["nepi_total_kwh"] = (
        lsoa["nepi_form_kwh"].fillna(0)
        + lsoa["nepi_mobility_kwh"].fillna(0)
        + lsoa["nepi_access_kwh"]
    )

    # --- Band assignment by percentile of composite kWh ---
    # Lower kWh = better = Band A
    valid = lsoa["nepi_total_kwh"].notna() & (lsoa["nepi_total_kwh"] > 0)
    pct_rank = lsoa.loc[valid, "nepi_total_kwh"].rank(pct=True) * 100

    lsoa["nepi_band"] = ""
    for band, (lo, hi) in BAND_PERCENTILES.items():
        mask = valid & (pct_rank > lo) & (pct_rank <= hi)
        lsoa.loc[mask.index[mask], "nepi_band"] = band
    # Bottom edge: assign band A to the lowest percentile
    mask_bottom = valid & (pct_rank <= BAND_PERCENTILES["A"][1])
    lsoa.loc[mask_bottom.index[mask_bottom], "nepi_band"] = "A"

    # --- Summary by type ---
    types = ["Flat", "Terraced", "Semi", "Detached"]
    print(
        f"\n  {'Type':<12s} {'Form':>8s} {'Mobility':>10s} {'Access':>8s} "
        f"{'Total':>8s} {'Band':>5s}"
    )
    print(f"  {'-' * 56}")
    for t in types:
        sub = lsoa[(lsoa["dominant_type"] == t) & valid]
        if len(sub) == 0:
            continue
        f = sub["nepi_form_kwh"].median()
        m = sub["nepi_mobility_kwh"].median()
        a = sub["nepi_access_kwh"].median()
        tot = sub["nepi_total_kwh"].median()
        # Modal band
        band = sub["nepi_band"].mode().iloc[0] if len(sub) > 0 else "?"
        print(
            f"  {t:<12s} {f:>8,.0f} {m:>10,.0f} {a:>8,.0f} "
            f"{tot:>8,.0f} {band:>5s}"
        )

    # Band distribution
    n_valid = valid.sum()
    print(f"\n  Band distribution (N={n_valid:,}):")
    for band in ["A", "B", "C", "D", "E", "F", "G"]:
        n = (lsoa["nepi_band"] == band).sum()
        pct = n / n_valid * 100 if n_valid > 0 else 0
        print(f"    {band}: {n:>7,d} ({pct:>5.1f}%)")

    # Band by type
    print(f"\n  Band distribution by dominant type:")
    print(
        f"  {'Type':<12s} {'A':>6s} {'B':>6s} {'C':>6s} {'D':>6s} "
        f"{'E':>6s} {'F':>6s} {'G':>6s}"
    )
    print(f"  {'-' * 54}")
    for t in types:
        sub = lsoa[lsoa["dominant_type"] == t]
        n_t = len(sub)
        parts = []
        for band in ["A", "B", "C", "D", "E", "F", "G"]:
            n = (sub["nepi_band"] == band).sum()
            pct = n / n_t * 100 if n_t > 0 else 0
            parts.append(f"{pct:>5.1f}%")
        print(f"  {t:<12s} {' '.join(parts)}")

    # Coverage by type for reporting
    print(f"\n  Local coverage by type:")
    for t in types:
        sub = lsoa[lsoa["dominant_type"] == t]
        print(f"    {t:<12s}: {sub['local_coverage'].median():.1%}")

    return lsoa


def fig_nepi_scorecard(lsoa: pd.DataFrame) -> None:
    """Stacked bar showing three kWh surfaces by housing type."""
    types = ["Flat", "Terraced", "Semi", "Detached"]

    form_vals = []
    mob_vals = []
    acc_vals = []
    for t in types:
        sub = lsoa[lsoa["dominant_type"] == t]
        form_vals.append(sub["nepi_form_kwh"].median())
        mob_vals.append(sub["nepi_mobility_kwh"].median())
        acc_vals.append(sub["nepi_access_kwh"].median())

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(types))

    b1 = ax.bar(x, form_vals, label="Form (building)", color="#5C6BC0",
                edgecolor="white")
    b2 = ax.bar(x, mob_vals, bottom=form_vals, label="Mobility (transport)",
                color="#26A69A", edgecolor="white")
    bottoms = [f + m for f, m in zip(form_vals, mob_vals)]
    b3 = ax.bar(x, acc_vals, bottom=bottoms, label="Access (penalty)",
                color="#EF5350", edgecolor="white")

    # Annotate totals
    for i, t in enumerate(types):
        total = form_vals[i] + mob_vals[i] + acc_vals[i]
        ax.text(i, total + 200, f"{total:,.0f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(types, fontsize=11)
    ax.set_ylabel("kWh per household per year", fontsize=11)
    ax.set_title("NEPI: Three Surfaces of Neighbourhood Energy Cost",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig_nepi_scorecard.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved fig_nepi_scorecard.png")


def fig_nepi_band_distribution(lsoa: pd.DataFrame) -> None:
    """Stacked bar chart showing NEPI band distribution by housing type."""
    types = ["Flat", "Terraced", "Semi", "Detached"]
    bands = ["A", "B", "C", "D", "E", "F", "G"]

    data = {}
    for t in types:
        sub = lsoa[lsoa["dominant_type"] == t]
        total = len(sub)
        data[t] = [
            (sub["nepi_band"] == b).sum() / total * 100 if total > 0 else 0
            for b in bands
        ]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(types))
    bottom = np.zeros(len(types))

    for i, band in enumerate(bands):
        vals = [data[t][i] for t in types]
        ax.bar(x, vals, bottom=bottom, label=f"Band {band}",
               color=BAND_COLORS[band], edgecolor="white", linewidth=0.5)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(types, fontsize=11)
    ax.set_ylabel("Share of OAs (%)", fontsize=10)
    ax.set_title("NEPI Band Distribution by Dominant Housing Type",
                 fontsize=12, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig_nepi_bands.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved fig_nepi_bands.png")


def fig_nepi_radar(lsoa: pd.DataFrame) -> None:
    """Radar chart showing three surfaces normalised to 0-100 for comparability."""
    types = ["Flat", "Terraced", "Semi", "Detached"]
    labels = ["Form\n(lower = better)", "Mobility\n(lower = better)",
              "Access coverage\n(higher = better)"]

    type_colors = {
        "Flat": "#2196F3",
        "Terraced": "#FF9800",
        "Semi": "#4CAF50",
        "Detached": "#E91E63",
    }

    # For radar, normalise each axis to 0-100 where 100 = best
    # Form and Mobility: invert (lower kWh = higher score)
    # Access: direct (higher coverage = higher score)
    form_min = lsoa["nepi_form_kwh"].quantile(0.02)
    form_max = lsoa["nepi_form_kwh"].quantile(0.98)
    mob_min = lsoa["nepi_mobility_kwh"].quantile(0.02)
    mob_max = lsoa["nepi_mobility_kwh"].quantile(0.98)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for t in types:
        sub = lsoa[lsoa["dominant_type"] == t]
        form_score = 100 * (1 - (sub["nepi_form_kwh"].median() - form_min) / (form_max - form_min))
        mob_score = 100 * (1 - (sub["nepi_mobility_kwh"].median() - mob_min) / (mob_max - mob_min))
        acc_score = sub["local_coverage"].median() * 100

        values = [
            max(0, min(100, form_score)),
            max(0, min(100, mob_score)),
            max(0, min(100, acc_score)),
        ]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=t, color=type_colors[t])
        ax.fill(angles, values, alpha=0.1, color=type_colors[t])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(["20", "40", "60", "80"], fontsize=8, color="#999")
    ax.set_title("NEPI Surface Performance\n(0–100, higher = better)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.2, 0), fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig_nepi_radar.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  Saved fig_nepi_radar.png")


def save_nepi_table(lsoa: pd.DataFrame) -> None:
    """Save NEPI scores to CSV for all OAs."""
    cols = [
        "OA21CD", "city", "dominant_type",
        "nepi_form_kwh", "nepi_mobility_kwh", "nepi_access_kwh",
        "nepi_total_kwh", "nepi_band", "local_coverage",
    ]
    out = lsoa[[c for c in cols if c in lsoa.columns]].copy()
    out_path = FIGURE_DIR / "nepi_scores.csv"
    out.to_csv(out_path, index=False)
    print(f"  Saved nepi_scores.csv ({len(out):,} OAs)")


def main() -> None:
    """Generate NEPI scores and figures."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    lsoa = load_and_aggregate()
    lsoa = build_accessibility(lsoa)
    lsoa = compute_nepi(lsoa)

    fig_nepi_scorecard(lsoa)
    fig_nepi_band_distribution(lsoa)
    fig_nepi_radar(lsoa)
    save_nepi_table(lsoa)

    print(f"\n  All NEPI outputs saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
