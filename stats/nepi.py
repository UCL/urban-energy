"""
Neighbourhood Energy Performance Index (NEPI).

Scores each Output Area on three surfaces of energy performance:

    1. Form    — thermal efficiency of built stock (metered building kWh/hh)
    2. Mobility — transport energy dependence (estimated transport kWh/hh)
    3. Access  — walkable coverage of essential services (local trip coverage %)

Each surface is scored 0–100 (higher = more efficient) using percentile ranks
against the national distribution. The composite NEPI is the mean of the three
surface scores, banded A–G following EPC conventions.

Usage:
    uv run python stats/nepi.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from proof_of_concept_oa import load_and_aggregate, build_accessibility
from basket_index_oa import compute_basket_index as compute_basket

FIGURE_DIR = Path(__file__).parent / "figures" / "nepi"

# EPC-style bands: A (most efficient) to G (least efficient)
BAND_THRESHOLDS = [
    (92, "A"),
    (81, "B"),
    (69, "C"),
    (55, "D"),
    (39, "E"),
    (21, "F"),
    (0, "G"),
]

BAND_COLORS = {
    "A": "#00845A",
    "B": "#2C9F29",
    "C": "#8CBF26",
    "D": "#FCD800",
    "E": "#F0AB00",
    "F": "#ED6F21",
    "G": "#E3242B",
}


def _score_inverse(series: pd.Series) -> pd.Series:
    """
    Score a 'lower is better' metric as 0–100 via inverted percentile rank.

    Parameters
    ----------
    series : pd.Series
        Raw values where lower = more efficient.

    Returns
    -------
    pd.Series
        Scores 0–100 where 100 = most efficient (lowest raw value).
    """
    # Percentile rank: 0 = lowest value, 100 = highest
    # Invert: lowest energy → highest score
    pct = series.rank(pct=True, na_option="keep") * 100
    return 100 - pct


def _score_direct(series: pd.Series) -> pd.Series:
    """
    Score a 'higher is better' metric as 0–100 via percentile rank.

    Parameters
    ----------
    series : pd.Series
        Raw values where higher = more efficient (e.g., coverage rate).

    Returns
    -------
    pd.Series
        Scores 0–100 where 100 = most efficient (highest raw value).
    """
    return series.rank(pct=True, na_option="keep") * 100


def _assign_band(score: float) -> str:
    """Assign EPC-style band from a 0–100 score."""
    for threshold, band in BAND_THRESHOLDS:
        if score >= threshold:
            return band
    return "G"


def compute_nepi(lsoa: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Neighbourhood Energy Performance Index for each OA.

    Parameters
    ----------
    lsoa : pd.DataFrame
        OA-level data from load_and_aggregate() with building energy,
        transport energy, and basket coverage columns.

    Returns
    -------
    pd.DataFrame
        Input data augmented with nepi_form, nepi_mobility, nepi_access,
        nepi_composite, and nepi_band columns.
    """
    print("=" * 70)
    print("NEIGHBOURHOOD ENERGY PERFORMANCE INDEX (NEPI)")
    print("=" * 70)

    # --- Surface 1: Form (building energy, lower = better) ---
    lsoa["nepi_form"] = _score_inverse(lsoa["building_kwh_per_hh"])

    # --- Surface 2: Mobility (transport energy, lower = better) ---
    transport_col = "transport_kwh_per_hh_total_est"
    if transport_col not in lsoa.columns:
        transport_col = "transport_kwh_per_hh_est"
    lsoa["nepi_mobility"] = _score_inverse(lsoa[transport_col])

    # --- Surface 3: Access (local service coverage, higher = better) ---
    # Computed directly from cityseer nearest-distance columns using
    # Gaussian decay: score = exp(-ln(2) * (d / d_half)^2)
    # where d_half is the walking-distance threshold per service type.
    # NaN (no destination within 4800m search radius) = 0 coverage.
    _SERVICE_THRESHOLDS: dict[str, tuple[str, int]] = {
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
    n_services_available = 0
    for col, (name, threshold) in _SERVICE_THRESHOLDS.items():
        if col not in lsoa.columns:
            continue
        n_services_available += 1
        vals = pd.to_numeric(lsoa[col], errors="coerce")
        # Gaussian decay: full credit at 0m, 50% at threshold
        score = np.exp(-np.log(2) * (vals / threshold) ** 2)
        score = score.fillna(0)  # No destination within search radius = zero
        cov_col = f"_cov_{name}"
        lsoa[cov_col] = score
        coverage_cols.append(cov_col)

    if coverage_cols:
        # Mean coverage across all service types (0–1)
        lsoa["local_coverage"] = lsoa[coverage_cols].mean(axis=1)
        # Convert to 0–100 score directly (already a meaningful 0–1 scale)
        lsoa["nepi_access"] = lsoa["local_coverage"] * 100
        print(f"\n  Access surface: {n_services_available} services, "
              f"Gaussian decay, median coverage = {lsoa['local_coverage'].median():.1%}")
        # Per-service medians
        for cov_col in coverage_cols:
            short = cov_col.replace("_cov_", "")
            print(f"    {short:<20s}: {lsoa[cov_col].median():.2f}")
    else:
        # Fallback to accessibility z-score if no nearest-distance columns
        acc = lsoa["accessibility"]
        lsoa["nepi_access"] = _score_direct(acc)
        print("  Access surface: fallback to accessibility z-score (no nearest-distance cols)")

    # --- Composite: equal-weighted mean of three surfaces ---
    surface_cols = ["nepi_form", "nepi_mobility", "nepi_access"]
    lsoa["nepi_composite"] = lsoa[surface_cols].mean(axis=1)

    # --- Band assignment ---
    lsoa["nepi_band"] = lsoa["nepi_composite"].apply(_assign_band)

    # --- Summary ---
    types = ["Flat", "Terraced", "Semi", "Detached"]
    print(
        f"\n  {'Type':<12s} {'Form':>6s} {'Mobility':>9s} {'Access':>7s} "
        f"{'Composite':>10s} {'Band':>5s}"
    )
    print(f"  {'-' * 54}")
    for t in types:
        sub = lsoa[lsoa["dominant_type"] == t]
        if len(sub) == 0:
            continue
        f = sub["nepi_form"].median()
        m = sub["nepi_mobility"].median()
        a = sub["nepi_access"].median()
        c = sub["nepi_composite"].median()
        band = _assign_band(c)
        print(f"  {t:<12s} {f:>6.1f} {m:>9.1f} {a:>7.1f} {c:>10.1f} {band:>5s}")

    # Band distribution
    print(f"\n  Band distribution (N={len(lsoa):,}):")
    for band in ["A", "B", "C", "D", "E", "F", "G"]:
        n = (lsoa["nepi_band"] == band).sum()
        pct = n / len(lsoa) * 100
        print(f"    {band}: {n:>7,d} ({pct:>5.1f}%)")

    # Band distribution by type
    print(f"\n  Band distribution by dominant type:")
    print(f"  {'Type':<12s} {'A':>6s} {'B':>6s} {'C':>6s} {'D':>6s} {'E':>6s} {'F':>6s} {'G':>6s}")
    print(f"  {'-' * 54}")
    for t in types:
        sub = lsoa[lsoa["dominant_type"] == t]
        parts = []
        for band in ["A", "B", "C", "D", "E", "F", "G"]:
            n = (sub["nepi_band"] == band).sum()
            pct = n / len(sub) * 100 if len(sub) > 0 else 0
            parts.append(f"{pct:>5.1f}%")
        print(f"  {t:<12s} {' '.join(parts)}")

    return lsoa


def fig_nepi_scorecard(lsoa: pd.DataFrame) -> None:
    """Generate the NEPI scorecard figure — three surfaces by housing type."""
    types = ["Flat", "Terraced", "Semi", "Detached"]
    surfaces = ["nepi_form", "nepi_mobility", "nepi_access"]
    labels = ["Form\n(building energy)", "Mobility\n(transport energy)", "Access\n(local services)"]

    medians = []
    for t in types:
        sub = lsoa[lsoa["dominant_type"] == t]
        row = [sub[s].median() for s in surfaces]
        medians.append(row)

    data = np.array(medians)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(labels))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    type_colors = {
        "Flat": "#2196F3",
        "Terraced": "#FF9800",
        "Semi": "#4CAF50",
        "Detached": "#E91E63",
    }

    for i, t in enumerate(types):
        bars = ax.bar(
            x + offsets[i] * width, data[i], width,
            label=t, color=type_colors[t], edgecolor="white",
        )
        for bar, val in zip(bars, data[i]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_ylabel("NEPI Score (0–100, higher = more efficient)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(
        "Neighbourhood Energy Performance Index by Housing Type",
        fontsize=12, fontweight="bold",
    )

    # Add band reference lines
    for threshold, band in BAND_THRESHOLDS:
        if threshold > 0:
            ax.axhline(y=threshold, color="#cccccc", linewidth=0.5, linestyle="--")
            ax.text(len(labels) - 0.5, threshold + 1, band,
                    fontsize=8, color="#999999", ha="center")

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
        data[t] = [(sub["nepi_band"] == b).sum() / total * 100 for b in bands]

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
    """Radar/spider chart showing three NEPI surfaces by housing type."""
    types = ["Flat", "Terraced", "Semi", "Detached"]
    surfaces = ["nepi_form", "nepi_mobility", "nepi_access"]
    labels = ["Form", "Mobility", "Access"]

    type_colors = {
        "Flat": "#2196F3",
        "Terraced": "#FF9800",
        "Semi": "#4CAF50",
        "Detached": "#E91E63",
    }

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for t in types:
        sub = lsoa[lsoa["dominant_type"] == t]
        values = [sub[s].median() for s in surfaces]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=t, color=type_colors[t])
        ax.fill(angles, values, alpha=0.1, color=type_colors[t])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(["20", "40", "60", "80"], fontsize=8, color="#999")
    ax.set_title("NEPI Surface Scores by Housing Type\n(0–100, higher = more efficient)",
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
        "building_kwh_per_hh", "transport_kwh_per_hh_total_est",
        "nepi_form", "nepi_mobility", "nepi_access",
        "nepi_composite", "nepi_band",
    ]
    out = lsoa[[c for c in cols if c in lsoa.columns]].copy()
    out_path = FIGURE_DIR / "nepi_scores.csv"
    out.to_csv(out_path, index=False)
    print(f"  Saved nepi_scores.csv ({len(out):,} OAs)")


def main() -> None:
    """Generate NEPI scores and figures."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Load OA data
    lsoa = load_and_aggregate()
    lsoa = build_accessibility(lsoa)

    # Compute basket coverage for the access surface
    try:
        lsoa = compute_basket(lsoa)
    except Exception as e:
        print(f"  Basket computation failed: {e}")
        print("  Falling back to accessibility z-score for access surface")

    # Compute NEPI
    lsoa = compute_nepi(lsoa)

    # Generate figures
    fig_nepi_scorecard(lsoa)
    fig_nepi_band_distribution(lsoa)
    fig_nepi_radar(lsoa)
    save_nepi_table(lsoa)

    print(f"\n  All NEPI outputs saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
