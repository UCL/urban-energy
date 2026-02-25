"""
LSOA Three Energy Surfaces: Publication-Ready Figures.

Generates 8 figures and 2 CSV tables for presenting the three energy
surfaces framework to colleagues and supervisors.

Hypothesis: Stop measuring buildings. Start measuring neighbourhoods.
A building's morphological type is not just a thermal envelope — it is
a commitment to a pattern of living.

Narrative arc:
    HYPOTHESIS → EVIDENCE (3 surfaces) → INTEGRATION (conduit) → ROBUSTNESS

Data loaded via proof_of_concept_lsoa functions (reads pre-aggregated
LSOA GeoPackage from processing/pipeline_lsoa.py).

Usage:
    uv run python stats/lsoa_figures.py
    uv run python stats/lsoa_figures.py canterbury
    uv run python stats/lsoa_figures.py manchester bristol york
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

# Import data pipeline from the PoC script (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from proof_of_concept_lsoa import (  # noqa: E402
    _run_ols,
    build_accessibility,
    compute_conduit,
    load_and_aggregate,
)

from urban_energy.paths import PROJECT_DIR  # noqa: E402

FIGURE_DIR = PROJECT_DIR / "stats" / "figures"

# Style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150})

# Compact → sprawl colour palette
TYPE_COLORS = {
    "Flat": "#3498db",
    "Terraced": "#2ecc71",
    "Semi": "#f39c12",
    "Detached": "#e74c3c",
}
TYPE_ORDER = ["Flat", "Terraced", "Semi", "Detached"]
COST_COLORS = {"building": "#3498db", "transport": "#e67e22"}


def _sigstars(p: float) -> str:
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Tables (CSV export)
# ---------------------------------------------------------------------------


def save_summary_tables(lsoa: pd.DataFrame) -> None:
    """
    Save three-surfaces summary and energy decomposition as CSVs.

    Parameters
    ----------
    lsoa : pd.DataFrame
        LSOA data with dominant_type, energy, and accessibility columns.
    """
    types = TYPE_ORDER
    valid = lsoa["transport_kwh_per_hh"].notna()
    sub = lsoa[valid]

    # Table 1: Three surfaces by housing type
    rows = []
    metrics = [
        ("N", None, None),
        ("S/V ratio", "lsoa_sv", lsoa),
        ("Height (m)", "height_mean", lsoa),
        ("People/ha", "people_per_ha", lsoa),
        ("People/hh", "avg_hh_size", lsoa),
        ("Building kWh/hh", "building_kwh_per_hh", lsoa),
        ("kWh/person (bldg)", "kwh_per_person", lsoa),
        ("Cars/hh", "cars_per_hh", sub),
        ("Transport kWh/hh", "transport_kwh_per_hh", sub),
        ("Total kWh/hh", "total_kwh_per_hh", sub),
        ("Accessibility PC1", "accessibility_pc1", lsoa),
        ("Conduit (access/kWh)", "conduit_total", sub),
    ]
    for label, col, src in metrics:
        row: dict[str, object] = {"metric": label}
        if col is None:
            for t in types:
                row[t] = int((lsoa["dominant_type"] == t).sum())
        else:
            for t in types:
                row[t] = src.loc[src["dominant_type"] == t, col].median()
        rows.append(row)
    df1 = pd.DataFrame(rows)
    df1.to_csv(FIGURE_DIR / "table1_three_surfaces.csv", index=False)
    print(f"  Saved table1_three_surfaces.csv ({len(df1)} rows)")

    # Table 2: Energy decomposition by housing type
    rows2 = []
    for t in types:
        s = sub[sub["dominant_type"] == t]
        if len(s) == 0:
            continue
        rows2.append({
            "type": t,
            "n": len(s),
            "building_kwh_hh": s["building_kwh_per_hh"].median(),
            "transport_kwh_hh": s["transport_kwh_per_hh"].median(),
            "total_kwh_hh": s["total_kwh_per_hh"].median(),
            "transport_share": (
                s["transport_kwh_per_hh"] / s["total_kwh_per_hh"]
            ).median(),
            "car_commute_share": s["car_commute_share"].median(),
            "avg_commute_km": s["avg_commute_km"].median(),
            "active_share": s["active_share"].median(),
        })
    df2 = pd.DataFrame(rows2)
    df2.to_csv(FIGURE_DIR / "table2_energy_decomposition.csv", index=False)
    print(f"  Saved table2_energy_decomposition.csv ({len(df2)} rows)")


# ---------------------------------------------------------------------------
# Figure 1: The Thermal Wash
# ---------------------------------------------------------------------------


def fig1_thermal_wash(lsoa: pd.DataFrame) -> None:
    """
    Grouped bar: building energy by housing type.

    Panel A: kWh/household (roughly flat).
    Panel B: kWh/person (per-capita inversion).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = [TYPE_COLORS[t] for t in TYPE_ORDER]

    for ax, col, ylabel, subtitle in [
        (
            axes[0],
            "building_kwh_per_hh",
            "Median kWh / household",
            "Metered household energy barely varies",
        ),
        (
            axes[1],
            "kwh_per_person",
            "Median kWh / person",
            "Per capita, compact is higher\u2009\u2014\u2009"
            "smaller households wash out S/V gain",
        ),
    ]:
        vals = [
            lsoa.loc[lsoa["dominant_type"] == t, col].median()
            for t in TYPE_ORDER
        ]
        x = np.arange(len(TYPE_ORDER))
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(
            lsoa[col].median(),
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Overall median",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(TYPE_ORDER)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle, fontsize=10)
        ax.legend(fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + ax.get_ylim()[1] * 0.01,
                f"{v:,.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    fig.suptitle(
        "Surface 1: The Thermal Wash", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "fig1_thermal_wash.png", bbox_inches="tight"
    )
    plt.close()
    print("  Saved fig1_thermal_wash.png")


# ---------------------------------------------------------------------------
# Figure 2: The Mobility Penalty
# ---------------------------------------------------------------------------


def fig2_mobility_penalty(lsoa: pd.DataFrame) -> None:
    """Stacked bar: building + transport energy by housing type."""
    sub = lsoa[lsoa["transport_kwh_per_hh"].notna()]
    fig, ax = plt.subplots(figsize=(10, 6))

    bldg = [
        sub.loc[sub["dominant_type"] == t, "building_kwh_per_hh"].median()
        for t in TYPE_ORDER
    ]
    trans = [
        sub.loc[sub["dominant_type"] == t, "transport_kwh_per_hh"].median()
        for t in TYPE_ORDER
    ]
    x = np.arange(len(TYPE_ORDER))
    w = 0.6

    ax.bar(
        x, bldg, w, label="Building", color=COST_COLORS["building"]
    )
    ax.bar(
        x,
        trans,
        w,
        bottom=bldg,
        label="Transport",
        color=COST_COLORS["transport"],
    )

    # Annotate totals and transport %
    for i, (b, t) in enumerate(zip(bldg, trans)):
        total = b + t
        tpct = t / total * 100
        ax.text(
            i, total + 200, f"{total:,.0f}", ha="center", fontsize=10, fontweight="bold"
        )
        ax.text(
            i,
            b + t / 2,
            f"{tpct:.0f}%",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
        )

    # Flat/Detached ratio
    total_flat = bldg[0] + trans[0]
    total_det = bldg[-1] + trans[-1]
    if total_det > 0:
        ratio = total_flat / total_det
        ax.annotate(
            f"Flat/Detached = {ratio:.2f}x",
            xy=(0.5, 0.95),
            xycoords="axes fraction",
            ha="center",
            fontsize=11,
            fontweight="bold",
            color="#c0392b",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(TYPE_ORDER)
    ax.set_ylabel("Median kWh / household")
    ax.set_title(
        "Surface 2: Transport Is the Dominant Cost Gradient",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper left")
    ax.annotate(
        "Building energy is ~flat. Transport doubles.",
        xy=(0.5, 0.02),
        xycoords="axes fraction",
        ha="center",
        fontsize=10,
        color="gray",
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "fig2_mobility_penalty.png", bbox_inches="tight"
    )
    plt.close()
    print("  Saved fig2_mobility_penalty.png")


# ---------------------------------------------------------------------------
# Figure 3: S/V vs Transport scatter
# ---------------------------------------------------------------------------


def fig3_sv_transport_scatter(lsoa: pd.DataFrame) -> None:
    """Scatter: S/V ratio vs transport energy, coloured by housing type."""
    sub = lsoa[lsoa["transport_kwh_per_hh"].notna()].copy()
    fig, ax = plt.subplots(figsize=(8, 6))

    for t in TYPE_ORDER:
        mask = sub["dominant_type"] == t
        ax.scatter(
            sub.loc[mask, "lsoa_sv"],
            sub.loc[mask, "transport_kwh_per_hh"],
            c=TYPE_COLORS[t],
            label=t,
            alpha=0.5,
            s=15,
            edgecolors="none",
        )

    # Linear fit
    valid = sub[["lsoa_sv", "transport_kwh_per_hh"]].dropna()
    slope, intercept, r, p, _ = sp_stats.linregress(
        valid["lsoa_sv"], valid["transport_kwh_per_hh"]
    )
    x_line = np.linspace(valid["lsoa_sv"].min(), valid["lsoa_sv"].max(), 100)
    ax.plot(x_line, intercept + slope * x_line, "k--", linewidth=1.5)
    ax.annotate(
        f"r = {r:.3f}  (p = {p:.1e})",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        fontweight="bold",
    )

    ax.set_xlabel("LSOA Aggregate S/V Ratio")
    ax.set_ylabel("Transport kWh / household")
    ax.set_title(
        "Morphological compactness predicts transport energy",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(title="Dominant type", fontsize=8)

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "fig3_sv_transport_scatter.png", bbox_inches="tight"
    )
    plt.close()
    print("  Saved fig3_sv_transport_scatter.png")


# ---------------------------------------------------------------------------
# Figure 4: Accessibility dividend (2-panel scatter)
# ---------------------------------------------------------------------------


def fig4_accessibility_dividend(lsoa: pd.DataFrame) -> None:
    """
    Two scatter panels showing the return side.

    Panel A: S/V → accessibility (compact delivers more city).
    Panel B: total energy → accessibility (less energy, more return).
    """
    sub = lsoa[lsoa["accessibility_pc1"].notna()].copy()
    # Log-transform accessibility to handle right-skewed distribution
    sub = sub[sub["accessibility_pc1"] > 0].copy()
    sub["log_accessibility"] = np.log10(sub["accessibility_pc1"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    panels = [
        (axes[0], "lsoa_sv", "LSOA Aggregate S/V",
         "Compact form delivers more city"),
        (
            axes[1],
            "total_kwh_per_hh",
            "Total kWh / household",
            "Less energy, more return",
        ),
    ]

    for ax, xcol, xlabel, title in panels:
        valid = sub[[xcol, "log_accessibility"]].dropna()
        for t in TYPE_ORDER:
            mask = sub["dominant_type"] == t
            d = sub.loc[mask]
            ax.scatter(
                d[xcol],
                d["log_accessibility"],
                c=TYPE_COLORS[t],
                label=t,
                alpha=0.5,
                s=12,
                edgecolors="none",
            )

        # Linear fit on log-transformed accessibility
        slope, intercept, r, p, _ = sp_stats.linregress(
            valid[xcol], valid["log_accessibility"]
        )
        x_line = np.linspace(valid[xcol].min(), valid[xcol].max(), 100)
        ax.plot(x_line, intercept + slope * x_line, "k--", linewidth=1.5)
        ax.annotate(
            f"r = {r:.3f}", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=10
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("log10(Accessibility PC1)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, title="Type")

    fig.suptitle(
        "Surface 3: The Accessibility Dividend",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "fig4_accessibility_dividend.png", bbox_inches="tight"
    )
    plt.close()
    print("  Saved fig4_accessibility_dividend.png")


# ---------------------------------------------------------------------------
# Figure 5: The Conduit (bar chart)
# ---------------------------------------------------------------------------


def fig5_conduit_bar(lsoa: pd.DataFrame) -> None:
    """Bar chart: conduit efficiency by housing type and S/V quartile."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = [TYPE_COLORS[t] for t in TYPE_ORDER]

    # Panel A: by housing type
    ax = axes[0]
    vals = [
        lsoa.loc[lsoa["dominant_type"] == t, "conduit_total"].median()
        for t in TYPE_ORDER
    ]
    iqr_lo = [
        lsoa.loc[lsoa["dominant_type"] == t, "conduit_total"].quantile(0.25)
        for t in TYPE_ORDER
    ]
    iqr_hi = [
        lsoa.loc[lsoa["dominant_type"] == t, "conduit_total"].quantile(0.75)
        for t in TYPE_ORDER
    ]
    x = np.arange(len(TYPE_ORDER))
    yerr_lo = [v - lo for v, lo in zip(vals, iqr_lo)]
    yerr_hi = [hi - v for v, hi in zip(vals, iqr_hi)]
    ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.errorbar(x, vals, yerr=[yerr_lo, yerr_hi], fmt="none", color="black", capsize=4)

    if vals[-1] > 0:
        ratio = vals[0] / vals[-1]
        ax.annotate(
            f"Flat/Detached = {ratio:.1f}x",
            xy=(0.5, 0.92),
            xycoords="axes fraction",
            ha="center",
            fontsize=11,
            fontweight="bold",
            color="#c0392b",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(TYPE_ORDER)
    ax.set_ylabel("Median Conduit (accessibility / kWh)")
    ax.set_title("By dominant housing type", fontsize=10)

    # Panel B: by S/V quartile
    ax = axes[1]
    qs = ["Q1 compact", "Q2", "Q3", "Q4 sprawl"]
    q_colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]
    vals_q = [
        lsoa.loc[lsoa["sv_quartile"] == q, "conduit_total"].median() for q in qs
    ]
    x = np.arange(len(qs))
    ax.bar(x, vals_q, color=q_colors, edgecolor="black", linewidth=0.5)
    if vals_q[-1] > 0:
        ratio = vals_q[0] / vals_q[-1]
        ax.annotate(
            f"Q1/Q4 = {ratio:.1f}x",
            xy=(0.5, 0.92),
            xycoords="axes fraction",
            ha="center",
            fontsize=11,
            fontweight="bold",
            color="#c0392b",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(qs, fontsize=9)
    ax.set_ylabel("Median Conduit (accessibility / kWh)")
    ax.set_title("By S/V compactness quartile", fontsize=10)

    fig.suptitle(
        "The Conduit: How Much City per kWh?",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "fig5_conduit_bar.png", bbox_inches="tight"
    )
    plt.close()
    print("  Saved fig5_conduit_bar.png")


# ---------------------------------------------------------------------------
# Figure 6: Correlation heatmap
# ---------------------------------------------------------------------------


def fig6_correlation_heatmap(lsoa: pd.DataFrame) -> None:
    """Seaborn heatmap of key variables: form -> cost -> return."""
    cols = [
        "lsoa_sv",
        "height_mean",
        "avg_hh_size",
        "building_kwh_per_hh",
        "cars_per_hh",
        "transport_kwh_per_hh",
        "total_kwh_per_hh",
        "accessibility_pc1",
        "conduit_total",
    ]
    if "people_per_ha" in lsoa.columns:
        cols.insert(2, "people_per_ha")

    present = [c for c in cols if c in lsoa.columns]
    corr = lsoa[present].corr()

    # Short labels
    short = {
        "lsoa_sv": "S/V",
        "height_mean": "Height",
        "people_per_ha": "Pop/ha",
        "avg_hh_size": "HH size",
        "building_kwh_per_hh": "Bldg kWh",
        "cars_per_hh": "Cars/hh",
        "transport_kwh_per_hh": "Trans kWh",
        "total_kwh_per_hh": "Total kWh",
        "accessibility_pc1": "Access",
        "conduit_total": "Conduit",
    }
    labels = [short.get(c, c) for c in present]

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(
        "Correlation Structure: Form \u2192 Cost \u2192 Return",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "fig6_correlation_heatmap.png", bbox_inches="tight"
    )
    plt.close()
    print("  Saved fig6_correlation_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 7: Deprivation control (forest plot)
# ---------------------------------------------------------------------------


def fig7_deprivation_forest(lsoa: pd.DataFrame) -> None:
    """Forest plot: S/V coefficient within each deprivation quintile."""
    if "deprivation_quintile" not in lsoa.columns:
        print("  Skipping fig7 — no deprivation quintiles")
        return

    quintiles = ["Q1 most", "Q2", "Q3", "Q4", "Q5 least"]
    physics = ["lsoa_sv", "height_mean"]
    results: list[dict[str, object]] = []

    for q in quintiles:
        sub = lsoa[lsoa["deprivation_quintile"] == q].copy()
        if len(sub) < 20:
            continue
        m = _run_ols(sub, "log_conduit_total", physics, f"dep-{q}")
        if m is None or "lsoa_sv" not in m.params:
            continue
        beta = m.params["lsoa_sv"]
        se = m.bse["lsoa_sv"]
        p = m.pvalues["lsoa_sv"]
        results.append({
            "quintile": q,
            "beta": beta,
            "ci_lo": beta - 1.96 * se,
            "ci_hi": beta + 1.96 * se,
            "p": p,
            "n": int(m.nobs),
        })

    if not results:
        print("  Skipping fig7 — no valid regressions")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(results))

    for i, r in enumerate(results):
        color = "#e74c3c" if r["p"] < 0.05 else "#95a5a6"
        ax.barh(i, r["beta"], color=color, height=0.5, alpha=0.7)
        ax.plot(
            [r["ci_lo"], r["ci_hi"]], [i, i], "k-", linewidth=1.5
        )
        sig = _sigstars(r["p"])
        ax.text(
            r["ci_hi"] + 0.02,
            i,
            f"{sig}  (N={r['n']})",
            va="center",
            fontsize=9,
        )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["quintile"] for r in results])
    ax.set_xlabel("S/V coefficient (\u03b2) on log(conduit efficiency)")
    ax.set_title(
        "S/V \u2192 Conduit Efficiency Within Each Deprivation Quintile",
        fontsize=12,
        fontweight="bold",
    )
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "fig7_deprivation_forest.png", bbox_inches="tight"
    )
    plt.close()
    print("  Saved fig7_deprivation_forest.png")


# ---------------------------------------------------------------------------
# Figure 8: Per-city forest plot
# ---------------------------------------------------------------------------


def fig8_city_forest(lsoa: pd.DataFrame) -> None:
    """Forest plot: S/V coefficient within each city."""
    if "city" not in lsoa.columns or lsoa["city"].nunique() < 2:
        print("  Skipping fig8 — single city")
        return

    physics = ["lsoa_sv", "height_mean"]
    cities = sorted(lsoa["city"].unique())
    results: list[dict[str, object]] = []

    for city in cities:
        sub = lsoa[lsoa["city"] == city].copy()
        m = _run_ols(sub, "log_conduit_total", physics, city)
        if m is None or "lsoa_sv" not in m.params:
            continue
        beta = m.params["lsoa_sv"]
        se = m.bse["lsoa_sv"]
        p = m.pvalues["lsoa_sv"]
        results.append({
            "city": city,
            "beta": beta,
            "ci_lo": beta - 1.96 * se,
            "ci_hi": beta + 1.96 * se,
            "p": p,
            "n": int(m.nobs),
        })

    if not results:
        print("  Skipping fig8 — no valid regressions")
        return

    fig, ax = plt.subplots(figsize=(8, max(3, len(results) * 0.5 + 1)))
    y_pos = np.arange(len(results))

    for i, r in enumerate(results):
        color = "#e74c3c" if r["p"] < 0.05 else "#95a5a6"
        ax.barh(i, r["beta"], color=color, height=0.5, alpha=0.7)
        ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], "k-", linewidth=1.5)
        sig = _sigstars(r["p"])
        ax.text(
            r["ci_hi"] + 0.02,
            i,
            f"{sig}  (N={r['n']})",
            va="center",
            fontsize=9,
        )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["city"] for r in results])
    ax.set_xlabel("S/V coefficient (\u03b2) on log(conduit efficiency)")
    ax.set_title(
        "Morphology\u2013Efficiency Relationship Across Cities",
        fontsize=12,
        fontweight="bold",
    )
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "fig8_city_forest.png", bbox_inches="tight"
    )
    plt.close()
    print("  Saved fig8_city_forest.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cities: list[str] | None = None) -> None:
    """Generate all figures and tables."""
    print("=" * 70)
    print("THREE ENERGY SURFACES: GENERATING FIGURES")
    print("=" * 70)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Load and prepare data (same pipeline as PoC analysis)
    lsoa = load_and_aggregate(cities)
    lsoa = build_accessibility(lsoa)
    lsoa = compute_conduit(lsoa)

    print(f"\n{'=' * 70}")
    print(f"GENERATING FIGURES ({len(lsoa):,} LSOAs)")
    print("=" * 70)

    save_summary_tables(lsoa)
    fig1_thermal_wash(lsoa)
    fig2_mobility_penalty(lsoa)
    fig3_sv_transport_scatter(lsoa)
    fig4_accessibility_dividend(lsoa)
    fig5_conduit_bar(lsoa)
    fig6_correlation_heatmap(lsoa)
    fig7_deprivation_forest(lsoa)
    fig8_city_forest(lsoa)

    print(f"\n{'=' * 70}")
    print(f"All outputs saved to: {FIGURE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    _cities = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(cities=_cities or None)
