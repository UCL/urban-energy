"""
LSOA Three Energy Surfaces: Publication-Ready Figures.

Generates 8 figures and 2 CSV tables for presenting the three energy
surfaces framework to colleagues and supervisors.

Hypothesis: Stop measuring buildings. Start measuring neighbourhoods.
A building's morphological type is not just a thermal envelope — it is
a commitment to a pattern of living.

Narrative arc:
    HYPOTHESIS → EVIDENCE (3 surfaces) → INTEGRATION (access/kWh) → ROBUSTNESS

Data loaded via proof_of_concept_lsoa functions (reads pre-aggregated
LSOA GeoPackage from processing/pipeline_lsoa.py). Transport energy is
an estimated commute-energy metric from Census ts058 (distance bands)
and ts061 (mode counts), using mode-specific energy intensities.

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
    _NTS_TOTAL_TO_COMMUTE_DISTANCE_FACTOR,
    build_accessibility,
    compute_access_per_kwh,
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


def _add_centroid_marker(
    ax: plt.Axes,
    x: float,
    y: float,
    color: str,
    *,
    x_is_log10: bool = False,
) -> None:
    """Draw a small centroid marker for a contour cluster."""
    if pd.isna(x) or pd.isna(y):
        return
    x_plot = 10**x if x_is_log10 else x
    ax.scatter(
        [x_plot],
        [y],
        s=26,
        color=color,
        edgecolors="black",
        linewidths=0.5,
        zorder=6,
    )


def _kde_peak_xy(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray
) -> tuple[float, float] | None:
    """Return the peak-density coordinate from a KDE surface grid."""
    if Z.size == 0 or not np.isfinite(Z).any():
        return None
    idx = np.unravel_index(np.nanargmax(Z), Z.shape)
    return float(X[idx]), float(Y[idx])


def _add_footnote(fig: plt.Figure, text: str) -> None:
    """Add a small data-source footnote at the bottom of a figure."""
    import textwrap

    # ~17 chars per inch at fontsize 8, leaving ~5% left+right margin.
    fig_width = fig.get_size_inches()[0]
    chars_per_line = int(fig_width * 17 * 0.9)
    wrapped = "\n".join(textwrap.wrap(text, width=chars_per_line))
    fig.text(
        0.5,
        -0.05,
        wrapped,
        ha="center",
        va="top",
        fontsize=8,
        color="#666666",
        style="italic",
    )


# -- Reusable source / methodology fragments -----------------------------------
_SRC_BLDG = (
    "Building energy: DESNZ metered domestic gas + electricity"
    " at LSOA level (2023). No per-building breakdown exists"
    " -- the whole LSOA total is divided by household count."
)
_SRC_TYPE = (
    "Housing type: each LSOA is labelled by its dominant"
    " Census 2021 accommodation type (ts044, highest share)."
    " This is an ecological grouping -- bars show median"
    " LSOA-level energy for LSOAs dominated by that type,"
    " not individual building energy."
)
_SRC_TRANSPORT = (
    "Transport energy estimate (commute only): ts061 mode counts"
    " (private = drive+passenger+taxi+motorcycle; public = bus+train+metro),"
    " multiplied by ts058 travelling-commuter distance and annualised"
    " (x2 x 220 days). Energy intensities: road passenger 34.3 ktoe/billion pkm"
    " and rail passenger 15.3 ktoe/billion pkm (ECUK 2025), converted to"
    " 0.399 and 0.178 kWh/pkm."
)
_SRC_TRANSPORT_TOTAL_EST = (
    "Overall-travel scenario: commute-energy estimate scaled by "
    f"{_NTS_TOTAL_TO_COMMUTE_DISTANCE_FACTOR:.2f}x "
    "(NTS 2024 total miles / commuting miles per person: 6,082 / 1,007)."
)
_SRC_TRANSPORT_MODE = (
    f"Mode decomposition uses the same commute-energy method: {_SRC_TRANSPORT}"
)
_SRC_DENSITY = "Population density: Census 2021 population / OA area (people/ha)."
_SRC_ACCESS = (
    "Accessibility: cityseer gravity-weighted counts at 800 m"
    " pedestrian catchment, per trophic layer (commercial,"
    " transport, green space, education, health)."
)


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
    valid = lsoa["transport_kwh_per_hh_est"].notna()
    sub = lsoa[valid]
    transport_summary = _transport_type_summary(lsoa)

    # Table 1: Three surfaces by housing type
    rows = []
    metrics = [
        ("N", None, None),
        ("People/ha", "people_per_ha", lsoa),
        ("Height (m)", "height_mean", lsoa),
        ("People/hh", "avg_hh_size", lsoa),
        ("Building kWh/hh", "building_kwh_per_hh", lsoa),
        ("kWh/person (bldg)", "kwh_per_person", lsoa),
        ("Cars/hh", "cars_per_hh", sub),
        ("Transport kWh/hh (commute est)", "transport_kwh_per_hh_est", sub),
        ("Transport kWh/hh (overall est)", "transport_kwh_per_hh_total_est", sub),
        ("Private commute kWh/hh (est)", "private_transport_kwh_per_hh_est", sub),
        ("Public commute kWh/hh (est)", "public_transport_kwh_per_hh_est", sub),
        ("Total kWh/hh (commute base)", "total_kwh_per_hh", sub),
        ("Total kWh/hh (overall est)", "total_kwh_per_hh_total_est", sub),
        ("Accessibility", "accessibility", lsoa),
        ("kWh / Access", "kwh_per_access", sub),
    ]
    for label, col, src in metrics:
        row: dict[str, object] = {"metric": label}
        if col is None:
            for t in types:
                row[t] = int((lsoa["dominant_type"] == t).sum())
        else:
            for t in types:
                if col == "transport_kwh_per_hh_est":
                    row[t] = transport_summary.loc[t, "transport_kwh_hh_commute_est"]
                elif col == "transport_kwh_per_hh_total_est":
                    row[t] = transport_summary.loc[t, "transport_kwh_hh_total_est"]
                elif col == "total_kwh_per_hh":
                    row[t] = transport_summary.loc[t, "total_kwh_hh_commute_base"]
                elif col == "total_kwh_per_hh_total_est":
                    row[t] = transport_summary.loc[t, "total_kwh_hh_total_est"]
                else:
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
        rows2.append(
            {
                "type": t,
                "n": len(s),
                "building_kwh_hh": transport_summary.loc[t, "building_kwh_hh"],
                "transport_kwh_hh_commute_est": transport_summary.loc[
                    t, "transport_kwh_hh_commute_est"
                ],
                "transport_kwh_hh_total_est": transport_summary.loc[
                    t, "transport_kwh_hh_total_est"
                ],
                "total_kwh_hh_commute_base": transport_summary.loc[
                    t, "total_kwh_hh_commute_base"
                ],
                "total_kwh_hh_total_est": transport_summary.loc[
                    t, "total_kwh_hh_total_est"
                ],
                "transport_share_commute_est": transport_summary.loc[
                    t, "transport_kwh_hh_commute_est"
                ]
                / transport_summary.loc[t, "total_kwh_hh_commute_base"],
                "transport_share_total_est": transport_summary.loc[
                    t, "transport_kwh_hh_total_est"
                ]
                / transport_summary.loc[t, "total_kwh_hh_total_est"],
                "car_commute_share": s["car_commute_share"].median(),
                "private_commute_share": s["private_commute_share"].median(),
                "public_commute_share": s["public_commute_share"].median(),
                "avg_commute_km": s["avg_commute_km"].median(),
                "avg_commute_km_travelling": s["avg_commute_km_travelling"].median(),
                "private_transport_kwh_hh_est": s[
                    "private_transport_kwh_per_hh_est"
                ].median(),
                "public_transport_kwh_hh_est": s[
                    "public_transport_kwh_per_hh_est"
                ].median(),
                "active_share": s["active_share"].median(),
            }
        )
    df2 = pd.DataFrame(rows2)
    df2.to_csv(FIGURE_DIR / "table2_energy_decomposition.csv", index=False)
    print(f"  Saved table2_energy_decomposition.csv ({len(df2)} rows)")


# ---------------------------------------------------------------------------
# Figure 1: Building Energy
# ---------------------------------------------------------------------------


def fig1_building_energy(lsoa: pd.DataFrame) -> None:
    """
    Grouped bar: building energy by housing type.

    Panel A: kWh/household.
    Panel B: kWh/hh stratified by deprivation tercile.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = [TYPE_COLORS[t] for t in TYPE_ORDER]

    # --- Panel A: kWh per household ---
    ax = axes[0]
    col = "building_kwh_per_hh"
    vals = [lsoa.loc[lsoa["dominant_type"] == t, col].median() for t in TYPE_ORDER]
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
    ax.set_ylabel("Median kWh / household")
    ax.set_title("(A) kWh per household", fontsize=10, fontweight="bold")
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
    axes[0].set_xlabel("Housing type", labelpad=10)

    # --- Panel B: kWh/hh by type within deprivation tercile ---
    ax = axes[1]
    sub = lsoa[
        lsoa["pct_not_deprived"].notna()
        & lsoa["building_kwh_per_hh"].notna()
        & lsoa["dominant_type"].notna()
    ].copy()
    dep_labels = ["Most deprived", "Middle", "Least deprived"]
    sub["dep_tercile"] = pd.qcut(sub["pct_not_deprived"], 3, labels=dep_labels)
    dep_colors = ["#d73027", "#fee090", "#4575b4"]
    bar_width = 0.22
    x = np.arange(len(TYPE_ORDER))
    for i, (dep, dc) in enumerate(zip(dep_labels, dep_colors)):
        vals = [
            sub.loc[
                (sub["dominant_type"] == t) & (sub["dep_tercile"] == dep),
                "building_kwh_per_hh",
            ].median()
            for t in TYPE_ORDER
        ]
        offset = (i - 1) * bar_width
        bars = ax.bar(
            x + offset,
            vals,
            bar_width,
            color=dc,
            edgecolor="black",
            linewidth=0.5,
            label=dep,
        )
        for bar, v in zip(bars, vals):
            if pd.notna(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 100,
                    f"{v:,.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    fontweight="bold",
                )
    ax.set_xticks(x)
    ax.set_xticklabels(TYPE_ORDER)
    ax.set_ylabel("Median kWh / household")
    ax.set_title("(B) kWh/hh by deprivation tercile", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, title="Deprivation", title_fontsize=7)
    axes[1].set_xlabel("Housing type", labelpad=10)
    fig.suptitle(
        "Surface 1: Building Energy by Housing Type",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    _add_footnote(
        fig,
        "(A) Each LSOA binned by dominant Census 2021 accommodation type (ts044, highest share). "
        "Bars show median LSOA-level metered energy (DESNZ) divided by household count.  "
        "(B) LSOAs split into thirds by % households not deprived (Census ts011). "
        "Deprivation is the strongest confounder of building energy (R\u00b2=0.12 vs R\u00b2=0.08 for type): "
        "deprived households use less energy. Within each deprivation band the compact\u2192sprawl gradient remains.",
    )
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig1_building_energy.png", bbox_inches="tight")
    plt.close()
    print("  Saved fig1_building_energy.png")


# ---------------------------------------------------------------------------
# Figure 2: Transport Energy by Housing Type
# ---------------------------------------------------------------------------


def _transport_type_summary(lsoa: pd.DataFrame) -> pd.DataFrame:
    """
    Shared type-level medians for transport figures.

    Uses one common subset and one aggregation rule so fig2 and fig2b
    stay numerically consistent.
    """
    needed = {
        "dominant_type",
        "building_kwh_per_hh",
        "private_transport_kwh_per_hh_est",
        "public_transport_kwh_per_hh_est",
    }
    sub = lsoa.dropna(subset=sorted(needed)).copy()
    rows: list[dict[str, float | str]] = []
    for t in TYPE_ORDER:
        s = sub[sub["dominant_type"] == t]
        rows.append(
            {
                "type": t,
                "building_kwh_hh": float(s["building_kwh_per_hh"].median()),
                "private_kwh_hh_est": float(
                    s["private_transport_kwh_per_hh_est"].median()
                ),
                "public_kwh_hh_est": float(
                    s["public_transport_kwh_per_hh_est"].median()
                ),
            }
        )
    out = pd.DataFrame(rows).set_index("type")
    out["transport_kwh_hh_commute_est"] = (
        out["private_kwh_hh_est"] + out["public_kwh_hh_est"]
    )
    out["transport_kwh_hh_total_est"] = (
        out["transport_kwh_hh_commute_est"] * _NTS_TOTAL_TO_COMMUTE_DISTANCE_FACTOR
    )
    out["total_kwh_hh_commute_base"] = (
        out["building_kwh_hh"] + out["transport_kwh_hh_commute_est"]
    )
    out["total_kwh_hh_total_est"] = (
        out["building_kwh_hh"] + out["transport_kwh_hh_total_est"]
    )
    return out


def fig2_mobility_penalty(lsoa: pd.DataFrame) -> None:
    """Stacked bars: commute-based and overall-scenario transport energy by housing type."""
    summary = _transport_type_summary(lsoa)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    bldg = [summary.loc[t, "building_kwh_hh"] for t in TYPE_ORDER]
    trans_commute = [summary.loc[t, "transport_kwh_hh_commute_est"] for t in TYPE_ORDER]
    trans_total = [summary.loc[t, "transport_kwh_hh_total_est"] for t in TYPE_ORDER]
    x = np.arange(len(TYPE_ORDER))
    w = 0.6

    def _draw_panel(
        ax: plt.Axes,
        transport_vals: list[float],
        panel_title: str,
        transport_label: str,
    ) -> None:
        ax.bar(x, bldg, w, label="Building", color=COST_COLORS["building"])
        ax.bar(
            x,
            transport_vals,
            w,
            bottom=bldg,
            label=transport_label,
            color=COST_COLORS["transport"],
        )

        for i, (b, t) in enumerate(zip(bldg, transport_vals)):
            total = b + t
            tpct = t / total * 100
            ax.text(
                i,
                total + 200,
                f"{total:,.0f}",
                ha="center",
                fontsize=10,
                fontweight="bold",
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

        total_flat = bldg[0] + transport_vals[0]
        total_det = bldg[-1] + transport_vals[-1]
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
        ax.set_title(panel_title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Housing type", labelpad=6)

    _draw_panel(
        axes[0],
        trans_commute,
        "(A) Commute-based transport estimate",
        "Transport (commute est.)",
    )
    _draw_panel(
        axes[1],
        trans_total,
        "(B) Overall-travel scenario estimate",
        "Transport (overall est.)",
    )

    axes[0].set_ylabel("Median kWh / household")
    axes[0].legend(loc="upper left")
    axes[1].legend(loc="upper left")
    fig.suptitle(
        "Surface 2: The Transport Cost Gradient",
        fontsize=13,
        fontweight="bold",
    )
    _add_footnote(
        fig,
        "LSOAs binned by dominant Census accommodation type (ts044). "
        "Building: DESNZ metered gas + electricity per household. "
        f"Commute transport: {_SRC_TRANSPORT} "
        f"Overall scenario: {_SRC_TRANSPORT_TOTAL_EST} "
        "For consistency with Figure 2b, commute transport bars are "
        "median(private) + median(public) within each type. "
        "Stack totals are sums of component medians.",
    )
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig2_mobility_penalty.png", bbox_inches="tight")
    plt.close()
    print("  Saved fig2_mobility_penalty.png")


# ---------------------------------------------------------------------------
# Figure 2b: Private vs Public commute energy estimate
# ---------------------------------------------------------------------------


def fig2b_private_public_transport(lsoa: pd.DataFrame) -> None:
    """Grouped bars: estimated private vs public commute energy by housing type."""
    needed = {"private_transport_kwh_per_hh_est", "public_transport_kwh_per_hh_est"}
    if not needed.issubset(lsoa.columns):
        print("  Skipping fig2b — private/public mode-energy columns not found")
        return

    summary = _transport_type_summary(lsoa)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(TYPE_ORDER))
    w = 0.36

    priv = [summary.loc[t, "private_kwh_hh_est"] for t in TYPE_ORDER]
    pub = [summary.loc[t, "public_kwh_hh_est"] for t in TYPE_ORDER]

    b1 = ax.bar(
        x - w / 2,
        priv,
        w,
        label="Private transport (est.)",
        color="#c0392b",
        edgecolor="black",
        linewidth=0.5,
    )
    b2 = ax.bar(
        x + w / 2,
        pub,
        w,
        label="Public transport (est.)",
        color="#2980b9",
        edgecolor="black",
        linewidth=0.5,
    )
    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            if pd.notna(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + max(5, ax.get_ylim()[1] * 0.01),
                    f"{v:,.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(TYPE_ORDER)
    ax.set_ylabel("Estimated commute energy (kWh / household / year)")
    ax.set_title(
        "Surface 2 Extension: Private vs Public Transport Energy",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Housing type", labelpad=8)
    ax.legend()

    # Add pair totals to match fig2 commute panel exactly.
    for i, (p1, p2) in enumerate(zip(priv, pub)):
        ax.text(
            i,
            p1 + p2 + max(10, ax.get_ylim()[1] * 0.015),
            f"{p1 + p2:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#444444",
        )

    _add_footnote(
        fig,
        f"{_SRC_TRANSPORT_MODE} Values are commute-energy estimates (not total annual travel"
        " across all trip purposes). Private+public totals use the same median rule as Figure 2.",
    )
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig2b_private_public_transport.png", bbox_inches="tight")
    plt.close()
    print("  Saved fig2b_private_public_transport.png")


# ---------------------------------------------------------------------------
# Figure 3: Density and Transport
# ---------------------------------------------------------------------------


def fig3_density_transport_scatter(lsoa: pd.DataFrame) -> None:
    """KDE contours: population density vs transport energy, per housing type."""
    from scipy.stats import gaussian_kde  # noqa: E402

    sub = lsoa[lsoa["transport_kwh_per_hh_est"].notna()].copy()
    sub = sub[sub["people_per_ha"] > 0].copy()

    valid = sub[["people_per_ha", "transport_kwh_per_hh_est", "dominant_type"]].dropna()
    log_dens_all = np.log10(valid["people_per_ha"])

    # Grid: wide enough to cover all types including low-density Detached
    X_LIM = (1.0, 1000.0)
    Y_LIM = (0.0, valid["transport_kwh_per_hh_est"].quantile(0.99))
    xmin_log, xmax_log = np.log10(X_LIM[0]), np.log10(X_LIM[1])
    xi = np.linspace(xmin_log, xmax_log, 300)
    yi = np.linspace(Y_LIM[0], Y_LIM[1], 300)
    Xi, Yi = np.meshgrid(xi, yi)
    grid_pts = np.vstack([Xi.ravel(), Yi.ravel()])

    fig, ax = plt.subplots(figsize=(8, 6))

    for t in TYPE_ORDER:
        mask = valid["dominant_type"] == t
        pts = valid.loc[mask]
        if len(pts) < 20:
            continue
        log_x = np.log10(pts["people_per_ha"])
        y_vals = pts["transport_kwh_per_hh_est"]
        xy = np.vstack([log_x, y_vals])
        kde = gaussian_kde(xy, bw_method=0.5)
        Z = kde(grid_pts).reshape(Xi.shape)
        # Normalise to [0,1] per type so contour levels are comparable
        Z = Z / Z.max()
        color = TYPE_COLORS[t]
        # Stepped filled bands: progressively more opaque toward centre
        band_levels = [0.35, 0.55, 0.72, 0.85, 0.93]
        band_alphas = [0.07, 0.13, 0.21, 0.33, 0.52]
        for lo, hi, a in zip(band_levels, band_levels[1:] + [1.01], band_alphas):
            ax.contourf(
                10**Xi,
                Yi,
                Z,
                levels=[lo, hi],
                colors=[color],
                alpha=a,
            )
        # Crisp outer boundary line
        ax.contour(
            10**Xi,
            Yi,
            Z,
            levels=[0.35],
            colors=[color],
            linewidths=[1.0],
            alpha=0.7,
        )
        # Thick semi-transparent proxy for legend
        ax.plot([], [], color=color, linewidth=8, alpha=0.35, label=t)
        peak_xy = _kde_peak_xy(Xi, Yi, Z)
        if peak_xy is not None:
            _add_centroid_marker(ax, peak_xy[0], peak_xy[1], color, x_is_log10=True)

    _, _, r, *_ = sp_stats.linregress(log_dens_all, valid["transport_kwh_per_hh_est"])

    ax.set_xscale("log")
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_ylabel("Transport kWh / household (est.)")
    ax.set_title(
        "Neighbourhood density is associated with transport energy",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(title="Dominant type", fontsize=8)
    ax.set_xlabel("Population density (people / ha, log scale)", labelpad=10)
    _add_footnote(
        fig,
        "Filled KDE bands: shading intensifies toward each type's peak."
        " Denser neighbourhoods are associated with lower transport energy."
        " Contours normalised within type. "
        f"{_SRC_DENSITY}  {_SRC_TRANSPORT}",
    )
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig3_density_transport.png", bbox_inches="tight")
    plt.close()
    print("  Saved fig3_density_transport.png")


# ---------------------------------------------------------------------------
# Figure 4: Accessibility dividend (2-panel scatter)
# ---------------------------------------------------------------------------


def _kde_gradient(
    ax: plt.Axes,
    log_x: np.ndarray,
    y: np.ndarray,
    color: str,
    xmin_log: float,
    xmax_log: float,
    ymin: float,
    ymax: float,
) -> tuple[float, float] | None:
    """Draw gradient-filled KDE contours and return peak density location."""
    from scipy.stats import gaussian_kde  # noqa: E402

    xy = np.vstack([log_x, y])
    kde = gaussian_kde(xy, bw_method=0.5)
    xi = np.linspace(xmin_log, xmax_log, 200)
    yi = np.linspace(ymin, ymax, 200)
    Xi, Yi = np.meshgrid(xi, yi)
    Z = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
    Z = Z / Z.max()
    band_levels = [0.35, 0.55, 0.72, 0.85, 0.93]
    band_alphas = [0.07, 0.13, 0.21, 0.33, 0.52]
    for lo, hi, a in zip(band_levels, band_levels[1:] + [1.01], band_alphas):
        ax.contourf(10**Xi, Yi, Z, levels=[lo, hi], colors=[color], alpha=a)
    # Crisp outer boundary line
    ax.contour(
        10**Xi, Yi, Z, levels=[0.35], colors=[color], linewidths=[1.0], alpha=0.7
    )
    return _kde_peak_xy(Xi, Yi, Z)


def fig4_accessibility_dividend(lsoa: pd.DataFrame) -> None:
    """KDE panel: population density vs street frontage by dwelling type."""
    sub = lsoa[lsoa["people_per_ha"] > 0].copy()

    X_LIM = (10.0, 500.0)
    xmin_log, xmax_log = np.log10(X_LIM[0]), np.log10(X_LIM[1])
    ycol, ylabel = "street_frontage", "Street network node density (800 m catchment)"
    ymin, ymax = 0.0, 275.0

    valid = sub[["people_per_ha", ycol, "dominant_type"]].dropna()
    valid = valid[
        (valid["people_per_ha"] >= X_LIM[0])
        & (valid["people_per_ha"] <= X_LIM[1])
        & (valid[ycol] >= ymin)
        & (valid[ycol] <= ymax)
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    for t in TYPE_ORDER:
        pts = valid[valid["dominant_type"] == t]
        if len(pts) < 20:
            continue
        x_vals = pts["people_per_ha"].values
        y_vals = pts[ycol].values
        peak_xy = _kde_gradient(
            ax,
            np.log10(x_vals),
            y_vals,
            TYPE_COLORS[t],
            xmin_log,
            xmax_log,
            ymin,
            ymax,
        )
        if peak_xy is not None:
            _add_centroid_marker(
                ax, peak_xy[0], peak_xy[1], TYPE_COLORS[t], x_is_log10=True
            )
        ax.plot([], [], color=TYPE_COLORS[t], linewidth=8, alpha=0.35, label=t)

    ax.set_xscale("log")
    ax.set_xlim(*X_LIM)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Population density (people / ha, log scale)", labelpad=10)
    ax.set_ylabel(ylabel)
    ax.set_title(
        "Denser neighbourhoods offer greater access to urban frontage",
        fontsize=10,
    )
    ax.legend(fontsize=7, title="Type")
    _add_footnote(fig, f"{_SRC_ACCESS}  {_SRC_DENSITY}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig4_accessibility_dividend.png", bbox_inches="tight")
    plt.close()
    print("  Saved fig4_accessibility_dividend.png")


# ---------------------------------------------------------------------------
# Figure 5: Trophic layers — per-category accessibility by housing type
# ---------------------------------------------------------------------------

# Each trophic layer: (column, short label, panel title)
# Uses _wt (gravity-weighted) columns at 800m pedestrian catchment.
# Falls back gracefully if school/health columns are not yet computed.
TROPHIC_LAYERS: list[tuple[str, str, str]] = [
    ("cc_density_800", "Street nodes", "Street network"),
    ("cc_fsa_restaurant_800_wt", "Restaurants (wt)", "Restaurants & cafes"),
    ("cc_fsa_pub_800_wt", "Pubs (wt)", "Pubs & bars"),
    ("cc_fsa_takeaway_800_wt", "Takeaways (wt)", "Takeaways"),
    ("cc_bus_800_wt", "Bus stops (wt)", "Bus"),
    ("cc_greenspace_800_wt", "Green spaces (wt)", "Green space"),
    ("cc_school_800_wt", "Schools (wt)", "Education"),
    ("cc_gp_practice_800_wt", "GP practices (wt)", "Health (GP)"),
    ("cc_pharmacy_800_wt", "Pharmacies (wt)", "Health (pharmacy)"),
]


def fig5_access_bar(lsoa: pd.DataFrame) -> None:
    """Small multiples: each trophic layer by housing type, plus energy cost."""
    # Filter to layers present in the data
    available = [
        (col, label, title)
        for col, label, title in TROPHIC_LAYERS
        if col in lsoa.columns
    ]
    if not available:
        print("  Skipping fig5 — no trophic layer columns found")
        return

    n_panels = len(available)
    # Arrange in rows of 3
    ncols = 3
    nrows = -(-n_panels // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    axes_flat = axes.flatten() if n_panels > 1 else [axes]

    for idx, (col, label, title) in enumerate(available):
        ax = axes_flat[idx]
        medians = []
        for t in TYPE_ORDER:
            vals = lsoa.loc[lsoa["dominant_type"] == t, col].dropna()
            medians.append(vals.median() if len(vals) > 0 else np.nan)

        x = np.arange(len(TYPE_ORDER))
        colors = [TYPE_COLORS[t] for t in TYPE_ORDER]
        ax.bar(x, medians, color=colors, edgecolor="black", linewidth=0.5)

        # IQR error bars
        for i, t in enumerate(TYPE_ORDER):
            vals = lsoa.loc[lsoa["dominant_type"] == t, col].dropna()
            if len(vals) > 0:
                q25, q75 = vals.quantile(0.25), vals.quantile(0.75)
                ax.errorbar(
                    i,
                    medians[i],
                    yerr=[[medians[i] - q25], [q75 - medians[i]]],
                    fmt="none",
                    color="black",
                    capsize=3,
                    linewidth=1,
                )

        # Detached/Flat ratio annotation
        m_flat = medians[0]  # Flat is first in TYPE_ORDER
        m_det = medians[-1]  # Detached is last
        if pd.notna(m_flat) and pd.notna(m_det) and m_flat > 0 and m_det > 0:
            ratio = m_flat / m_det if m_det > 0 else np.nan
            label_txt = f"Flat/Det = {ratio:.1f}x" if pd.notna(ratio) else ""
            if label_txt:
                ax.annotate(
                    label_txt,
                    xy=(0.5, 0.93),
                    xycoords="axes fraction",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                    color="#c0392b",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(TYPE_ORDER, fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8)

    # Hide unused axes
    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Trophic Layers: What Does the Energy Buy?",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    _add_footnote(
        fig,
        "Each panel shows median LSOA-level metric by dominant Census housing type (ts044). "
        "Error bars: IQR. Accessibility metrics: cityseer gravity-weighted counts "
        "within 800 m pedestrian catchment.",
    )
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig5_access_bar.png", bbox_inches="tight")
    plt.close()
    print("  Saved fig5_access_bar.png")


# ---------------------------------------------------------------------------
# Figure 6: Correlation heatmap
# ---------------------------------------------------------------------------


def fig6_correlation_heatmap(lsoa: pd.DataFrame) -> None:
    """Seaborn heatmap of key variables: form -> cost -> return."""
    cols = [
        "people_per_ha",
        "height_mean",
        "avg_hh_size",
        "building_kwh_per_hh",
        "cars_per_hh",
        "transport_kwh_per_hh_est",
        "total_kwh_per_hh",
        "cc_density_800",
        "fsa_count",
        "cc_bus_800_wt",
        "cc_rail_800_wt",
        "cc_greenspace_800_wt",
        "cc_school_800_wt",
        "cc_gp_practice_800_wt",
    ]

    present = [c for c in cols if c in lsoa.columns]
    corr = lsoa[present].corr()

    # Short labels
    short = {
        "people_per_ha": "Pop/ha",
        "height_mean": "Height",
        "avg_hh_size": "HH size",
        "building_kwh_per_hh": "Bldg kWh",
        "cars_per_hh": "Cars/hh",
        "transport_kwh_per_hh_est": "Trans kWh",
        "total_kwh_per_hh": "Total kWh",
        "cc_density_800": "Streets",
        "fsa_count": "FSA",
        "cc_bus_800_wt": "Bus",
        "cc_rail_800_wt": "Rail",
        "cc_greenspace_800_wt": "Green",
        "cc_school_800_wt": "Schools",
        "cc_gp_practice_800_wt": "GPs",
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

    _add_footnote(
        fig,
        f"{_SRC_DENSITY}  {_SRC_BLDG}  {_SRC_TRANSPORT}  {_SRC_ACCESS}",
    )
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig6_correlation_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  Saved fig6_correlation_heatmap.png")


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
    lsoa = compute_access_per_kwh(lsoa)

    print(f"\n{'=' * 70}")
    print(f"GENERATING FIGURES ({len(lsoa):,} LSOAs)")
    print("=" * 70)

    save_summary_tables(lsoa)
    fig1_building_energy(lsoa)
    fig2_mobility_penalty(lsoa)
    fig2b_private_public_transport(lsoa)
    fig3_density_transport_scatter(lsoa)
    fig4_accessibility_dividend(lsoa)
    fig5_access_bar(lsoa)
    fig6_correlation_heatmap(lsoa)

    print(f"\n{'=' * 70}")
    print(f"All outputs saved to: {FIGURE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    _cities = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(cities=_cities or None)
