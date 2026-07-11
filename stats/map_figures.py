"""
Map figures for the NEPI paper: the pattern is national and structural.

  F9 England — a choropleth of every English Output Area by energy spent per
     dwelling. Most of the land is dispersed, high-energy form; the compact,
     low-energy neighbourhoods are small bright islands (cities).
  F10 one city — the same city mapped twice, energy spent beside access on foot,
     so the core-to-edge flip is visible over a few kilometres.

Output Areas are equal-population, so rural areas are large on the map: the visual
weight of the high-energy dispersed form is itself part of the point. Colours come
from ``figstyle`` (the validated sequential ramp). Figures write to
``paper/figures/`` as PNG + PDF.

Reproduce:
    uv run python stats/oa_network_access.py   # for the access panel
    uv run python stats/map_figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import figstyle as fs  # noqa: E402
import geopandas as gpd  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import (  # noqa: E402
    BoundaryNorm,
    LinearSegmentedColormap,
    LogNorm,
)
from oa_data import load_and_aggregate  # noqa: E402

from urban_energy.paths import DATA_DIR  # noqa: E402

_STATS = DATA_DIR / "statistics"
_BOUND = _STATS / "oa_boundaries.gpkg"
_NET = _STATS / "oa_network_access.parquet"
# Energy uses the warm ramp, access the green ramp, so the two panels never read
# as one flipped blue scale (both run normally: darker = more of that quantity).
_WARM = LinearSegmentedColormap.from_list("warm", fs.WARM_SEQ)
_GREEN = LinearSegmentedColormap.from_list("green", fs.GREEN_SEQ)

# A city with a compact core and a clear sprawling edge (Sheffield), as a British
# National Grid bounding box (easting/northing metres), ~14 km across.
_CITY_NAME = "Sheffield"
_CITY_BBOX = (428_000, 380_000, 446_000, 396_000)

# Cities to label on the national map (name, easting, northing).
_CITIES = [
    ("London", 530_000, 180_000),
    ("Birmingham", 407_000, 287_000),
    ("Manchester", 384_000, 398_000),
    ("Leeds", 430_000, 434_000),
    ("Newcastle", 424_000, 565_000),
    ("Bristol", 359_000, 173_000),
]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _measures() -> pd.DataFrame:
    """Per-OA energy per dwelling and on-foot amenity count."""
    df = load_and_aggregate()
    df["energy"] = _num(df["building_kwh_per_hh"]) + _num(
        df["transport_kwh_per_hh_total_est"]
    )
    net = pd.read_parquet(_NET, columns=["net_total_1600"])
    df = df.merge(
        net.rename(columns={"net_total_1600": "access"}),
        left_on="OA21CD",
        right_index=True,
        how="left",
        validate="m:1",
    )
    return df[["OA21CD", "energy", "access"]]


def _colourbar(fig, ax, cmap, norm, label: str, compact: bool = False) -> None:
    """Add a slim horizontal colourbar under a map axis."""
    sm = ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8, colors=fs.INK_SECONDARY)
    if compact:  # thousands as "18k" so the ticks cannot crowd each other
        cb.ax.xaxis.set_major_formatter(lambda v, _p: f"{v / 1000:.0f}k")
    cb.set_label(label, fontsize=8.5, color=fs.INK_SECONDARY)


def _quantile_norm(values: pd.Series, n_classes: int = 5) -> BoundaryNorm:
    """Class the ramp on quantiles so within-country structure is visible
    (a linear ramp compresses most areas into one mid tone)."""
    qs = np.linspace(0.05, 0.95, n_classes + 1)
    bounds = np.unique(_num(values).quantile(qs).to_numpy())
    return BoundaryNorm(bounds, ncolors=256)


def _scale_bar(ax, x0: float, y0: float, length_m: float, label: str) -> None:
    """Draw a simple map scale bar (a rule with a centred length label)."""
    ax.plot(
        [x0, x0 + length_m],
        [y0, y0],
        color=fs.INK,
        lw=2.0,
        solid_capstyle="butt",
        zorder=9,
    )
    ax.text(
        x0 + length_m / 2,
        y0 + length_m * 0.14,
        label,
        ha="center",
        va="bottom",
        fontsize=7,
        color=fs.INK_SECONDARY,
        zorder=9,
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    )


def national(gdf: gpd.GeoDataFrame) -> None:
    """F9: England mapped twice — energy per dwelling beside access on foot.

    The two axes at national scale in one display item: dispersed high-energy
    form covers most of the land (left, warm ramp), and the high-access places
    are the compact cities, visible as green islands (right, log-scaled green
    ramp — walkable access spans orders of magnitude).
    """
    halo = [pe.withStroke(linewidth=2.4, foreground="white")]
    # Built at print width so fonts render at their designed sizes.
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 5.6))
    fig.subplots_adjust(top=0.80, bottom=0.10, left=0.01, right=0.99, wspace=0.02)

    e_norm = _quantile_norm(gdf["energy"])
    gdf.plot(
        column="energy",
        ax=axes[0],
        cmap=_WARM,
        norm=e_norm,
        linewidth=0,
        antialiased=False,
    )
    axes[0].set_title(
        "Energy spent", fontsize=11, color=fs.HEAT, fontweight="bold", loc="left"
    )
    _colourbar(
        fig,
        axes[0],
        _WARM,
        e_norm,
        "kWh / dwelling / yr, quantile classes  (darker = more)",
        compact=True,
    )

    # Walkable access spans orders of magnitude (0 in deep rural, thousands in
    # inner cities): a log ramp keeps both ends legible. Zeros clip to the floor.
    acc = pd.to_numeric(gdf["access"], errors="coerce").fillna(0).clip(lower=1)
    a_lo = max(float(acc.quantile(0.05)), 1.0)
    a_hi = float(acc.quantile(0.95))
    a_norm = LogNorm(vmin=a_lo, vmax=a_hi)
    gdf.assign(_acc=acc).plot(
        column="_acc",
        ax=axes[1],
        cmap=_GREEN,
        norm=a_norm,
        linewidth=0,
        antialiased=False,
    )
    axes[1].set_title(
        "Access on foot", fontsize=11, color=fs.ACCESS, fontweight="bold", loc="left"
    )
    _colourbar(
        fig, axes[1], _GREEN, a_norm, "amenities ≤1.6 km, log scale  (darker = more)"
    )

    for ax in axes:
        ax.set_facecolor("#f3f3f1")
        ax.set_axis_off()
        ax.set_aspect("equal")
        for name, e, n in _CITIES:
            ax.scatter(e, n, s=7, color=fs.INK, zorder=5)
            ax.annotate(
                name,
                (e, n),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=7,
                color=fs.INK,
                fontweight="bold",
                zorder=6,
                path_effects=halo,
            )
    _scale_bar(axes[0], 90_000, 25_000, 100_000, "100 km")

    # Greater London inset on each panel: at national scale the compact cores
    # compress to a few pixels, so the sharpest contrast is unreadable. The
    # inset repeats the same colour scale, placed over the Irish Sea corner.
    lon = (503_000, 155_000, 563_000, 200_000)
    panels = [
        (axes[0], gdf, "energy", _WARM, e_norm),
        (axes[1], gdf.assign(_acc=acc), "_acc", _GREEN, a_norm),
    ]
    for ax, frame, col, cmap, norm in panels:
        axins = ax.inset_axes([0.01, 0.63, 0.34, 0.33])
        clip = frame.cx[lon[0] : lon[2], lon[1] : lon[3]]
        clip.plot(
            column=col, ax=axins, cmap=cmap, norm=norm, linewidth=0, antialiased=False
        )
        axins.set_xlim(lon[0], lon[2])
        axins.set_ylim(lon[1], lon[3])
        axins.set_aspect("equal")
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_facecolor("#f3f3f1")
        for spine in axins.spines.values():
            spine.set_edgecolor(fs.INK)
            spine.set_linewidth(0.7)
        axins.annotate(
            "London",
            (0.05, 0.86),
            xycoords="axes fraction",
            fontsize=7,
            fontweight="bold",
            color=fs.INK,
            path_effects=halo,
        )
        ax.indicate_inset(
            bounds=(lon[0], lon[1], lon[2] - lon[0], lon[3] - lon[1]),
            inset_ax=None,
            edgecolor=fs.INK,
            linewidth=0.8,
        )

    fig.text(
        0.02,
        0.955,
        "THE TWO AXES, MAPPED",
        fontsize=8.5,
        fontweight="bold",
        color=fs.ACCENT,
    )
    fig.text(
        0.02,
        0.91,
        "Where energy runs high, access runs low",
        fontsize=13,
        fontweight="bold",
        color=fs.INK,
    )
    fs.footer(fig)
    print(f"  F9 england (two-axis): {len(gdf):,} OAs")
    fs.save(fig, "fig9_england", pdf=False)


def city(gdf: gpd.GeoDataFrame) -> None:
    """F10: one city, energy spent beside access on foot (the core-to-edge flip)."""
    x0, y0, x1, y1 = _CITY_BBOX
    sub = gdf.cx[x0:x1, y0:y1]
    fig, axes = plt.subplots(1, 2, figsize=(fs.COL2 * 1.15, 5.4))
    # bottom margin keeps the two colourbars clear of the source footer
    fig.subplots_adjust(top=0.8, bottom=0.12, left=0.02, right=0.98, wspace=0.06)

    e_norm = _quantile_norm(sub["energy"])
    sub.plot(
        column="energy",
        ax=axes[0],
        cmap=_WARM,
        norm=e_norm,
        linewidth=0,
        antialiased=False,
    )
    axes[0].set_title("Energy spent", fontsize=10.5, color=fs.HEAT, fontweight="bold")
    _colourbar(
        fig,
        axes[0],
        _WARM,
        e_norm,
        "kWh / dwelling / yr, quantile classes  (darker = more)",
        compact=True,
    )

    # Log ramp to match the national access panel (one convention set-wide).
    acc = _num(sub["access"]).fillna(0).clip(lower=1)
    a_lo = max(float(acc.quantile(0.05)), 1.0)
    a_norm = LogNorm(vmin=a_lo, vmax=float(acc.quantile(0.95)))
    sub.assign(_acc=acc).plot(
        column="_acc",
        ax=axes[1],
        cmap=_GREEN,
        norm=a_norm,
        linewidth=0,
        antialiased=False,
    )
    axes[1].set_title(
        "Access on foot", fontsize=10.5, color=fs.ACCESS, fontweight="bold"
    )
    _colourbar(
        fig, axes[1], _GREEN, a_norm, "amenities ≤1.6 km, log scale  (darker = more)"
    )

    halo = [pe.withStroke(linewidth=2, foreground="white")]
    centre = ((x0 + x1) / 2, (y0 + y1) / 2 + 500)
    for ax in axes:
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.scatter(*centre, s=8, color=fs.INK, zorder=5)
        ax.annotate(
            "city centre",
            centre,
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=7,
            color=fs.INK,
            fontweight="bold",
            zorder=6,
            path_effects=halo,
        )
    _scale_bar(axes[0], x0 + 500, y0 + 500, 5_000, "5 km")
    fig.text(
        0.02, 0.95, "INSIDE ONE CITY", fontsize=8.5, fontweight="bold", color=fs.ACCENT
    )
    fig.text(
        0.02,
        0.9,
        f"{_CITY_NAME}: the compact core spends less and reaches more",
        fontsize=12.5,
        fontweight="bold",
        color=fs.INK,
    )
    fs.footer(fig)
    print(f"  F10 city: {len(sub):,} OAs in {_CITY_NAME}")
    fs.save(fig, "fig10_city", pdf=False)


def main() -> None:
    """Build the national and city maps into paper/figures/."""
    fs.apply_style()
    gdf = gpd.read_file(_BOUND, columns=["OA21CD", "geometry"]).to_crs(27700)
    gdf = gdf.merge(_measures(), on="OA21CD", how="inner", validate="1:1")
    national(gdf)
    city(gdf)
    print(f"\n  → {fs.FIG_DIR}")


if __name__ == "__main__":
    main()
