"""
Argument figures for the NEPI paper and summary.md.

The set carries the argument on its own. Read in order it is the paper (the full
story scaffold is in ``paper/figure_design.md``):

  F1 inversion   — detached spends more energy and reaches less (the hook).
  F2 country     — the same pattern across all 178,353 Output Areas.
  F3 energy      — the energy axis, ~2.1×, heat vs car travel.
  F4 decomposition — how much of the heat gap is the form itself (~17%).
  F5 doorstep    — what a flat reaches on foot that a detached does not.
  F6 reach       — access against distance, widest on foot.
  F7 rate        — same reach, about a third of the fuel (~3.9× per kWh).
  F8 scenarios   — no decarbonisation lever closes much of the gap; access is fixed.
  F11 forest     — every headline ratio with its 95% CI on one log ruler.

Maps (F9 England, F10 one city) are built by ``stats/map_figures.py``.

Every chart shows compositional (method-D) pure-type predictions at mean confounds
unless noted (F2 and F5 are the raw per-OA distribution), so the plotted number
matches summary.md. Palette and rcParams come from ``figstyle`` (both palettes
validated). Figures write to ``paper/figures/`` as 300-dpi PNG + vector PDF.

Reproduce (build the network cache first):
    uv run python stats/oa_network_access.py
    uv run python stats/argument_figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import os  # noqa: E402
from typing import Any  # noqa: E402

import figstyle as fs  # noqa: E402
import ledger  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from access_profile import _comp_poisson  # noqa: E402
from form_size_decomposition import (  # noqa: E402
    _SHARE_FRACS,
    _comp_ols,
    _compositional_frame,
    _deprivation_cols,
    _hdd_cols,
    _imd_income_col,
    _tenure_cols,
)
from inference import CLUSTER_COL, log_contrast_ci  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from oa_data import load_and_aggregate  # noqa: E402
from scenarios import SCENARIOS, scenario_energy  # noqa: E402

from urban_energy.paths import DATA_DIR  # noqa: E402

_NET_CACHE = DATA_DIR / "statistics" / "oa_network_access.parquet"
_TYPES = fs.DWELLING_ORDER
_DENSITY = LinearSegmentedColormap.from_list("density", ["#eaf1fb", *fs.SEQUENTIAL])

# On-foot services (network count within 1,600 m), for the doorstep figure.
# Hospitals are omitted: the NHS ETS layer counts all trust sites (wards, clinics,
# community units), not hospitals, so its count is not credible on the doorstep.
# Health is represented by GP surgeries and pharmacies, both correctly counted.
_SERVICES = [
    ("GP surgeries", "net_gp_1600"),
    ("Pharmacies", "net_pharmacy_1600"),
    ("Schools", "net_school_1600"),
    ("Food outlets", "net_food_1600"),
    ("Food shops", "net_grocery_1600"),
    ("Parks & greenspace", "net_greenspace_1600"),
]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _pure_preds(model: Any, frame: pd.DataFrame, confounds: list[str]) -> dict:
    """Predicted level for each pure dwelling type at mean confounds (log link)."""
    base = sum(float(model.params[c]) * _num(frame[c]).mean() for c in confounds)
    return {t: float(np.exp(model.params[f"s_{t.lower()}"] + base)) for t in _TYPES}


# ---------------------------------------------------------------------------
# Act 1 — the wrong question, the right question
# ---------------------------------------------------------------------------


def inversion(cf: pd.DataFrame, confounds: list[str], income: list[str]) -> None:
    """F1: the two axes cross — more energy spent, less access reached."""
    cf = cf.copy()
    heat_kwh = _num(cf["building_kwh_per_hh"])
    trav_kwh = _num(cf["transport_kwh_per_hh_total_est"])
    cf["_tot"] = np.log((heat_kwh + trav_kwh).clip(lower=1))
    energy = _pure_preds(
        _comp_ols(cf, "_tot", _SHARE_FRACS + confounds, "total_hh"), cf, confounds
    )
    cf["_amen"] = _num(cf["net_total_1600"])
    access = _pure_preds(
        _comp_poisson(cf, "_amen", _SHARE_FRACS + income, "total_hh"), cf, income
    )

    # X-Y plot of the four principal types: energy on x, walkable access on a
    # log y. The morphological path runs from the top left (flat: low energy,
    # high access) to the bottom right (detached), with no constructed scaling.
    fig, ax = plt.subplots(figsize=(fs.COL2, 4.6))
    # Explicit bottom margin: the source footer must clear the x-axis title.
    fig.subplots_adjust(bottom=0.17, top=0.78, left=0.11, right=0.96)
    xs = [energy[t] for t in _TYPES]
    ys = [access[t] for t in _TYPES]
    ax.plot(xs, ys, color=fs.BASELINE, lw=1.4, zorder=1)
    for t in _TYPES:
        c = fs.DWELLING[t]
        ax.scatter(energy[t], access[t], s=110, color=c, zorder=3)
        ax.annotate(
            t,
            xy=(energy[t], access[t]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            color=c,
        )
        ax.annotate(
            f"{energy[t]:,.0f} kWh · {access[t]:,.0f} amenities",
            xy=(energy[t], access[t]),
            xytext=(10, -2),
            textcoords="offset points",
            fontsize=8,
            color=fs.INK_SECONDARY,
        )
    e_ratio = energy["Detached"] / energy["Flat"]
    a_ratio = access["Flat"] / access["Detached"]
    ax.set_yscale("log")
    ax.set_ylim(min(ys) * 0.45, max(ys) * 2.1)
    ax.margins(x=0.16)
    ax.set_xlabel("Total energy per dwelling (kWh / yr)")
    ax.set_ylabel("Everyday amenities within a 1.6 km walk (log scale)")
    fs.comma(ax, "x")
    fs.deck(
        ax,
        "Energy against access",
        "A detached neighbourhood spends more energy and reaches less",
    )
    fs.footer(fig)
    print(f"  F1 inversion: energy {e_ratio:.2f}× · access {a_ratio:.0f}×")
    fs.save(fig, "fig1_inversion")


def national_scatter(cf: pd.DataFrame) -> None:
    """F2: every Output Area, energy spent vs access reached — the whole country."""
    d = cf.copy()
    energy = _num(d["building_kwh_per_hh"]) + _num(d["transport_kwh_per_hh_total_est"])
    access = _num(d["net_total_1600"])
    dom = d["dominant_type"].astype("object")
    ok = energy.between(1, 60_000) & (access > 0)

    e_ok = energy[ok].to_numpy()
    a_ok = access[ok].to_numpy()

    ycap = float(np.quantile(a_ok, 0.985))  # linear y, clip the long right-skew tail
    fig, ax = plt.subplots(figsize=(fs.COL2, 5.4))
    # A single-int gridsize keeps the hexagons regular (a tuple stretches
    # them); face-coloured hairline edges stop same-colour cells merging
    # into irregular blobs.
    hb = ax.hexbin(
        e_ok,
        a_ok,
        gridsize=46,
        extent=(2000, 42000, 0, ycap),
        bins="log",
        cmap=_DENSITY,
        mincnt=1,
        linewidths=0.25,
        edgecolors="face",
    )
    # Binned-median trend, labelled at the right end in clear space (leader, not on it).
    bins = np.quantile(e_ok, np.linspace(0, 1, 19))
    mids, meds = [], []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        sel = (e_ok >= lo) & (e_ok < hi)
        if sel.sum() > 50:
            mids.append((lo + hi) / 2)
            meds.append(float(np.median(a_ok[sel])))
    ax.plot(mids, meds, color=fs.INK, lw=2.6, zorder=6, solid_capstyle="round")
    # Label above the line, anchored at the last binned point INSIDE the x-limit
    # (the top quantile bin's midpoint falls beyond it and would clip).
    anchor = len(mids) // 2  # mid-curve: clear of the Detached label at right
    ax.annotate(
        "median neighbourhood",
        xy=(mids[anchor], meds[anchor]),
        xytext=(-10, 18),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
        color=fs.INK,
        ha="right",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
        arrowprops=dict(arrowstyle="-", color=fs.INK, lw=0.8, shrinkB=4),
        zorder=7,
    )
    # Flat and Detached anchors, labelled with leaders into clear space, white halo.
    anchors = {"Flat": (10, 40), "Detached": (62, 34)}
    for t, off in anchors.items():
        m = ok & (dom == t)
        mx, my = float(energy[m].median()), float(access[m].median())
        ax.scatter(
            mx,
            my,
            s=180,
            facecolor=fs.DWELLING[t],
            edgecolor="white",
            linewidths=2.0,
            zorder=8,
        )
        ax.annotate(
            f"{t} areas",
            xy=(mx, my),
            xytext=off,
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color=fs.DWELLING[t],
            ha="center",
            zorder=9,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
            arrowprops=dict(arrowstyle="-", color=fs.DWELLING[t], lw=1.1, shrinkB=10),
        )
    ax.set_xlim(2000, 42000)
    ax.set_ylim(0, ycap)
    ax.set_xlabel("Energy spent (kWh / dwelling / year)")
    ax.set_ylabel("Amenities reachable on foot")
    fs.comma(ax, "x")
    cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=7.5, colors=fs.INK_SECONDARY)
    cb.set_label(
        "Output Areas per cell (log scale)", fontsize=8, color=fs.INK_SECONDARY
    )
    fs.deck(ax, "At national scale", "Access falls as energy rises")
    fs.footer(fig)
    print(f"  F2 country: {int(ok.sum()):,} OAs plotted")
    fs.save(fig, "fig2_country")


# ---------------------------------------------------------------------------
# Act 2 — the energy axis
# ---------------------------------------------------------------------------


def energy_gradient(cf: pd.DataFrame, confounds: list[str]) -> None:
    """F3: stacked heat + car travel per dwelling, flat to detached (~2.1×)."""
    cf = cf.copy()
    heat_kwh = _num(cf["building_kwh_per_hh"])
    trav_kwh = _num(cf["transport_kwh_per_hh_total_est"])
    cf["_heat"] = np.log(heat_kwh.clip(lower=1))
    cf["_trav"] = np.log(trav_kwh.clip(lower=1))
    cf["_tot"] = np.log((heat_kwh + trav_kwh).clip(lower=1))
    heat = _pure_preds(
        _comp_ols(cf, "_heat", _SHARE_FRACS + confounds, "total_hh"), cf, confounds
    )
    trav = _pure_preds(
        _comp_ols(cf, "_trav", _SHARE_FRACS + confounds, "total_hh"), cf, confounds
    )
    tot = _pure_preds(
        _comp_ols(cf, "_tot", _SHARE_FRACS + confounds, "total_hh"), cf, confounds
    )
    totals = [tot[t] for t in _TYPES]
    heat_seg = [tot[t] * heat[t] / (heat[t] + trav[t]) for t in _TYPES]
    trav_seg = [totals[i] - heat_seg[i] for i in range(len(_TYPES))]
    grad = totals[-1] / totals[0]

    fig, ax = plt.subplots(figsize=(fs.COL2, 4.5))
    x = np.arange(len(_TYPES))
    ax.bar(x, heat_seg, 0.62, label="Heat (metered gas + electricity)", color=fs.HEAT)
    ax.bar(
        x,
        trav_seg,
        0.62,
        bottom=heat_seg,
        label="Car travel (NTS-anchored)",
        color=fs.TRAVEL,
    )
    for i, t in enumerate(totals):
        ax.text(
            i,
            t + 500,
            f"{t:,.0f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9.5,
        )
    # The claim is the component slopes, so state them on the chart: how much
    # steeper each component rises from the flat bar to the detached bar.
    tratio = trav_seg[-1] / trav_seg[0]
    hratio = heat_seg[-1] / heat_seg[0]
    ax.annotate(
        f"car travel ×{tratio:.1f}",
        xy=(len(_TYPES) - 1 + 0.36, heat_seg[-1] + trav_seg[-1] / 2),
        fontsize=9,
        fontweight="bold",
        color=fs.TRAVEL,
        va="center",
        ha="left",
        annotation_clip=False,
    )
    ax.annotate(
        f"home energy ×{hratio:.1f}",
        xy=(len(_TYPES) - 1 + 0.36, heat_seg[-1] / 2),
        fontsize=9,
        fontweight="bold",
        color=fs.HEAT,
        va="center",
        ha="left",
        annotation_clip=False,
    )
    ax.set_xticks(x, _TYPES)
    ax.set_xlim(-0.6, len(_TYPES) + 0.7)
    ax.set_ylabel("Energy spent (kWh / dwelling / year)")
    fs.comma(ax, "y")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.margins(y=0.18)
    fs.deck(
        ax, "The energy axis", f"A detached home spends {grad:.1f}× a flat's energy"
    )
    fs.footer(fig)
    print(f"  F3 energy: {totals[0]:,.0f} → {totals[-1]:,.0f} ({grad:.2f}×)")
    fs.save(fig, "fig3_energy_gradient")


def decomposition(cf: pd.DataFrame, confounds: list[str]) -> None:
    """F4: how much of the heat gap is the form itself (1.60 → 1.27 → 1.17×)."""
    cf = cf.copy()
    cf["_lh"] = np.log(_num(cf["building_kwh_per_hh"]).clip(lower=1))
    cf["log_hh_size"] = np.log(_num(cf["avg_hh_size"]).clip(lower=1))
    cf["log_floor"] = np.log(_num(cf["oa_median_floor_area_m2"]).clip(lower=1))
    steps = [
        ("Raw heat gap", _SHARE_FRACS + confounds),
        ("Same family size", _SHARE_FRACS + confounds + ["log_hh_size"]),
        (
            "Same floor area\n(the form alone)",
            _SHARE_FRACS + confounds + ["log_hh_size", "log_floor"],
        ),
    ]
    ratios = []
    for _, xcols in steps:
        m = _comp_ols(cf, "_lh", xcols, "total_hh")
        assert m is not None, "decomposition fit failed"
        ratios.append(float(np.exp(m.params["s_detached"] - m.params["s_flat"])))
    labels = [s[0] for s in steps]

    fig, ax = plt.subplots(figsize=(fs.COL2, 4.2))
    y = np.arange(len(labels))[::-1]
    prev: float | None = None
    for yi, r in zip(y, ratios, strict=True):
        is_form = yi == 0  # the last row: form held alone
        ax.barh(
            yi,
            r - 1.0,
            left=1.0,
            height=0.62,
            color=fs.DWELLING["Detached"] if is_form else fs.HEAT,
            zorder=3,
        )
        if prev is not None:
            # the slice this control removed, a faded ghost back to the row above
            ax.barh(
                yi, prev - r, left=r, height=0.62, color=fs.HEAT, alpha=0.16, zorder=2
            )
            ax.text(
                (r + prev) / 2,
                yi + 0.36,
                f"−{prev - r:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=fs.INK_SECONDARY,
            )
        ax.text(
            r + 0.012,
            yi,
            f"{r:.2f}×",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=fs.INK,
        )
        prev = r
    ax.axvline(1.0, color=fs.INK, lw=1.3, zorder=4)
    ax.set_yticks(y, labels)
    ax.set_xlim(1.0, ratios[0] + 0.13)
    ax.set_xlabel("Detached : Flat heat per dwelling")
    fig.subplots_adjust(left=0.24, right=0.96, top=0.80, bottom=0.17)
    fs.deck(
        ax,
        "Form and household size",
        f"At equal size and occupancy the home-energy gap falls to "
        f"{ratios[-1]:.2f}\u00d7",
    )
    fs.footer(fig)
    print(f"  F4 decomposition: {ratios[0]:.2f}× → {ratios[-1]:.2f}×")
    fs.save(fig, "fig4_decomposition")


# ---------------------------------------------------------------------------
# Act 3 — the access axis
# ---------------------------------------------------------------------------


def doorstep(cf: pd.DataFrame) -> None:
    """F5: what a flat reaches on foot that a detached does not, service by service."""
    d = cf.copy()
    rows = []
    for name, col in _SERVICES:
        x = _num(d[col])
        f = float(x[d["dominant_type"] == "Flat"].median())
        de = float(x[d["dominant_type"] == "Detached"].median())
        rows.append((name, f, de))
    rows.sort(key=lambda r: r[1])  # ascending, so the biggest gap sits at the top

    def _dx(v: float) -> float:
        return v  # symlog (linthresh=1) renders zero exactly at the axis origin

    fig, ax = plt.subplots(figsize=(fs.COL2, 4.7))
    fig.subplots_adjust(left=0.23, right=0.9, top=0.7, bottom=0.13)
    y = np.arange(len(rows))
    for yi, (_name, f, de) in enumerate(rows):
        ax.plot([_dx(de), f], [yi, yi], color=fs.BASELINE, lw=2.5, zorder=1)
        ax.scatter(_dx(de), yi, s=90, color=fs.DWELLING["Detached"], zorder=3)
        ax.scatter(f, yi, s=90, color=fs.ACCESS, zorder=3)
        ax.annotate(
            f"{f:.0f}",
            (f, yi),
            textcoords="offset points",
            xytext=(8, 0),
            va="center",
            ha="left",
            fontsize=8.5,
            color=fs.ACCESS,
            fontweight="bold",
        )
        # Zero-count dots sit on the axis; drop their value label below the
        # dot so it cannot collide with the row label at the plot edge.
        de_off = (6, -13) if de < 1 else (-8, 0)
        ax.annotate(
            f"{de:.0f}",
            (_dx(de), yi),
            textcoords="offset points",
            xytext=de_off,
            va="center",
            ha="left" if de < 1 else "right",
            fontsize=8.5,
            color=fs.DWELLING["Detached"],
            fontweight="bold",
        )
    # A legend avoids anchoring type labels to dots that move per row.
    handles = [
        plt.Line2D(
            [], [], marker="o", ls="", ms=8, color=fs.ACCESS, label="Flat areas"
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=8,
            color=fs.DWELLING["Detached"],
            label="Detached areas",
        ),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, handletextpad=0.3)
    # Mark where the axis switches from linear (below one) to logarithmic.
    ax.axvline(1.0, color=fs.GRID, lw=0.8, ls=":", zorder=0)
    # Context: the share of detached areas that reach no GP at all on foot.
    gp_idx = next((i for i, r in enumerate(rows) if r[0] == "GP surgeries"), None)
    if gp_idx is not None:
        z = (
            _num(d.loc[d["dominant_type"] == "Detached", "net_gp_1600"]).fillna(0) == 0
        ).mean() * 100
        ax.annotate(
            f"{z:.0f}% of detached areas reach no GP at all",
            (_dx(rows[gp_idx][2]), gp_idx),
            textcoords="offset points",
            xytext=(12, -14),
            ha="left",
            fontsize=7.5,
            color=fs.INK_SECONDARY,
        )
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.8)
    ax.set_xticks([0, 1, 10, 100])
    ax.set_xticklabels(["0", "1", "10", "100"])
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) + 0.35)
    ax.set_xlabel(
        "Reachable within a 1.6 km walk (median neighbourhood; log scale above 1)"
    )
    big = max(rows, key=lambda r: r[1])
    fs.deck(
        ax,
        "On the doorstep",
        f"A flat reaches {big[1]:.0f} {big[0].lower()} on foot; "
        f"a detached, {big[2]:.0f}",
    )
    fs.footer(fig)
    print(f"  F5 doorstep: {len(rows)} services")
    fs.save(fig, "fig5_doorstep")


def access_curve(cf: pd.DataFrame, income: list[str], dists: list[int]) -> None:
    """F6: amenities vs network distance, one slate line per type (~27× on foot)."""
    cf = cf.copy()
    series: dict[str, list[float]] = {t: [] for t in _TYPES}
    for dd in dists:
        cf["_y"] = _num(cf[f"net_total_{dd}"])
        p = _pure_preds(
            _comp_poisson(cf, "_y", _SHARE_FRACS + income, "total_hh"), cf, income
        )
        for t in _TYPES:
            series[t].append(p[t])
    km = np.array(dists) / 1000
    curves = {t: np.array(v) for t, v in series.items()}
    foot = curves["Flat"][0] / curves["Detached"][0]
    far = curves["Flat"][-1] / curves["Detached"][-1]

    fig, ax = plt.subplots(figsize=(fs.COL2, 4.6))
    _halo = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9)
    label_y = {"Terraced": 1.18, "Semi": 0.86}  # nudge the two middle labels apart
    for t in _TYPES:
        ax.plot(km, curves[t], color=fs.DWELLING[t], lw=2.4, label=t)
        ax.text(
            km[-1] + 0.5,
            curves[t][-1] * label_y.get(t, 1.0),
            t,
            color=fs.DWELLING[t],
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox=_halo,
        )
    # Name the protagonist where its curve starts.
    ax.text(
        km[0] + 0.2,
        curves["Flat"][0] * 1.3,
        "Flat",
        color=fs.DWELLING["Flat"],
        fontsize=9.5,
        fontweight="bold",
        bbox=_halo,
    )
    ax.set_yscale("log")
    ax.axvline(1.6, color=fs.MUTED, lw=0.9, ls=":", zorder=1)
    ax.axvline(km[-1], color=fs.MUTED, lw=0.9, ls=":", zorder=1)
    _thr = dict(
        ha="center",
        fontsize=8,
        color=fs.INK_SECONDARY,
        fontweight="bold",
        linespacing=1.3,
        transform=ax.get_xaxis_transform(),
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
        zorder=6,
    )
    ax.text(1.6, 0.98, "on foot\n(a 20-minute walk)", va="top", **_thr)
    # Bottom-right corner is clear of every curve at the drive threshold.
    ax.text(
        km[-1], 0.03, "25.6 km drive\n(a typical rural commute)", va="bottom", **_thr
    )
    # No in-plot multiples: the deck headline and the caption carry the 27x
    # and 11x figures; brackets over converging log curves read awkwardly.
    ax.set_xlabel("Network distance reachable (km)")
    ax.set_ylabel("Everyday amenities reachable (log scale)")
    ax.set_xlim(km[0] - 0.5, km[-1] + 5.6)
    ax.set_ylim(None, curves["Flat"][-1] * 1.8)  # headroom for the top label
    # Keep the x-axis label clear of the figure-level source footer.
    fig.subplots_adjust(bottom=0.15)
    fs.deck(
        ax,
        "Reach and distance",
        f"A flat reaches ≈{foot:.0f}× more on foot, ≈{far:.0f}× at a 25.6 km drive",
    )
    fs.footer(fig)
    print(f"  F6 reach: ≈{foot:.0f}× on foot → ≈{far:.0f}× at 25 km")
    fs.save(fig, "fig6_access_curve")


# ---------------------------------------------------------------------------
# Act 4 — the rate
# ---------------------------------------------------------------------------


def rate(cf: pd.DataFrame, income: list[str], confounds: list[str]) -> None:
    """F7: the rate SHOWN as its construction — reach ÷ energy → access per kWh.

    Three panels: catchment amenities (nearly equal across types), car-travel
    energy (three times higher for detached), and the resulting rate, with the
    cluster-bootstrap interval read back from the ledger. The paper's sentence
    ("the rate is by construction the product of the two quantities") becomes
    visible instead of asserted.
    """
    cf = cf.copy()
    cf["_amen"] = _num(cf["net_amen"])
    amen = _pure_preds(
        _comp_poisson(cf, "_amen", _SHARE_FRACS + income, "total_hh"), cf, income
    )
    cf["_le"] = np.log(_num(cf["transport_kwh_per_hh_total_est"]).clip(lower=1))
    energy = _pure_preds(
        _comp_ols(cf, "_le", _SHARE_FRACS + confounds, "total_hh"), cf, confounds
    )
    vals = [amen[t] / energy[t] for t in _TYPES]
    ratio = vals[0] / vals[-1] if vals[-1] else float("nan")
    catch = amen["Flat"] / amen["Detached"]
    esave = energy["Detached"] / energy["Flat"]
    rate_ci = ledger.value("rateCI").replace("--", ", ")

    fig, axes = plt.subplots(1, 3, figsize=(fs.COL2, 3.8))
    fig.subplots_adjust(
        top=0.66, bottom=0.16, left=0.09, right=0.985, wspace=0.55
    )
    x = np.arange(len(_TYPES))
    panels = [
        (
            axes[0],
            [amen[t] for t in _TYPES],
            "Amenities at own\ncar catchment",
            f"nearly equal ({catch:.2f}×)",
            fs.ACCESS,
        ),
        (
            axes[1],
            [energy[t] for t in _TYPES],
            "Car-travel energy\n(kWh / dwelling / yr)",
            f"detached : flat {esave:.1f}×",
            fs.TRAVEL,
        ),
        (
            axes[2],
            vals,
            "Access per kWh\nof car travel",
            f"flat : detached {ratio:.1f}×"
            + (f"\n[{rate_ci}]" if rate_ci else ""),
            fs.ACCESS,
        ),
    ]
    for ax, v, title, note, accent in panels:
        ax.bar(x, v, 0.6, color=[fs.DWELLING[t] for t in _TYPES], zorder=2)
        ax.set_title(
            title, loc="left", fontsize=8.5, fontweight="bold", color=accent, pad=6
        )
        ax.set_xticks(x, ["F", "T", "S", "D"], fontsize=8)
        ax.margins(y=0.24)
        ax.tick_params(axis="y", labelsize=7)
        ax.annotate(
            note,
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            xytext=(0, -6),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color=accent,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
            zorder=5,
        )
        if v is not vals:
            fs.comma(ax, "y")
    fig.text(
        0.09,
        0.945,
        "THE RATE, CONSTRUCTED" if os.environ.get("NEPI_PLAIN_FIGS") != "1" else "",
        fontsize=8.5,
        fontweight="bold",
        color=fs.ACCENT,
    )
    if os.environ.get("NEPI_PLAIN_FIGS") != "1":
        fig.text(
            0.09,
            0.885,
            f"Similar reach, a third of the energy: {ratio:.1f}× access per kWh",
            fontsize=13,
            fontweight="bold",
            color=fs.INK,
        )
        fig.text(
            0.09,
            0.835,
            "F flat · T terraced · S semi-detached · D detached",
            fontsize=9,
            color=fs.INK_SECONDARY,
        )
    fs.footer(fig)
    print(
        f"  F7 rate: Flat:Det {ratio:.1f}× "
        f"(catch {catch:.2f}× · energy {esave:.1f}×)"
    )
    fs.save(fig, "fig7_rate")
    fs.save(fig, "fig7_rate")


# ---------------------------------------------------------------------------
# Act 5 — locked in
# ---------------------------------------------------------------------------


def scenario_ladder(cf: pd.DataFrame, confounds: list[str]) -> None:
    """F8: the flat→detached energy gap under each decarbonisation lever."""
    cf = cf.copy()
    hh = _num(cf["total_hh"])
    gas = (_num(cf["oa_gas_mean_kwh"]) * _num(cf["oa_gas_num_meters"])).fillna(0) / hh
    elec = (_num(cf["oa_elec_mean_kwh"]) * _num(cf["oa_elec_num_meters"])).fillna(
        0
    ) / hh
    travel = _num(cf["transport_kwh_per_hh_total_est"])

    pretty = {
        "S0 status quo": "Status quo",
        "Fabric only (100%)": "Insulation only",
        "Heat pumps only (100%)": "Heat pumps only",
        "EVs only (100%)": "Electric vehicles only",
        "CCC Balanced Pathway 2040": "CCC pathway (2040)",
        "Full rollout (100%)": "Full rollout of all three",
    }
    cf["log_hh_size"] = np.log(_num(cf["avg_hh_size"]).clip(lower=1))
    labels, ratios, fam_ratios = [], [], []
    for label, heat_transform, u_heat, u_ev in SCENARIOS:
        heat, trav = scenario_energy(
            cf, gas, elec, travel, heat_transform, u_heat, u_ev
        )
        cf["_lt"] = np.log((heat + trav).clip(lower=1).to_numpy())
        preds = _pure_preds(
            _comp_ols(cf, "_lt", _SHARE_FRACS + confounds, "total_hh"), cf, confounds
        )
        fam = _pure_preds(
            _comp_ols(
                cf, "_lt", _SHARE_FRACS + confounds + ["log_hh_size"], "total_hh"
            ),
            cf,
            confounds + ["log_hh_size"],
        )
        labels.append(pretty.get(label, label))
        ratios.append(preds["Detached"] / preds["Flat"])
        fam_ratios.append(fam["Detached"] / fam["Flat"])
    status_quo = ratios[0]

    # The frozen access gap: on-foot amenities, compositional flat:detached.
    cf["_amen"] = _num(cf["net_total_1600"])
    ma = _comp_poisson(cf, "_amen", _SHARE_FRACS + _imd_income_col(cf), "total_hh")
    assert ma is not None, "access fit failed"
    access_ratio = float(np.exp(ma.params["s_flat"] - ma.params["s_detached"]))

    fig, (ax_e, ax_a) = plt.subplots(
        1,
        2,
        figsize=(fs.COL2, 5.0),
        gridspec_kw={"width_ratios": [3.1, 1.0], "wspace": 0.12},
        sharey=True,
    )
    fig.subplots_adjust(top=0.72, bottom=0.12, left=0.19, right=0.97)
    y = np.arange(len(labels))[::-1]

    # Left: the energy gap, each lever a bar from parity (1.0) to its ratio.
    colours = [fs.CLOSES if r <= status_quo + 1e-9 else fs.WIDENS for r in ratios]
    colours[0] = fs.NEUTRAL
    ax_e.barh(
        y, [r - 1.0 for r in ratios], left=1.0, height=0.62, color=colours, zorder=2
    )
    # The equal-household-size companion of each scenario ratio, as an open
    # marker on the same row, so both views appear together.
    ax_e.scatter(
        fam_ratios,
        y,
        s=42,
        facecolor="white",
        edgecolor=fs.INK,
        linewidths=1.1,
        zorder=4,
        label="equal household size",
    )
    ax_e.legend(loc="lower right", fontsize=7.5, frameon=False)
    ax_e.axvline(status_quo, color=fs.NEUTRAL, lw=1.0, ls=(0, (4, 3)), zorder=1)
    for yi, r in enumerate(ratios):
        closed = 1 - np.log(r) / np.log(status_quo) if status_quo > 1 and r > 1 else 0.0
        if yi == 0:
            note = ""
        elif closed >= 0:
            note = f"  ({closed:.0%} closed)"
        else:
            note = f"  ({-closed:.0%} wider)"
        ax_e.text(
            r + 0.02, y[yi], f"{r:.2f}×{note}", va="center", fontsize=8.8, color=fs.INK
        )
    ax_e.set_yticks(y, labels)
    ax_e.tick_params(axis="y", labelcolor=fs.INK)
    ax_e.set_xlim(1.0, max(ratios) + 0.5)  # label space right of the longest bar
    ax_e.set_ylim(-0.6, len(labels) - 1 + 1.05)
    ax_e.set_xlabel("Detached : Flat energy per dwelling")
    ax_e.set_title(
        "Energy gap: narrows a little",
        loc="left",
        fontsize=9.5,
        color=fs.HEAT,
        fontweight="bold",
        pad=8,
    )
    # "today" rides the top of the dashed status-quo line, inside the axes,
    # where it cannot collide with the x-tick labels.
    ax_e.text(
        status_quo,
        len(labels) - 1 + 0.62,
        "today",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=fs.INK_SECONDARY,
    )

    # Right: the access gap, identical in every scenario. Equal-length bars
    # would imply a measured scale, so each row simply states the number.
    for yi in y:
        ax_a.text(
            0.5,
            yi,
            f"{access_ratio:.0f}×",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=fs.ACCESS,
            zorder=3,
        )
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(-0.6, len(labels) - 1 + 1.05)
    ax_a.set_xticks([])
    ax_a.set_title(
        f"Access gap: {access_ratio:.0f}×, frozen",
        loc="left",
        fontsize=9.5,
        color=fs.ACCESS,
        fontweight="bold",
        pad=8,
    )
    ax_a.set_xlabel("")
    ax_a.spines[["bottom", "left"]].set_visible(False)
    ax_a.tick_params(left=False)

    # Title block at figure level, clear of both panels.
    fig.text(0.19, 0.94, "LOCKED IN", fontsize=8.5, fontweight="bold", color=fs.ACCENT)
    fig.text(
        0.19,
        0.885,
        "Technology narrows the energy gap only part way",
        fontsize=13.5,
        fontweight="bold",
        color=fs.INK,
    )
    fs.footer(fig)
    print(
        f"  F8 scenarios: energy {status_quo:.2f}×→{ratios[-1]:.2f}× · "
        f"access {access_ratio:.0f}× frozen"
    )
    fs.save(fig, "fig8_scenarios")


def forest(cf: pd.DataFrame, confounds: list[str], income: list[str]) -> None:
    """F11: every headline compositional ratio with its clustered 95% CI.

    Two panels on separate log rulers: the energy gaps (Detached:Flat, log-OLS)
    above, the access gaps (Flat:Detached, Poisson) below. Each interval is the
    single-model delta-method CI with LAD-clustered covariance — the same
    numbers as the text tables. The access-per-kWh rate joins the access panel
    from the ledger (diamond marker, cluster-bootstrap interval).
    """
    cf = cf.copy()
    heat_kwh = _num(cf["building_kwh_per_hh"])
    trav_kwh = _num(cf["transport_kwh_per_hh_total_est"])
    cf["_lh"] = np.log(heat_kwh.clip(lower=1))
    cf["_le"] = np.log(trav_kwh.clip(lower=1))
    cf["_lt"] = np.log((heat_kwh + trav_kwh).clip(lower=1))
    cf["log_hh_size"] = np.log(_num(cf["avg_hh_size"]).clip(lower=1))
    cf["log_floor"] = np.log(_num(cf["oa_median_floor_area_m2"]).clip(lower=1))

    def _ols_ci(y_col: str, x_cols: list[str]) -> tuple | None:
        m = _comp_ols(cf, y_col, x_cols, "total_hh", cluster_col=CLUSTER_COL)
        return None if m is None else log_contrast_ci(m, "s_detached", "s_flat")

    def _pois_ci(col: str) -> tuple | None:
        cf["_y"] = _num(cf[col])
        m = _comp_poisson(
            cf, "_y", _SHARE_FRACS + income, "total_hh", cluster_col=CLUSTER_COL
        )
        return None if m is None else log_contrast_ci(m, "s_flat", "s_detached")

    base = _SHARE_FRACS + confounds
    groups: list[tuple[str, str, list[tuple[str, tuple | None]]]] = [
        (
            "ENERGY — Detached : Flat, per dwelling",
            fs.HEAT,
            [
                ("Total energy", _ols_ci("_lt", base)),
                (
                    "Total energy,\nequal household size",
                    _ols_ci("_lt", base + ["log_hh_size"]),
                ),
                ("Car travel", _ols_ci("_le", base)),
                (
                    "Car travel,\nequal household size",
                    _ols_ci("_le", base + ["log_hh_size"]),
                ),
                ("Home energy", _ols_ci("_lh", base)),
                (
                    "Home energy,\nequal household size",
                    _ols_ci("_lh", base + ["log_hh_size"]),
                ),
                (
                    "Home energy,\nequal floor area too",
                    _ols_ci("_lh", base + ["log_hh_size", "log_floor"]),
                ),
            ],
        ),
        (
            "ACCESS — Flat : Detached, within reach",
            fs.ACCESS,
            [
                ("Amenities on foot", _pois_ci("net_total_1600")),
                ("Jobs on foot", _pois_ci("net_jobs_1600")),
                ("People on foot", _pois_ci("net_pop_1600")),
                ("Amenities, own catchment", _pois_ci("net_amen")),
                ("Amenities, 25.6 km drive", _pois_ci("net_total_25600")),
            ],
        ),
    ]

    # The rate joins the access panel from the ledger (cluster-bootstrap CI),
    # so the figure genuinely holds ALL headline ratios.
    try:
        r_pt = float(ledger.value("rate"))
        r_lo, r_hi = (float(v) for v in ledger.value("rateCI").split("--"))
        groups[1][2].append(("Access per kWh (rate)", (r_pt, r_lo, r_hi)))
    except (ValueError, IndexError):
        pass

    def _fmt(v: float) -> str:
        return f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"

    # Two panels: the energy ratios sit between 1x and ~3.5x, the access ratios
    # reach ~50x, so a shared ruler wastes most of its width on one group.
    # Each panel gets its own log axis spanning only its data.
    fig, axes = plt.subplots(
        2, 1, figsize=(fs.COL2, 6.4), gridspec_kw={"hspace": 0.42}
    )
    fig.subplots_adjust(left=0.30, right=0.80, top=0.80, bottom=0.09)
    ticks = {
        0: ([1, 1.5, 2, 3, 4], ["1×", "1.5×", "2×", "3×", "4×"]),
        1: (
            [1, 2, 5, 10, 20, 50, 100],
            ["1×", "2×", "5×", "10×", "20×", "50×", "100×"],
        ),
    }
    drawn = 0
    for i, (ax, (header, colour, rows)) in enumerate(
        zip(axes, groups, strict=True)
    ):
        ypos = 0.0
        yticks: list[float] = []
        ylabels: list[str] = []
        for label, ci in rows:
            if ci is None:
                continue
            point, lo, hi = ci[0], ci[1], ci[2]
            marker = "D" if "rate" in label else "o"
            ax.plot([lo, hi], [ypos, ypos], color=colour, lw=1.8, zorder=2)
            ax.plot(
                [lo, lo], [ypos - 0.14, ypos + 0.14], color=colour, lw=1.4, zorder=2
            )
            ax.plot(
                [hi, hi], [ypos - 0.14, ypos + 0.14], color=colour, lw=1.4, zorder=2
            )
            ax.scatter(point, ypos, s=42, color=colour, marker=marker, zorder=3)
            ax.annotate(
                f"{_fmt(point)}× [{_fmt(lo)}, {_fmt(hi)}]",
                xy=(1.02, ypos),
                xycoords=("axes fraction", "data"),
                fontsize=8,
                color=fs.INK_SECONDARY,
                va="center",
                ha="left",
            )
            yticks.append(ypos)
            ylabels.append(label)
            ypos -= 1.0
        drawn += len(yticks)
        ax.axvline(1.0, color=fs.INK, lw=1.1, zorder=1)
        ax.set_xscale("log")
        ax.set_xticks(ticks[i][0])
        ax.set_xticklabels(ticks[i][1])
        ax.minorticks_off()
        ax.set_yticks(yticks, ylabels)
        ax.set_ylim(ypos + 0.4, 0.6)
        ax.set_title(
            header, loc="left", fontsize=8, fontweight="bold", color=colour, pad=6
        )
        ax.set_xlabel("Ratio (log scale; 1× = no difference)", fontsize=8)
        ax.tick_params(left=False)
        ax.spines["left"].set_visible(False)
    axes[0].set_xlim(0.95, 4.2)
    axes[1].set_xlim(0.95, 110)
    # Figure-level title block: anchored to the figure, not the top axes, so it
    # can never overprint the first panel's header.
    if os.environ.get("NEPI_PLAIN_FIGS") != "1":
        fig.text(
            0.30, 0.965, "SUMMARY OF RESULTS",
            fontsize=8.5, fontweight="bold", color=fs.ACCENT,
        )
        fig.text(
            0.30, 0.925,
            "All headline ratios with their confidence intervals",
            fontsize=13, fontweight="bold", color=fs.INK,
        )
    fs.footer(fig)
    print(f"  F11 forest: {drawn} intervals drawn")
    fs.save(fig, "fig11_forest")


def equity(cf: pd.DataFrame) -> None:
    """F12 (Extended Data): the deprivation gradient of walkable access.

    Left panel: median on-foot amenities by IMD income-deprivation decile —
    nationally the most deprived areas hold the most access. Right panel: the
    rank correlation by stratum, showing the flattening and inversion in the
    strongest housing markets (inner London, Manchester, Bristol).
    """
    inc = _num(cf["imd_income_score"])
    amen = _num(cf["net_total_1600"])
    dec = pd.qcut(inc, 10, labels=False)
    med = [float(amen[dec == i].median()) for i in range(10)]

    lad = cf["LAD22CD"].astype(str)
    inner_london = {
        "E09000001", "E09000007", "E09000012", "E09000013", "E09000014",
        "E09000019", "E09000020", "E09000022", "E09000023", "E09000025",
        "E09000028", "E09000030", "E09000032", "E09000033",
    }  # fmt: skip
    strata = [
        ("England", pd.Series(True, index=cf.index)),
        ("Inner London", lad.isin(inner_london)),
        ("Manchester", lad == "E08000003"),
        ("Bristol", lad == "E06000023"),
    ]
    rhos = []
    for name, mask in strata:
        sub = pd.DataFrame({"a": amen[mask], "i": inc[mask]}).dropna()
        rhos.append((name, float(sub["a"].corr(sub["i"], "spearman"))))

    fig, (ax_d, ax_r) = plt.subplots(
        1, 2, figsize=(fs.COL2, 4.1), gridspec_kw={"width_ratios": [1.5, 1.0]}
    )
    fig.subplots_adjust(top=0.70, bottom=0.26, left=0.10, right=0.96, wspace=0.42)

    x = np.arange(10)
    ax_d.bar(x, med, 0.7, color=fs.ACCESS, zorder=2)
    ax_d.set_xticks([0, 4, 9], ["1\nleast\ndeprived", "5", "10\nmost\ndeprived"])
    ax_d.tick_params(axis="x", labelsize=7.5)
    ax_d.set_ylabel("Median amenities within 1.6 km")
    ax_d.set_xlabel("IMD income-deprivation decile")
    ax_d.set_title(
        "Access rises with deprivation, nationally",
        loc="left", fontsize=8.5, fontweight="bold", color=fs.ACCESS, pad=6,
    )

    yr = np.arange(len(rhos))[::-1]
    for yi, (_name, rho) in zip(yr, rhos, strict=True):
        colour = fs.ACCESS if rho > 0.05 else fs.WIDENS
        ax_r.plot([0, rho], [yi, yi], color=colour, lw=2.2, zorder=2)
        ax_r.scatter(rho, yi, s=55, color=colour, zorder=3)
        ax_r.annotate(
            f"{rho:+.2f}",
            (rho, yi),
            textcoords="offset points",
            xytext=(6 if rho >= 0 else -6, 6),
            ha="left" if rho >= 0 else "right",
            fontsize=8.5,
            fontweight="bold",
            color=colour,
        )
    ax_r.axvline(0, color=fs.INK, lw=1.0, zorder=1)
    ax_r.set_yticks(yr, [n for n, _ in rhos])
    ax_r.tick_params(left=False, axis="y", labelcolor=fs.INK)
    ax_r.spines["left"].set_visible(False)
    ax_r.set_xlim(-0.45, 0.62)
    ax_r.set_xlabel("Rank correlation, access vs deprivation")
    ax_r.set_title(
        "…but flattens or inverts in the\nstrongest housing markets",
        loc="left", fontsize=8.5, fontweight="bold", color=fs.WIDENS, pad=6,
    )
    if os.environ.get("NEPI_PLAIN_FIGS") != "1":
        fig.text(
            0.10, 0.955, "WHO HOLDS THE ACCESS",
            fontsize=8.5, fontweight="bold", color=fs.ACCENT,
        )
        fig.text(
            0.10, 0.90,
            "Deprived areas hold the walkable access, except where "
            "proximity is priced",
            fontsize=12.5, fontweight="bold", color=fs.INK,
        )
    fs.footer(fig)
    print(f"  F12 equity: national ρ {rhos[0][1]:+.2f} → Bristol {rhos[-1][1]:+.2f}")
    fs.save(fig, "fig12_equity")


def main() -> None:
    """Build the argument chart figures into paper/figures/ (method-D basis)."""
    fs.apply_style()
    df = load_and_aggregate()
    net = pd.read_parquet(_NET_CACHE)
    cf = _compositional_frame(
        df.merge(net, left_on="OA21CD", right_index=True, how="left", validate="m:1")
    )
    confounds = (
        ["median_build_year"] + _deprivation_cols(cf) + _tenure_cols(cf) + _hdd_cols(cf)
    )
    income = _imd_income_col(cf)
    dists = sorted(
        int(c.rsplit("_", 1)[1]) for c in net.columns if c.startswith("net_total_")
    )
    inversion(cf, confounds, income)
    national_scatter(cf)
    energy_gradient(cf, confounds)
    decomposition(cf, confounds)
    doorstep(cf)
    access_curve(cf, income, dists)
    rate(cf, income, confounds)
    scenario_ladder(cf, confounds)
    forest(cf, confounds, income)
    equity(cf)
    print(f"\n  → {fs.FIG_DIR}")


if __name__ == "__main__":
    main()
