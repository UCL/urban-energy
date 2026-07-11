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

from typing import Any  # noqa: E402

import figstyle as fs  # noqa: E402
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

    poles = ["Flat", "Detached"]
    emax = max(energy[t] for t in poles)
    amin, amax = min(access[t] for t in poles), max(access[t] for t in poles)

    def y_e(v: float) -> float:
        return 0.16 + 0.66 * v / emax

    def y_a(v: float) -> float:
        return 0.16 + 0.66 * (np.log(v) - np.log(amin)) / (np.log(amax) - np.log(amin))

    fig, ax = plt.subplots(figsize=(fs.COL2, 4.6))
    ax.axvline(0, color=fs.BASELINE, lw=1.0, ymax=0.92)
    ax.axvline(1, color=fs.BASELINE, lw=1.0, ymax=0.92)
    for t in poles:
        c = fs.DWELLING[t]
        ye, ya = y_e(energy[t]), y_a(access[t])
        ax.plot([0, 1], [ye, ya], color=c, lw=3.0, marker="o", ms=9, zorder=3)
        ax.text(
            -0.03,
            ye,
            f"{t}\n{energy[t]:,.0f} kWh",
            ha="right",
            va="center",
            color=fs.INK,
            fontsize=9.5,
            fontweight="bold",
            linespacing=1.4,
        )
        ax.text(
            1.03,
            ya,
            f"{access[t]:,.0f}\namenities",
            ha="left",
            va="center",
            color=fs.INK,
            fontsize=9.5,
            fontweight="bold",
            linespacing=1.4,
        )
    e_ratio = energy["Detached"] / energy["Flat"]
    a_ratio = access["Flat"] / access["Detached"]
    _hdr = dict(
        ha="center",
        va="bottom",
        fontsize=8,
        color=fs.MUTED,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", pad=2),
    )
    ax.text(0.0, 0.99, "ENERGY SPENT", **_hdr)
    ax.text(1.0, 0.99, "ACCESS ON FOOT", **_hdr)
    ax.text(
        0.5,
        0.05,
        f"Detached spends {e_ratio:.1f}× the energy to reach "
        f"1/{a_ratio:.0f} the amenities",
        ha="center",
        va="center",
        fontsize=11,
        color=fs.INK,
        fontweight="medium",
    )
    ax.set_xlim(-0.24, 1.18)
    ax.set_ylim(0, 1.04)
    ax.axis("off")
    fs.deck(
        ax,
        "The inversion",
        "A detached neighbourhood spends more energy and reaches less",
        "kWh per dwelling per year, and everyday amenities within a 1.6 km walk",
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
    hb = ax.hexbin(
        e_ok,
        a_ok,
        gridsize=(48, 40),
        extent=(2000, 42000, 0, ycap),
        bins="log",
        cmap=_DENSITY,
        mincnt=1,
        linewidths=0,
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
    cb.set_label("Output Areas per cell", fontsize=8, color=fs.INK_SECONDARY)
    fs.deck(
        ax,
        "At national scale",
        "More energy, less access, in 178,353 neighbourhoods",
        "Hexagonal bins over every Output Area; "
        "the median line falls as spending rises",
    )
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
        # Per-segment values, so the 1.6× heat / 3.1× travel split is legible
        # off the bars, not only from the totals.
        ax.text(
            i,
            heat_seg[i] / 2,
            f"{heat_seg[i]:,.0f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
        )
        ax.text(
            i,
            heat_seg[i] + trav_seg[i] / 2,
            f"{trav_seg[i]:,.0f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
        )
    ax.set_xticks(x, _TYPES)
    ax.set_ylabel("Energy spent (kWh / dwelling / year)")
    fs.comma(ax, "y")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.margins(y=0.18)
    fs.deck(
        ax,
        "The energy axis",
        f"A detached home spends {grad:.1f}× a flat's energy",
        "Car travel is the steeper part (3.1×), heat the rest (1.6×)",
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
        "Form versus family",
        f"At equal size and occupancy the heat gap falls to {ratios[-1]:.2f}\u00d7",
        "Bigger homes and larger households account for two-thirds of the log gap",
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
    fig.subplots_adjust(left=0.19, right=0.9, top=0.7, bottom=0.13)
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
        ax.annotate(
            f"{de:.0f}",
            (_dx(de), yi),
            textcoords="offset points",
            xytext=(-8, 0),
            va="center",
            ha="right",
            fontsize=8.5,
            color=fs.DWELLING["Detached"],
            fontweight="bold",
        )
    top = len(rows) - 1
    ax.annotate(
        "Flat",
        (rows[top][1], top),
        textcoords="offset points",
        xytext=(8, 15),
        color=fs.ACCESS,
        fontsize=9.5,
        fontweight="bold",
    )
    ax.annotate(
        "Detached",
        (_dx(rows[top][2]), top),
        textcoords="offset points",
        xytext=(-8, 15),
        ha="right",
        color=fs.DWELLING["Detached"],
        fontsize=9.5,
        fontweight="bold",
    )
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
    ax.set_xlabel("Reachable within a 1.6 km walk (median neighbourhood)")
    big = max(rows, key=lambda r: r[1])
    fs.deck(
        ax,
        "On the doorstep",
        f"A flat reaches {big[1]:.0f} {big[0].lower()} on foot; "
        f"a detached, {big[2]:.0f}",
        "Median count within a 1.6 km walk, by service",
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
        )
    ax.set_yscale("log")
    ax.axvline(1.6, color=fs.MUTED, lw=0.9, ls=":", zorder=1)
    ax.axvline(km[-1], color=fs.MUTED, lw=0.9, ls=":", zorder=1)
    _thr = dict(
        ha="center",
        va="top",
        fontsize=8,
        color=fs.INK_SECONDARY,
        fontweight="bold",
        linespacing=1.3,
        transform=ax.get_xaxis_transform(),
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
        zorder=6,
    )
    ax.text(1.6, 0.98, "on foot\n(a 20-minute walk)", **_thr)
    ax.text(km[-1], 0.98, "25.6 km drive\n(a typical rural commute)", **_thr)
    # White halo boxes: both multiples sit in the band the middle curves cross.
    _halo = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)
    ax.annotate(
        f"{foot:.0f}×",
        (1.6, np.sqrt(curves["Flat"][0] * curves["Detached"][0])),
        textcoords="offset points",
        xytext=(7, 0),
        fontsize=10.5,
        fontweight="bold",
        color=fs.INK,
        va="center",
        bbox=_halo,
        zorder=6,
    )
    xb = km[-1] + 3.6
    ax.plot(
        [xb, xb],
        [curves["Detached"][-1], curves["Flat"][-1]],
        color=fs.INK_SECONDARY,
        lw=1.0,
        zorder=5,
        clip_on=False,
    )
    ax.annotate(
        f"{far:.0f}×",
        (xb, np.sqrt(curves["Flat"][-1] * curves["Detached"][-1])),
        textcoords="offset points",
        xytext=(5, 0),
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color=fs.INK,
        va="center",
        zorder=6,
        annotation_clip=False,
    )
    ax.set_xlabel("Network distance reachable (km)")
    ax.set_ylabel("Everyday amenities reachable")
    ax.set_xlim(km[0] - 0.5, km[-1] + 5.6)
    # Keep the x-axis label clear of the figure-level source footer.
    fig.subplots_adjust(bottom=0.15)
    fs.deck(
        ax,
        "Reach and distance",
        f"A flat reaches ≈{foot:.0f}× more on foot, ≈{far:.0f}× at a 25.6 km drive",
        "The gap is widest at the doorstep and never closes",
    )
    fs.footer(fig)
    print(f"  F6 reach: ≈{foot:.0f}× on foot → ≈{far:.0f}× at 25 km")
    fs.save(fig, "fig6_access_curve")


# ---------------------------------------------------------------------------
# Act 4 — the rate
# ---------------------------------------------------------------------------


def rate(cf: pd.DataFrame, income: list[str], confounds: list[str]) -> None:
    """F7: amenities reachable per kWh of car travel, by type (flat ≈3.9×)."""
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

    fig, ax = plt.subplots(figsize=(fs.COL2, 4.6))
    x = np.arange(len(_TYPES))
    # Flat and Detached carry the comparison, in green; mute the middle types.
    colours = [fs.ACCESS, "#c8d5cc", "#c8d5cc", fs.ACCESS]
    ax.bar(x, vals, 0.62, color=colours, zorder=2)
    for i, r in enumerate(vals):
        strong = i in (0, len(_TYPES) - 1)
        ax.text(
            i,
            r + 0.015,
            f"{r:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold" if strong else "normal",
            color=fs.INK if strong else fs.MUTED,
        )
    # Bracket the Flat vs Detached comparison and label the rate.
    top = max(vals[0], vals[-1]) * 1.28
    ax.plot(
        [0, 0, 3, 3],
        [vals[0] * 1.05, top, top, vals[-1] * 1.45],
        color=fs.INK_SECONDARY,
        lw=1.0,
        zorder=3,
    )
    ax.text(
        1.5,
        top * 1.02,
        f"{ratio:.1f}×",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=fs.ACCESS,
        zorder=3,
    )
    ax.set_xticks(x, _TYPES)
    ax.set_ylabel("Amenities reachable per kWh of car travel")
    ax.set_ylim(0, top * 1.16)
    fs.deck(
        ax,
        "The rate",
        f"A flat returns {ratio:.1f}× a detached home's access per kWh",
        "Same reach at each area's own catchment, about a third of the fuel",
    )
    fs.footer(fig)
    print(f"  F7 rate: Flat:Det {ratio:.1f}×")
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
    labels, ratios = [], []
    for label, heat_transform, u_heat, u_ev in SCENARIOS:
        heat, trav = scenario_energy(
            cf, gas, elec, travel, heat_transform, u_heat, u_ev
        )
        cf["_lt"] = np.log((heat + trav).clip(lower=1).to_numpy())
        preds = _pure_preds(
            _comp_ols(cf, "_lt", _SHARE_FRACS + confounds, "total_hh"), cf, confounds
        )
        labels.append(pretty.get(label, label))
        ratios.append(preds["Detached"] / preds["Flat"])
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
    ax_e.set_xlim(1.0, max(ratios) + 0.5)
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

    # Right: the access gap, identical in every scenario (frozen). Each bar
    # carries its value so the panel reads as data, not decoration.
    ax_a.barh(
        y,
        [access_ratio - 1.0] * len(y),
        left=1.0,
        height=0.62,
        color=fs.DWELLING["Semi"],
        zorder=2,
    )
    for yi in y:
        ax_a.text(
            access_ratio * 0.985,
            yi,
            f"{access_ratio:.0f}×",
            ha="right",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color="white",
            zorder=3,
        )
    ax_a.set_xlim(1.0, access_ratio * 1.08)
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
    ax_a.set_xlabel("every scenario")
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
    fig.text(
        0.19,
        0.84,
        "Insulation, heat pumps and EVs close some of the energy gap and none "
        "of the access gap",
        fontsize=9.5,
        color=fs.INK_SECONDARY,
    )
    fs.footer(fig)
    print(
        f"  F8 scenarios: energy {status_quo:.2f}×→{ratios[-1]:.2f}× · "
        f"access {access_ratio:.0f}× frozen"
    )
    fs.save(fig, "fig8_scenarios")


def forest(cf: pd.DataFrame, confounds: list[str], income: list[str]) -> None:
    """F11: every headline compositional ratio with its clustered 95% CI.

    One log ruler for the whole evidence base: the energy gaps (Detached:Flat,
    log-OLS) above, the access gaps (Flat:Detached, Poisson) below. Each interval
    is the single-model delta-method CI with LAD-clustered covariance — the same
    numbers as the text tables. The access-per-kWh rate is deliberately absent:
    it is the product of the catchment-amenity and car-travel rows and carries a
    cluster-bootstrap interval, reported in the text.
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
                ("Car travel", _ols_ci("_le", base)),
                ("Heat", _ols_ci("_lh", base)),
                ("Heat, same family size", _ols_ci("_lh", base + ["log_hh_size"])),
                (
                    "Heat, same floor area",
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

    fig, ax = plt.subplots(figsize=(fs.COL2, 6.2))
    fig.subplots_adjust(left=0.30, right=0.82, top=0.80, bottom=0.10)
    ypos = 0.0
    yticks: list[float] = []
    ylabels: list[str] = []
    for header, colour, rows in groups:
        ax.annotate(
            header,
            xy=(-0.34, ypos),
            xycoords=("axes fraction", "data"),
            fontsize=8,
            fontweight="bold",
            color=colour,
            va="center",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"),
            zorder=5,
        )
        ypos -= 1.0
        for label, ci in rows:
            if ci is None:
                continue
            point, lo, hi = ci[0], ci[1], ci[2]
            ax.plot([lo, hi], [ypos, ypos], color=colour, lw=1.8, zorder=2)
            ax.plot(
                [lo, lo], [ypos - 0.14, ypos + 0.14], color=colour, lw=1.4, zorder=2
            )
            ax.plot(
                [hi, hi], [ypos - 0.14, ypos + 0.14], color=colour, lw=1.4, zorder=2
            )
            ax.scatter(point, ypos, s=42, color=colour, zorder=3)
            ax.annotate(
                f"{point:.2f}× [{lo:.2f}, {hi:.2f}]",
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
        ypos -= 0.6  # gap between groups
    ax.axvline(1.0, color=fs.INK, lw=1.1, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20, 30, 50, 100])
    ax.set_xticklabels(["1×", "2×", "3×", "5×", "10×", "20×", "30×", "50×", "100×"])
    ax.set_yticks(yticks, ylabels)
    ax.set_ylim(ypos + 0.4, 0.8)
    ax.set_xlabel("Pure-type ratio (log scale; 1× = no difference)")
    ax.tick_params(left=False)
    ax.spines["left"].set_visible(False)
    fs.deck(
        ax,
        "The evidence in one view",
        "Every headline gap, with its uncertainty",
        "Compositional pure-type ratios; 95% CIs clustered by local authority "
        "(309 districts)",
    )
    fs.footer(fig)
    print(f"  F11 forest: {len(yticks)} intervals drawn")
    fs.save(fig, "fig11_forest")


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
    print(f"\n  → {fs.FIG_DIR}")


if __name__ == "__main__":
    main()
