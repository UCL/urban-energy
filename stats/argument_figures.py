"""
Argument figures for the NEPI paper and summary.md.

The set carries the argument on its own. Read in order it is the paper (the full
story scaffold is in ``paper/figure_notes.md``):

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
import matplotlib.transforms as mtransforms  # noqa: E402
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
from matplotlib.lines import Line2D  # noqa: E402
from oa_data import load_and_aggregate  # noqa: E402
from scenarios import SCENARIOS, scenario_energy  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from urban_energy.paths import DATA_DIR  # noqa: E402

_NET_CACHE = DATA_DIR / "statistics" / "oa_network_access.parquet"
_TYPES = fs.DWELLING_ORDER

# On-foot services (network count within 1,600 m), for the doorstep figure.
# Hospitals are omitted: the NHS ETS layer counts all trust sites (wards, clinics,
# community units), not hospitals, so its count is not credible on the doorstep.
# Health is represented by GP surgeries and pharmacies, both correctly counted.
_SERVICES = [
    ("GP surgeries", "net_gp_1600"),
    ("Pharmacies", "net_pharmacy_1600"),
    ("Schools", "net_school_1600"),
    ("Eat & drink", "net_food_1600"),
    ("Grocery & convenience", "net_grocery_1600"),
    ("Parks & greenspace", "net_greenspace_1600"),
]

#: Single-line prose form of each service label, for the deck headline.
_SERVICE_PROSE = {
    "Eat & drink": "places to eat & drink",
    "Grocery & convenience": "grocery shops",
}


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
    # Bottom margin matched to F2's footer-to-axis-title spacing.
    fig.subplots_adjust(bottom=0.13, top=0.78, left=0.11, right=0.96)
    xs = [energy[t] for t in _TYPES]
    ys = [access[t] for t in _TYPES]
    ax.plot(xs, ys, color=fs.BASELINE, lw=1.4, zorder=1)
    for t in _TYPES:
        c = fs.DWELLING[t]
        ax.scatter(energy[t], access[t], s=110, color=c, zorder=3)
        label_bbox = dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.5)
        ax.annotate(
            fs.DWELLING_LABEL[t],
            xy=(energy[t], access[t]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            color=c,
            bbox=label_bbox,
        )
        ax.annotate(
            f"{energy[t]:,.0f} kWh\n{access[t]:,.0f} amenities",
            xy=(energy[t], access[t]),
            xytext=(10, 6),
            textcoords="offset points",
            va="top",
            fontsize=8,
            color=fs.INK_SECONDARY,
            bbox=label_bbox,
        )
    e_ratio = energy["Detached"] / energy["Flat"]
    a_ratio = access["Flat"] / access["Detached"]
    ax.set_yscale("log")
    ax.set_ylim(min(ys) * 0.45, max(ys) * 2.1)
    ax.margins(x=0.16)
    ax.set_xlabel("Total energy per dwelling (kWh / yr)")
    ax.set_ylabel("Amenities within a 1.6 km walk (log)")
    fs.comma(ax, "x")
    fs.deck(
        ax,
        "Energy against access",
        "Detached areas spend more energy and reach fewer amenities",
    )
    fs.footer(fig)
    print(f"  F1 inversion: energy {e_ratio:.2f}× · access {a_ratio:.0f}×")
    fs.save(fig, "fig1_inversion")


def national_scatter(cf: pd.DataFrame) -> None:
    """F2: every Output Area as a grey cloud, with kernel-density contours
    for the flat- and detached-dominant extremes (slate pair)."""
    d = cf.copy()
    energy = _num(d["building_kwh_per_hh"]) + _num(d["transport_kwh_per_hh_total_est"])
    access = _num(d["net_total_1600"])
    dom = d["dominant_type"].astype("object")
    ok = energy.between(1, 60_000) & (access > 0)

    e_ok = energy[ok].to_numpy()
    a_ok = access[ok].to_numpy()

    ytop = float(a_ok.max()) * 1.1  # log y: no cap needed, close every contour
    fig, ax = plt.subplots(figsize=(fs.COL2, 5.4))
    ax.scatter(
        e_ok,
        a_ok,
        s=1.6,
        color="#d3d7db",
        alpha=0.25,
        linewidths=0,
        zorder=1,
        rasterized=True,
    )
    # Per-class KDE fitted on LOG access (the y-axis is log, so density must be
    # estimated in the displayed space); a deterministic subsample keeps the
    # fit tractable. Contour levels are fractions of each class's own peak.
    rng = np.random.default_rng(0)
    gx, gy = np.meshgrid(
        np.linspace(2000, 42000, 160), np.linspace(0, np.log10(ytop), 160)
    )
    grid = np.vstack([gx.ravel(), gy.ravel()])
    extremes = ("Flat", "Detached")
    for z, t in enumerate(extremes, start=3):
        m = ok & (dom == t)
        xs = energy[m].to_numpy()
        ys = np.log10(access[m].to_numpy())
        if len(xs) > 20_000:
            idx = rng.choice(len(xs), 20_000, replace=False)
            xs, ys = xs[idx], ys[idx]
        kde = gaussian_kde(np.vstack([xs, ys]))
        dens = kde(grid).reshape(gx.shape)
        levels = np.array([0.08, 0.15, 0.25, 0.38, 0.55, 0.75, 0.92]) * dens.max()
        ax.contour(
            gx,
            10**gy,
            dens,
            levels=levels,
            colors=fs.DWELLING[t],
            linewidths=np.linspace(0.7, 2.2, len(levels)),
            zorder=z,
        )
    handles = [
        Line2D([], [], color=fs.DWELLING[t], lw=2.2, label=fs.DWELLING_LABEL[t])
        for t in extremes
    ]
    ax.legend(
        handles=handles, loc="upper right", frameon=False, fontsize=9, handlelength=1.6
    )
    ax.set_yscale("log")
    ax.set_xlim(2000, 42000)
    ax.set_ylim(4, ytop)
    ax.set_xlabel("Total energy per dwelling (kWh / yr)")
    ax.set_ylabel("Amenities within a 1.6 km walk (log)")
    fs.comma(ax, "x")
    fs.deck(ax, "At national scale", "Access falls as energy rises")
    fs.footer(fig)
    print(f"  F2 country: {int(ok.sum()):,} OAs, KDE contours for the extremes")
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
    # The claim is the component slopes, so state them on the bars: each
    # segment carries its multiple of the flat segment, centred in white.
    gap = 350.0  # kWh inset below each segment's top edge
    for i in range(len(_TYPES)):
        for y_pos, mult in [
            (heat_seg[i] - gap, heat_seg[i] / heat_seg[0]),
            (heat_seg[i] + trav_seg[i] - gap, trav_seg[i] / trav_seg[0]),
        ]:
            ax.text(
                i,
                y_pos,
                f"×{mult:.1f}",
                ha="center",
                va="top",
                fontsize=9.5,
                fontweight="bold",
                color="white",
            )
    # Category names are labels, not measurement ticks: match the axis-label
    # style rather than the muted tick style.
    ax.set_xticks(x, [fs.DWELLING_LABEL[t] for t in _TYPES])
    ax.tick_params(axis="x", labelsize=9, labelcolor=fs.INK_SECONDARY)
    ax.set_xlim(-0.6, len(_TYPES) - 0.4)
    fig.subplots_adjust(bottom=0.133)  # ~0.60 in to the footer, as figs 1/2/5/7
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
        ("Unadjusted gap", _SHARE_FRACS + confounds),
        ("At equal household size", _SHARE_FRACS + confounds + ["log_hh_size"]),
        (
            "At equal household size\n& floor area (the form alone)",
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
        ax.text(
            r + 0.012,
            yi,
            f"{r:.2f}×",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=fs.INK,
        )
    ax.axvline(1.0, color=fs.INK, lw=1.3, zorder=4)
    # Category names are labels, not measurement ticks: match the axis-label style.
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", labelsize=9, labelcolor=fs.INK_SECONDARY)
    ax.set_xlim(1.0, ratios[0] + 0.13)
    ax.set_xlabel("Ratio of home energy per dwelling, detached : flat")
    fig.subplots_adjust(left=0.27, right=0.96, top=0.80, bottom=0.143)
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
        med = {t: float(x[d["dominant_type"] == t].median()) for t in _TYPES}
        rows.append((name, med))
    rows.sort(key=lambda r: r[1]["Flat"])  # ascending: biggest gap at the top

    def _dx(v: float) -> float:
        return v  # symlog (linthresh=1) renders zero exactly at the axis origin

    fig, ax = plt.subplots(figsize=(fs.COL2, 4.7))
    fig.subplots_adjust(left=0.23, right=0.9, top=0.7, bottom=0.13)
    y = np.arange(len(rows))
    for yi, (_name, med) in enumerate(rows):
        f, de = med["Flat"], med["Detached"]
        ax.plot([_dx(de), f], [yi, yi], color=fs.BASELINE, lw=2.5, zorder=1)
        for t in ("Terraced", "Semi"):
            ax.scatter(
                _dx(med[t]),
                yi,
                s=60,
                facecolor="white",
                edgecolor=fs.DWELLING[t],
                linewidths=1.1,
                zorder=2,
            )
        ax.scatter(_dx(de), yi, s=90, color=fs.DWELLING["Detached"], zorder=3)
        ax.scatter(f, yi, s=90, color=fs.DWELLING["Flat"], zorder=3)
        ax.annotate(
            f"{f:.0f}",
            (f, yi),
            textcoords="offset points",
            xytext=(8, 0),
            va="center",
            ha="left",
            fontsize=8.5,
            color=fs.DWELLING["Flat"],
            fontweight="bold",
        )
        # Zero-count dots sit on the axis and carry no value label.
        if de >= 1:
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
    # A legend avoids anchoring type labels to dots that move per row.
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=8,
            markerfacecolor="white" if t in ("Terraced", "Semi") else fs.DWELLING[t],
            markeredgecolor=fs.DWELLING[t],
            markeredgewidth=1.1,
            color=fs.DWELLING[t],
            label=fs.DWELLING_LABEL[t],
        )
        for t in _TYPES
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, handletextpad=0.3)
    # Mark where the axis switches from linear (below one) to logarithmic.
    ax.axvline(1.0, color=fs.GRID, lw=0.8, ls=":", zorder=0)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, max(r[1]["Flat"] for r in rows) * 1.8)
    ax.set_xticks([0, 1, 10, 100])
    ax.set_xticklabels(["0", "1", "10", "100"])
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) + 0.35)
    # Tick text carries category/label weight on this chart: label ink, not
    # muted tick grey.
    ax.tick_params(axis="both", labelsize=9, labelcolor=fs.INK_SECONDARY)
    ax.set_xlabel("Median count within a 1.6 km walk (log above 1)")
    big = max(rows, key=lambda r: r[1]["Flat"])
    big_name = _SERVICE_PROSE.get(big[0], big[0].lower())
    fs.deck(
        ax,
        "On the doorstep",
        f"On foot, a flat reaches {big[1]['Flat']:.0f} {big_name} "
        f"and a detached reaches {big[1]['Detached']:.0f}",
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
    for t in _TYPES:
        ax.plot(km, curves[t], color=fs.DWELLING[t], lw=2.4, label=fs.DWELLING_LABEL[t])
    ax.set_yscale("log")
    ax.legend(loc="lower right", frameon=False, fontsize=9, handlelength=1.6)
    ax.set_xlabel("Network distance reachable (km)")
    ax.set_ylabel("Amenities reachable (log)")
    ax.set_xlim(km[0], km[-1])
    ax.set_ylim(None, curves["Flat"][-1] * 1.25)
    # Keep the x-axis label clear of the figure-level source footer.
    fig.subplots_adjust(bottom=0.13)
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
    fig, axes = plt.subplots(1, 3, figsize=(fs.COL2, 3.8))
    fig.subplots_adjust(top=0.66, bottom=0.16, left=0.09, right=0.985, wspace=0.35)
    x = np.arange(len(_TYPES))
    panels = [
        (
            axes[0],
            [amen[t] for t in _TYPES],
            "Amenities at car catchment",
            fs.ACCESS,
        ),
        (
            axes[1],
            [energy[t] for t in _TYPES],
            "Car energy (kWh / yr)",
            fs.TRAVEL,
        ),
        (
            axes[2],
            vals,
            "Access per kWh",
            fs.ACCESS,
        ),
    ]
    for ax, v, title, accent in panels:
        ax.bar(x, v, 0.6, color=[fs.DWELLING[t] for t in _TYPES], zorder=2)
        # Each bar carries its multiple of the flat bar, centred at the top.
        for i, val in enumerate(v):
            ax.annotate(
                f"{val / v[0]:.1f}×",
                (i, val),
                textcoords="offset points",
                xytext=(0, -2),
                ha="center",
                va="top",
                fontsize=6.5,
                fontweight="bold",
                color="white",
                zorder=5,
            )
        ax.set_title(
            title, loc="left", fontsize=8.5, fontweight="bold", color=accent, pad=6
        )
        ax.set_xticks(x, ["F", "T", "S", "D"], fontsize=8)
        ax.margins(y=0.24)
        ax.tick_params(axis="y", labelsize=7)
        if v is not vals:
            fs.comma(ax, "y")
    fig.text(
        fs.LEFT_X,
        0.945,
        "BUILDING THE RATE" if os.environ.get("NEPI_PLAIN_FIGS") != "1" else "",
        fontsize=8.5,
        fontweight="bold",
        color=fs.ACCENT,
    )
    if os.environ.get("NEPI_PLAIN_FIGS") != "1":
        fig.text(
            fs.LEFT_X,
            0.885,
            f"A flat obtains {ratio:.1f}× the access per kWh of car travel",
            fontsize=13,
            fontweight="bold",
            color=fs.INK,
        )
        fig.text(
            fs.LEFT_X,
            0.835,
            "F flats · T terraced · S semi-detached · D detached",
            fontsize=9,
            color=fs.INK_SECONDARY,
        )
    fs.footer(fig)
    print(
        f"  F7 rate: Flat:Det {ratio:.1f}× (catch {catch:.2f}× · energy {esave:.1f}×)"
    )
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
        if label == "Fabric + EVs (100%)":
            continue  # ladder rung reported in Extended Data Table 2 only

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

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(fs.COL2, 4.6), sharey=True, gridspec_kw={"wspace": 0.08}
    )
    fig.subplots_adjust(top=0.76, bottom=0.15, left=0.19, right=0.97)
    y = np.arange(len(labels))[::-1]

    # One colour for the levers; grey baseline. The value labels carry the
    # sign ("closed" / "wider"), so no colour emphasis is needed.
    slate = fs.DWELLING["Semi"]
    colours = [slate] * len(ratios)
    colours[0] = fs.NEUTRAL
    panels = [
        (ax_l, ratios, status_quo, "Unadjusted"),
        (ax_r, fam_ratios, fam_ratios[0], "At equal household size"),
    ]
    for ax, vals_, sq, ptitle in panels:
        ax.barh(
            y, [r - 1.0 for r in vals_], left=1.0, height=0.62, color=colours, zorder=2
        )
        ax.axvline(sq, color=fs.MUTED, lw=0.8, zorder=1)
        for yi, r in enumerate(vals_):
            ax.text(
                r - 0.02,
                y[yi],
                f"{r:.2f}×",
                va="center",
                ha="right",
                fontsize=7,
                fontweight="bold",
                color="white",
                zorder=5,
            )
        ax.set_xlim(1.0, max(vals_) + 0.08)
        ax.set_ylim(-0.6, len(labels) - 1 + 1.0)
        ax.text(
            0.03,
            0.97,
            ptitle,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            fontweight="bold",
            color=fs.INK_SECONDARY,
        )
    ax_l.set_yticks(y, labels)
    ax_l.tick_params(axis="y", labelsize=9, labelcolor=fs.INK_SECONDARY)
    fig.supxlabel(
        "Detached : Flat total energy (home + car travel) per dwelling",
        fontsize=9,
        color=fs.INK_SECONDARY,
        y=0.045,
    )
    fs.deck(
        ax_l,
        "Locked in",
        "Technology narrows the energy gap only part way",
    )
    fs.footer(fig)
    print(
        f"  F8 scenarios: energy {status_quo:.2f}×→{ratios[-1]:.2f}× · "
        f"access {access_ratio:.0f}× frozen"
    )
    fs.save(fig, "fig8_scenarios")


def forest(cf: pd.DataFrame, confounds: list[str], income: list[str]) -> None:
    """F11: every headline compositional ratio with its clustered 95% CI.

    Two stacked panels on independent log rulers, in the stem-and-dot grammar
    of F12: the energy gaps (detached:flat, log-OLS) above, the access gaps
    (flat:detached, Poisson) below, each estimate a stem from 1x with the 95%
    CI as a capped whisker over the tip. Each interval is the single-model
    delta-method CI with LAD-clustered covariance — the same numbers as the
    text tables. The access-per-kWh rate joins the access panel from the
    ledger (diamond tip, cluster-bootstrap interval).
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
    hh = ["log_hh_size"]
    # Energy panel: one row per component, the unadjusted and the
    # equal-household-size interval paired on the row (filled vs open marker).
    energy_rows: list[tuple[str, tuple | None, tuple | None, tuple[str, str]]] = [
        (
            "Total energy",
            _ols_ci("_lt", base),
            _ols_ci("_lt", base + hh),
            ("totalGap", "famNowGap"),
        ),
        (
            "Car travel",
            _ols_ci("_le", base),
            _ols_ci("_le", base + hh),
            ("travelGap", "travelFamGap"),
        ),
        (
            "Home energy",
            _ols_ci("_lh", base),
            _ols_ci("_lh", base + hh),
            ("heatGap", "heatFamGap"),
        ),
    ]
    access_rows: list[tuple[str, tuple | None]] = [
        ("Amenities on foot", _pois_ci("net_total_1600")),
        ("Jobs on foot", _pois_ci("net_jobs_1600")),
        ("People on foot", _pois_ci("net_pop_1600")),
        ("Amenities, own catchment", _pois_ci("net_amen")),
        ("Amenities, 25.6 km drive", _pois_ci("net_total_25600")),
    ]

    # The rate joins the access panel from the ledger (cluster-bootstrap CI),
    # so the figure genuinely holds ALL headline ratios.
    try:
        r_pt = float(ledger.value("rate"))
        r_lo, r_hi = (float(v) for v in ledger.value("rateCI").split("--"))
        access_rows.append(("Access per kWh (rate)", (r_pt, r_lo, r_hi)))
    except (ValueError, IndexError):
        pass

    def _fmt(v: float) -> str:
        return f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"

    # Printed row values come from the ledger where a key exists, so the
    # figure can never disagree with the manuscript text at the last decimal
    # (the plotted positions still come from the recomputed models).
    _LEDGER_LABELS = {
        "Total energy": ("totalGap", "totalGapCI"),
        "Total energy,\nequal household size": ("famNowGap", "famNowGapCI"),
        "Car travel": ("travelGap", "travelGapCI"),
        "Car travel,\nequal household size": ("travelFamGap", "travelFamGapCI"),
        "Home energy": ("heatGap", "heatGapCI"),
        "Home energy,\nequal household size": ("heatFamGap", "heatFamGapCI"),
        "Home energy,\nequal floor area too": ("heatSizeGap", "heatSizeGapCI"),
        "Amenities on foot": ("walkAmen", "walkAmenCI"),
        "Jobs on foot": ("walkJobs", "walkJobsCI"),
        "People on foot": ("walkPeople", "walkPeopleCI"),
        "Amenities, own catchment": ("catchAmen", "catchAmenCI"),
        "Amenities, 25.6 km drive": ("driveAmen", "driveAmenCI"),
    }

    # Two stacked panels with independent log rulers, drawn in the stem-and-dot
    # grammar of F12: each estimate is a stem from 1x to the point with the
    # 95% CI as a thin capped whisker over the tip. The stems make distance
    # from parity the ink itself, so no 1x reference line and no row
    # separators are needed (a floating-dot forest was tried and read as
    # unanchored; a shared ruler crushed the energy block into a sliver).
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(fs.COL2, 6.4),
        gridspec_kw={"hspace": 0.42, "height_ratios": [0.66, 1.0]},
    )
    fig.subplots_adjust(left=0.30, right=0.97, top=0.84, bottom=0.09)
    drawn = 0

    def _stem(ax, ci, ypos, colour, gap=0.99, face="solid", muted=False, marker="o"):
        """One estimate: stem from 1x, then the CI band, marker at the tip.

        The stem stops a small white breath short of the CI's lower bound, so
        the hand-off from "distance from parity" to "interval" is articulated;
        band and stem share one line weight, and the cap ticks mark the bound.
        The ``gap`` factor is per panel: a log ruler makes a fixed multiplier
        a fixed visual gap, so the wider access ruler uses a smaller factor.
        """
        point, lo, hi = ci[0], ci[1], ci[2]
        alpha = 0.4 if muted else 1.0
        if lo * gap > 1.0:
            ax.plot(
                [1.0, lo * gap],
                [ypos, ypos],
                color=colour,
                lw=1.8,
                alpha=alpha,
                solid_capstyle="butt",
                zorder=2,
            )
        ax.errorbar(
            point,
            ypos,
            xerr=[[point - lo], [hi - point]],
            fmt="none",
            ecolor=colour,
            elinewidth=1.8,
            capsize=3.0,
            capthick=1.2,
            alpha=alpha,
            zorder=3,
        )
        ax.scatter(
            point,
            ypos,
            s=55 if face == "solid" else 52,
            facecolor=colour if face == "solid" else "white",
            edgecolor=colour,
            linewidths=1.5,
            marker=marker,
            alpha=0.8 if muted else 1.0,
            zorder=4,
        )
        return point

    def _value(ax, label_key, point, ypos, colour, muted=False):
        # Values are bold, series-coloured, and centred directly above the tip
        # (the adjusted pair member goes below the tip, muted, a size smaller).
        pt = ledger.value(label_key) if label_key else ""
        ax.annotate(
            f"{pt or _fmt(point)}×",
            xy=(point, ypos),
            textcoords="offset points",
            xytext=(0, -5 if muted else 5),
            fontsize=7.5 if muted else 8,
            fontweight=600 if muted else "bold",
            color=fs.INK_SECONDARY if muted else colour,
            va="top" if muted else "bottom",
            ha="center",
            zorder=5,
        )

    def _row_labels(ax, ys, labels):
        # Manual row labels: tick-label vertical centring sits visibly low
        # (measured ~2 pt below the row lines), so place them by hand with a
        # small optical lift.
        tf = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        for y_row, lab in zip(ys, labels, strict=True):
            ax.annotate(
                lab,
                xy=(-0.02, y_row),
                xycoords=tf,
                xytext=(0, 2.0),
                textcoords="offset points",
                ha="right",
                va="center_baseline",
                fontsize=9,
                color=fs.INK_SECONDARY,
            )

    # Energy panel: one row per component, the unadjusted and the
    # equal-household-size estimate paired (solid vs open tip), the value
    # above-right of the solid tip and below-right of the open one.
    ax = axes[0]
    E_STEP = 1.35
    yticks, ylabels = [], []
    ypos = 0.0
    for label, ci_u, ci_f, (key_u, key_f) in energy_rows:
        if ci_u is None or ci_f is None:
            continue
        p_u = _stem(ax, ci_u, ypos + 0.24, fs.HEAT, gap=0.99, face="solid")
        p_f = _stem(ax, ci_f, ypos - 0.24, fs.HEAT, gap=0.99, face="white", muted=True)
        _value(ax, key_u, p_u, ypos + 0.24, fs.HEAT)
        _value(ax, key_f, p_f, ypos - 0.24, fs.HEAT, muted=True)
        yticks.append(ypos)
        ylabels.append(label)
        ypos -= E_STEP
        drawn += 2
    _row_labels(ax, yticks, ylabels)
    ax.set_yticks([])
    ax.set_ylim(ypos + E_STEP - 0.9, 0.85)
    ax.set_xscale("log")
    ax.set_xticks([1, 1.5, 2, 3, 4], ["1×", "1.5×", "2×", "3×", "4×"])
    ax.set_xlim(1.0, 4.2)
    ax.set_title(
        "ENERGY — detached to flat, per dwelling",
        loc="left",
        fontsize=8,
        fontweight="bold",
        color=fs.HEAT,
        pad=6,
    )
    handles = [
        plt.Line2D([], [], marker="o", ls="-", ms=6, color=fs.HEAT, label="unadjusted"),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="-",
            ms=6,
            color=fs.HEAT,
            markerfacecolor="white",
            alpha=0.7,
            label="equal household size",
        ),
    ]
    # A conventional stacked legend in the panel's empty upper-right corner,
    # text first with the symbol on the right edge.
    ax.legend(
        handles=handles,
        loc="upper right",
        markerfirst=False,
        fontsize=7.5,
        frameon=False,
        handletextpad=0.4,
        handlelength=1.4,
        borderaxespad=0.2,
    )

    # Access panel: one stem per row (diamond tip for the rate).
    ax = axes[1]
    yticks, ylabels = [], []
    ypos = 0.0
    for label, ci in access_rows:
        if ci is None:
            continue
        marker = "D" if "rate" in label else "o"
        point = _stem(ax, ci, ypos, fs.ACCESS, gap=0.97, face="solid", marker=marker)
        keys = _LEDGER_LABELS.get(label)
        _value(ax, keys[0] if keys else "", point, ypos, fs.ACCESS)
        yticks.append(ypos)
        ylabels.append(label)
        ypos -= 1.0
        drawn += 1
    _row_labels(ax, yticks, ylabels)
    ax.set_yticks([])
    ax.set_ylim(ypos + 1.0 - 0.6, 0.75)
    ax.set_xscale("log")
    ax.set_xticks(
        [1, 2, 5, 10, 20, 50, 100],
        ["1×", "2×", "5×", "10×", "20×", "50×", "100×"],
    )
    ax.set_xlim(1.0, 110)
    ax.set_title(
        "ACCESS — flat to detached, within reach",
        loc="left",
        fontsize=8,
        fontweight="bold",
        color=fs.ACCESS,
        pad=6,
    )
    ax.set_xlabel("Ratio (log; 1× = no difference)", fontsize=9)

    for ax in axes:
        ax.minorticks_off()
        # Category and value text are labels, not measurement ticks.
        ax.tick_params(axis="y", left=False, labelsize=9, labelcolor=fs.INK_SECONDARY)
        ax.tick_params(axis="x", labelsize=8, labelcolor=fs.INK_SECONDARY)
        ax.spines["left"].set_visible(False)
    # Figure-level title block: anchored to the figure, not the top axes, so it
    # can never overprint the first panel's header.
    if os.environ.get("NEPI_PLAIN_FIGS") != "1":
        fig.text(
            fs.LEFT_X,
            0.965,
            "SUMMARY OF RESULTS",
            fontsize=8.5,
            fontweight="bold",
            color=fs.ACCENT,
        )
        fig.text(
            fs.LEFT_X,
            0.925,
            "All headline ratios between flats and detached neighbourhoods",
            fontsize=13,
            fontweight="bold",
            color=fs.INK,
        )
    fs.footer(fig)
    print(f"  F11 summary: {drawn} stems drawn")
    fs.save(fig, "fig11_forest")


def equity(cf: pd.DataFrame) -> None:
    """F12 (Extended Data): the deprivation gradient of walkable access.

    Left panel: median on-foot amenities by IMD income-deprivation decile —
    nationally the most deprived areas hold the most access. Right panel: the
    rank correlation by area — England, inner London and twenty major cities,
    sorted by correlation — showing where the national gradient flattens and
    inverts.
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
    # Twenty-six further cities and large towns (single 2022 local-authority
    # districts), spanning weak and strong housing markets so the inversion
    # claim is tested on both tails rather than illustrated on hand-picked
    # cases.
    cities = [
        ("Birmingham", "E08000025"),
        ("Leeds", "E08000035"),
        ("Sheffield", "E08000019"),
        ("Liverpool", "E08000012"),
        ("Newcastle", "E08000021"),
        ("Bradford", "E08000032"),
        ("Coventry", "E08000026"),
        ("Leicester", "E06000016"),
        ("Nottingham", "E06000018"),
        ("Kingston upon Hull", "E06000010"),
        ("Stoke-on-Trent", "E06000021"),
        ("Derby", "E06000015"),
        ("Southampton", "E06000045"),
        ("Portsmouth", "E06000044"),
        ("Plymouth", "E06000026"),
        ("Brighton and Hove", "E06000043"),
        ("Oxford", "E07000178"),
        ("Cambridge", "E07000008"),
        ("Norwich", "E07000148"),
        ("York", "E06000014"),
    ]
    strata += [(name, lad == code) for name, code in cities]
    rhos = []
    for name, mask in strata:
        sub = pd.DataFrame({"a": amen[mask], "i": inc[mask]}).dropna()
        if len(sub) < 100:  # guard against a mis-coded or empty district
            continue
        rhos.append((name, float(sub["a"].corr(sub["i"], "spearman")), len(sub)))
    # England becomes a reference line rather than a row of its own; the
    # areas are sorted by correlation so the flatten-then-invert shape reads
    # top-down.
    eng_rho = rhos[0][1]
    rows = sorted(rhos[1:], key=lambda t: t[1], reverse=True)

    # One column, two rows: the decile gradient is a squat wide panel, the
    # correlation ladder a tall one, so neither fights the other's aspect.
    fig, (ax_d, ax_r) = plt.subplots(
        2,
        1,
        figsize=(fs.COL2 * 0.8, 7.6),
        gridspec_kw={"hspace": 0.52, "height_ratios": [1.0, 2.2]},
    )
    _L, _R = 0.19, 0.96
    fig.subplots_adjust(top=0.86, bottom=0.08, left=_L, right=_R)

    x = np.arange(10)
    ax_d.bar(x, med, 0.7, color=fs.ACCESS, zorder=2)
    ax_d.set_xticks([0, 4, 9], ["1\nleast\ndeprived", "5", "10\nmost\ndeprived"])
    # Words embedded in tick text are labels: style them as label ink.
    ax_d.tick_params(axis="x", labelsize=7.5, labelcolor=fs.INK_SECONDARY)
    ax_d.tick_params(axis="y", labelsize=8, labelcolor=fs.INK_SECONDARY)
    ax_d.set_ylabel("Median amenities within 1.6 km")
    ax_d.set_xlabel("IMD income-deprivation decile")
    ax_d.set_title(
        "National gradient by decile",
        loc="left",
        fontsize=8.5,
        fontweight="bold",
        color=fs.INK_SECONDARY,
        pad=6,
    )

    # One stem per area from zero, F11 grammar: 1.8 pt lines, values at the
    # tips, manual row labels with the optical lift. Near-zero correlations
    # are neutral grey so "flat" is not painted as "inverted".
    lab_tf = mtransforms.blended_transform_factory(ax_r.transAxes, ax_r.transData)
    yr = np.arange(len(rows))[::-1]
    leaders: list[tuple[float, float, float]] = []  # (y, rho, value pad in pt)
    for yi, (name, rho, n) in zip(yr, rows, strict=True):
        if abs(rho) <= 0.05:
            colour = fs.INK_SECONDARY
        elif rho > 0:
            colour = fs.ACCESS
        else:
            colour = fs.WIDENS
        ax_r.plot(
            [0, rho], [yi, yi], color=colour, lw=1.8,
            solid_capstyle="butt", zorder=2,
        )  # fmt: skip
        # Marker area tracks city size (Output-Area count; OAs are
        # equal-population); the cap keeps the largest dots inside their rows.
        size = float(np.clip(n / 32, 12, 110))
        ax_r.scatter(rho, yi, s=size, color=colour, zorder=3)
        pad = float(np.sqrt(size)) / 2 + 3.5
        leaders.append((yi, rho, pad))
        ax_r.annotate(
            f"{rho:+.2f}",
            (rho, yi),
            textcoords="offset points",
            xytext=(pad if rho >= 0 else -pad, 1.2),
            ha="left" if rho >= 0 else "right",
            va="center_baseline",
            fontsize=6.5,
            fontweight="bold",
            color=colour,
            zorder=4,
            # Knock out the England reference line where a value crosses it.
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
        )
        ax_r.annotate(
            name,
            xy=(-0.02, yi),
            xycoords=lab_tf,
            xytext=(0, 1.5),
            textcoords="offset points",
            ha="right",
            va="center_baseline",
            fontsize=7.5,
            color=fs.INK_SECONDARY,
        )
    # England as a faint dashed reference line rather than a row of its own.
    ax_r.axvline(
        eng_rho, color=fs.INK_SECONDARY, lw=0.8, ls=(0, (4, 3)), alpha=0.7, zorder=1
    )
    ax_r.annotate(
        f"England {eng_rho:+.2f}",
        xy=(eng_rho, -0.6),
        xytext=(3, 0),
        textcoords="offset points",
        ha="left",
        va="center_baseline",
        fontsize=6.5,
        color=fs.INK_SECONDARY,
    )
    ax_r.axvline(0, color=fs.INK, lw=0.8, zorder=1)
    ax_r.set_yticks([])
    ax_r.set_ylim(-1.05, len(rows) - 0.4)
    ax_r.spines["left"].set_visible(False)
    lo = min(r for _, r, _n in rows)
    hi = max(r for _, r, _n in rows)
    ax_r.set_xlim(lo - 0.17, hi + 0.12)
    ax_r.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    # Almost-imperceptible leaders from the name gutter to each row's
    # leftmost ink, so the eye tracks across the wide panel.
    x_lo, x_hi = ax_r.get_xlim()
    ppu = (fs.COL2 * 0.8) * (_R - _L) * 72 / (x_hi - x_lo)
    for yi, rho, pad in leaders:
        end = -3.0 / ppu if rho >= 0 else rho - (pad + 21.0) / ppu
        ax_r.plot(
            [x_lo, end], [yi, yi], color="#ebebe9", lw=0.5,
            solid_capstyle="butt", zorder=0.4,
        )  # fmt: skip
    ax_r.tick_params(axis="x", labelsize=8, labelcolor=fs.INK_SECONDARY)
    ax_r.set_xlabel("Rank correlation, access vs deprivation")
    ax_r.set_title(
        "Rank correlation by area",
        loc="left",
        fontsize=8.5,
        fontweight="bold",
        color=fs.INK_SECONDARY,
        pad=6,
    )
    if os.environ.get("NEPI_PLAIN_FIGS") != "1":
        fig.text(
            0.10,
            0.965,
            "ACCESS AND DEPRIVATION",
            fontsize=8.5,
            fontweight="bold",
            color=fs.ACCENT,
        )
        fig.text(
            0.10,
            0.935,
            "How walkable access relates to income deprivation",
            fontsize=12.5,
            fontweight="bold",
            color=fs.INK,
        )
    fs.footer(fig)
    n_inv = sum(1 for _, r, _n in rows if r < -0.05)
    for name, rho, n in rows:
        print(f"    ρ {rho:+.2f}  {name} ({n:,} OAs)")
    print(
        f"  F12 equity: national ρ {eng_rho:+.2f} (reference line); "
        f"{n_inv}/{len(rows)} strata inverted"
    )
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
