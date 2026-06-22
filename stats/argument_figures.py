"""
Figures for paper/summary.md — the two-axis story in three charts, on the D basis.

Every chart shows **compositional (method-D)** predictions for the four pure dwelling
types (all-flat, all-terraced, all-semi, all-detached) at mean confounds, to match the
ratios in ``paper/summary.md`` (no dominant-type medians, so the figures cannot drift
from the text):

1. ``energy_gradient.png`` — stacked heat + car-travel energy by dwelling type (the
   Flat→Detached total-energy gradient, ~2.1× per dwelling).
2. ``access_per_kwh.png`` — amenities reachable per kWh: catchment access ÷ car-travel
   energy (the rate, ~3.6× flat vs detached = 1.2× access × 3.1× energy saving).
3. ``access_curve.png`` — predicted amenities reachable vs network distance by dwelling
   type (the gap is ~24× flat vs detached on foot, narrowing to ~10× at a 25 km drive).

Energy axes use the log compositional model (``_comp_ols``); access uses the Poisson
count model (``_comp_poisson``). Both predict a pure-type level as
``exp(b_type + mean-confound offset)``.

Reproduce (build the network cache first):
    uv run python stats/oa_network_access.py
    uv run python stats/argument_figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

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
from oa_data import load_and_aggregate  # noqa: E402

from urban_energy.paths import DATA_DIR, PROJECT_DIR  # noqa: E402

_NET_CACHE = DATA_DIR / "statistics" / "oa_network_access.parquet"

_OUT = PROJECT_DIR / "stats" / "figures" / "argument"
_HEAT_C = "#c1543b"  # warm: energy to heat the home
_TRAVEL_C = "#3b6ea5"  # cool: energy to travel
_ACCESS_C = "#3d8a5f"  # green: access return


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _pure_preds(model: Any, frame: pd.DataFrame, confounds: list[str]) -> dict:
    """Predicted level for a pure all-flat / all-detached area at mean confounds.

    Works for both the log energy model and the Poisson access model (both have a
    log link, so the level is ``exp(b_type + sum(b_conf * mean_conf))``).
    """
    base = sum(float(model.params[c]) * _num(frame[c]).mean() for c in confounds)
    return {
        "Flat": float(np.exp(model.params["s_flat"] + base)),
        "Terraced": float(np.exp(model.params["s_terraced"] + base)),
        "Semi": float(np.exp(model.params["s_semi"] + base)),
        "Detached": float(np.exp(model.params["s_detached"] + base)),
    }


def energy_gradient(cf: pd.DataFrame, confounds: list[str], out: Path) -> None:
    """Stacked heat + travel, pure-flat vs pure-detached (method-D predictions)."""
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
    # Bar height is the total-model prediction; split into heat/travel by the two
    # component models' proportions.
    types = ["Flat", "Terraced", "Semi", "Detached"]
    totals = [tot[t] for t in types]
    heat_seg = [tot[t] * heat[t] / (heat[t] + trav[t]) for t in types]
    trav_seg = [totals[i] - heat_seg[i] for i in range(len(types))]
    grad = totals[-1] / totals[0]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(types))
    ax.bar(x, heat_seg, 0.62, label="Heat (metered)", color=_HEAT_C)
    ax.bar(
        x,
        trav_seg,
        0.62,
        bottom=heat_seg,
        label="Car travel (NTS-anchored)",
        color=_TRAVEL_C,
    )
    for i, t in enumerate(totals):
        ax.text(
            i,
            t + 300,
            f"{t:,.0f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    ax.set_xticks(x, types)
    ax.set_ylabel("Energy spent (kWh / household / year)")
    ax.set_title(
        f"Energy axis: a detached home spends {grad:.1f}× a flat's energy",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}  ({totals[0]:,.0f} → {totals[-1]:,.0f}, {grad:.2f}×)")


def access_per_kwh(
    cf: pd.DataFrame, income: list[str], confounds: list[str], out: Path
) -> None:
    """Amenities reachable per kWh, pure-flat vs pure-detached.

    The rate is access ÷ energy, so each type's level is its predicted catchment
    amenities (income-controlled Poisson) divided by its predicted car-travel energy
    (full-confound log-OLS). The flat-to-detached ratio is therefore the product of
    the access advantage and the energy saving — reconstructable from the two axes,
    not a per-OA ratio modelled directly (which double-counted to a spurious 6.3×).
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
    types = ["Flat", "Terraced", "Semi", "Detached"]
    vals = [amen[t] / energy[t] for t in types]
    ratio = vals[0] / vals[-1] if vals[-1] else float("nan")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(types))
    ax.bar(x, vals, 0.62, color=_ACCESS_C)
    for i, r in enumerate(vals):
        ax.text(i, r + 0.02, f"{r:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x, types)
    ax.set_ylabel("Network amenities reachable per kWh")
    ax.set_title(
        f"Access per kWh: a flat returns {ratio:.1f}× a detached home",
        fontsize=12,
        fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}  (Flat:Det {ratio:.1f}×)")


def access_curve(
    cf: pd.DataFrame, income: list[str], dists: list[int], out: Path
) -> None:
    """Predicted amenities vs network distance, pure-flat vs detached (Poisson D)."""
    cf = cf.copy()
    types = ["Flat", "Terraced", "Semi", "Detached"]
    series = {t: [] for t in types}
    for dd in dists:
        cf["_y"] = _num(cf[f"net_total_{dd}"])
        p = _pure_preds(
            _comp_poisson(cf, "_y", _SHARE_FRACS + income, "total_hh"), cf, income
        )
        for t in types:
            series[t].append(p[t])
    km = np.array(dists) / 1000
    curves = {t: np.array(v) for t, v in series.items()}
    foot = curves["Flat"][0] / curves["Detached"][0]
    far = curves["Flat"][-1] / curves["Detached"][-1]
    colours = {
        "Flat": "#3d8a5f",
        "Terraced": "#7aa66b",
        "Semi": "#c9a13b",
        "Detached": "#c1543b",
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for t in types:
        ax.plot(km, curves[t], color=colours[t], lw=2.2, label=t)
    ax.set_yscale("log")
    ax.set_xlabel("Network distance reachable (km)")
    ax.set_ylabel("Everyday amenities reachable (predicted)")
    ax.set_title(
        f"Access: a flat reaches ≈{foot:.0f}× more on foot, ≈{far:.0f}× at 25 km",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}  (≈{foot:.0f}× on foot → ≈{far:.0f}× at 25 km)")


def main() -> None:
    """Build the argument figures (method-D basis) into stats/figures/argument/."""
    _OUT.mkdir(parents=True, exist_ok=True)
    df = load_and_aggregate()
    net = pd.read_parquet(_NET_CACHE)
    cf = _compositional_frame(
        df.merge(net, left_on="OA21CD", right_index=True, how="left")
    )
    # Energy figure: the full energy-ladder confound set (deprivation + climate),
    # so the figure's gradient matches summary.md. Access figures stay income-only
    # (overall IMD's barriers sub-domains are collinear with access — see
    # access_profile.compositional_access).
    confounds = (
        ["median_build_year"] + _deprivation_cols(cf) + _tenure_cols(cf) + _hdd_cols(cf)
    )
    income = _imd_income_col(cf)
    dists = sorted(
        int(c.rsplit("_", 1)[1]) for c in net.columns if c.startswith("net_total_")
    )
    energy_gradient(cf, confounds, _OUT / "energy_gradient.png")
    access_per_kwh(cf, income, confounds, _OUT / "access_per_kwh.png")
    access_curve(cf, income, dists, _OUT / "access_curve.png")


if __name__ == "__main__":
    main()
