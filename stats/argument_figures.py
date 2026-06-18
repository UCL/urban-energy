"""
Figures for paper/summary.md — the two-axis story in three charts, on the D basis.

Every chart shows the **compositional (method-D)** contrast — a pure all-flat versus
a pure all-detached area, predicted at mean confounds — to match the ratios in
``paper/summary.md`` (no dominant-type medians, so the figures cannot drift from the
text):

1. ``energy_gradient.png`` — stacked heat + car-travel energy, pure-flat vs
   pure-detached (the Flat→Detached total-energy gradient, ~2.0× per household).
2. ``access_per_kwh.png`` — network amenities reachable per kWh within each OA's own
   car-trip catchment (the drivable rate, ~6.3×).
3. ``access_curve.png`` — predicted amenities reachable vs network distance, pure-flat
   vs pure-detached (the gap is ~24× on foot, narrowing to ~10× at a 25 km drive).

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
    _imd_income_col,
    _tenure_cols,
)
from oa_data import load_and_aggregate  # noqa: E402

from urban_energy.paths import DATA_DIR, PROJECT_DIR  # noqa: E402

_NET_CACHE = DATA_DIR / "statistics" / "oa_network_access.parquet"

_OUT = PROJECT_DIR / "stats" / "figures" / "argument"
_TYPES = ["Flat", "Detached"]
_HEAT_C = "#c1543b"  # warm — energy to heat the home
_TRAVEL_C = "#3b6ea5"  # cool — energy to travel
_ACCESS_C = "#3d8a5f"  # green — access return
_FLAT_C = "#3d8a5f"
_DET_C = "#c1543b"


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _pure_preds(model: Any, frame: pd.DataFrame, confounds: list[str]) -> dict:
    """Predicted level for a pure all-flat / all-detached area at mean confounds.

    Works for both the log energy model and the Poisson access model (both have a
    log link, so the level is ``exp(b_type + sum(b_conf * mean_conf))``).
    """
    base = sum(
        float(model.params[c]) * _num(frame[c]).mean() for c in confounds
    )
    return {
        "Flat": float(np.exp(model.params["s_flat"] + base)),
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
    # Bar height is the total-model prediction (so the gradient matches the text);
    # split into heat/travel by the two component models' proportions.
    totals = [tot[t] for t in _TYPES]
    heat_seg = [tot[t] * heat[t] / (heat[t] + trav[t]) for t in _TYPES]
    trav_seg = [totals[i] - heat_seg[i] for i in range(len(_TYPES))]
    grad = totals[1] / totals[0]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(_TYPES))
    ax.bar(x, heat_seg, 0.5, label="Heat (metered)", color=_HEAT_C)
    ax.bar(
        x, trav_seg, 0.5, bottom=heat_seg, label="Car travel (NTS-anchored)",
        color=_TRAVEL_C,
    )
    for i, tot in enumerate(totals):
        ax.text(i, tot + 300, f"{tot:,.0f}", ha="center", va="bottom",
                fontweight="bold", fontsize=10)
    ax.annotate(
        f"{grad:.1f}× the energy",
        xy=(1, totals[1]), xytext=(0.5, totals[1] + 1500), ha="center",
        fontsize=11, fontweight="bold", color="#444",
        arrowprops=dict(arrowstyle="->", color="#888", lw=1.2),
    )
    ax.set_xticks(x, ["Pure flat area", "Pure detached area"])
    ax.set_ylabel("Energy spent (kWh / household / year)")
    ax.set_title(
        f"Energy axis — a detached home spends {grad:.1f}× a flat's energy",
        fontsize=12, fontweight="bold",
    )
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}  ({totals[0]:,.0f} → {totals[1]:,.0f}, {grad:.2f}×)")


def access_per_kwh(cf: pd.DataFrame, income: list[str], out: Path) -> None:
    """Network amenities reachable per kWh, pure-flat vs pure-detached (Poisson D)."""
    cf = cf.copy()
    travel = _num(cf["transport_kwh_per_hh_total_est"]).replace(0, np.nan)
    cf["_y"] = _num(cf["net_amen"]) / travel
    rate = _pure_preds(
        _comp_poisson(cf, "_y", _SHARE_FRACS + income, "total_hh"), cf, income
    )
    vals = [rate[t] for t in _TYPES]
    ratio = vals[0] / vals[1] if vals[1] else float("nan")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(_TYPES))
    ax.bar(x, vals, 0.5, color=_ACCESS_C)
    for i, r in enumerate(vals):
        ax.text(i, r + 0.02, f"{r:.2f}", ha="center", va="bottom", fontsize=10)
    ax.annotate(
        f"{ratio:.1f}× the access per kWh",
        xy=(0, vals[0]), xytext=(0.5, vals[0] + 0.1), ha="center",
        fontsize=11, fontweight="bold", color="#444",
        arrowprops=dict(arrowstyle="->", color="#888", lw=1.2),
    )
    ax.set_xticks(x, ["Pure flat area", "Pure detached area"])
    ax.set_ylabel("Network amenities reachable per kWh")
    ax.set_title(
        f"Access axis — a flat reaches {ratio:.1f}× the amenities per kWh",
        fontsize=12, fontweight="bold",
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
    flat_y, det_y = [], []
    for dd in dists:
        cf["_y"] = _num(cf[f"net_total_{dd}"])
        p = _pure_preds(
            _comp_poisson(cf, "_y", _SHARE_FRACS + income, "total_hh"), cf, income
        )
        flat_y.append(p["Flat"])
        det_y.append(p["Detached"])
    km = np.array(dists) / 1000
    flat_y, det_y = np.array(flat_y), np.array(det_y)
    foot = flat_y[0] / det_y[0]
    far = flat_y[-1] / det_y[-1]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(km, flat_y, color=_FLAT_C, lw=2.2, label="Pure flat area")
    ax.plot(km, det_y, color=_DET_C, lw=2.2, label="Pure detached area")
    ax.set_yscale("log")
    ax.annotate(
        f"≈{foot:.0f}× on foot",
        xy=(km[0], flat_y[0]), xytext=(km[0] + 3, flat_y[0] * 0.4),
        fontsize=10, fontweight="bold", color="#444",
        arrowprops=dict(arrowstyle="->", color="#888", lw=1.0),
    )
    ax.set_xlabel("Network distance reachable (km)")
    ax.set_ylabel("Everyday amenities reachable (predicted)")
    ax.set_title(
        f"Access — a flat reaches ≈{foot:.0f}× more on foot, ≈{far:.0f}× at 25 km",
        fontsize=12, fontweight="bold",
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
    confounds = ["median_build_year"] + _imd_income_col(cf) + _tenure_cols(cf)
    income = _imd_income_col(cf)
    dists = sorted(
        int(c.rsplit("_", 1)[1]) for c in net.columns if c.startswith("net_total_")
    )
    energy_gradient(cf, confounds, _OUT / "energy_gradient.png")
    access_per_kwh(cf, income, _OUT / "access_per_kwh.png")
    access_curve(cf, income, dists, _OUT / "access_curve.png")


if __name__ == "__main__":
    main()
