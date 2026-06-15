"""
Figures for paper/argument.md — the two-axis story in two charts.

1. ``energy_gradient.png`` — stacked heat + car-travel energy by dominant dwelling
   type, showing the Flat→Detached total-energy gradient (1.78×).
2. ``access_per_kwh.png`` — everyday access bought per kWh, a flat vs a detached
   home, one bar per service (the ~10× headline).

Both are computed from the same canonical sources as the prose (``oa_data`` +
``access_profile``), so the figures cannot drift from the numbers.

Reproduce:
    uv run python stats/argument_figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from access_profile import CATCHMENT_M, SERVICES, _series, _with_counts  # noqa: E402
from oa_data import load_and_aggregate  # noqa: E402

from urban_energy.paths import PROJECT_DIR  # noqa: E402

_OUT = PROJECT_DIR / "stats" / "figures" / "argument"
_TYPES = ["Flat", "Terraced", "Semi", "Detached"]
_HEAT_C = "#c1543b"  # warm — energy to heat the home
_TRAVEL_C = "#3b6ea5"  # cool — energy to travel
_ACCESS_C = "#3d8a5f"  # green — access return


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def energy_gradient(df: pd.DataFrame, out: Path) -> None:
    """Stacked heat + travel kWh/hh by dominant type, with the Flat→Det gradient.

    Bar height and the gradient use the **median of the per-OA total** (the
    canonical headline, ``argument.md`` §2). Heat/travel are split by their
    component-median proportion — medians are not additive, so the segments are
    shown proportionally rather than labelled, and only the totals carry numbers.
    """
    heat_med = [df.loc[df["dominant_type"] == t, "building_kwh_per_hh"].median()
                for t in _TYPES]
    trav_med = [_num(df.loc[df["dominant_type"] == t,
                            "transport_kwh_per_hh_total_est"]).median()
                for t in _TYPES]
    totals = [df.loc[df["dominant_type"] == t, "energy"].median() for t in _TYPES]
    heat_seg = [tot * h / (h + v)
                for h, v, tot in zip(heat_med, trav_med, totals, strict=True)]
    trav_seg = [tot - hs for tot, hs in zip(totals, heat_seg, strict=True)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(_TYPES))
    ax.bar(x, heat_seg, 0.62, label="Heat (metered)", color=_HEAT_C)
    ax.bar(x, trav_seg, 0.62, bottom=heat_seg,
           label="Car travel (NTS-anchored)", color=_TRAVEL_C)
    for i, tot in enumerate(totals):
        ax.text(i, tot + 350, f"{tot:,.0f}", ha="center", va="bottom",
                fontweight="bold", fontsize=10)

    grad = totals[-1] / totals[0]
    ax.annotate(
        f"{grad:.2f}× the energy",
        xy=(3, totals[-1]), xytext=(1.3, totals[-1] + 1500),
        ha="center", fontsize=11, fontweight="bold", color="#444",
        arrowprops=dict(arrowstyle="->", color="#888", lw=1.2),
    )
    ax.set_xticks(x, _TYPES)
    ax.set_ylabel("Energy spent (kWh / household / year)")
    ax.set_title(f"Energy axis — a detached home spends {grad:.2f}× a flat's energy",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}  (Flat {totals[0]:,.0f} → Detached {totals[-1]:,.0f})")


def access_per_kwh(df: pd.DataFrame, out: Path) -> None:
    """Per-service access bought per kWh, flat vs detached, + the geometric mean."""
    energy = {t: df.loc[df["dominant_type"] == t, "energy"].mean() for t in _TYPES}
    names, ratios = [], []
    for nm, base in SERVICES:
        col = f"{base}_{CATCHMENT_M}_nw"
        det = _series(df, "Detached", col).mean() / energy["Detached"]
        flt = _series(df, "Flat", col).mean() / energy["Flat"]
        names.append(nm)
        ratios.append(flt / det if det else np.nan)
    order = np.argsort(ratios)
    names = [names[i] for i in order]
    ratios = [ratios[i] for i in order]
    geomean = float(np.exp(np.nanmean(np.log(ratios))))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = np.arange(len(names))
    ax.barh(y, ratios, 0.66, color=_ACCESS_C)
    for i, r in enumerate(ratios):
        ax.text(r + 0.3, i, f"{r:.0f}×", va="center", fontsize=10)
    ax.axvline(geomean, color="#444", ls="--", lw=1.2)
    ax.text(geomean + 0.3, -0.7, f"geometric mean {geomean:.0f}×",
            color="#444", fontsize=10, fontweight="bold")
    ax.set_yticks(y, names)
    ax.set_xlabel("× more access per kWh (a flat vs a detached home)")
    ax.set_title("Access axis — same energy, ~10× the everyday access",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}  (geometric mean {geomean:.1f}×)")


def main() -> None:
    """Build both argument figures into stats/figures/argument/."""
    _OUT.mkdir(parents=True, exist_ok=True)
    df = _with_counts(load_and_aggregate())
    energy_gradient(df, _OUT / "energy_gradient.png")
    access_per_kwh(df, _OUT / "access_per_kwh.png")


if __name__ == "__main__":
    main()
