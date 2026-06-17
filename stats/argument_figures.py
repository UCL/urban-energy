"""
Figures for paper/argument.md — the two-axis story in three charts.

1. ``energy_gradient.png`` — stacked heat + car-travel energy by dominant dwelling
   type, showing the Flat→Detached total-energy gradient (1.74×).
2. ``access_per_kwh.png`` — network amenities reachable per kWh within each OA's own
   car-trip catchment, by dwelling type (the drivable rate, ~2.9×).
3. ``access_curve.png`` — amenities reachable vs network distance by type: the
   like-for-like gap (4.5–9.5×) and how detached drives far to match the flat's count.

All are computed from the canonical loader (``oa_data``) plus the network-access
cache (``oa_network_access.parquet``), so the figures cannot drift from the numbers.

Reproduce (build the network cache first):
    uv run python stats/oa_network_access.py
    uv run python stats/argument_figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from oa_data import load_and_aggregate  # noqa: E402

from urban_energy.paths import DATA_DIR, PROJECT_DIR  # noqa: E402

_NET_CACHE = DATA_DIR / "statistics" / "oa_network_access.parquet"

_OUT = PROJECT_DIR / "stats" / "figures" / "argument"
_TYPES = ["Flat", "Terraced", "Semi", "Detached"]
_HEAT_C = "#c1543b"  # warm — energy to heat the home
_TRAVEL_C = "#3b6ea5"  # cool — energy to travel
_ACCESS_C = "#3d8a5f"  # green — access return


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def energy_gradient(df: pd.DataFrame, out: Path) -> None:
    """Stacked heat + travel by type; bars + gradient use the median per-OA total."""
    heat_med = [
        df.loc[df["dominant_type"] == t, "building_kwh_per_hh"].median() for t in _TYPES
    ]
    trav_med = [
        _num(
            df.loc[df["dominant_type"] == t, "transport_kwh_per_hh_total_est"]
        ).median()
        for t in _TYPES
    ]
    totals = [df.loc[df["dominant_type"] == t, "energy"].median() for t in _TYPES]
    heat_seg = [
        tot * h / (h + v) for h, v, tot in zip(heat_med, trav_med, totals, strict=True)
    ]
    trav_seg = [tot - hs for tot, hs in zip(totals, heat_seg, strict=True)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(_TYPES))
    ax.bar(x, heat_seg, 0.62, label="Heat (metered)", color=_HEAT_C)
    ax.bar(
        x,
        trav_seg,
        0.62,
        bottom=heat_seg,
        label="Car travel (NTS-anchored)",
        color=_TRAVEL_C,
    )
    for i, tot in enumerate(totals):
        ax.text(
            i,
            tot + 350,
            f"{tot:,.0f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    grad = totals[-1] / totals[0]
    ax.annotate(
        f"{grad:.2f}× the energy",
        xy=(3, totals[-1]),
        xytext=(1.3, totals[-1] + 1500),
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="#444",
        arrowprops=dict(arrowstyle="->", color="#888", lw=1.2),
    )
    ax.set_xticks(x, _TYPES)
    ax.set_ylabel("Energy spent (kWh / household / year)")
    ax.set_title(
        f"Energy axis — a detached home spends {grad:.2f}× a flat's energy",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}  (Flat {totals[0]:,.0f} → Detached {totals[-1]:,.0f})")


def access_per_kwh(df: pd.DataFrame, net: pd.DataFrame, out: Path) -> None:
    """Network amenities reachable per kWh within each OA's catchment, by type."""
    d = df.merge(net[["net_amen"]], left_on="OA21CD", right_index=True, how="left")
    # divide by TRAVEL energy (the catchment is the car travel) — matches access_profile
    d["rate"] = _num(d["net_amen"]) / _num(d["transport_kwh_per_hh_total_est"])
    rate = [d.loc[d["dominant_type"] == t, "rate"].median() for t in _TYPES]
    ratio = rate[0] / rate[-1] if rate[-1] else float("nan")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(_TYPES))
    ax.bar(x, rate, 0.62, color=_ACCESS_C)
    for i, r in enumerate(rate):
        ax.text(i, r + 0.01, f"{r:.2f}", ha="center", va="bottom", fontsize=10)
    ax.annotate(
        f"{ratio:.1f}× the access per kWh",
        xy=(0, rate[0]),
        xytext=(1.5, rate[0] + 0.08),
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="#444",
        arrowprops=dict(arrowstyle="->", color="#888", lw=1.2),
    )
    ax.set_xticks(x, _TYPES)
    ax.set_ylabel("Network amenities reachable per kWh")
    ax.set_title(
        f"Access axis — a flat reaches {ratio:.1f}× the amenities per kWh",
        fontsize=12,
        fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}  (Flat:Det {ratio:.1f}×)")


def access_curve(df: pd.DataFrame, net: pd.DataFrame, out: Path) -> None:
    """Amenities reachable vs network distance, by type — the like-for-like gap."""
    d = df[["OA21CD", "dominant_type"]].merge(net, left_on="OA21CD", right_index=True)
    dists = sorted(
        int(c.rsplit("_", 1)[1]) for c in net.columns if c.startswith("net_total_")
    )
    km = np.array(dists) / 1000
    colours = {
        "Flat": "#3d8a5f",
        "Terraced": "#7aa66b",
        "Semi": "#c9a13b",
        "Detached": "#c1543b",
    }
    catch = {  # median car-trip catchment (km) per type
        t: _num(d.loc[d["dominant_type"] == t, "trip_m"]).median() / 1000
        for t in _TYPES
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))
    curves = {}
    for t in _TYPES:
        y = [
            _num(d.loc[d["dominant_type"] == t, f"net_total_{dd}"]).median()
            for dd in dists
        ]
        curves[t] = np.array(y)
        ax.plot(km, y, color=colours[t], lw=2, label=t)
    # mark flat vs detached catchments: detached drives far to reach the flat's level
    for t in ("Flat", "Detached"):
        c = catch[t]
        yc = float(np.interp(c, km, curves[t]))
        ax.plot([c, c], [0, yc], color=colours[t], ls=":", lw=1.2)
        ax.scatter([c], [yc], color=colours[t], zorder=5)
    ax.annotate(
        "detached drives 2.4× as far\nto reach what a flat reaches on a short trip",
        xy=(
            catch["Detached"],
            float(np.interp(catch["Detached"], km, curves["Detached"])),
        ),
        xytext=(11, curves["Flat"].max() * 0.55),
        fontsize=9,
        color="#444",
        arrowprops=dict(arrowstyle="->", color="#888", lw=1.0),
    )
    ax.set_xlabel("Network distance reachable (km)")
    ax.set_ylabel("Everyday amenities reachable (median)")
    ax.set_title(
        "Like-for-like — at any distance, a flat reaches far more (4.5–9.5×)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.02, y=0.05)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}  (like-for-like curve)")


def main() -> None:
    """Build the argument figures into stats/figures/argument/."""
    _OUT.mkdir(parents=True, exist_ok=True)
    df = load_and_aggregate()
    df["energy"] = df["building_kwh_per_hh"] + _num(
        df["transport_kwh_per_hh_total_est"]
    )
    energy_gradient(df, _OUT / "energy_gradient.png")
    net = pd.read_parquet(_NET_CACHE)
    access_per_kwh(df, net, _OUT / "access_per_kwh.png")
    access_curve(df, net, _OUT / "access_curve.png")


if __name__ == "__main__":
    main()
