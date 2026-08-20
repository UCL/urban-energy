"""The forgone economy, priced, and the prospectus figures.

Counterfactual: every neighbourhood whose fabric post-dates 1918 is re-priced
at the national pre-1919 dwelling-type composition (normalised over the four
named types, each OA keeping its own residual 'other' share), using the same
no-intercept compositional model as the companion analysis. The difference is
the energy the motor-era fabric spends above the compactness it inherited.

Outputs the national premium (TWh/yr), the per-district 'bloat' ranking, and
the two prospectus figures, written next to this script:

- ``fig_era.png``   — compact share of each era's fabric by settlement-size tercile
- ``fig_bloat.png`` — each district's energy vs its own fabric at pre-1919 compactness

Run::

    uv run python papers/allometry/allometry_premium.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stats"))

import matplotlib

matplotlib.use("Agg")
import figstyle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from form_size_decomposition import (
    _SHARE_FRACS,
    _comp_ols,
    _compositional_frame,
    _deprivation_cols,
    _hdd_cols,
    _tenure_cols,
)
from inference import CLUSTER_COL
from oa_data import load_and_aggregate

from urban_energy.paths import DATA_DIR

_OUT = Path(__file__).resolve().parent
_FOUR = ["s_flat", "s_terraced", "s_semi", "s_detached"]


def counterfactual(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Observed and counterfactual total energy per OA.

    Fits the compositional model on log total energy per household, computes the
    hh-weighted pre-1919 four-type composition, and re-prices every post-1918 OA
    at that composition (own 'other' share retained).

    Returns
    -------
    (E_obs, E_cf, valid) : per-OA observed kWh, counterfactual kWh, and the
    boolean mask of rows where both are defined.
    """
    confounds = (
        ["median_build_year"] + _deprivation_cols(df) + _tenure_cols(df) + _hdd_cols(df)
    )
    m = _comp_ols(
        df,
        "log_total_kwh_per_hh",
        _SHARE_FRACS + confounds,
        "total_hh",
        cluster_col=CLUSTER_COL,
    )
    if m is None:
        raise RuntimeError("compositional model failed to fit")

    year = pd.to_numeric(df["median_build_year"], errors="coerce")
    hh = pd.to_numeric(df["total_hh"], errors="coerce").fillna(0)
    pre = year <= 1918
    raw = {s: float((df.loc[pre, s] * hh[pre]).sum() / hh[pre].sum()) for s in _FOUR}
    tot4 = sum(raw.values())
    target = {s: v / tot4 for s, v in raw.items()}
    print("Pre-1919 composition (hh-weighted, normalised over the four types):")
    print("  " + "  ".join(f"{s[2:]}: {v * 100:.1f}%" for s, v in target.items()))

    s_other = pd.to_numeric(df["s_other"], errors="coerce")
    delta_log = pd.Series(0.0, index=df.index)
    for s in _FOUR:
        delta_log += m.params[s] * (
            target[s] * (1.0 - s_other) - pd.to_numeric(df[s], errors="coerce")
        )
    delta_log = delta_log.where(year > 1918, 0.0)

    E_obs = df["total_kwh_per_hh"] * hh
    E_cf = E_obs * np.exp(delta_log)
    return E_obs, E_cf, E_obs.notna() & E_cf.notna()


def bloat_frame(
    df: pd.DataFrame, E_obs: pd.Series, E_cf: pd.Series, valid: pd.Series
) -> pd.DataFrame:
    """Per-district population, observed and counterfactual energy, and bloat."""
    names = pd.read_parquet(
        DATA_DIR / "statistics" / "oa_lookup.parquet", columns=["OA21CD", "LAD22NM"]
    ).drop_duplicates("OA21CD")
    lad_name = df["OA21CD"].map(names.set_index("OA21CD")["LAD22NM"])
    grp = lad_name[valid]
    lad = pd.DataFrame(
        {
            "pop": pd.to_numeric(df.loc[valid, "total_people"], errors="coerce")
            .groupby(grp)
            .sum(),
            "E_obs": E_obs[valid].groupby(grp).sum(),
            "E_cf": E_cf[valid].groupby(grp).sum(),
        }
    ).dropna()
    lad["bloat"] = lad["E_obs"] / lad["E_cf"]
    return lad[lad["pop"] > 20_000]


def fig_bloat(lad: pd.DataFrame) -> None:
    """Scatter of district bloat against population, largest places named."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    above = lad["bloat"] >= 1
    ax.axhline(1.0, color="0.35", lw=1.0, zorder=1)
    for mask, colour in [(above, "#c26d4b"), (~above, "#4b7fc2")]:
        ax.scatter(
            lad.loc[mask, "pop"],
            lad.loc[mask, "bloat"],
            s=np.sqrt(lad.loc[mask, "pop"]) / 9,
            alpha=0.5,
            lw=0,
            color=colour,
            zorder=2,
        )
    labels = [
        "Birmingham",
        "Leeds",
        "Manchester",
        "Liverpool",
        "Bristol",
        "Sheffield",
        "Wiltshire",
        "Cornwall",
        "Newham",
        "Tower Hamlets",
        "Cheshire East",
        "East Riding of Yorkshire",
    ]
    for name in labels:
        if name in lad.index:
            row = lad.loc[name]
            ax.annotate(
                name,
                (row["pop"], row["bloat"]),
                fontsize=7,
                xytext=(4, 3),
                textcoords="offset points",
            )
    ax.set_xscale("log")
    ax.set_ylim(0.82, 1.2)
    ax.set_xlabel("Population (log)")
    ax.set_ylabel("Energy vs own fabric at pre-1919 compactness")
    ax.set_title("The energy above each district's inherited benchmark", loc="left")
    fig.tight_layout()
    fig.savefig(_OUT / "fig_bloat.png", dpi=200)
    print(f"fig -> {_OUT / 'fig_bloat.png'}")


def fig_era(df: pd.DataFrame) -> None:
    """Compact share of each era's fabric, by settlement-size tercile."""
    year = pd.to_numeric(df["median_build_year"], errors="coerce")
    hh = pd.to_numeric(df["total_hh"], errors="coerce").fillna(0)
    pop_lad = (
        pd.to_numeric(df["total_people"], errors="coerce")
        .groupby(df["LAD22CD"])
        .transform("sum")
    )
    size_ter = pd.qcut(pop_lad, 3, labels=["smaller", "middle", "larger"])
    cohort = pd.cut(
        year,
        bins=[0, 1918, 1944, 1979, 2100],
        labels=["pre-1919", "1919–1944", "1945–1979", "1980+"],
    )
    compact = pd.to_numeric(df["pct_flat"], errors="coerce") + pd.to_numeric(
        df["pct_terraced"], errors="coerce"
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cohorts = ["pre-1919", "1919–1944", "1945–1979", "1980+"]
    colours = {"smaller": "#a8c6e8", "middle": "#6f9fd8", "larger": "#2f5f9e"}
    xbase = np.arange(len(cohorts))
    for i, t in enumerate(["smaller", "middle", "larger"]):
        vals = []
        for c in cohorts:
            sel = (cohort == c) & (size_ter == t)
            vals.append(float((compact[sel] * hh[sel]).sum() / hh[sel].sum()))
        ax.bar(
            xbase + (i - 1) * 0.27,
            vals,
            width=0.25,
            color=colours[t],
            label=f"{t} districts",
        )
    ax.set_xticks(xbase, cohorts)
    ax.set_ylabel("Flats + terraces in the fabric (%)")
    ax.set_xlabel("Median build year of the neighbourhood")
    ax.set_title(
        "Compact fabric by era: the walking constraint, then its release", loc="left"
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(_OUT / "fig_era.png", dpi=200)
    print(f"fig -> {_OUT / 'fig_era.png'}")


def main() -> None:
    """Price the counterfactual, print the ranking, write both figures."""
    oa = load_and_aggregate()
    building = pd.to_numeric(oa["building_kwh_per_hh"], errors="coerce")
    travel = pd.to_numeric(oa["transport_kwh_per_hh_total_est"], errors="coerce")
    oa["log_total_kwh_per_hh"] = np.log((building + travel).clip(lower=1))
    oa["total_kwh_per_hh"] = building + travel
    df = _compositional_frame(oa)

    E_obs, E_cf, valid = counterfactual(df)
    tot_obs = E_obs[valid].sum() / 1e9
    tot_cf = E_cf[valid].sum() / 1e9
    print(f"\nObserved total (homes + car travel): {tot_obs:.0f} TWh/yr")
    print(f"Counterfactual at inherited compactness: {tot_cf:.0f} TWh/yr")
    print(
        f"Premium of the dispersed century: {tot_obs - tot_cf:.0f} TWh/yr "
        f"({(tot_obs - tot_cf) / tot_obs * 100:.0f}%)"
    )

    lad = bloat_frame(df, E_obs, E_cf, valid)
    print("\nMost and least bloated of the 30 largest districts:")
    big = lad.nlargest(30, "pop").sort_values("bloat")
    for name, row in pd.concat([big.head(5), big.tail(5)]).iterrows():
        print(f"  {name:<28s} pop {row['pop']:>9,.0f}   bloat {row['bloat']:.2f}x")

    figstyle.apply_style()
    fig_era(df)
    fig_bloat(lad)


if __name__ == "__main__":
    main()
