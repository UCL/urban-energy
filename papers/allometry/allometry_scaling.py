"""Pilot scaling analyses for the settlement energy-allometry paper.

Ecology's allometric laws are laws of energy (Kleiber 1932; West, Brown and
Enquist 1997): metabolic rate rises sublinearly with body mass because shared
distribution structure serves a larger body more efficiently. This script asks
where that law now stands for England's settlements, using the same built
artefacts as the companion (lock-in) analysis in ``stats/``.

Sections
--------
1. Scaling exponent across LADs (total / home / travel, with and without
   dwelling-type composition).
2. Exponent within compactness terciles, and the equal-population energy ratio
   between the dispersed and compact terciles.
3. Size-compactness coupling (does growth densify?).
4. Per-household energy elasticity to population density at OA level.
5. Exponent of each build cohort's fabric (pre-1919 sublinear, motor decades
   linear, post-1980 partially re-coupled).
6. Compositional detached:flat home-energy gap within each RUC 2021 settlement
   class (the gap is not a big-city artefact).

Pilot caveats: local authority districts stand in for settlements (final
estimation is planned on ONS Built-Up Areas), travel energy is a survey-anchored
construction (the clean metered exponent is the home-energy one), and today's
district populations proxy historic settlement size by rank.

Print-only; run on demand::

    uv run python papers/allometry/allometry_scaling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stats"))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from form_size_decomposition import (
    _SHARE_FRACS,
    _comp_ols,
    _compositional_frame,
    _deprivation_cols,
    _hdd_cols,
    _tenure_cols,
)
from inference import CLUSTER_COL, fmt_ci, log_contrast_ci
from oa_data import load_and_aggregate

_SHARES = ["pct_flat", "pct_terraced", "pct_semi", "pct_detached"]


def prepare(oa: pd.DataFrame) -> pd.DataFrame:
    """Add the per-OA energy totals and log DVs used throughout."""
    building = pd.to_numeric(oa["building_kwh_per_hh"], errors="coerce")
    travel = pd.to_numeric(oa["transport_kwh_per_hh_total_est"], errors="coerce")
    hh = pd.to_numeric(oa["total_hh"], errors="coerce")
    oa = oa.copy()
    oa["oa_building_kwh"] = building * hh
    oa["oa_travel_kwh"] = travel * hh
    oa["oa_total_kwh"] = (building + travel) * hh
    oa["log_total_per_hh"] = np.log((building + travel).clip(lower=1))
    return oa


def lad_frame(oa: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the OA frame to LADs: population, households, energy totals and
    household-weighted dwelling-type shares."""
    hh = pd.to_numeric(oa["total_hh"], errors="coerce")
    w = hh.fillna(0)
    grp = oa["LAD22CD"]
    lad = pd.DataFrame(
        {
            "pop": pd.to_numeric(oa["total_people"], errors="coerce")
            .groupby(grp)
            .sum(),
            "hh": hh.groupby(grp).sum(),
            "E_total": oa["oa_total_kwh"].groupby(grp).sum(),
            "E_building": oa["oa_building_kwh"].groupby(grp).sum(),
            "E_travel": oa["oa_travel_kwh"].groupby(grp).sum(),
        }
    )
    for c in _SHARES:
        x = pd.to_numeric(oa[c], errors="coerce")
        lad[c] = (x * w).groupby(grp).sum() / w.where(x.notna()).groupby(grp).sum()
    lad = lad.dropna()
    lad["log_pop"] = np.log(lad["pop"])
    return lad


def _beta(lad: pd.DataFrame, y_col: str, extra: list[str], label: str) -> None:
    """Fit log(energy) on log(population) plus ``extra`` and print the exponent."""
    df = lad[lad[y_col] > 0].copy()
    df["log_E"] = np.log(df[y_col])
    m = sm.OLS(df["log_E"], sm.add_constant(df[["log_pop", *extra]])).fit(
        cov_type="HC1"
    )
    b = m.params["log_pop"]
    lo, hi = m.conf_int().loc["log_pop"]
    print(f"  {label:<50s} beta = {b:.3f} [{lo:.3f}, {hi:.3f}]  R2={m.rsquared:.3f}")


def section_1_lad_betas(lad: pd.DataFrame) -> None:
    """Scaling exponents across LADs, with and without composition."""
    print("\n" + "=" * 76)
    print("1. SCALING EXPONENT ACROSS LADs")
    print("=" * 76)
    print(f"  {len(lad)} LADs; pop {lad['pop'].min():,.0f} - {lad['pop'].max():,.0f}")
    for y, name in [
        ("E_total", "total (home+travel)"),
        ("E_building", "home (metered)"),
        ("E_travel", "travel (constructed)"),
    ]:
        _beta(lad, y, [], f"log E_{name} ~ log pop")
    comp = ["pct_flat", "pct_terraced", "pct_semi"]
    for y, name in [("E_total", "total"), ("E_building", "home")]:
        _beta(lad, y, comp, f"log E_{name} ~ log pop + composition")


def section_2_terciles(lad: pd.DataFrame) -> None:
    """Exponent within compactness terciles + equal-population energy ratio."""
    print("\n" + "=" * 76)
    print("2. EXPONENT WITHIN COMPACTNESS TERCILES (LADs by flat share)")
    print("=" * 76)
    lad = lad.copy()
    lad["tercile"] = pd.qcut(
        lad["pct_flat"], 3, labels=["dispersed", "middle", "compact"]
    )
    pop_ref = lad["pop"].median()
    fitted: dict[str, float] = {}
    for t in ["dispersed", "middle", "compact"]:
        sub = lad[lad["tercile"] == t]
        m = sm.OLS(np.log(sub["E_total"]), sm.add_constant(sub[["log_pop"]])).fit(
            cov_type="HC1"
        )
        b = m.params["log_pop"]
        lo, hi = m.conf_int().loc["log_pop"]
        fitted[t] = float(np.exp(m.params["const"] + b * np.log(pop_ref)))
        print(
            f"  {t:<11s} N={len(sub):>4d}  flat share {sub['pct_flat'].median():.1f}%"
            f"   beta = {b:.3f} [{lo:.3f}, {hi:.3f}]"
        )
    print(
        f"  At the same population ({pop_ref:,.0f}), dispersed-tercile energy is "
        f"{fitted['dispersed'] / fitted['compact']:.2f}x compact-tercile."
    )


def section_3_coupling(lad: pd.DataFrame) -> None:
    """Does growth densify? Flat share on log population across LADs."""
    print("\n" + "=" * 76)
    print("3. SIZE-COMPACTNESS COUPLING")
    print("=" * 76)
    m = sm.OLS(lad["pct_flat"], sm.add_constant(lad[["log_pop"]])).fit(cov_type="HC1")
    slope = m.params["log_pop"] * np.log(10)
    print(
        f"  flat share vs population: {slope:+.1f} pp per tenfold, "
        f"R2 = {m.rsquared:.3f}"
    )


def section_4_density(oa: pd.DataFrame) -> None:
    """Per-household energy elasticity to population density (OA level).

    Near-zero: raw density is the wrong compactness measure, and the economy is
    carried by dwelling-type composition instead.
    """
    print("\n" + "=" * 76)
    print("4. PER-HH ENERGY ELASTICITY TO DENSITY (OA level, hh-weighted WLS)")
    print("=" * 76)
    df = oa.copy()
    df["log_density"] = np.log(pd.to_numeric(df["pop_density"], errors="coerce"))
    confounds = [
        "median_build_year",
        "imd_overall_score",
        "imd_income_score",
        "pct_social_rented",
        "pct_private_rented",
    ] + (["hdd"] if "hdd" in df.columns else [])
    for xs, label in [([], "bivariate"), (confounds, "with confounds")]:
        cols = ["log_total_per_hh", "log_density", *xs, "total_hh", "LAD22CD"]
        sub = df[cols].copy()
        for c in cols[:-1]:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
        m = sm.WLS(
            sub["log_total_per_hh"],
            sm.add_constant(sub[["log_density", *xs]]),
            weights=sub["total_hh"],
        ).fit(cov_type="cluster", cov_kwds={"groups": sub["LAD22CD"]})
        d = m.params["log_density"]
        lo, hi = m.conf_int().loc["log_density"]
        print(
            f"  total E/hh ~ density ({label:<15s}) "
            f"delta = {d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        )


def section_5_cohorts(oa: pd.DataFrame) -> None:
    """Scaling exponent of each build cohort's fabric, aggregated to LADs."""
    print("\n" + "=" * 76)
    print("5. EXPONENT OF EACH ERA'S FABRIC")
    print("=" * 76)
    year = pd.to_numeric(oa["median_build_year"], errors="coerce")
    cohort = pd.cut(
        year,
        bins=[0, 1918, 1944, 1979, 2100],
        labels=["pre-1919", "1919-1944", "1945-1979", "1980+"],
    )
    print(f"  {'cohort':<12s}{'LADs':>6s}{'beta total':>24s}{'beta home':>24s}")
    for c in ["pre-1919", "1919-1944", "1945-1979", "1980+"]:
        sub = oa[cohort == c]
        agg = pd.DataFrame(
            {
                "pop": pd.to_numeric(sub["total_people"], errors="coerce")
                .groupby(sub["LAD22CD"])
                .sum(),
                "E": sub["oa_total_kwh"].groupby(sub["LAD22CD"]).sum(),
                "Eb": sub["oa_building_kwh"].groupby(sub["LAD22CD"]).sum(),
            }
        ).dropna()
        agg = agg[(agg["pop"] > 1000) & (agg["E"] > 0)]
        out = []
        for col in ["E", "Eb"]:
            m = sm.OLS(np.log(agg[col]), sm.add_constant(np.log(agg[["pop"]]))).fit(
                cov_type="HC1"
            )
            b = m.params["pop"]
            lo, hi = m.conf_int().loc["pop"]
            out.append(f"{b:.3f} [{lo:.3f}, {hi:.3f}]")
        print(f"  {c:<12s}{len(agg):>6d}{out[0]:>24s}{out[1]:>24s}")


def section_6_ruc_gap(oa: pd.DataFrame) -> None:
    """Compositional detached:flat home-energy gap within each RUC class."""
    print("\n" + "=" * 76)
    print("6. COMPOSITIONAL Det:Flat HOME-ENERGY GAP WITHIN RUC 2021 CLASS")
    print("  Option-D model (hh-weighted, LAD-clustered), fit per class. Travel")
    print("  is constructed FROM the class, so only home energy is tested here.")
    print("=" * 76)
    counts = oa["RUC21NM"].value_counts()
    print(f"  {'class':<48s}{'N':>9s}  {'heat Det:Flat':>20s}{'mean flat%':>11s}")
    for ruc in counts.index:
        sub = oa[oa["RUC21NM"] == ruc]
        if len(sub) < 200:
            continue
        df = _compositional_frame(sub)
        confounds = (
            ["median_build_year"]
            + _deprivation_cols(df)
            + _tenure_cols(df)
            + _hdd_cols(df)
        )
        m = _comp_ols(
            df,
            "log_building_kwh_per_hh",
            _SHARE_FRACS + confounds,
            "total_hh",
            cluster_col=CLUSTER_COL,
        )
        if m is None:
            continue
        heat = log_contrast_ci(m, "s_detached", "s_flat")
        mean_flat = pd.to_numeric(sub["pct_flat"], errors="coerce").mean()
        print(f"  {ruc:<48s}{len(sub):>9,d}  {fmt_ci(heat):>20s}{mean_flat:>10.1f}%")
    print(
        "  Classes with a tiny flat share (the smaller-rural pair) put the pure-"
        "type\n  contrast out of support; read those rows as unsupported."
    )


def main() -> None:
    """Run all six pilot sections on the national OA frame."""
    oa = prepare(load_and_aggregate())
    lad = lad_frame(oa)
    section_1_lad_betas(lad)
    section_2_terciles(lad)
    section_3_coupling(lad)
    section_4_density(oa)
    section_5_cohorts(oa)
    section_6_ruc_gap(oa)


if __name__ == "__main__":
    main()
