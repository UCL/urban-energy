"""
MAUP scale check for the two-axis energy gap.

The headline flat-to-detached energy gaps are computed at the 2021 Census Output
Area. A standard reviewer question is whether they are an artefact of that one
zonation — the modifiable areal unit problem (Openshaw, 1984): the same
underlying pattern can yield different coefficients at different aggregations.

This script re-aggregates the national OA frame to the two coarser census
zonations that nest cleanly above the OA — the Lower- and Middle-layer Super
Output Area (LSOA, MSOA) — household-weighted, and re-fits the *same*
no-intercept compositional model (``form_size_decomposition`` option D) at each
scale. Every dwelling-type share, the metered energy, and the confounds are
carried up as household-weighted means; the local-authority district (the
clustering unit) nests above all three scales, so cluster-robust SEs remain
defined. A gap that is stable from OA to MSOA is a property of the pattern, not
of the zonation.

Print-only; run on demand::

    uv run python stats/maup_scale.py
"""

from __future__ import annotations

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
from inference import CLUSTER_COL, fmt_ci, log_contrast_ci
from oa_data import load_and_aggregate

from urban_energy.paths import DATA_DIR

_STATS = DATA_DIR / "statistics"

# Dwelling-type shares (percent), the two metered energy components, and the
# confounds — all carried up to each coarse zonation as household-weighted means.
_SHARES = ["pct_flat", "pct_terraced", "pct_semi", "pct_detached"]
_ENERGY = ["building_kwh_per_hh", "transport_kwh_per_hh_total_est"]
_CONFOUNDS = [
    "median_build_year",
    "imd_overall_score",
    "imd_income_score",
    "pct_social_rented",
    "pct_private_rented",
    "hdd",
]


def _hh_weighted(oa: pd.DataFrame, key: str, cols: list[str]) -> pd.DataFrame:
    """Aggregate ``cols`` to ``key`` as household-weighted means.

    For each column the coarse value is ``Σ(x·hh) / Σ(hh)`` over the child OAs,
    with the denominator restricted to OAs where ``x`` is present (so a missing
    child does not deflate the mean). Households are summed. The local-authority
    district is nested above LSOA/MSOA, so its first (unique) value is kept.

    Parameters
    ----------
    oa : pandas.DataFrame
        National OA frame from :func:`oa_data.load_and_aggregate`.
    key : str
        Grouping column defining the coarse zonation (``LSOA21CD``/``MSOA21CD``).
    cols : list of str
        Numeric columns to carry up as household-weighted means.

    Returns
    -------
    pandas.DataFrame
        One row per coarse unit with ``total_hh``, the weighted ``cols`` and
        ``LAD22CD``.
    """
    w = pd.to_numeric(oa["total_hh"], errors="coerce")
    grp = oa[key]
    out = pd.DataFrame({"total_hh": w.groupby(grp).sum(min_count=1)})
    for c in cols:
        x = pd.to_numeric(oa[c], errors="coerce")
        num = (x * w).groupby(grp).sum(min_count=1)
        den = w.where(x.notna()).groupby(grp).sum(min_count=1)
        out[c] = num / den
    out["LAD22CD"] = oa.groupby(key)["LAD22CD"].first()
    return out.reset_index()


def _add_dvs(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the log heat and log total per-household energy DVs."""
    out = frame.copy()
    building = pd.to_numeric(out["building_kwh_per_hh"], errors="coerce")
    travel = pd.to_numeric(out["transport_kwh_per_hh_total_est"], errors="coerce")
    out["log_building_kwh_per_hh"] = np.log(building.clip(lower=1))
    out["log_total_kwh_per_hh"] = np.log((building + travel).clip(lower=1))
    return out


def _ratio(frame: pd.DataFrame, dv: str) -> tuple[float, float, float, float]:
    """D0 compositional Detached:Flat ratio + LAD-clustered 95% CI for ``dv``."""
    df = _compositional_frame(frame)
    confounds = (
        ["median_build_year"] + _deprivation_cols(df) + _tenure_cols(df) + _hdd_cols(df)
    )
    m = _comp_ols(df, dv, _SHARE_FRACS + confounds, "total_hh", cluster_col=CLUSTER_COL)
    if m is None:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    return log_contrast_ci(m, "s_detached", "s_flat")


def maup_ladder(oa: pd.DataFrame) -> None:
    """Print the flat-to-detached heat and total gaps at OA, LSOA and MSOA."""
    # MSOA key is not in the base frame; bring it in from the OA lookup.
    msoa = pd.read_parquet(
        _STATS / "oa_lookup.parquet", columns=["OA21CD", "MSOA21CD"]
    ).drop_duplicates("OA21CD")
    oa = oa.merge(msoa, on="OA21CD", how="left", validate="m:1")

    scales: list[tuple[str, pd.DataFrame]] = [("OA", _add_dvs(oa))]
    for name, key in [("LSOA", "LSOA21CD"), ("MSOA", "MSOA21CD")]:
        coarse = _hh_weighted(oa, key, _SHARES + _ENERGY + _CONFOUNDS)
        scales.append((name, _add_dvs(coarse)))

    print("\n" + "=" * 74)
    print("MAUP SCALE LADDER — compositional flat→detached gap by zonation")
    print("  Same no-intercept household-weighted model (option D), LAD-clustered CIs.")
    print("  Coarser units aggregate OA shares + metered energy household-weighted.")
    print("=" * 74)
    print(
        f"\n  {'scale':<6s}{'units':>10s}   "
        f"{'heat Det:Flat':>22s}   {'total Det:Flat':>22s}"
    )
    print("  " + "-" * 66)
    for name, frame in scales:
        n = int(pd.to_numeric(frame["total_hh"], errors="coerce").notna().sum())
        heat = _ratio(frame, "log_building_kwh_per_hh")
        total = _ratio(frame, "log_total_kwh_per_hh")
        print(f"  {name:<6s}{n:>10,d}   {fmt_ci(heat):>22s}   {fmt_ci(total):>22s}")
    print(
        "\n  Reading: the compositional contrast reads the gap at pure-type "
        "vertices (100%\n  flat vs 100% detached). Few LSOAs and almost no MSOAs "
        "approach a pure mix, so at\n  coarse scales this is increasingly an "
        "out-of-support extrapolation — which is\n  why the heat sub-component "
        "(already the confound-entangled, non-robust piece;\n  §4.5) attenuates "
        "fastest. The support-respecting companion below compares\n  typical units."
    )

    # Support-respecting companion: dominant-type median ratio at each scale. This
    # compares actual Flat-dominant against Detached-dominant units (no vertex
    # extrapolation), so it is directly comparable across scales and is the fair
    # MAUP test of whether the gap between typical neighbourhoods survives re-zoning.
    print("\n" + "=" * 74)
    print("  SUPPORT-RESPECTING COMPANION — dominant-type median ratio by zonation")
    print("=" * 74)
    print(
        f"\n  {'scale':<6s}{'flat n':>9s}{'det n':>9s}   "
        f"{'heat Det:Flat':>16s}   {'total Det:Flat':>16s}"
    )
    print("  " + "-" * 60)
    for name, frame in scales:
        dom = frame[_SHARES].apply(pd.to_numeric, errors="coerce").idxmax(axis=1)
        building = pd.to_numeric(frame["building_kwh_per_hh"], errors="coerce")
        travel = pd.to_numeric(frame["transport_kwh_per_hh_total_est"], errors="coerce")
        total = building + travel
        is_flat = dom == "pct_flat"
        is_det = dom == "pct_detached"
        heat_r = building[is_det].median() / building[is_flat].median()
        total_r = total[is_det].median() / total[is_flat].median()
        print(
            f"  {name:<6s}{int(is_flat.sum()):>9,d}{int(is_det.sum()):>9,d}   "
            f"{heat_r:>15.2f}×   {total_r:>15.2f}×"
        )


def main() -> None:
    """Load the national OA frame and print the MAUP scale ladder."""
    oa = load_and_aggregate()
    maup_ladder(oa)


if __name__ == "__main__":
    main()
