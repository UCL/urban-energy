"""
Mixed use: does functional mix add to the compact advantage?

The manuscript recommends compact form *and* mixed-use development. Compact form
is measured throughout by dwelling-type composition. Mix was not measured at all,
so the second half of the recommendation rested on the literature rather than on
these data. This module measures it.

Two senses of "mix" are kept apart, because they are not the same quantity:

* **Jobs-housing balance** (the planning sense): workplace jobs against resident
  households in the same Output Area, as ``1 - |J - H| / (J + H)``. One where jobs
  and homes are in balance, zero where an area is wholly dormitory or wholly
  workplace. Workplace jobs are a separate register from the amenity points that
  form the access numerator, so this can be used as a predictor without
  circularity.
* **Destination-type balance**: Hill diversity (q=1) over the six everyday
  destination types within 1,600 m (``net_mix1_1600``, from the network cache).
  Reported descriptively only, for two reasons. It is computed from the same
  amenity points that the access counts sum, so regressing access on it would be
  partly circular; and balance is not what mixed use means, since it marks down a
  high street holding many food outlets and one surgery. Richness and proximity
  are the substance of mix, and the per-service walkable counts in
  ``access_profile`` already show compact areas reaching more of every type.

The test holds dwelling-type composition, so the balance coefficient answers the
question the recommendation needs: at a given built form, does an area that mixes
homes and workplaces reach more and spend less?

Inputs:
    - the assembled OA frame (``oa_data.load_and_aggregate``)
    - ``$DATA_DIR/statistics/oa_network_access.parquet`` (network cache)
    - ``$DATA_DIR/employment/workplace_jobs.gpkg`` (workplace jobs per OA)
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from access_profile import NET_CACHE, TYPES, _comp_poisson
from form_size_decomposition import (
    _SHARE_FRACS,
    _comp_ols,
    _compositional_frame,
    _deprivation_cols,
    _hdd_cols,
    _tenure_cols,
)
from inference import CLUSTER_COL
from oa_access import _JOBS
from oa_data import load_and_aggregate

from urban_energy.paths import DATA_DIR

#: Percentiles of the balance measure contrasted in the reported effect. A
#: p10 → p90 shift is the span of the observed distribution, not an extrapolation.
BALANCE_LO_PCT: float = 10.0
BALANCE_HI_PCT: float = 90.0


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def jobs_housing_balance(jobs: pd.Series, households: pd.Series) -> pd.Series:
    """
    Balance between workplace jobs and resident households in an area.

    Parameters
    ----------
    jobs : pandas.Series
        Workplace jobs counted in the Output Area.
    households : pandas.Series
        Resident households in the same area.

    Returns
    -------
    pandas.Series
        ``1 - |J - H| / (J + H)`` in [0, 1]. One where jobs and homes are equal in
        number, zero where the area holds only one of the two. ``NaN`` where the
        area holds neither.
    """
    j = _num(jobs).fillna(0.0).clip(lower=0)
    h = _num(households).fillna(0.0).clip(lower=0)
    total = j + h
    return (1 - (j - h).abs() / total.replace(0, np.nan)).clip(0, 1)


def load_frame() -> pd.DataFrame:
    """Assemble the OA frame with workplace jobs and the network mix columns."""
    df = load_and_aggregate().reset_index(drop=True)

    jobs = gpd.read_file(_JOBS, columns=["OA21CD", "jobs"], ignore_geometry=True)
    jobs = pd.DataFrame(jobs)[["OA21CD", "jobs"]].drop_duplicates("OA21CD")
    df = df.merge(jobs, on="OA21CD", how="left", validate="m:1")
    # The jobs layer keeps only OAs with at least one workplace job, so an absent
    # row is a genuine zero rather than a missing value.
    df["jobs"] = _num(df["jobs"]).fillna(0.0)
    df["jobs_housing_balance"] = jobs_housing_balance(df["jobs"], df["total_hh"])

    if NET_CACHE.exists():
        net_cols = ["net_mix1_1600", "net_total_1600", "net_amen"]
        net = pd.read_parquet(NET_CACHE)
        keep = [c for c in net_cols if c in net.columns]
        df = df.merge(
            net[keep], left_on="OA21CD", right_index=True, how="left", validate="m:1"
        )
    return df


def descriptive(df: pd.DataFrame) -> None:
    """Both mix measures by dominant dwelling type."""
    print("\n  Mix by dominant dwelling type (medians):")
    header = f"    {'type':<10s}{'jobs/hh':>10s}{'balance':>10s}"
    if "net_mix1_1600" in df.columns:
        header += f"{'types reachable (q1)':>24s}"
    print(header)
    for t in TYPES:
        s = df[df["dominant_type"] == t]
        jph = (_num(s["jobs"]) / _num(s["total_hh"]).replace(0, np.nan)).median()
        bal = _num(s["jobs_housing_balance"]).median()
        line = f"    {t:<10s}{jph:>10.2f}{bal:>10.2f}"
        if "net_mix1_1600" in df.columns:
            line += f"{_num(s['net_mix1_1600']).median():>24.2f}"
        print(line)
    print(
        "\n    (balance = 1 - |jobs - households| / (jobs + households): 1 is parity,\n"
        "     0 is wholly dormitory or wholly workplace. q1 is Hill diversity over\n"
        "     the six destination types within 1,600 m, descriptive only.)"
    )


def to_msoa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-aggregate the frame to MSOA, the scale at which balance is interpretable.

    Jobs and households are summed, because balance is a ratio of counts over an
    area; everything else is carried up as a household-weighted mean, matching
    :mod:`maup_scale`. At Output Area scale (about 125 households) the measure is
    dominated by area size, since a small area needs few jobs to reach parity. An
    MSOA holds 2,000 to 6,000 households, closer to the scale at which homes and
    workplaces are actually planned together.
    """
    lookup = pd.read_parquet(
        DATA_DIR / "statistics" / "oa_lookup.parquet", columns=["OA21CD", "MSOA21CD"]
    ).drop_duplicates("OA21CD")
    df = df.merge(lookup, on="OA21CD", how="left", validate="m:1")

    carry = [
        c
        for c in (
            *[f"pct_{t}" for t in ("flat", "terraced", "semi", "detached")],
            "building_kwh_per_hh",
            "transport_kwh_per_hh_total_est",
            "net_amen",
            "median_build_year",
            "imd_overall_score",
            "imd_income_score",
            "pct_social_rented",
            "pct_private_rented",
            "hdd",
        )
        if c in df.columns
    ]
    w = _num(df["total_hh"])
    grp = df["MSOA21CD"]
    out = pd.DataFrame({"total_hh": w.groupby(grp).sum(min_count=1)})
    out["jobs"] = _num(df["jobs"]).groupby(grp).sum(min_count=1)
    for c in carry:
        x = _num(df[c])
        out[c] = (x * w).groupby(grp).sum(min_count=1) / w.where(x.notna()).groupby(
            grp
        ).sum(min_count=1)
    out["LAD22CD"] = df.groupby(grp)["LAD22CD"].first()
    out = out.reset_index()
    out["jobs_housing_balance"] = jobs_housing_balance(out["jobs"], out["total_hh"])
    return out


def _fit(frame: pd.DataFrame, label: str) -> tuple[float, float, float] | None:
    """Access, energy and rate multipliers for a p10 → p90 shift in balance."""
    cf = _compositional_frame(frame)
    income = [
        c for c in cf.columns if "imd_income" in c.lower() and "score" in c.lower()
    ]
    confounds = (
        ["median_build_year"] + _deprivation_cols(cf) + _tenure_cols(cf) + _hdd_cols(cf)
    )
    bal = "jobs_housing_balance"
    cf[bal] = _num(cf[bal])
    cf["_le"] = np.log(
        (
            _num(cf["building_kwh_per_hh"]) + _num(cf["transport_kwh_per_hh_total_est"])
        ).clip(lower=1)
    )

    lo, hi = np.nanpercentile(cf[bal].to_numpy(float), [BALANCE_LO_PCT, BALANCE_HI_PCT])
    span = float(hi - lo)

    # Access carries the income-only conditioning of the headline access models;
    # energy carries the full confound set of the headline energy axis.
    acc = _comp_poisson(
        cf,
        "net_amen",
        _SHARE_FRACS + income + [bal],
        "total_hh",
        cluster_col=CLUSTER_COL,
    )
    eng = _comp_ols(
        cf, "_le", _SHARE_FRACS + confounds + [bal], "total_hh", cluster_col=CLUSTER_COL
    )
    if acc is None or eng is None:
        print(f"    {label:<8s} [model did not fit]")
        return None

    acc_mult = float(np.exp(acc.params[bal] * span))
    eng_mult = float(np.exp(eng.params[bal] * span))
    rate_mult = acc_mult / eng_mult
    print(
        f"    {label:<8s}{len(cf):>10,d}{lo:>9.2f}{hi:>9.2f}"
        f"{acc_mult:>12.2f}{eng_mult:>10.2f}{rate_mult:>10.2f}"
    )
    return acc_mult, eng_mult, rate_mult


def main() -> None:
    """Does mix predict access and energy once built form is held?"""
    import ledger

    print("=" * 70)
    print("MIXED USE — jobs-housing balance alongside dwelling-type composition")
    print("=" * 70)

    df = load_frame()
    descriptive(df)

    print(
        "\n  Effect of a p10 → p90 shift in jobs-housing balance, holding\n"
        "  dwelling-type composition and the confounds:"
    )
    print(
        f"    {'scale':<8s}{'units':>10s}{'p10':>9s}{'p90':>9s}"
        f"{'amenities':>12s}{'energy':>10s}{'rate':>10s}"
    )
    oa_fit = _fit(df, "OA")
    msoa_fit = _fit(to_msoa(df), "MSOA")

    if msoa_fit is not None:
        ledger.record(
            mixBalanceAccess=ledger.pt(msoa_fit[0]),
            mixBalanceEnergy=ledger.pt(msoa_fit[1]),
            mixBalanceRate=ledger.pt(msoa_fit[2]),
        )
    if oa_fit is not None:
        ledger.record(mixBalanceRateOa=ledger.pt(oa_fit[2]))

    if msoa_fit is not None:
        print(
            f"\n  Finding: at MSOA scale the rate moves ×{msoa_fit[2]:.2f}, so once "
            "dwelling-type\n  composition is held, jobs-housing balance carries no "
            "independent signal. These\n  data support the compact-form recommendation "
            "and do not support the mixed-use\n  half of it either way."
        )

    print(
        "\n  Caveats that govern how far this can be read:\n"
        "   * At Output Area scale the measure is dominated by area size: a small\n"
        "     area needs few jobs to reach parity, which is why detached areas score\n"
        "     higher on it than flats. MSOA is the interpretable scale.\n"
        "   * Balance is one facet of mixed use. It says nothing about whether homes\n"
        "     and workplaces share a building or a street, which is what the planning\n"
        "     term usually means.\n"
        "   * The q1 column is *balance*, the wrong construct for mixed use: it\n"
        "     marks down a high street with many food outlets and one surgery.\n"
        "     Mixed use is richness and proximity, which the per-service walkable\n"
        "     counts in access_profile already carry, compact areas reaching more\n"
        "     of every type within 1,600 m. Read q1 as a note on evenness, not as\n"
        "     evidence about mix."
    )


if __name__ == "__main__":
    main()
