"""
Access profile — network access by dwelling type (from ``oa_network_access``).

Sections [1]-[3] report dominant-type medians on the network ruler (so the walkable set
is a true subset of the drivable); section [4] reports the headline **compositional**
flat-vs-detached ratios used in ``paper/summary.md``:

  [1] WALKABLE CATCHMENT — amenities within a 1,600 m walk: per-service counts + the
      share with ZERO. The doorstep, reached without travel energy.
  [2] LIKE-FOR-LIKE DRIVABLE — amenities within the SAME fixed distance for every OA,
      by type at each ladder rung: pure density/connectivity, no catchment scaling.
  [3] DRIVABLE RATE — each OA at its OWN car-trip catchment (NTS mileage ÷ trips) ÷ its
      car-travel energy: the access-per-kWh rate (dominant-type median ~3.0×).
  [4] COMPOSITIONAL — Poisson flat-vs-detached contrasts (the headline): on foot a flat
      reaches ~27× the amenities of a detached, and returns ~3.9× the access per kWh
      (access advantage 1.26× × energy saving 3.07×).

Jobs are reported alongside the amenities (the total reachable jobs, summed over
workplaces), so the same flat-vs-detached comparison can be read for employment access.

Run:
    uv run python stats/oa_network_access.py   # build the cache first
    uv run python stats/access_profile.py
"""

from __future__ import annotations

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
from inference import (
    CLUSTER_COL,
    NAN_CI,
    cluster_bootstrap_multi,
    fmt_ci,
    log_contrast_ci,
)
from oa_access import DEST
from oa_data import load_and_aggregate
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
from travel_energy import KWH_PER_MILE_EV, fleet_intensity_kwh_per_mile

from urban_energy.paths import DATA_DIR

NET_CACHE = DATA_DIR / "statistics" / "oa_network_access.parquet"
TYPES = ["Flat", "Terraced", "Semi", "Detached"]
LABELS = {
    "gp": "GP",
    "pharmacy": "Pharmacy",
    "school": "School",
    "food": "Food",
    "grocery": "Grocery",
    "greenspace": "Greenspace",
}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _med(df: pd.DataFrame, col: str) -> dict[str, float]:
    return {
        t: float(_num(df.loc[df["dominant_type"] == t, col]).median()) for t in TYPES
    }


def _ratio(m: dict[str, float]) -> float:
    return m["Flat"] / m["Detached"] if m["Detached"] else float("nan")


def _comp_poisson(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    weight_col: str,
    cluster_col: str | None = None,
) -> GLMResultsWrapper | None:
    """No-intercept Poisson (log-link) GLM, household-weighted, on complete cases.

    The right estimator for non-negative access counts: the log link makes
    predictions strictly positive (a linear model predicts negative amenity
    counts for sparse detached areas), and with shares summing to 1 the
    Detached:Flat contrast ``exp(b_detached - b_flat)`` is invariant to the
    (uncentred) confounds.

    Households enter as ``var_weights`` (analytic weights), NOT ``freq_weights``:
    frequency weights would treat each OA as replicated by its household count,
    inflating the effective sample to tens of millions and collapsing every
    standard error. Analytic weights keep the effective sample at ~178k OAs, so
    the reported CIs are honest. When ``cluster_col`` is supplied the primary fit
    carries cluster-robust SEs, with the HC1 fit kept on ``._hc1``.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame.
    y_col : str
        Non-negative outcome (an access count or the access-per-kWh rate).
    x_cols : list of str
        Type-share fractions plus confounds. No constant (shares carry the level).
    weight_col : str
        Household-count column used as analytic weights.
    cluster_col : str or None, default None
        Column grouping OAs for cluster-robust SEs (typically ``LAD22CD``).

    Returns
    -------
    statsmodels results or None
        Fitted GLM results, or ``None`` if too few valid cases.
    """
    cols = [y_col, *x_cols, weight_col]
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    if cluster_col and cluster_col in df.columns:
        sub[cluster_col] = df[cluster_col]
    sub = sub.dropna()
    sub = sub[(sub[y_col] >= 0) & (sub[weight_col] > 0)]
    if len(sub) < len(x_cols) + 10:
        return None
    model = sm.GLM(
        sub[y_col],
        sub[x_cols],
        family=sm.families.Poisson(),
        var_weights=sub[weight_col],
    )
    if cluster_col and cluster_col in sub.columns:
        try:
            clustered = model.fit(
                cov_type="cluster", cov_kwds={"groups": sub[cluster_col]}
            )
            clustered._hc1 = model.fit(cov_type="HC1")
            clustered._n_clusters = int(sub[cluster_col].nunique())
            return clustered
        except Exception:
            pass
    return model.fit(cov_type="HC1")


def compositional_access(d: pd.DataFrame) -> None:
    """Option D on the access axis: a pure all-flat vs all-detached area.

    The no-intercept compositional idea from the energy axes, but fitted with a
    Poisson (log-link) GLM because access measures are non-negative, zero-inflated
    counts a linear model would push negative. Household-weighted and
    **income-controlled but not density-controlled** — density is the mechanism by
    which compact form delivers access, so netting it out would erase the very
    effect under study. Deprivation is held via the **income domain only**, not the
    overall IMD used on the energy axes: the overall index's geographic-barriers and
    living-environment sub-domains are themselves access measures, so controlling for
    them would absorb part of the effect under study. Each ratio is the predicted
    access of a pure all-flat area over a pure all-detached one (invariant to income;
    levels shown at mean income).

    Parameters
    ----------
    d : pandas.DataFrame
        The access frame assembled in :func:`main` (dwelling-type shares, network
        access columns, households, income, travel energy).
    """
    cf = _compositional_frame(d)
    income = [
        c for c in cf.columns if "imd_income" in c.lower() and "score" in c.lower()
    ][:1]

    measures = [
        ("amenities, walk 1,600 m", "net_total_1600"),
        ("jobs, walk 1,600 m", "net_jobs_1600"),
        ("people, walk 1,600 m", "net_pop_1600"),
        ("amenities, catchment", "net_amen"),
        ("jobs, catchment", "net_jobs_catch"),
        ("people, catchment", "net_pop_catch"),
        ("amenities, drive 25.6 km", "net_total_25600"),
        ("jobs, drive 25.6 km", "net_jobs_25600"),
        ("people, drive 25.6 km", "net_pop_25600"),
    ]
    print("\n  [4] COMPOSITIONAL (option D) — pure all-flat vs all-detached area")
    print(
        "      Poisson log-link · hh-weighted (var_weights) · income-only · NOT density"
    )
    print("      Flat:Det CIs: 95% LAD-clustered (delta method, share contrast).")
    print(f"  {'measure':<26s}{'Flat':>14s}{'Det':>14s}{'Flat:Det [95% CI]':>26s}")
    access_ratio = float("nan")
    access_ratio_ci: tuple[float, float, float, float] = NAN_CI
    for label, col in measures:
        cf["_y"] = _num(cf[col])
        m = _comp_poisson(
            cf, "_y", _SHARE_FRACS + income, "total_hh", cluster_col=CLUSTER_COL
        )
        if m is None:
            continue
        base = sum(float(m.params[c]) * _num(cf[c]).mean() for c in income)
        pf = float(np.exp(m.params["s_flat"] + base))
        pdet = float(np.exp(m.params["s_detached"] + base))
        ci = log_contrast_ci(m, "s_flat", "s_detached")  # flat:det
        if col == "net_amen":
            access_ratio = ci[0]  # catchment-amenity advantage, flat:det
            access_ratio_ci = ci
        print(f"  {label:<26s}{pf:>14,.1f}{pdet:>14,.1f}{fmt_ci(ci):>26s}")

    # Common-support companions for the on-foot gaps. The Poisson rows above are
    # pure-type predictions read at a 100%-of-type vertex (where several detached
    # service medians are 0); these grounded comparisons sit beside them. Jobs get
    # the same companions as amenities because their vertex-vs-support divergence
    # is the widest of the three measures.
    for m_label, m_col in (("amenity", "net_total_1600"), ("jobs", "net_jobs_1600")):
        walk = _num(cf[m_col])
        med_flat = float(walk[cf["dominant_type"] == "Flat"].median())
        med_det = float(walk[cf["dominant_type"] == "Detached"].median())
        dom_ratio = med_flat / med_det if med_det else float("nan")
        hi_flat = walk[_num(cf["s_flat"]) >= 0.5]
        hi_det = walk[_num(cf["s_detached"]) >= 0.5]
        sup_ratio = (
            float(hi_flat.median()) / float(hi_det.median())
            if len(hi_det) and hi_det.median()
            else float("nan")
        )
        print(
            f"\n  on-foot {m_label} gap companions (vs the pure-type Poisson above):"
            f"\n    dominant-type medians   Flat {med_flat:,.0f} / Det {med_det:,.0f} "
            f"= {dom_ratio:.1f}×"
            f"\n    support-restricted (≥50% share) medians = {sup_ratio:.1f}×  "
            f"(n≥50%-flat {len(hi_flat):,}, n≥50%-det {len(hi_det):,})"
        )

    # The access-per-kWh RATE is a ratio of two divisions (access ÷ energy): for a
    # flat area over a detached one it equals the access advantage (flat:det
    # catchment amenities) times the energy saving (det:flat car-travel energy).
    # This is a derived product of the two reported axes, reconstructable from the
    # tables — NOT a per-OA ratio modelled directly (which double-counts: an earlier
    # version did that and reported a spurious 6.3×).
    conf = (
        ["median_build_year"] + _deprivation_cols(cf) + _tenure_cols(cf) + _hdd_cols(cf)
    )
    cf["_le"] = np.log(_num(cf["transport"]).clip(lower=1))

    # Energy saving (det:flat car-travel energy), two conditioning sets so the
    # asymmetry with the income-only access side is explicit:
    #   full  — the headline energy axis (build age, IMD overall+income, tenure, HDD)
    #   inc   — income only, matched to the access side (like-for-like conditioning)
    me_full = _comp_ols(
        cf, "_le", _SHARE_FRACS + conf, "total_hh", cluster_col=CLUSTER_COL
    )
    me_inc = _comp_ols(
        cf, "_le", _SHARE_FRACS + income, "total_hh", cluster_col=CLUSTER_COL
    )
    er_full = (
        log_contrast_ci(me_full, "s_detached", "s_flat")
        if me_full is not None
        else NAN_CI
    )
    er_inc = (
        log_contrast_ci(me_inc, "s_detached", "s_flat")
        if me_inc is not None
        else NAN_CI
    )

    # Circularity-robust denominator (ROADMAP "rate circularity"): heat + travel at
    # the EV fleet intensity, so the denominator no longer scales with ICE driving
    # distance (the term that also drives the access catchment). Access rated
    # against this optimised combined energy is a lower, harder rate.
    fleet = fleet_intensity_kwh_per_mile(cf)
    cf["_le_alt"] = np.log(
        (
            _num(cf["building_kwh_per_hh"])
            + _num(cf["transport"]) * (KWH_PER_MILE_EV / fleet)
        ).clip(lower=1)
    )
    me_alt = _comp_ols(
        cf, "_le_alt", _SHARE_FRACS + conf, "total_hh", cluster_col=CLUSTER_COL
    )
    er_alt = (
        log_contrast_ci(me_alt, "s_detached", "s_flat")
        if me_alt is not None
        else NAN_CI
    )

    # All three rate variants multiply the income-only access advantage by an
    # energy saving; each variant only changes the energy conditioning, so one
    # cluster-bootstrap pass (shared access Poisson, three cheap WLS) gives all
    # three intervals.
    def _rates_stat(frame: pd.DataFrame) -> dict[str, float]:
        a = _comp_poisson(frame, "net_amen", _SHARE_FRACS + income, "total_hh")
        fl = fleet_intensity_kwh_per_mile(frame)
        frame = frame.assign(
            _le=np.log(
                pd.to_numeric(frame["transport"], errors="coerce").clip(lower=1)
            ),
            _le_alt=np.log(
                (
                    _num(frame["building_kwh_per_hh"])
                    + _num(frame["transport"]) * (KWH_PER_MILE_EV / fl)
                ).clip(lower=1)
            ),
        )
        e_full = _comp_ols(frame, "_le", _SHARE_FRACS + conf, "total_hh")
        e_inc = _comp_ols(frame, "_le", _SHARE_FRACS + income, "total_hh")
        e_alt = _comp_ols(frame, "_le_alt", _SHARE_FRACS + conf, "total_hh")
        nan = float("nan")
        if a is None or e_full is None or e_inc is None or e_alt is None:
            return {"headline": nan, "harmonised": nan, "circularity": nan}
        ar = float(np.exp(a.params["s_flat"] - a.params["s_detached"]))
        er_f = float(np.exp(e_full.params["s_detached"] - e_full.params["s_flat"]))
        er_i = float(np.exp(e_inc.params["s_detached"] - e_inc.params["s_flat"]))
        er_a = float(np.exp(e_alt.params["s_detached"] - e_alt.params["s_flat"]))
        return {
            "headline": ar * er_f,
            "harmonised": ar * er_i,
            "circularity": ar * er_a,
        }

    rc = cluster_bootstrap_multi(cf, _rates_stat)
    print(
        f"\n  access-per-kWh RATE (headline) = access advantage × energy saving"
        f"\n    = {access_ratio:.2f} × {er_full[0]:.2f} = {rc['headline'][0]:.2f}×  "
        f"[95% CI {rc['headline'][1]:.2f}, {rc['headline'][2]:.2f}, cluster bootstrap]"
    )
    print(
        f"      access advantage {fmt_ci(access_ratio_ci)} · "
        f"energy saving (full conf) {fmt_ci(er_full)}"
    )
    print(
        f"  harmonised rate (both income-only, like-for-like conditioning) "
        f"= {access_ratio:.2f} × {er_inc[0]:.2f} = {rc['harmonised'][0]:.2f}×  "
        f"[95% CI {rc['harmonised'][1]:.2f}, {rc['harmonised'][2]:.2f}]"
    )
    print(
        f"  circularity-robust rate (access ÷ [heat + electrified travel]) "
        f"= {access_ratio:.2f} × {er_alt[0]:.2f} = {rc['circularity'][0]:.2f}×  "
        f"[95% CI {rc['circularity'][1]:.2f}, {rc['circularity'][2]:.2f}]"
    )
    print(
        "  non-parametric companion: dominant-type median access/kWh — see the "
        "'access / kWh' row in section [3] above."
    )


def main() -> None:
    """Print the three access numbers from the network curve cache."""
    df = load_and_aggregate().reset_index(drop=True)
    if not NET_CACHE.exists():
        print(
            f"\n  [network cache not found ({NET_CACHE.name}) — build it first:\n"
            "     uv run python stats/oa_network_access.py]"
        )
        return
    net = pd.read_parquet(NET_CACHE)
    d = df.merge(net, left_on="OA21CD", right_index=True, how="left", validate="m:1")
    d["transport"] = _num(d["transport_kwh_per_hh_total_est"])
    ladder = sorted(
        int(c.rsplit("_", 1)[1]) for c in net.columns if c.startswith("net_total_")
    )

    # ---- [1] WALKABLE CATCHMENT (network, within 1,600 m) ----
    print("\n  [1] WALKABLE — network count within 1,600 m, by type (the doorstep)")
    print(f"  {'service':<11s}{'Flat':>8s}{'Det':>8s}{'%Det=0':>9s}")
    for svc in DEST:
        m = _med(d, f"net_{svc}_1600")
        zdet = (
            _num(d.loc[d["dominant_type"] == "Detached", f"net_{svc}_1600"]).fillna(0)
            == 0
        ).mean() * 100
        print(
            f"  {LABELS[svc]:<11s}{m['Flat']:>8.0f}{m['Detached']:>8.0f}{zdet:>8.0f}%"
        )
    d["walk_basket"] = sum((_num(d[f"net_{s}_1600"]) > 0).astype(int) for s in DEST)
    mb = {t: float(d.loc[d["dominant_type"] == t, "walk_basket"].mean()) for t in TYPES}
    print(
        f"  walkable basket (of {len(DEST)} on foot, mean):  "
        + "   ".join(f"{t} {mb[t]:.1f}" for t in TYPES)
    )
    mj = _med(d, "net_jobs_1600")
    print(
        "  jobs within 1,600 m (median):  "
        + "  ".join(f"{t} {mj[t]:,.0f}" for t in TYPES)
        + f"   (Flat:Det {_ratio(mj):.1f}x)"
    )
    mp = _med(d, "net_pop_1600")
    print(
        "  people within 1,600 m (median):  "
        + "  ".join(f"{t} {mp[t]:,.0f}" for t in TYPES)
        + f"   (Flat:Det {_ratio(mp):.1f}x)"
    )

    # ---- [1b] STRUCTURAL INTENSITY (network + population, within 1,600 m) ----
    print("\n  [1b] STRUCTURAL INTENSITY — by type (compactness → complexity)")
    print(f"  {'':16s}" + "".join(f"{t:>10s}" for t in TYPES) + f"{'Flat:Det':>10s}")
    for label, col, fmt in [
        ("closeness", "net_closeness_1600", "{:>10.2f}"),
        ("node density", "net_density_1600", "{:>10.0f}"),
        ("pop /ha", "pop_density", "{:>10.1f}"),
    ]:
        m = _med(d, col)
        print(
            f"  {label:<16s}"
            + "".join(fmt.format(m[t]) for t in TYPES)
            + f"{_ratio(m):>8.1f}x"
        )

    # ---- [2] LIKE-FOR-LIKE DRIVABLE (same network distance for both) ----
    print("\n  [2] LIKE-FOR-LIKE — amenities within the SAME network distance, by type")
    print(
        f"  {'dist (m)':<10s}"
        + "".join(f"{t:>10s}" for t in TYPES)
        + f"{'Flat:Det':>10s}"
    )
    for dist in ladder:
        m = _med(d, f"net_total_{dist}")
        print(
            f"  {dist:<10d}"
            + "".join(f"{m[t]:>10.0f}" for t in TYPES)
            + f"{_ratio(m):>8.1f}x"
        )

    # ---- [2b] LIKE-FOR-LIKE JOBS (same network distance, weighted sum) ----
    print(
        "\n  [2b] LIKE-FOR-LIKE JOBS — jobs reachable within the SAME network distance"
    )
    print(
        f"  {'dist (m)':<10s}"
        + "".join(f"{t:>12s}" for t in TYPES)
        + f"{'Flat:Det':>10s}"
    )
    for dist in ladder:
        m = _med(d, f"net_jobs_{dist}")
        print(
            f"  {dist:<10d}"
            + "".join(f"{m[t]:>12,.0f}" for t in TYPES)
            + f"{_ratio(m):>8.1f}x"
        )

    # ---- [2c] LIKE-FOR-LIKE PEOPLE (same network distance, resident population) ----
    print(
        "\n  [2c] LIKE-FOR-LIKE PEOPLE — residents reachable within the SAME "
        "network distance"
    )
    print(
        f"  {'dist (m)':<10s}"
        + "".join(f"{t:>12s}" for t in TYPES)
        + f"{'Flat:Det':>10s}"
    )
    for dist in ladder:
        m = _med(d, f"net_pop_{dist}")
        print(
            f"  {dist:<10d}"
            + "".join(f"{m[t]:>12,.0f}" for t in TYPES)
            + f"{_ratio(m):>8.1f}x"
        )

    # ---- [3] DRIVABLE RATE (each OA at its own catchment) ----
    d["trip_km"] = _num(d["trip_m"]) / 1000
    d["amenities"] = _num(d["net_amen"])
    d["rate"] = d["amenities"] / d["transport"].replace(0, np.nan)
    print("\n  [3] DRIVABLE RATE — amenities per kWh, at each OA's own catchment")
    print(f"  {'':14s}" + "".join(f"{t:>10s}" for t in TYPES) + f"{'Flat:Det':>10s}")
    for label, col in [
        ("trip dist (km)", "trip_km"),
        ("amenities", "amenities"),
        ("travel kWh", "transport"),
        ("access / kWh", "rate"),
    ]:
        m = _med(d, col)
        print(
            f"  {label:<14s}"
            + "".join(f"{m[t]:>10.1f}" for t in TYPES)
            + f"{_ratio(m):>8.1f}x"
        )
    # jobs and people at the same own-catchment radius (weighted sums)
    d["jobs_catch"] = _num(d["net_jobs_catch"])
    d["jobs_rate"] = d["jobs_catch"] / d["transport"].replace(0, np.nan)
    d["pop_catch"] = _num(d["net_pop_catch"])
    for label, col, fmt in [
        ("jobs (catchment)", "jobs_catch", "{:>10,.0f}"),
        ("jobs / kWh", "jobs_rate", "{:>10,.0f}"),
        ("people (catch)", "pop_catch", "{:>10,.0f}"),
    ]:
        m = _med(d, col)
        print(
            f"  {label:<14s}"
            + "".join(fmt.format(m[t]) for t in TYPES)
            + f"{_ratio(m):>8.1f}x"
        )

    compositional_access(d)


if __name__ == "__main__":
    main()
