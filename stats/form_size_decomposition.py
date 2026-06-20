"""
Form vs size decomposition — untangling the dwelling-type energy gradient.

The Form (building-energy) gap between flats and detached houses is partly a
*size* effect: detached dwellings are bigger and hold more people, and low-density
form *causes* both. So the raw per-household gap is the **total effect** of form
(fabric + induced size + occupancy), while the gap *holding size fixed* is the
**direct effect** (fabric/exposure only). Neither is "the" answer — the pair
brackets the truth, and the difference between them is the size-mediation channel.

This module makes that explicit, because no fixed denominator is a valid
inferential unit — each silently nails an elasticity the data reject:

* **Per person** forces energy ∝ people (occupancy elasticity = 1). But heating
  is largely a property of the *building*, not its occupants, so the true
  household-size elasticity of energy is well below 1 (economies of scale —
  Huebner & Shipworth 2017; Druckman & Jackson 2008). Per-person therefore
  over-credits the larger households that self-select into detached homes,
  manufacturing apparent parity. It is kept here only as a *descriptive* lens.
* **Per m²** forces energy ∝ floor area — false via surface-to-volume geometry
  (the elasticity is < 1), so per-m² mechanically flatters large dwellings.

The honest tool is a per-dwelling regression that enters family size and floor
area as FREE regressors (their elasticities estimated, not assumed) and reads off
the partial form effect at equal family size and equal size. This script reports:

1. A **bivariate floor-area elasticity** of building energy — the empirical
   kill-shot for per-m² (elasticity < 1 ⇒ kWh/m² declines with dwelling size).
2. A **descriptive panel** (per-hh / per-person / per-m²) by dominant type, now
   framed as presentation units, not competing truths.
3. A **floor-area-quintile stratification** — non-parametric "like-for-like"
   size control, with cell counts to expose the limited Flat/Detached common
   support.
4. The **total → direct regression ladder** on one fixed common-support sample:
   form → + family size → + dwelling size, reporting the modelled Detached:Flat
   per-household ratio at each step and the share of the gap mediated by size,
   plus the estimated household-size elasticity (the per-person validity check).
5. The **compositional (no-intercept) ladder** (option D) — the same mediation,
   but read from every dwelling-type share at once (fractions summing to 1, with
   an "other" residual), household-weighted, with no dominant-type label and no
   dropped reference category, so each coefficient is a pure-type mean.

Run:
    uv run python stats/form_size_decomposition.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Reuse the canonical loader and OLS plumbing (data construction is already
# correct: genuine per-household energy, EPC floor-area merge, dominant_type).
from oa_data import _run_ols, load_and_aggregate

# Type shares are in percent (0–100); semi-detached is the omitted reference.
_FORM = ["pct_detached", "pct_flat", "pct_terraced"]
_DV = "log_building_kwh_per_hh"


def _imd_income_col(df: pd.DataFrame) -> list[str]:
    """Return the IMD income-score column name as a one-element list (or empty)."""
    hits = [c for c in df.columns if "imd_income" in c.lower() and "score" in c.lower()]
    return hits[:1]


def _imd_overall_col(df: pd.DataFrame) -> list[str]:
    """Return the overall IMD-score column name as a one-element list (or empty)."""
    hits = [
        c for c in df.columns if "imd_overall" in c.lower() and "score" in c.lower()
    ]
    return hits[:1]


def _deprivation_cols(df: pd.DataFrame) -> list[str]:
    """Return the deprivation controls present: overall IMD plus the income domain.

    The overall Index of Multiple Deprivation (IoD25) is the broad deprivation
    confound; the income domain is retained as the sharper material-resources
    control. Both are nuisance covariates, so their collinearity is harmless — it
    inflates only their own standard errors, never the form coefficient.
    """
    return _imd_overall_col(df) + _imd_income_col(df)


def _tenure_cols(df: pd.DataFrame) -> list[str]:
    """Return the tenure-share confound columns present (social + private rented).

    Owner-occupation is the omitted reference, so the two renting shares carry the
    energy-relevant tenure gradient (heating control, ability to upgrade fabric).
    """
    return [c for c in ("pct_social_rented", "pct_private_rented") if c in df.columns]


def _hdd_cols(df: pd.DataFrame) -> list[str]:
    """Return the climate confound column if present (annual heating-degree-days).

    Populated by ``data/process_climate.py`` from HadUK-Grid; absent until that
    has run, so the ladder runs with or without the climate control.
    """
    return ["hdd"] if "hdd" in df.columns else []


def _detached_flat_ratio(
    model: object | None, params: pd.Series | None = None
) -> float:
    """
    Modelled Detached:Flat per-household energy ratio from form coefficients.

    With type shares in percent and semi as reference, an all-detached OA differs
    from an all-flat OA by ``100 * (beta_detached - beta_flat)`` in log energy.

    Parameters
    ----------
    model : statsmodels results or None
        Fitted model exposing ``.params``. Ignored if ``params`` is given.
    params : pandas.Series or None
        Coefficient series (used directly when supplied).

    Returns
    -------
    float
        ``exp(100 * (beta_detached - beta_flat))`` — NaN if coefficients absent.
    """
    p = params if params is not None else getattr(model, "params", None)
    if p is None or "pct_detached" not in p or "pct_flat" not in p:
        return float("nan")
    return float(np.exp(100.0 * (p["pct_detached"] - p["pct_flat"])))


def bivariate_floor_area_elasticity(lsoa: pd.DataFrame) -> None:
    """
    Report the floor-area elasticity of per-household building energy.

    An elasticity below 1 is the empirical reason per-m² is misleading: if energy
    rises less than proportionally with floor area, kWh/m² *falls* as dwellings
    get bigger, handing large/detached dwellings a mechanical advantage.
    """
    print("=" * 70)
    print("1. FLOOR-AREA ELASTICITY OF BUILDING ENERGY (the per-m² kill-shot)")
    print("=" * 70)

    df = lsoa.copy()
    df["log_floor_area"] = np.log(
        pd.to_numeric(df["oa_median_floor_area_m2"], errors="coerce").clip(lower=1)
    )
    m = _run_ols(df, _DV, ["log_floor_area"], "elasticity")
    if m is None:
        print("  insufficient data")
        return
    e = float(m.params["log_floor_area"])
    print(f"\n  d log(kWh/hh) / d log(m²) = {e:.3f}  (N = {int(m.nobs):,})")
    print(f"  → energy scales as floor_area^{e:.2f}")
    if e < 1:
        print(
            f"  → kWh/m² scales as floor_area^{e - 1:.2f} (DECLINES with size): "
            "per-m² mechanically favours larger dwellings."
        )
    else:
        print("  → elasticity ≥ 1: per-m² would not mechanically favour size.")


def descriptive_panel(lsoa: pd.DataFrame) -> None:
    """Median energy by dominant type under each presentation unit."""
    print("\n" + "=" * 70)
    print("2. DESCRIPTIVE PANEL — presentation units, not competing truths")
    print("=" * 70)

    units = [
        ("per household", "building_kwh_per_hh", ",.0f"),
        ("per person", "building_kwh_per_person", ",.0f"),
        ("per m²", "building_kwh_per_m2", ".1f"),
    ]
    types = ["Flat", "Terraced", "Semi", "Detached"]
    have = [(lbl, col, fmt) for lbl, col, fmt in units if col in lsoa.columns]

    header = f"  {'Dominant type':<14s}" + "".join(f"{lbl:>16s}" for lbl, _, _ in have)
    print("\n" + header)
    print("  " + "-" * (14 + 16 * len(have)))
    medians: dict[str, dict[str, float]] = {}
    for t in types:
        sub = lsoa[lsoa["dominant_type"] == t]
        if sub.empty:
            continue
        row = f"  {t:<14s}"
        medians[t] = {}
        for _lbl, col, fmt in have:
            v = pd.to_numeric(sub[col], errors="coerce").median()
            medians[t][col] = v
            row += f"{v:>16{fmt}}"
        print(row)

    # Flat→Detached gradient under each unit
    if "Flat" in medians and "Detached" in medians:
        print(f"\n  {'Flat→Detached gradient:':<24s}", end="")
        for _lbl, col, _fmt in have:
            f = medians["Flat"].get(col)
            d = medians["Detached"].get(col)
            ratio = d / f if (f and d) else float("nan")
            print(f"  {col.split('_')[-1]}={ratio:.2f}×", end="")
        print(
            "\n  (DESCRIPTIVE ONLY. per-hh rises toward detached; per-person"
            " compresses\n  toward parity because detached house more people"
            " (γ<1, not efficiency);\n  per-m² reverses via the floor-area"
            " artefact. The inferential gap is the\n  family-size-controlled"
            " ladder below, not any of these ratios.)"
        )


def quintile_stratification(lsoa: pd.DataFrame) -> None:
    """Flat vs Detached per-household energy within floor-area quintiles."""
    print("\n" + "=" * 70)
    print("3. FLOOR-AREA-QUINTILE STRATIFICATION — non-parametric size control")
    print("=" * 70)

    df = lsoa.copy()
    fa = pd.to_numeric(df["oa_median_floor_area_m2"], errors="coerce")
    valid = fa.notna() & df["building_kwh_per_hh"].notna()
    df = df[valid].copy()
    df["fa_q"] = pd.qcut(fa[valid], 5, labels=[f"Q{i}" for i in range(1, 6)])

    print(
        f"\n  {'Quintile':<8s} {'floor m²':>10s} "
        f"{'Flat n':>8s} {'Flat kWh':>10s} "
        f"{'Det n':>8s} {'Det kWh':>10s} {'Det:Flat':>9s}"
    )
    print("  " + "-" * 66)
    for q in [f"Q{i}" for i in range(1, 6)]:
        sub = df[df["fa_q"] == q]
        fa_med = pd.to_numeric(sub["oa_median_floor_area_m2"], errors="coerce").median()
        flat = sub[sub["dominant_type"] == "Flat"]["building_kwh_per_hh"]
        det = sub[sub["dominant_type"] == "Detached"]["building_kwh_per_hh"]
        f_n, d_n = len(flat), len(det)
        f_m = flat.median() if f_n else float("nan")
        d_m = det.median() if d_n else float("nan")
        ratio = d_m / f_m if (f_n and d_n and f_m) else float("nan")
        print(
            f"  {q:<8s} {fa_med:>10.0f} "
            f"{f_n:>8d} {f_m:>10,.0f} "
            f"{d_n:>8d} {d_m:>10,.0f} "
            f"{ratio:>8.2f}×"
        )
    print(
        "\n  Within-quintile Det:Flat ratios isolate form from size. Thin cells in\n"
        "  the tails (few large flats / small detached) = limited common support:\n"
        "  in the real stock, low-density form and large dwellings co-occur by design."
    )


def regression_ladder(lsoa: pd.DataFrame) -> None:
    """
    Total → direct mediation ladder on one fixed common-support sample.

    M0 (total)  : form + confounds (build year, deprivation, tenure, climate)
    M1          : + family size (log household size — a FREE elasticity)
    M2 (direct) : + dwelling size (log floor area)

    Household size enters as ``log_hh_size`` with a free coefficient, NOT as a
    per-person denominator: per-person division silently nails the occupancy
    elasticity to 1, whereas heating is largely a property of the building, so the
    true elasticity is well below 1 (economies of scale — Huebner & Shipworth
    2017; Druckman & Jackson 2008). The estimated elasticity is printed at M1; if
    it is far from 1, that is the quantitative reason per-person is not a valid
    inferential unit. The modelled Detached:Flat per-household ratio is reported at
    each step; M1 is the family-size-controlled form gap, M2 the size-held direct
    (fabric/exposure) gap. Confounds are held throughout; family/dwelling size are
    mediators, in only for the direct effect.
    """
    print("\n" + "=" * 70)
    print("4. TOTAL → DIRECT REGRESSION LADDER (fixed common-support sample)")
    print("=" * 70)

    df = lsoa.copy()
    df["log_floor_area"] = np.log(
        pd.to_numeric(df["oa_median_floor_area_m2"], errors="coerce").clip(lower=1)
    )
    df["log_hh_size"] = np.log(
        pd.to_numeric(df["avg_hh_size"], errors="coerce").clip(lower=1)
    )
    confounds = (
        ["median_build_year"] + _deprivation_cols(df) + _tenure_cols(df) + _hdd_cols(df)
    )
    occupancy = ["log_hh_size"]
    size = ["log_floor_area"]

    # One fixed sample: complete cases on EVERY variable the ladder will touch,
    # so coefficient movement reflects mediator inclusion, not sample change.
    all_vars = [_DV, *_FORM, *confounds, *occupancy, *size]
    sample = df.dropna(subset=all_vars).copy()
    print(
        f"\n  Common sample: N = {len(sample):,} OAs "
        f"(complete cases on form, {', '.join(confounds)}, family size, floor area)"
    )
    print(f"  Confounds held throughout: {confounds}")

    steps = [
        ("M0 total  (form + confounds)", _FORM + confounds),
        ("M1        (+ family size)", _FORM + confounds + occupancy),
        ("M2 direct (+ dwelling size)", _FORM + confounds + occupancy + size),
    ]
    print(
        f"\n  {'Model':<30s} {'R²':>7s} {'β_det':>8s} {'β_flat':>8s} "
        f"{'Det:Flat':>9s} {'floor e':>8s}"
    )
    print("  " + "-" * 74)
    ratios: list[float] = []
    hh_elasticity = float("nan")
    for label, xcols in steps:
        m = _run_ols(sample, _DV, xcols, label)
        if m is None:
            print(f"  {label:<30s} SKIPPED")
            continue
        bdet = float(m.params.get("pct_detached", np.nan))
        bflat = float(m.params.get("pct_flat", np.nan))
        ratio = _detached_flat_ratio(m)
        ratios.append(ratio)
        if "log_hh_size" in m.params and np.isnan(hh_elasticity):
            hh_elasticity = float(m.params["log_hh_size"])
        floor_e = (
            float(m.params["log_floor_area"])
            if "log_floor_area" in m.params
            else np.nan
        )
        floor_str = f"{floor_e:>8.3f}" if not np.isnan(floor_e) else f"{'—':>8s}"
        print(
            f"  {label:<30s} {m.rsquared:>7.3f} {bdet:>8.4f} {bflat:>8.4f} "
            f"{ratio:>8.2f}× {floor_str}"
        )

    if not np.isnan(hh_elasticity):
        print(
            f"\n  Household-size elasticity of heat: γ = {hh_elasticity:.2f} "
            f"(per-person division would impose γ = 1)."
        )
        print(
            "  γ < 1 ⇒ heat is sub-linear in occupants (economies of scale), so\n"
            "  per-person flatters large households — the form gap is read off the\n"
            "  family-size-controlled M1, not a per-person ratio."
        )

    if len(ratios) >= 3 and ratios[0] > 1:
        total_gap = ratios[0] - 1.0
        direct_gap = ratios[-1] - 1.0
        mediated = (total_gap - direct_gap) / total_gap if total_gap else float("nan")
        print(
            f"\n  Total form gap: {ratios[0]:.2f}×  →  family-size-held "
            f"{ratios[1]:.2f}×  →  size-held direct {ratios[-1]:.2f}×"
        )
        print(
            f"  Family size + dwelling size mediate {mediated:.0%} of the gap; "
            f"the residual {direct_gap:+.0%} is direct fabric/exposure."
        )

    if _hdd_cols(df):
        print("\n  Direct term is net of build era, income, tenure and climate (HDD).")
    else:
        print(
            "\n  Caveat: climate (heating-degree-days) is not yet in the dataset\n"
            "  (HadUK-Grid pending — run data/process_climate.py once the .nc lands);\n"
            "  it is the remaining confound, and would further adjust the direct term."
        )


# ---------------------------------------------------------------------------
# Option D — compositional (no-intercept) ecological regression
# ---------------------------------------------------------------------------
# Dwelling-type shares as fractions that sum to 1 (an explicit "other" closes the
# composition) enter a no-intercept, household-weighted regression, so each
# coefficient is the (log) per-household energy of a pure area of that type and
# the whole dwelling mix is used — no single dominant-type label, no dropped
# reference category. The Detached:Flat ratio is exp(b_detached - b_flat), and is
# invariant to the (uncentred) confound values.

_SHARE_FRACS = ["s_flat", "s_terraced", "s_semi", "s_detached", "s_other"]


def _compositional_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add dwelling-type share fractions that sum to 1 for every OA.

    Parameters
    ----------
    df : pandas.DataFrame
        OA frame carrying the ``pct_*`` dwelling-type shares (percent).

    Returns
    -------
    pandas.DataFrame
        ``df`` with ``s_flat`` / ``s_terraced`` / ``s_semi`` / ``s_detached`` and
        a residual ``s_other`` that closes the composition, row-normalised to sum
        to exactly 1.
    """
    out = df.copy()
    for frac, pct in [
        ("s_flat", "pct_flat"),
        ("s_terraced", "pct_terraced"),
        ("s_semi", "pct_semi"),
        ("s_detached", "pct_detached"),
    ]:
        out[frac] = pd.to_numeric(out[pct], errors="coerce") / 100
    four = out[["s_flat", "s_terraced", "s_semi", "s_detached"]].sum(axis=1)
    out["s_other"] = (1.0 - four).clip(lower=0)
    rowsum = out[_SHARE_FRACS].sum(axis=1).replace(0, np.nan)
    out[_SHARE_FRACS] = out[_SHARE_FRACS].div(rowsum, axis=0)
    return out


def _comp_ols(
    df: pd.DataFrame, y_col: str, x_cols: list[str], weight_col: str
) -> sm.regression.linear_model.RegressionResultsWrapper | None:
    """No-intercept WLS (household-weighted, HC1 robust) on complete cases.

    Parameters
    ----------
    df : pandas.DataFrame
        Source frame.
    y_col : str
        Dependent variable (log per-household energy).
    x_cols : list of str
        Regressors — the type-share fractions plus any confounds/mediators. No
        constant is added; the shares carry the level.
    weight_col : str
        Household-count column used as regression weights.

    Returns
    -------
    statsmodels results or None
        Fitted WLS results, or ``None`` if too few complete cases.
    """
    cols = [y_col, *x_cols, weight_col]
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < len(x_cols) + 10:
        return None
    return sm.WLS(sub[y_col], sub[x_cols], weights=sub[weight_col]).fit(cov_type="HC1")


def compositional_ladder(lsoa: pd.DataFrame) -> None:
    """Option D: the no-intercept compositional mediation ladder.

    Mirrors :func:`regression_ladder` (same total → direct mediation), but
    parameterised with every dwelling-type share at once and no intercept, and
    household-weighted. The Detached:Flat per-household ratio at each rung is
    ``exp(b_detached - b_flat)``; its attenuation as occupancy and floor area
    enter is the size-mediation channel. A pure-type predicted-heat row checks
    the fit against the descriptive medians.

    Parameters
    ----------
    lsoa : pandas.DataFrame
        National OA frame from :func:`oa_data.load_and_aggregate`.
    """
    print("\n" + "=" * 70)
    print("5. COMPOSITIONAL NO-INTERCEPT LADDER — option D (all shares, hh-weighted)")
    print("=" * 70)

    df = _compositional_frame(lsoa)
    df["log_floor_area"] = np.log(
        pd.to_numeric(df["oa_median_floor_area_m2"], errors="coerce").clip(lower=1)
    )
    df["_log_travel"] = np.log(
        pd.to_numeric(df["transport_kwh_per_hh_total_est"], errors="coerce").clip(
            lower=1
        )
    )
    df["_log_total"] = np.log(
        (
            pd.to_numeric(df["building_kwh_per_hh"], errors="coerce")
            + pd.to_numeric(df["transport_kwh_per_hh_total_est"], errors="coerce")
        ).clip(lower=1)
    )
    df["log_hh_size"] = np.log(
        pd.to_numeric(df["avg_hh_size"], errors="coerce").clip(lower=1)
    )
    confounds = (
        ["median_build_year"] + _deprivation_cols(df) + _tenure_cols(df) + _hdd_cols(df)
    )
    occupancy = ["log_hh_size"]
    size = ["log_floor_area"]

    keep = [_DV, *_SHARE_FRACS, *confounds, *occupancy, *size, "total_hh"]
    sample = df.dropna(subset=keep).copy()
    print(f"\n  Common sample: N = {len(sample):,} OAs (household-weighted, HC1 SEs)")
    print(f"  Confounds held throughout: {confounds}")

    steps = [
        ("D0 total  (shares + confounds)", _SHARE_FRACS + confounds),
        ("D1        (+ family size)", _SHARE_FRACS + confounds + occupancy),
        ("D2 direct (+ dwelling size)", _SHARE_FRACS + confounds + occupancy + size),
    ]
    print(f"\n  {'Model':<32s} {'Det:Flat':>9s} {'b_flat':>9s} {'b_det':>9s}")
    print("  " + "-" * 62)
    ratios: list[float] = []
    hh_elasticity = float("nan")
    m0 = None
    for label, xcols in steps:
        m = _comp_ols(sample, _DV, xcols, "total_hh")
        if m is None:
            print(f"  {label:<32s} SKIPPED")
            continue
        bflat, bdet = float(m.params["s_flat"]), float(m.params["s_detached"])
        ratio = float(np.exp(bdet - bflat))
        ratios.append(ratio)
        if "log_hh_size" in m.params and np.isnan(hh_elasticity):
            hh_elasticity = float(m.params["log_hh_size"])
        if m0 is None:
            m0 = m
        print(f"  {label:<32s} {ratio:>8.2f}× {bflat:>9.3f} {bdet:>9.3f}")

    if not np.isnan(hh_elasticity):
        print(
            f"\n  Household-size elasticity of heat: γ = {hh_elasticity:.2f} "
            f"(per-person would impose γ = 1; economies of scale ⇒ γ < 1)."
        )

    if len(ratios) >= 3 and ratios[0] > 1:
        total_gap, direct_gap = ratios[0] - 1.0, ratios[-1] - 1.0
        mediated = (total_gap - direct_gap) / total_gap if total_gap else float("nan")
        print(
            f"\n  Total form gap: {ratios[0]:.2f}×  →  family-size-held "
            f"{ratios[1]:.2f}×  →  size-held direct {ratios[-1]:.2f}×"
        )
        print(
            f"  Family + dwelling size mediate {mediated:.0%}; "
            f"residual {direct_gap:+.0%} is direct fabric/exposure."
        )

    print("\n  Same estimator on each energy axis (shares + confounds):")
    for label, ycol in (
        ("heat", _DV),
        ("travel", "_log_travel"),
        ("total", "_log_total"),
    ):
        m = _comp_ols(sample, ycol, _SHARE_FRACS + confounds, "total_hh")
        if m is not None:
            r = float(np.exp(float(m.params["s_detached"] - m.params["s_flat"])))
            print(f"    {label:<7s} Det:Flat {r:.2f}×")

    if m0 is not None:
        base = sum(
            float(m0.params[c]) * pd.to_numeric(sample[c], errors="coerce").mean()
            for c in confounds
        )
        pred = {
            t: float(np.exp(float(m0.params[s]) + base))
            for t, s in (("Flat", "s_flat"), ("Detached", "s_detached"))
        }
        med = {
            t: float(
                pd.to_numeric(
                    sample.loc[sample["dominant_type"] == t, "building_kwh_per_hh"],
                    errors="coerce",
                ).median()
            )
            for t in ("Flat", "Detached")
        }
        print(
            "\n  D0 predicted heat at mean confounds (kWh/hh):  "
            f"Flat {pred['Flat']:,.0f}   Detached {pred['Detached']:,.0f}"
        )
        print(
            "  vs descriptive medians (same sample):          "
            f"Flat {med['Flat']:,.0f}   Detached {med['Detached']:,.0f}"
        )


def main() -> None:
    """Run the full form-vs-size decomposition on the national OA dataset."""
    lsoa = load_and_aggregate()
    print(f"\nLoaded {len(lsoa):,} OAs\n")

    if "oa_median_floor_area_m2" not in lsoa.columns:
        print("ERROR: oa_epc.parquet not merged — run data/aggregate_epc_oa.py first.")
        return

    bivariate_floor_area_elasticity(lsoa)
    descriptive_panel(lsoa)
    quintile_stratification(lsoa)
    regression_ladder(lsoa)
    compositional_ladder(lsoa)


if __name__ == "__main__":
    main()
