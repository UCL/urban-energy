"""
Form vs size decomposition — untangling the dwelling-type energy gradient.

The Form (building-energy) gap between flats and detached houses is partly a
*size* effect: detached dwellings are bigger and hold more people, and low-density
form *causes* both. So the raw per-household gap is the **total effect** of form
(fabric + induced size + occupancy), while the gap *holding size fixed* is the
**direct effect** (fabric/exposure only). Neither is "the" answer — the pair
brackets the truth, and the difference between them is the size-mediation channel.

This module makes that explicit, because no single denominator can:

* **Per household** forces energy ∝ households (occupancy elasticity = 1).
* **Per person** forces energy ∝ people.
* **Per m²** forces energy ∝ floor area — false via surface-to-volume geometry
  (the elasticity is < 1), so per-m² mechanically flatters large dwellings.

Each ratio is a one-covariate regression with the slope nailed to 1. The honest
tool is a regression that conditions on the drivers and reads off the partial
form effect. This script reports:

1. A **bivariate floor-area elasticity** of building energy — the empirical
   kill-shot for per-m² (elasticity < 1 ⇒ kWh/m² declines with dwelling size).
2. A **descriptive panel** (per-hh / per-person / per-m²) by dominant type, now
   framed as presentation units, not competing truths.
3. A **floor-area-quintile stratification** — non-parametric "like-for-like"
   size control, with cell counts to expose the limited Flat/Detached common
   support.
4. The **total → direct regression ladder** on one fixed common-support sample:
   form → + occupancy → + dwelling size, reporting the modelled Detached:Flat
   per-household ratio at each step and the share of the gap mediated by size.

Run:
    uv run python stats/form_size_decomposition.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Reuse the canonical loader and OLS plumbing (data construction is already
# correct: genuine per-household energy, EPC floor-area merge, dominant_type).
from proof_of_concept_oa import _run_ols, load_and_aggregate

# Type shares are in percent (0–100); semi-detached is the omitted reference.
_FORM = ["pct_detached", "pct_flat", "pct_terraced"]
_DV = "log_building_kwh_per_hh"


def _imd_income_col(df: pd.DataFrame) -> list[str]:
    """Return the IMD income-score column name as a one-element list (or empty)."""
    hits = [c for c in df.columns if "imd_income" in c.lower() and "score" in c.lower()]
    return hits[:1]


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
            f, d = medians["Flat"].get(col), medians["Detached"].get(col)
            ratio = d / f if f else float("nan")
            print(f"  {col.split('_')[-1]}={ratio:.2f}×", end="")
        print(
            "\n  (per-hh/per-person rise toward detached; per-m² reverses — "
            "the size artefact, not efficiency.)"
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

    M0 (total)  : form + confounds (build year, income)
    M1          : + occupancy (avg_hh_size)
    M2 (direct) : + dwelling size (log floor area)

    The modelled Detached:Flat per-household ratio is reported at each step; its
    attenuation from M0 to M2 is the share of the form gap mediated by size and
    occupancy. Confounds (build era, income) are held throughout; size/occupancy
    are mediators, in only for the direct effect.
    """
    print("\n" + "=" * 70)
    print("4. TOTAL → DIRECT REGRESSION LADDER (fixed common-support sample)")
    print("=" * 70)

    df = lsoa.copy()
    df["log_floor_area"] = np.log(
        pd.to_numeric(df["oa_median_floor_area_m2"], errors="coerce").clip(lower=1)
    )
    confounds = ["median_build_year"] + _imd_income_col(df)
    occupancy = ["avg_hh_size"]
    size = ["log_floor_area"]

    # One fixed sample: complete cases on EVERY variable the ladder will touch,
    # so coefficient movement reflects mediator inclusion, not sample change.
    all_vars = [_DV, *_FORM, *confounds, *occupancy, *size]
    sample = df.dropna(subset=all_vars).copy()
    print(
        f"\n  Common sample: N = {len(sample):,} OAs "
        f"(complete cases on form, {', '.join(confounds)}, occupancy, floor area)"
    )
    print(f"  Confounds held throughout: {confounds}")

    steps = [
        ("M0 total  (form + confounds)", _FORM + confounds),
        ("M1        (+ occupancy)", _FORM + confounds + occupancy),
        ("M2 direct (+ dwelling size)", _FORM + confounds + occupancy + size),
    ]
    print(
        f"\n  {'Model':<30s} {'R²':>7s} {'β_det':>8s} {'β_flat':>8s} "
        f"{'Det:Flat':>9s} {'floor e':>8s}"
    )
    print("  " + "-" * 74)
    ratios: list[float] = []
    for label, xcols in steps:
        m = _run_ols(sample, _DV, xcols, label)
        if m is None:
            print(f"  {label:<30s} SKIPPED")
            continue
        bdet = float(m.params.get("pct_detached", np.nan))
        bflat = float(m.params.get("pct_flat", np.nan))
        ratio = _detached_flat_ratio(m)
        ratios.append(ratio)
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

    if len(ratios) >= 2 and ratios[0] > 1:
        total_gap = ratios[0] - 1.0
        direct_gap = ratios[-1] - 1.0
        mediated = (total_gap - direct_gap) / total_gap if total_gap else float("nan")
        print(
            f"\n  Total form gap (Det:Flat): {ratios[0]:.2f}×  →  "
            f"direct (size-held): {ratios[-1]:.2f}×"
        )
        print(
            f"  Size + occupancy mediate {mediated:.0%} of the gap; "
            f"the residual {direct_gap:+.0%} is direct fabric/exposure."
        )

    print(
        "\n  Caveat: tenure (TS054) and climate (HDD) are not yet in the dataset;\n"
        "  they are confounds to add next, and would adjust the *direct* term."
    )


def main() -> None:
    """Run the full form-vs-size decomposition on the national OA dataset."""
    lsoa = load_and_aggregate()
    print(f"\nLoaded {len(lsoa):,} OAs\n")

    if "oa_median_floor_area_m2" not in lsoa.columns:
        print("ERROR: oa_floor_area.parquet not merged — run "
              "data/aggregate_epc_floor_area_oa.py first.")
        return

    bivariate_floor_area_elasticity(lsoa)
    descriptive_panel(lsoa)
    quintile_stratification(lsoa)
    regression_ladder(lsoa)


if __name__ == "__main__":
    main()
