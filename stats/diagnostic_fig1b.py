"""
Diagnostic: identify confounders of building_kwh_per_hh by housing type.

Questions addressed:
1. What share of variance in building_kwh_per_hh is explained by each
   candidate confounder (bivariate R²)?
2. What is the partial R² of dominant_type after controlling for each
   confounder (OLS with and without type dummies)?
3. Cross-tabulation of confounders by dominant type to show which types
   are most contaminated.

Candidate confounders:
  - avg_hh_size     (household size — dilutes per-capita)
  - median_build_year (stock age — thermal performance era)
  - pct_not_deprived  (deprivation — fuel poverty suppresses usage)
  - lsoa_sv           (S/V ratio — thermal envelope geometry)
  - people_per_ha     (density — urban heat island, wind shelter)

Usage:
    uv run python stats/diagnostic_fig1b.py
    uv run python stats/diagnostic_fig1b.py manchester
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

sys.path.insert(0, str(Path(__file__).parent))
from proof_of_concept_lsoa import load_and_aggregate  # noqa: E402

TYPE_ORDER = ["Flat", "Terraced", "Semi", "Detached"]

CONFOUNDERS = {
    "avg_hh_size": "Household size",
    "median_build_year": "Build year",
    "pct_not_deprived": "% not deprived",
    "lsoa_sv": "S/V ratio",
    "people_per_ha": "Density (ppl/ha)",
}


def _r2(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 10:
        return float("nan")
    return linregress(x[mask], y[mask]).rvalue ** 2


def _ols_r2(X: pd.DataFrame, y: pd.Series) -> float:
    """R² of OLS y ~ X (with intercept) using numpy lstsq."""
    mask = X.notna().all(axis=1) & y.notna()
    Xm = np.column_stack([np.ones(mask.sum()), X[mask].values])
    ym = y[mask].values
    coef, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
    y_hat = Xm @ coef
    ss_res = ((ym - y_hat) ** 2).sum()
    ss_tot = ((ym - ym.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def run_diagnostic(cities: list[str] | None = None) -> None:
    lsoa = load_and_aggregate(cities)
    y = lsoa["building_kwh_per_hh"]

    available = [c for c in CONFOUNDERS if c in lsoa.columns]

    # -----------------------------------------------------------------------
    # 1. Bivariate R² of each confounder with building_kwh_per_hh
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("1. BIVARIATE R²: confounder vs building_kwh_per_hh")
    print("=" * 70)
    for col in available:
        r2 = _r2(lsoa[col], y)
        print(f"  {CONFOUNDERS[col]:<22s}  R²={r2:.3f}")

    # Type dummies bivariate R²
    type_dummies = pd.get_dummies(
        lsoa["dominant_type"], drop_first=True, dtype=float,
    )
    r2_type_only = _ols_r2(type_dummies, y)
    print(f"  {'Housing type (dummies)':<22s}  R²={r2_type_only:.3f}")

    # -----------------------------------------------------------------------
    # 2. Partial R²: what does type add after each confounder?
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("2. PARTIAL R²: type dummies after controlling for confounder")
    print("   (R² of model with confounder+type vs confounder alone)")
    print("=" * 70)
    for col in available:
        mask = lsoa[col].notna() & y.notna() & lsoa["dominant_type"].notna()
        sub = lsoa[mask]
        ysub = y[mask]
        conf_only = sub[[col]]
        conf_type = pd.concat(
            [sub[[col]], pd.get_dummies(
                sub["dominant_type"], drop_first=True, dtype=float,
            )],
            axis=1,
        )
        r2_conf = _ols_r2(conf_only, ysub)
        r2_full = _ols_r2(conf_type, ysub)
        partial = r2_full - r2_conf
        print(
            f"  {CONFOUNDERS[col]:<22s}  "
            f"confounder R²={r2_conf:.3f}  "
            f"+type R²={r2_full:.3f}  "
            f"partial={partial:.3f}"
        )

    # -----------------------------------------------------------------------
    # 3. All confounders together, then +type
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("3. FULL MODEL: all confounders vs all confounders + type")
    print("=" * 70)
    mask = (
        lsoa[available].notna().all(axis=1)
        & y.notna()
        & lsoa["dominant_type"].notna()
    )
    sub = lsoa[mask]
    ysub = y[mask]
    conf_all = sub[available]
    conf_all_type = pd.concat(
        [sub[available], pd.get_dummies(
            sub["dominant_type"], drop_first=True, dtype=float,
        )],
        axis=1,
    )
    r2_conf_all = _ols_r2(conf_all, ysub)
    r2_full_all = _ols_r2(conf_all_type, ysub)
    print(f"  All confounders R²={r2_conf_all:.3f}")
    partial_all = r2_full_all - r2_conf_all
    print(
        f"  + type dummies  R²={r2_full_all:.3f}"
        f"  (partial={partial_all:.3f})"
    )

    # -----------------------------------------------------------------------
    # 4. Confounder means by dominant type
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("4. CONFOUNDER MEDIANS BY DOMINANT TYPE")
    print("=" * 70)
    rows = []
    for col in available:
        row = {"Confounder": CONFOUNDERS[col]}
        for t in TYPE_ORDER:
            mask_t = lsoa["dominant_type"] == t
            row[t] = lsoa.loc[mask_t, col].median()
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Confounder")
    print(df.round(2).to_string())


if __name__ == "__main__":
    _cities = [a for a in sys.argv[1:] if not a.startswith("-")]
    run_diagnostic(cities=_cities or None)
