"""
Core OA dataset for the two-axis analysis (energy spent vs access gained).

Loads the national integrated OA GeoPackage produced by
``processing/pipeline_oa.py`` and derives the household-level variables the
two-axis layer consumes:

* **Heat** — genuine metered building energy per household (DESNZ gas + elec).
* **Travel** — total car-travel energy from the NTS-anchored disaggregation
  (``stats/travel_energy.py``).
* **Form** — Census dwelling-type shares and the dominant type.
* **Size** — EPC median dwelling floor area and best-fabric (POTENTIAL)
  intensity (``data/aggregate_epc_oa.py``).

Consumers: ``access_profile``, ``lock_in``, ``form_size_decomposition`` and
``travel_energy`` (which this module calls). The small OLS helpers used by the
form/size decomposition live here too, so the analysis scripts share one core.

Usage::

    from oa_data import load_and_aggregate
    oa = load_and_aggregate()
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from travel_energy import TS058_BAND_MIDPOINTS_KM, compute_travel_energy

from urban_energy.paths import DATA_DIR, PROCESSING_DIR

DATA_PATH = PROCESSING_DIR / "combined" / "oa_integrated.gpkg"
_EPC_PATH = DATA_DIR / "statistics" / "oa_epc.parquet"

# --- Census 2021 column names ---
_TS001_POP = "ts001_Residence type: Lives in a household; measures: Value"
_TS017_TOTAL = "ts017_Household size: Total: All household spaces; measures: Value"
_TS017_ZERO = "ts017_Household size: 0 people in household; measures: Value"
_TS045_TOTAL = "ts045_Number of cars or vans: Total: All households"
_TS045_ONE = "ts045_Number of cars or vans: 1 car or van in household"
_TS045_TWO = "ts045_Number of cars or vans: 2 cars or vans in household"
_TS045_THREE = "ts045_Number of cars or vans: 3 or more cars or vans in household"
_TS044_TOTAL = "ts044_Accommodation type: Total: All households"
_TS044_DETACHED = "ts044_Accommodation type: Detached"
_TS044_SEMI = "ts044_Accommodation type: Semi-detached"
_TS044_TERRACED = "ts044_Accommodation type: Terraced"
_TS044_FLAT = "ts044_Accommodation type: In a purpose-built block of flats or tenement"

# Metered energy, postcode-aggregated to OA (per-meter mean × meter count).
_OA_ENERGY = [
    "oa_gas_mean_kwh",
    "oa_gas_num_meters",
    "oa_elec_mean_kwh",
    "oa_elec_num_meters",
]

# Census columns the loader requires (a hard error if absent).
_CENSUS_REQUIRED = [
    _TS001_POP,
    _TS017_TOTAL,
    _TS017_ZERO,
    _TS045_TOTAL,
    _TS045_ONE,
    _TS045_TWO,
    _TS045_THREE,
    _TS044_TOTAL,
    _TS044_DETACHED,
    _TS044_SEMI,
    _TS044_TERRACED,
    _TS044_FLAT,
    *TS058_BAND_MIDPOINTS_KM,  # commute distance → travel disaggregation
]
_REQUIRED = ["OA21CD", *_OA_ENERGY, *_CENSUS_REQUIRED]

# Kept when present; no error if absent.
_EXTRA = ["city", "n_uprns", "median_build_year", "bev_share"]

_TYPE_MAP = {
    "pct_flat": "Flat",
    "pct_terraced": "Terraced",
    "pct_semi": "Semi",
    "pct_detached": "Detached",
}
_TYPE_ORDER = ["Flat", "Terraced", "Semi", "Detached"]


def _num(series: pd.Series) -> pd.Series:
    """Coerce to numeric, non-parseable → NaN."""
    return pd.to_numeric(series, errors="coerce")


def _validate(available: set[str]) -> None:
    """Raise if any hard-required column is missing from the dataset."""
    missing = [c for c in _REQUIRED if c not in available]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )


def load_and_aggregate(cities: list[str] | None = None) -> pd.DataFrame:
    """
    Load the integrated OA GeoPackage and derive the two-axis variables.

    Parameters
    ----------
    cities : list[str] or None
        Restrict to these ``city`` values, or ``None`` for all of England.

    Returns
    -------
    pandas.DataFrame
        One row per OA with ``building_kwh_per_hh`` (heat),
        ``transport_kwh_per_hh_total_est`` (travel), ``dominant_type``, the
        dwelling-type shares, household size, EPC floor area + best-fabric
        intensity, and the travel-energy intermediates.
    """
    print("Loading OA data …")
    probe = gpd.read_file(DATA_PATH, rows=1)
    available = set(probe.columns)
    del probe
    _validate(available)

    imd_cols = [
        c for c in available if "imd_income" in c.lower() and "score" in c.lower()
    ]
    cols = sorted({*_REQUIRED, *_EXTRA, *imd_cols} & available)
    oa = gpd.read_file(DATA_PATH, columns=cols)
    if "geometry" in oa.columns:
        oa = pd.DataFrame(oa.drop(columns="geometry"))
    print(f"  {len(oa):,} OA rows, {len(oa.columns)} columns")

    # EPC median floor area + best-fabric (POTENTIAL) intensity, per OA.
    if _EPC_PATH.exists():
        keep = ["OA21CD", "oa_median_floor_area_m2", "epc_potential_kwh_m2"]
        epc = pd.read_parquet(_EPC_PATH)
        oa = oa.merge(
            epc[[c for c in keep if c in epc.columns]], on="OA21CD", how="left"
        )

    oa = oa[oa["OA21CD"].notna()].copy()
    if cities and "city" in oa.columns:
        oa = oa[oa["city"].isin(cities)].copy()

    # De-duplicate boundary-straddling OAs. An OA intersecting more than one
    # Built-Up Area is emitted once per BUA; census + energy columns are
    # BUA-invariant. Keep the row from the OA's "home" BUA (max n_uprns) so
    # boundary replicas don't inflate N and deflate regression SEs.
    if oa["OA21CD"].duplicated().any():
        n0 = len(oa)
        rank = (
            _num(oa["n_uprns"]).fillna(0)
            if "n_uprns" in oa.columns
            else pd.Series(0, index=oa.index)
        )
        oa = (
            oa.assign(_rank=rank)
            .sort_values("_rank", ascending=False, kind="stable")
            .drop_duplicates("OA21CD", keep="first")
            .drop(columns="_rank")
            .reset_index(drop=True)
        )
        print(f"  de-duplicated {n0 - len(oa):,} boundary rows → {len(oa):,}")

    # --- Population and households ---
    oa["total_people"] = _num(oa[_TS001_POP])
    oa["total_hh"] = _num(oa[_TS017_TOTAL]) - _num(oa[_TS017_ZERO])
    oa["avg_hh_size"] = oa["total_people"] / oa["total_hh"].replace(0, np.nan)

    # --- Heat: genuine metered energy per household ---
    # OA total = per-meter mean × meter count, summed across fuels over a common
    # household denominator (TS017). Off-gas / communal-gas under-recording is
    # flagged separately by urban_energy.form_bias (pipeline Stage 3).
    energy_total = (_num(oa["oa_gas_mean_kwh"]) * _num(oa["oa_gas_num_meters"])).fillna(
        0
    ) + (_num(oa["oa_elec_mean_kwh"]) * _num(oa["oa_elec_num_meters"])).fillna(0)
    oa["building_kwh_per_hh"] = energy_total / oa["total_hh"].replace(0, np.nan)
    oa["building_kwh_per_person"] = energy_total / oa["total_people"].replace(0, np.nan)
    oa["log_building_kwh_per_hh"] = np.log(oa["building_kwh_per_hh"].clip(lower=1))
    if "oa_median_floor_area_m2" in oa.columns:
        oa["building_kwh_per_m2"] = oa["building_kwh_per_hh"] / _num(
            oa["oa_median_floor_area_m2"]
        ).replace(0, np.nan)

    # --- Cars per household (TS045) ---
    oa["cars_per_hh"] = (
        _num(oa[_TS045_ONE]) + 2 * _num(oa[_TS045_TWO]) + 3 * _num(oa[_TS045_THREE])
    ) / _num(oa[_TS045_TOTAL]).replace(0, np.nan)

    # --- Dwelling-type shares (TS044) + dominant type (compact → sprawl) ---
    denom = _num(oa[_TS044_TOTAL]).replace(0, np.nan)
    oa["pct_detached"] = _num(oa[_TS044_DETACHED]) / denom * 100
    oa["pct_semi"] = _num(oa[_TS044_SEMI]) / denom * 100
    oa["pct_terraced"] = _num(oa[_TS044_TERRACED]) / denom * 100
    oa["pct_flat"] = _num(oa[_TS044_FLAT]) / denom * 100
    oa["dominant_type"] = pd.Categorical(
        oa[list(_TYPE_MAP)].fillna(0).idxmax(axis=1).map(_TYPE_MAP),
        categories=_TYPE_ORDER,
        ordered=True,
    )

    # --- Travel: NTS-anchored car-travel energy (constrained disaggregation) ---
    oa = compute_travel_energy(oa)
    oa["transport_kwh_per_hh_total_est"] = oa["travel_kwh_per_hh_car"]

    # --- Filter (relaxed for small OA units) ---
    valid = (
        (oa["total_people"] > 10)
        & oa["building_kwh_per_hh"].notna()
        & (oa["building_kwh_per_hh"] > 0)
    )
    if "n_uprns" in oa.columns:
        valid &= _num(oa["n_uprns"]) >= 5
    oa = oa[valid].copy()

    print(f"  {len(oa):,} OAs after filter")
    print(f"    heat   median {oa['building_kwh_per_hh'].median():>8,.0f} kWh/hh")
    print(
        f"    travel median {oa['transport_kwh_per_hh_total_est'].median():>8,.0f} "
        "kWh/hh"
    )
    counts = "  ".join(f"{t} {(oa['dominant_type'] == t).sum():,}" for t in _TYPE_ORDER)
    print(f"    dominant type: {counts}")
    return oa


# ---------------------------------------------------------------------------
# Shared OLS helpers (used by form_size_decomposition)
# ---------------------------------------------------------------------------


def _sigstars(p: float) -> str:
    """Significance stars for a p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _demean_by_group(df: pd.DataFrame, cols: list[str], group_col: str) -> pd.DataFrame:
    """Within-group demean (Frisch-Waugh-Lovell): absorb a fixed effect."""
    return df[cols] - df.groupby(group_col)[cols].transform("mean")


def _run_ols(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    label: str,
    cluster_col: str | None = None,
    fe_col: str | None = None,
) -> sm.regression.linear_model.RegressionResultsWrapper | None:
    """
    Run OLS with HC1 robust SEs on complete cases.

    Parameters
    ----------
    cluster_col : str or None
        If given, also fit cluster-robust SEs, stored on ``result._clustered_fit``.
    fe_col : str or None
        If given, absorb this categorical as a fixed effect via within-group
        demeaning (coefficients identical to explicit dummies).
    """
    extra: list[str] = []
    if cluster_col and cluster_col not in [y_col, *x_cols]:
        extra.append(cluster_col)
    if fe_col and fe_col not in [y_col, *x_cols, *extra]:
        extra.append(fe_col)
    sub = df[[y_col, *x_cols, *extra]].dropna()
    if len(sub) < len(x_cols) + 10:
        print(f"  {label}: insufficient data (N={len(sub)})")
        return None

    if fe_col:
        demeaned = _demean_by_group(sub, [y_col, *x_cols], fe_col)
        y, x = demeaned[y_col], demeaned[x_cols]  # demeaned: no constant
    else:
        y, x = sub[y_col], sm.add_constant(sub[x_cols])

    result = sm.OLS(y, x).fit(cov_type="HC1")
    if fe_col:
        result._n_fe_groups = sub[fe_col].nunique()
        result._fe_col = fe_col
    if cluster_col and cluster_col in sub.columns:
        try:
            result._clustered_fit = sm.OLS(y, x).fit(
                cov_type="cluster", cov_kwds={"groups": sub[cluster_col]}
            )
        except Exception:
            pass
    return result
