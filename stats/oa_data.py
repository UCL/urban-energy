"""
Core OA dataset for the two-axis analysis (energy spent vs access gained).

Assembled in the stats layer from lightweight, fast-to-build artefacts — no
cityseer pipeline, no 2.4 GB integrated GeoPackage:

* **Census 2021** — ``census_oa_joined.gpkg`` (dwelling type, cars, household
  size, commute bands) + the OA→LSOA key.
* **Heat** — ``oa_energy_consumption.parquet`` (DESNZ metered gas + elec → OA).
* **Size / fabric** — ``oa_epc.parquet`` (floor area, best-fabric intensity,
  build year).
* **Fleet / deprivation** — ``lsoa_vehicles.parquet`` (BEV share) and
  ``lsoa_imd2025.parquet`` (income), joined via LSOA.
* **Travel** — NTS-anchored disaggregation (``travel_energy.py``).
* **Access** — straight-line KD-tree counts (``oa_access.py``).

Consumers: ``access_profile``, ``lock_in``, ``form_size_decomposition``,
``travel_energy``. The shared OLS helpers live here too.

Usage::

    from oa_data import load_and_aggregate
    oa = load_and_aggregate()
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from oa_access import access_table
from travel_energy import TS058_BAND_MIDPOINTS_KM, compute_travel_energy

from urban_energy.paths import DATA_DIR

_STATS = DATA_DIR / "statistics"
_CENSUS = _STATS / "census_oa_joined.gpkg"

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

_CENSUS_COLS = [
    "OA21CD",
    "LSOA21CD",
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
_ENERGY_COLS = [
    "OA21CD",
    "oa_gas_mean_kwh",
    "oa_gas_num_meters",
    "oa_elec_mean_kwh",
    "oa_elec_num_meters",
]

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


def load_and_aggregate(cities: list[str] | None = None) -> pd.DataFrame:
    """
    Assemble the per-OA two-axis dataset from the primary artefacts.

    Returns one row per OA with ``building_kwh_per_hh`` (heat),
    ``transport_kwh_per_hh_total_est`` (travel), ``dominant_type``, the
    dwelling-type shares, household size, EPC floor area + best-fabric intensity +
    build year, fleet/deprivation context, and straight-line access columns.
    """
    print("Assembling OA data …")
    oa = gpd.read_file(_CENSUS, columns=_CENSUS_COLS, ignore_geometry=True)
    oa = pd.DataFrame(oa)

    # OA land area (hectares), for population density.
    area = gpd.read_file(_CENSUS, columns=["OA21CD"]).to_crs(27700)
    area["oa_area_ha"] = area.geometry.area / 1e4
    oa = oa.merge(
        pd.DataFrame(area.drop(columns="geometry"))[["OA21CD", "oa_area_ha"]],
        on="OA21CD",
        how="left",
    )

    oa = oa.merge(
        pd.read_parquet(_STATS / "oa_energy_consumption.parquet", columns=_ENERGY_COLS),
        on="OA21CD",
        how="left",
    )
    oa = oa.merge(pd.read_parquet(_STATS / "oa_epc.parquet"), on="OA21CD", how="left")
    oa = oa.merge(access_table().reset_index(), on="OA21CD", how="left")

    # Fleet (BEV share) and deprivation (income), via the OA→LSOA key.
    veh = pd.read_parquet(_STATS / "lsoa_vehicles.parquet")
    veh["bev_share"] = _num(veh["ulev_battery_electric"]) / _num(
        veh["cars_total"]
    ).replace(0, np.nan)
    oa = oa.merge(veh[["LSOA21CD", "bev_share"]], on="LSOA21CD", how="left")
    imd = pd.read_parquet(
        _STATS / "lsoa_imd2025.parquet", columns=["LSOA21CD", "imd_income_score"]
    )
    oa = oa.merge(imd, on="LSOA21CD", how="left")
    oa = oa.rename(columns={"oa_median_build_year": "median_build_year"})

    # --- Population and households ---
    oa["total_people"] = _num(oa[_TS001_POP])
    oa["pop_density"] = oa["total_people"] / _num(oa["oa_area_ha"]).replace(0, np.nan)
    oa["total_hh"] = _num(oa[_TS017_TOTAL]) - _num(oa[_TS017_ZERO])
    oa["avg_hh_size"] = oa["total_people"] / oa["total_hh"].replace(0, np.nan)

    # --- Heat: genuine metered energy per household (gas + elec, common denom) ---
    energy_total = (_num(oa["oa_gas_mean_kwh"]) * _num(oa["oa_gas_num_meters"])).fillna(
        0
    ) + (_num(oa["oa_elec_mean_kwh"]) * _num(oa["oa_elec_num_meters"])).fillna(0)
    oa["building_kwh_per_hh"] = energy_total / oa["total_hh"].replace(0, np.nan)
    oa["building_kwh_per_person"] = energy_total / oa["total_people"].replace(0, np.nan)
    oa["log_building_kwh_per_hh"] = np.log(oa["building_kwh_per_hh"].clip(lower=1))
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

    # --- Filter (relaxed; one row per OA, so no dedup needed) ---
    oa = oa[
        (oa["total_people"] > 10)
        & oa["building_kwh_per_hh"].notna()
        & (oa["building_kwh_per_hh"] > 0)
    ].copy()

    print(f"  {len(oa):,} OAs")
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
    """Run OLS with HC1 robust SEs on complete cases (optional FE / clustering)."""
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
