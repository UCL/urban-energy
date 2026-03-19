"""
Draft v1 basket index for morphology-linked energy cost of access.

This script builds a TCPA-aligned (but explicitly provisional) local-access basket
using currently available 800m cityseer accessibility layers, combines it with the
existing LSOA energy-cost metrics, and exports:

- CSV schema and summary tables
- draft figures
- a markdown summary document for review

The current basket uses trip-type-specific distance-decay from nearest-network distance
for local access, with a 4.8km wider-access bound. It is not yet a
full 20-minute multimodal basket. The goal is to produce a clear
end-to-end draft that can be iterated into the final index design.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from proof_of_concept_lsoa import load_and_aggregate  # noqa: E402

from urban_energy.paths import PROJECT_DIR  # noqa: E402

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150})

TYPE_ORDER = ["Flat", "Terraced", "Semi", "Detached"]
TYPE_LABELS = {
    "Flat": "Flat",
    "Terraced": "Terraced",
    "Semi": "Semi-detached",
    "Detached": "Detached",
}
TYPE_COLORS = {
    "Flat": "#3498db",
    "Terraced": "#2ecc71",
    "Semi": "#f39c12",
    "Detached": "#e74c3c",
}
LOCAL_ACCESS_M = 800.0
WIDER_ACCESS_M = 4800.0
LOCAL_WALK_HALF_DISTANCE_M = 500.0
LOCAL_WALK_DECAY_SHAPE = 2.0

OUT_DIR = PROJECT_DIR / "stats" / "figures" / "basket_v1"
DOC_PATH = OUT_DIR / "basket_v1_summary.md"

# ---------------------------------------------------------------------------
# Trip-demand assumptions for access-energy budgeting
# ---------------------------------------------------------------------------

# Real-world anchors (England):
# - NTS 2024 total trips/person = 922
# - NTS 2024 shopping trips/person = 167
# - NTS 2024 commuting trips/person = 111
# - NTS 2024 "just walk" trips/person = 85
# - NTS 2024 local bus trips/person = 41
# - NTS 2024 surface rail trips/person = 21
# - NHS England GP appointments (Apr 2024-Mar 2025): 383.3m
# - NHS Digital outpatient attended appointments (2024/25): 113.2m
# - NHSBSA prescription items dispensed (2024/25): 1.26bn
# - DfE pupil headcount (all schools, 2024/25): 9,032,426
# - School year length: 190 days (380 sessions)
#
# Healthcare and school conversions are explicit approximations.
_NTS_TOTAL_TRIPS_PER_PERSON = 922.0
_NTS_COMMUTE_TRIPS_PER_PERSON = 111.0
_ENGLAND_POP_APPROX_2024 = 58_000_000.0
_GP_APPOINTMENTS_ANNUAL = 383_300_000.0
_OUTPATIENT_ATTENDED_ANNUAL = 113_200_000.0
_PRESCRIPTION_ITEMS_ANNUAL = 1_260_000_000.0
_PRESCRIPTION_ITEMS_PER_COLLECTION_TRIP = 3.0
_NTS_BUS_TRIPS_PER_PERSON = 41.0
_NTS_SURFACE_RAIL_TRIPS_PER_PERSON = 21.0
_DFE_PUPIL_HEADCOUNT_2024_25 = 9_032_426.0
_SCHOOL_DAYS_PER_YEAR = 190.0
_SCHOOL_TRIPS_PER_PUPIL_PER_DAY = 2.0

TRIP_DEMAND_ASSUMPTIONS: dict[str, dict[str, object]] = {
    "food_services_proxy": {
        "annual_trips_per_person": 167.0,
        "annual_trips_per_household": np.nan,
        "include_in_trip_budget": True,
        "source": "DfT NTS 2024 shopping trips per person",
        "method_note": "Direct NTS trip count",
    },
    "gp": {
        "annual_trips_per_person": _GP_APPOINTMENTS_ANNUAL / _ENGLAND_POP_APPROX_2024,
        "annual_trips_per_household": np.nan,
        "include_in_trip_budget": True,
        "source": "NHS England GP appointments (Apr 2024-Mar 2025)",
        "method_note": (
            "Appointments converted to per-person with England population approx"
        ),
    },
    "pharmacy": {
        "annual_trips_per_person": (
            (_PRESCRIPTION_ITEMS_ANNUAL / _ENGLAND_POP_APPROX_2024)
            / _PRESCRIPTION_ITEMS_PER_COLLECTION_TRIP
        ),
        "annual_trips_per_household": np.nan,
        "include_in_trip_budget": True,
        "source": "NHSBSA PCA 2024/25 prescription items",
        "method_note": "Assumes 3 prescription items collected per pharmacy trip",
    },
    "school": {
        "annual_trips_per_person": (
            _DFE_PUPIL_HEADCOUNT_2024_25
            * _SCHOOL_TRIPS_PER_PUPIL_PER_DAY
            * _SCHOOL_DAYS_PER_YEAR
            / _ENGLAND_POP_APPROX_2024
        ),
        "annual_trips_per_household": np.nan,
        "include_in_trip_budget": True,
        "source": (
            "DfE Schools, pupils and their characteristics"
            " 2024/25 + ONS England population approx"
        ),
        "method_note": (
            "Pupil headcount x 2 school trips/day x 190 days, converted to per-person"
        ),
    },
    "greenspace": {
        "annual_trips_per_person": 85.0,
        "annual_trips_per_household": np.nan,
        "include_in_trip_budget": True,
        "source": "DfT NTS 2024 'just walk' trips per person",
        "method_note": "Used as local green-space access demand proxy",
    },
    "bus": {
        "annual_trips_per_person": _NTS_BUS_TRIPS_PER_PERSON,
        "annual_trips_per_household": np.nan,
        "include_in_trip_budget": False,
        "source": "DfT NTS 2024 public transport trends (local bus trips/person)",
        "method_note": (
            "Retained as mode-access enabler;"
            " excluded from summed destination trip budget"
        ),
    },
    "rail": {
        "annual_trips_per_person": _NTS_SURFACE_RAIL_TRIPS_PER_PERSON,
        "annual_trips_per_household": np.nan,
        "include_in_trip_budget": False,
        "source": "DfT NTS 2024 public transport trends (surface rail trips/person)",
        "method_note": (
            "Retained as mode-access enabler;"
            " excluded from summed destination trip budget"
        ),
    },
    "hospital": {
        "annual_trips_per_person": _OUTPATIENT_ATTENDED_ANNUAL
        / _ENGLAND_POP_APPROX_2024,
        "annual_trips_per_household": np.nan,
        "include_in_trip_budget": True,
        "source": "NHS Digital Hospital Outpatient Activity 2024/25",
        "method_note": "Attended outpatient appointments converted to per-person",
    },
}


# Category schema. Quantiles are calibration points for this
# draft, not final normative thresholds.
BASKET_SCHEMA = [
    {
        "id": "food_services_proxy",
        "label": "Local food/services (FSA proxy)",
        "group": "core",
        "weight": 0.20,
        "local_walk_half_distance_m": 400.0,
        "source_column_local": "cc_fsa_total_800_wt",
        "source_column_wider": "cc_fsa_total_4800_wt",
        "nearest_column": "cc_fsa_total_nearest_max_4800",
        "floor_quantile": 0.25,
        "target_quantile": 0.75,
    },
    {
        "id": "gp",
        "label": "GP practice access",
        "group": "core",
        "weight": 0.15,
        "local_walk_half_distance_m": 700.0,
        "source_column_local": "cc_gp_practice_800_wt",
        "source_column_wider": "cc_gp_practice_4800_wt",
        "nearest_column": "cc_gp_practice_nearest_max_4800",
        "floor_quantile": 0.25,
        "target_quantile": 0.75,
    },
    {
        "id": "pharmacy",
        "label": "Pharmacy access",
        "group": "core",
        "weight": 0.10,
        "local_walk_half_distance_m": 650.0,
        "source_column_local": "cc_pharmacy_800_wt",
        "source_column_wider": "cc_pharmacy_4800_wt",
        "nearest_column": "cc_pharmacy_nearest_max_4800",
        "floor_quantile": 0.25,
        "target_quantile": 0.75,
    },
    {
        "id": "school",
        "label": "School access",
        "group": "core",
        "weight": 0.15,
        "local_walk_half_distance_m": 900.0,
        "source_column_local": "cc_school_800_wt",
        "source_column_wider": "cc_school_4800_wt",
        "nearest_column": "cc_school_nearest_max_4800",
        "floor_quantile": 0.25,
        "target_quantile": 0.75,
    },
    {
        "id": "greenspace",
        "label": "Green space access",
        "group": "core",
        "weight": 0.15,
        "local_walk_half_distance_m": 1000.0,
        "source_column_local": "cc_greenspace_800_wt",
        "source_column_wider": "cc_greenspace_4800_wt",
        "nearest_column": "cc_greenspace_nearest_max_4800",
        "floor_quantile": 0.25,
        "target_quantile": 0.75,
    },
    {
        "id": "bus",
        "label": "Bus access",
        "group": "core",
        "weight": 0.15,
        "local_walk_half_distance_m": 500.0,
        "source_column_local": "cc_bus_800_wt",
        "source_column_wider": "cc_bus_4800_wt",
        "nearest_column": "cc_bus_nearest_max_4800",
        "floor_quantile": 0.25,
        "target_quantile": 0.75,
    },
    {
        "id": "rail",
        "label": "Rail/metro access",
        "group": "support",
        "weight": 0.05,
        "local_walk_half_distance_m": 800.0,
        "source_column_local": "cc_rail_800_wt",
        "source_column_wider": "cc_rail_4800_wt",
        "nearest_column": "cc_rail_nearest_max_4800",
        "floor_quantile": 0.90,
        "target_quantile": 0.95,
    },
    {
        "id": "hospital",
        "label": "Hospital access",
        "group": "support",
        "weight": 0.05,
        "local_walk_half_distance_m": 700.0,
        "source_column_local": "cc_hospital_800_wt",
        "source_column_wider": "cc_hospital_4800_wt",
        "nearest_column": "cc_hospital_nearest_max_4800",
        "floor_quantile": 0.25,
        "target_quantile": 0.75,
    },
]


def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _local_access_distance_share(
    nearest_m: pd.Series,
    *,
    half_distance_m: float = LOCAL_WALK_HALF_DISTANCE_M,
    decay_shape: float = LOCAL_WALK_DECAY_SHAPE,
) -> pd.Series:
    """
    Distance-weighted local-access share in [0, 1].

    Uses a smooth decay where share=1 at distance 0 and share=0.5 at
    ``half_distance_m``. Distances beyond the wider search bound are 0.
    """
    if not np.isfinite(half_distance_m) or half_distance_m <= 0:
        half_distance_m = LOCAL_WALK_HALF_DISTANCE_M
    if not np.isfinite(decay_shape) or decay_shape <= 0:
        decay_shape = LOCAL_WALK_DECAY_SHAPE
    d = pd.to_numeric(nearest_m, errors="coerce")
    share = pd.Series(0.0, index=d.index, dtype=float)
    valid = d.notna() & (d >= 0) & (d <= WIDER_ACCESS_M)
    if valid.any():
        scaled = (d.loc[valid] / half_distance_m).pow(decay_shape)
        share.loc[valid] = np.exp(-np.log(2.0) * scaled)
    return share.clip(lower=0.0, upper=1.0)


def prepare_lsoa(cities: list[str] | None = None) -> pd.DataFrame:
    """Load PoC LSOA dataset and derive basket inputs."""
    lsoa = load_and_aggregate(cities=cities)

    fsa_cols_local = [
        c for c in lsoa.columns if c.startswith("cc_fsa_") and c.endswith("_800_wt")
    ]
    fsa_cols_wider = [
        c for c in lsoa.columns if c.startswith("cc_fsa_") and c.endswith("_4800_wt")
    ]
    lsoa = _ensure_numeric(lsoa, fsa_cols_local + fsa_cols_wider)
    if not fsa_cols_local:
        raise ValueError(
            "No cc_fsa_*_800_wt columns found for FSA-based local basket proxy"
        )
    if not fsa_cols_wider:
        raise ValueError(
            "No cc_fsa_*_4800_wt columns found for FSA-based wider basket proxy"
        )
    lsoa["cc_fsa_total_800_wt"] = lsoa[fsa_cols_local].sum(axis=1)
    lsoa["cc_fsa_total_4800_wt"] = lsoa[fsa_cols_wider].sum(axis=1)
    fsa_nearest_cols = [
        c
        for c in lsoa.columns
        if c.startswith("cc_fsa_")
        and c.endswith("_nearest_max_4800")
        and c != "cc_fsa_total_nearest_max_4800"
    ]
    lsoa = _ensure_numeric(lsoa, fsa_nearest_cols)
    if fsa_nearest_cols:
        lsoa["cc_fsa_total_nearest_max_4800"] = lsoa[fsa_nearest_cols].min(axis=1)
    else:
        lsoa["cc_fsa_total_nearest_max_4800"] = np.nan

    needed = (
        [row["source_column_local"] for row in BASKET_SCHEMA]
        + [row["source_column_wider"] for row in BASKET_SCHEMA]
        + [row["nearest_column"] for row in BASKET_SCHEMA]
        + [
            "total_kwh_per_hh",
            "total_kwh_per_hh_total_est",
            "transport_kwh_per_hh_est",
            "transport_kwh_per_hh_total_est",
            "total_kwh_per_person",
            "dominant_type",
            "people_per_ha",
            "avg_hh_size",
            "pct_not_deprived",
            "cc_density_800",
            "cc_harmonic_800",
            "city",
        ]
    )
    _ensure_numeric(
        lsoa,
        [
            str(c)
            for c in needed
            if str(c) in lsoa.columns and str(c) != "dominant_type" and str(c) != "city"
        ],
    )
    return lsoa


def calibrate_schema(lsoa: pd.DataFrame) -> pd.DataFrame:
    """Convert schema config into a calibrated table with numeric thresholds."""
    rows: list[dict[str, object]] = []
    hh_size_ref = (
        float(pd.to_numeric(lsoa["avg_hh_size"], errors="coerce").median())
        if "avg_hh_size" in lsoa.columns
        else 2.4
    )
    for cfg in BASKET_SCHEMA:
        cid = cfg["id"]
        local_half_distance = float(
            cfg.get("local_walk_half_distance_m", LOCAL_WALK_HALF_DISTANCE_M)
        )
        local_decay_shape = float(
            cfg.get("local_walk_decay_shape", LOCAL_WALK_DECAY_SHAPE)
        )
        col_local = cfg["source_column_local"]
        col_wider = cfg["source_column_wider"]
        if col_local not in lsoa.columns:
            raise ValueError(f"Missing required basket column (local): {col_local}")
        if col_wider not in lsoa.columns:
            raise ValueError(f"Missing required basket column (wider): {col_wider}")
        s = pd.to_numeric(lsoa[col_local], errors="coerce").dropna()
        s_wider = pd.to_numeric(lsoa[col_wider], errors="coerce").dropna()
        if len(s) == 0:
            raise ValueError(
                f"No non-null values for basket column (local): {col_local}"
            )
        if len(s_wider) == 0:
            raise ValueError(
                f"No non-null values for basket column (wider): {col_wider}"
            )
        floor_q = float(cfg["floor_quantile"])
        target_q = float(cfg["target_quantile"])
        floor_v = float(s.quantile(floor_q))
        target_v = float(s.quantile(target_q))
        if target_v <= 0:
            positive = s[s > 0]
            target_v = float(positive.quantile(0.75)) if len(positive) else 1.0
        if target_v <= floor_v:
            # For sparse layers (notably rail), keep a strict
            # floor but ensure target > floor.
            target_v = max(target_v, floor_v * 1.25, floor_v + 1e-6, 1e-6)

        trip_meta: dict[str, object] = TRIP_DEMAND_ASSUMPTIONS.get(str(cid)) or {}
        _trips_pp = trip_meta.get("annual_trips_per_person", np.nan)
        annual_trips_pp = float(_trips_pp)  # type: ignore[arg-type]
        _trips_hh = trip_meta.get("annual_trips_per_household", np.nan)
        annual_trips_hh = float(_trips_hh)  # type: ignore[arg-type]
        include_trip_budget = bool(trip_meta.get("include_in_trip_budget", True))
        trip_equiv_pp = 0.0
        if include_trip_budget:
            trip_equiv_pp = annual_trips_pp
            if np.isnan(trip_equiv_pp):
                trip_equiv_pp = 0.0
            if not np.isnan(annual_trips_hh) and hh_size_ref > 0:
                trip_equiv_pp += annual_trips_hh / hh_size_ref

        rows.append(
            {
                **cfg,
                "local_walk_half_distance_m": local_half_distance,
                "local_walk_decay_shape": local_decay_shape,
                "source_column": col_local,
                "floor_value": floor_v,
                "target_value": target_v,
                "non_null_n": int(s.notna().sum()),
                "positive_share": float((s > 0).mean()),
                "median_value": float(s.median()),
                "median_value_wider": float(s_wider.median()),
                "q90_value": float(s.quantile(0.90)),
                "annual_trips_per_person": annual_trips_pp,
                "annual_trips_per_household": annual_trips_hh,
                "include_in_trip_budget": include_trip_budget,
                "trip_demand_equiv_per_person": trip_equiv_pp,
                "trip_source": str(trip_meta.get("source", "not specified")),
                "trip_method_note": str(trip_meta.get("method_note", "")),
            }
        )

    schema = pd.DataFrame(rows)
    trip_sum = float(schema["trip_demand_equiv_per_person"].sum())
    if trip_sum > 0:
        schema["trip_weight"] = schema["trip_demand_equiv_per_person"] / trip_sum
    else:
        schema["trip_weight"] = np.nan
    if not np.isclose(schema["weight"].sum(), 1.0):
        raise ValueError(
            f"Basket weights must sum to 1.0 (got {schema['weight'].sum():.3f})"
        )
    return schema


def compute_basket_index(lsoa: pd.DataFrame, schema: pd.DataFrame) -> pd.DataFrame:
    """Compute category scores, basket scores, and energy-cost-of-access metrics."""
    out = lsoa.copy()
    core_ids: list[str] = []

    for row in schema.to_dict("records"):
        cid = row["id"]
        col_local = row["source_column_local"]
        col_wider = row["source_column_wider"]
        nearest_col = row["nearest_column"]
        target = float(row["target_value"])
        floor = float(row["floor_value"])
        vals_local = pd.to_numeric(out[col_local], errors="coerce")
        vals_wider = pd.to_numeric(out[col_wider], errors="coerce")
        nearest_m = pd.to_numeric(out[nearest_col], errors="coerce")

        local_score = (vals_local / target).clip(lower=0, upper=1)
        wider_score = (vals_wider / target).clip(lower=0, upper=1)
        wider_score = np.maximum(wider_score, local_score)

        out[f"basket_{cid}_score_local"] = local_score
        out[f"basket_{cid}_score_wider"] = wider_score
        out[f"basket_{cid}_score_travel_gap"] = (wider_score - local_score).clip(
            lower=0, upper=1
        )
        # Backward-compatibility aliases (local baseline).
        out[f"basket_{cid}_score"] = out[f"basket_{cid}_score_local"]

        out[f"basket_{cid}_core_floor_met_local"] = vals_local >= floor
        out[f"basket_{cid}_core_floor_met_wider"] = vals_wider >= floor
        out[f"basket_{cid}_core_floor_met"] = out[f"basket_{cid}_core_floor_met_local"]

        out[f"basket_{cid}_value_local"] = vals_local
        out[f"basket_{cid}_value_wider"] = vals_wider
        out[f"basket_{cid}_value"] = out[f"basket_{cid}_value_local"]
        out[f"basket_{cid}_nearest_m"] = nearest_m
        out[f"basket_{cid}_local_access_flag"] = nearest_m <= LOCAL_ACCESS_M
        out[f"basket_{cid}_wider_access_flag"] = nearest_m <= WIDER_ACCESS_M
        if row["group"] == "core":
            core_ids.append(cid)

    # Weighted raw scores in [0,1] for local and wider catchments.
    weighted_local = np.zeros(len(out), dtype=float)
    weighted_wider = np.zeros(len(out), dtype=float)
    for row in schema.to_dict("records"):
        weighted_local += pd.to_numeric(
            out[f"basket_{row['id']}_score_local"], errors="coerce"
        ).fillna(0).to_numpy() * float(row["weight"])
        weighted_wider += pd.to_numeric(
            out[f"basket_{row['id']}_score_wider"], errors="coerce"
        ).fillna(0).to_numpy() * float(row["weight"])
    out["basket_raw_score_local"] = weighted_local
    out["basket_raw_score_wider"] = weighted_wider
    out["basket_raw_score"] = out["basket_raw_score_local"]

    # Core coverage (non-compensatory penalty proxy)
    core_floor_cols_local = [f"basket_{cid}_core_floor_met_local" for cid in core_ids]
    core_floor_cols_wider = [f"basket_{cid}_core_floor_met_wider" for cid in core_ids]
    out["basket_core_coverage_rate_local"] = (
        out[core_floor_cols_local].astype(float).mean(axis=1)
        if core_floor_cols_local
        else 1.0
    )
    out["basket_core_coverage_rate_wider"] = (
        out[core_floor_cols_wider].astype(float).mean(axis=1)
        if core_floor_cols_wider
        else 1.0
    )
    out["basket_core_coverage_rate"] = out["basket_core_coverage_rate_local"]
    out["basket_core_missing_count"] = (
        len(core_floor_cols_local) - out[core_floor_cols_local].astype(int).sum(axis=1)
        if core_floor_cols_local
        else 0
    )
    out["basket_core_missing_count_wider"] = (
        len(core_floor_cols_wider) - out[core_floor_cols_wider].astype(int).sum(axis=1)
        if core_floor_cols_wider
        else 0
    )

    # Penalise missing essentials without collapsing the scale entirely.
    out["basket_core_penalty_local"] = (
        0.5 + 0.5 * out["basket_core_coverage_rate_local"]
    )
    out["basket_core_penalty_wider"] = (
        0.5 + 0.5 * out["basket_core_coverage_rate_wider"]
    )
    out["basket_core_penalty"] = out["basket_core_penalty_local"]
    out["basket_score_local"] = (
        100 * out["basket_raw_score_local"] * out["basket_core_penalty_local"]
    )
    out["basket_score_wider"] = (
        100 * out["basket_raw_score_wider"] * out["basket_core_penalty_wider"]
    )
    out["basket_score"] = out["basket_score_local"]
    out["basket_score_travel_gap"] = (
        out["basket_score_wider"] - out["basket_score_local"]
    ).clip(lower=0)

    # Trip-demand weighted companion scores (local/wider).
    if (
        "trip_weight" in schema.columns
        and pd.to_numeric(schema["trip_weight"], errors="coerce").notna().any()
    ):
        weighted_trip_local = np.zeros(len(out), dtype=float)
        weighted_trip_wider = np.zeros(len(out), dtype=float)
        for row in schema.to_dict("records"):
            w_trip = float(row.get("trip_weight", 0.0))
            if not np.isfinite(w_trip):
                continue
            weighted_trip_local += (
                pd.to_numeric(out[f"basket_{row['id']}_score_local"], errors="coerce")
                .fillna(0)
                .to_numpy()
                * w_trip
            )
            weighted_trip_wider += (
                pd.to_numeric(out[f"basket_{row['id']}_score_wider"], errors="coerce")
                .fillna(0)
                .to_numpy()
                * w_trip
            )
        out["basket_trip_weighted_raw_score_local"] = weighted_trip_local
        out["basket_trip_weighted_raw_score_wider"] = weighted_trip_wider
        out["basket_trip_weighted_raw_score"] = out[
            "basket_trip_weighted_raw_score_local"
        ]
        out["basket_trip_weighted_score_local"] = (
            100
            * out["basket_trip_weighted_raw_score_local"]
            * out["basket_core_penalty_local"]
        )
        out["basket_trip_weighted_score_wider"] = (
            100
            * out["basket_trip_weighted_raw_score_wider"]
            * out["basket_core_penalty_wider"]
        )
        out["basket_trip_weighted_score"] = out["basket_trip_weighted_score_local"]
        out["basket_trip_weighted_score_travel_gap"] = (
            out["basket_trip_weighted_score_wider"]
            - out["basket_trip_weighted_score_local"]
        ).clip(lower=0)

    # Annual access trip budgets by catchment.
    if "avg_hh_size" not in out.columns:
        out["avg_hh_size"] = np.nan
    hh_size = pd.to_numeric(out["avg_hh_size"], errors="coerce")
    trip_budget_cols: list[str] = []
    trip_local_cols: list[str] = []
    trip_wider_cols: list[str] = []
    trip_extra_cols: list[str] = []
    trip_residual_cols: list[str] = []
    trip_rows = [
        row
        for row in schema.to_dict("records")
        if bool(row.get("include_in_trip_budget", True))
    ]
    for row in trip_rows:
        cid = row["id"]
        nearest_col = row.get("nearest_column")
        nearest_m = pd.to_numeric(out[nearest_col], errors="coerce")
        wider_access_flag = nearest_m <= WIDER_ACCESS_M
        local_access_share = _local_access_distance_share(
            nearest_m,
            half_distance_m=float(
                row.get("local_walk_half_distance_m", LOCAL_WALK_HALF_DISTANCE_M)
            ),
            decay_shape=float(
                row.get("local_walk_decay_shape", LOCAL_WALK_DECAY_SHAPE)
            ),
        )
        wider_access_share = wider_access_flag.astype(float)
        trips_pp = float(row.get("annual_trips_per_person", np.nan))
        trips_hh = float(row.get("annual_trips_per_household", np.nan))
        base_trips = pd.Series(0.0, index=out.index, dtype=float)
        if np.isfinite(trips_pp):
            base_trips = base_trips + (hh_size * trips_pp).fillna(0)
        if np.isfinite(trips_hh):
            base_trips = base_trips + trips_hh
        trips_col = f"basket_{cid}_trips_hh_est"
        local_col = f"basket_{cid}_local_accessible_trips_hh_est"
        wider_col = f"basket_{cid}_wider_accessible_trips_hh_est"
        extra_col = f"basket_{cid}_extra_travel_trips_hh_est"
        residual_col = f"basket_{cid}_residual_trips_hh_est"
        out[trips_col] = base_trips
        out[f"basket_{cid}_local_access_share"] = local_access_share
        out[f"basket_{cid}_wider_access_share"] = wider_access_share
        out[local_col] = base_trips * local_access_share
        out[wider_col] = base_trips * wider_access_share
        out[extra_col] = (out[wider_col] - out[local_col]).clip(lower=0)
        out[residual_col] = (base_trips - out[wider_col]).clip(lower=0)
        # Backward-compatibility aliases.
        out[f"basket_{cid}_accessible_trips_hh_est"] = out[local_col]
        out[f"basket_{cid}_unmet_trips_hh_est"] = out[extra_col]
        trip_budget_cols.append(trips_col)
        trip_local_cols.append(local_col)
        trip_wider_cols.append(wider_col)
        trip_extra_cols.append(extra_col)
        trip_residual_cols.append(residual_col)

    if trip_budget_cols:
        out["basket_trip_budget_hh_est"] = out[trip_budget_cols].sum(axis=1)
        out["basket_trip_local_accessible_hh_est"] = out[trip_local_cols].sum(axis=1)
        out["basket_trip_wider_accessible_hh_est"] = out[trip_wider_cols].sum(axis=1)
        out["basket_trip_extra_travel_hh_est"] = out[trip_extra_cols].sum(axis=1)
        out["basket_trip_residual_hh_est"] = out[trip_residual_cols].sum(axis=1)
        # Backward-compatibility aliases.
        out["basket_trip_accessible_hh_est"] = out[
            "basket_trip_local_accessible_hh_est"
        ]
        out["basket_trip_unmet_hh_est"] = out["basket_trip_extra_travel_hh_est"]
        out["basket_trip_coverage_rate_est"] = out[
            "basket_trip_local_accessible_hh_est"
        ] / out["basket_trip_budget_hh_est"].replace(0, np.nan)
        out["basket_trip_coverage_rate_local_est"] = out[
            "basket_trip_coverage_rate_est"
        ]
        out["basket_trip_coverage_rate_wider_est"] = out[
            "basket_trip_wider_accessible_hh_est"
        ] / out["basket_trip_budget_hh_est"].replace(0, np.nan)
    else:
        out["basket_trip_budget_hh_est"] = np.nan
        out["basket_trip_local_accessible_hh_est"] = np.nan
        out["basket_trip_wider_accessible_hh_est"] = np.nan
        out["basket_trip_extra_travel_hh_est"] = np.nan
        out["basket_trip_residual_hh_est"] = np.nan
        out["basket_trip_accessible_hh_est"] = np.nan
        out["basket_trip_unmet_hh_est"] = np.nan
        out["basket_trip_coverage_rate_est"] = np.nan
        out["basket_trip_coverage_rate_local_est"] = np.nan
        out["basket_trip_coverage_rate_wider_est"] = np.nan

    # Convert transport totals into trip-energy intensity to
    # estimate access-energy budgets.
    commute_trips_hh = (_NTS_COMMUTE_TRIPS_PER_PERSON * hh_size).replace(0, np.nan)
    total_trips_hh = (_NTS_TOTAL_TRIPS_PER_PERSON * hh_size).replace(0, np.nan)
    if "transport_kwh_per_hh_est" in out.columns:
        out["transport_kwh_per_trip_commute_est"] = (
            pd.to_numeric(out["transport_kwh_per_hh_est"], errors="coerce")
            / commute_trips_hh
        )
        out["basket_trip_budget_kwh_hh_commute_est"] = (
            out["basket_trip_budget_hh_est"] * out["transport_kwh_per_trip_commute_est"]
        )
        out["basket_trip_local_accessible_kwh_hh_commute_est"] = (
            out["basket_trip_local_accessible_hh_est"]
            * out["transport_kwh_per_trip_commute_est"]
        )
        out["basket_trip_wider_accessible_kwh_hh_commute_est"] = (
            out["basket_trip_wider_accessible_hh_est"]
            * out["transport_kwh_per_trip_commute_est"]
        )
        out["basket_trip_extra_travel_kwh_hh_commute_est"] = (
            out["basket_trip_extra_travel_hh_est"]
            * out["transport_kwh_per_trip_commute_est"]
        )
        out["basket_trip_residual_kwh_hh_commute_est"] = (
            out["basket_trip_residual_hh_est"]
            * out["transport_kwh_per_trip_commute_est"]
        )
        out["basket_trip_accessible_kwh_hh_commute_est"] = out[
            "basket_trip_local_accessible_kwh_hh_commute_est"
        ]
        out["basket_unmet_access_kwh_hh_commute_est"] = (
            out["basket_trip_extra_travel_hh_est"]
            * out["transport_kwh_per_trip_commute_est"]
        )
        out["land_use_access_penalty_kwh_hh_commute_est"] = out[
            "basket_unmet_access_kwh_hh_commute_est"
        ]
    if "transport_kwh_per_hh_total_est" in out.columns:
        out["transport_kwh_per_trip_total_est"] = (
            pd.to_numeric(out["transport_kwh_per_hh_total_est"], errors="coerce")
            / total_trips_hh
        )
        out["basket_trip_budget_kwh_hh_total_est"] = (
            out["basket_trip_budget_hh_est"] * out["transport_kwh_per_trip_total_est"]
        )
        out["basket_trip_local_accessible_kwh_hh_total_est"] = (
            out["basket_trip_local_accessible_hh_est"]
            * out["transport_kwh_per_trip_total_est"]
        )
        out["basket_trip_wider_accessible_kwh_hh_total_est"] = (
            out["basket_trip_wider_accessible_hh_est"]
            * out["transport_kwh_per_trip_total_est"]
        )
        out["basket_trip_extra_travel_kwh_hh_total_est"] = (
            out["basket_trip_extra_travel_hh_est"]
            * out["transport_kwh_per_trip_total_est"]
        )
        out["basket_trip_residual_kwh_hh_total_est"] = (
            out["basket_trip_residual_hh_est"] * out["transport_kwh_per_trip_total_est"]
        )
        out["basket_trip_accessible_kwh_hh_total_est"] = out[
            "basket_trip_local_accessible_kwh_hh_total_est"
        ]
        out["basket_unmet_access_kwh_hh_total_est"] = (
            out["basket_trip_extra_travel_hh_est"]
            * out["transport_kwh_per_trip_total_est"]
        )
        out["land_use_access_penalty_kwh_hh_total_est"] = out[
            "basket_unmet_access_kwh_hh_total_est"
        ]
        out["transport_kwh_per_local_accessible_trip_total_est"] = pd.to_numeric(
            out["transport_kwh_per_hh_total_est"], errors="coerce"
        ) / out["basket_trip_local_accessible_hh_est"].replace(0, np.nan)
        out["transport_kwh_per_accessible_trip_total_est"] = out[
            "transport_kwh_per_local_accessible_trip_total_est"
        ]
        out["kwh_per_local_accessible_trip_total_est"] = pd.to_numeric(
            out["total_kwh_per_hh_total_est"], errors="coerce"
        ) / out["basket_trip_local_accessible_hh_est"].replace(0, np.nan)
        out["kwh_per_wider_accessible_trip_total_est"] = pd.to_numeric(
            out["total_kwh_per_hh_total_est"], errors="coerce"
        ) / out["basket_trip_wider_accessible_hh_est"].replace(0, np.nan)
        out["kwh_per_accessible_trip_total_est"] = out[
            "kwh_per_local_accessible_trip_total_est"
        ]
    out["kwh_per_local_accessible_trip_commute_base"] = pd.to_numeric(
        out["total_kwh_per_hh"], errors="coerce"
    ) / out["basket_trip_local_accessible_hh_est"].replace(0, np.nan)
    out["kwh_per_accessible_trip_commute_base"] = out[
        "kwh_per_local_accessible_trip_commute_base"
    ]

    # Energy-cost-of-access metrics
    # (primary = household denominator for continuity with PoC)
    basket = out["basket_score"].replace(0, np.nan)
    out["kwh_per_basket_point_hh"] = out["total_kwh_per_hh"] / basket
    out["basket_points_per_10mwh_hh"] = basket / (out["total_kwh_per_hh"] / 10_000)
    if "total_kwh_per_hh_total_est" in out.columns:
        out["kwh_per_basket_point_hh_total_est"] = (
            out["total_kwh_per_hh_total_est"] / basket
        )
        out["basket_points_per_10mwh_hh_total_est"] = basket / (
            out["total_kwh_per_hh_total_est"] / 10_000
        )
    if "total_kwh_per_person" in out.columns:
        out["kwh_per_basket_point_person"] = out["total_kwh_per_person"] / basket
        out["basket_points_per_mwh_person"] = basket / (
            out["total_kwh_per_person"] / 1_000
        )

    # Deprivation quintiles for summary figure/table
    if "pct_not_deprived" in out.columns:
        valid = out["pct_not_deprived"].notna()
        if valid.sum() > 10:
            out.loc[valid, "basket_dep_quintile"] = pd.qcut(
                out.loc[valid, "pct_not_deprived"],
                5,
                labels=["Q1 most", "Q2", "Q3", "Q4", "Q5 least"],
            )
    return out


def save_schema_table(schema: pd.DataFrame) -> Path:
    out_path = OUT_DIR / "table_basket_v1_schema.csv"
    cols = [
        "id",
        "label",
        "group",
        "weight",
        "local_walk_half_distance_m",
        "local_walk_decay_shape",
        "source_column",
        "source_column_local",
        "source_column_wider",
        "floor_quantile",
        "target_quantile",
        "floor_value",
        "target_value",
        "median_value",
        "median_value_wider",
        "q90_value",
        "positive_share",
        "annual_trips_per_person",
        "annual_trips_per_household",
        "include_in_trip_budget",
        "trip_demand_equiv_per_person",
        "trip_weight",
        "trip_source",
        "trip_method_note",
    ]
    schema[cols].to_csv(out_path, index=False)
    return out_path


def summarise_by_type(lsoa: pd.DataFrame, schema: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for t in TYPE_ORDER:
        sub = lsoa[lsoa["dominant_type"] == t]
        if len(sub) == 0:
            continue
        row: dict[str, object] = {
            "type": t,
            "type_label": TYPE_LABELS.get(t, t),
            "n_lsoas": int(len(sub)),
            "people_per_ha_median": float(sub["people_per_ha"].median())
            if "people_per_ha" in sub
            else np.nan,
            "total_kwh_per_hh_median": float(sub["total_kwh_per_hh"].median()),
            "total_kwh_per_person_median": float(sub["total_kwh_per_person"].median()),
            "basket_score_median": float(sub["basket_score"].median()),
            "basket_score_local_median": float(sub["basket_score_local"].median())
            if "basket_score_local" in sub
            else np.nan,
            "basket_score_wider_median": float(sub["basket_score_wider"].median())
            if "basket_score_wider" in sub
            else np.nan,
            "basket_score_travel_gap_median": float(
                sub["basket_score_travel_gap"].median()
            )
            if "basket_score_travel_gap" in sub
            else np.nan,
            "basket_raw_score_median": float(sub["basket_raw_score"].median()),
            "basket_core_coverage_rate_median": float(
                sub["basket_core_coverage_rate"].median()
            ),
            "basket_core_coverage_rate_local_median": float(
                sub["basket_core_coverage_rate_local"].median()
            )
            if "basket_core_coverage_rate_local" in sub
            else np.nan,
            "basket_core_coverage_rate_wider_median": float(
                sub["basket_core_coverage_rate_wider"].median()
            )
            if "basket_core_coverage_rate_wider" in sub
            else np.nan,
            "kwh_per_basket_point_hh_median": float(
                sub["kwh_per_basket_point_hh"].median()
            ),
            "basket_points_per_10mwh_hh_median": float(
                sub["basket_points_per_10mwh_hh"].median()
            ),
            "basket_trip_weighted_score_median": float(
                sub["basket_trip_weighted_score"].median()
            )
            if "basket_trip_weighted_score" in sub
            else np.nan,
            "basket_trip_budget_hh_est_median": float(
                sub["basket_trip_budget_hh_est"].median()
            )
            if "basket_trip_budget_hh_est" in sub
            else np.nan,
            "basket_trip_local_accessible_hh_est_median": float(
                sub["basket_trip_local_accessible_hh_est"].median()
            )
            if "basket_trip_local_accessible_hh_est" in sub
            else np.nan,
            "basket_trip_wider_accessible_hh_est_median": float(
                sub["basket_trip_wider_accessible_hh_est"].median()
            )
            if "basket_trip_wider_accessible_hh_est" in sub
            else np.nan,
            "basket_trip_extra_travel_hh_est_median": float(
                sub["basket_trip_extra_travel_hh_est"].median()
            )
            if "basket_trip_extra_travel_hh_est" in sub
            else np.nan,
            "basket_trip_residual_hh_est_median": float(
                sub["basket_trip_residual_hh_est"].median()
            )
            if "basket_trip_residual_hh_est" in sub
            else np.nan,
            "basket_trip_accessible_hh_est_median": float(
                sub["basket_trip_accessible_hh_est"].median()
            )
            if "basket_trip_accessible_hh_est" in sub
            else np.nan,
            "basket_trip_unmet_hh_est_median": float(
                sub["basket_trip_unmet_hh_est"].median()
            )
            if "basket_trip_unmet_hh_est" in sub
            else np.nan,
            "basket_trip_coverage_rate_est_median": float(
                sub["basket_trip_coverage_rate_est"].median()
            )
            if "basket_trip_coverage_rate_est" in sub
            else np.nan,
            "basket_trip_coverage_rate_local_est_median": float(
                sub["basket_trip_coverage_rate_local_est"].median()
            )
            if "basket_trip_coverage_rate_local_est" in sub
            else np.nan,
            "basket_trip_coverage_rate_wider_est_median": float(
                sub["basket_trip_coverage_rate_wider_est"].median()
            )
            if "basket_trip_coverage_rate_wider_est" in sub
            else np.nan,
            "basket_trip_budget_kwh_hh_commute_est_median": float(
                sub["basket_trip_budget_kwh_hh_commute_est"].median()
            )
            if "basket_trip_budget_kwh_hh_commute_est" in sub
            else np.nan,
            "basket_trip_accessible_kwh_hh_commute_est_median": float(
                sub["basket_trip_accessible_kwh_hh_commute_est"].median()
            )
            if "basket_trip_accessible_kwh_hh_commute_est" in sub
            else np.nan,
            "basket_unmet_access_kwh_hh_commute_est_median": float(
                sub["basket_unmet_access_kwh_hh_commute_est"].median()
            )
            if "basket_unmet_access_kwh_hh_commute_est" in sub
            else np.nan,
            "land_use_access_penalty_kwh_hh_commute_est_median": float(
                sub["land_use_access_penalty_kwh_hh_commute_est"].median()
            )
            if "land_use_access_penalty_kwh_hh_commute_est" in sub
            else np.nan,
            "basket_trip_extra_travel_kwh_hh_commute_est_median": float(
                sub["basket_trip_extra_travel_kwh_hh_commute_est"].median()
            )
            if "basket_trip_extra_travel_kwh_hh_commute_est" in sub
            else np.nan,
            "kwh_per_accessible_trip_commute_base_median": float(
                sub["kwh_per_accessible_trip_commute_base"].median()
            )
            if "kwh_per_accessible_trip_commute_base" in sub
            else np.nan,
            "cc_density_800_median": float(sub["cc_density_800"].median())
            if "cc_density_800" in sub
            else np.nan,
            "cc_harmonic_800_median": float(sub["cc_harmonic_800"].median())
            if "cc_harmonic_800" in sub
            else np.nan,
        }
        if "total_kwh_per_hh_total_est" in sub.columns:
            row["total_kwh_per_hh_total_est_median"] = float(
                sub["total_kwh_per_hh_total_est"].median()
            )
            row["kwh_per_basket_point_hh_total_est_median"] = float(
                sub["kwh_per_basket_point_hh_total_est"].median()
            )
            row["basket_points_per_10mwh_hh_total_est_median"] = float(
                sub["basket_points_per_10mwh_hh_total_est"].median()
            )
            if "kwh_per_accessible_trip_total_est" in sub.columns:
                row["kwh_per_accessible_trip_total_est_median"] = float(
                    sub["kwh_per_accessible_trip_total_est"].median()
                )
            if "transport_kwh_per_accessible_trip_total_est" in sub.columns:
                row["transport_kwh_per_accessible_trip_total_est_median"] = float(
                    sub["transport_kwh_per_accessible_trip_total_est"].median()
                )
            if "basket_trip_budget_kwh_hh_total_est" in sub.columns:
                row["basket_trip_budget_kwh_hh_total_est_median"] = float(
                    sub["basket_trip_budget_kwh_hh_total_est"].median()
                )
            if "basket_trip_local_accessible_kwh_hh_total_est" in sub.columns:
                row["basket_trip_local_accessible_kwh_hh_total_est_median"] = float(
                    sub["basket_trip_local_accessible_kwh_hh_total_est"].median()
                )
            if "basket_trip_wider_accessible_kwh_hh_total_est" in sub.columns:
                row["basket_trip_wider_accessible_kwh_hh_total_est_median"] = float(
                    sub["basket_trip_wider_accessible_kwh_hh_total_est"].median()
                )
            if "basket_trip_extra_travel_kwh_hh_total_est" in sub.columns:
                row["basket_trip_extra_travel_kwh_hh_total_est_median"] = float(
                    sub["basket_trip_extra_travel_kwh_hh_total_est"].median()
                )
            if "basket_trip_residual_kwh_hh_total_est" in sub.columns:
                row["basket_trip_residual_kwh_hh_total_est_median"] = float(
                    sub["basket_trip_residual_kwh_hh_total_est"].median()
                )
            if "basket_trip_accessible_kwh_hh_total_est" in sub.columns:
                row["basket_trip_accessible_kwh_hh_total_est_median"] = float(
                    sub["basket_trip_accessible_kwh_hh_total_est"].median()
                )
            if "basket_unmet_access_kwh_hh_total_est" in sub.columns:
                row["basket_unmet_access_kwh_hh_total_est_median"] = float(
                    sub["basket_unmet_access_kwh_hh_total_est"].median()
                )
            if "land_use_access_penalty_kwh_hh_total_est" in sub.columns:
                row["land_use_access_penalty_kwh_hh_total_est_median"] = float(
                    sub["land_use_access_penalty_kwh_hh_total_est"].median()
                )
        for r in schema.to_dict("records"):
            cid = r["id"]
            row[f"{cid}_score_median"] = float(sub[f"basket_{cid}_score"].median())
            row[f"{cid}_score_local_median"] = float(
                sub[f"basket_{cid}_score_local"].median()
            )
            row[f"{cid}_score_wider_median"] = float(
                sub[f"basket_{cid}_score_wider"].median()
            )
            row[f"{cid}_value_median"] = float(sub[f"basket_{cid}_value"].median())
            row[f"{cid}_value_local_median"] = float(
                sub[f"basket_{cid}_value_local"].median()
            )
            row[f"{cid}_value_wider_median"] = float(
                sub[f"basket_{cid}_value_wider"].median()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def save_summary_tables(lsoa: pd.DataFrame, schema: pd.DataFrame) -> tuple[Path, Path]:
    type_summary = summarise_by_type(lsoa, schema)
    type_path = OUT_DIR / "table_basket_v1_by_type.csv"
    type_summary.to_csv(type_path, index=False)

    dep_rows: list[dict[str, object]] = []
    if "basket_dep_quintile" in lsoa.columns:
        for dep in ["Q1 most", "Q2", "Q3", "Q4", "Q5 least"]:
            sub = lsoa[lsoa["basket_dep_quintile"] == dep]
            if len(sub) == 0:
                continue
            dep_rows.append(
                {
                    "deprivation_quintile": dep,
                    "n_lsoas": int(len(sub)),
                    "total_kwh_per_hh_median": float(sub["total_kwh_per_hh"].median()),
                    "basket_score_median": float(sub["basket_score"].median()),
                    "kwh_per_basket_point_hh_median": float(
                        sub["kwh_per_basket_point_hh"].median()
                    ),
                    "core_coverage_rate_median": float(
                        sub["basket_core_coverage_rate"].median()
                    ),
                    "basket_trip_coverage_rate_est_median": float(
                        sub["basket_trip_coverage_rate_est"].median()
                    )
                    if "basket_trip_coverage_rate_est" in sub
                    else np.nan,
                    "basket_trip_coverage_rate_wider_est_median": float(
                        sub["basket_trip_coverage_rate_wider_est"].median()
                    )
                    if "basket_trip_coverage_rate_wider_est" in sub
                    else np.nan,
                }
            )
            if "total_kwh_per_hh_total_est" in sub.columns:
                dep_rows[-1]["total_kwh_per_hh_total_est_median"] = float(
                    sub["total_kwh_per_hh_total_est"].median()
                )
                dep_rows[-1]["kwh_per_basket_point_hh_total_est_median"] = float(
                    sub["kwh_per_basket_point_hh_total_est"].median()
                )
                if "basket_trip_extra_travel_kwh_hh_total_est" in sub.columns:
                    dep_rows[-1]["basket_trip_extra_travel_kwh_hh_total_est_median"] = (
                        float(sub["basket_trip_extra_travel_kwh_hh_total_est"].median())
                    )
                if "basket_unmet_access_kwh_hh_total_est" in sub.columns:
                    dep_rows[-1]["basket_unmet_access_kwh_hh_total_est_median"] = float(
                        sub["basket_unmet_access_kwh_hh_total_est"].median()
                    )
    dep_path = OUT_DIR / "table_basket_v1_by_deprivation.csv"
    pd.DataFrame(dep_rows).to_csv(dep_path, index=False)
    return type_path, dep_path


def save_lsoa_scores(lsoa: pd.DataFrame, schema: pd.DataFrame) -> Path:
    """Export a slim LSOA-level scored table for inspection and mapping."""
    keep = [
        "LSOA21CD",
        "city",
        "dominant_type",
        "people_per_ha",
        "total_kwh_per_hh",
        "total_kwh_per_person",
        "basket_raw_score",
        "basket_raw_score_local",
        "basket_raw_score_wider",
        "basket_core_coverage_rate",
        "basket_core_coverage_rate_local",
        "basket_core_coverage_rate_wider",
        "basket_core_missing_count",
        "basket_score",
        "basket_score_local",
        "basket_score_wider",
        "basket_score_travel_gap",
        "kwh_per_basket_point_hh",
        "basket_points_per_10mwh_hh",
        "basket_trip_weighted_score",
        "basket_trip_budget_hh_est",
        "basket_trip_local_accessible_hh_est",
        "basket_trip_wider_accessible_hh_est",
        "basket_trip_extra_travel_hh_est",
        "basket_trip_residual_hh_est",
        "basket_trip_accessible_hh_est",
        "basket_trip_unmet_hh_est",
        "basket_trip_coverage_rate_est",
        "basket_trip_coverage_rate_wider_est",
        "basket_trip_budget_kwh_hh_commute_est",
        "basket_trip_local_accessible_kwh_hh_commute_est",
        "basket_trip_wider_accessible_kwh_hh_commute_est",
        "basket_trip_extra_travel_kwh_hh_commute_est",
        "basket_trip_residual_kwh_hh_commute_est",
        "basket_trip_accessible_kwh_hh_commute_est",
        "basket_unmet_access_kwh_hh_commute_est",
        "land_use_access_penalty_kwh_hh_commute_est",
        "basket_trip_budget_kwh_hh_total_est",
        "basket_trip_local_accessible_kwh_hh_total_est",
        "basket_trip_wider_accessible_kwh_hh_total_est",
        "basket_trip_extra_travel_kwh_hh_total_est",
        "basket_trip_residual_kwh_hh_total_est",
        "basket_trip_accessible_kwh_hh_total_est",
        "basket_unmet_access_kwh_hh_total_est",
        "land_use_access_penalty_kwh_hh_total_est",
        "transport_kwh_per_local_accessible_trip_total_est",
        "transport_kwh_per_accessible_trip_total_est",
        "kwh_per_local_accessible_trip_total_est",
        "kwh_per_wider_accessible_trip_total_est",
        "kwh_per_local_accessible_trip_commute_base",
        "kwh_per_accessible_trip_commute_base",
        "kwh_per_basket_point_person",
        "basket_points_per_mwh_person",
        "cc_density_800",
        "cc_harmonic_800",
        "pct_not_deprived",
    ]
    for row in schema.to_dict("records"):
        cid = row["id"]
        keep.extend(
            [
                f"basket_{cid}_value",
                f"basket_{cid}_value_local",
                f"basket_{cid}_value_wider",
                f"basket_{cid}_score",
                f"basket_{cid}_score_local",
                f"basket_{cid}_score_wider",
                f"basket_{cid}_score_travel_gap",
                f"basket_{cid}_core_floor_met",
                f"basket_{cid}_core_floor_met_local",
                f"basket_{cid}_core_floor_met_wider",
                f"basket_{cid}_nearest_m",
                f"basket_{cid}_local_access_flag",
                f"basket_{cid}_wider_access_flag",
                f"basket_{cid}_local_access_share",
                f"basket_{cid}_wider_access_share",
            ]
        )
    keep = [c for c in keep if c in lsoa.columns]
    out_path = OUT_DIR / "lsoa_basket_v1_scores.csv"
    lsoa[keep].to_csv(out_path, index=False)
    return out_path


def fig1_category_scores_heatmap(lsoa: pd.DataFrame, schema: pd.DataFrame) -> Path:
    """Heatmap of local-access rates by dominant housing type."""
    rows = []
    schema_rows = schema.to_dict("records")
    for t in TYPE_ORDER:
        sub = lsoa[lsoa["dominant_type"] == t]
        if len(sub) == 0:
            continue
        for r in schema_rows:
            rows.append(
                {
                    "type": TYPE_LABELS.get(t, t),
                    "category": r["label"],
                    "score": float(
                        sub[f"basket_{r['id']}_local_access_flag"].astype(float).mean()
                    ),
                    "group": r["group"],
                }
            )
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="category", columns="type", values="score")
    col_order = [TYPE_LABELS[t] for t in TYPE_ORDER if TYPE_LABELS[t] in pivot.columns]
    pivot = pivot.reindex(columns=col_order)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        pivot.loc[[r["label"] for r in schema_rows]],
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "Share of LSOAs with local access <= 800m"},
        ax=ax,
    )
    ax.set_title(
        "Basket v1: Local Access Presence by Dominant Housing Type", fontweight="bold"
    )
    ax.set_xlabel("Dominant housing type (Census TS044)")
    ax.set_ylabel("Basket category")
    plt.tight_layout()
    out_path = OUT_DIR / "fig_basket_v1_category_scores_heatmap.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def fig2_basket_and_cost_bars(lsoa: pd.DataFrame) -> Path:
    """Two-panel bars: local trip coverage and access penalty.

    Grouped by housing type.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(TYPE_ORDER))
    colors = [TYPE_COLORS[t] for t in TYPE_ORDER]

    local_cov_vals = [
        100
        * lsoa.loc[
            lsoa["dominant_type"] == t, "basket_trip_coverage_rate_local_est"
        ].mean()
        for t in TYPE_ORDER
    ]
    penalty_vals = [
        lsoa.loc[
            lsoa["dominant_type"] == t, "land_use_access_penalty_kwh_hh_total_est"
        ].mean()
        for t in TYPE_ORDER
    ]

    # Panel A
    ax = axes[0]
    bars = ax.bar(x, local_cov_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([TYPE_LABELS[t] for t in TYPE_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Distance-weighted local trip coverage (%)")
    ax.set_title(
        "(A) Local access delivered with distance decay", fontweight="bold", fontsize=10
    )
    for bar, v in zip(bars, local_cov_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.8,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Panel B
    ax = axes[1]
    bars = ax.bar(x, penalty_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([TYPE_LABELS[t] for t in TYPE_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Additional access energy (kWh / household / year)")
    ax.set_title("(B) Land-use access penalty", fontweight="bold", fontsize=10)
    for bar, v in zip(bars, penalty_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + ax.get_ylim()[1] * 0.01,
            f"{v:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Headline ratio annotation
    flat_pen = penalty_vals[0]
    det_pen = penalty_vals[-1]
    if pd.notna(flat_pen) and pd.notna(det_pen) and flat_pen > 0:
        axes[1].annotate(
            f"Detached / Flat = {det_pen / flat_pen:.2f}x",
            xy=(0.5, 0.94),
            xycoords="axes fraction",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#c0392b",
        )

    fig.suptitle(
        "Basket v1: Local Access and Land-Use Travel Penalty by Housing Type",
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out_path = OUT_DIR / "fig_basket_v1_by_type.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def _kde_gradient_linear(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> tuple[float, float] | None:
    """Draw gradient-filled KDE contours and return the KDE peak point."""
    from scipy.stats import gaussian_kde  # noqa: E402

    if len(x) < 20:
        return None
    # Guard against singular covariance when values are nearly constant.
    if np.isclose(np.nanstd(x), 0) or np.isclose(np.nanstd(y), 0):
        return None

    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, bw_method=0.5)
    xi = np.linspace(xmin, xmax, 220)
    yi = np.linspace(ymin, ymax, 220)
    Xi, Yi = np.meshgrid(xi, yi)
    Z = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
    if np.nanmax(Z) <= 0:
        return None
    Z = Z / Z.max()

    band_levels = [0.35, 0.55, 0.72, 0.85, 0.93]
    band_alphas = [0.07, 0.13, 0.21, 0.33, 0.52]
    for lo, hi, a in zip(band_levels, band_levels[1:] + [1.01], band_alphas):
        ax.contourf(Xi, Yi, Z, levels=[lo, hi], colors=[color], alpha=a)
    ax.contour(Xi, Yi, Z, levels=[0.35], colors=[color], linewidths=[1.0], alpha=0.7)
    peak_idx = np.unravel_index(np.nanargmax(Z), Z.shape)
    return float(Xi[peak_idx]), float(Yi[peak_idx])


def _add_centroid_marker(ax: plt.Axes, x: float, y: float, color: str) -> None:
    """Draw a centroid marker on top of a contour cluster."""
    if pd.isna(x) or pd.isna(y):
        return
    ax.scatter(
        [x],
        [y],
        s=26,
        color=color,
        edgecolors="black",
        linewidths=0.5,
        zorder=6,
    )


def fig3_energy_vs_basket_scatter(lsoa: pd.DataFrame) -> Path:
    """KDE contours: total energy vs local trip coverage by dominant housing type."""
    sub = (
        lsoa[
            ["total_kwh_per_hh", "basket_trip_coverage_rate_local_est", "dominant_type"]
        ]
        .dropna()
        .copy()
    )
    sub = sub[
        (sub["basket_trip_coverage_rate_local_est"] >= 0)
        & (sub["basket_trip_coverage_rate_local_est"] <= 1)
        & (sub["total_kwh_per_hh"] > 0)
    ].copy()

    x = sub["total_kwh_per_hh"]
    y = 100 * sub["basket_trip_coverage_rate_local_est"]
    x_lo, x_hi = float(x.quantile(0.01)), float(x.quantile(0.99))
    y_lo, y_hi = 0.0, float(min(100.0, y.quantile(0.995) * 1.03))

    fig, ax = plt.subplots(figsize=(8, 6))

    for t in TYPE_ORDER:
        pts = sub[sub["dominant_type"] == t]
        if len(pts) < 20:
            continue
        x_vals = pts["total_kwh_per_hh"].to_numpy()
        y_vals = (100 * pts["basket_trip_coverage_rate_local_est"]).to_numpy()
        peak_xy = _kde_gradient_linear(
            ax, x_vals, y_vals, TYPE_COLORS[t], x_lo, x_hi, y_lo, y_hi
        )
        if peak_xy is not None:
            _add_centroid_marker(ax, peak_xy[0], peak_xy[1], TYPE_COLORS[t])
        ax.plot(
            [], [], color=TYPE_COLORS[t], linewidth=8, alpha=0.35, label=TYPE_LABELS[t]
        )

    x_med = float(sub["total_kwh_per_hh"].median())
    y_med = float((100 * sub["basket_trip_coverage_rate_local_est"]).median())
    ax.axvline(x_med, color="#666", linestyle="--", linewidth=1)
    ax.axhline(y_med, color="#666", linestyle="--", linewidth=1)
    ax.text(
        x_med + (x_hi - x_lo) * 0.01,
        y_med + (y_hi - y_lo) * 0.02,
        "Pilot medians",
        fontsize=8,
        color="#555",
    )

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Total energy (kWh / household)")
    ax.set_ylabel("Local trip coverage (0-100)")
    ax.set_title(
        "Basket v1: Energy vs Local Access Coverage (KDE by Housing Type)",
        fontweight="bold",
    )
    ax.legend(title="Dominant type", fontsize=8)
    plt.tight_layout()
    out_path = OUT_DIR / "fig_basket_v1_scatter_energy_vs_basket.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def fig4_deprivation_gradient(lsoa: pd.DataFrame) -> Path | None:
    """Line plot of land-use access penalty by deprivation quintile."""
    if "basket_dep_quintile" not in lsoa.columns:
        return None
    dep_order = ["Q1 most", "Q2", "Q3", "Q4", "Q5 least"]
    rows = []
    for dep in dep_order:
        for t in TYPE_ORDER:
            sub = lsoa[
                (lsoa["basket_dep_quintile"] == dep) & (lsoa["dominant_type"] == t)
            ]
            if len(sub) < 10:
                continue
            rows.append(
                {
                    "deprivation_quintile": dep,
                    "type": TYPE_LABELS.get(t, t),
                    "land_use_access_penalty_kwh_hh_total_est": float(
                        sub["land_use_access_penalty_kwh_hh_total_est"].median()
                    ),
                    "local_trip_coverage_pct": float(
                        100 * sub["basket_trip_coverage_rate_local_est"].median()
                    ),
                    "n": len(sub),
                }
            )
    if not rows:
        return None
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for t in TYPE_ORDER:
        label = TYPE_LABELS[t]
        pts = df[df["type"] == label]
        if len(pts) == 0:
            continue
        pts = pts.set_index("deprivation_quintile").reindex(dep_order).reset_index()
        axes[0].plot(
            dep_order,
            pts["land_use_access_penalty_kwh_hh_total_est"],
            marker="o",
            label=label,
            color=TYPE_COLORS[t],
        )
        axes[1].plot(
            dep_order,
            pts["local_trip_coverage_pct"],
            marker="o",
            label=label,
            color=TYPE_COLORS[t],
        )

    axes[0].set_title("(A) Land-use access penalty", fontweight="bold", fontsize=10)
    axes[0].set_ylabel("Additional access energy (kWh/hh/yr)")
    axes[1].set_title("(B) Local trip coverage", fontweight="bold", fontsize=10)
    axes[1].set_ylabel("Local trip coverage (%)")
    for ax in axes:
        ax.set_xlabel("Deprivation quintile (% households not deprived)")
        ax.tick_params(axis="x", rotation=20)
    axes[1].legend(title="Dominant type", fontsize=8)
    fig.suptitle(
        "Basket v1: Deprivation Gradient of Access Burden", fontweight="bold", y=1.03
    )
    plt.tight_layout()
    out_path = OUT_DIR / "fig_basket_v1_deprivation_gradient.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def _markdown_table(df: pd.DataFrame) -> str:
    """Format a small dataframe as GitHub-flavored markdown without external deps."""
    if df.empty:
        return "_No rows_"

    def _fmt(v: object) -> str:
        if pd.isna(v):
            return ""
        return str(v)

    headers = [str(c) for c in df.columns]
    rows = [[_fmt(v) for v in row] for row in df.to_numpy()]
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))
    ]

    def _line(vals: list[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(vals)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    parts = [_line(headers), sep]
    parts.extend(_line(r) for r in rows)
    return "\n".join(parts)


def _round_table_for_markdown(
    df: pd.DataFrame, cols_map: list[tuple[str, str, int]]
) -> str:
    """Return a compact markdown table for selected columns."""
    out = df.copy()
    for src, _, decimals in cols_map:
        if not pd.api.types.is_numeric_dtype(out[src]):
            out[src] = out[src].fillna("").astype(str)
        elif decimals == 0:
            out[src] = out[src].round(0).astype("Int64")
        else:
            out[src] = out[src].round(decimals)
    rename = {src: dst for src, dst, _ in cols_map}
    out = out[[src for src, _, _ in cols_map]].rename(columns=rename)
    return _markdown_table(out)


def write_summary_document(
    lsoa: pd.DataFrame,
    schema: pd.DataFrame,
    type_summary: pd.DataFrame,
    figure_paths: list[Path],
) -> Path:
    """Write a draft markdown summary of the basket v1 results."""
    total_n = len(lsoa)
    n_cities = lsoa["city"].nunique() if "city" in lsoa.columns else np.nan

    flat = type_summary[type_summary["type"] == "Flat"].iloc[0]
    det = type_summary[type_summary["type"] == "Detached"].iloc[0]
    _terr = type_summary[type_summary["type"] == "Terraced"].iloc[0]  # noqa: F841
    _semi = type_summary[type_summary["type"] == "Semi"].iloc[0]  # noqa: F841

    detached_vs_flat_cost = (
        det["kwh_per_basket_point_hh_median"] / flat["kwh_per_basket_point_hh_median"]
    )
    flat_vs_det_yield = (
        flat["basket_points_per_10mwh_hh_median"]
        / det["basket_points_per_10mwh_hh_median"]
    )
    detached_vs_flat_energy = (
        det["total_kwh_per_hh_median"] / flat["total_kwh_per_hh_median"]
    )
    flat_vs_det_basket = flat["basket_score_median"] / det["basket_score_median"]

    schema_md = _round_table_for_markdown(
        schema,
        [
            ("label", "Category", 2),
            ("group", "Group", 2),
            ("weight", "Weight", 2),
            ("trip_weight", "Trip wt", 2),
            ("include_in_trip_budget", "In trip budget", 2),
            ("source_column_local", "Local col (800m)", 2),
            ("source_column_wider", "Wider col (4800m)", 2),
            ("floor_value", "Core floor (q)", 3),
            ("target_value", "Target (q)", 3),
            ("annual_trips_per_person", "Trips/person/yr", 1),
            ("annual_trips_per_household", "Trips/hh/yr", 1),
            ("positive_share", "Positive share", 2),
        ],
    )
    type_md = _round_table_for_markdown(
        type_summary,
        [
            ("type_label", "Type", 2),
            ("n_lsoas", "N", 0),
            ("total_kwh_per_hh_median", "Total kWh/hh", 0),
            ("basket_score_median", "Basket score", 1),
            ("basket_core_coverage_rate_median", "Core coverage", 2),
            ("kwh_per_basket_point_hh_median", "kWh / basket pt", 0),
            ("basket_points_per_10mwh_hh_median", "Basket pts / 10MWh", 1),
            ("basket_trip_coverage_rate_local_est_median", "Local trip coverage", 2),
            ("basket_trip_coverage_rate_wider_est_median", "Wider trip coverage", 2),
            ("basket_trip_extra_travel_hh_est_median", "Extra-travel trips/hh/yr", 1),
            (
                "basket_trip_extra_travel_kwh_hh_total_est_median",
                "Extra-travel kWh/hh/yr (overall est)",
                0,
            ),
            (
                "kwh_per_accessible_trip_total_est_median",
                "kWh / accessible trip (overall est)",
                2,
            ),
        ],
    )

    dep_df = pd.DataFrame()
    if "basket_dep_quintile" in lsoa.columns:
        dep_rows = []
        for dep in ["Q1 most", "Q2", "Q3", "Q4", "Q5 least"]:
            sub = lsoa[lsoa["basket_dep_quintile"] == dep]
            if len(sub) == 0:
                continue
            dep_rows.append(
                {
                    "Deprivation quintile": dep,
                    "N": len(sub),
                    "Total kWh/hh": round(float(sub["total_kwh_per_hh"].median())),
                    "Basket score": round(float(sub["basket_score"].median()), 1),
                    "kWh / basket pt": round(
                        float(sub["kwh_per_basket_point_hh"].median())
                    ),
                    "Trip coverage": round(
                        float(sub["basket_trip_coverage_rate_est"].median()), 2
                    )
                    if "basket_trip_coverage_rate_est" in sub
                    else np.nan,
                }
            )
        dep_df = pd.DataFrame(dep_rows)

    dep_md = _markdown_table(dep_df) if not dep_df.empty else "_Not available_"

    rel_figs = [p.relative_to(PROJECT_DIR) for p in figure_paths]
    fig_map = {p.name: p for p in rel_figs}

    def _fig_md(name: str, alt: str) -> str:
        p = fig_map.get(name)
        if p is None:
            return f"_Missing figure: {name}_"
        return f"![{alt}](../{p.as_posix()})"

    figure_lines = "\n".join(f"- `../{p.as_posix()}`" for p in rel_figs)
    fig_heatmap = _fig_md(
        "fig_basket_v1_category_scores_heatmap.png",
        "Basket v1 category scores heatmap",
    )
    fig_by_type = _fig_md(
        "fig_basket_v1_by_type.png",
        "Basket attainment and energy cost by housing type",
    )
    fig_scatter = _fig_md(
        "fig_basket_v1_scatter_energy_vs_basket.png",
        "KDE contours of energy vs local trip coverage by housing type",
    )
    fig_dep = _fig_md(
        "fig_basket_v1_deprivation_gradient.png",
        "Basket v1 deprivation gradient",
    )

    text = (
        "# Morphology, Energy, and the Cost of Ordinary Access"
        " (Pilot): Data and Figures Appendix\n"
        "\n"
        "## Purpose\n"
        "\n"
        "This is the **generated data-and-figures appendix** for the"
        " pilot case note on morphology-\n"
        "linked energy cost of access at LSOA level.\n"
        "\n"
        "Use this alongside the canonical narrative case document:\n"
        "\n"
        "- `paper/case_v1.md`\n"
        "\n"
        "This appendix consolidates:\n"
        "\n"
        "- basket v1 schema and calibration\n"
        "- summary tables\n"
        "- generated figure inventory\n"
        "- headline pilot metrics\n"
        "- technical caveats and upgrade path\n"
        "\n"
        "## Status (What v1 Is)\n"
        "\n"
        f"- **Pilot geography:** {total_n:,} LSOAs across"
        f" {int(n_cities)} English cities\n"
        "- **Accessibility basis:** trip-type distance-decay"
        " local-access model from nearest-network distance,"
        " with wider-access bound at 4.8km"
        " (not yet full 20-minute multimodal)\n"
        "- **Energy basis (primary):** total household energy ="
        " metered building energy (DESNZ) + modelled commute"
        " transport energy"
        " (Census `TS058` + `TS061`, mode-based estimate)\n"
        "- **Energy basis (secondary scenario):** commute"
        " estimate scaled to overall travel using NTS 2024"
        " distance ratio (6,082 / 1,007 = 6.04x)\n"
        "- **Basket type:** TCPA-aligned category draft using"
        " currently available cityseer layers\n"
        "\n"
        "## How To Use This Appendix\n"
        "\n"
        "This appendix is technical support for the main case"
        " note, not the primary narrative.\n"
        "\n"
        "Recommended reading order:\n"
        "\n"
        "1. Read the case note (`paper/case_v1.md`)\n"
        "2. Use this appendix to verify methods, thresholds,"
        " and exact summary values\n"
        "3. Use the CSV outputs in `stats/figures/basket_v1/`"
        " for further analysis or mapping\n"
        "\n"
        "## v1 Basket Design (Draft)\n"
        "\n"
        "### Structure\n"
        "\n"
        "- **Core categories (90% weight):** local"
        " food/services proxy, GP, pharmacy, school,"
        " green space, bus\n"
        "- **Support categories (10% weight):**"
        " rail/metro, hospital\n"
        "- **Category scores:** `min(1, value / target)`\n"
        "- **Core penalty:** raw weighted score is multiplied"
        " by `0.5 + 0.5 * core_coverage_rate`\n"
        "- **Final basket score:** `0-100`\n"
        "\n"
        "This gives a basket that is partly"
        " non-compensatory: high performance in optional"
        " or abundant\n"
        "categories cannot fully offset missing"
        " essentials.\n"
        "\n"
        "### Draft Calibration Table\n"
        "\n"
        "Thresholds are quantile-calibrated on the pilot"
        " dataset for this draft (not final normative\n"
        "thresholds for England-wide deployment).\n"
        "\n"
        f"{schema_md}\n"
        "\n"
        "## Main Metrics\n"
        "\n"
        "- **Basket attainment:** `basket_score` (0-100)\n"
        "- **Energy cost of access (primary):**"
        " `kWh_per_basket_point_hh`\n"
        "- **Energy productivity (companion):**"
        " `basket_points_per_10mwh_hh`\n"
        "- **Trip-demand coverage (new):**"
        " `basket_trip_coverage_rate_est`\n"
        "- **Energy per accessible trip (overall scenario):**"
        " `kwh_per_accessible_trip_total_est`\n"
        "\n"
        "## Headline Pilot Result\n"
        "\n"
        "The pilot reproduces the core morphology lock-in"
        " pattern under a basket formulation:\n"
        "\n"
        f"- Detached-dominated LSOAs have"
        f" **{detached_vs_flat_energy:.2f}x** the median total"
        f" household energy of flat-dominant LSOAs\n"
        f"- Flat-dominant LSOAs have"
        f" **{flat_vs_det_basket:.2f}x** the median basket"
        f" attainment of detached-dominant LSOAs\n"
        f"- Detached-dominated LSOAs face"
        f" **{detached_vs_flat_cost:.2f}x** higher energy cost"
        f" of access (`kWh / basket point`) than"
        f" flat-dominant LSOAs\n"
        f"- Equivalently, flat-dominant LSOAs deliver"
        f" **{flat_vs_det_yield:.2f}x** more basket points"
        f" per 10 MWh than detached-dominant LSOAs\n"
        "\n"
        "## Narrative Walkthrough"
        " (Figures + Interpretation)\n"
        "\n"
        "### 1. What counts as the basket?\n"
        "\n"
        "The v1 basket is a structured bundle of local-access"
        " functions using currently available\n"
        "cityseer layers (health, schools, green space,"
        " bus/rail, and an FSA-based local\n"
        "food/services proxy). Each category is scored"
        " separately before being combined, and missing\n"
        "core services reduce the final score via a"
        " core-coverage penalty.\n"
        "\n"
        f"{fig_heatmap}\n"
        "\n"
        "What this figure shows:\n"
        "\n"
        "- The morphology gradient is visible across multiple"
        " categories, not just one amenity type.\n"
        "- Compact forms (flat- and terraced-dominant LSOAs)"
        " perform strongly across more basket\n"
        "  categories.\n"
        "- Detached-dominant LSOAs score lower across most"
        " core categories, especially local\n"
        "  food/services and primary care access.\n"
        "- Rail is intentionally a low-weight support category"
        " and sparse in the pilot, so it should\n"
        "  not be overinterpreted.\n"
        "\n"
        "### 2. Why the basket framing matters\n"
        "\n"
        "The core claim of the project is not simply that some"
        " neighbourhoods use more energy. It is\n"
        "that neighbourhood morphology changes both sides of"
        " the fraction:\n"
        "\n"
        "- **Cost side:** total energy required for ordinary"
        " living (building + commute transport estimate)\n"
        "- **Return side:** local basket attainment (everyday"
        " functions reachable locally)\n"
        "\n"
        f"{fig_by_type}\n"
        "\n"
        "This is the clearest summary figure in the draft:\n"
        "\n"
        "- The energy gap alone is meaningful but limited.\n"
        "- Once local basket attainment is included, the"
        " difference becomes much larger.\n"
        "- Detached-dominant LSOAs show both **higher total"
        " energy** and **lower basket attainment**,\n"
        "  so the **energy cost of access** rises sharply.\n"
        "\n"
        "### 3. Is this only an artifact of medians?\n"
        "\n"
        "No. The KDE contour plot shows the relationship"
        " across all pilot LSOAs and preserves the\n"
        "same visual language as the existing three-surfaces"
        " figures.\n"
        "\n"
        f"{fig_scatter}\n"
        "\n"
        "What to look for:\n"
        "\n"
        "- The type-specific distributions occupy different"
        " regions of the energy/local-coverage plane.\n"
        "- Flat- and terraced-dominant LSOAs cluster toward"
        " **higher local trip coverage at lower energy**.\n"
        "- Semi- and detached-dominant LSOAs shift toward"
        " **lower local trip coverage and higher energy**.\n"
        "- The morphology pattern is distributed, not just"
        " a result of a few extreme outliers.\n"
        "\n"
        "### 4. Equity lens (pilot)\n"
        "\n"
        "The deprivation view is included as a diagnostic,"
        " not as the main framing. It helps confirm\n"
        "that the basket can support an equity analysis later,"
        " but the central story remains the\n"
        "morphology-linked energy dependency pattern.\n"
        "\n"
        f"{fig_dep}\n"
        "\n"
        "## Results by Dominant Housing Type (Median LSOA)\n"
        "\n"
        f"{type_md}\n"
        "\n"
        "## Deprivation Summary (All Types Combined)\n"
        "\n"
        f"{dep_md}\n"
        "\n"
        "## Figures Generated\n"
        "\n"
        f"{figure_lines}\n"
        "\n"
        "## Interpretation (Single-Sentence Version)\n"
        "\n"
        "Neighbourhood morphology changes the energy cost of"
        " ordinary access by shifting both\n"
        "household energy demand and the local availability"
        " of everyday services.\n"
        "\n"
        "## Interpretation (Working Narrative)\n"
        "\n"
        "This v1 basket draft makes the argument more legible"
        " than the prior z-scored access proxy:\n"
        "the index now reports a structured bundle of everyday"
        " functions, then expresses the energy\n"
        "cost of attaining that bundle.\n"
        "\n"
        "The pilot result remains the same in direction:\n"
        "\n"
        "- compact forms (especially flat-dominant and"
        " terraced-dominant areas) tend to have lower\n"
        "  energy cost and higher local basket attainment;\n"
        "- more sprawling forms (semi- and"
        " detached-dominant areas) tend to have higher"
        " energy cost\n"
        "  and lower local basket attainment.\n"
        "\n"
        "## Current Limitations (v1 Draft)\n"
        "\n"
        "1. The basket uses a **distance-decay walkability"
        " proxy** with trip-type half-distance"
        " assumptions, not a full 20-minute multimodal"
        " travel-time model.\n"
        "2. The FSA-based category is a **local food/services"
        " proxy**, not a clean grocery/essential"
        " retail layer.\n"
        "3. Category thresholds are **pilot-calibrated"
        " quantiles**, not yet normative TCPA/UK policy"
        " thresholds.\n"
        "4. Trip-demand mapping combines observed counts and"
        " explicit assumptions (notably pharmacy"
        " collection conversion).\n"
        "5. The primary denominator is **household energy**;"
        " a per-person version should be co-reported"
        " in the next iteration.\n"
        "\n"
        "## Next Upgrade Path (v2)\n"
        "\n"
        "1. Replace the FSA proxy with explicit grocery /"
        " daily retail classes.\n"
        "2. Add a true 20-minute multimodal accessibility"
        " definition (walk + public transport"
        " travel time).\n"
        "3. Set category targets from policy/literature"
        " rather than pilot quantiles.\n"
        "4. Test sensitivity to weights, thresholds, and"
        " catchment assumptions.\n"
        "5. Produce England-wide calibration and"
        " city-specific comparison profiles.\n"
    )

    DOC_PATH.write_text(text)
    return DOC_PATH


def main(cities: list[str] | None = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BASKET INDEX V1 (PILOT APPENDIX BUILD)")
    print("=" * 70)

    lsoa = prepare_lsoa(cities=cities)
    schema = calibrate_schema(lsoa)
    lsoa = compute_basket_index(lsoa, schema)

    schema_path = save_schema_table(schema)
    type_path, dep_path = save_summary_tables(lsoa, schema)
    lsoa_scores_path = save_lsoa_scores(lsoa, schema)
    fig_paths: list[Path] = []
    fig_paths.append(fig1_category_scores_heatmap(lsoa, schema))
    fig_paths.append(fig2_basket_and_cost_bars(lsoa))
    fig_paths.append(fig3_energy_vs_basket_scatter(lsoa))
    maybe_dep = fig4_deprivation_gradient(lsoa)
    if maybe_dep is not None:
        fig_paths.append(maybe_dep)

    print("\nOutputs")
    print(f"  Schema table: {schema_path}")
    print(f"  Type summary: {type_path}")
    print(f"  Deprivation summary: {dep_path}")
    print(f"  LSOA scores: {lsoa_scores_path}")
    for p in fig_paths:
        print(f"  Figure: {p}")
    print(f"  Figures/tables dir: {OUT_DIR}")


if __name__ == "__main__":
    _cities = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(cities=_cities or None)
