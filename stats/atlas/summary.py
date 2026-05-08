"""
Build the summary JSON consumed by the frontend.

The summary is the schema-stable contract between the Python pipeline and
the JS frontend. New blocks (units, scenarios) are populated as B1 and B4
land; for now they expose a placeholder shape so the frontend can render a
disabled UI without special-casing absence.
"""

from __future__ import annotations

import pandas as pd

from nepi import BAND_COLORS, BAND_PERCENTILES

from .demand import (
    EV_EFFICIENCY_RATIO,
    EV_ROLLOUT_80,
    HEAT_PUMP_COP,
    HP_ROLLOUT_80,
    NEUTRAL,
    DemandScenario,
)
from .schema import SCHEMA_VERSION
from .units import units_block


def _band_thresholds(df: pd.DataFrame) -> dict[str, dict[str, float | str]]:
    """kWh values at each band's percentile boundaries (national)."""
    valid = df["nepi_total_kwh"].notna() & (df["nepi_total_kwh"] > 0)
    s = df.loc[valid, "nepi_total_kwh"]
    return {
        band: {
            "lo_pct": lo,
            "hi_pct": hi,
            "lo_kwh": float(s.quantile(lo / 100)),
            "hi_kwh": float(s.quantile(hi / 100)),
            "color": BAND_COLORS[band],
        }
        for band, (lo, hi) in BAND_PERCENTILES.items()
    }


def _feature_ranges(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    National p5/p50/p95 for the 9 XGBoost features.

    Used by the in-browser planning-tool sliders to set sensible default
    ranges. Per-feature so each slider can use its own scale.
    """
    features = [
        "people_per_ha",
        "pct_detached", "pct_semi", "pct_terraced", "pct_flat",
        "local_coverage",
        "cc_bus_800_wt", "cc_rail_800_wt",
        "median_build_year",
    ]
    out: dict[str, dict[str, float]] = {}
    for col in features:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        out[col] = {
            "p5": float(s.quantile(0.05)),
            "p50": float(s.quantile(0.50)),
            "p95": float(s.quantile(0.95)),
        }
    return out


def _surface_percentiles(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    National percentile breaks per surface.

    Used by the frontend to colour-ramp non-composite surface views.
    p05/p95 anchor the ends of the ramp so outliers don't dominate;
    p25/p50/p75 give the intermediate stops.
    """
    surfaces = {
        "form": "nepi_form_kwh",
        "mobility": "nepi_mobility_kwh",
        "access": "nepi_access_kwh",
        "composite": "nepi_total_kwh",
    }
    out: dict[str, dict[str, float]] = {}
    for key, col in surfaces.items():
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        out[key] = {
            "p05": float(vals.quantile(0.05)),
            "p25": float(vals.quantile(0.25)),
            "p50": float(vals.quantile(0.50)),
            "p75": float(vals.quantile(0.75)),
            "p95": float(vals.quantile(0.95)),
        }
    return out


def _archetype_medians(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """National median for each surface, by dominant_type — peer-group reference."""
    out: dict[str, dict[str, float | int]] = {}
    for t in ["Flat", "Terraced", "Semi", "Detached"]:
        sub = df[df["dominant_type"] == t]
        if len(sub) == 0:
            continue
        out[t] = {
            "n": int(len(sub)),
            "form_kwh": float(sub["nepi_form_kwh"].median()),
            "mobility_kwh": float(sub["nepi_mobility_kwh"].median()),
            "access_kwh": float(sub["nepi_access_kwh"].median()),
            "total_kwh": float(sub["nepi_total_kwh"].median()),
            "local_coverage": float(sub["local_coverage"].median()),
        }
    return out


def _distribution(df: pd.DataFrame) -> dict[str, float | int | dict[str, int]]:
    """Summary stats over a dataframe of NEPI scores."""
    valid = df["nepi_total_kwh"].notna() & (df["nepi_total_kwh"] > 0)
    s = df.loc[valid, "nepi_total_kwh"]
    bands = df.loc[valid, "nepi_band"]
    return {
        "n_oas": int(valid.sum()),
        "median_kwh": float(s.median()) if len(s) else 0.0,
        "p25_kwh": float(s.quantile(0.25)) if len(s) else 0.0,
        "p75_kwh": float(s.quantile(0.75)) if len(s) else 0.0,
        "min_kwh": float(s.min()) if len(s) else 0.0,
        "max_kwh": float(s.max()) if len(s) else 0.0,
        "band_counts": {b: int((bands == b).sum()) for b in "ABCDEFG"},
    }


def _per_oa_form(
    national: pd.DataFrame, factors: dict[str, float], demand: DemandScenario
) -> pd.Series:
    """Form-surface contribution per OA, in the unit defined by `factors`."""
    elec = national["oa_elec_mean_kwh"].fillna(0)
    gas = national["oa_gas_mean_kwh"].fillna(0)
    hp = demand.hp_share
    new_gas = gas * (1 - hp)
    new_elec = elec + gas * hp / HEAT_PUMP_COP
    return new_elec * factors.get("elec", 0) + new_gas * factors.get("gas", 0)


def _per_oa_mobility(
    national: pd.DataFrame, factors: dict[str, float], demand: DemandScenario
) -> pd.Series:
    """Mobility-surface contribution per OA, with per-OA fleet split + EV adoption."""
    mob = national["nepi_mobility_kwh"].fillna(0)
    bev = national["bev_share"].fillna(0).clip(0, 1)
    bev_target = bev.where(bev >= demand.ev_share, demand.ev_share)
    ice = mob * (1 - bev_target)
    ev = mob * bev_target / EV_EFFICIENCY_RATIO
    return ice * factors.get("mobility", 0) + ev * factors.get("elec", 0)


def _per_oa_access(
    national: pd.DataFrame, factors: dict[str, float], demand: DemandScenario
) -> pd.Series:
    """Access-penalty contribution per OA, follows Mobility's fuel mix."""
    acc = national["nepi_access_kwh"].fillna(0)
    bev = national["bev_share"].fillna(0).clip(0, 1)
    bev_target = bev.where(bev >= demand.ev_share, demand.ev_share)
    ice = acc * (1 - bev_target)
    ev = acc * bev_target / EV_EFFICIENCY_RATIO
    return ice * factors.get("access", 0) + ev * factors.get("elec", 0)


_SURFACE_FNS = {
    "form": _per_oa_form,
    "mobility": _per_oa_mobility,
    "access": _per_oa_access,
}


def _per_oa_total_in_view(
    national: pd.DataFrame,
    factors: dict[str, float],
    demand: DemandScenario = NEUTRAL,
) -> pd.Series:
    """Sum of all three surfaces per OA in the view defined by `factors` × `demand`."""
    return (
        _per_oa_form(national, factors, demand)
        + _per_oa_mobility(national, factors, demand)
        + _per_oa_access(national, factors, demand)
    )


def _per_oa_surface(
    national: pd.DataFrame,
    factors: dict[str, float],
    demand: DemandScenario,
    surface: str,
) -> pd.Series:
    """Per-OA value for a given surface — composite sums all three."""
    if surface == "composite":
        return _per_oa_total_in_view(national, factors, demand)
    return _SURFACE_FNS[surface](national, factors, demand)


def _delta_distributions(
    national: pd.DataFrame, scenarios: dict[str, dict]
) -> dict[str, dict[str, float]]:
    """
    For each (non-present mode, unit, surface), compute the % delta distribution
    vs Present.

    Frontend uses `abs_p98` as the symmetric ±range for the diverging colour
    palette so each view normalises to its own data spread — keeping the
    colours saturated rather than pastel.

    Keyed by `"{mode}.{unit}.{surface}"` (e.g. `"y2050_hp.kgco2.composite"`).
    """
    present = scenarios["present"]
    out: dict[str, dict[str, float]] = {}
    for mode_key, scenario in scenarios.items():
        if mode_key == "present":
            continue
        for unit_key, unit_data in scenario["units"].items():
            cur_factors = unit_data["factors"]
            cur_demand = scenario["_demand"]
            base_factors = present["units"][unit_key]["factors"]
            base_demand = present["_demand"]
            for surface in ("composite", "form", "mobility", "access"):
                cur = _per_oa_surface(national, cur_factors, cur_demand, surface)
                base = _per_oa_surface(national, base_factors, base_demand, surface)
                # % delta vs present, only for OAs with positive baseline
                # (avoids div-by-zero on Access for compact OAs that have ~0 penalty)
                valid = base > 1e-6
                if not valid.any():
                    continue
                d_pct = 100 * (cur[valid] - base[valid]) / base[valid]
                abs_d = d_pct.abs()
                # Range is p90 of |delta| — ensures ~90% of OAs fall inside
                # the vivid colour band, with rare outliers clipping to the
                # endpoints. p98 alone gets dragged by tail outliers
                # (e.g. all-electric OAs whose Form CO2 collapses), making
                # the bulk of the map look pastel.
                out[f"{mode_key}.{unit_key}.{surface}"] = {
                    "p2": float(d_pct.quantile(0.02)),
                    "p10": float(d_pct.quantile(0.10)),
                    "p50": float(d_pct.quantile(0.50)),
                    "p90": float(d_pct.quantile(0.90)),
                    "p98": float(d_pct.quantile(0.98)),
                    # Symmetric envelope used by the diverging-palette legend.
                    # Floor at 5% so a kWh-grid-only view (all zeros) still
                    # has a non-degenerate legend.
                    "range_pct": max(5.0, float(abs_d.quantile(0.90))),
                }
    return out


def _band_thresholds_by_view(
    national: pd.DataFrame, scenarios: dict[str, dict]
) -> dict[str, dict[str, dict[str, float | str]]]:
    """
    For each (mode, unit) combination, compute band thresholds on per-OA totals.

    Each scenario carries its own `demand` block (heat-pump and EV adoption
    shares); the per-OA totals reflect both grid factors and demand-side
    adjustments so that hypothetical scenarios actually shift the band
    boundaries.

    Keyed by `"{mode}.{unit}"` (e.g. `"y2050_hp.kgco2"`).
    """
    out: dict[str, dict[str, dict[str, float | str]]] = {}
    for mode_key, scenario in scenarios.items():
        demand = scenario.get("_demand", NEUTRAL)
        for unit_key, unit_data in scenario["units"].items():
            factors = unit_data["factors"]
            total = _per_oa_total_in_view(national, factors, demand)
            t = total[total > 0]
            if len(t) == 0:
                continue
            thresholds: dict[str, dict[str, float | str]] = {}
            for band, (lo, hi) in BAND_PERCENTILES.items():
                thresholds[band] = {
                    "lo": float(t.quantile(lo / 100)),
                    "hi": float(t.quantile(hi / 100)),
                    "color": BAND_COLORS[band],
                }
            out[f"{mode_key}.{unit_key}"] = thresholds
    return out


def _scenarios(national: pd.DataFrame) -> dict[str, dict[str, object]]:
    """
    Scenario specifications.

    Each entry carries:
      - `units`: per-fuel and per-surface multipliers for kWh / kgCO2 / £
      - `_demand`: a DemandScenario controlling demand-side shifts
        (heat-pump rollout, EV rollout). Hidden from JSON serialisation;
        the frontend reads `demand_share` instead.

    Year-keyed scenarios apply the projected grid factors with neutral demand.
    Hypothetical scenarios additionally apply uniform adoption rates across
    all OAs.
    """
    f_med = float(national["nepi_form_kwh"].dropna().median() or 0.0)
    m_med = float(national["nepi_mobility_kwh"].dropna().median() or 0.0)
    a_med = float(national["nepi_access_kwh"].dropna().median() or 0.0)

    def scenario(
        slug: str,
        label: str,
        year: int,
        demand: DemandScenario = NEUTRAL,
        kind: str = "grid",
        default: bool = False,
    ) -> dict:
        out: dict = {
            "label": label,
            "year": year,
            "kind": kind,  # 'grid' | 'hypothetical'
            "available": True,
            "units": units_block(f_med, m_med, a_med, year=year),
            # Frontend-facing demand spec (JSON-serialisable)
            "demand": {
                "hp_share": demand.hp_share,
                "ev_share": demand.ev_share,
                "hp_cop": HEAT_PUMP_COP,
                "ev_efficiency": EV_EFFICIENCY_RATIO,
            },
            # Internal handle for server-side band threshold computation
            "_demand": demand,
        }
        if default:
            out["default"] = True
        return out

    # User-facing Mode toggle is year-only. Demand-side rollout lives on
    # interactive sliders in the frontend (HP and EV adoption shares). The
    # `_demand_envelope` entries below are NOT shown as buttons but are
    # included so the frontend can read their precomputed delta distributions
    # and use them as the colour-range upper bound when sliders are active.
    return {
        "present": scenario("present", "Present", 2025, default=True),
        "y2030":   scenario("y2030", "2030", 2030),
        "y2040":   scenario("y2040", "2040", 2040),
        "y2050":   scenario("y2050", "2050", 2050),
        "_envelope_hp": scenario(
            "_envelope_hp", "[envelope] 2050 + 100% HP", 2050,
            demand=DemandScenario(hp_share=1.0, ev_share=0.0),
            kind="envelope",
        ),
        "_envelope_ev": scenario(
            "_envelope_ev", "[envelope] 2050 + 100% EV", 2050,
            demand=DemandScenario(hp_share=0.0, ev_share=1.0),
            kind="envelope",
        ),
        "_envelope_both": scenario(
            "_envelope_both", "[envelope] 2050 + 100% HP + 100% EV", 2050,
            demand=DemandScenario(hp_share=1.0, ev_share=1.0),
            kind="envelope",
        ),
    }


def _strip_internal_keys(scenarios: dict) -> dict:
    """
    Remove internal-only entries (`_envelope_*`) and Python-only fields
    (`_demand`) before JSON serialisation. Envelope scenarios still flow
    through delta distributions but aren't presented as Mode buttons.
    """
    out: dict = {}
    for k, v in scenarios.items():
        if k.startswith("_"):
            continue
        out[k] = {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
    return out


def build_summary(
    national: pd.DataFrame,
    bua: str,
    bua_df: pd.DataFrame,
    lad_gdf=None,
) -> dict[str, object]:
    """
    Assemble the summary JSON.

    Parameters
    ----------
    national : pd.DataFrame
        Full national NEPI dataframe (used for band thresholds, peer medians,
        national distribution).
    bua : str
        BUA name, included for frontend display.
    bua_df : pd.DataFrame
        Subset for the selected BUA (used for BUA-level distribution).
    """
    scenarios = _scenarios(national)
    band_thresholds_by_view = _band_thresholds_by_view(national, scenarios)
    delta_distributions = _delta_distributions(national, scenarios)

    out = {
        "schema_version": SCHEMA_VERSION,
        "bua": bua,
        "national": _distribution(national),
        "bua_distribution": _distribution(bua_df),
        # OA-level (the default)
        "band_thresholds": _band_thresholds(national),
        "band_thresholds_by_view": band_thresholds_by_view,
        "delta_distributions": delta_distributions,
        "surface_percentiles": _surface_percentiles(national),
        "feature_ranges": _feature_ranges(national),
        "archetype_medians": _archetype_medians(national),
        # Top-level `units` is the present-year units block — kept for
        # frontend backward-compat. Year-specific units live under
        # `scenarios[year].units`.
        "units": scenarios["present"]["units"],
        "scenarios": _strip_internal_keys(scenarios),
    }

    # LAD-level stats — re-banded on the LAD distribution so A–G remains
    # meaningful at this aggregation level (otherwise everything clusters
    # in the middle bands since OA totals average out at LAD scale).
    if lad_gdf is not None:
        out["lad_level"] = {
            "band_thresholds": _band_thresholds(lad_gdf),
            "band_thresholds_by_view": _band_thresholds_by_view(lad_gdf, scenarios),
            "delta_distributions": _delta_distributions(lad_gdf, scenarios),
            "surface_percentiles": _surface_percentiles(lad_gdf),
            "distribution": _distribution(lad_gdf),
        }

    return out
