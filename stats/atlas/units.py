"""
Unit conversion factors for the Atlas — kWh → kgCO2e and kWh → £.

Per-NEPI-surface blended factors. Form blends gas + electricity by typical
UK domestic share; Mobility blends petrol + diesel by typical road-transport
share; Access uses the same factor as Mobility (the Access surface is itself
a transport-energy penalty).

Year-aware: under scenario modes (2030/2040/2050) the per-fuel inputs come
from `projections.py` and the surface blends are recomputed for that year.
"""

from __future__ import annotations

from .projections import PROJECTION_YEARS, fuel_factors

# Blend shares — weights used to combine fuel-specific factors into a single
# multiplier per NEPI surface. National averages, year-invariant.
DOMESTIC_GAS_SHARE = 0.70
DOMESTIC_ELEC_SHARE = 0.30
ROAD_PETROL_SHARE = 0.50
ROAD_DIESEL_SHARE = 0.50


def _blend_form(fuels: dict[str, float]) -> float:
    return DOMESTIC_GAS_SHARE * fuels["gas"] + DOMESTIC_ELEC_SHARE * fuels["elec"]


def _blend_mobility(fuels: dict[str, float]) -> float:
    return ROAD_PETROL_SHARE * fuels["petrol"] + ROAD_DIESEL_SHARE * fuels["diesel"]


def _surface_factors(unit: str, year: int) -> dict[str, float]:
    """
    Per-surface (and per-fuel) multipliers for `unit` at `year`.

    The frontend uses:
      - `elec`, `gas` to compute Form **per-OA** from oa_elec_mean_kwh /
        oa_gas_mean_kwh (so OAs with different fuel mixes evolve differently
        under scenario decarbonisation).
      - `form` is the national 70/30 blend, kept for fallback / summary stats.
      - `mobility`, `access` are uniform blends (we don't have per-OA petrol/
        diesel splits — known limitation, called out in About).
    """
    if unit == "kwh":
        return {
            "elec": 1.0, "gas": 1.0, "petrol": 1.0, "diesel": 1.0,
            "form": 1.0, "mobility": 1.0, "access": 1.0,
        }
    fuels = fuel_factors(year, "kgco2" if unit == "kgco2" else "gbp")
    form_blend = _blend_form(fuels)
    mobility_blend = _blend_mobility(fuels)
    return {
        "elec": fuels["elec"],
        "gas": fuels["gas"],
        "petrol": fuels["petrol"],
        "diesel": fuels["diesel"],
        "form": form_blend,
        "mobility": mobility_blend,
        "access": mobility_blend,
    }


def composite_factor(
    unit: str,
    year: int,
    form_kwh: float,
    mobility_kwh: float,
    access_kwh: float,
) -> float:
    """Composite multiplier weighted by a national-share kWh decomposition."""
    factors = _surface_factors(unit, year)
    total = form_kwh + mobility_kwh + access_kwh
    if total <= 0:
        return 1.0 if unit == "kwh" else 0.0
    return (
        form_kwh * factors["form"]
        + mobility_kwh * factors["mobility"]
        + access_kwh * factors["access"]
    ) / total


def units_block(
    form_median: float,
    mobility_median: float,
    access_median: float,
    year: int = 2025,
) -> dict[str, dict]:
    """
    Build a `units` block for a single year.

    The frontend uses this nested under `summary.scenarios[year].units` so
    each scenario year carries its own conversion table.
    """
    def entry(unit: str, label: str, short: str, source: str | None = None) -> dict:
        per_surf = _surface_factors(unit, year)
        comp = composite_factor(
            unit, year, form_median, mobility_median, access_median
        )
        out: dict = {
            "label": label,
            "short": short,
            "available": True,
            "factors": {**per_surf, "composite": comp},
        }
        if source:
            out["source"] = source
        return out

    return {
        "kwh": {**entry("kwh", "kWh / household / year", "kWh"), "default": True},
        "kgco2": entry(
            "kgco2",
            "kg CO₂e / household / year",
            "kgCO₂",
            "DUKES / NESO FES Holistic Transition; Form is national gas/elec blend (70/30)",
        ),
        "gbp": entry(
            "gbp",
            "£ / household / year",
            "£",
            "Ofgem cap (2025) + DESNZ Fossil Fuel Price Assumptions central case",
        ),
    }


__all__ = ["PROJECTION_YEARS", "composite_factor", "units_block"]
