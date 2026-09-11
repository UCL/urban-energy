"""Tests for the meter-weighted OA energy aggregation."""

import pandas as pd
import pytest
from aggregate_energy_oa import aggregate_postcode_to_oa


def _energy() -> pd.DataFrame:
    """
    Two OAs of two postcodes each.

    E1 exercises the ordinary meter-weighted mean. E2 exercises the zero-meter
    and missing-fuel handling: P3 has zero meters of every fuel and P4 is an
    electricity-only postcode (absent from the gas file, so its gas columns are
    NaN after the outer join). Neither may dilute the fuel means it lacks.
    """
    return pd.DataFrame(
        {
            "Postcode": ["AB1 1AA", "AB1 2AA", "AB2 1AA", "AB2 2AA", "AB2 3AA"],
            "elec_num_meters": [10, 30, 0, 6, 12],
            "elec_mean_kwh": [3000, 3500, 5000, 2000, 4000],
            "gas_num_meters": [8, 20, 0, float("nan"), 4],
            "gas_mean_kwh": [12000, 13000, 0, float("nan"), 9000],
            "total_mean_kwh": [15000, 16500, 5000, 2000, 13000],
        }
    )


def _lookup() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Postcode": ["AB1 1AA", "AB1 2AA", "AB2 1AA", "AB2 2AA", "AB2 3AA"],
            "OA21CD": ["E1", "E1", "E2", "E2", "E2"],
            "LSOA21CD": ["L1", "L1", "L2", "L2", "L2"],
        }
    )


def test_meter_weighted_means() -> None:
    out = aggregate_postcode_to_oa(_energy(), _lookup()).set_index("OA21CD")

    # elec: (10*3000 + 30*3500) / 40 = 3375
    assert out.loc["E1", "oa_elec_mean_kwh"] == pytest.approx(3375.0)
    # gas: (8*12000 + 20*13000) / 28
    assert out.loc["E1", "oa_gas_mean_kwh"] == pytest.approx(356000 / 28)
    # total weighted by combined meters (_meters): (18*15000 + 50*16500) / 68
    assert out.loc["E1", "oa_total_mean_kwh"] == pytest.approx(1095000 / 68)
    # combined meters: (10+8) + (30+20) = 68
    assert out.loc["E1", "oa_num_meters"] == 68
    assert out.loc["E1", "oa_num_postcodes"] == 2
    # per-fuel meter totals
    assert out.loc["E1", "oa_elec_num_meters"] == 40
    assert out.loc["E1", "oa_gas_num_meters"] == 28


def test_zero_meter_and_missing_fuel_rows_carry_no_weight() -> None:
    out = aggregate_postcode_to_oa(_energy(), _lookup()).set_index("OA21CD")

    # E2 elec: P3 has 0 meters, so it is dropped from the mean rather than
    # entering with a floored weight: (6*2000 + 12*4000) / 18
    assert out.loc["E2", "oa_elec_mean_kwh"] == pytest.approx(60000 / 18)
    # E2 gas: P3 has 0 meters and P4 has no gas at all (NaN), so only P5
    # contributes; the mean must not be diluted by either row.
    assert out.loc["E2", "oa_gas_mean_kwh"] == pytest.approx(9000.0)
    assert out.loc["E2", "oa_gas_num_meters"] == 4
    # E2 combined meters: P3 0, P4 6+0, P5 12+4 = 22.
    assert out.loc["E2", "oa_num_meters"] == 22
    # The stored mean x meter count reproduces the postcode gas total exactly.
    assert out.loc["E2", "oa_gas_mean_kwh"] * out.loc["E2", "oa_gas_num_meters"] == (
        pytest.approx(4 * 9000)
    )


def test_low_meter_oas_are_filtered() -> None:
    energy = _energy().iloc[[2]].copy()  # single postcode, 0 meters
    lookup = _lookup().iloc[[2]].copy()
    out = aggregate_postcode_to_oa(energy, lookup)
    # oa_num_meters is 0, below MIN_METERS_PER_OA (5), so the OA is dropped.
    assert out.empty
