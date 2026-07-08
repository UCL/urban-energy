"""Tests for the meter-weighted OA energy aggregation."""

import pandas as pd
import pytest
from aggregate_energy_oa import aggregate_postcode_to_oa


def _energy() -> pd.DataFrame:
    """
    Two OAs of two postcodes each.

    E1 exercises the ordinary meter-weighted mean. E2 exercises the zero/
    suppressed-meter weight handling: P3 has zero meters of every fuel, so its
    weight is floored to 1 rather than dropped.
    """
    return pd.DataFrame(
        {
            "Postcode": ["AB1 1AA", "AB1 2AA", "AB2 1AA", "AB2 2AA"],
            "elec_num_meters": [10, 30, 0, 6],
            "elec_mean_kwh": [3000, 3500, 5000, 2000],
            "gas_num_meters": [8, 20, 0, 6],
            "gas_mean_kwh": [12000, 13000, 0, 10000],
            "total_mean_kwh": [15000, 16500, 5000, 12000],
        }
    )


def _lookup() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Postcode": ["AB1 1AA", "AB1 2AA", "AB2 1AA", "AB2 2AA"],
            "OA21CD": ["E1", "E1", "E2", "E2"],
            "LSOA21CD": ["L1", "L1", "L2", "L2"],
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


def test_zero_meter_weight_is_floored_to_one() -> None:
    out = aggregate_postcode_to_oa(_energy(), _lookup()).set_index("OA21CD")

    # E2 elec: P3 has 0 meters, so its weight floors to 1 (not dropped):
    # (1*5000 + 6*2000) / (1 + 6) = 17000 / 7
    assert out.loc["E2", "oa_elec_mean_kwh"] == pytest.approx(17000 / 7)
    # E2 combined meters: P3's 0+0 floors to 1, P4's 6+6 = 12, total 13.
    assert out.loc["E2", "oa_num_meters"] == 13


def test_low_meter_oas_are_filtered() -> None:
    energy = _energy().iloc[[2]].copy()  # single postcode, 0 meters → floored 1
    lookup = _lookup().iloc[[2]].copy()
    out = aggregate_postcode_to_oa(energy, lookup)
    # oa_num_meters would be 1, below MIN_METERS_PER_OA (5), so the OA is dropped.
    assert out.empty
