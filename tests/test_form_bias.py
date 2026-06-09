"""Tests for the Form-surface under-recording flags (methodology TODO #6)."""

import pandas as pd

from urban_energy.form_bias import (
    TS044_FLAT,
    TS044_TOTAL,
    compute_form_bias_flags,
)


def _frame() -> pd.DataFrame:
    """Three OAs: normal, off-gas, and majority-flat bulk-gas."""
    return pd.DataFrame(
        {
            "OA21CD": ["E1", "E2", "E3"],
            TS044_TOTAL: [100, 100, 100],
            TS044_FLAT: [10, 10, 80],
            "oa_gas_num_meters": [95, 20, 55],
        }
    )


def test_derived_ratios() -> None:
    out = compute_form_bias_flags(_frame())
    assert list(out["form_flat_share"].round(2)) == [0.10, 0.10, 0.80]
    assert list(out["form_gas_meter_coverage"].round(2)) == [0.95, 0.20, 0.55]


def test_flag_logic() -> None:
    out = compute_form_bias_flags(_frame())
    # Normal OA: high coverage, few flats — nothing flagged.
    assert bool(out.loc[0, "form_offgas_flag"]) is False
    assert bool(out.loc[0, "form_bulkgas_flag"]) is False
    assert bool(out.loc[0, "form_underrecorded_flag"]) is False
    # Off-gas OA: coverage 0.20 < 0.50 — off-gas, hence under-recorded.
    assert bool(out.loc[1, "form_offgas_flag"]) is True
    assert bool(out.loc[1, "form_bulkgas_flag"]) is False
    assert bool(out.loc[1, "form_underrecorded_flag"]) is True
    # Majority-flat OA, coverage 0.55: not off-gas (>=0.50) but bulk-gas.
    assert bool(out.loc[2, "form_offgas_flag"]) is False
    assert bool(out.loc[2, "form_bulkgas_flag"]) is True
    assert bool(out.loc[2, "form_underrecorded_flag"]) is True


def test_missing_gas_column_yields_na_not_error() -> None:
    df = _frame().drop(columns=["oa_gas_num_meters"])
    out = compute_form_bias_flags(df)
    # Coverage undeterminable → coverage NaN and all gas-derived flags NA.
    assert out["form_gas_meter_coverage"].isna().all()
    assert out["form_offgas_flag"].isna().all()
    assert out["form_underrecorded_flag"].isna().all()
    # Flat share still derives from Census TS044.
    assert list(out["form_flat_share"].round(2)) == [0.10, 0.10, 0.80]


def test_input_not_mutated() -> None:
    df = _frame()
    before = list(df.columns)
    compute_form_bias_flags(df)
    assert list(df.columns) == before
