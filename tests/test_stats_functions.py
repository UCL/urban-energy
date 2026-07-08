"""Tests for pure stats-layer functions (data-free, synthetic frames)."""

import math

import numpy as np
import pandas as pd
from form_size_decomposition import _SHARE_FRACS, _compositional_frame
from lock_in import _fabric_factor
from travel_energy import (
    KWH_PER_MILE_EV,
    KWH_PER_MILE_ICE,
    TS058_BAND_MIDPOINTS_KM,
    fleet_intensity_kwh_per_mile,
    mean_commute_km,
)


def test_compositional_frame_shares_sum_to_one() -> None:
    df = pd.DataFrame(
        {
            "pct_flat": [100.0, 0.0, 25.0, 0.0],
            "pct_terraced": [0.0, 50.0, 25.0, 0.0],
            "pct_semi": [0.0, 50.0, 25.0, 0.0],
            "pct_detached": [0.0, 0.0, 25.0, 40.0],  # last row leaves 60% "other"
        }
    )
    out = _compositional_frame(df)
    rowsums = out[_SHARE_FRACS].sum(axis=1)
    assert np.allclose(rowsums, 1.0)
    # The residual "other" closes the composition on the under-100 row.
    assert math.isclose(out.loc[3, "s_other"], 0.60, rel_tol=1e-9)
    assert math.isclose(out.loc[0, "s_flat"], 1.0, rel_tol=1e-9)


def test_fabric_factor_clipped_to_unit_interval() -> None:
    df = pd.DataFrame(
        {
            # potential/current: 0.5 (typical), 2.0 (would raise heat → clip to 1),
            # 0.01 (implausibly low → clip to 0.1).
            "epc_potential_kwh_m2": [100.0, 200.0, 1.0],
            "epc_current_kwh_m2": [200.0, 100.0, 100.0],
        }
    )
    f = _fabric_factor(df)
    assert math.isclose(f.iloc[0], 0.5, rel_tol=1e-9)
    assert f.iloc[1] == 1.0  # improvement cannot exceed current
    assert f.iloc[2] == 0.1  # floored


def test_fleet_intensity_blends_by_bev_share() -> None:
    df = pd.DataFrame({"bev_share": [0.0, 1.0, 0.5]})
    intensity = fleet_intensity_kwh_per_mile(df)
    assert math.isclose(intensity.iloc[0], KWH_PER_MILE_ICE, rel_tol=1e-9)
    assert math.isclose(intensity.iloc[1], KWH_PER_MILE_EV, rel_tol=1e-9)
    assert math.isclose(
        intensity.iloc[2], 0.5 * (KWH_PER_MILE_ICE + KWH_PER_MILE_EV), rel_tol=1e-9
    )


def test_fleet_intensity_defaults_to_ice_without_bev() -> None:
    df = pd.DataFrame({"x": [1, 2]})
    intensity = fleet_intensity_kwh_per_mile(df)
    assert (intensity == KWH_PER_MILE_ICE).all()


def test_mean_commute_km_is_band_weighted() -> None:
    bands = list(TS058_BAND_MIDPOINTS_KM)
    # All commuters in the shortest band → mean equals that band midpoint.
    row = {b: (100.0 if b == bands[0] else 0.0) for b in bands}
    df = pd.DataFrame([row])
    assert math.isclose(
        mean_commute_km(df).iloc[0], TS058_BAND_MIDPOINTS_KM[bands[0]], rel_tol=1e-9
    )
