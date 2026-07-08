"""Tests for the EPC construction-age-band midpoint parser."""

import math

from aggregate_epc_oa import _band_to_year


def test_closed_band_returns_midpoint() -> None:
    assert _band_to_year("1900-1929") == 1914.5
    assert _band_to_year("1930-1949") == 1939.5
    assert _band_to_year("1983-1990") == 1986.5


def test_prefixed_closed_band() -> None:
    # Real EPC values carry an "England and Wales: " prefix.
    assert _band_to_year("England and Wales: 1900-1929") == 1914.5


def test_open_upper_band_uses_boundary_year() -> None:
    assert _band_to_year("2007 onwards") == 2007.0
    assert _band_to_year("England and Wales: 2007 onwards") == 2007.0


def test_before_band_uses_boundary_year() -> None:
    # "before 1900" is open on the lower side; it returns its boundary year 1900
    # (a conservative proxy) so the old-housing signal is kept in the confound
    # rather than dropped to NaN.
    assert _band_to_year("before 1900") == 1900.0
    assert _band_to_year("England and Wales: before 1900") == 1900.0
    assert _band_to_year("Before 1900") == 1900.0


def test_unparseable_bands_return_nan() -> None:
    assert math.isnan(_band_to_year("NO DATA!"))
    assert math.isnan(_band_to_year("INVALID"))
    assert math.isnan(_band_to_year(""))
    assert math.isnan(_band_to_year(None))
    assert math.isnan(_band_to_year(float("nan")))
