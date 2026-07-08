"""Tests for postcode normalisation and its per-call-site space handling."""

import pandas as pd
from aggregate_energy_oa import normalise_postcode as energy_normalise

from urban_energy.text import normalise_postcode


def test_shared_keep_space_preserves_internal_space() -> None:
    s = pd.Series([" ab10 1au ", "SW1A 1AA", "m1 1ae"])
    out = normalise_postcode(s, keep_space=True)
    assert list(out) == ["AB10 1AU", "SW1A 1AA", "M1 1AE"]


def test_shared_no_keep_space_strips_all_spaces() -> None:
    s = pd.Series([" ab10 1au ", "SW1A 1AA", "m1 1ae"])
    out = normalise_postcode(s, keep_space=False)
    assert list(out) == ["AB101AU", "SW1A1AA", "M11AE"]


def test_energy_wrapper_keeps_the_space() -> None:
    # The metered-energy path joins to the spaced Code-Point lookup, so it must
    # retain the internal space.
    s = pd.Series(["ab10 1au", " SW1A 1AA "])
    assert list(energy_normalise(s)) == ["AB10 1AU", "SW1A 1AA"]


def test_nhs_style_strips_the_space() -> None:
    # The NHS path routes through the shared normaliser with keep_space=False,
    # producing a spaceless key on both sides of its geocode join.
    s = pd.Series(["ab10 1au", " SW1A 1AA "])
    assert list(normalise_postcode(s, keep_space=False)) == ["AB101AU", "SW1A1AA"]


def test_spaced_and_spaceless_keys_do_not_match() -> None:
    # Guards the invariant that the two conventions must never be joined: the
    # same postcode normalises to different keys under the two flags.
    s = pd.Series(["AB10 1AU"])
    spaced = normalise_postcode(s, keep_space=True).iloc[0]
    spaceless = normalise_postcode(s, keep_space=False).iloc[0]
    assert spaced != spaceless
