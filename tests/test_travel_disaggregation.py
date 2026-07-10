"""Tests for the NTS-constrained travel disaggregation (synthetic classes).

The class join is monkeypatched so no data files are needed. What is tested is
the load-bearing property: the population-weighted class mean of allocated
per-person mileage reproduces the NTS class figure EXACTLY, at every allocator
setting, and the ownership elasticity behaves as documented (alpha = 0 removes
the ownership signal; alpha = 1 allocates proportionally).
"""

import numpy as np
import pandas as pd
import pytest
import travel_energy

_ANCHOR = {"Urban": 2_500.0, "Rural": 5_200.0}


def _fake_join(lsoa: pd.DataFrame) -> pd.DataFrame:
    out = lsoa.copy()
    out["RUC21NM"] = out["ruc"]
    out["ruc_class_miles_pp"] = out["ruc"].map(_ANCHOR)
    return out


@pytest.fixture()
def frame(monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    monkeypatch.setattr(travel_energy, "_join_ruc_mileage", _fake_join)
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame(
        {
            "ruc": np.where(rng.random(n) < 0.5, "Urban", "Rural"),
            "cars_per_hh": rng.uniform(0.3, 2.4, n),
            "avg_hh_size": rng.uniform(1.8, 2.9, n),
            "total_people": rng.integers(200, 400, n).astype(float),
        }
    )
    # Identical commute distribution everywhere → commute factor exactly 1,
    # so the allocator is the ownership term alone.
    for col in travel_energy.TS058_BAND_MIDPOINTS_KM:
        df[col] = 10.0
    return df


def _class_means(d: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for cls, g in d.groupby("RUC21NM"):
        out[str(cls)] = float(
            np.average(g["car_miles_per_person"], weights=g["total_people"])
        )
    return out


def test_class_marginals_preserved_exactly(frame: pd.DataFrame) -> None:
    d = travel_energy.compute_travel_energy(frame)
    for cls, got in _class_means(d).items():
        assert got == pytest.approx(_ANCHOR[cls], rel=1e-12)


@pytest.mark.parametrize("alpha", [0.0, 0.6, 1.0])
def test_class_marginals_preserved_at_every_alpha(
    frame: pd.DataFrame, alpha: float
) -> None:
    d = travel_energy.compute_travel_energy(frame, ownership_elasticity=alpha)
    for cls, got in _class_means(d).items():
        assert got == pytest.approx(_ANCHOR[cls], rel=1e-12)


def test_alpha_zero_removes_ownership_signal(frame: pd.DataFrame) -> None:
    """With equal commutes and alpha 0 every OA in a class gets the class figure."""
    d = travel_energy.compute_travel_energy(frame, ownership_elasticity=0.0)
    for cls, g in d.groupby("RUC21NM"):
        assert np.allclose(g["car_miles_per_person"], _ANCHOR[str(cls)])


def test_alpha_one_is_proportional_to_ownership(frame: pd.DataFrame) -> None:
    """At alpha 1 (equal commutes) miles per person scale with cars per person."""
    d = travel_energy.compute_travel_energy(frame, ownership_elasticity=1.0)
    for _cls, g in d.groupby("RUC21NM"):
        cars_pp = g["cars_per_hh"] / g["avg_hh_size"]
        implied = g["car_miles_per_person"] / cars_pp
        # miles ∝ cars_pp ⇒ miles / cars_pp is one constant per class
        assert float(implied.std() / implied.mean()) < 1e-9


def test_energy_multiplies_back_household_size(frame: pd.DataFrame) -> None:
    d = travel_energy.compute_travel_energy(frame)
    expected = d["car_miles_per_person"] * d["avg_hh_size"] * d["travel_kwh_per_mile"]
    assert np.allclose(d["travel_kwh_per_hh_car"], expected, equal_nan=True)
