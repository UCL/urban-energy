"""NEPI score: banding, threshold freezing and the potential transform."""

import json

import nepi_score
import numpy as np
import pandas as pd
import pytest
from nepi_score import (
    LETTERS,
    N_BANDS,
    assign_letters,
    potential_heat,
    weighted_thresholds,
)


class TestWeightedThresholds:
    def test_equal_weights_equal_bands(self):
        values = np.arange(1.0, 701.0)
        weights = np.ones_like(values)
        cuts = weighted_thresholds(values, weights)
        assert len(cuts) == N_BANDS - 1
        assert cuts == sorted(cuts)
        # Each band should hold ~1/7 of the observations.
        bands = np.searchsorted(cuts, values, side="right")
        counts = np.bincount(bands, minlength=N_BANDS)
        assert counts.max() - counts.min() <= 2

    def test_weights_shift_cuts(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        heavy_low = weighted_thresholds(values, np.array([100, 1, 1, 1, 1, 1, 1]))
        heavy_high = weighted_thresholds(values, np.array([1, 1, 1, 1, 1, 1, 100]))
        assert heavy_low[0] < heavy_high[0]

    def test_ignores_non_finite(self):
        values = np.array([1.0, 2.0, np.nan, 3.0, np.inf])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        cuts = weighted_thresholds(values, weights)
        assert all(np.isfinite(cuts))
        assert cuts[0] >= 1.0 and cuts[-1] <= 3.0

    def test_log_invariance(self):
        """Septiles of log(x) are the log of septiles of x (score_spec §Bands)."""
        rng = np.random.default_rng(7)
        values = np.exp(rng.normal(0, 1, 5000))
        weights = rng.integers(50, 200, 5000).astype(float)
        raw = weighted_thresholds(values, weights)
        logged = weighted_thresholds(np.log(values), weights)
        assert np.allclose(np.log(raw), logged, atol=1e-9)


class TestAssignLetters:
    CUTS = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

    def test_best_high(self):
        letters = assign_letters([65.0, 35.0, 5.0], self.CUTS, best="high")
        assert list(letters) == ["A", "D", "G"]

    def test_best_low(self):
        letters = assign_letters([5.0, 35.0, 65.0], self.CUTS, best="low")
        assert list(letters) == ["A", "D", "G"]

    def test_full_ladder(self):
        values = [5, 15, 25, 35, 45, 55, 65]
        assert list(assign_letters(values, self.CUTS, best="low")) == list(LETTERS)

    def test_nan_is_na(self):
        letters = assign_letters([np.nan, 65.0], self.CUTS, best="high")
        assert pd.isna(letters.iloc[0]) and letters.iloc[1] == "A"

    def test_bad_best_raises(self):
        with pytest.raises(ValueError):
            assign_letters([1.0], self.CUTS, best="up")


class TestPotentialTransform:
    def test_matches_scenario_energy(self):
        """The score's potential must equal scenarios.py at full deployment."""
        from scenarios import scenario_energy

        df = pd.DataFrame(
            {
                "epc_potential_kwh_m2": [120.0, 90.0, 200.0],
                "epc_current_kwh_m2": [240.0, 300.0, 220.0],
            }
        )
        gas = pd.Series([12000.0, 9000.0, 15000.0])
        elec = pd.Series([3000.0, 2800.0, 3500.0])
        travel = pd.Series([5000.0, 4000.0, 8000.0])
        heat_ref, _ = scenario_energy(df, gas, elec, travel, "fabric+hp", 1.0, 0.0)
        from scenarios import _fabric_factor

        heat = potential_heat(gas, elec, _fabric_factor(df))
        assert np.allclose(heat, heat_ref)

    def test_potential_never_exceeds_current(self):
        gas = pd.Series([12000.0, 0.0])
        elec = pd.Series([3000.0, 4500.0])
        factor = pd.Series([0.6, 1.0])
        heat = potential_heat(gas, elec, factor)
        assert (heat <= gas + elec + 1e-9).all()


class TestBandFreezing:
    def test_roundtrip(self, tmp_path, monkeypatch):
        """Bands are written once, then loaded verbatim on the next run."""
        monkeypatch.setattr(nepi_score, "BANDS_PATH", tmp_path / "bands.json")
        rng = np.random.default_rng(11)
        frame = pd.DataFrame(
            {
                "rate": np.exp(rng.normal(0, 1, 800)),
                "total_kwh_hh": rng.uniform(8000, 40000, 800),
                "access_walk": rng.integers(0, 500, 800).astype(float),
                "total_hh": rng.integers(80, 180, 800).astype(float),
            }
        )
        first = nepi_score.load_or_freeze_bands(frame)
        assert nepi_score.BANDS_PATH.exists()
        # A different frame must NOT move the frozen thresholds.
        second = nepi_score.load_or_freeze_bands(frame.iloc[:100])
        assert second == first
        on_disk = json.loads(nepi_score.BANDS_PATH.read_text())
        assert on_disk["thresholds"] == first["thresholds"]
        assert len(on_disk["thresholds"]["rate"]) == N_BANDS - 1
