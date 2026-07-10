"""Recovery tests for the compositional (no-intercept) model — the headline
estimator. Synthetic frames with known pure-type levels: the fitted
Detached:Flat contrast must recover the truth, must be invariant to uncentred
confounds, and the Poisson variant must recover a known count ratio.
"""

import numpy as np
import pandas as pd
from access_profile import _comp_poisson
from form_size_decomposition import _SHARE_FRACS, _comp_ols
from inference import log_contrast_ci

_LEVELS = {
    "s_flat": 10_000.0,
    "s_terraced": 13_000.0,
    "s_semi": 14_000.0,
    "s_detached": 21_000.0,
    "s_other": 15_000.0,
}
_TRUE_RATIO = _LEVELS["s_detached"] / _LEVELS["s_flat"]  # 2.1


def _synthetic_frame(n: int = 4000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shares = rng.dirichlet(np.ones(len(_SHARE_FRACS)), size=n)
    df = pd.DataFrame(shares, columns=_SHARE_FRACS)
    logy = shares @ np.log([_LEVELS[c] for c in _SHARE_FRACS])
    df["confound"] = rng.normal(5.0, 1.0, n)  # deliberately uncentred
    df["log_energy"] = logy + 0.30 * df["confound"] + rng.normal(0, 0.05, n)
    df["total_hh"] = rng.integers(80, 160, n).astype(float)
    return df


def test_comp_ols_recovers_pure_type_ratio() -> None:
    df = _synthetic_frame()
    m = _comp_ols(df, "log_energy", _SHARE_FRACS + ["confound"], "total_hh")
    assert m is not None
    ratio = float(np.exp(m.params["s_detached"] - m.params["s_flat"]))
    assert abs(ratio - _TRUE_RATIO) < 0.06


def test_comp_ols_contrast_invariant_to_uncentred_confound() -> None:
    """Shifting a confound's location must not move the share contrast."""
    df = _synthetic_frame()
    m1 = _comp_ols(df, "log_energy", _SHARE_FRACS + ["confound"], "total_hh")
    shifted = df.copy()
    shifted["confound"] = shifted["confound"] + 100.0
    m2 = _comp_ols(shifted, "log_energy", _SHARE_FRACS + ["confound"], "total_hh")
    assert m1 is not None and m2 is not None
    c1 = float(m1.params["s_detached"] - m1.params["s_flat"])
    c2 = float(m2.params["s_detached"] - m2.params["s_flat"])
    assert abs(c1 - c2) < 1e-8


def test_comp_ols_ci_covers_truth() -> None:
    df = _synthetic_frame()
    m = _comp_ols(df, "log_energy", _SHARE_FRACS + ["confound"], "total_hh")
    assert m is not None
    point, lo, hi, se = log_contrast_ci(m, "s_detached", "s_flat")
    assert lo < _TRUE_RATIO < hi
    assert se > 0


def test_comp_poisson_recovers_count_ratio() -> None:
    rng = np.random.default_rng(11)
    n = 4000
    shares = rng.dirichlet(np.ones(len(_SHARE_FRACS)), size=n)
    df = pd.DataFrame(shares, columns=_SHARE_FRACS)
    # Pure-type expected counts: flat 200, detached 8 → flat:det 25×.
    levels = {
        "s_flat": 200.0,
        "s_terraced": 90.0,
        "s_semi": 40.0,
        "s_detached": 8.0,
        "s_other": 60.0,
    }
    mu = np.exp(shares @ np.log([levels[c] for c in _SHARE_FRACS]))
    df["count"] = rng.poisson(mu).astype(float)
    df["total_hh"] = rng.integers(80, 160, n).astype(float)
    m = _comp_poisson(df, "count", _SHARE_FRACS, "total_hh")
    assert m is not None
    ratio = float(np.exp(m.params["s_flat"] - m.params["s_detached"]))
    true_ratio = levels["s_flat"] / levels["s_detached"]
    assert abs(ratio - true_ratio) / true_ratio < 0.15
