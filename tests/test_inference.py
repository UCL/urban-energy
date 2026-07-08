"""Tests for the uncertainty utilities in stats/inference.py (data-free)."""

import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from inference import (
    NAN_CI,
    cluster_bootstrap,
    cluster_bootstrap_multi,
    fmt_ci,
    log_contrast_ci,
)


def _fit(seed: int = 0):
    """A small OLS with a known coefficient gap b_x1 - b_x2 = 1.0."""
    rng = np.random.default_rng(seed)
    n = 4000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 2.0 * x1 + 1.0 * x2 + rng.normal(scale=0.5, size=n)
    x = pd.DataFrame({"x1": x1, "x2": x2})
    return sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC1")


def test_log_contrast_point_and_interval() -> None:
    m = _fit()
    ratio, lo, hi, se = log_contrast_ci(m, "x1", "x2")
    assert math.isclose(ratio, math.exp(1.0), rel_tol=0.05)
    assert lo < ratio < hi
    assert se > 0


def test_log_contrast_scale_factor() -> None:
    m = _fit()
    # Scaling the contrast by 100 exponentiates 100×(b_x1 - b_x2).
    r1 = log_contrast_ci(m, "x1", "x2", scale=1.0)[0]
    r100 = log_contrast_ci(m, "x1", "x2", scale=100.0)[0]
    assert math.isclose(math.log(r100), 100.0 * math.log(r1), rel_tol=1e-6)


def test_log_contrast_missing_coeff_returns_nan() -> None:
    m = _fit()
    assert all(math.isnan(v) for v in log_contrast_ci(m, "x1", "absent"))


def test_fmt_ci() -> None:
    assert fmt_ci((1.6, 1.5, 1.7, 0.02)) == "1.60× [1.50, 1.70]"
    assert fmt_ci(NAN_CI) == "n/a"


def test_cluster_bootstrap_brackets_point() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "g": np.repeat(np.arange(50), 20),
            "v": rng.normal(loc=3.0, size=1000),
        }
    )
    point, lo, hi = cluster_bootstrap(
        frame, lambda f: float(f["v"].mean()), group_col="g", reps=200
    )
    assert lo < point < hi
    assert math.isclose(point, frame["v"].mean(), rel_tol=1e-9)


def test_cluster_bootstrap_is_deterministic() -> None:
    frame = pd.DataFrame({"g": np.repeat(np.arange(30), 10), "v": np.arange(300.0)})
    a = cluster_bootstrap(
        frame, lambda f: float(f["v"].mean()), group_col="g", reps=100
    )
    b = cluster_bootstrap(
        frame, lambda f: float(f["v"].mean()), group_col="g", reps=100
    )
    assert a == b


def test_cluster_bootstrap_multi_brackets_and_keys() -> None:
    rng = np.random.default_rng(2)
    frame = pd.DataFrame(
        {"g": np.repeat(np.arange(50), 20), "v": rng.normal(loc=3.0, size=1000)}
    )

    def stat(f: pd.DataFrame) -> dict[str, float]:
        m = float(f["v"].mean())
        return {"mean": m, "double": 2.0 * m}

    out = cluster_bootstrap_multi(frame, stat, group_col="g", reps=200)
    assert set(out) == {"mean", "double"}
    for point, lo, hi in out.values():
        assert lo < point < hi
    # The two statistics come from the SAME resample each rep, so the point
    # estimates keep their exact algebraic relationship.
    assert math.isclose(out["double"][0], 2.0 * out["mean"][0], rel_tol=1e-9)


def test_cluster_bootstrap_multi_is_deterministic() -> None:
    frame = pd.DataFrame({"g": np.repeat(np.arange(30), 10), "v": np.arange(300.0)})

    def stat(f: pd.DataFrame) -> dict[str, float]:
        return {"m": float(f["v"].mean())}

    a = cluster_bootstrap_multi(frame, stat, group_col="g", reps=100)
    b = cluster_bootstrap_multi(frame, stat, group_col="g", reps=100)
    assert a == b
