"""
Uncertainty utilities for the two-axis analysis.

The headline results are ratios of the form ``exp(b_detached - b_flat)`` read off
one or more regressions. This module attaches confidence intervals to them two
ways:

* :func:`log_contrast_ci` — the delta-method CI for a single-model log contrast
  ``exp(scale * (b_a - b_b))``, taken straight from the (robust or clustered)
  coefficient covariance. Exact for one fitted model, effectively free.
* :func:`cluster_bootstrap` — a cluster-resampled bootstrap for quantities that
  combine several model fits on the same rows (the access-per-kWh rate, the
  surviving-gap share, the mediated fraction), where no closed-form covariance is
  available. Clusters are resampled with replacement so the interval reflects
  spatial dependence between neighbouring areas, not 178k independent draws.

All intervals are reported at the 95% level unless stated. Seeds are fixed so a
rerun reproduces the interval.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

#: Local-authority district is the clustering unit for spatial dependence: ~309
#: English LADs, each holding many contiguous OAs, so the interval is not built on
#: the false premise of 178k independent areas.
CLUSTER_COL = "LAD22CD"

#: A "no interval" sentinel: (ratio, low, high, se_log) all NaN.
NAN_CI: tuple[float, float, float, float] = (
    float("nan"),
    float("nan"),
    float("nan"),
    float("nan"),
)

#: Default bootstrap replications for composite (multi-fit) quantities.
BOOTSTRAP_REPS = 300

#: Default RNG seed (reproducible intervals).
SEED = 20260708


def _z(level: float) -> float:
    """Two-sided normal critical value for a confidence level."""
    return float(stats.norm.ppf(0.5 + level / 2.0))


def log_contrast_ci(
    result: Any,
    a: str,
    b: str,
    scale: float = 1.0,
    level: float = 0.95,
) -> tuple[float, float, float, float]:
    """
    Delta-method CI for the ratio ``exp(scale * (b_a - b_b))``.

    The contrast ``scale * (params[a] - params[b])`` is linear in the coefficients,
    so its variance comes directly from the model covariance
    ``scale**2 * (V_aa + V_bb - 2 V_ab)``. Because that covariance is whatever the
    model was fitted with (HC1 or cluster-robust), the interval inherits the same
    robustness.

    Parameters
    ----------
    result : statsmodels results
        A fitted model exposing ``.params`` and ``.cov_params()``.
    a, b : str
        Coefficient names for the numerator and denominator log-levels.
    scale : float, default 1.0
        Multiplier on the contrast (100 for percent-share coefficients, 1 for
        compositional share fractions).
    level : float, default 0.95
        Confidence level.

    Returns
    -------
    tuple of float
        ``(ratio, low, high, se_log)`` — the point ratio, its CI bounds, and the
        standard error on the log scale. NaNs if the coefficients are absent.
    """
    params = getattr(result, "params", None)
    if params is None or a not in params or b not in params:
        return NAN_CI
    cov = result.cov_params()
    logr = scale * (float(params[a]) - float(params[b]))
    var = scale**2 * (
        float(cov.loc[a, a]) + float(cov.loc[b, b]) - 2.0 * float(cov.loc[a, b])
    )
    se = float(np.sqrt(var)) if var > 0 else float("nan")
    z = _z(level)
    return (
        float(np.exp(logr)),
        float(np.exp(logr - z * se)),
        float(np.exp(logr + z * se)),
        se,
    )


def fmt_ci(ci: tuple[float, float, float, float] | tuple[float, float, float]) -> str:
    """Format a ``(point, low, high, ...)`` interval as ``1.60× [1.55, 1.65]``."""
    point, low, high = ci[0], ci[1], ci[2]
    if any(np.isnan(v) for v in (point, low, high)):
        return "n/a"
    return f"{point:.2f}× [{low:.2f}, {high:.2f}]"


def cluster_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    group_col: str = CLUSTER_COL,
    reps: int = BOOTSTRAP_REPS,
    level: float = 0.95,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """
    Percentile CI for a composite statistic via cluster resampling.

    Whole clusters (local authorities) are drawn with replacement and the
    statistic recomputed on the reassembled frame, so the interval reflects
    between-area dependence. Use this for quantities that combine several fits on
    the same rows and therefore have no single-model covariance.

    Parameters
    ----------
    frame : pandas.DataFrame
        The analysis frame; must contain ``group_col``.
    statistic : callable
        Maps a frame to a scalar (refits whatever models it needs). Should return
        NaN on a degenerate resample rather than raising.
    group_col : str, default :data:`CLUSTER_COL`
        Cluster identifier.
    reps : int, default :data:`BOOTSTRAP_REPS`
        Number of resamples.
    level : float, default 0.95
        Confidence level.
    seed : int, default :data:`SEED`
        RNG seed.

    Returns
    -------
    tuple of float
        ``(point, low, high)`` — the statistic on the observed frame and the
        percentile CI bounds across resamples.
    """
    point = float(statistic(frame))
    if group_col not in frame.columns:
        return (point, float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    groups = frame[group_col].dropna().unique()
    n = len(groups)
    # Pre-split once so each resample is a cheap concat of cluster blocks.
    blocks = {g: sub for g, sub in frame.groupby(group_col)}
    estimates: list[float] = []
    for _ in range(reps):
        picked = rng.choice(groups, size=n, replace=True)
        resample = pd.concat([blocks[g] for g in picked], ignore_index=True)
        try:
            val = float(statistic(resample))
        except Exception:
            val = float("nan")
        if not np.isnan(val):
            estimates.append(val)
    if len(estimates) < max(10, reps // 10):
        return (point, float("nan"), float("nan"))
    lo = float(np.percentile(estimates, 100 * (0.5 - level / 2)))
    hi = float(np.percentile(estimates, 100 * (0.5 + level / 2)))
    return (point, lo, hi)


def cluster_bootstrap_multi(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], dict[str, float]],
    group_col: str = CLUSTER_COL,
    reps: int = BOOTSTRAP_REPS,
    level: float = 0.95,
    seed: int = SEED,
) -> dict[str, tuple[float, float, float]]:
    """
    Percentile CIs for several statistics sharing one cluster resampling pass.

    Like :func:`cluster_bootstrap`, but ``statistic`` returns a dict of named
    scalars, all computed on the same resample. Use it when several quantities
    share the expensive fits (e.g. the three rate variants, which reuse one access
    Poisson), so the models are refitted once per resample, not once per quantity.

    Returns
    -------
    dict
        ``{name: (point, low, high)}`` for each key the statistic returns.
    """
    point = statistic(frame)
    keys = list(point.keys())
    if group_col not in frame.columns:
        return {k: (float(point[k]), float("nan"), float("nan")) for k in keys}
    rng = np.random.default_rng(seed)
    groups = frame[group_col].dropna().unique()
    n = len(groups)
    blocks = {g: sub for g, sub in frame.groupby(group_col)}
    collected: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(reps):
        picked = rng.choice(groups, size=n, replace=True)
        resample = pd.concat([blocks[g] for g in picked], ignore_index=True)
        try:
            vals = statistic(resample)
        except Exception:
            vals = None
        if vals is None:
            continue
        for k in keys:
            v = float(vals.get(k, float("nan")))
            if not np.isnan(v):
                collected[k].append(v)
    out: dict[str, tuple[float, float, float]] = {}
    for k in keys:
        ests = collected[k]
        if len(ests) < max(10, reps // 10):
            out[k] = (float(point[k]), float("nan"), float("nan"))
        else:
            lo = float(np.percentile(ests, 100 * (0.5 - level / 2)))
            hi = float(np.percentile(ests, 100 * (0.5 + level / 2)))
            out[k] = (float(point[k]), lo, hi)
    return out
