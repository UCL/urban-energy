"""Coarser-cluster sensitivity for the headline total-energy gap interval.

The 95% intervals cluster on local-authority districts, which assumes
independence between districts. This check merges the districts into ~50
contiguous spatial blocks (k-means on district centroids, seeded), so that
dependence crossing district boundaries is absorbed, and re-estimates the S0
compositional Detached:Flat total-energy interval. The point estimate is
unchanged by construction (clustering only affects the covariance); the
question is how far the interval widens.

Print-only, run on demand; records ``spatialBlockGap``/``spatialBlockGapCI``
(k=50) to the ledger for the Robustness sentence and Methods clause.
"""

from __future__ import annotations

import geopandas as gpd
import ledger
import numpy as np
import pandas as pd
from form_size_decomposition import (
    _SHARE_FRACS,
    _comp_ols,
    _compositional_frame,
    _deprivation_cols,
    _hdd_cols,
    _tenure_cols,
)
from inference import CLUSTER_COL, fmt_ci, log_contrast_ci
from oa_data import _num, load_and_aggregate

from urban_energy.paths import DATA_DIR

_CENSUS = DATA_DIR / "statistics" / "census_oa_joined.gpkg"

#: Spatial block counts to evaluate; the first is recorded to the ledger.
BLOCK_COUNTS: tuple[int, ...] = (50, 100)


def _spatial_blocks(cf: pd.DataFrame, k: int) -> pd.Series:
    """District labels merged into ``k`` contiguous spatial blocks.

    District centroids (mean of member-OA representative points) are grouped
    by seeded k-means, so the assignment is deterministic and every district's
    areas stay together inside one block.

    Parameters
    ----------
    cf : pandas.DataFrame
        Compositional frame with ``OA21CD`` and the district column.
    k : int
        Number of spatial blocks.

    Returns
    -------
    pandas.Series
        Block label per row of ``cf``.
    """
    from scipy.cluster.vq import kmeans2

    geo = gpd.read_file(_CENSUS, columns=["OA21CD"])
    pts = geo.geometry.representative_point()
    coords = (
        pd.DataFrame({"OA21CD": geo["OA21CD"], "x": pts.x, "y": pts.y})
        .merge(cf[["OA21CD", CLUSTER_COL]], on="OA21CD", how="inner")
        .groupby(CLUSTER_COL)[["x", "y"]]
        .mean()
    )
    _, labels = kmeans2(coords.to_numpy(), k, seed=42, minit="++")
    return cf[CLUSTER_COL].map(dict(zip(coords.index, labels, strict=True)))


def main() -> None:
    """Print the total-gap interval under district and spatial-block clustering."""
    df = load_and_aggregate()
    cf = _compositional_frame(df)
    total = _num(cf["building_kwh_per_hh"]) + _num(cf["transport_kwh_per_hh_total_est"])
    cf["_log_total"] = np.log(total.clip(lower=1).to_numpy())
    conf = (
        ["median_build_year"] + _deprivation_cols(cf) + _tenure_cols(cf) + _hdd_cols(cf)
    )

    print("\n  Detached:Flat total-energy gap by clustering unit:")
    runs: list[tuple[str, str]] = [(CLUSTER_COL, "LAD (headline)")]
    for k in BLOCK_COUNTS:
        col = f"block{k}"
        cf[col] = _spatial_blocks(cf, k)
        runs.append((col, f"{k} spatial blocks"))

    for col, note in runs:
        m = _comp_ols(
            cf, "_log_total", _SHARE_FRACS + conf, "total_hh", cluster_col=col
        )
        if m is None:
            print(f"  {note:<22s} fit failed")
            continue
        n_cl = getattr(m, "_n_clusters", None)
        ci = log_contrast_ci(m, "s_detached", "s_flat")
        print(f"  {note:<22s} G={n_cl:>4}  {fmt_ci(ci)}")
        if col == f"block{BLOCK_COUNTS[0]}":
            ledger.record(
                spatialBlockGap=ledger.pt(ci[0]),
                spatialBlockGapCI=ledger.ci(ci[1], ci[2]),
            )


if __name__ == "__main__":
    main()
