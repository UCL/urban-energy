"""
Compute the national NEPI dataframe — cached on disk.

This is the expensive step (~30s national load + aggregate + access + NEPI).
Cached as parquet in `$DATA_DIR/stats/nepi_national.parquet` so iterating on
the export side is instant. The cache is invalidated by passing
`use_cache=False` (e.g. after underlying data or NEPI logic changes).
"""

from pathlib import Path

import pandas as pd

from .schema import NEPI_CACHE_PATH


def compute_national_nepi(use_cache: bool = True) -> pd.DataFrame:
    """
    Run the canonical NEPI pipeline on the full national OA dataset.

    Parameters
    ----------
    use_cache : bool
        If True (default) and a cached parquet exists, load it. If False,
        re-run the pipeline and overwrite the cache.

    Returns
    -------
    pd.DataFrame
        OA-level dataframe with NEPI surfaces, band, and supporting columns.
        See `schema.OA_PROPERTIES` for the fields downstream consumers rely on.
    """
    if use_cache and NEPI_CACHE_PATH.exists():
        print(f"  Loading cached national NEPI: {NEPI_CACHE_PATH}")
        return pd.read_parquet(NEPI_CACHE_PATH)

    # Imports deferred so that consumers loading from cache don't pay the cost
    # of pulling in statsmodels, geopandas, etc. just to read a parquet.
    from nepi import compute_nepi
    from proof_of_concept_oa import build_accessibility, load_and_aggregate

    print("  Computing national NEPI from scratch (this takes ~30s)...")
    df = load_and_aggregate()
    df = build_accessibility(df)
    df = compute_nepi(df)

    NEPI_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(NEPI_CACHE_PATH)
    print(f"  Cached → {NEPI_CACHE_PATH} ({NEPI_CACHE_PATH.stat().st_size / 1e6:.0f} MB)")
    return df
