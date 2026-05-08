"""
Local-Authority-District aggregation for the Atlas.

Builds a per-LAD GeoDataFrame by spatially joining OAs to LAD polygons
(OS Boundary Line `district_borough_unitary` layer) and aggregating each
NEPI surface as a population-weighted mean (UPRN count as the weight —
reasonable proxy for household count without bringing in another dataset).

Cache the OA→LAD spatial-join lookup at
`$DATA_DIR/lookups/oa_lad_lookup.parquet` so the heavy spatial join runs
once per dataset vintage.

Public entry point: `build_lad_dataset(national_df) -> gpd.GeoDataFrame`.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from urban_energy.paths import DATA_DIR, PROCESSING_DIR

from .geometry import GPKG_PATH
from .region import BDLINE_PATH

LOOKUP_PATH = DATA_DIR / "lookups" / "oa_lad_lookup.parquet"

# LAD-level NEPI fields that need population-weighted aggregation.
# These mirror the per-OA columns the frontend already consumes via pmtiles.
_AGG_COLS = [
    "nepi_total_kwh",
    "nepi_form_kwh",
    "nepi_mobility_kwh",
    "nepi_access_kwh",
    "oa_elec_mean_kwh",
    "oa_gas_mean_kwh",
    "bev_share",
    "local_coverage",
    "people_per_ha",
    "pct_detached",
    "pct_semi",
    "pct_terraced",
    "pct_flat",
    "cc_bus_800_wt",
    "cc_rail_800_wt",
    "median_build_year",
]


def build_oa_lad_lookup(force: bool = False) -> pd.DataFrame:
    """
    Spatially join OA centroids to LAD polygons. Cached as parquet.

    Returns a DataFrame with columns OA21CD, LAD22CD, LAD22NM.
    """
    if LOOKUP_PATH.exists() and not force:
        return pd.read_parquet(LOOKUP_PATH)

    print("  Building OA→LAD spatial-join lookup...")
    oas = gpd.read_file(GPKG_PATH, columns=["OA21CD", "geometry"])
    oas = oas.drop_duplicates(subset=["OA21CD"]).copy()
    oas["geometry"] = oas.geometry.centroid

    lads = gpd.read_file(BDLINE_PATH, layer="district_borough_unitary")
    lads = lads[["Name", "Census_Code", "geometry"]].rename(
        columns={"Name": "LAD22NM", "Census_Code": "LAD22CD"}
    )

    print(f"    Joining {len(oas):,} OAs to {len(lads):,} LAD polygons...")
    joined = gpd.sjoin(oas, lads, how="left", predicate="within")
    out = joined[["OA21CD", "LAD22CD", "LAD22NM"]].copy()

    n_unmatched = out["LAD22CD"].isna().sum()
    if n_unmatched > 0:
        print(f"    {n_unmatched:,} OAs without a LAD match (likely outside England)")

    LOOKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LOOKUP_PATH, index=False)
    print(f"    Cached → {LOOKUP_PATH}")
    return out


def _pop_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return float("nan")
    return float((values[valid] * weights[valid]).sum() / weights[valid].sum())


def build_lad_dataset(national_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Aggregate the national NEPI dataframe to LAD level.

    Returns a GeoDataFrame in EPSG:4326 with one row per LAD, geometry,
    and population-weighted means of the per-OA NEPI fields. `dominant_type`
    is set to the modal type weighted by OA UPRN counts.
    """
    print("[LAD] Building OA→LAD lookup...")
    lookup = build_oa_lad_lookup()

    print("[LAD] Joining lookup with national NEPI dataframe...")
    df = national_df.merge(lookup[["OA21CD", "LAD22CD"]], on="OA21CD", how="left")
    df = df[df["LAD22CD"].notna()].copy()
    weight_col = "n_uprns" if "n_uprns" in df.columns else None

    print(f"[LAD] Aggregating {len(df):,} OAs to LAD level...")
    rows = []
    for lad_code, group in df.groupby("LAD22CD"):
        weights = group[weight_col].fillna(0) if weight_col else pd.Series([1] * len(group))
        row: dict[str, object] = {
            "LAD22CD": lad_code,
            "n_oas": int(len(group)),
            "n_uprns": int(weights.sum()) if weight_col else int(len(group)),
        }
        for col in _AGG_COLS:
            if col in group.columns:
                row[col] = _pop_weighted_mean(group[col], weights)
        # Modal dominant_type weighted by UPRN
        if "dominant_type" in group.columns:
            type_weights = (
                group.assign(_w=weights)
                .groupby("dominant_type")["_w"].sum()
                .sort_values(ascending=False)
            )
            row["dominant_type"] = (
                str(type_weights.index[0]) if len(type_weights) else None
            )
        rows.append(row)

    lad_df = pd.DataFrame(rows)

    # Re-band on the LAD distribution. Keeps A–G meaningful at the LAD scale
    # (otherwise everyone would cluster in the middle bands since OA totals
    # average out at LAD scale).
    from nepi import BAND_PERCENTILES
    valid = lad_df["nepi_total_kwh"].notna() & (lad_df["nepi_total_kwh"] > 0)
    pct_rank = lad_df.loc[valid, "nepi_total_kwh"].rank(pct=True) * 100
    lad_df["nepi_band"] = ""
    for band, (lo, hi) in BAND_PERCENTILES.items():
        mask = valid & (pct_rank > lo) & (pct_rank <= hi)
        lad_df.loc[mask.index[mask], "nepi_band"] = band
    bottom = valid & (pct_rank <= BAND_PERCENTILES["A"][1])
    lad_df.loc[bottom.index[bottom], "nepi_band"] = "A"

    # Geometry
    print("[LAD] Attaching LAD geometry...")
    lads = gpd.read_file(BDLINE_PATH, layer="district_borough_unitary")
    lads = lads.rename(columns={"Census_Code": "LAD22CD", "Name": "LAD22NM"})
    lads = lads[["LAD22CD", "LAD22NM", "geometry"]]
    out = lads.merge(lad_df, on="LAD22CD", how="inner")
    out = out.to_crs(4326)
    print(f"[LAD] Built {len(out):,} LAD rows")
    return out
