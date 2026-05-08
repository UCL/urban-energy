"""
Top-level orchestration: take national NEPI + a Region, write outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .compute import compute_national_nepi
from .geometry import attach_region_geometry
from .lad import build_lad_dataset
from .region import Region, get_region
from .schema import (
    GEOJSON_COORDINATE_PRECISION,
    OA_PROPERTIES,
    OUTPUT_DIR,
)
from .summary import build_summary
from .tiles import (
    ToolingMissingError,
    geojson_to_pmtiles,
    multilayer_geojson_to_pmtiles,
)


def _select_properties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Project to OA_PROPERTIES.

    Float NaNs are preserved — Fiona's GeoJSON driver writes them as JSON
    null, which MapLibre treats as a missing property (correct behaviour
    for our colour-ramp expressions). Casting to object dtype here would
    cause Fiona to fall back to string serialisation for the entire column.
    """
    keep = [c for c in OA_PROPERTIES if c in df.columns]
    return df[keep].copy()


def _print_distribution(label: str, dist: dict) -> None:
    print(
        f"  {label}: N={dist['n_oas']:,} · median={dist['median_kwh']:,.0f}"
        f" kWh/hh/yr · IQR={dist['p25_kwh']:,.0f}–{dist['p75_kwh']:,.0f}"
    )
    bands = "  ".join(f"{b}={dist['band_counts'][b]:,}" for b in "ABCDEFG")
    print(f"      bands: {bands}")


def _select_lad_properties(gdf) -> "gpd.GeoDataFrame":
    """Trim LAD GeoDataFrame to fields the frontend needs."""
    keep = [
        "LAD22CD", "LAD22NM", "n_oas", "n_uprns", "dominant_type", "nepi_band",
        "nepi_total_kwh", "nepi_form_kwh", "nepi_mobility_kwh", "nepi_access_kwh",
        "oa_elec_mean_kwh", "oa_gas_mean_kwh", "bev_share", "local_coverage",
        "people_per_ha", "pct_detached", "pct_semi", "pct_terraced", "pct_flat",
        "cc_bus_800_wt", "cc_rail_800_wt", "median_build_year",
        "geometry",
    ]
    return gdf[[c for c in keep if c in gdf.columns]]


def export_region(
    region: Region | str,
    use_cache: bool = True,
    out_dir: Path = OUTPUT_DIR,
    tiles: bool = True,
    include_lad: bool = True,
) -> dict[str, Path]:
    """
    Export per-region artefacts for the Atlas frontend.

    Outputs:
        {region.slug}_oas.geojson     OA polygons (intermediate for tippecanoe)
        {region.slug}_lads.geojson    LAD polygons (intermediate)
        {region.slug}.pmtiles         Multi-layer vector tiles (oa + lad)
        summary.json                  Schema-stable contract for the frontend

    Two layers in the same pmtiles archive: `oa` (zoom 9–14) and `lad`
    (zoom 4–14) so the frontend can switch between Output-Area and
    Local-Authority-District resolution without reloading.
    """
    if isinstance(region, str):
        region = get_region(region)

    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] National NEPI")
    national = compute_national_nepi(use_cache=use_cache)

    print(f"[2/6] Spatial filter (OA): {region.label}")
    properties = _select_properties(national)
    gdf = attach_region_geometry(region, properties)
    if len(gdf) == 0:
        raise ValueError(f"No OAs found in region {region.slug!r}")
    print(f"      {len(gdf):,} OAs in region")

    region_df = national[national["OA21CD"].isin(set(gdf["OA21CD"]))].drop_duplicates(
        subset=["OA21CD"]
    )

    lad_gdf = None
    if include_lad:
        print("[3/6] Aggregating to LAD level")
        lad_gdf = _select_lad_properties(build_lad_dataset(region_df))
        print(f"      {len(lad_gdf):,} LADs")
    else:
        print("[3/6] Skipping LAD aggregation (include_lad=False)")

    print("[4/6] Building summary")
    summary = build_summary(national, region.label, region_df, lad_gdf=lad_gdf)
    summary["region"] = {"slug": region.slug, "label": region.label, "kind": region.kind}
    bounds = gdf.total_bounds
    summary["bounds"] = [
        [float(bounds[0]), float(bounds[1])],
        [float(bounds[2]), float(bounds[3])],
    ]

    print("[5/6] Writing GeoJSON + summary")
    oa_geojson = out_dir / f"{region.slug}_oas.geojson"
    lad_geojson = out_dir / f"{region.slug}_lads.geojson"
    summary_path = out_dir / "summary.json"

    for p in (oa_geojson, lad_geojson):
        if p.exists():
            p.unlink()

    gdf.to_file(
        oa_geojson, driver="GeoJSON",
        COORDINATE_PRECISION=GEOJSON_COORDINATE_PRECISION,
    )
    if lad_gdf is not None:
        lad_gdf.to_file(
            lad_geojson, driver="GeoJSON",
            COORDINATE_PRECISION=GEOJSON_COORDINATE_PRECISION,
        )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"      {oa_geojson.name}: {oa_geojson.stat().st_size / 1024:,.0f} KB")
    if lad_gdf is not None:
        print(f"      {lad_geojson.name}: {lad_geojson.stat().st_size / 1024:,.0f} KB")
    print(f"      {summary_path.name}: {summary_path.stat().st_size / 1024:.1f} KB")

    pmtiles_paths: dict[str, Path] = {}
    if tiles:
        print("[6/6] Generating vector tiles (LAD + OA as separate pmtiles)")
        try:
            # LAD pmtiles — full fidelity. ~300 polygons, no aggressive
            # measures; just modest simplification. Visible at all zooms.
            if lad_gdf is not None:
                lad_pmtiles = out_dir / f"{region.slug}_lad.pmtiles"
                geojson_to_pmtiles(
                    lad_geojson, lad_pmtiles,
                    layer_name="lad",
                    min_zoom=4, max_zoom=14,
                    simplification=5,
                    drop_densest=False,
                )
                pmtiles_paths["lad"] = lad_pmtiles

            # OA pmtiles — aggressive size control needed for national.
            # Min zoom 11: whole country at zoom <11 uses LAD layer only.
            oa_pmtiles = out_dir / f"{region.slug}_oa.pmtiles"
            geojson_to_pmtiles(
                oa_geojson, oa_pmtiles,
                layer_name="oa",
                min_zoom=11, max_zoom=14,
                simplification=15,
                drop_densest=True,
                max_tile_bytes=500_000,
            )
            pmtiles_paths["oa"] = oa_pmtiles

            # Read optional R2/external host base URL from env so deploys
            # can point the (large, > 100 MB) OA pmtiles at Cloudflare R2
            # without changing code. Default: relative URL (local + GH Pages).
            import os as _os
            oa_host = _os.environ.get("ATLAS_OA_TILES_URL_BASE", "").rstrip("/")
            oa_url = (
                f"{oa_host}/{region.slug}_oa.pmtiles" if oa_host
                else f"{region.slug}_oa.pmtiles"
            )
            summary["tiles"] = {
                "lad": {
                    "url": f"{region.slug}_lad.pmtiles",
                    "layer": "lad",
                    "size_bytes": pmtiles_paths["lad"].stat().st_size if "lad" in pmtiles_paths else None,
                } if "lad" in pmtiles_paths else None,
                "oa": {
                    "url": oa_url,
                    "layer": "oa",
                    "size_bytes": pmtiles_paths["oa"].stat().st_size,
                },
            }
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
        except ToolingMissingError as e:
            print(f"      SKIPPED: {e}")
    else:
        print("[6/6] Skipping vector tiles (tiles=False)")

    print()
    _print_distribution("National OA", summary["national"])
    _print_distribution(region.label, summary["bua_distribution"])

    out = {"geojson": oa_geojson, "summary": summary_path}
    if lad_gdf is not None:
        out["lad_geojson"] = lad_geojson
    out.update(pmtiles_paths)
    return out


# Backward-compat shim — older callers / tests.
def export_bua(bua: str, **kwargs) -> dict[str, Path]:
    return export_region(bua, **kwargs)
