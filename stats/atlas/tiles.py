"""
Vector-tile generation: GeoJSON → MBTiles → PMTiles.

Two CLI tools (must be on PATH):

    tippecanoe   GeoJSON → MBTiles    (https://github.com/felt/tippecanoe)
    pmtiles      MBTiles → PMTiles    (https://github.com/protomaps/go-pmtiles)

PMTiles is the deployable artefact: a single archive file, served as a
static asset over HTTP `Range:` requests by GitHub Pages.

Tuning notes:
    -Z / -z       Zoom range (set in `schema`).
    -l            Layer name in the resulting tile schema.
    --simplification=10
                  Drop coordinates where they round-trip to within ~10
                  pixels at the given zoom. Cheap geometric reduction.
    --coalesce-densest-as-needed
                  When a tile would exceed the 500 KB hard limit, merge
                  the densest features. Lets us scale to national without
                  hitting per-tile limits.
    --no-tile-size-limit
                  Hard ceiling override — only enable for big regions
                  where coalescing isn't enough.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .schema import TILE_LAYER_NAME, TIPPECANOE_MAX_ZOOM, TIPPECANOE_MIN_ZOOM


class ToolingMissingError(RuntimeError):
    """Raised when tippecanoe or pmtiles isn't on PATH."""


def _check_tooling() -> None:
    missing = [t for t in ("tippecanoe", "pmtiles") if shutil.which(t) is None]
    if missing:
        raise ToolingMissingError(
            f"Required CLI tool(s) not found on PATH: {', '.join(missing)}. "
            f"Install with: brew install tippecanoe pmtiles"
        )


def geojson_to_pmtiles(
    geojson_path: Path,
    pmtiles_path: Path,
    layer_name: str = TILE_LAYER_NAME,
    min_zoom: int = TIPPECANOE_MIN_ZOOM,
    max_zoom: int = TIPPECANOE_MAX_ZOOM,
    simplification: int = 10,
    drop_densest: bool = False,
    max_tile_bytes: int | None = None,
) -> Path:
    """
    Convert a single-layer GeoJSON to PMTiles via tippecanoe + pmtiles.

    Per-layer options are exposed because LAD layers and OA layers want
    very different settings: LADs are small (~300 polygons, no drops, low
    simplification, full fidelity) while national OA layers need aggressive
    drop-densest to fit anywhere usable.
    """
    _check_tooling()
    if not geojson_path.exists():
        raise FileNotFoundError(geojson_path)
    pmtiles_path.parent.mkdir(parents=True, exist_ok=True)
    mbtiles_path = pmtiles_path.with_suffix(".mbtiles")

    cmd: list[str] = [
        "tippecanoe",
        "-o", str(mbtiles_path),
        "-l", layer_name,
        f"-Z{min_zoom}",
        f"-z{max_zoom}",
        f"--simplification={simplification}",
        "--force",
    ]
    if drop_densest:
        cmd.append("--drop-densest-as-needed")
    if max_tile_bytes is not None:
        cmd.append(f"--maximum-tile-bytes={max_tile_bytes}")
    cmd.append(str(geojson_path))

    print(f"  tippecanoe → {mbtiles_path.name} (layer={layer_name}, "
          f"z={min_zoom}-{max_zoom}, simp={simplification}, "
          f"drop_densest={drop_densest})")
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    print(f"  pmtiles convert → {pmtiles_path.name}")
    if pmtiles_path.exists():
        pmtiles_path.unlink()
    subprocess.run(
        ["pmtiles", "convert", str(mbtiles_path), str(pmtiles_path)],
        check=True, capture_output=True, text=True,
    )
    mbtiles_path.unlink()
    size_mb = pmtiles_path.stat().st_size / 1e6
    print(f"  pmtiles ready: {pmtiles_path} ({size_mb:.1f} MB)")
    return pmtiles_path


# Kept for callers that want a single multi-layer archive (smaller regions
# where size pressure is low). The national export uses two separate files.
def multilayer_geojson_to_pmtiles(
    layers: list[tuple[Path, str, dict]],
    pmtiles_path: Path,
) -> Path:
    _check_tooling()
    pmtiles_path.parent.mkdir(parents=True, exist_ok=True)
    mbtiles_path = pmtiles_path.with_suffix(".mbtiles")
    cmd: list[str] = ["tippecanoe", "-o", str(mbtiles_path), "--force"]
    for geojson_path, layer_name, opts in layers:
        if not geojson_path.exists():
            raise FileNotFoundError(geojson_path)
        cmd += [
            f"-L{layer_name}:{geojson_path}",
            f"-Z{opts.get('min_zoom', TIPPECANOE_MIN_ZOOM)}",
            f"-z{opts.get('max_zoom', TIPPECANOE_MAX_ZOOM)}",
            f"--simplification={opts.get('simplification', 10)}",
        ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    if pmtiles_path.exists():
        pmtiles_path.unlink()
    subprocess.run(
        ["pmtiles", "convert", str(mbtiles_path), str(pmtiles_path)],
        check=True, capture_output=True, text=True,
    )
    mbtiles_path.unlink()
    size_mb = pmtiles_path.stat().st_size / 1e6
    print(f"  pmtiles ready: {pmtiles_path} ({size_mb:.1f} MB)")
    return pmtiles_path
