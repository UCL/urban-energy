"""
Climate confound — annual heating-degree-days (HDD) per Output Area.

Turns a HadUK-Grid 1 km monthly mean-temperature grid into an annual HDD value
for every England OA, used as the climate confound in the form/size energy
ladder (``stats/form_size_decomposition.py``). Colder places burn more heat; if
detached homes cluster in colder (rural / northern) areas, climate confounds the
raw form gap — so it is held alongside build era, income and tenure.

Input (manual download — HadUK-Grid is gated behind a free CEDA account):

    $DATA_DIR/climate/tas_hadukgrid_uk_1km_mon-30y_199101-202012.nc

        1 km monthly mean air temperature (``tas``), 1991-2020 30-year
        climatology — 12 monthly grids on the OSGB grid (EPSG:27700).

    Source: HadUK-Grid, Met Office (Open Government Licence), via the CEDA Archive
        https://catalogue.ceda.ac.uk/uuid/4dc8450d889a491ebb20e724debe2dfb
        latest version → 1km → tas → mon-30y → 1991-2020 → the ``.nc`` file.

Any NetCDF with a ``tas`` variable on 12 monthly steps over the OSGB grid works
(e.g. a single recent year of ``mon`` data); the file is located by glob.

Method
------
Per grid cell, for each calendar month::

    HDD(month) = days_in_month * max(0, BASE_TEMP_C - tas_month)

and the annual HDD is the sum over the twelve months, with
``BASE_TEMP_C = 15.5`` (the UK-standard base temperature). Computing HDD from
monthly means rather than daily values slightly understates the absolute total,
but is a consistent *relative* climate measure across OAs — all a confound needs.

Each OA's representative point (EPSG:27700) is matched to the nearest grid cell
with a valid (land) HDD value via a KD-tree, so coastal centroids that fall on a
sea cell still receive the nearest land value.

Output: ``$DATA_DIR/statistics/oa_hdd.parquet`` (``OA21CD``, ``hdd``).

    uv run python data/process_climate.py
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import KDTree

from urban_energy.paths import DATA_DIR

CLIMATE_DIR = DATA_DIR / "climate"
CENSUS = DATA_DIR / "statistics" / "census_oa_joined.gpkg"
OUT = DATA_DIR / "statistics" / "oa_hdd.parquet"

#: UK-standard heating base temperature (degrees Celsius).
BASE_TEMP_C: float = 15.5

#: Days per calendar month (climatology — leap day ignored), January first.
DAYS_IN_MONTH: list[int] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

#: Glob patterns tried in order to locate the temperature grid.
_TAS_GLOBS: list[str] = [
    "tas_hadukgrid_uk_1km_mon-30y_*.nc",
    "tas_hadukgrid_uk_1km_mon_*.nc",
    "tas_hadukgrid_uk_*.nc",
    "tas*.nc",
    "*.nc",
]


def _find_tas_file() -> Path:
    """Locate the HadUK-Grid temperature NetCDF, or raise with instructions."""
    for pattern in _TAS_GLOBS:
        hits = sorted(CLIMATE_DIR.glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No HadUK-Grid temperature NetCDF found in {CLIMATE_DIR}.\n\n"
        "Download manually (HadUK-Grid needs a free CEDA account):\n"
        "  1. Open the HadUK-Grid record:\n"
        "     https://catalogue.ceda.ac.uk/uuid/4dc8450d889a491ebb20e724debe2dfb\n"
        "  2. Latest 1 km version -> tas -> mon-30y -> 1991-2020.\n"
        "  3. Download: tas_hadukgrid_uk_1km_mon-30y_199101-202012.nc\n"
        f"  4. Save it to: {CLIMATE_DIR}/\n"
        "\n(Any NetCDF with a `tas` variable on 12 monthly steps also works.)"
    )


def _spatial_coord_names(da: xr.DataArray) -> tuple[str, str]:
    """Return the (x, y) 1-D projection-coordinate names of ``da``."""

    def _pick(axis: str, names: tuple[str, ...]) -> str:
        for name in da.coords:
            coord = da.coords[name]
            if coord.ndim != 1:
                continue
            attrs = coord.attrs
            if (
                name in names
                or attrs.get("axis", "").lower() == axis
                or attrs.get("standard_name", "") == f"projection_{axis}_coordinate"
            ):
                return str(name)
        raise ValueError(
            f"Could not identify the {axis}-coordinate in {list(da.coords)}"
        )

    x = _pick("x", ("projection_x_coordinate", "x", "easting"))
    y = _pick("y", ("projection_y_coordinate", "y", "northing"))
    return x, y


def _annual_hdd_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Open the NetCDF and return (x_coords, y_coords, annual-HDD grid [ny, nx])."""
    ds = xr.open_dataset(path, decode_times=False)
    name = (
        "tas"
        if "tas" in ds.data_vars
        else next(
            (
                v
                for v in ds.data_vars
                if ds[v].attrs.get("standard_name") == "air_temperature"
            ),
            None,
        )
    )
    if name is None:
        raise ValueError(f"No `tas` / air_temperature variable in {path.name}")

    da = ds[name]
    xname, yname = _spatial_coord_names(da)
    month_dims = [d for d in da.dims if d not in (xname, yname)]
    if len(month_dims) != 1:
        raise ValueError(f"Expected one non-spatial dim, found {month_dims}")
    mdim = month_dims[0]
    if da.sizes[mdim] != 12:
        raise ValueError(
            f"Expected 12 monthly steps along '{mdim}', found {da.sizes[mdim]} — "
            "supply a monthly (mon / mon-30y) grid, not annual/seasonal."
        )

    vals = np.asarray(da.transpose(mdim, yname, xname).values, dtype=float)
    if da.attrs.get("units", "").lower() in ("k", "kelvin"):
        vals = vals - 273.15

    days = np.asarray(DAYS_IN_MONTH, dtype=float)[:, None, None]
    hdd = np.sum(np.clip(BASE_TEMP_C - vals, 0.0, None) * days, axis=0)
    # Any month missing (sea / outside-UK cell) → no valid annual HDD.
    hdd[np.any(~np.isfinite(vals), axis=0)] = np.nan

    xs = np.asarray(ds[xname].values, dtype=float)
    ys = np.asarray(ds[yname].values, dtype=float)
    ds.close()
    return xs, ys, hdd


def _oa_points() -> pd.DataFrame:
    """OA representative points in EPSG:27700 (OA21CD, x, y)."""
    g = gpd.read_file(CENSUS, columns=["OA21CD"]).to_crs(27700)
    pt = g.geometry.representative_point()
    return pd.DataFrame(
        {"OA21CD": g["OA21CD"].to_numpy(), "x": pt.x.to_numpy(), "y": pt.y.to_numpy()}
    )


def main() -> pd.DataFrame:
    """Compute annual HDD per OA from the HadUK-Grid grid; write the parquet."""
    path = _find_tas_file()
    print(f"[1/3] Reading temperature grid: {path.name}")
    xs, ys, hdd = _annual_hdd_grid(path)

    print("[2/3] Sampling HDD at OA representative points …")
    oa = _oa_points()
    gx, gy = np.meshgrid(xs, ys)
    valid = np.isfinite(hdd)
    tree = KDTree(np.c_[gx[valid], gy[valid]])
    _, idx = tree.query(oa[["x", "y"]].to_numpy())
    oa["hdd"] = hdd[valid][idx]

    out = oa[["OA21CD", "hdd"]].copy()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    print(f"[3/3] Wrote {OUT}  ({len(out):,} OAs)")
    print(
        f"  HDD (base {BASE_TEMP_C:.1f} C): median {out['hdd'].median():,.0f}  "
        f"range {out['hdd'].min():,.0f}–{out['hdd'].max():,.0f}"
    )
    return out


if __name__ == "__main__":
    main()
