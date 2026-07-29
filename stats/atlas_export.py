"""
Atlas export — static artefacts for the NEPI Atlas site (dissemination Phase 1).

Implements the export half of ``dissemination/atlas_architecture.md``. From the
score frame (``oa_nepi_score.parquet``) and the OA boundaries this writes, all
static, no backend:

* **Tiles** — three PMTiles archives (LAD / LSOA / OA), switched by zoom in the
  site. Each level carries its own pre-aggregated attributes, so a click at any
  scale reads summary statistics straight from the tile. Built via
  GeoJSONSeq → tippecanoe → pmtiles (both must be on PATH).
* **Aggregates** — ``docs/data/aggregates.json``: the England panel plus a
  lightweight LAD rank table. Slider maths needs only five linear sums per
  unit (gas, fabric-treated gas, electricity, travel, EV-treated travel), all
  in GWh/yr; letter shifts ship only for current and full deployment.
* **Meta** — ``docs/data/meta.json``: frozen band thresholds, technology
  constants and the paper's headline numbers (read from the ledger's
  ``numbers.json``, never hand-typed).
* **Postcode index** — ``docs/data/pc/{OUTCODE}.json`` shards mapping each
  postcode to its OA and centroid, for client-side search.

Tile property names are compressed; the mapping is in ``_OA_PROPS`` /
``_UNIT_PROPS`` below and mirrored in ``docs/app.js``.

Run (score first: ``uv run python stats/nepi_score.py``):
    uv run python stats/atlas_export.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import geopandas as gpd
import pandas as pd

from urban_energy.paths import DATA_DIR, PROJECT_DIR

_STATS = DATA_DIR / "statistics"
SCORE_PATH = _STATS / "oa_nepi_score.parquet"
BOUNDS_PATH = _STATS / "oa_boundaries.gpkg"
LOOKUP_PATH = _STATS / "oa_lookup.parquet"
PCODE_PATH = _STATS / "postcode_oa_lookup.parquet"
BANDS_PATH = PROJECT_DIR / "dissemination" / "nepi_bands_2021.json"
NUMBERS_PATH = PROJECT_DIR / "paper" / "latex" / "numbers.json"

DOCS = PROJECT_DIR / "docs"
BUILD = PROJECT_DIR / "temp" / "atlas_build"

LETTERS = "ABCDEFG"

#: OA tile properties (compressed → meaning; mirrored in docs/app.js).
_OA_PROPS = {
    "id": "OA21CD",
    "dt": "dominant type",
    "hh": "households",
    "hs": "avg household size",
    "fa": "median floor area m2",
    "g": "gas kWh/hh",
    "e": "elec kWh/hh",
    "t": "travel kWh/hh",
    "f": "fabric factor",
    "v": "EV intensity ratio",
    "aw": "amenities on foot (1.6 km)",
    "ac": "amenities at own catchment",
    "r": "rate (amenities/kWh)",
    "p": "rate percentile (hh-weighted)",
    "lr": "letter rate",
    "lrp": "letter rate potential",
    "le": "letter energy",
    "lep": "letter energy potential",
    "la": "letter access",
    "fx": "flags (1 no EPC, 2 low meters)",
}
#: LAD/LSOA tile properties: id, nm, hh, n (OAs), mr/me/ma (hh-weighted
#: medians: rate, total energy, walk access), p (rank percentile), cA..cG /
#: pA..pG (rate letter household counts, current/potential), sg/sgf/se/st/sv
#: (GWh/yr sums: gas, fabric-treated gas, elec, travel, EV-treated travel).
_UNIT_PROPS = "documented above"


def _wmedian(values: pd.Series, weights: pd.Series) -> float:
    """Weighted median (household-weighted throughout the Atlas)."""
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    ok = v.notna() & w.notna() & (w > 0)
    if not ok.any():
        return float("nan")
    sv = v[ok].sort_values()
    cw = w[ok].reindex(sv.index).cumsum()
    return float(sv[cw >= 0.5 * float(cw.iloc[-1])].iloc[0])


def _wpercentile(values: pd.Series, weights: pd.Series) -> pd.Series:
    """Weighted percentile rank (0–100) of each value within the frame."""
    order = values.sort_values().index
    cum = weights.reindex(order).cumsum()
    pct = (cum - 0.5 * weights.reindex(order)) / float(cum.iloc[-1]) * 100
    return pct.reindex(values.index)


def _letter_counts(letters: pd.Series, hh: pd.Series, prefix: str) -> dict:
    """Household counts per letter, keyed ``{prefix}{letter}``."""
    return {f"{prefix}{letter}": int(hh[letters == letter].sum()) for letter in LETTERS}


def _unit_row(g: pd.DataFrame) -> dict:
    """Aggregate attributes for one unit (an LSOA, LAD, or England)."""
    hh = g["total_hh"]
    gwh = 1e-6
    return {
        "hh": int(hh.sum()),
        "n": int(len(g)),
        "mr": round(_wmedian(g["rate"], hh), 4),
        "me": round(_wmedian(g["total_kwh_hh"], hh)),
        "ma": round(_wmedian(g["access_walk"], hh)),
        **_letter_counts(g["letter_rate"], hh, "c"),
        **_letter_counts(g["letter_rate_potential"], hh, "p"),
        "sg": round(float((g["gas_kwh_hh"] * hh).sum()) * gwh, 3),
        "sgf": round(float((g["gas_kwh_hh"] * g["fabric_factor"] * hh).sum()) * gwh, 3),
        "se": round(float((g["elec_kwh_hh"] * hh).sum()) * gwh, 3),
        "st": round(float((g["travel_kwh_hh"] * hh).sum()) * gwh, 3),
        "sv": round(float((g["travel_kwh_hh"] * g["ev_ratio"] * hh).sum()) * gwh, 3),
    }


def _load_frame() -> pd.DataFrame:
    """Score frame joined to LAD/LSOA names."""
    score = pd.read_parquet(SCORE_PATH)
    names = pd.read_parquet(
        LOOKUP_PATH, columns=["OA21CD", "LSOA21NM", "LAD22NM"]
    ).drop_duplicates("OA21CD")
    return score.merge(names, on="OA21CD", how="left", validate="1:1")


def write_aggregates(df: pd.DataFrame) -> dict:
    """England panel + LAD rank table → ``docs/data/aggregates.json``."""
    england = _unit_row(df)
    england.update(_letter_counts(df["letter_energy"], df["total_hh"], "e"))
    england.update(_letter_counts(df["letter_access"], df["total_hh"], "a"))
    # Dominant-type poles, for the live morphological-penalty tile: the same
    # linear sums, restricted to flat- and detached-dominant areas.
    england["types"] = {
        t: _unit_row(df[df["dominant_type"] == t]) for t in ("Flat", "Detached")
    }
    lads = []
    lad_rows = df.groupby("LAD22CD", observed=True)
    med = lad_rows.apply(lambda g: _wmedian(g["rate"], g["total_hh"]))
    rank = med.rank(pct=True) * 100
    for code, g in lad_rows:
        lads.append(
            {
                "id": code,
                "nm": str(g["LAD22NM"].iloc[0]),
                "hh": int(g["total_hh"].sum()),
                "mr": round(float(med[code]), 4),
                "p": round(float(rank[code]), 1),
            }
        )
    out = {"england": england, "lad": sorted(lads, key=lambda r: -r["mr"])}
    (DOCS / "data").mkdir(parents=True, exist_ok=True)
    (DOCS / "data" / "aggregates.json").write_text(json.dumps(out))
    return out


def write_meta() -> None:
    """Bands, constants and paper headlines → ``docs/data/meta.json``."""
    bands = json.loads(BANDS_PATH.read_text())
    numbers = json.loads(NUMBERS_PATH.read_text())
    from scenarios import BOILER_EFF, COP

    meta = {
        "bands": bands,
        "constants": {"boiler_eff": BOILER_EFF, "cop": COP},
        "headlines": {
            k: numbers[k]
            for k in (
                "walkAmen",
                "rate",
                "rateCI",
                "totalGap",
                "totalGapCI",
                "famNowGap",
                "travelGap",
                "catchAmen",
                "fullSurvives",
                "cccGap",
                "sampleN",
            )
            if k in numbers
        },
    }
    (DOCS / "data" / "meta.json").write_text(json.dumps(meta))


def write_postcode_index(df: pd.DataFrame, centroids: pd.DataFrame) -> int:
    """Postcode → (OA, lon, lat) shards keyed by outcode → ``docs/data/pc/``."""
    pc = pd.read_parquet(PCODE_PATH, columns=["Postcode", "OA21CD"])
    pc = pc.merge(centroids, on="OA21CD", how="inner")
    pc = pc[pc["OA21CD"].isin(df["OA21CD"])]
    key = pc["Postcode"].str.replace(" ", "", regex=False).str.upper()
    pc = pc.assign(_key=key, _out=key.str[:-3])
    shard_dir = DOCS / "data" / "pc"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True)
    n = 0
    for outcode, g in pc.groupby("_out"):
        shard = {
            k: [o, round(x, 5), round(y, 5)]
            for k, o, x, y in zip(
                g["_key"], g["OA21CD"], g["LONG"], g["LAT"], strict=True
            )
        }
        (shard_dir / f"{outcode}.json").write_text(json.dumps(shard))
        n += 1
    return n


def _tippecanoe(src: Path, layer: str, minz: int, maxz: int) -> Path:
    """GeoJSONSeq → MBTiles → PMTiles under ``docs/tiles/``."""
    mb = BUILD / f"{layer}.mbtiles"
    subprocess.run(
        [
            "tippecanoe",
            "-o",
            str(mb),
            "--force",
            "-l",
            layer,
            f"--minimum-zoom={minz}",
            f"--maximum-zoom={maxz}",
            "--simplification=10",
            "--coalesce-densest-as-needed",
            "--detect-shared-borders",
            "-P",
            str(src),
        ],
        check=True,
    )
    (DOCS / "tiles").mkdir(parents=True, exist_ok=True)
    pm = DOCS / "tiles" / f"{layer}.pmtiles"
    subprocess.run(["pmtiles", "convert", str(mb), str(pm)], check=True)
    return pm


def build_tiles(df: pd.DataFrame) -> pd.DataFrame:
    """Dissolve OA → LSOA → LAD, attach attributes, tile all three levels.

    Returns the OA centroid table (OA21CD, LONG, LAT) for the postcode index.
    """
    BUILD.mkdir(parents=True, exist_ok=True)
    print("  [tiles 1/4] reading OA boundaries …")
    geo = gpd.read_file(BOUNDS_PATH, columns=["OA21CD", "LAT", "LONG"])
    centroids = pd.DataFrame(geo[["OA21CD", "LONG", "LAT"]])
    geo = geo.merge(df, on="OA21CD", how="inner", validate="1:1")

    print("  [tiles 2/4] OA level …")
    oa = geo.copy()
    oa["fx"] = oa["flag_no_epc"].astype(int) + 2 * oa["flag_low_meters"].astype(int)
    oa["p"] = _wpercentile(oa["rate"], oa["total_hh"]).round(1)
    short = {
        "OA21CD": "id",
        "dominant_type": "dt",
        "total_hh": "hh",
        "avg_hh_size": "hs",
        "floor_area_m2": "fa",
        "gas_kwh_hh": "g",
        "elec_kwh_hh": "e",
        "travel_kwh_hh": "t",
        "fabric_factor": "f",
        "ev_ratio": "v",
        "access_walk": "aw",
        "access_catchment": "ac",
        "rate": "r",
        "letter_rate": "lr",
        "letter_rate_potential": "lrp",
        "letter_energy": "le",
        "letter_energy_potential": "lep",
        "letter_access": "la",
    }
    oa = oa[[*short, "p", "fx", "geometry"]].rename(columns=short)
    for col, nd in (("hs", 2), ("f", 3), ("v", 3), ("r", 4)):
        oa[col] = pd.to_numeric(oa[col], errors="coerce").round(nd)
    for col in ("fa", "g", "e", "t", "aw", "ac"):
        oa[col] = pd.to_numeric(oa[col], errors="coerce").round()
    src = BUILD / "oa.geojsonl"
    oa.to_crs(4326).to_file(src, driver="GeoJSONSeq")
    # Max zoom 12 keeps the national OA archive under the 100 MB static-host
    # file cap; MapLibre overzooms the polygons cleanly beyond that.
    _tippecanoe(src, "oa", 9, 12)

    print("  [tiles 3/4] LSOA level (dissolve) …")
    lsoa_geo = geo[["LSOA21CD", "geometry"]].dissolve(by="LSOA21CD")
    lsoa_rows = []
    for code, g in df.groupby("LSOA21CD"):
        row = _unit_row(g)
        row["id"] = code
        row["nm"] = str(g["LSOA21NM"].iloc[0])
        lsoa_rows.append(row)
    lsoa_at = pd.DataFrame(lsoa_rows).set_index("id")
    lsoa_at["p"] = _wpercentile(lsoa_at["mr"], lsoa_at["hh"]).round(1)
    lsoa = gpd.GeoDataFrame(
        lsoa_at.join(lsoa_geo), geometry="geometry", crs=geo.crs
    ).reset_index(names="id")
    src = BUILD / "lsoa.geojsonl"
    lsoa.to_crs(4326).to_file(src, driver="GeoJSONSeq")
    _tippecanoe(src, "lsoa", 7, 10)

    print("  [tiles 4/4] LAD level (dissolve) …")
    lad_key = df[["LSOA21CD", "LAD22CD", "LAD22NM"]].drop_duplicates("LSOA21CD")
    lad_geo = lsoa_geo.join(lad_key.set_index("LSOA21CD")).dissolve(by="LAD22CD")[
        ["geometry"]
    ]
    lad_rows = []
    for code, g in df.groupby("LAD22CD", observed=True):
        row = _unit_row(g)
        row["id"] = code
        row["nm"] = str(g["LAD22NM"].iloc[0])
        lad_rows.append(row)
    lad_at = pd.DataFrame(lad_rows).set_index("id")
    lad_at["p"] = _wpercentile(lad_at["mr"], lad_at["hh"]).round(1)
    lad = gpd.GeoDataFrame(
        lad_at.join(lad_geo), geometry="geometry", crs=geo.crs
    ).reset_index(names="id")
    src = BUILD / "lad.geojsonl"
    lad.to_crs(4326).to_file(src, driver="GeoJSONSeq")
    _tippecanoe(src, "lad", 4, 8)
    return centroids


def main() -> None:
    """Run the full export: aggregates, meta, tiles, postcode shards."""
    for tool in ("tippecanoe", "pmtiles"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not on PATH — install it and rerun")
    df = _load_frame()
    print(f"Atlas export — {len(df):,} scored OAs")
    print("  [1/4] aggregates.json …")
    agg = write_aggregates(df)
    print(f"    England: {agg['england']['hh']:,} households, {len(agg['lad'])} LADs")
    print("  [2/4] meta.json …")
    write_meta()
    print("  [3/4] tiles …")
    centroids = build_tiles(df)
    print("  [4/4] postcode index …")
    n = write_postcode_index(df, centroids)
    print(f"    {n:,} outcode shards")
    for pm in sorted((DOCS / "tiles").glob("*.pmtiles")):
        print(f"    {pm.name}: {pm.stat().st_size / 1e6:,.1f} MB")
    print(f"  → {DOCS}")


if __name__ == "__main__":
    main()
