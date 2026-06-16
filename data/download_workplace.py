"""
Download Census 2021 workplace population (jobs) per Output Area → points.

Access to **employment** is the largest single travel driver and is otherwise
missing from the access axis. This builds a job-location point layer for the
cityseer accessibility step: one point per OA at its centroid, carrying the
workplace-population count (the number of people whose workplace is that OA).

Source: NOMIS table **WP101EW** ("Population (Workplace population)", dataset
``NM_1300_1``) at 2011 Output Area level — the finest geography available; 2021
OAs are mostly unchanged 2011 OAs, so the code join is near-complete. OA geometry
is reused from ``census_oa_joined.gpkg`` (no extra boundary download).

Caveat: Census 2021 fell during a COVID lockdown, so more respondents than usual
coded "works from home", modestly under-recording people at a workplace address.
The measure still captures the *spatial pattern* of where jobs are.

Output:
    $DATA_DIR/employment/workplace_jobs.gpkg  (points, EPSG:27700; column `jobs`)
"""

from __future__ import annotations

import io

import geopandas as gpd
import pandas as pd
import requests

from urban_energy.paths import CACHE_DIR, DATA_DIR

# WP101EW workplace population, 2011 OA (TYPE299), count measure (20100).
NOMIS_URL = (
    "https://www.nomisweb.co.uk/api/v01/dataset/NM_1300_1.data.csv"
    "?geography=TYPE299&measures=20100&select=geography_code,obs_value"
)
CENSUS_OA = DATA_DIR / "statistics" / "census_oa_joined.gpkg"
OUTPUT_PATH = DATA_DIR / "employment" / "workplace_jobs.gpkg"
_CACHE = CACHE_DIR / "workplace" / "wp101ew_oa.csv"


#: NOMIS caps each response at 25,000 rows; page through with recordoffset.
_PAGE = 25000


def _fetch_jobs() -> pd.DataFrame:
    """Workplace population per 2011 OA (England), cached. Paged (NOMIS 25k cap)."""
    if _CACHE.exists():
        df = pd.read_csv(_CACHE)
    else:
        print("  downloading WP101EW (workplace population) from NOMIS (paged)…")
        frames, offset = [], 0
        while True:
            url = f"{NOMIS_URL}&recordlimit={_PAGE}&recordoffset={offset}"
            r = requests.get(
                url, timeout=300, headers={"User-Agent": "urban-energy-research/1.0"}
            )
            r.raise_for_status()
            part = pd.read_csv(io.BytesIO(r.content))
            if part.empty:
                break
            frames.append(part)
            print(f"    page {offset // _PAGE + 1}: {len(part):,} rows")
            offset += _PAGE
            if len(part) < _PAGE:
                break
        df = pd.concat(frames, ignore_index=True)
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(_CACHE, index=False)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"geography_code": "OA21CD", "obs_value": "jobs"})
    df = df[df["OA21CD"].astype(str).str.startswith("E")]  # England only
    df["jobs"] = pd.to_numeric(df["jobs"], errors="coerce").fillna(0).astype(int)
    return df[["OA21CD", "jobs"]]


def main() -> None:
    """Build the job-location point layer (one weighted point per OA)."""
    print("Building Census 2021 workplace (jobs) point layer")
    jobs = _fetch_jobs()
    print(f"  {len(jobs):,} England OAs; total jobs = {jobs['jobs'].sum():,}")

    oa = gpd.read_file(CENSUS_OA, columns=["OA21CD"]).to_crs(27700)
    oa["geometry"] = oa.geometry.representative_point()
    merged = oa.merge(jobs, on="OA21CD", how="inner")
    print(
        f"  matched {len(merged):,} / {len(jobs):,} OAs to geometry "
        f"({len(merged) / len(jobs):.1%})"
    )
    merged = merged[merged["jobs"] > 0][["OA21CD", "jobs", "geometry"]]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(OUTPUT_PATH, driver="GPKG")
    print(f"  wrote {len(merged):,} job-points → {OUTPUT_PATH}")
    print(
        f"    jobs/OA: median {merged['jobs'].median():.0f}, "
        f"max {merged['jobs'].max():,}"
    )


if __name__ == "__main__":
    main()
