"""
Prepare Get Information About Schools (GIAS) data for England.

Reads the manually downloaded GIAS establishment CSV from cache and outputs
a geocoded GeoPackage for use as the education accessibility layer in the
pedestrian catchment analysis.

Data source: https://get-information-schools.service.gov.uk/Downloads
License: UK Open Government Licence (OGL)

Manual download:
    Go to https://get-information-schools.service.gov.uk/Downloads
    Download "All establishment data (open establishments)" as CSV.
    Save to: cache/gias/  (any filename matching edubasealldata*.csv is accepted,
    e.g. edubasealldata20260226.csv, as downloaded from the GIAS website.)

Key features:
    - Authoritative DfE register of all state-funded and independent establishments
    - Includes easting/northing (OSGB36) — no geocoding step required
    - Covers schools, academies, free schools, pupil referral units, colleges
    - Monthly updates; only open establishments are included

Output:
    temp/education/gias_schools.gpkg
        Columns: urn, establishment_name, type_of_establishment,
                 phase_of_education, statutory_low_age, statutory_high_age,
                 easting, northing, geometry
"""

import datetime
from io import BytesIO

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from urban_energy.paths import CACHE_DIR as _CACHE_ROOT
from urban_energy.paths import DATA_DIR

OUTPUT_DIR = DATA_DIR / "education"
CACHE_DIR = _CACHE_ROOT / "gias"

# Establishment status codes — include only open establishments
OPEN_STATUS_CODES = {
    1,  # Open
    4,  # Open, but proposed to close
}


GIAS_URL = (
    "https://ea-edubase-api-prod.azurewebsites.net/edubase/downloads/"
    "public/edubasealldata{date}.csv"
)
_HEADERS = {"User-Agent": "urban-energy-research/1.0"}


def _download_gias() -> bytes | None:
    """
    Download the latest GIAS all-establishments CSV automatically.

    The GIAS bulk file is published daily at a date-stamped URL; this tries the
    last few days until one resolves, caching the result.

    Returns
    -------
    bytes | None
        Raw CSV content, or None if no recent date could be fetched.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today()
    for back in range(7):
        date = (today - datetime.timedelta(days=back)).strftime("%Y%m%d")
        try:
            resp = requests.get(
                GIAS_URL.format(date=date), headers=_HEADERS, timeout=300
            )
        except requests.RequestException:
            continue
        # A real CSV starts with the header row, not an HTML error page.
        if resp.status_code == 200 and resp.content[:1] not in (b"<", b""):
            out = CACHE_DIR / f"edubasealldata{date}.csv"
            out.write_bytes(resp.content)
            print(f"  Downloaded {out.name} ({len(resp.content) / 1e6:.0f} MB)")
            return resp.content
    return None


def load_gias() -> bytes:
    """
    Load GIAS establishment data, downloading it if not already cached.

    Accepts a manually-saved fixed filename (gias_all_establishments.csv) or a
    dated filename (e.g. edubasealldata20260226.csv); if neither is present the
    latest CSV is fetched automatically from the GIAS bulk endpoint.

    Returns
    -------
    bytes
        Raw CSV content.

    Raises
    ------
    FileNotFoundError
        If no cached CSV exists and the automatic download failed.
    """
    # Accept fixed rename
    fixed_path = CACHE_DIR / "gias_all_establishments.csv"
    if fixed_path.exists():
        print(f"  Loading {fixed_path.name}")
        return fixed_path.read_bytes()

    # Accept dated filename as downloaded from GIAS (most recent if multiple)
    dated_files = sorted(CACHE_DIR.glob("edubasealldata*.csv"))
    if dated_files:
        dated_path = dated_files[-1]
        print(f"  Loading {dated_path.name}")
        return dated_path.read_bytes()

    # Nothing cached — fetch automatically.
    print("  No cached GIAS CSV; downloading from the GIAS bulk endpoint...")
    content = _download_gias()
    if content is not None:
        return content

    raise FileNotFoundError(
        f"No GIAS CSV in {CACHE_DIR} and automatic download failed.\n"
        "Download manually from:\n"
        "  https://get-information-schools.service.gov.uk/Downloads\n"
        "  → 'All establishment data (open establishments)' → CSV\n"
        f"Save to: {CACHE_DIR}/"
    )


def parse_gias(content: bytes) -> gpd.GeoDataFrame:
    """
    Parse GIAS CSV and convert to GeoDataFrame.

    Parameters
    ----------
    content : bytes
        Raw GIAS CSV content (Windows-1252 encoding).

    Returns
    -------
    gpd.GeoDataFrame
        Open schools and colleges with point geometry in EPSG:27700.
    """
    # GIAS CSVs use Windows-1252 (Latin-1 superset) encoding
    df = pd.read_csv(BytesIO(content), encoding="cp1252", low_memory=False)
    print(f"  Loaded {len(df):,} establishments")

    # Standardise column names: GIAS uses mixed case with spaces
    col_map = {}
    for col in df.columns:
        col_lower = (
            col.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
        )
        col_map[col] = col_lower
    df = df.rename(columns=col_map)

    # Find status column
    status_col = next(
        (c for c in df.columns if "establishment_status" in c and "name" not in c), None
    )
    if status_col:
        status_code = pd.to_numeric(df[status_col], errors="coerce")
        df = df[status_code.isin(OPEN_STATUS_CODES)].copy()
        print(f"  Open establishments: {len(df):,}")
    else:
        print("  Warning: establishment status column not found — keeping all records")

    # Find coordinate columns (easting/northing)
    easting_col = next((c for c in df.columns if "easting" in c), None)
    northing_col = next((c for c in df.columns if "northing" in c), None)

    if not easting_col or not northing_col:
        raise ValueError(
            f"Cannot find easting/northing columns. Available: {list(df.columns[:20])}"
        )

    df["easting"] = pd.to_numeric(df[easting_col], errors="coerce")
    df["northing"] = pd.to_numeric(df[northing_col], errors="coerce")

    # Drop records without valid coordinates
    valid_coords = df["easting"].notna() & df["northing"].notna()
    n_no_coords = (~valid_coords).sum()
    if n_no_coords > 0:
        print(f"  Dropping {n_no_coords:,} establishments without coordinates")
    df = df[valid_coords].copy()

    # Select and rename key columns
    keep = {
        "urn": "urn",
        "establishmentname": "establishment_name",
        "typeofestablishment_name": "type_of_establishment",
        "phasesofeducation_name": "phase_of_education",
        "statutorylowage": "statutory_low_age",
        "statutoryhighage": "statutory_high_age",
        "numberofpupils": "number_of_pupils",
    }
    available = {k: v for k, v in keep.items() if k in df.columns}
    output_cols = list(available.keys()) + ["easting", "northing"]
    df = df[[c for c in output_cols if c in df.columns]].rename(columns=available)

    # Build geometry
    geometry = [Point(e, n) for e, n in zip(df["easting"], df["northing"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:27700")

    return gdf


def main() -> None:
    """Prepare GIAS school data."""
    print("=" * 60)
    print("GIAS Schools — Prepare")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load
    print("\n[1/3] Loading GIAS data from cache...")
    content = load_gias()

    # 2. Parse
    print("\n[2/3] Parsing establishments...")
    gdf = parse_gias(content)
    print(f"  Final dataset: {len(gdf):,} establishments")

    if "phase_of_education" in gdf.columns:
        print("\n  By phase of education:")
        for phase, count in gdf["phase_of_education"].value_counts().items():
            print(f"    {phase}: {count:,}")

    # 3. Save
    print("\n[3/3] Saving output...")
    output_path = OUTPUT_DIR / "gias_schools.gpkg"
    gdf.to_file(output_path, driver="GPKG")
    print(f"  Saved {len(gdf):,} establishments to {output_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
