"""
Prepare NHS facility location data for England.

Reads manually downloaded NHS Organisation Data Service (ODS) CSV files and
produces a geocoded GeoPackage for use as the health accessibility layer in the
pedestrian catchment analysis.

Data sources:
    NHS ODS (Organisation Data Service) — https://digital.nhs.uk/services/organisation-data-service
    License: UK Open Government Licence (OGL)

Manual download:
    ODS files must be downloaded manually from the NHS Digital website and saved
    to cache/nhs_ods/ with the exact ODS filenames:

    ets.csv (hospitals/trusts):
        https://digital.nhs.uk/services/organisation-data-service/data-search-and-export/csv-downloads/other-nhs-organisations
        → "NHS Trusts and Sites"

    epraccur.csv (GP practices):
        https://digital.nhs.uk/services/organisation-data-service/data-search-and-export/csv-downloads/gp-and-gp-practice-related-data
        → "GP Practices"

    edispensary.csv (pharmacies):
        https://digital.nhs.uk/services/organisation-data-service/data-search-and-export/csv-downloads/gp-and-gp-practice-related-data
        → "Dispensaries"

Key features:
    - Authoritative NHS register of all active sites in England
    - Postcode-geocoded via Code-Point Open (included in OS Open Data)
    - Covers hospitals (trusts), GP surgeries, and community pharmacies
    - Updated monthly

Geocoding strategy:
    NHS ODS provides postcodes but not OS grid references. Coordinates are derived
    by joining ODS postcodes to the OS Code-Point Open postcode-centroid lookup,
    which ships as part of OS open data and is already available at:
        temp/codepo_gpkg_gb/

Output:
    temp/health/nhs_facilities.gpkg
        Columns: ods_code, name, facility_type, postcode, easting, northing, geometry
"""

from io import BytesIO

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from urban_energy.paths import CACHE_DIR as _CACHE_ROOT
from urban_energy.paths import TEMP_DIR

OUTPUT_DIR = TEMP_DIR / "health"
CACHE_DIR = _CACHE_ROOT / "nhs_ods"

# ODS CSV column positions (fixed-width positional — no header row)
# Reference: ODS Data Dictionary https://digital.nhs.uk/services/organisation-data-service
# Columns: ODS_Code, Name, National_Grouping, High_Level_Health_Authority,
#          Address_Line_1-5, Postcode, Open_Date, Close_Date, ...
ODS_COL_NAMES = [
    "ods_code",
    "name",
    "national_grouping",
    "high_level_health_authority",
    "address_line_1",
    "address_line_2",
    "address_line_3",
    "address_line_4",
    "address_line_5",
    "postcode",
    "open_date",
    "close_date",
    "status_code",
    "organisation_sub_type_code",
    "parent_organisation_code",
    "join_parent_date",
    "left_parent_date",
    "contact_telephone_number",
    "null1",
    "null2",
    "null3",
    "amended_record_indicator",
    "null4",
    "gor_code",
    "null5",
    "null6",
    "null7",
]


def load_ods_file(filename: str) -> bytes | None:
    """
    Load an NHS ODS CSV file from the manual download cache.

    ODS bulk files must be downloaded manually from the NHS Digital website.
    Place the downloaded CSV files in cache/nhs_ods/ using the exact ODS
    filenames (e.g. ets.csv, epraccur.csv, edispensary.csv).

    Parameters
    ----------
    filename : str
        ODS base filename without extension (e.g. "ets", "epraccur").

    Returns
    -------
    bytes | None
        Raw CSV content, or None if the file is not found.
    """
    cache_path = CACHE_DIR / f"{filename}.csv"

    if cache_path.exists():
        print(f"  Loading {filename}.csv from cache")
        return cache_path.read_bytes()

    print(
        f"  {filename}.csv not found in {CACHE_DIR}\n"
        f"  Download manually from:\n"
        f"  https://digital.nhs.uk/services/organisation-data-service"
        f"/data-search-and-export/csv-downloads\n"
        f"  Save as: {cache_path}"
    )
    return None


def parse_ods_csv(content: bytes, facility_type: str) -> pd.DataFrame:
    """
    Parse an NHS ODS CSV (no header, fixed columns) to a DataFrame.

    Parameters
    ----------
    content : bytes
        Raw CSV content from ODS file.
    facility_type : str
        Label for the facility type column (e.g. "hospital", "gp_practice").

    Returns
    -------
    pd.DataFrame
        Records with ods_code, name, postcode, facility_type columns.
        Only active (not closed) organisations are included.
    """
    n_cols = len(ODS_COL_NAMES)
    df = pd.read_csv(
        BytesIO(content),
        header=None,
        names=ODS_COL_NAMES[:n_cols],
        usecols=["ods_code", "name", "postcode", "open_date", "close_date"],
        dtype=str,
        low_memory=False,
    )

    # Keep only active organisations: open_date set, close_date empty
    df["postcode"] = df["postcode"].str.strip()
    df["close_date"] = df["close_date"].str.strip()
    active = df["close_date"].isna() | (df["close_date"] == "")
    df = df[active].copy()

    df["facility_type"] = facility_type

    return df[["ods_code", "name", "postcode", "facility_type"]].copy()


def load_postcode_lookup() -> pd.DataFrame:
    """
    Load OS Code-Point Open postcode → easting/northing lookup.

    Reads the Code-Point Open GeoPackage already downloaded to
    temp/codepo_gpkg_gb/ and extracts postcode centroids.

    Returns
    -------
    pd.DataFrame
        Columns: postcode (normalised, no space), easting, northing.
    """
    codepoint_path = TEMP_DIR / "codepo_gpkg_gb"

    gpkg_files = list(codepoint_path.glob("**/*.gpkg"))
    if not gpkg_files:
        raise FileNotFoundError(
            f"OS Code-Point Open GeoPackage not found in {codepoint_path}\n"
            "Download from: https://osdatahub.os.uk/downloads/open/CodePointOpen\n"
            "Extract to: temp/codepo_gpkg_gb/"
        )

    print(f"  Loading Code-Point Open from {gpkg_files[0]}...")
    gdf = gpd.read_file(gpkg_files[0])

    # Column names vary by release — find postcode column
    pc_col = next(
        (c for c in gdf.columns if "postcode" in c.lower() or c.upper() == "PC"),
        None,
    )
    if pc_col is None:
        raise ValueError(
            f"Cannot find postcode column in Code-Point Open. Columns: {list(gdf.columns)}"
        )

    gdf["easting"] = gdf.geometry.x
    gdf["northing"] = gdf.geometry.y

    lookup = pd.DataFrame(
        {
            "postcode": gdf[pc_col].str.upper().str.replace(" ", "", regex=False),
            "easting": gdf["easting"],
            "northing": gdf["northing"],
        }
    )
    print(f"  Loaded {len(lookup):,} postcode centroids")
    return lookup


def geocode_by_postcode(
    df: pd.DataFrame, lookup: pd.DataFrame
) -> gpd.GeoDataFrame:
    """
    Join NHS ODS records to OS postcode centroids and build GeoDataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        ODS facilities with a postcode column.
    lookup : pd.DataFrame
        Code-Point Open lookup with postcode, easting, northing.

    Returns
    -------
    gpd.GeoDataFrame
        Facilities with point geometry in EPSG:27700.
    """
    df = df.copy()
    df["_pc_key"] = df["postcode"].str.upper().str.replace(" ", "", regex=False)

    lookup_renamed = lookup.rename(columns={"postcode": "_pc_key"})
    merged = df.merge(lookup_renamed, on="_pc_key", how="left")

    n_matched = merged["easting"].notna().sum()
    n_total = len(merged)
    print(f"  Geocoded {n_matched:,}/{n_total:,} facilities by postcode")

    merged = merged[merged["easting"].notna()].copy()
    geometry = [Point(e, n) for e, n in zip(merged["easting"], merged["northing"])]

    return gpd.GeoDataFrame(
        merged[["ods_code", "name", "facility_type", "postcode", "easting", "northing"]],
        geometry=geometry,
        crs="EPSG:27700",
    )


def main() -> None:
    """Prepare NHS facility location data."""
    print("=" * 60)
    print("NHS ODS Facility Locations — Prepare")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load all ODS files from manual cache
    print("\n[1/4] Loading NHS ODS files from cache...")
    ODS_FILENAMES = {
        "hospitals": "ets",
        "gp_practices": "epraccur",
        "pharmacies": "edispensary",
    }
    frames: list[pd.DataFrame] = []
    for facility_type, filename in ODS_FILENAMES.items():
        raw = load_ods_file(filename)
        if raw is None:
            print(f"  Skipping {facility_type} (file not found)")
            continue
        df = parse_ods_csv(raw, facility_type)
        print(f"  {facility_type}: {len(df):,} active facilities")
        frames.append(df)

    if not frames:
        print(
            "\nNo ODS data loaded. Download the CSV files manually from:\n"
            "https://digital.nhs.uk/services/organisation-data-service"
            "/data-search-and-export/csv-downloads\n"
            f"and save to: {CACHE_DIR}"
        )
        return

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Total active facilities: {len(combined):,}")

    # 2. Load postcode lookup
    print("\n[2/4] Loading OS Code-Point Open postcode lookup...")
    try:
        lookup = load_postcode_lookup()
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        return

    # 3. Geocode
    print("\n[3/4] Geocoding by postcode...")
    gdf = geocode_by_postcode(combined, lookup)

    print("\n  By facility type:")
    for ftype, count in gdf["facility_type"].value_counts().items():
        print(f"    {ftype}: {count:,}")

    # 4. Save
    print("\n[4/4] Saving output...")
    output_path = OUTPUT_DIR / "nhs_facilities.gpkg"
    gdf.to_file(output_path, driver="GPKG")
    print(f"  Saved {len(gdf):,} facilities to {output_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
