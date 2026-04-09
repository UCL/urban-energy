"""
Build a postcode-to-OA lookup by spatial joining Code-Point Open to OA boundaries.

Produces the same mapping as the ONS NSPL (National Statistics Postcode Lookup)
but using data already present in the pipeline: OS Code-Point Open postcode
centroids and Census 2021 OA boundary polygons.

Inputs (must already exist):
    - temp/codepo_gpkg_gb/  — OS Code-Point Open (postcode centroids, EPSG:27700)
    - temp/statistics/census_oa_joined.gpkg — OA boundaries with OA21CD, LSOA21CD

Output:
    - temp/statistics/postcode_oa_lookup.parquet
        Columns: Postcode, OA21CD, LSOA21CD
"""

import geopandas as gpd
import pandas as pd

from urban_energy.paths import DATA_DIR

OUTPUT_DIR = DATA_DIR / "statistics"

# Input paths
CODEPOINT_DIR = DATA_DIR / "codepo_gpkg_gb"
CENSUS_OA_PATH = DATA_DIR / "statistics" / "census_oa_joined.gpkg"


def _find_codepoint_gpkg() -> str:
    """Find the Code-Point Open GeoPackage file in the directory."""
    matches = list(CODEPOINT_DIR.glob("*.gpkg"))
    if not matches:
        # Try nested Data/ directory (OS packaging varies)
        matches = list(CODEPOINT_DIR.glob("**/*.gpkg"))
    if not matches:
        msg = (
            f"No GeoPackage found in {CODEPOINT_DIR}. "
            "Download Code-Point Open from https://osdatahub.os.uk/downloads/open/CodePointOpen"
        )
        raise FileNotFoundError(msg)
    return str(matches[0])


def load_codepoint_postcodes() -> gpd.GeoDataFrame:
    """
    Load Code-Point Open postcode centroids, filtered to England.

    Returns
    -------
    gpd.GeoDataFrame
        Point geometries with 'Postcode' column in normalised format
        (uppercase, stripped, with space).
    """
    gpkg_path = _find_codepoint_gpkg()
    print(f"  Loading Code-Point Open from {gpkg_path}...")
    postcodes = gpd.read_file(gpkg_path)
    print(f"  Loaded {len(postcodes):,} postcodes (all UK)")

    # Identify the postcode column (varies by Code-Point version)
    pc_col = None
    for candidate in ["Postcode", "postcode", "POSTCODE", "PCD"]:
        if candidate in postcodes.columns:
            pc_col = candidate
            break
    if pc_col is None:
        # Fall back to first string column
        for col in postcodes.columns:
            if postcodes[col].dtype == "object" and col != "geometry":
                pc_col = col
                break
    if pc_col is None:
        cols = list(postcodes.columns)
        raise ValueError(f"Cannot find postcode column. Columns: {cols}")

    postcodes = postcodes.rename(columns={pc_col: "Postcode"})
    postcodes["Postcode"] = postcodes["Postcode"].str.strip().str.upper()

    # Filter to England postcodes (exclude Scottish and Welsh areas)
    scottish_areas = {
        "AB",
        "DD",
        "DG",
        "EH",
        "FK",
        "G",
        "HS",
        "IV",
        "KA",
        "KW",
        "KY",
        "ML",
        "PA",
        "PH",
        "TD",
        "ZE",
    }
    welsh_areas = {"CF", "LD", "LL", "NP", "SA"}
    exclude = scottish_areas | welsh_areas
    area = postcodes["Postcode"].str.extract(r"^([A-Z]{1,2})", expand=False)
    england_mask = ~area.isin(exclude)
    n_before = len(postcodes)
    postcodes = postcodes[england_mask].copy()
    n_removed = n_before - len(postcodes)
    print(f"  Filtered to England: {len(postcodes):,} (removed {n_removed:,})")

    # Keep only Postcode and geometry
    postcodes = postcodes[["Postcode", "geometry"]].copy()

    # Ensure CRS is EPSG:27700
    if postcodes.crs is None:
        postcodes = postcodes.set_crs(epsg=27700)
    elif postcodes.crs.to_epsg() != 27700:
        postcodes = postcodes.to_crs(epsg=27700)

    return postcodes


def load_oa_boundaries() -> gpd.GeoDataFrame:
    """
    Load OA boundaries with OA21CD and LSOA21CD.

    Returns
    -------
    gpd.GeoDataFrame
        OA polygons with 'OA21CD' and 'LSOA21CD' columns.
    """
    if not CENSUS_OA_PATH.exists():
        msg = (
            f"Census OA boundaries not found at {CENSUS_OA_PATH}. "
            "Run: uv run python data/download_census.py"
        )
        raise FileNotFoundError(msg)

    print(f"  Loading OA boundaries from {CENSUS_OA_PATH}...")
    oa = gpd.read_file(CENSUS_OA_PATH, columns=["OA21CD", "LSOA21CD", "geometry"])
    print(f"  Loaded {len(oa):,} OAs")
    return oa


def build_lookup(
    postcodes: gpd.GeoDataFrame,
    oa_boundaries: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Spatial join postcode centroids to OA boundaries.

    Parameters
    ----------
    postcodes : gpd.GeoDataFrame
        Postcode point centroids with 'Postcode' column.
    oa_boundaries : gpd.GeoDataFrame
        OA polygon boundaries with 'OA21CD' and 'LSOA21CD' columns.

    Returns
    -------
    pd.DataFrame
        Lookup with columns: Postcode, OA21CD, LSOA21CD.
    """
    print("  Spatial joining postcodes to OA boundaries...")
    joined = gpd.sjoin(
        postcodes,
        oa_boundaries[["OA21CD", "LSOA21CD", "geometry"]],
        how="left",
        predicate="within",
    )

    # Drop duplicates (boundary postcodes may match multiple OAs)
    n_before = len(joined)
    joined = joined.drop_duplicates(subset=["Postcode"], keep="first")
    n_dups = n_before - len(joined)
    if n_dups > 0:
        print(f"  Removed {n_dups:,} boundary duplicates")

    # Report match rate
    matched = joined["OA21CD"].notna().sum()
    unmatched = joined["OA21CD"].isna().sum()
    print(f"  Matched: {matched:,} ({matched / len(joined):.1%})")
    if unmatched > 0:
        print(f"  Unmatched: {unmatched:,} (postcodes outside OA boundaries)")

    # Keep only matched postcodes
    lookup = joined[joined["OA21CD"].notna()][["Postcode", "OA21CD", "LSOA21CD"]].copy()
    lookup = lookup.reset_index(drop=True)

    return lookup


def main() -> None:
    """Build the postcode-to-OA lookup table."""
    print("=" * 60)
    print("Build Postcode → OA Lookup (spatial join)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("\n[1/3] Loading postcode centroids...")
    postcodes = load_codepoint_postcodes()

    print("\n[2/3] Loading OA boundaries...")
    oa_boundaries = load_oa_boundaries()

    # 3. Spatial join
    print("\n[3/3] Building lookup...")
    lookup = build_lookup(postcodes, oa_boundaries)

    # Save
    output_path = OUTPUT_DIR / "postcode_oa_lookup.parquet"
    lookup.to_parquet(output_path, index=False)
    print(f"\n  Saved {len(lookup):,} postcodes to {output_path}")

    # Summary
    n_oas = lookup["OA21CD"].nunique()
    n_lsoas = lookup["LSOA21CD"].nunique()
    print(f"  Covers {n_oas:,} OAs, {n_lsoas:,} LSOAs")
    print("=" * 60)


if __name__ == "__main__":
    main()
