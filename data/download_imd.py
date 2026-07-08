"""
Download English Indices of Deprivation 2025 (IoD25 / IMD25).

Downloads the all-in-one CSV (File 7) containing ranks, scores, deciles, and
population denominators for all 33,755 English LSOAs on 2021 boundaries.

Data source: https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025
Publisher: Ministry of Housing, Communities and Local Government (MHCLG)
Published: 2025-10-30
License: UK Open Government Licence (OGL)

Key features:
- 7 domains: Income, Employment, Education, Health, Crime, Barriers, Living Environment
- 6 sub-domains: Children & Young People, Adult Skills, Geographical Barriers,
  Wider Barriers, Indoors, Outdoors
- 2 supplementary indices: IDACI, IDAOPI
- Population denominators (mid-2022 estimates)
- Uses 2021 Census LSOA boundaries and 2024 LAD codes
- NOT directly comparable to IoD 2019 (new indicators and methodology)

Output:
    - temp/statistics/lsoa_imd2025.parquet
        All 56 columns with standardised names (snake_case, prefixed imd_).
"""

import re
from io import BytesIO

import pandas as pd

from urban_energy.fetch import download_and_cache
from urban_energy.paths import CACHE_DIR as _CACHE_ROOT
from urban_energy.paths import DATA_DIR
from urban_energy.text import england_code_mask

OUTPUT_DIR = DATA_DIR / "statistics"
CACHE_DIR = _CACHE_ROOT / "imd"

# IoD 2025 File 7: All Ranks, Scores, Deciles, Population Denominators
IMD_URL = (
    "https://assets.publishing.service.gov.uk/media/"
    "691ded56d140bbbaa59a2a7d/"
    "File_7_IoD2025_All_Ranks_Scores_Deciles_Population_Denominators.csv"
)
IMD_FILENAME = "IoD2025_File7_All_Ranks_Scores_Deciles.csv"


def _standardise_column_name(col: str) -> str:
    """
    Convert IoD column names to snake_case with imd_ prefix.

    Parameters
    ----------
    col : str
        Original column name from the CSV.

    Returns
    -------
    str
        Standardised column name.

    Examples
    --------
    >>> _standardise_column_name("LSOA code (2021)")
    'LSOA21CD'
    >>> _standardise_column_name("Index of Multiple Deprivation (IMD) Score")
    'imd_score'
    >>> _standardise_column_name("Income Score (rate)")
    'imd_income_score'
    """
    col = col.strip()

    # Direct mappings for identifier columns
    if col == "LSOA code (2021)":
        return "LSOA21CD"
    if col == "LSOA name (2021)":
        return "LSOA21NM"
    if col == "Local Authority District code (2024)":
        return "LAD24CD"
    if col == "Local Authority District name (2024)":
        return "LAD24NM"

    # Population denominators
    if "Total population" in col:
        return "imd_pop_total"
    if "Dependent Children" in col:
        return "imd_pop_children_0_15"
    if "Older population" in col:
        return "imd_pop_older_60plus"
    if "Working age" in col:
        return "imd_pop_working_age"

    # Build snake_case name for domain/index columns
    name = col.lower()

    # Remove parenthetical content like "(rate)", "(IMD)", "(IDACI)", "(IDAOPI)"
    # but keep descriptive text
    name = name.replace("(rate)", "")
    name = name.replace("(imd)", "")
    name = name.replace("(idaci)", "")
    name = name.replace("(idaopi)", "")

    # Clarifying replacements
    name = name.replace("index of multiple deprivation", "overall")
    name = name.replace("income deprivation affecting children index", "idaci")
    name = name.replace("income deprivation affecting older people", "idaopi")
    name = name.replace("education, skills and training", "education")
    name = name.replace("health deprivation and disability", "health")
    name = name.replace("barriers to housing and services", "barriers")
    name = name.replace("living environment", "living_env")
    name = name.replace("children and young people", "children_young_people")
    name = name.replace("adult skills", "adult_skills")
    name = name.replace("geographical barriers", "geo_barriers")
    name = name.replace("wider barriers", "wider_barriers")
    name = name.replace("sub-domain", "")
    name = name.replace("where 1 is most deprived 10% of lsoas", "")
    name = name.replace("where 1 is most deprived", "")

    # Remove all parenthetical content (closed or truncated)
    name = re.sub(r"\([^)]*\)?", "", name)

    # Clean up whitespace and special characters
    name = name.strip()
    name = name.replace("  ", " ")
    name = name.replace(" ", "_")
    name = name.replace("-", "_")
    # Collapse multiple underscores
    while "__" in name:
        name = name.replace("__", "_")
    name = name.strip("_")

    return f"imd_{name}"


def parse_imd_data(content: bytes) -> pd.DataFrame:
    """
    Parse the IoD 2025 File 7 CSV.

    Parameters
    ----------
    content : bytes
        Raw CSV file content.

    Returns
    -------
    pd.DataFrame
        Parsed IMD data with standardised column names, filtered to England.
    """
    df = pd.read_csv(BytesIO(content), encoding="utf-8-sig")
    print(f"  Raw CSV: {len(df):,} rows, {len(df.columns)} columns")

    # Standardise column names
    rename_map = {col: _standardise_column_name(col) for col in df.columns}
    df = df.rename(columns=rename_map)

    # Validate LSOA column exists
    if "LSOA21CD" not in df.columns:
        raise ValueError(
            f"LSOA21CD column not found after renaming. "
            f"Columns: {list(df.columns)[:10]}..."
        )

    # Filter to England only (should already be England-only but be safe)
    df["LSOA21CD"] = df["LSOA21CD"].astype(str).str.strip()
    england_mask = england_code_mask(df["LSOA21CD"])
    n_before = len(df)
    df = df[england_mask].copy()
    if len(df) < n_before:
        n_removed = n_before - len(df)
        print(f"  Filtered to England: {len(df):,} LSOAs (removed {n_removed:,})")
    else:
        print(f"  All {len(df):,} LSOAs are English")

    # Coerce numeric columns
    skip_cols = {"LSOA21CD", "LSOA21NM", "LAD24CD", "LAD24NM"}
    for col in df.columns:
        if col not in skip_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def main() -> None:
    """Download and process English Indices of Deprivation 2025."""
    print("=" * 60)
    print("English Indices of Deprivation 2025 (IoD25)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download
    print("\n[1/3] Downloading IoD 2025 File 7...")
    imd_path = download_and_cache(IMD_URL, CACHE_DIR / IMD_FILENAME, timeout=300)
    content = imd_path.read_bytes()

    # 2. Parse
    print("\n[2/3] Parsing data...")
    imd_df = parse_imd_data(content)

    # 3. Save
    print("\n[3/3] Saving output...")
    output_path = OUTPUT_DIR / "lsoa_imd2025.parquet"
    imd_df.to_parquet(output_path, index=False)
    print(f"  Saved {len(imd_df):,} LSOAs to {output_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("Columns:")
    for col in imd_df.columns:
        dtype = imd_df[col].dtype
        if dtype in ("float64", "int64"):
            valid = imd_df[col].notna().sum()
            print(f"  {col}: {dtype}, {valid:,} valid")
        else:
            print(f"  {col}: {dtype}")
    print("\n7 domains: Income, Employment, Education, Health,")
    print("  Crime, Barriers, Living Environment")
    print("6 sub-domains: Children & Young People, Adult Skills,")
    print("  Geo Barriers, Wider Barriers, Indoors, Outdoors")
    print("2 supplementary: IDACI, IDAOPI")
    print("=" * 60)


if __name__ == "__main__":
    main()
