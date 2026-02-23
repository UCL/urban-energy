"""
Download and process DESNZ sub-national energy consumption statistics.

Downloads LSOA-level domestic gas and electricity consumption data
from the Department for Energy Security and Net Zero (DESNZ).

Data source: https://www.gov.uk/government/collections/sub-national-energy-consumption-data
License: UK Open Government Licence (OGL)

Key features:
- Actual metered consumption (not SAP-modelled like EPC)
- Gas is weather-corrected; electricity is not
- Domestic meters only at LSOA level (~33,000 LSOAs in England)
- Available 2010-2024

Output:
    - temp/statistics/lsoa_energy_consumption.parquet
        Columns: LSOA21CD, lsoa_elec_num_meters, lsoa_elec_total_kwh,
                 lsoa_elec_mean_kwh, lsoa_elec_median_kwh,
                 lsoa_gas_num_meters, lsoa_gas_total_kwh,
                 lsoa_gas_mean_kwh, lsoa_gas_median_kwh,
                 lsoa_total_mean_kwh, lsoa_gas_share
"""

from io import BytesIO

import pandas as pd
import requests
from tqdm import tqdm

from urban_energy.paths import CACHE_DIR as _CACHE_ROOT
from urban_energy.paths import TEMP_DIR

OUTPUT_DIR = TEMP_DIR / "statistics"
CACHE_DIR = _CACHE_ROOT / "desnz"

# DESNZ Sub-National Energy Statistics URLs (2010-2024 release, Dec 2025)
ELEC_URL = (
    "https://assets.publishing.service.gov.uk/media/"
    "69427b7bd8156a816c419351/"
    "LSOA_domestic_elec_2010-2024.xlsx"
)
GAS_URL = (
    "https://assets.publishing.service.gov.uk/media/"
    "694578171a2e540ccd8a5426/"
    "LSOA_domestic_gas_2010-2024.xlsx"
)

# Year to extract
TARGET_YEAR = 2023


def download_file(url: str, desc: str, timeout: int = 600) -> bytes:
    """
    Download a file with progress bar.

    Parameters
    ----------
    url : str
        URL to download.
    desc : str
        Description for progress bar.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    bytes
        Downloaded file content.
    """
    headers = {"User-Agent": "urban-energy-research/1.0"}
    response = requests.get(url, stream=True, timeout=timeout, headers=headers)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    content = BytesIO()

    with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            content.write(chunk)
            pbar.update(len(chunk))

    return content.getvalue()


def download_and_cache(url: str, filename: str) -> bytes:
    """
    Download a file with caching to avoid re-downloading.

    Parameters
    ----------
    url : str
        URL to download.
    filename : str
        Cache filename.

    Returns
    -------
    bytes
        File content (from cache or fresh download).
    """
    cache_path = CACHE_DIR / filename

    if cache_path.exists():
        print(f"  Loading cached {filename}")
        return cache_path.read_bytes()

    print(f"  Downloading {filename}...")
    content = download_file(url, filename)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(content)
    print(f"  Cached to {cache_path}")
    return content


def _find_lsoa_col(df: pd.DataFrame) -> str:
    """Find the LSOA code column by name pattern."""
    for col in df.columns:
        col_lower = str(col).lower().replace(" ", "")
        if "lsoacode" in col_lower or "lsoaCode" in col_lower:
            return col
        if col_lower in ("lsoa_code", "lsoa code", "lsoacd"):
            return col
    # Fall back to any column containing "lsoa" and "code"
    for col in df.columns:
        col_lower = str(col).lower()
        if "lsoa" in col_lower and "code" in col_lower:
            return col
    raise ValueError(f"Cannot find LSOA code column. Columns: {list(df.columns)}")


def _find_metric_cols(
    df: pd.DataFrame,
) -> dict[str, str]:
    """
    Find meter count, total, mean, and median consumption columns by pattern.

    Returns
    -------
    dict[str, str]
        Mapping from original column name to standardised suffix
        (e.g., "num_meters", "total_kwh", "mean_kwh", "median_kwh").
    """
    mapping: dict[str, str] = {}
    for col in df.columns:
        col_lower = str(col).lower()
        # Skip "non-consuming meters" — we want the main meter count
        if "non" in col_lower and "consum" in col_lower:
            continue
        if "num" in col_lower and "meter" in col_lower:
            mapping[col] = "num_meters"
        elif "total" in col_lower and ("cons" in col_lower or "kwh" in col_lower):
            mapping[col] = "total_kwh"
        elif "mean" in col_lower and ("cons" in col_lower or "kwh" in col_lower):
            mapping[col] = "mean_kwh"
        elif "median" in col_lower and ("cons" in col_lower or "kwh" in col_lower):
            mapping[col] = "median_kwh"
    return mapping


def _find_year_sheet(sheets: dict[str, pd.DataFrame], year: int) -> pd.DataFrame:
    """Find the sheet for a given year from an XLSX workbook."""
    # Try exact match first
    for name in sheets:
        if str(year) == str(name).strip():
            return sheets[name]
    # Try partial match
    for name in sheets:
        if str(year) in str(name):
            return sheets[name]
    # Fall back to last sheet
    last_name = list(sheets.keys())[-1]
    print(f"  Warning: sheet for {year} not found, using '{last_name}'")
    return sheets[last_name]


def parse_consumption_data(content: bytes, year: int, prefix: str) -> pd.DataFrame:
    """
    Parse LSOA domestic consumption from a DESNZ XLSX file.

    Parameters
    ----------
    content : bytes
        Raw XLSX file content.
    year : int
        Target year to extract.
    prefix : str
        Column prefix, e.g. "lsoa_elec" or "lsoa_gas".

    Returns
    -------
    pd.DataFrame
        Consumption data with columns: LSOA21CD, {prefix}_num_meters,
        {prefix}_total_kwh, {prefix}_mean_kwh, {prefix}_median_kwh.
    """
    print("  Reading XLSX sheets (this may take a moment)...")
    # Read with no header first to detect the actual header row
    # (DESNZ XLSX files have title rows above the real column headers)
    sheets = pd.read_excel(
        BytesIO(content), sheet_name=None, header=None, engine="openpyxl"
    )
    print(f"  Found {len(sheets)} sheets: {list(sheets.keys())}")

    df = _find_year_sheet(sheets, year)

    # Find the header row: look for a row with a SHORT cell (< 30 chars)
    # containing both "lsoa" and "code" — this distinguishes actual column
    # headers like "LSOA code" from description rows that mention
    # "LSOA code column" in a long sentence.
    header_idx = 0
    for i in range(min(10, len(df))):
        row_vals = df.iloc[i].astype(str)
        row_lower = row_vals.str.lower()
        has_short_lsoa_code = (
            row_lower.str.contains("lsoa")
            & row_lower.str.contains("code")
            & (row_lower.str.len() < 30)
        ).any()
        if has_short_lsoa_code:
            header_idx = i
            break

    # Set the detected row as column headers and drop rows above it
    df.columns = df.iloc[header_idx]
    df = df.iloc[header_idx + 1 :].reset_index(drop=True)
    # Clean column names (strip whitespace, convert to string)
    df.columns = [str(c).strip() for c in df.columns]
    print(f"  Using sheet with {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Header row index: {header_idx}")

    # Find LSOA code column
    lsoa_col = _find_lsoa_col(df)
    df = df.rename(columns={lsoa_col: "LSOA21CD"})

    # Filter to England only (LSOA codes starting with E)
    df["LSOA21CD"] = df["LSOA21CD"].astype(str).str.strip()
    df = df[df["LSOA21CD"].str.startswith("E")].copy()
    print(f"  England LSOAs: {len(df):,}")

    # Find and rename metric columns
    metric_mapping = _find_metric_cols(df)
    if not metric_mapping:
        print(f"  Warning: no metric columns matched. Available: {list(df.columns)}")
        return df[["LSOA21CD"]]

    rename_map = {orig: f"{prefix}_{suffix}" for orig, suffix in metric_mapping.items()}
    df = df.rename(columns=rename_map)

    # Keep only LSOA code and renamed metric columns
    keep_cols = ["LSOA21CD"] + list(rename_map.values())
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Coerce to numeric
    for col in df.columns:
        if col != "LSOA21CD":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  Columns: {list(df.columns)}")
    return df


def join_gas_electricity(elec_df: pd.DataFrame, gas_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join gas and electricity data on LSOA code and compute derived metrics.

    Parameters
    ----------
    elec_df : pd.DataFrame
        Electricity consumption by LSOA.
    gas_df : pd.DataFrame
        Gas consumption by LSOA (weather-corrected).

    Returns
    -------
    pd.DataFrame
        Merged data with total energy and gas share columns.
    """
    merged = elec_df.merge(gas_df, on="LSOA21CD", how="outer")

    # Derived: total mean consumption per meter (gas + electricity)
    gas_mean = merged.get("lsoa_gas_mean_kwh")
    elec_mean = merged.get("lsoa_elec_mean_kwh")
    if gas_mean is not None and elec_mean is not None:
        merged["lsoa_total_mean_kwh"] = gas_mean.fillna(0) + elec_mean.fillna(0)
        merged["lsoa_gas_share"] = gas_mean / merged["lsoa_total_mean_kwh"].replace(
            0, pd.NA
        )

    print(f"  Joined: {len(merged):,} LSOAs")
    return merged


def main() -> None:
    """Download and process DESNZ sub-national energy statistics."""
    print("=" * 60)
    print("DESNZ Sub-National Energy Statistics Download")
    print(f"Target year: {TARGET_YEAR}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download electricity data
    print("\n[1/4] Electricity consumption data...")
    elec_content = download_and_cache(ELEC_URL, "LSOA_domestic_elec_2010-2024.xlsx")

    # 2. Download gas data
    print("\n[2/4] Gas consumption data...")
    gas_content = download_and_cache(GAS_URL, "LSOA_domestic_gas_2010-2024.xlsx")

    # 3. Parse and join
    print("\n[3/4] Parsing data...")
    print("\n  --- Electricity ---")
    elec_df = parse_consumption_data(elec_content, TARGET_YEAR, "lsoa_elec")
    print("\n  --- Gas (weather-corrected) ---")
    gas_df = parse_consumption_data(gas_content, TARGET_YEAR, "lsoa_gas")

    print("\n  --- Joining ---")
    energy_df = join_gas_electricity(elec_df, gas_df)

    # 4. Save
    print("\n[4/4] Saving output...")
    output_path = OUTPUT_DIR / "lsoa_energy_consumption.parquet"
    energy_df.to_parquet(output_path, index=False)
    print(f"  Saved {len(energy_df):,} LSOAs to {output_path}")

    # Summary statistics
    print(f"\n{'=' * 60}")
    print("Summary:")
    for col in energy_df.columns:
        if col == "LSOA21CD":
            continue
        if energy_df[col].dtype in ("float64", "int64"):
            valid = energy_df[col].notna().sum()
            mean_val = energy_df[col].mean()
            print(f"  {col}: {valid:,} valid, mean = {mean_val:,.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
