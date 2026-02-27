"""
Download ONS Small Area GVA data at LSOA level.

Downloads economic output data to test the productivity side of Bettencourt
urban scaling: dense, connected urban form generates more economic value per
person, not just less energy per person.

Sources:
    - ONS Small Area GVA Estimates (NOMIS NM_2400_1)
      GVA(B) in £ million at LSOA level, 1998-2023

Note: GVA data uses 2011 LSOA boundaries. Most LSOA codes (~95%) are
unchanged between censuses, so a direct join on LSOA21CD captures the vast
majority. LSOAs that were split or merged between 2011 and 2021 will be
unmatched — acceptable for a PoC.

License: UK Open Government Licence (OGL)

Output:
    - temp/statistics/lsoa_scaling.parquet
        Columns: LSOA_CODE, gva_year, lsoa_gva_millions
"""

from io import BytesIO

import pandas as pd
import requests
from tqdm import tqdm

from urban_energy.paths import CACHE_DIR as _CACHE_ROOT
from urban_energy.paths import TEMP_DIR

OUTPUT_DIR = TEMP_DIR / "statistics"
CACHE_DIR = _CACHE_ROOT / "scaling"

# NOMIS API: ONS Small Area GVA (NM_2400_1)
# cell=0: Total GVA (balanced approach), measures=20100: Value
GVA_URL = (
    "https://www.nomisweb.co.uk/api/v01/dataset/NM_2400_1.data.csv"
    "?geography=TYPE298"
    "&date=latest"
    "&cell=0"
    "&measures=20100"
    "&select=date_name,geography_code,obs_value"
    "&recordlimit=100000"
)



def download_csv(url: str, desc: str, cache_name: str) -> pd.DataFrame:
    """
    Download CSV from NOMIS API with caching.

    Parameters
    ----------
    url : str
        NOMIS API URL returning CSV.
    desc : str
        Description for progress bar.
    cache_name : str
        Filename for local cache.

    Returns
    -------
    pd.DataFrame
        Parsed CSV data.
    """
    cache_path = CACHE_DIR / cache_name

    if cache_path.exists():
        print(f"  Loading cached {cache_name}")
        return pd.read_csv(cache_path)

    print(f"  Downloading {desc}...")
    headers = {"User-Agent": "urban-energy-research/1.0"}
    response = requests.get(url, stream=True, timeout=300, headers=headers)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    content = BytesIO()

    with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            content.write(chunk)
            pbar.update(len(chunk))

    # Cache raw CSV
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(content.getvalue())

    content.seek(0)
    return pd.read_csv(content)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise NOMIS column names to uppercase."""
    df.columns = [c.upper().strip() for c in df.columns]
    return df


def process_gva(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw GVA download into clean LSOA-level data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw NOMIS CSV output.

    Returns
    -------
    pd.DataFrame
        Clean data with columns: LSOA_CODE, gva_year, lsoa_gva_millions.
    """
    df = _normalise_columns(df)

    # Filter to England LSOAs only
    df = df[df["GEOGRAPHY_CODE"].astype(str).str.startswith("E")].copy()

    df = df.rename(
        columns={
            "GEOGRAPHY_CODE": "LSOA_CODE",
            "OBS_VALUE": "lsoa_gva_millions",
            "DATE_NAME": "gva_year",
        }
    )
    df["lsoa_gva_millions"] = pd.to_numeric(df["lsoa_gva_millions"], errors="coerce")

    print(f"  England LSOAs with GVA: {len(df):,}")
    if len(df) > 0:
        print(f"  GVA year: {df['gva_year'].iloc[0]}")
        print(f"  Mean GVA per LSOA: £{df['lsoa_gva_millions'].mean():.2f}m")
        print(
            f"  Range: £{df['lsoa_gva_millions'].min():.2f}m"
            f" – £{df['lsoa_gva_millions'].max():.2f}m"
        )
    else:
        print("  WARNING: No England LSOAs returned — check NOMIS API")

    return df[["LSOA_CODE", "gva_year", "lsoa_gva_millions"]]


def main() -> None:
    """Download and process ONS Small Area GVA data."""
    print("=" * 60)
    print("ONS Scaling Data Download (GVA)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download GVA
    print("\n[1/2] Small Area GVA estimates...")
    gva_raw = download_csv(GVA_URL, "Small Area GVA", "gva_lsoa.csv")
    gva_df = process_gva(gva_raw)

    # 2. Save
    print("\n[2/2] Saving...")
    output_path = OUTPUT_DIR / "lsoa_scaling.parquet"
    gva_df.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")
    print(f"  {len(gva_df):,} LSOAs with GVA data")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
