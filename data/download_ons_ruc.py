"""
Download the 2021 Rural-Urban Classification of Output Areas.

Assigns each 2021 Output Area to one of six 2021 Rural-Urban Classification
residence categories (Urban / Larger Rural / Smaller Rural × Nearer / Further
from a major town or city). Used to join each OA to its NTS9904 measured
car-mileage class in the travel-energy disaggregation (`stats/travel_energy.py`).

OA21-native, so it joins directly to the integrated dataset with no boundary
re-mapping, and uses the same six categories as the NTS9904 2024 edition.

Data source: ONS Open Geography Portal — "Rural Urban Classification (2021)
    of Output Areas in EW" (service OA21_RUC21_EW_LU)
    https://www.data.gov.uk/dataset/5ef3842b-a7d3-410c-a96a-bd8e73011eac
Publisher: Office for National Statistics
License: UK Open Government Licence (OGL) v3.0

Output:
    - $DATA_DIR/statistics/oa21_ruc21.parquet
        Columns: OA21CD, RUC21NM (residence class), urban_rural_flag
"""

from pathlib import Path

import pandas as pd
import requests

from urban_energy.paths import CACHE_DIR as _CACHE_ROOT
from urban_energy.paths import DATA_DIR

OUTPUT_DIR = DATA_DIR / "statistics"
CACHE_DIR = _CACHE_ROOT / "ons_geography"

RUC21_URL = (
    "https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/"
    "items/ed33e08c81244b77a15e00545be084e1/csv?layers=0"
)
RUC21_FILENAME = "oa21_ruc21_ew_lu.csv"


def download_and_cache(url: str, filename: str) -> Path:
    """Download a file with caching; return the cached path."""
    cache_path = CACHE_DIR / filename
    if cache_path.exists():
        print(f"  Loading cached {filename} ({cache_path.stat().st_size / 1e6:.1f} MB)")
        return cache_path
    print(f"  Downloading {filename}...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "urban-energy-research/1.0"}
    resp = requests.get(url, timeout=300, headers=headers)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    print(f"  Cached to {cache_path}")
    return cache_path


def main() -> None:
    """Build the OA21 → 2021 Rural-Urban Classification lookup."""
    print("Downloading ONS 2021 Rural-Urban Classification of Output Areas")
    path = download_and_cache(RUC21_URL, RUC21_FILENAME)

    df = pd.read_csv(path, usecols=["OA21CD", "RUC21NM", "Urban_rural_flag"])
    df = df.rename(columns={"Urban_rural_flag": "urban_rural_flag"})
    df["RUC21NM"] = df["RUC21NM"].astype(str).str.strip()
    # England only, consistent with the rest of the pipeline.
    df = df[df["OA21CD"].astype(str).str.startswith("E")].reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "oa21_ruc21.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  {len(df):,} English OAs classified")
    print(df["RUC21NM"].value_counts().to_string())
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
