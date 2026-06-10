"""
Download National Travel Survey car mileage by rural-urban classification.

Provides the measured coarse-unit constraint for the travel-energy
disaggregation: average car/van-driver distance per person, by 2021
Rural-Urban Classification of residence. The local allocation (cars/hh,
commute distance) redistributes this measured total within each class onto
Output Areas, so OA car travel reproduces the NTS class average while varying
locally (see `stats/travel_energy.py`).

Data source: DfT National Travel Survey, table NTS9904
    https://www.gov.uk/government/statistical-data-sets/nts99-travel-by-region-and-area-type-of-residence
Publisher: Department for Transport
License: UK Open Government Licence (OGL) v3.0

The 2024 edition reports by the 2021 Rural-Urban Classification (the same six
categories as the ONS OA21→RUC21 lookup), so the join to Output Areas is exact
and 2021-native.

Output:
    - $DATA_DIR/statistics/nts_mileage_by_ruc.parquet
        Columns: ruc21_name, car_miles_per_person, year
        One row per 2021 rural-urban class.
"""

from pathlib import Path

import pandas as pd
import requests

from urban_energy.paths import CACHE_DIR as _CACHE_ROOT
from urban_energy.paths import DATA_DIR

OUTPUT_DIR = DATA_DIR / "statistics"
CACHE_DIR = _CACHE_ROOT / "nts"

NTS9904_URL = (
    "https://assets.publishing.service.gov.uk/media/"
    "68a42b19f49bec79d23d2986/nts9904.ods"
)
NTS9904_FILENAME = "nts9904.ods"
SHEET = "NTS9904b_rural_urban"
CAR_COL = "Car or van driver"

# The six 2021 Rural-Urban Classification residence categories (NTS9904, 2024).
RUC21_CLASSES = {
    "Urban: Nearer to a major town or city",
    "Urban: Further from a major town or city",
    "Larger Rural: Nearer to a major town or city",
    "Larger Rural: Further from a major town or city",
    "Smaller Rural: Nearer to a major town or city",
    "Smaller Rural: Further from a major town or city",
}


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
    """Extract latest-year car miles/person by 2021 rural-urban class."""
    print("Downloading NTS9904 car mileage by rural-urban classification")
    path = download_and_cache(NTS9904_URL, NTS9904_FILENAME)

    df = pd.read_excel(path, sheet_name=SHEET, engine="odf", header=5)
    df.columns = [str(c).strip() for c in df.columns]
    year_col, ruc_col = df.columns[0], df.columns[1]

    # Latest year reported on the 2021 (six-category) classification.
    df["_is_ruc21"] = df[ruc_col].astype(str).str.strip().isin(RUC21_CLASSES)
    ruc21 = df[df["_is_ruc21"]].copy()
    latest_year = ruc21[year_col].dropna().astype(str).unique()[-1]
    latest = ruc21[ruc21[year_col].astype(str) == latest_year]

    out = pd.DataFrame(
        {
            "ruc21_name": latest[ruc_col].astype(str).str.strip(),
            "car_miles_per_person": pd.to_numeric(latest[CAR_COL], errors="coerce"),
            "year": latest_year,
        }
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "nts_mileage_by_ruc.parquet"
    out.to_parquet(out_path, index=False)
    print(f"  Year: {latest_year}  ({len(out)} rural-urban classes)")
    for _, r in out.iterrows():
        print(f"    {r['ruc21_name']:<48s} {r['car_miles_per_person']:>7.0f}")
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
