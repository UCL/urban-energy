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
        Columns: ruc21_name, car_miles_per_person, pt_miles_per_person,
                 pt_rail_miles_per_person, year
        One row per 2021 rural-urban class.

``pt_miles_per_person`` sums the bus, rail, Underground and other public-transport
columns of the same table. It anchors the public-transport sensitivity in
``stats/travel_energy.py`` and does not enter the headline energy axis, which is
car travel only.
"""

import pandas as pd

from urban_energy.fetch import download_and_cache
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

# Public-transport modes, summed into one passenger-mileage figure per class for
# the public-transport sensitivity (stats/travel_energy.py). Taxi and minicab are
# excluded: those are car journeys, whose driver miles already sit in CAR_COL.
PT_COLS = [
    "Bus in London",
    "Other local bus",
    "Non-local bus",
    "London Underground",
    "Surface Rail",
    "Other public transport [note 7]",
]

# Rail-type modes within PT_COLS. Reported separately because rail carries most
# public-transport mileage and is far less energy-intensive per passenger-mile
# than road modes, which is what justifies the assumed intensity in the
# public-transport sensitivity (stats/travel_energy.py).
PT_RAIL_COLS = ["London Underground", "Surface Rail"]

# The six 2021 Rural-Urban Classification residence categories (NTS9904, 2024).
RUC21_CLASSES = {
    "Urban: Nearer to a major town or city",
    "Urban: Further from a major town or city",
    "Larger Rural: Nearer to a major town or city",
    "Larger Rural: Further from a major town or city",
    "Smaller Rural: Nearer to a major town or city",
    "Smaller Rural: Further from a major town or city",
}


def main() -> None:
    """Extract latest-year car miles/person by 2021 rural-urban class."""
    print("Downloading NTS9904 car mileage by rural-urban classification")
    path = download_and_cache(NTS9904_URL, CACHE_DIR / NTS9904_FILENAME, timeout=300)

    df = pd.read_excel(path, sheet_name=SHEET, engine="odf", header=5)
    df.columns = [str(c).strip() for c in df.columns]
    year_col, ruc_col = df.columns[0], df.columns[1]

    # Latest year reported on the 2021 (six-category) classification. Select by
    # the numeric maximum of the parsed year, not by row order: the year label
    # can be a single year or a rolling range ("2023-2024"), so parse the last
    # four-digit year and take max() to stay deterministic across re-downloads.
    df["_is_ruc21"] = df[ruc_col].astype(str).str.strip().isin(RUC21_CLASSES)
    ruc21 = df[df["_is_ruc21"]].copy()
    year_num = pd.to_numeric(
        ruc21[year_col].astype(str).str.extract(r"(\d{4})")[0], errors="coerce"
    )
    if year_num.notna().any():
        latest_num = year_num.max()
        latest = ruc21[year_num == latest_num]
        latest_year = str(int(latest_num))
    else:
        # Fallback: no four-digit year parses — keep the last distinct label.
        latest_year = ruc21[year_col].dropna().astype(str).unique()[-1]
        latest = ruc21[ruc21[year_col].astype(str) == latest_year]

    missing_pt = [c for c in PT_COLS if c not in latest.columns]
    if missing_pt:
        msg = f"NTS9904 sheet {SHEET} is missing public-transport columns: {missing_pt}"
        raise KeyError(msg)
    pt = sum(pd.to_numeric(latest[c], errors="coerce").fillna(0.0) for c in PT_COLS)
    rail = sum(
        pd.to_numeric(latest[c], errors="coerce").fillna(0.0) for c in PT_RAIL_COLS
    )

    out = pd.DataFrame(
        {
            "ruc21_name": latest[ruc_col].astype(str).str.strip(),
            "car_miles_per_person": pd.to_numeric(latest[CAR_COL], errors="coerce"),
            "pt_miles_per_person": pt,
            "pt_rail_miles_per_person": rail,
            "year": latest_year,
        }
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "nts_mileage_by_ruc.parquet"
    out.to_parquet(out_path, index=False)
    print(f"  Year: {latest_year}  ({len(out)} rural-urban classes)")
    print(f"    {'class':<48s} {'car':>7s} {'public transport':>17s}")
    for _, r in out.iterrows():
        print(
            f"    {r['ruc21_name']:<48s} {r['car_miles_per_person']:>7.0f} "
            f"{r['pt_miles_per_person']:>17.0f}"
        )
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
