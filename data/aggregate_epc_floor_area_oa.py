"""
Aggregate EPC dwelling floor area to Output Area (robustness Step 4).

Produces a per-OA median dwelling floor area, used for the per-m² Form
normalisation (energy per unit floor area isolates thermal efficiency from
dwelling size). EPC certificates are mapped to OAs via the postcode→OA lookup
(a postcode lies within a single OA, so this is fast and sufficient for an OA
median, avoiding a national UPRN spatial join).

Inputs:
    - $DATA_DIR/epc/epc_domestic_spatial.parquet  (POSTCODE, TOTAL_FLOOR_AREA)
    - $DATA_DIR/statistics/postcode_oa_lookup.parquet  (Postcode, OA21CD)

Output:
    - $DATA_DIR/statistics/oa_floor_area.parquet
        Columns: OA21CD, oa_median_floor_area_m2, oa_n_epc_floor
"""

import pandas as pd

from urban_energy.paths import DATA_DIR

EPC_PATH = DATA_DIR / "epc" / "epc_domestic_spatial.parquet"
LOOKUP_PATH = DATA_DIR / "statistics" / "postcode_oa_lookup.parquet"
OUTPUT_PATH = DATA_DIR / "statistics" / "oa_floor_area.parquet"

# Plausible domestic floor-area range (m²); excludes data-entry errors/outliers.
MIN_FLOOR_AREA_M2 = 10
MAX_FLOOR_AREA_M2 = 1000
# Minimum EPC certificates per OA for a stable median.
MIN_EPC_PER_OA = 5


def main() -> None:
    """Aggregate EPC median floor area to OA via the postcode→OA lookup."""
    print("Aggregating EPC floor area → Output Area")

    epc = pd.read_parquet(EPC_PATH, columns=["POSTCODE", "TOTAL_FLOOR_AREA"])
    epc["POSTCODE"] = epc["POSTCODE"].astype(str).str.strip().str.upper()
    epc["TOTAL_FLOOR_AREA"] = pd.to_numeric(epc["TOTAL_FLOOR_AREA"], errors="coerce")
    epc = epc[
        epc["TOTAL_FLOOR_AREA"].between(MIN_FLOOR_AREA_M2, MAX_FLOOR_AREA_M2)
    ]
    print(f"  {len(epc):,} EPC certificates with plausible floor area")

    lookup = pd.read_parquet(LOOKUP_PATH, columns=["Postcode", "OA21CD"])
    lookup["Postcode"] = lookup["Postcode"].astype(str).str.strip().str.upper()

    merged = epc.merge(
        lookup, left_on="POSTCODE", right_on="Postcode", how="inner"
    )
    print(f"  {len(merged):,} matched to an OA ({len(merged) / len(epc):.1%})")

    oa = (
        merged.groupby("OA21CD")["TOTAL_FLOOR_AREA"]
        .agg(["median", "count"])
        .reset_index()
    )
    oa.columns = ["OA21CD", "oa_median_floor_area_m2", "oa_n_epc_floor"]
    oa = oa[oa["oa_n_epc_floor"] >= MIN_EPC_PER_OA]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    oa.to_parquet(OUTPUT_PATH, index=False)
    print(f"  Saved {len(oa):,} OAs to {OUTPUT_PATH}")
    print(
        f"  Median dwelling floor area: "
        f"{oa['oa_median_floor_area_m2'].median():.0f} m² "
        f"(IQR {oa['oa_median_floor_area_m2'].quantile(0.25):.0f}–"
        f"{oa['oa_median_floor_area_m2'].quantile(0.75):.0f})"
    )


if __name__ == "__main__":
    main()
