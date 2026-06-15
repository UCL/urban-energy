"""
Aggregate EPC dwelling attributes to Output Area.

Two OA-median attributes the two-axis analysis needs, from one pass over the
domestic EPC register. Certificates are mapped to OAs via the postcode→OA lookup
(a postcode lies within a single OA, so this is exact for an OA median and
avoids a national UPRN spatial join):

* ``oa_median_floor_area_m2`` — median dwelling floor area. Feeds the
  heat-vs-size decomposition (``stats/form_size_decomposition.py``) and the
  lock-in size counterfactual (``stats/lock_in.py``).
* ``epc_potential_kwh_m2`` — median best-practice-fabric energy intensity
  (EPC ``ENERGY_CONSUMPTION_POTENTIAL``). The lock-in "perfect insulation" basis.

Inputs:
    - $DATA_DIR/epc/epc_domestic_spatial.parquet
    - $DATA_DIR/statistics/postcode_oa_lookup.parquet

Output:
    - $DATA_DIR/statistics/oa_epc.parquet
        Columns: OA21CD, oa_median_floor_area_m2, epc_potential_kwh_m2,
        oa_n_epc_floor
"""

import pandas as pd

from urban_energy.paths import DATA_DIR

EPC_PATH = DATA_DIR / "epc" / "epc_domestic_spatial.parquet"
LOOKUP_PATH = DATA_DIR / "statistics" / "postcode_oa_lookup.parquet"
OUTPUT_PATH = DATA_DIR / "statistics" / "oa_epc.parquet"

# Plausible domestic ranges; exclude data-entry errors/outliers.
FLOOR_MIN_M2, FLOOR_MAX_M2 = 10, 1000
INTENSITY_MIN, INTENSITY_MAX = 10, 1000  # kWh/m²/yr
# Minimum certificates per OA for a stable floor-area median.
MIN_EPC_PER_OA = 5


def main() -> None:
    """Aggregate EPC floor area + best-fabric intensity to OA medians."""
    print("Aggregating EPC attributes → Output Area")

    epc = pd.read_parquet(
        EPC_PATH,
        columns=["POSTCODE", "TOTAL_FLOOR_AREA", "ENERGY_CONSUMPTION_POTENTIAL"],
    )
    epc["POSTCODE"] = epc["POSTCODE"].astype(str).str.strip().str.upper()
    floor = pd.to_numeric(epc["TOTAL_FLOOR_AREA"], errors="coerce")
    pot = pd.to_numeric(epc["ENERGY_CONSUMPTION_POTENTIAL"], errors="coerce")
    epc = epc.assign(
        floor=floor.where(floor.between(FLOOR_MIN_M2, FLOOR_MAX_M2)),
        pot=pot.where(pot.between(INTENSITY_MIN, INTENSITY_MAX)),
    )

    lookup = pd.read_parquet(LOOKUP_PATH, columns=["Postcode", "OA21CD"])
    lookup["Postcode"] = lookup["Postcode"].astype(str).str.strip().str.upper()
    merged = epc.merge(lookup, left_on="POSTCODE", right_on="Postcode", how="inner")
    print(f"  {len(merged):,} certificates matched to an OA")

    # Floor area: median over valid-floor certs, OA kept only if ≥ MIN_EPC_PER_OA.
    floor_oa = (
        merged[merged["floor"].notna()]
        .groupby("OA21CD")["floor"]
        .agg(oa_median_floor_area_m2="median", oa_n_epc_floor="size")
        .reset_index()
    )
    floor_oa = floor_oa[floor_oa["oa_n_epc_floor"] >= MIN_EPC_PER_OA]

    # Best-fabric intensity: median over valid-potential certs (no min count).
    pot_oa = (
        merged[merged["pot"].notna()]
        .groupby("OA21CD")["pot"]
        .median()
        .reset_index()
        .rename(columns={"pot": "epc_potential_kwh_m2"})
    )

    oa = floor_oa.merge(pot_oa, on="OA21CD", how="outer")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    oa.to_parquet(OUTPUT_PATH, index=False)

    print(f"  wrote {len(oa):,} OAs → {OUTPUT_PATH}")
    print(f"    floor median     {oa['oa_median_floor_area_m2'].median():.0f} m²")
    print(f"    potential median {oa['epc_potential_kwh_m2'].median():.0f} kWh/m²/yr")


if __name__ == "__main__":
    main()
