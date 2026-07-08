"""
Aggregate EPC dwelling attributes to Output Area.

Three OA-median attributes the two-axis analysis needs, from one pass over the
domestic EPC register. Certificates are mapped to OAs via the postcode→OA lookup
(a postcode lies within a single OA, so this is exact for an OA median and avoids
a national UPRN spatial join):

* ``oa_median_floor_area_m2`` — median dwelling floor area. Feeds the
  heat-vs-size decomposition (``stats/form_size_decomposition.py``) and the
  lock-in size counterfactual (``stats/lock_in.py``).
* ``epc_potential_kwh_m2`` — median best-practice-fabric energy intensity
  (EPC ``ENERGY_CONSUMPTION_POTENTIAL``). The lock-in "perfect insulation" basis.
* ``epc_current_kwh_m2`` — median as-is energy intensity (EPC
  ``ENERGY_CONSUMPTION_CURRENT``). With potential it gives the fabric-improvement
  ratio (potential/current) the lock-in applies to metered gas; both are EPC
  modelled, so the performance gap cancels in the ratio.
* ``oa_median_build_year`` — median construction year (from
  ``CONSTRUCTION_AGE_BAND`` midpoints). The build-age confound in the form/size
  regression.

Inputs:
    - $DATA_DIR/epc/epc_domestic_spatial.parquet
    - $DATA_DIR/statistics/postcode_oa_lookup.parquet

Output:
    - $DATA_DIR/statistics/oa_epc.parquet
"""

import re

import pandas as pd

from urban_energy.checks import assert_match_rate
from urban_energy.paths import DATA_DIR
from urban_energy.text import normalise_postcode

EPC_PATH = DATA_DIR / "epc" / "epc_domestic_spatial.parquet"
LOOKUP_PATH = DATA_DIR / "statistics" / "postcode_oa_lookup.parquet"
OUTPUT_PATH = DATA_DIR / "statistics" / "oa_epc.parquet"

# Plausible domestic ranges; exclude data-entry errors/outliers.
FLOOR_MIN_M2, FLOOR_MAX_M2 = 10, 1000
INTENSITY_MIN, INTENSITY_MAX = 10, 1000  # kWh/m²/yr
# Minimum certificates per OA for a stable floor-area median.
MIN_EPC_PER_OA = 5


def _band_to_year(band: object) -> float:
    """
    Midpoint construction year from an EPC age band.

    Closed bands ("1900-1929") return the arithmetic midpoint. Open bands return
    their single boundary year: "2007 onwards" gives 2007, and "before 1900" gives
    1900 (its upper bound, a conservative lower-bound proxy for the age of a
    pre-1900 dwelling). This preserves the old-housing signal in the build-year
    confound rather than discarding it. A band with no parseable year returns NaN.

    Parameters
    ----------
    band : object
        Raw EPC ``CONSTRUCTION_AGE_BAND`` value.

    Returns
    -------
    float
        Representative year, or NaN when no year is parseable.
    """
    years = [int(y) for y in re.findall(r"\b(1[89]\d{2}|20\d{2})\b", str(band))]
    return sum(years) / len(years) if years else float("nan")


def main() -> None:
    """Aggregate EPC floor area + best-fabric intensity + build year to OA medians."""
    print("Aggregating EPC attributes → Output Area")

    epc = pd.read_parquet(
        EPC_PATH,
        columns=[
            "POSTCODE",
            "TOTAL_FLOOR_AREA",
            "ENERGY_CONSUMPTION_CURRENT",
            "ENERGY_CONSUMPTION_POTENTIAL",
            "CONSTRUCTION_AGE_BAND",
        ],
    )
    epc["POSTCODE"] = normalise_postcode(epc["POSTCODE"], keep_space=True)
    floor = pd.to_numeric(epc["TOTAL_FLOOR_AREA"], errors="coerce")
    cur = pd.to_numeric(epc["ENERGY_CONSUMPTION_CURRENT"], errors="coerce")
    pot = pd.to_numeric(epc["ENERGY_CONSUMPTION_POTENTIAL"], errors="coerce")
    epc = epc.assign(
        floor=floor.where(floor.between(FLOOR_MIN_M2, FLOOR_MAX_M2)),
        cur=cur.where(cur.between(INTENSITY_MIN, INTENSITY_MAX)),
        pot=pot.where(pot.between(INTENSITY_MIN, INTENSITY_MAX)),
        year=epc["CONSTRUCTION_AGE_BAND"].map(_band_to_year),
    )

    lookup = pd.read_parquet(LOOKUP_PATH, columns=["Postcode", "OA21CD"])
    lookup["Postcode"] = normalise_postcode(lookup["Postcode"], keep_space=True)
    # m:1 — many certificates share a postcode, but the lookup holds one OA per
    # postcode, so the lookup cannot fan out the certificate rows.
    merged = epc.merge(
        lookup, left_on="POSTCODE", right_on="Postcode", how="inner", validate="m:1"
    )
    assert_match_rate(len(epc), len(merged), name="EPC postcode ↔ OA lookup")

    # Floor area: median over valid-floor certs, OA kept only if ≥ MIN_EPC_PER_OA.
    floor_oa = (
        merged[merged["floor"].notna()]
        .groupby("OA21CD")["floor"]
        .agg(oa_median_floor_area_m2="median", oa_n_epc_floor="size")
        .reset_index()
    )
    floor_oa = floor_oa[floor_oa["oa_n_epc_floor"] >= MIN_EPC_PER_OA]

    # Best-fabric intensity + build year: medians over valid certs (no min count).
    pot_oa = (
        merged[merged["pot"].notna()]
        .groupby("OA21CD")["pot"]
        .median()
        .reset_index()
        .rename(columns={"pot": "epc_potential_kwh_m2"})
    )
    cur_oa = (
        merged[merged["cur"].notna()]
        .groupby("OA21CD")["cur"]
        .median()
        .reset_index()
        .rename(columns={"cur": "epc_current_kwh_m2"})
    )
    year_oa = (
        merged[merged["year"].notna()]
        .groupby("OA21CD")["year"]
        .median()
        .reset_index()
        .rename(columns={"year": "oa_median_build_year"})
    )

    # Outer merge — each column has its own OA coverage. Floor area is gated by
    # MIN_EPC_PER_OA (≥5 certs), while intensity and build year keep any OA with
    # at least one valid cert. So an OA can carry an intensity/build-year value
    # but a NaN floor area (fewer than 5 floor certs). Downstream consumers must
    # treat the columns as independently populated, not row-complete.
    oa = (
        floor_oa.merge(pot_oa, on="OA21CD", how="outer", validate="1:1")
        .merge(cur_oa, on="OA21CD", how="outer", validate="1:1")
        .merge(year_oa, on="OA21CD", how="outer", validate="1:1")
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    oa.to_parquet(OUTPUT_PATH, index=False)

    print(f"  wrote {len(oa):,} OAs → {OUTPUT_PATH}")
    print("    per-column coverage (populated / total OAs):")
    for col in (
        "oa_median_floor_area_m2",
        "epc_current_kwh_m2",
        "epc_potential_kwh_m2",
        "oa_median_build_year",
    ):
        n_pop = int(oa[col].notna().sum())
        print(f"      {col:<24s} {n_pop:,} / {len(oa):,} ({n_pop / len(oa):.1%})")
    print(f"    floor median     {oa['oa_median_floor_area_m2'].median():.0f} m²")
    print(f"    current median   {oa['epc_current_kwh_m2'].median():.0f} kWh/m²/yr")
    print(f"    potential median {oa['epc_potential_kwh_m2'].median():.0f} kWh/m²/yr")
    print(f"    build year median {oa['oa_median_build_year'].median():.0f}")


if __name__ == "__main__":
    main()
