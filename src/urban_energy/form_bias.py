"""
Form-surface under-recording flags (methodology TODO #6).

The Form surface uses DESNZ postcode-level *domestic* metered energy aggregated
to Output Area. Two known mechanisms bias it downward for specific OA classes:

1. **Bulk / communal gas in flats.** Blocks of flats served by a single
   non-domestic gas meter (district heating, communal boilers) record their
   gas in the non-domestic dataset, so it is absent from domestic postcode
   totals. Affected OAs read as more efficient than they are.
2. **Off-gas-grid OAs.** OAs with little or no mains-gas connection (rural, and
   some post-1990 estates) heat with oil/LPG/biomass/electric, which is not
   metered at OA granularity. Affected OAs read as having little heating load.

This module adds diagnostic columns and a composite flag so affected OAs can be
identified in the dataset, excluded or down-weighted in analysis, and called out
on the Atlas About page and in the paper. The transform is additive and
NaN-safe: missing inputs yield ``NA`` flags rather than raising.

Thresholds are heuristic; tuning them (and an EPC-based correction for the most
affected classes) is tracked as forward work in ROADMAP.md.
"""

import numpy as np
import pandas as pd

# Census TS044 accommodation-type columns (the domestic dwelling counts).
TS044_FLAT = "ts044_Accommodation type: In a purpose-built block of flats or tenement"
TS044_TOTAL = "ts044_Accommodation type: Total: All households"

# Domestic gas-meter count per OA (from aggregate_energy_oa.py).
GAS_METERS_COL = "oa_gas_num_meters"

# Heuristic thresholds (fraction of dwellings, 0..1).
OFFGAS_COVERAGE_MAX = 0.5  # < half of dwellings hold a domestic gas meter
FLAT_SHARE_MIN = 0.5  # majority-flat OA
BULKGAS_COVERAGE_MAX = 0.6  # flats + low gas coverage → likely communal/bulk gas


def compute_form_bias_flags(
    oa: pd.DataFrame,
    *,
    flat_col: str = TS044_FLAT,
    total_col: str = TS044_TOTAL,
    gas_meters_col: str = GAS_METERS_COL,
) -> pd.DataFrame:
    """
    Add Form-surface under-recording flags to an OA-level frame.

    Parameters
    ----------
    oa : pandas.DataFrame
        OA-level data, expected to carry the Census TS044 dwelling counts and
        the domestic gas-meter count. Missing columns are tolerated.
    flat_col, total_col : str
        Census TS044 flat-count and total-household-count column names.
    gas_meters_col : str
        Domestic gas-meter count column name.

    Returns
    -------
    pandas.DataFrame
        A copy of ``oa`` with these additive columns:

        ``form_flat_share`` : float
            Flats as a fraction of all households (0..1).
        ``form_gas_meter_coverage`` : float
            Domestic gas meters as a fraction of households (0..1). Values well
            below 1 indicate off-gas dwellings and/or communal gas supply.
        ``form_offgas_flag`` : boolean
            Gas-meter coverage below ``OFFGAS_COVERAGE_MAX``.
        ``form_bulkgas_flag`` : boolean
            Majority-flat OA with gas coverage below ``BULKGAS_COVERAGE_MAX``.
        ``form_underrecorded_flag`` : boolean
            Either component flag set — the headline "treat Form with caution"
            indicator. ``NA`` only when both components are undetermined.
    """
    oa = oa.copy()

    # --- Flat share ---
    if flat_col in oa.columns and total_col in oa.columns:
        total = pd.to_numeric(oa[total_col], errors="coerce")
        flat = pd.to_numeric(oa[flat_col], errors="coerce")
        oa["form_flat_share"] = flat / total.replace(0, np.nan)
    else:
        oa["form_flat_share"] = np.nan

    # --- Domestic gas-meter coverage ---
    if gas_meters_col in oa.columns and total_col in oa.columns:
        total = pd.to_numeric(oa[total_col], errors="coerce")
        gas_m = pd.to_numeric(oa[gas_meters_col], errors="coerce")
        oa["form_gas_meter_coverage"] = gas_m / total.replace(0, np.nan)
    else:
        oa["form_gas_meter_coverage"] = np.nan

    cov = oa["form_gas_meter_coverage"]
    flat_share = oa["form_flat_share"]

    # Comparisons on NaN yield False under numpy; mask back to NA where the
    # underlying inputs are undetermined so flags stay honest.
    offgas = (cov < OFFGAS_COVERAGE_MAX).astype("boolean").where(cov.notna())
    bulkgas = (
        ((flat_share >= FLAT_SHARE_MIN) & (cov < BULKGAS_COVERAGE_MAX))
        .astype("boolean")
        .where(cov.notna() & flat_share.notna())
    )

    oa["form_offgas_flag"] = offgas
    oa["form_bulkgas_flag"] = bulkgas

    both_na = offgas.isna() & bulkgas.isna()
    under = (offgas.fillna(False) | bulkgas.fillna(False)).astype("boolean")
    oa["form_underrecorded_flag"] = under.mask(both_na, pd.NA)

    return oa
