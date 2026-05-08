# Projection inputs for the NEPI Atlas

Hand-tabulated CSVs of fuel emission factors and prices through 2050.
Consumed by [stats/atlas/projections.py](../../stats/atlas/projections.py)
via the parquet built by [data/build_projections.py](../build_projections.py).

## Why CSV (not XLSX scraping)

NESO and DESNZ publish their projection workbooks annually, the URLs
change with each vintage, and the workbook layouts have multi-tab merged-cell
quirks that resist robotic parsing. A small CSV with **per-row source
citation** gives:

- explicit provenance — every value is auditable to a specific publication
- a clean update workflow — edit the CSV, run `build_projections.py`
- no URL-decay risk
- the same shape that automated download would produce after cleaning

## Files

- [carbon_factors.csv](carbon_factors.csv) — NESO FES (Holistic Transition + Counterfactual) + DESNZ Conversion Factors. ~4 fuels × 4 years × scenarios.
- [fuel_prices.csv](fuel_prices.csv) — DESNZ Fossil Fuel Price Assumptions + Ofgem default cap. ~4 fuels × 4 years × scenarios (incl. low / high envelope rows).

## Schema

Both files share columns:

- `year` (int) — 2025, 2030, 2040, 2050
- `scenario` (str) — `central`, `low`, `high`, `counterfactual` (where applicable)
- `fuel` (str) — `elec`, `gas`, `petrol`, `diesel`
- `metric` (str) — `kgco2_per_kwh` or `gbp_per_kwh`
- `value` (float) — the number, in stated metric units
- `source` (str) — publication citation (year, table/page, scenario name)
- `notes` (str) — optional caveats

## Authoritative source URLs (validate against these)

### NESO Future Energy Scenarios 2025

- **Publication landing page** — <https://www.neso.energy/publications/future-energy-scenarios-fes>
- **FES 2025 documents** — <https://www.neso.energy/publications/future-energy-scenarios-fes/fes-documents>
- **FES 2025 Pathways to Net Zero (full report PDF)** — <https://www.neso.energy/document/364541/download>
- **FES 2025 Economics Annex** — <https://www.neso.energy/document/374246/download>
- **FES 2025 Data Workbook** (linked from the documents page; opens directly in Excel)

Use the **Holistic Transition** central pathway as the central scenario;
**Counterfactual** as a downside sensitivity. Headline figure for 2050:
~16 gCO₂/kWh under Holistic Transition (operational generation; full
lifecycle factor incl. T&D losses is ~1.3× higher).

### DESNZ Fossil Fuel Price Assumptions 2025

Published January 2026.

- **Publication landing page** — <https://www.gov.uk/government/publications/fossil-fuel-price-assumptions-2025>
- **Final report (PDF)** — <https://assets.publishing.service.gov.uk/media/696939b3448fedc1eb424870/fossil-fuel-price-assumptions-2025.pdf>
- **Workbook (XLSX)** — <https://assets.publishing.service.gov.uk/media/69693a291c8a70fc0a3b0434/Fossil_Fuel_Price_Assumptions_2025_publication.xlsx>
- **Collection (all years)** — <https://www.gov.uk/government/collections/fossil-fuel-price-assumptions>

Use the **central case**. Low/High scenarios are present in the same
workbook and already exposed in [fuel_prices.csv](fuel_prices.csv) as
extra rows — frontend doesn't yet show them but the data is available.

### Supporting sources

- **DESNZ "UK Government GHG Conversion Factors for Company Reporting"** (annual) —
  <https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024>
  Used for: per-fuel domestic emission factors (gas, petrol, diesel — combustion physics, near-constant across vintages).
- **DUKES (Digest of UK Energy Statistics)** —
  <https://www.gov.uk/government/statistics/digest-of-uk-energy-statistics-dukes-2024>
  Used for: per-litre energy density and emission factors for petrol/diesel.
- **Ofgem default tariff cap** —
  <https://www.ofgem.gov.uk/decision/changes-default-tariff-cap>
  Used for: domestic retail electricity and gas unit rates (baseline year).
- **DESNZ Quarterly Energy Prices, table QEP 4.1.1** —
  <https://www.gov.uk/government/statistical-data-sets/monthly-and-annual-prices-of-road-fuels-and-petroleum-products>
  Used for: pump prices (petrol/diesel).

## Validation status

Values currently in the CSVs are **best-effort indicative** — they reflect
the *shape* of the published trajectories (steep grid decarbonisation,
near-constant gas/diesel emissions, modest price drift) but have not been
pinned to specific cell references in the source workbooks.

Specific known refinements pending:

- 2030/2040 electricity gCO₂/kWh under Holistic Transition central — needs
  validation against FES 2025 workbook (currently linearly approximated)
- All price values — needs validation against FFPA 2025 workbook tabs
- Gas price escalation in late years — speculative; FFPA 2025 may show
  different shape

The schema is authoritative. Update the values when a researcher walks
through the source PDFs/workbooks.

## Update workflow

1. Open the relevant publication PDF/workbook from the URLs above
2. Locate the central-pathway table (FES) or central-case price assumption (FFPA)
3. Update `value` for affected rows in the appropriate CSV
4. Update `source` to cite the new publication (vintage, table, page)
5. Run `uv run python data/build_projections.py` to rebuild the parquet
6. Re-run `uv run python stats/export_atlas_data.py greater_manchester` to regenerate the Atlas summary with new factors
7. Commit both the CSVs and the new parquet
