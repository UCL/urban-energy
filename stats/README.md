# Statistical Analysis

OA-level analysis of urban form and energy consumption across English Built-Up Areas.

---

## Prerequisites

Complete the data and processing pipelines first:

```text
1. DATA ACQUISITION      → data/README.md (download/prep scripts)
2. BUILDING MORPHOLOGY   → processing/process_morphology.py
3. OA PIPELINE           → processing/pipeline_oa.py
4. STATISTICAL ANALYSIS  → (this directory)
```

**Required input:** `temp/processing/combined/oa_integrated.gpkg`

---

## Running the Analysis

```bash
# Regenerate all OA-level figures and tables
uv run python stats/build_case_oa.py

# Or run individual scripts
uv run python stats/oa_figures.py            # Three-surfaces figures (OA)
uv run python stats/basket_index_oa.py       # Basket index + land-use penalty (OA)

# Optionally restrict to specific cities
uv run python stats/build_case_oa.py "Manchester" "London"
```

## Scripts

| Script | Purpose | Output |
| ------ | ------- | ------ |
| `build_case_oa.py` | Entry point: regenerates all OA figures | Calls the two scripts below |
| `oa_figures.py` | Three energy surfaces publication figures (OA) | `figures/oa/fig1_*` through `fig8_*`, summary CSVs |
| `basket_index_oa.py` | Illustrative basket case: access penalty for selected land uses (OA) | `figures/basket_oa/` (figures + tables) |
| `proof_of_concept_oa.py` | Core OA data loading and analysis functions | Imported by the above scripts |

## Output

```text
stats/figures/
├── oa/
│   ├── fig1_building_energy.png
│   ├── fig2_mobility_penalty.png
│   ├── fig2b_private_public_transport.png
│   ├── fig3_density_transport.png
│   ├── fig4_accessibility_dividend.png
│   ├── fig5_access_bar.png
│   ├── fig6_correlation_heatmap.png
│   ├── fig7_three_surface_composite.png
│   ├── fig8_plurality_sensitivity.png
│   ├── table1_three_surfaces.csv
│   └── table2_energy_decomposition.csv
├── basket_oa/
│   ├── fig_basket_oa_by_type.png
│   ├── fig_basket_oa_category_scores_heatmap.png
│   ├── fig_basket_oa_deprivation_gradient.png
│   ├── fig_basket_oa_scatter_energy_vs_basket.png
│   ├── oa_basket_scores.csv
│   ├── table_basket_oa_by_type.csv
│   ├── table_basket_oa_by_deprivation.csv
│   └── table_basket_oa_schema.csv
└── archive_lsoa/            ← Archived LSOA-level figures (reference only)
    ├── fig1–fig6 PNGs
    ├── table CSVs
    └── basket_v1/
```

## Analytical Framework

The analysis constructs three "energy surfaces" per OA, then compounds them:

1. **Building energy** (kWh/hh): DESNZ metered gas + electricity per household
2. **Transport energy** (kWh/hh): Census commute distance x mode-specific intensity
3. **Accessibility** (gravity-weighted count): land-use destinations within 800m walk

The basket index is an illustrative case that asks: for a particular set of land uses (food retail, healthcare, education, greenspace, public transport) at their observed trip rates, how much of a household's routine travel can be satisfied locally?

## Key Finding

The compounding widens the efficiency gap at each normalisation level:

- Building kWh/hh: 1.55x (detached vs flat)
- Transport kWh/hh: 1.83x
- kWh per unit access: 2.69x

## Archived LSOA Scripts

The original LSOA-level analysis scripts are preserved in `archive/`:

| Script | Purpose |
| ------ | ------- |
| `archive/build_case.py` | LSOA case-one figure entry point |
| `archive/lsoa_figures.py` | LSOA three-surfaces figures |
| `archive/basket_index_v1.py` | LSOA basket case |
| `archive/proof_of_concept_lsoa.py` | LSOA data loading and analysis |
| `archive/diagnostic_fig1b.py` | LSOA confounder diagnostics |
