# Statistical Analysis

LSOA-level analysis of urban form and energy consumption across 18 English cities.

---

## Prerequisites

Complete the data and processing pipelines first:

```text
1. DATA ACQUISITION      → data/README.md (10 download/prep scripts)
2. BUILDING MORPHOLOGY   → processing/process_morphology.py
3. LSOA PIPELINE         → processing/pipeline_lsoa.py
4. STATISTICAL ANALYSIS  → (this directory)
```

**Required input:** `temp/processing/combined/lsoa_integrated.gpkg`

---

## Running the Analysis

```bash
# Regenerate all case-one figures and tables
uv run python stats/build_case.py

# Or run individual scripts
uv run python stats/lsoa_figures.py           # Three-surfaces figures
uv run python stats/basket_index_v1.py        # Basket index + land-use penalty
uv run python stats/diagnostic_fig1b.py       # Confounder diagnostics

# Optionally restrict to specific cities
uv run python stats/build_case.py "Manchester" "London"
```

## Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `build_case.py` | Entry point: regenerates all case-one figures | Calls the two scripts below |
| `lsoa_figures.py` | Three energy surfaces publication figures | `figures/fig1_*` through `fig6_*`, summary CSVs |
| `basket_index_v1.py` | TCPA-aligned basket index, land-use penalty | `figures/basket_v1/` (figures + tables) |
| `proof_of_concept_lsoa.py` | Core LSOA data loading and analysis functions | Imported by the above scripts |
| `diagnostic_fig1b.py` | Confounder scatter diagnostics | Diagnostic figure |

## Output

```text
stats/figures/
├── fig1_building_energy.png          # Building energy by dominant housing type
├── fig2_mobility_penalty.png         # Building + transport energy by type
├── fig2b_private_public_transport.png # Private vs public commute decomposition
├── fig3_density_transport.png        # Density and transport energy (KDE)
├── fig4_accessibility_dividend.png   # Local accessibility network frontage
├── fig5_access_bar.png               # Accessibility components by type
├── fig6_correlation_heatmap.png      # Correlation matrix
├── table1_three_surfaces.csv         # Energy decomposition summary
├── table2_energy_decomposition.csv   # Component breakdown
└── basket_v1/
    ├── fig_basket_v1_by_type.png
    ├── fig_basket_v1_category_scores_heatmap.png
    ├── fig_basket_v1_deprivation_gradient.png
    ├── fig_basket_v1_scatter_energy_vs_basket.png
    ├── lsoa_basket_v1_scores.csv
    ├── table_basket_v1_by_type.csv
    ├── table_basket_v1_by_deprivation.csv
    └── table_basket_v1_schema.csv
```

## Analytical Framework

The analysis constructs three "energy surfaces" per LSOA, then compounds them:

1. **Building energy** (kWh/m2): DESNZ metered gas + electricity, normalised by floor area
2. **Transport energy** (kWh/capita): Census commute distance x mode-specific intensity
3. **Accessibility** (gravity-weighted count): land-use destinations within 800m walk

The basket index adds a trip-demand model (TCPA-aligned) to quantify how much of a household's routine travel can be satisfied locally, producing a "land-use access penalty" that compounds with building and transport energy.

## Key Finding

The compounding widens the efficiency gap at each normalisation level:
- kWh/m2: modest difference (~1.04x detached vs flat)
- kWh/capita: gap widens (~1.76x)
- kWh/capita/accessibility: gap widens further (~2.79x)
- With basket penalty: 3.5x between detached-dominant and flat-dominant LSOAs
