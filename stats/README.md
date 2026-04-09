# Statistical Analysis

OA-level analysis of urban form and energy consumption across English Built-Up Areas,
delivered as the **Neighbourhood Energy Performance Index (NEPI)** scorecard plus an
interactive XGBoost-driven planning tool.

---

## Prerequisites

Complete the data and processing pipelines first:

```text
1. DATA ACQUISITION       → data/README.md (download/prep scripts)
2. BUILDING MORPHOLOGY    → processing/process_morphology.py
3. NATIONAL OA PIPELINE   → processing/pipeline_oa.py
4. STATISTICAL ANALYSIS   → (this directory)
```

**Required input:** `$URBAN_ENERGY_DATA_DIR/processing/combined/oa_integrated.gpkg`

---

## Running the Analysis

```bash
# Regenerate all OA-level case figures and tables
uv run python stats/build_case_oa.py

# Or run individual stages
uv run python stats/oa_figures.py            # Three-surfaces figures (OA)
uv run python stats/basket_index_oa.py       # Basket index + land-use penalty (OA)

# NEPI scorecard, bands, and access penalty
uv run python stats/nepi.py                       # Scorecard, A–G bands, surface decomposition
uv run python stats/access_penalty_model.py       # Empirical OLS access-energy penalty

# NEPI planning tool: XGBoost models + interactive app
uv run python stats/nepi_model.py                 # Train four models (form / mobility / cars / commute)
uv run streamlit run stats/nepi_app.py            # Launch Streamlit interactive tool

# Optionally restrict to specific cities
uv run python stats/build_case_oa.py "Manchester" "London"
```

## Scripts

### Case figures (three energy surfaces)

| Script | Purpose | Output |
| ------ | ------- | ------ |
| `build_case_oa.py` | Entry point: regenerates case figures | Calls `oa_figures` + `basket_index_oa` |
| `oa_figures.py` | Three-surfaces publication figures | `figures/oa/fig1_*` through `fig8_*`, summary CSVs |
| `basket_index_oa.py` | Illustrative basket case: access penalty for selected land uses | `figures/basket_oa/` (figures + tables) |
| `proof_of_concept_oa.py` | Core OA data loading and aggregation functions | Imported by all scripts above |

### NEPI scorecard and access penalty

| Script | Purpose | Output |
| ------ | ------- | ------ |
| `nepi.py` | NEPI scorecard, A–G bands, surface decomposition | `figures/nepi/fig_nepi_scorecard.png`, `fig_nepi_bands.png`, `fig_nepi_radar.png` |
| `access_penalty_model.py` | Empirical OLS: transport energy ~ local coverage + controls | `figures/nepi/fig_empirical_penalty.png`, `fig_coverage_vs_transport.png` |

### NEPI planning tool

| Script | Purpose | Output |
| ------ | ------- | ------ |
| `nepi_model.py` | Train four XGBoost models (form / mobility / cars / commute) with monotonic constraints + SHAP | `$DATA_DIR/models/nepi/nepi_model_*.json`, archetype profiles, band thresholds, `figures/nepi/nepi_shap_global_*.png` |
| `nepi_app.py` | Streamlit interactive tool driven by trained models with SHAP waterfalls | (interactive) |
| `nepi_static/index.html` | Static HTML/JS planning tool — runs the same trees in the browser | mirrored to `docs/` for GitHub Pages |

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
├── nepi/
│   ├── fig_nepi_scorecard.png
│   ├── fig_nepi_bands.png
│   ├── fig_nepi_radar.png
│   ├── fig_empirical_penalty.png
│   ├── fig_coverage_vs_transport.png
│   ├── nepi_shap_global_form.png
│   └── nepi_shap_global_mobility.png
└── archive_lsoa/                              ← Archived LSOA-level figures
    ├── fig1–fig6 PNGs
    ├── table CSVs
    └── basket_v1/
```

## Analytical Framework

The analysis constructs three "energy surfaces" per OA, all in **kWh/hh/yr** so the
composite NEPI requires no arbitrary weighting:

1. **Form** (building energy): DESNZ metered gas + electricity per household
2. **Mobility** (transport energy): Census commute distance × mode-specific intensity, scaled to overall travel via NTS 2024
3. **Access** (energy penalty): empirical OLS estimate of the additional transport energy attributable to poor walkable service coverage, relative to a compact reference (85% coverage)

The composite NEPI is the sum of the three surfaces; A–G bands are assigned by national
percentile.

## Key Finding

The compounding widens the efficiency gap at each surface (median Flat → median Detached):

- Building kWh/hh: 1.46×
- Total energy kWh/hh (overall): 1.67×
- kWh per unit access: 2.68×

NEPI: median Flat-dominant OA scores **Band A** (15,982 kWh/hh/yr); median
Detached-dominant scores **Band F** (26,897 kWh/hh/yr). Gap decomposition: Form 45% /
Mobility 43% / Access penalty 14%.

## Archived LSOA Scripts

The original LSOA-level analysis scripts are preserved in `archive/`:

| Script | Purpose |
| ------ | ------- |
| `archive/build_case.py` | LSOA case-one figure entry point |
| `archive/lsoa_figures.py` | LSOA three-surfaces figures |
| `archive/basket_index_v1.py` | LSOA basket case |
| `archive/proof_of_concept_lsoa.py` | LSOA data loading and analysis |
| `archive/diagnostic_fig1b.py` | LSOA confounder diagnostics |
