# Urban Energy

Investigating the relationship between urban morphology and building energy consumption across England.

## Quick Start

```bash
uv sync                              # Install dependencies
cd data && cat README.md             # See data setup instructions
```

## Project Structure

```text
urban-energy/
├── src/urban_energy/    # Python package
├── data/                # Data download & initial processing scripts
├── processing/          # Derivative processing (morphology, UPRN, network)
├── stats/               # Statistical analysis plan and scripts
├── paper/               # Academic paper (LaTeX)
├── temp/                # Downloaded/processed data (not in git)
└── tests/               # Test suite
```

## Research Questions

1. How strongly do urban morphological characteristics correlate with per-capita building energy consumption?
2. Which morphological features (density, compactness, connectivity) have the largest effect sizes?
3. Do these relationships persist after controlling for building fabric, tenure, and socio-economic factors?

## Data Sources

- **Energy Performance Certificates** - 30M+ domestic properties with energy ratings
- **Census 2021** - Population, households, tenure at Output Area level
- **OS Open Data** - Built-up areas, street networks, building footprints
- **Environment Agency LiDAR** - Building heights from DSM/DTM
- **Food Standards Agency** - Eating/drinking establishments as walkability proxy

See [data/README.md](data/README.md) for download and processing instructions.

## Statistical Approach

**Primary method:** Multi-level regression accounting for nested structure (UPRN → Building → Segment → LSOA)

**Complementary methods:**

| Method | Purpose |
| ------ | ------- |
| **Multi-level regression** | Hypothesis testing, variance decomposition, inference |
| **SHAP values** | Feature importance, interaction detection, non-linear effects |
| **Spatial regression** | Account for spatial autocorrelation |
| **Propensity matching** | Causal inference approximation |

**Key outputs:**

- Variance decomposition (building vs neighbourhood vs area level)
- Standardised effect sizes for morphology variables
- SHAP dependence plots for non-linear relationships

See [stats/README.md](stats/README.md) for full research design.

## Development Status

### Completed

- [x] Census data pipeline (`download_census.py`)
- [x] Built-up area boundary processing (`process_boundaries.py`)
- [x] EPC data processing with UPRN linkage (`process_epc.py`)
- [x] LiDAR building height extraction (`process_lidar.py`)
- [x] FSA establishments download (`download_fsa.py`)

### Next Steps

| Priority | Task                      | Description                                                     |
| -------- | ------------------------- | --------------------------------------------------------------- |
| 1        | **Morphology metrics**    | Compute building-level metrics (area, compactness, FAR, height) |
| 2        | **UPRN integration**      | Spatial join buildings → UPRNs; interpolate Census; join EPCs   |
| 3        | **Street network**        | Load OS Open Roads, build pedestrian network via cityseer       |
| 4        | **Network aggregation**   | Assign UPRNs to segments, compute 400m catchment metrics        |
| 5        | **Statistical modelling** | Multi-level regression with building, segment, LSOA effects     |
| 6        | **Paper completion**      | Results, discussion, and conclusion sections                    |

### Workflow Overview

```text
Buildings (OS Open Map Local)
    │
    ├── heights from LiDAR ✓
    ↓
Buildings + morphology metrics
    │
    ↓ spatial join
UPRNs with building attributes
    │
    ├── + Census (spatial interpolation)
    ├── + EPCs (join on UPRN)
    ↓
UPRN integrated dataset
    │
    ↓ cityseer network assignment
Street segment aggregations → Regression analysis
```

## License

GPL-3.0-only
