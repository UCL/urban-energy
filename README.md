# Urban Energy

Investigating the relationship between urban morphology and building energy consumption across England and Wales.

## Quick Start

```bash
uv sync                              # Install dependencies
cd data && cat README.md             # See data setup instructions
```

## Project Structure

```text
urban-energy/
├── src/urban_energy/    # Python package
├── data/                # Data download & processing scripts
├── paper/               # Academic paper (LaTeX)
├── temp/                # Downloaded data (not in git)
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

See [data/README.md](data/README.md) for download and processing instructions.

## Development Status

### Completed

- [x] Census data pipeline (`download_census.py`)
- [x] Built-up area boundary processing (`process_boundaries.py`)
- [x] EPC data processing with UPRN linkage (`process_epc.py`)
- [x] LiDAR building height extraction (`process_lidar.py`)

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
