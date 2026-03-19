# Processing Pipeline

Derivative processing scripts that transform raw data into analysis-ready datasets.

**Scope:** Domestic (residential) buildings only.

**Distinction from `data/`:**

- `data/` -- Download and initial processing of raw data sources
- `processing/` -- Derived computations building on those outputs

---

## Scripts

| Script | Purpose | Output |
| ------ | ------- | ------ |
| `pipeline_oa.py` | National OA pipeline (CityNetwork API, all BUAs) | `temp/processing/combined/oa_integrated.gpkg` |
| `process_morphology.py` | Building morphology metrics from LiDAR | `temp/morphology/buildings_morphology.gpkg` |

---

## OA Pipeline

Processes all English Built-Up Areas using the CityNetwork API (cityseer 4.25+).

Three stages per BUA:

1. **Stage 1:** Building morphology (from cached LiDAR + momepy metrics)
2. **Stage 2:** Network analysis (CityNetwork: centrality + accessibility)
3. **Stage 3:** OA aggregation (transient UPRN joins -> aggregate to OA polygons)

### Running

```bash
uv run python processing/pipeline_oa.py                  # all BUAs
uv run python processing/pipeline_oa.py cambridge        # by name
uv run python processing/pipeline_oa.py E63010556        # by code
```

### Output structure

```text
temp/processing/
├── {bua_name}/oa_integrated.gpkg    -- per-BUA OA polygons
└── combined/oa_integrated.gpkg      -- all BUAs merged
```

---

## Building Morphology Processing

Computes shape and shared-wall metrics from OS building footprints and LiDAR heights.

### Running Morphology

```bash
uv run python processing/process_morphology.py
```

### Input Requirements

| Input | Source |
| ----- | ------ |
| `temp/lidar/building_heights.gpkg` | `data/process_lidar.py` |
| `temp/boundaries/built_up_areas.gpkg` | `data/process_boundaries.py` |

### Output Schema

`temp/morphology/buildings_morphology.gpkg`:

| Column | Type | Description |
| ------ | ---- | ----------- |
| `id` | string | OS building identifier |
| `geometry` | polygon | Building footprint (EPSG:27700) |
| `height_*` | float | LiDAR-derived heights (from input) |
| `footprint_area_m2` | float | Building footprint area |
| `perimeter_m` | float | Building perimeter |
| `orientation` | float | Deviation from cardinal directions (0-45 deg, 0=N-S/E-W aligned) |
| `convexity` | float | Area / convex hull area (1=simple, <1=L-shapes/courtyards) |
| `compactness` | float | Circular compactness (1=circle, <1=elongated/complex) |
| `elongation` | float | Longest / shortest axis ratio (1=square, >1=elongated) |
| `shared_wall_length_m` | float | Total shared wall length (momepy, 1.5m tolerance) |
| `shared_wall_ratio` | float | shared_wall_length / perimeter (0=detached, ~0.5=terraced) |

### Caching

The script maintains a per-boundary cache in `temp/morphology/cache/`:

- Files named by BUA22CD (e.g., `E37000001.gpkg`)
- Delete cache files to reprocess specific boundaries
- Delete entire cache directory to reprocess all

---

## Archived LSOA Pipeline

The original LSOA-level pipeline is preserved in `archive/pipeline_lsoa.py`. It is still
imported by `pipeline_oa.py` for shared constants and the Stage 1 morphology function.

---

## Relationship to Analysis

The lock-in analysis in `stats/` uses:

1. **EPC data** -- Building characteristics (built form, floor area, energy)
2. **Census data** -- Area-level demographics (car ownership, household size)
3. **Spatial join** -- EPC properties to Census OA

The morphology processing provides supplementary building-level metrics. The core lock-in findings derive from EPC categorical variables (`BUILT_FORM`) rather than computed morphology.

See [stats/README.md](../stats/README.md) for the statistical analysis pipeline.
