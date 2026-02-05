# Processing Pipeline

Derivative processing scripts that transform raw data into analysis-ready datasets.

**Scope:** Domestic (residential) buildings only.

**Distinction from `data/`:**

- `data/` — Download and initial processing of raw data sources
- `processing/` — Derived computations building on those outputs

---

## Scripts

| Script                  | Purpose                                           | Output                                      |
| ----------------------- | ------------------------------------------------- | ------------------------------------------- |
| `process_morphology.py` | Building morphology metrics from LiDAR            | `temp/morphology/buildings_morphology.gpkg` |
| `test_pipeline.py`      | Validation pipeline (morphology + network + UPRN) | `temp/processing/test/`                     |

---

## Building Morphology Processing

Computes shape and shared-wall metrics from OS building footprints and LiDAR heights.

### Running

```bash
uv run python processing/process_morphology.py
```

### Input Requirements

| Input                                 | Source                       |
| ------------------------------------- | ---------------------------- |
| `temp/lidar/building_heights.gpkg`    | `data/process_lidar.py`      |
| `temp/boundaries/built_up_areas.gpkg` | `data/process_boundaries.py` |

### Output Schema

`temp/morphology/buildings_morphology.gpkg`:

| Column                 | Type    | Description                                                   |
| ---------------------- | ------- | ------------------------------------------------------------- |
| `id`                   | string  | OS building identifier                                        |
| `geometry`             | polygon | Building footprint (EPSG:27700)                               |
| `height_*`             | float   | LiDAR-derived heights (from input)                            |
| `footprint_area_m2`    | float   | Building footprint area                                       |
| `perimeter_m`          | float   | Building perimeter                                            |
| `orientation`          | float   | Deviation from cardinal directions (0-45°, 0=N-S/E-W aligned) |
| `convexity`            | float   | Area / convex hull area (1=simple, <1=L-shapes/courtyards)    |
| `compactness`          | float   | Circular compactness (1=circle, <1=elongated/complex)         |
| `elongation`           | float   | Longest / shortest axis ratio (1=square, >1=elongated)        |
| `shared_wall_length_m` | float   | Total shared wall length (momepy, 1.5m tolerance)             |
| `shared_wall_ratio`    | float   | shared_wall_length / perimeter (0=detached, ~0.5=terraced)    |

### Shape Metrics

| Metric        | Energy Relevance                                               |
| ------------- | -------------------------------------------------------------- |
| `orientation` | Affects passive solar gain (N-S vs E-W orientation)            |
| `convexity`   | Indented shapes have higher surface-area-to-volume → heat loss |
| `compactness` | Less compact = more envelope per floor area → higher heat loss |
| `elongation`  | Elongated buildings have higher surface-area-to-volume ratio   |

### Caching

The script maintains a per-boundary cache in `temp/morphology/cache/`:

- Files named by BUA22CD (e.g., `E37000001.gpkg`)
- Delete cache files to reprocess specific boundaries
- Delete entire cache directory to reprocess all

---

## Relationship to Analysis

The lock-in analysis in `stats/` uses:

1. **EPC data** — Building characteristics (built form, floor area, energy)
2. **Census data** — Area-level demographics (car ownership, household size)
3. **Spatial join** — EPC properties to Census OA/LSOA

The morphology processing provides supplementary building-level metrics. The core lock-in findings derive from EPC categorical variables (`BUILT_FORM`) rather than computed morphology.

See [stats/README.md](../stats/README.md) for the statistical analysis pipeline.
