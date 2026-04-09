# Processing Pipeline

Derivative processing scripts that transform raw data into the analysis-ready
national OA dataset.

**Scope:** Domestic (residential) buildings only.

**Distinction from `data/`:**

- `data/` — Download and initial processing of raw data sources
- `processing/` — Derived computations building on those outputs (national pipeline)

All paths below are relative to `$URBAN_ENERGY_DATA_DIR` (configured in `.env` at the
repo root). The `DATA_DIR` and `PROCESSING_DIR` constants are exported by
[`src/urban_energy/paths.py`](../src/urban_energy/paths.py).

---

## Scripts

| Script | Purpose | Output |
| ------ | ------- | ------ |
| `pipeline_oa.py` | National OA pipeline (CityNetwork API, all BUAs) | `$PROCESSING_DIR/combined/oa_integrated.gpkg` |
| `process_morphology.py` | Building morphology metrics from LiDAR + OS footprints | `$DATA_DIR/morphology/buildings_morphology.gpkg` |

---

## OA Pipeline

Processes all 7,147 English Built-Up Areas using the cityseer 4.25 CityNetwork API
(`from_geopandas`). 6,687 BUAs / 198,779 OAs are produced after filtering for valid data.

Three stages per BUA:

1. **Stage 1:** Building morphology (from cached LiDAR + momepy metrics)
2. **Stage 2:** Network analysis (CityNetwork: centrality + Gaussian-weighted accessibility to FSA, NaPTAN, GIAS, NHS, OS Open Greenspace)
3. **Stage 3:** OA aggregation (transient UPRN joins → meter-weighted aggregation to OA polygons; postcode energy joined via `oa_energy_consumption.parquet`)

The pipeline is **resumable**: per-BUA outputs are written under `$PROCESSING_DIR/{bua_name}/`
and skipped on re-run if present. The merged national output is regenerated each invocation.

### Running

```bash
uv run python processing/pipeline_oa.py                  # all BUAs
uv run python processing/pipeline_oa.py cambridge        # by name
uv run python processing/pipeline_oa.py E63010556        # by code
```

### Output structure

```text
$PROCESSING_DIR/
├── {bua_name}/oa_integrated.gpkg    ← per-BUA OA polygons
└── combined/oa_integrated.gpkg      ← all BUAs merged (national, ~199k OAs)
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
| `$DATA_DIR/lidar/building_heights.gpkg` | `data/process_lidar.py` |
| `$DATA_DIR/boundaries/built_up_areas.gpkg` | `data/process_boundaries.py` |

### Output Schema

`$DATA_DIR/morphology/buildings_morphology.gpkg`:

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

The script maintains a per-boundary cache in `$DATA_DIR/morphology/cache/`:

- Files named by BUA22CD (e.g., `E37000001.gpkg`)
- Delete cache files to reprocess specific boundaries
- Delete entire cache directory to reprocess all

---

## Archived LSOA Pipeline

The original LSOA-level pipeline is preserved in `archive/pipeline_lsoa.py`. It is still
imported by `pipeline_oa.py` for shared morphology constants and the Stage 1 morphology
function (`run_stage1_morphology`, `_MORPH_MEAN_COLS`, `ERA_MAP`, etc.).

---

## Relationship to Analysis

The OA pipeline produces the integrated national dataset that drives all downstream
statistical work in [stats/](../stats/README.md):

1. **Form surface** — postcode energy aggregated to OA via meter-weighted means
2. **Mobility surface** — Census 2021 commute distance × mode × ECUK intensities
3. **Access surface** — cityseer CityNetwork accessibility + empirical OLS penalty (`stats/access_penalty_model.py`)

The result is fed into the NEPI scorecard (`stats/nepi.py`) and the XGBoost planning
tool (`stats/nepi_model.py`, `stats/nepi_app.py`, `stats/nepi_static/`).

See [stats/README.md](../stats/README.md) for the analysis pipeline.
