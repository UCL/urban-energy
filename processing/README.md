# Processing Pipeline

Derivative processing scripts that transform raw data into analysis-ready datasets.

**Distinction from `data/`:**
- `data/` contains download and initial processing of raw data sources
- `processing/` contains derived computations building on those outputs

## Data Integration Workflow

The analysis unit is the **UPRN** (Unique Property Reference Number). All data sources are linked to UPRNs, which are then aggregated to street network segments via cityseer.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PREPARATION                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OS Open Map Local (buildings)                                              │
│        │                                                                    │
│        ├── heights attached via data/process_lidar.py                       │
│        │                                                                    │
│        ↓                                                                    │
│  Buildings with heights                                                     │
│        │                                                                    │
│        ↓ compute morphology metrics (process_morphology.py)                 │
│        │                                                                    │
│  Buildings with heights + morphology                                        │
│        │                                                                    │
│        ↓ spatial join (UPRN point ∈ building polygon)                       │
│        │                                                                    │
│  UPRNs inherit building attributes ──────────────────────┐                  │
│                                                          │                  │
│  Census (OA-level)                                       │                  │
│        │                                                 │                  │
│        ↓ spatial interpolation (UPRN point ∈ OA)         │                  │
│        │                                                 ↓                  │
│  UPRNs with census attributes ─────────────────────► UPRN Dataset           │
│                                                          ↑                  │
│  EPCs                                                    │                  │
│        │                                                 │                  │
│        ↓ direct join on UPRN field                       │                  │
│        │                                                 │                  │
│  UPRNs with EPC attributes ──────────────────────────────┘                  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                           NETWORK AGGREGATION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  UPRN Dataset (morphology + census + EPC)                                   │
│        │                                                                    │
│        ↓ cityseer network assignment                                        │
│        │                                                                    │
│  Street segments with aggregated metrics                                    │
│        │                                                                    │
│        ├── + FSA establishments (walkability/accessibility proxy)           │
│        │                                                                    │
│        ↓ 400m network catchment computation                                 │
│        │                                                                    │
│  Analysis-ready dataset for regression modelling                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Linkage Methods

| Source → Target         | Method                          | Notes                                                |
| ----------------------- | ------------------------------- | ---------------------------------------------------- |
| Buildings → UPRNs       | Spatial join (point-in-polygon) | Multiple UPRNs per building (flats) share morphology |
| Census → UPRNs          | Spatial interpolation           | UPRN inherits OA-level attributes                    |
| EPCs → UPRNs            | Direct join on `UPRN` field     | Post-Nov 2021 records only                           |
| FSA → Street segments   | cityseer network assignment     | Walkability proxy (eating/drinking density)          |
| UPRNs → Street segments | cityseer network assignment     | Automated via pedestrian network analysis            |

## Processing Stages

| Stage | Script                  | Input                                  | Output                                       |
| ----- | ----------------------- | -------------------------------------- | -------------------------------------------- |
| 1     | `process_morphology.py` | `temp/lidar/building_heights.gpkg`     | `temp/morphology/buildings_morphology.gpkg`  |
| 2     | `process_uprn.py`       | Buildings, Census, EPCs, OS UPRNs      | `temp/uprn/uprn_integrated.gpkg`             |
| 3     | `process_network.py`    | UPRN dataset, OS Open Roads, FSA       | `temp/network/segments_analysis.gpkg`        |

## Running Scripts

All scripts follow the same pattern:

- Process boundary-by-boundary (largest to smallest, same order as LiDAR)
- Cache results per boundary for resumability
- Support partial runs for testing (Ctrl+C safe)

```bash
# Run from project root
uv run python processing/process_morphology.py
```

---

## Output Schemas

### Building Morphology (`temp/morphology/`)

`buildings_morphology.gpkg`:

| Column                 | Type    | Description                                                    |
| ---------------------- | ------- | -------------------------------------------------------------- |
| `id`                   | string  | OS building identifier                                         |
| `geometry`             | polygon | Building footprint (EPSG:27700)                                |
| `height_*`             | float   | LiDAR-derived heights (from input)                             |
| `footprint_area_m2`    | float   | Building footprint area                                        |
| `perimeter_m`          | float   | Building perimeter                                             |
| `orientation`          | float   | Deviation from cardinal directions (0-45°, 0=N-S/E-W aligned)  |
| `convexity`            | float   | Area / convex hull area (1=simple, <1=L-shapes/courtyards)     |
| `compactness`          | float   | Circular compactness (1=circle, <1=elongated/complex)          |
| `elongation`           | float   | Shortest / longest axis ratio (1=square, <1=elongated)         |
| `shared_wall_length_m` | float   | Total shared wall length (momepy, 1.5m tolerance)              |
| `shared_wall_ratio`    | float   | shared_wall_length / perimeter (0=detached, ~0.5=terraced)     |

**Shape metrics (via momepy):**

| Metric        | Energy relevance                                                |
| ------------- | --------------------------------------------------------------- |
| `orientation` | Affects passive solar gain (N-S vs E-W orientation)             |
| `convexity`   | Indented shapes have higher surface-area-to-volume → heat loss  |
| `compactness` | Less compact = more envelope per floor area → higher heat loss  |
| `elongation`  | Elongated buildings have higher surface-area-to-volume ratio    |

### UPRN Integrated (`temp/uprn/`) - Planned

`uprn_integrated.gpkg`:

| Column                  | Type   | Source            | Description                    |
| ----------------------- | ------ | ----------------- | ------------------------------ |
| `uprn`                  | string | OS Open UPRN      | Unique Property Reference      |
| `geometry`              | point  | OS Open UPRN      | Address location (EPSG:27700)  |
| `building_id`           | string | Spatial join      | Linked building ID             |
| `footprint_area_m2`     | float  | Building          | From morphology                |
| `building_type`         | string | Building          | Inferred type                  |
| `height_median`         | float  | Building          | LiDAR height                   |
| `oa21cd`                | string | Spatial join      | Output Area code               |
| `lsoa21cd`              | string | Census lookup     | LSOA code                      |
| `population_density`    | float  | Census            | Persons per hectare            |
| `deprivation_*`         | int    | Census            | Deprivation dimensions         |
| `epc_*`                 | varies | EPC join          | Energy certificate fields      |

### Street Segments (`temp/network/`) - Planned

`segments_analysis.gpkg`:

| Column                   | Type   | Description                              |
| ------------------------ | ------ | ---------------------------------------- |
| `segment_id`             | string | Street segment identifier                |
| `geometry`               | line   | Street segment (EPSG:27700)              |
| `n_uprns`                | int    | UPRNs in 400m catchment                  |
| `building_coverage`      | float  | Σ footprint / catchment area             |
| `far`                    | float  | Σ floor area / catchment area            |
| `mean_height_m`          | float  | Mean building height                     |
| `pct_terraced`           | float  | % terraced buildings                     |
| `pct_detached`           | float  | % detached buildings                     |
| `mean_shared_wall_ratio` | float  | Mean party wall ratio                    |
| `fsa_density`            | float  | FSA establishments per hectare           |
| `betweenness`            | float  | Network centrality (cityseer)            |
| `closeness`              | float  | Network integration (cityseer)           |

---

## Caching

Each script maintains a per-boundary cache in `temp/{output}/cache/`:
- Files named by BUA22CD (e.g., `E37000001.gpkg`)
- Empty files cached to mark "no data" boundaries
- Delete cache files to reprocess specific boundaries
- Delete entire cache directory to reprocess all

---

## Dependencies

Requires outputs from `data/` scripts:

1. `data/process_boundaries.py` → `temp/boundaries/built_up_areas.gpkg`
2. `data/process_lidar.py` → `temp/lidar/building_heights.gpkg`
3. `data/download_census.py` → `temp/census_oa_joined.gpkg`
4. `data/process_epc.py` → `temp/epc_domestic_spatial.gpkg`
5. `data/download_fsa.py` → `temp/fsa/fsa_establishments.gpkg`
