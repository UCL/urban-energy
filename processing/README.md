# Processing Pipeline

Derivative processing scripts that transform raw data into analysis-ready datasets.

**Distinction from `data/`:**

- `data/` contains download and initial processing of raw data sources
- `processing/` contains derived computations building on those outputs

## Data Integration Workflow

The analysis unit is the **UPRN** (Unique Property Reference Number). Processing follows three stages:

1. **Building Morphology** - Compute shape metrics on building footprints
2. **Network Analysis** - Compute network-centric accessibility and centrality metrics
3. **UPRN Integration** - Link all attributes to UPRNs as the atomic unit of analysis

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       STAGE 1: BUILDING MORPHOLOGY                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OS Open Map Local (buildings) + LiDAR heights                              │
│        │                                                                    │
│        ↓ compute morphology metrics (process_morphology.py)                 │
│        │                                                                    │
│  Buildings with heights + morphology ──────────────────────────────────┐    │
│                                                                        │    │
├────────────────────────────────────────────────────────────────────────│────┤
│                       STAGE 2: NETWORK ANALYSIS                        │    │
├────────────────────────────────────────────────────────────────────────│────┤
│                                                                        │    │
│  OS Open Roads (street network)                                        │    │
│        │                                                               │    │
│        ↓ cityseer network construction                                 │    │
│        │                                                               │    │
│        ├── compute centrality (betweenness, closeness)                 │    │
│        │                                                               │    │
│        ├── compute accessibility to green spaces                       │    │
│        │                                                               │    │
│        ├── compute accessibility to FSA establishments (by category)   │    │
│        │                                                               │    │
│        └── compute accessibility to transport nodes                    │    │
│                                                                        │    │
│  Street segments with network metrics ─────────────────────────────┐   │    │
│                                                                    │   │    │
├────────────────────────────────────────────────────────────────────│───│────┤
│                       STAGE 3: UPRN INTEGRATION                    │   │    │
├────────────────────────────────────────────────────────────────────│───│────┤
│                                                                    │   │    │
│  OS Open UPRN (address points)                                     │   │    │
│        │                                                           │   │    │
│        ├── spatial join to buildings ◄─────────────────────────────│───┘    │
│        │   (UPRN point ∈ building polygon → morphology)            │        │
│        │                                                           │        │
│        ├── spatial join to Census OA                               │        │
│        │   (UPRN point ∈ OA polygon → demographics)                │        │
│        │                                                           │        │
│        ├── direct join to EPC on UPRN field                        │        │
│        │   (UPRN = UPRN → energy performance)                      │        │
│        │                                                           │        │
│        └── nearest street segment ◄────────────────────────────────┘        │
│            (UPRN → closest segment → network metrics)                       │
│                                                                             │
│  UPRN Dataset (morphology + census + EPC + network context)                 │
│        │                                                                    │
│        ↓                                                                    │
│  Analysis-ready dataset for regression modelling                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Linkage Methods

| Source → Target            | Method                          | Notes                                                |
| -------------------------- | ------------------------------- | ---------------------------------------------------- |
| Buildings → UPRNs          | Spatial join (point-in-polygon) | Multiple UPRNs per building (flats) share morphology |
| Census OA → UPRNs          | Spatial join (point-in-polygon) | UPRN inherits OA-level attributes                    |
| EPCs → UPRNs               | Direct join on `UPRN` field     | Post-Nov 2021 records only                           |
| Street segments → UPRNs    | Nearest segment                 | UPRN inherits network metrics from closest street    |
| Green spaces → Segments    | cityseer accessibility          | Network distance to nearest green space              |
| FSA establishments → Segs  | cityseer accessibility          | Accessibility by food category                       |
| Transport nodes → Segments | cityseer accessibility          | Accessibility to bus stops, rail stations            |

## Processing Stages

| Stage | Script                  | Input                                       | Output                                      |
| ----- | ----------------------- | ------------------------------------------- | ------------------------------------------- |
| 1     | `process_morphology.py` | `temp/lidar/building_heights.gpkg`          | `temp/morphology/buildings_morphology.gpkg` |
| 2     | `process_network.py`    | OS Open Roads, Green spaces, FSA, Transport | `temp/network/segments_metrics.gpkg`        |
| 3     | `process_uprn.py`       | Buildings, Census, EPCs, OS UPRNs, Segments | `temp/uprn/uprn_integrated.gpkg`            |

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
| `elongation`           | float   | Shortest / longest axis ratio (1=square, <1=elongated)        |
| `shared_wall_length_m` | float   | Total shared wall length (momepy, 1.5m tolerance)             |
| `shared_wall_ratio`    | float   | shared_wall_length / perimeter (0=detached, ~0.5=terraced)    |

**Shape metrics (via momepy):**

| Metric        | Energy relevance                                               |
| ------------- | -------------------------------------------------------------- |
| `orientation` | Affects passive solar gain (N-S vs E-W orientation)            |
| `convexity`   | Indented shapes have higher surface-area-to-volume → heat loss |
| `compactness` | Less compact = more envelope per floor area → higher heat loss |
| `elongation`  | Elongated buildings have higher surface-area-to-volume ratio   |

### Street Segments (`temp/network/`) - Planned

`segments_metrics.gpkg`:

Network-centric metrics only. No aggregated building/UPRN attributes.

Walking time thresholds (cityseer converts to network distances):

- **Centrality:** 10, 20, 40, 60, 120 minutes
- **Accessibility:** 5, 10, 20, 60 minutes

| Column                     | Type   | Description                                        |
| -------------------------- | ------ | -------------------------------------------------- |
| `segment_id`               | string | Street segment identifier                          |
| `geometry`                 | line   | Street segment (EPSG:27700)                        |
| **Centrality**             |        |                                                    |
| `betweenness_{t}`          | float  | Betweenness centrality (t = 10, 20, 40, 60, 120 min) |
| `closeness_{t}`            | float  | Closeness centrality (t = 10, 20, 40, 60, 120 min)   |
| **Green space access**     |        |                                                    |
| `green_nearest_min`        | float  | Walking time to nearest green space                |
| `green_count_{t}`          | int    | Green spaces within t (5, 10, 20, 60 min)          |
| `green_area_{t}`           | float  | Total green space area (m²) within t               |
| **FSA accessibility**      |        |                                                    |
| `fsa_restaurant_{t}`       | int    | Restaurants within t (5, 10, 20, 60 min)           |
| `fsa_takeaway_{t}`         | int    | Takeaways within t                                 |
| `fsa_retail_{t}`           | int    | Food retail within t                               |
| `fsa_total_{t}`            | int    | All FSA establishments within t                    |
| **Transport access**       |        |                                                    |
| `bus_stop_nearest_min`     | float  | Walking time to nearest bus stop                   |
| `bus_stops_{t}`            | int    | Bus stops within t (5, 10, 20, 60 min)             |
| `rail_station_nearest_min` | float  | Walking time to nearest rail station               |
| `rail_stations_{t}`        | int    | Rail stations within t                             |

### UPRN Integrated (`temp/uprn/`) - Planned

`uprn_integrated.gpkg`:

Final analysis-ready dataset with all attributes linked to individual properties.

| Column                     | Type   | Source          | Description                            |
| -------------------------- | ------ | --------------- | -------------------------------------- |
| **Identity**               |        |                 |                                        |
| `uprn`                     | string | OS Open UPRN    | Unique Property Reference              |
| `geometry`                 | point  | OS Open UPRN    | Address location (EPSG:27700)          |
| **Building (Stage 1)**     |        |                 |                                        |
| `building_id`              | string | Spatial join    | Linked building ID                     |
| `footprint_area_m2`        | float  | Building        | From morphology                        |
| `height_median`            | float  | Building        | LiDAR height                           |
| `compactness`              | float  | Building        | Shape compactness                      |
| `shared_wall_ratio`        | float  | Building        | Party wall ratio                       |
| `orientation`              | float  | Building        | Cardinal deviation                     |
| **Census**                 |        |                 |                                        |
| `oa21cd`                   | string | Spatial join    | Output Area code                       |
| `lsoa21cd`                 | string | Census lookup   | LSOA code                              |
| `population_density`       | float  | Census          | Persons per hectare                    |
| `deprivation_*`            | int    | Census          | Deprivation dimensions                 |
| **EPC**                    |        |                 |                                        |
| `epc_rating`               | string | EPC join        | Energy rating (A-G)                    |
| `epc_*`                    | varies | EPC join        | Energy certificate fields              |
| **Network (Stage 2)**      |        |                 |                                        |
| `segment_id`               | string | Nearest segment | Linked street segment                  |
| `betweenness_{t}`          | float  | Segment         | Street centrality (10, 20, 40, 60, 120 min) |
| `closeness_{t}`            | float  | Segment         | Street integration                     |
| `green_nearest_min`        | float  | Segment         | Walking time to green space            |
| `fsa_total_{t}`            | int    | Segment         | Food accessibility (5, 10, 20, 60 min) |
| `bus_stop_nearest_min`     | float  | Segment         | Walking time to bus stop               |
| `rail_station_nearest_min` | float  | Segment         | Walking time to rail station           |

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

| Script                        | Output                                  | Used by         |
| ----------------------------- | --------------------------------------- | --------------- |
| `data/process_boundaries.py`  | `temp/boundaries/built_up_areas.gpkg`   | All stages      |
| `data/process_lidar.py`       | `temp/lidar/building_heights.gpkg`      | Stage 1         |
| `data/download_census.py`     | `temp/census_oa_joined.gpkg`            | Stage 3         |
| `data/process_epc.py`         | `temp/epc_domestic_spatial.parquet`     | Stage 3         |
| `data/download_fsa.py`        | `temp/fsa/fsa_establishments.gpkg`      | Stage 2         |
| `data/download_greenspace.py` | `temp/greenspace/greenspace.gpkg`       | Stage 2 (TODO)  |
| `data/download_transport.py`  | `temp/transport/naptan.gpkg`            | Stage 2 (TODO)  |
| OS Open Roads                 | `temp/roads/open_roads.gpkg`            | Stage 2 (TODO)  |
| OS Open UPRN                  | `temp/uprn/os_open_uprn.parquet`        | Stage 3         |

---

## Next Steps

Once the UPRN integrated dataset is complete (`temp/uprn/uprn_integrated.gpkg`):

**Proceed to [stats/README.md](../stats/README.md)** for statistical analysis:

1. **Exploratory analysis** - Descriptive statistics, correlation matrix, spatial maps
2. **Multi-level modelling** - Variance decomposition, morphology effect sizes
3. **SHAP analysis** - Feature importance, interaction detection
4. **Robustness checks** - Different scales, property subsets, spatial regression

The analysis-ready dataset contains all variables needed for the statistical workflow:

- Dependent: energy intensity (kWh/m²) from EPC
- Independent: building morphology metrics from Stage 1
- Controls: neighbourhood context from Stage 2, socio-demographics from Census
