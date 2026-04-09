# Urban Energy Project Guidelines

## Project Overview

This project investigates the relationship between urban form (morphology, density, accessibility) and per-capita energy consumption in England. It combines geospatial data processing with academic research on sustainable urban development, targeting a research paper.

**Author:** Gareth Simons
**License:** GPL-3.0-only

## Project Structure

```
urban-energy/
├── src/urban_energy/          # Python package (paths.py centralises storage config)
├── data/                      # Data acquisition and preprocessing scripts
├── processing/                # OA-level pipeline (pipeline_oa.py)
│   └── archive/               # Archived LSOA pipeline
├── stats/                     # Statistical analysis and figure generation
│   ├── figures/oa/            # Active OA-level figures (fig1-fig8)
│   ├── figures/basket_oa/     # OA basket figures and tables
│   ├── figures/archive_lsoa/  # Archived LSOA figures
│   └── archive/               # Archived LSOA analysis scripts
├── paper/                     # Academic paper
│   └── archive/               # Archived LSOA case and stale LaTeX
├── docs/                      # Archived working notes (v0 methodology, roadmap, log)
├── tests/                     # Test suite (framework configured, tests pending)
├── temp/                      # Data, processing outputs, and caches (gitignored)
└── .claude/                   # Claude Code configuration
```

### Key entry points

| Script | Purpose |
|--------|---------|
| `stats/build_case_oa.py` | Regenerate all OA-level figures and tables |
| `processing/pipeline_oa.py` | National OA integration pipeline (7,147 BUAs) |
| `processing/process_morphology.py` | Building shape metrics from LiDAR + OS footprints |

### Canonical documents

| File | Status |
|------|--------|
| `paper/case_v2.md` | **Current** — OA-level analysis with robustness (IMRaD transition in progress) |
| `paper/data.md` | **Current** — data source methodology |
| `paper/literature_review.md` | Current — thematic literature review |
| `paper/references.bib` | Partial |
| `TODO.md` | Development status and forward plan |

### Storage layout

The base data directory is configured via `URBAN_ENERGY_DATA_DIR` in a `.env` file at the repo root (gitignored). `src/urban_energy/paths.py` loads it and exports:

- `DATA_DIR` = `$URBAN_ENERGY_DATA_DIR/data` — all datasets
- `PROCESSING_DIR` = `$URBAN_ENERGY_DATA_DIR/processing` — per-BUA pipeline outputs
- `CACHE_DIR` = `$URBAN_ENERGY_DATA_DIR/cache` — download caches
- `PROJECT_DIR` — source repo root (for `stats/figures/` outputs)

Expected layout under `$URBAN_ENERGY_DATA_DIR` (default: `./temp/`):

```text
temp/
├── cache/                                     ← download caches (CACHE_DIR)
├── processing/                                ← per-BUA pipeline outputs (PROCESSING_DIR)
│   ├── combined/oa_integrated.gpkg            ← Final integrated OA dataset (national)
│   └── {bua_name}/oa_integrated.gpkg          ← Per-BUA OA outputs
└── data/                                      ← datasets (DATA_DIR)
    ├── boundaries/built_up_areas.gpkg
    ├── lidar/building_heights.gpkg
    ├── morphology/buildings_morphology.gpkg
    ├── statistics/
    │   ├── census_oa_joined.gpkg
    │   ├── lsoa_energy_consumption.parquet    ← DESNZ metered (LSOA level)
    │   ├── postcode_energy_consumption.parquet ← DESNZ metered (postcode level)
    │   ├── lsoa_imd2025.parquet               ← IoD25 indices of deprivation
    │   ├── lsoa_vehicles.parquet              ← DVLA vehicle licensing
    │   └── lsoa_scaling.parquet
    ├── epc/epc_domestic_spatial.parquet
    ├── fsa/fsa_establishments.gpkg
    ├── transport/naptan_england.gpkg
    ├── education/gias_schools.gpkg
    ├── health/nhs_facilities.gpkg
    ├── models/nepi/                           ← trained XGBoost models
    └── stats/results/                         ← JSON analysis outputs
```

## Data Source Inventory

### Currently integrated (Case One)

| # | Source | Granularity | Script | Output | Role |
|---|--------|------------|--------|--------|------|
| 1 | Census 2021 (10 topic tables) | OA → LSOA | `data/download_census.py` | `census_oa_joined.gpkg` | Population, accommodation type (TS044), commute distance/mode, deprivation, cars, tenure |
| 2 | DESNZ metered energy (LSOA) | LSOA | `data/download_energy_stats.py` | `lsoa_energy_consumption.parquet` | **Primary DV**: domestic gas + electricity (actual metered, weather-corrected gas) |
| 3 | EPC domestic certificates | Property (UPRN) | `data/process_epc.py` | `epc_domestic_spatial.parquet` | SAP modelled energy, building fabric, construction age, floor area |
| 4 | Environment Agency LiDAR | 2m raster | `data/process_lidar.py` | `building_heights.gpkg` | nDSM building heights → S/V ratio, volume |
| 5 | OS Open Map Local | Building footprint | (manual) | `os_open_local/` | Building geometry for morphology |
| 6 | OS Open Roads | Road network | (manual) | `oproad_gpkg_gb/` | Street network for cityseer centrality |
| 7 | OS Open Greenspace | Polygon | (manual) | `opgrsp_gpkg_gb/` | Recreation/restoration trophic layer |
| 8 | OS Open UPRN | Point | (manual) | `osopenuprn_*/` | Property-level geocoding |
| 9 | OS Code-Point Open | Postcode centroid | (manual) | `codepo_gpkg_gb/` | Postcode geocoding (NHS, etc.) |
| 10 | OS Built Up Areas | Polygon | `data/process_boundaries.py` | `built_up_areas.gpkg` | 18 study city boundaries |
| 11 | FSA establishments | Point | `data/download_fsa.py` | `fsa_establishments.gpkg` | Commercial exchange trophic layer (~500k) |
| 12 | NaPTAN transport stops | Point | `data/download_naptan.py` | `naptan_england.gpkg` | Mobility trophic layer (~434k stops) |
| 13 | GIAS schools (DfE) | Point | `data/prepare_gias.py` | `gias_schools.gpkg` | Education accessibility (~25k) |
| 14 | NHS ODS facilities | Point | `data/prepare_nhs.py` | `nhs_facilities.gpkg` | Health accessibility (~24k) |
| 15 | ONS Small Area GVA + BRES | LSOA | `data/download_scaling.py` | `lsoa_scaling.parquet` | Scaling analysis (forward work) |

### New data sources (in progress)

| # | Source | Granularity | Script | Output | Role |
|---|--------|------------|--------|--------|------|
| 16 | DESNZ postcode-level energy | Postcode | `data/download_energy_postcode.py` | `postcode_energy_consumption.parquet` | Within-LSOA energy variation, link to building type |
| 17 | IoD 2025 (IMD) | LSOA (2021) | `data/download_imd.py` | `lsoa_imd2025.parquet` | 7-domain deprivation control (replaces Census TS011) |
| 18 | DVLA vehicle licensing | LSOA (2021) | `data/download_vehicles.py` | `lsoa_vehicles.parquet` | Fleet composition by fuel type, transport lock-in |

### Data sources identified but not yet integrated

| Source | Granularity | Value | Priority |
|--------|------------|-------|----------|
| Carbon & Place (Leeds/CREDS) | LSOA | Independent validation of transport energy estimates | Medium |
| ONS Energy Efficiency of Housing | LSOA | Median EPC band per LSOA (avoids raw EPC coverage bias) | Medium |
| VOA dwelling stock | LSOA | Property counts by Council Tax band, type, build period | Low |
| Sub-regional fuel poverty (LILEE) | LSOA | Fuel-poor household proportion | Low |
| NEED anonymised data | No geography | Metered consumption linked to property attributes | Reference only |
| SERL smart meter data | Household (restricted) | Half-hourly consumption | Not accessible |
| ADR UK meter-level | Property (restricted) | Individual property metered consumption | Not accessible |
| NEBULA (2025 academic) | Postcode | 242 integrated variables | To evaluate |

## Study Design

### Sample

All 7,147 English Built-Up Areas at Output Area resolution (6,687 BUAs / 198,779 OAs processed as of 2026-03-20). Each OA contains ~130 households.

### Primary stratification

`dominant_type` — plurality accommodation type from Census TS044 per OA:

- Flat-dominant: 36,502 OAs
- Terraced-dominant: 50,592 OAs
- Semi-detached-dominant: 65,986 OAs
- Detached-dominant: 45,699 OAs

### Three energy surfaces

1. **Thermal (building):** DESNZ metered gas + electricity (postcode-level, aggregated to OA)
2. **Mobility (transport):** Census commute distance (TS058) × mode (TS061) × energy intensity (road: 0.399, rail: 0.178 kWh/pkm)
3. **Accessibility (return):** Cityseer network centrality + gravity-weighted land-use counts at 800m walk

### Key results (OA level, 168,225 OAs)

The gradient widens across surfaces (Flat-dominant → Detached-dominant medians):

- Building energy: 10,755 → 15,713 kWh/hh (1.46x)
- Total energy (overall scenario): 14,906 → 24,898 kWh/hh (1.67x)
- Energy per unit access: 3,292 → 8,820 kWh/access (2.68x)

Plurality sensitivity: gradient steepens at stricter thresholds (1.84x at 60%), confirming the headline is conservative.

## Core Thesis: The Trophic Layers Framework

### The Argument

Cities are conduits that capture energy and recycle it through layers of human interaction
(Jacobs, 2000, "The Nature of Economies"). The measure of urban efficiency is not how much
energy a neighbourhood consumes, but how many transactions, connections, and transformations
that energy enables before it dissipates. This connects to Bettencourt et al.'s (2007)
urban scaling laws: cities scale superlinearly in socioeconomic output (~N^1.15) and
sublinearly in infrastructure (~N^0.85). Proximity is the mechanism.

### The Rainforest Analogy

A rainforest and a desert receive the same solar radiation per m². The difference is
**trophic depth** — how many times energy is captured and re-used before it dissipates.

- **Rainforest** (dense urban neighbourhood): Energy passes through dozens of layers —
  canopy, understorey, epiphytes, soil biome, mycorrhizal networks. Each layer captures
  energy from the one above. Thousands of species, millions of interactions.
- **Desert** (suburban sprawl): Energy hits the ground and radiates straight back out.
  One trophic level. Minimal recycling.

### The Trophic Layers (mapped to our data)

| Layer | Ecological equivalent | Urban function | Our metric |
|-------|----------------------|----------------|------------|
| **Physical substrate** | Soil/root network | Street network connectivity | `cc_harmonic_800`, `cc_density_800` |
| **Commercial exchange** | Canopy photosynthesis | Places where economic transactions happen | `cc_fsa_restaurant_800_wt`, `cc_fsa_pub_800_wt`, `cc_fsa_takeaway_800_wt` |
| **Mobility** | Seed dispersal/pollinators | Connections to wider city network | `cc_bus_800_wt`, `cc_rail_800_wt` |
| **Recreation/restoration** | Water cycle/shade | Green space — regenerative capacity | `cc_greenspace_800_wt` |

All metrics at 800m (~10 min walk), the pedestrian catchment. The `_wt` suffix means
gravity-weighted count (more establishments closer = higher score).

### The PoC Structure (stats/proof_of_concept_lsoa.py)

| Step | Test | Role |
|------|------|------|
| 1 | Building typology → building energy | Foundation |
| 2 | Neighbourhood morphology → transport energy | Gap widens |
| 3 | Neighbourhood morphology → local amenity access | Sets up compounding |
| 4 | Trip-demand schedule for selected land uses | Demand side |
| 5 | Distance allocation: local vs wider access | Supply side |
| 6 | Access delivery and penalty comparison | **Centerpiece** |
| 7 | Pattern holds within deprivation quintiles | Rules out wealth |
| 8 | Relationship across all LSOAs (distribution-wide) | Generalisability |
| 9 | Aggregate cost of sprawl (18-city sample) | Policy implication |

### Key References

- Jacobs, J. (2000). *The Nature of Economies*. Random House. — Cities as ecosystems
- Bettencourt, L.M.A. et al. (2007). "Growth, innovation, scaling, and the pace of life in cities." *PNAS*, 104(17). — Superlinear/sublinear scaling
- Norman, J. et al. (2006). "Comparing high and low residential density." *J. Urban Planning*. — Functional unit matters (per m² vs per capita)
- Newman, P. & Kenworthy, J. (1989). *Cities and Automobile Dependence*. — Density-transport energy
- Rode, P. et al. (2014). "Cities and energy: urban morphology and residential heat-energy demand." *Env. & Planning B*. — S/V ratio and building physics
- Few, J. et al. (2023). "The over-prediction of energy use by EPCs." *Energy & Buildings*. — SAP performance gap
- Ewing, R. & Cervero, R. (2010). "Travel and the built environment: A meta-analysis." *JAPA*. — Destination accessibility > density

## Development Standards

### Python Code Quality

1. **Type Annotations:** All functions must have complete type hints. Use modern Python typing (3.10+) including `|` union syntax and `list[]`/`dict[]` generics.

2. **Docstrings:** Use NumPy-style docstrings for all public functions and classes:

   ```python
   def calculate_density(geometries: gpd.GeoSeries, population: pd.Series) -> pd.Series:
       """
       Calculate population density per unit area.

       Parameters
       ----------
       geometries : gpd.GeoSeries
           Polygon geometries with projected CRS (units in meters).
       population : pd.Series
           Population counts aligned with geometries index.

       Returns
       -------
       pd.Series
           Population density in persons per square kilometer.

       Raises
       ------
       ValueError
           If CRS is geographic (unprojected) or series lengths mismatch.
       """
   ```

3. **Linting:** Code must pass `ruff check` and `ruff format`. Configuration in pyproject.toml.

4. **Type Checking:** Code must pass `ty check` (or `uv run ty check`).

5. **Testing:** Write pytest tests for all non-trivial functions. Tests should cover edge cases and validate geospatial operations with known inputs.

### Script conventions

All data scripts in `data/` follow this pattern:
- Import `DATA_DIR` and `CACHE_DIR` from `urban_energy.paths`
- Use `download_and_cache()` to avoid re-downloading
- Output parquet (tabular) or GeoPackage (spatial) under `DATA_DIR`
- Print progress with step numbers `[1/N]` and summary statistics
- Filter to England only (codes starting with "E")
- Use `requests` with `User-Agent: urban-energy-research/1.0`

### Geospatial Best Practices

1. **CRS:** EPSG:27700 (British National Grid) for all analysis. EPSG:4326 only for data interchange.
2. **Spatial Joins:** Prefer `sjoin` with explicit `predicate` parameter.
3. **Memory:** Use appropriate dtypes, process large datasets in chunks, release with `del` and `gc.collect()`.
4. **Raster:** Always use context managers with rasterio. Respect nodata values.
5. **Reproducibility:** Set random seeds, document data source versions, use deterministic ordering.

### Data Handling

1. **Provenance:** Document all data sources, licenses, and access dates.
2. **Sensitive Data:** Never commit raw data containing personal information.
3. **File Paths:** Use `pathlib.Path` for all file operations.
4. **Validation:** Check expected columns, CRS, null values, geometry validity at function boundaries.

## Academic Writing Standards

1. **Style:** Formal academic conventions. No contractions, colloquialisms, or first-person singular.
2. **Citations:** Author-date format (APA/Harvard). All claims require supporting citations.
3. **Precision:** Quantitative claims must specify units, confidence intervals, and data sources.
4. **Reproducibility:** Methods must provide sufficient detail for independent replication.
5. **Structure:** IMRaD (Introduction, Methods, Results, Discussion).
6. **Literature Review:** Critically evaluate sources thematically, not serially. Identify gaps. Prioritise peer-reviewed; note preprints explicitly.

## Commands Reference

```bash
# Development
uv sync                      # Install dependencies
uv run pytest                # Run tests
uv run ruff check .          # Lint code
uv run ruff format .         # Format code
uv run ty check              # Type check

# Data pipeline
uv run python data/download_census.py          # Census 2021
uv run python data/download_energy_stats.py    # DESNZ energy (LSOA)
uv run python data/download_energy_postcode.py # DESNZ energy (postcode)
uv run python data/download_imd.py             # IMD 2025
uv run python data/download_vehicles.py        # DVLA vehicles
uv run python data/download_fsa.py             # FSA establishments
uv run python data/download_naptan.py          # NaPTAN transport
uv run python data/download_scaling.py         # BRES + GVA

# Analysis
uv run python stats/build_case_oa.py           # Regenerate all OA-level figures

# Git workflow
git status                   # Check changes
git diff                     # Review changes
git log --oneline -10        # Recent history
```

## Code Review Checklist

Before committing, verify:

- [ ] All functions have type annotations
- [ ] Public functions have docstrings
- [ ] `ruff check .` passes
- [ ] `ruff format --check .` passes
- [ ] `ty check` passes (or type errors are intentional/documented)
- [ ] Tests pass: `uv run pytest`
- [ ] CRS handling is explicit and correct
- [ ] No hardcoded absolute paths (all storage paths derive from `URBAN_ENERGY_DATA_DIR` via `src/urban_energy/paths.py`)
- [ ] No sensitive data included
- [ ] Commit message is descriptive and follows conventional format

## Dependencies

Core geospatial stack:

- **geopandas:** Vector data operations
- **shapely:** Geometry primitives
- **rasterio:** Raster I/O and operations
- **pyproj:** CRS transformations (via geopandas)

Analysis:

- **numpy / pandas:** Numerical and tabular data
- **scipy / statsmodels:** Statistical methods and regression
- **xgboost / shap:** ML methods (forward work)
- **cityseer:** Network centrality and accessibility
- **momepy:** Building morphology metrics
- **esda / libpysal:** Spatial statistics

Visualisation:

- **seaborn:** Publication figures

I/O:

- **requests / aiohttp:** HTTP downloads
- **openpyxl:** DESNZ XLSX parsing
- **pyarrow:** Parquet I/O

Development:

- **ruff:** Linting and formatting
- **ty:** Type checking
- **pytest:** Testing framework
