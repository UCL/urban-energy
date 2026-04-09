# Urban Energy Project Guidelines

## Project Overview

This project investigates the relationship between urban form (morphology, density, accessibility) and per-capita energy consumption in England. It combines a national geospatial pipeline with statistical analysis and an interactive planning tool, all targeting a peer-reviewed research paper.

The deliverable is the **Neighbourhood Energy Performance Index (NEPI)** — a place-level rating analogous to a building EPC, computed for every English Output Area from open data, and exposed both as a research scorecard and as an interactive XGBoost-driven planning tool.

**Author:** Gareth Simons
**License:** GPL-3.0-only
**GitHub:** [UCL/urban-energy](https://github.com/UCL/urban-energy) — live tool at [UCL.github.io/urban-energy](https://UCL.github.io/urban-energy/)

## Project Structure

```
urban-energy/
├── src/urban_energy/          # Python package (paths.py centralises storage config)
├── data/                      # Data acquisition and preprocessing scripts
├── processing/                # National OA pipeline (pipeline_oa.py, CityNetwork API)
│   └── archive/               # Archived LSOA pipeline (still imported for shared morphology constants)
├── stats/                     # Statistical analysis, NEPI scorecard, planning tool
│   ├── nepi.py                # NEPI scorecard (Form / Mobility / Access)
│   ├── access_penalty_model.py # Empirical OLS access-energy penalty
│   ├── nepi_model.py          # XGBoost training (form / mobility / cars / commute)
│   ├── nepi_app.py            # Streamlit planning tool
│   ├── nepi_static/           # Static HTML planning tool (no Python)
│   ├── figures/oa/            # Active OA-level case figures (fig1-fig8)
│   ├── figures/basket_oa/     # OA basket figures and tables
│   ├── figures/nepi/          # NEPI scorecard, bands, empirical penalty figures
│   ├── figures/archive_lsoa/  # Archived LSOA figures
│   └── archive/               # Archived LSOA analysis scripts
├── paper/                     # Academic paper (case_v2.md is canonical)
│   └── archive/               # Archived LSOA case and stale LaTeX
├── docs/                      # GitHub Pages site — mirror of stats/nepi_static/
├── notes/                     # Archived v0 working notes (methodology, roadmap, log)
├── tests/                     # Test suite (framework configured, tests pending)
├── temp/                      # Data, processing outputs, and caches (gitignored)
└── .claude/                   # Claude Code configuration (this file lives here)
```

### Key entry points

| Script | Purpose |
|--------|---------|
| `processing/pipeline_oa.py` | National OA integration pipeline (CityNetwork API, all 7,147 BUAs) |
| `processing/process_morphology.py` | Building shape metrics from LiDAR + OS footprints |
| `stats/build_case_oa.py` | Regenerate all OA case figures (three surfaces + basket) |
| `stats/nepi.py` | NEPI scorecard, A–G bands, surface decomposition |
| `stats/access_penalty_model.py` | Empirical access-energy penalty (OLS on observed transport behaviour) |
| `stats/nepi_model.py` | Train four XGBoost models (form, mobility, cars, commute) with monotonic constraints + SHAP |
| `stats/nepi_app.py` | Streamlit interactive planning tool |
| `stats/nepi_static/index.html` | Static HTML/JS planning tool — runs models in the browser, also mirrored to `docs/` |

### Canonical documents

| File | Status |
|------|--------|
| `paper/case_v2.md` | **Current** — full IMRaD narrative for the NEPI paper (198,779 OAs, 6,687 BUAs) |
| `paper/data.md` | **Current** — OA-level data source methodology |
| `paper/literature_review.md` | Current — thematic literature review |
| `paper/references.bib` | Partial — to be finalised before submission |
| `TODO.md` | Development status and forward plan |
| `notes/` | Archived v0 working notes (LSOA-era) — read-only snapshots, not authoritative |

### Storage layout

The base data directory is configured via `URBAN_ENERGY_DATA_DIR` in a `.env` file at the repo root (gitignored). `src/urban_energy/paths.py` loads it and exports:

- `DATA_DIR` = `$URBAN_ENERGY_DATA_DIR/data` — all datasets
- `PROCESSING_DIR` = `$URBAN_ENERGY_DATA_DIR/processing` — per-BUA pipeline outputs
- `CACHE_DIR` = `$URBAN_ENERGY_DATA_DIR/cache` — download caches
- `PROJECT_DIR` — source repo root (for `stats/figures/` outputs)

Expected layout under `$URBAN_ENERGY_DATA_DIR` (default: `./temp/`):

```text
$URBAN_ENERGY_DATA_DIR/
├── cache/                                     ← download caches (CACHE_DIR)
├── processing/                                ← per-BUA pipeline outputs (PROCESSING_DIR)
│   ├── combined/oa_integrated.gpkg            ← Final integrated OA dataset (national)
│   └── {bua_name}/oa_integrated.gpkg          ← Per-BUA OA outputs
└── data/                                      ← datasets (DATA_DIR)
    ├── boundaries/built_up_areas.gpkg
    ├── lidar/building_heights.gpkg
    ├── morphology/buildings_morphology.gpkg
    ├── statistics/
    │   ├── census_oa_joined.gpkg              ← Census 2021 (10 topic tables joined to OA boundaries)
    │   ├── census_2011_commute_oa.parquet    ← Census 2011 QS701/QS702 (pre-pandemic validation)
    │   ├── postcode_energy_consumption.parquet ← DESNZ metered (postcode level — primary energy source)
    │   ├── postcode_oa_lookup.parquet         ← Postcode → OA21CD spatial lookup
    │   ├── oa_energy_consumption.parquet      ← Postcode energy aggregated to OA (meter-weighted)
    │   ├── lsoa_energy_consumption.parquet    ← DESNZ metered (LSOA level — legacy/cross-check)
    │   ├── lsoa_imd2025.parquet               ← IoD25 indices of deprivation (7 domains)
    │   ├── lsoa_vehicles.parquet              ← DVLA vehicle licensing (fuel type, ULEV share)
    │   ├── lsoa_scaling.parquet               ← BRES + small-area GVA
    │   ├── msoa_od_commute.parquet            ← Census 2021 origin-destination workplace flows
    │   └── census_ts*_oa.parquet              ← Individual Census topic tables
    ├── epc/epc_domestic_spatial.parquet
    ├── fsa/fsa_establishments.gpkg
    ├── transport/naptan_england.gpkg
    ├── education/gias_schools.gpkg
    ├── health/nhs_facilities.gpkg
    ├── models/nepi/                           ← trained XGBoost models + band thresholds + archetypes
    └── stats/results/                         ← legacy JSON/CSV analysis outputs
```

## Data Source Inventory

### Currently integrated

| # | Source | Granularity | Script | Output | Role |
|---|--------|------------|--------|--------|------|
| 1 | Census 2021 (10 topic tables) | OA | `data/download_census.py` | `census_oa_joined.gpkg` | Population, accommodation type (TS044), commute distance/mode, deprivation, cars, tenure |
| 2 | Census 2011 commute (QS701/QS702) | OA | `data/download_census_2011.py` | `census_2011_commute_oa.parquet` | Pre-pandemic validation of the morphology-transport gradient |
| 3 | DESNZ postcode-level energy | Postcode | `data/download_energy_postcode.py` → `data/aggregate_energy_oa.py` | `oa_energy_consumption.parquet` | **Primary DV (Form surface)**: domestic gas + electricity, weather-corrected, aggregated to OA via postcode→OA lookup |
| 4 | DESNZ LSOA energy | LSOA | `data/download_energy_stats.py` | `lsoa_energy_consumption.parquet` | Legacy / LSOA-level cross-check |
| 5 | EPC domestic certificates | Property (UPRN) | `data/process_epc.py` | `epc_domestic_spatial.parquet` | Building fabric, construction age, floor area (no longer the primary DV) |
| 6 | Environment Agency LiDAR | 2m raster | `data/process_lidar.py` | `building_heights.gpkg` | nDSM building heights → S/V ratio, volume |
| 7 | OS Open Map Local | Building footprint | (manual) | `os_open_local/` | Building geometry for morphology |
| 8 | OS Open Roads | Road network | (manual) | `oproad_gpkg_gb/` | Street network for CityNetwork centrality + accessibility |
| 9 | OS Open Greenspace | Polygon | (manual) | `opgrsp_gpkg_gb/` | Recreation/restoration trophic layer |
| 10 | OS Open UPRN | Point | (manual) | `osopenuprn_*/` | Property-level geocoding |
| 11 | OS Code-Point Open | Postcode centroid | (manual) | `codepo_gpkg_gb/` | Postcode → OA spatial join, NHS site geocoding |
| 12 | OS Built Up Areas | Polygon | `data/process_boundaries.py` | `built_up_areas.gpkg` | 7,147 English BUA processing units |
| 13 | FSA establishments | Point | `data/download_fsa.py` | `fsa_establishments.gpkg` | Food retail accessibility (~500k) |
| 14 | NaPTAN transport stops | Point | `data/download_naptan.py` | `naptan_england.gpkg` | Bus/rail accessibility (~434k stops) |
| 15 | GIAS schools (DfE) | Point | `data/prepare_gias.py` | `gias_schools.gpkg` | Education accessibility (~25k) |
| 16 | NHS ODS facilities | Point | `data/prepare_nhs.py` | `nhs_facilities.gpkg` | Health accessibility (~24k: GPs, pharmacies, hospitals) |
| 17 | IoD 2025 (IMD) | LSOA (2021) | `data/download_imd.py` | `lsoa_imd2025.parquet` | 7-domain deprivation control (income domain used in OLS) |
| 18 | DVLA vehicle licensing | LSOA (2021) | `data/download_vehicles.py` | `lsoa_vehicles.parquet` | Fleet composition by fuel type, transport lock-in |
| 19 | ONS Small Area GVA + BRES | LSOA | `data/download_scaling.py` | `lsoa_scaling.parquet` | Scaling analysis (forward work) |
| 20 | Census 2021 OD workplace flows | MSOA→MSOA | (manual / `data/extract_gm_roads.py` for support) | `msoa_od_commute.parquet` | OD commute distance robustness check (case_v2 §4.6) |

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

All 7,147 English Built-Up Areas at Output Area resolution (6,687 BUAs / 198,779 OAs processed and filtered for valid data). Each OA contains ~130 households.

### Primary stratification

`dominant_type` — plurality accommodation type from Census TS044 per OA:

- Flat-dominant: 36,502 OAs
- Terraced-dominant: 50,592 OAs
- Semi-detached-dominant: 65,986 OAs
- Detached-dominant: 45,699 OAs

### The NEPI: three energy surfaces in a common kWh/hh/yr unit

1. **Form (building energy):** DESNZ metered gas + electricity, postcode-level, aggregated to OA via meter-weighted means.
2. **Mobility (transport energy):** Census 2021 commute distance (TS058 band midpoints) × mode (TS061) × ECUK 2025 energy intensities (road 0.399, rail 0.178 kWh/pkm), annualised at 220 workdays × return × NTS 6.04× total-to-commute scalar.
3. **Access (energy penalty):** Empirical OLS — `transport_energy ~ local_coverage + controls`. The Access surface is the *additional* transport energy predicted at the OA's actual coverage relative to a compact reference (85% coverage = flat-dominant median). Local coverage itself is computed via Gaussian decay over network distances to 9 service types using the cityseer CityNetwork API.

The composite NEPI is the sum (kWh/hh/yr); A–G bands are assigned by national percentile.

### Headline results (198,779 OAs, 6,687 BUAs)

Median by dominant type (Flat → Detached):

- Form (building): 10,755 → 15,713 kWh/hh (1.46×)
- Mobility (overall): 4,150 → 9,185 kWh/hh (2.21×)
- Total NEPI: 15,982 (Band A) → 26,897 (Band F) — gap 10,915 kWh/hh/yr
- kWh per unit access: 3,292 → 8,820 (2.68×)

Gap decomposition: Form 45% / Mobility 43% / Access penalty 14%.

Plurality sensitivity: gradient *steepens* at stricter thresholds (1.84× at 60% purity), so the headline is conservative. Pre-pandemic Census 2011 validation shows the transport gradient is **steeper** than the COVID-affected 2021 figure (2.00× vs 1.70×).

### NEPI planning tool

Four XGBoost models (form / mobility / cars / commute) with monotonic constraints predict each surface from planner-controllable inputs only:

- Density (people/ha)
- Built form mix (% detached / semi / terraced / flat)
- Local amenity coverage (0–1)
- Transit access (bus + rail gravity-weighted counts)
- Median build year (form model only)

SHAP TreeExplainer provides per-prediction attributions in the Streamlit app. The static HTML version executes the same trees in JavaScript with no Python runtime.

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

### Analytical structure (stats/proof_of_concept_oa.py + nepi.py)

| Step | Test | Role |
|------|------|------|
| 1 | Building typology → building energy (Form) | Foundation |
| 2 | Neighbourhood morphology → transport energy (Mobility) | Gap widens |
| 3 | Neighbourhood morphology → walkable service coverage | Sets up Access |
| 4 | Empirical OLS: transport energy ~ coverage + controls | Access penalty mechanism |
| 5 | NEPI scorecard: A–G bands from composite kWh | Headline result |
| 6 | Pattern holds within deprivation quintiles (IMD25) | Rules out wealth |
| 7 | Pattern holds across all 198,779 OAs (distribution) | Generalisability |
| 8 | Plurality + NTS scalar + 2011 + OD distance robustness | Sensitivity (case_v2 §4) |
| 9 | NEPI planning tool: XGBoost forecasts under intervention | Policy lever |

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

# Data acquisition (see data/README.md for the manual OS downloads)
uv run python data/download_census.py            # Census 2021 topic tables
uv run python data/download_census_2011.py       # Census 2011 commute (validation)
uv run python data/download_energy_stats.py      # DESNZ energy (LSOA, legacy)
uv run python data/download_energy_postcode.py   # DESNZ energy (postcode, primary)
uv run python data/download_imd.py               # IoD 2025
uv run python data/download_vehicles.py          # DVLA vehicles
uv run python data/download_fsa.py               # FSA establishments
uv run python data/download_naptan.py            # NaPTAN transport
uv run python data/download_scaling.py           # BRES + GVA
uv run python data/prepare_gias.py               # GIAS schools (after manual CSV)
uv run python data/prepare_nhs.py                # NHS ODS (after manual CSVs)
uv run python data/process_boundaries.py         # OS BUAs → study boundaries
uv run python data/process_lidar.py              # LiDAR → building heights
uv run python data/process_epc.py                # EPC → spatial parquet
uv run python data/build_postcode_oa_lookup.py   # Postcode → OA lookup
uv run python data/aggregate_energy_oa.py        # Postcode energy → OA energy

# National pipeline
uv run python processing/process_morphology.py   # Building morphology metrics
uv run python processing/pipeline_oa.py          # All BUAs → oa_integrated.gpkg

# Analysis + NEPI
uv run python stats/build_case_oa.py             # Regenerate three-surfaces + basket figures
uv run python stats/nepi.py                      # NEPI scorecard + bands
uv run python stats/access_penalty_model.py      # Empirical access-energy penalty
uv run python stats/nepi_model.py                # Train XGBoost planning-tool models
uv run streamlit run stats/nepi_app.py           # Launch interactive planning tool

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
- **scipy / statsmodels:** Statistical methods and regression (NEPI access-penalty OLS)
- **xgboost / shap:** Planning-tool gradient-boosted models with monotonic constraints + SHAP attributions
- **scikit-learn:** Train/test splits, stratified k-fold for cross-validation
- **cityseer (≥4.25):** CityNetwork API for centrality and accessibility (used by `pipeline_oa.py`)
- **momepy:** Building morphology metrics
- **esda / libpysal:** Spatial statistics

Visualisation:

- **seaborn / matplotlib:** Publication figures
- **streamlit:** Interactive NEPI planning tool

I/O:

- **requests / aiohttp:** HTTP downloads
- **openpyxl:** DESNZ XLSX parsing
- **pyarrow:** Parquet I/O

Development:

- **ruff:** Linting and formatting
- **ty:** Type checking
- **pytest:** Testing framework
