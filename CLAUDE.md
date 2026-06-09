# Urban Energy — Technical Brief

Codebase reference and Claude Code briefing. For the project pitch, headline result,
theory synopsis, and current status read [README.md](README.md). For the paper itself
read [PAPER.md](PAPER.md).

**Rebuilding the data?** The pipeline is driven by an executable orchestrator:
`uv run python -m urban_energy.pipeline {doctor,status,list,run}`. The step-by-step
recipe is [REPRODUCTION.md](REPRODUCTION.md); current scope and what is KEEP / DEFER /
CUT lives in [ROADMAP.md](ROADMAP.md). Prefer those over the prose command lists below.

**Author:** Gareth Simons · **License:** GPL-3.0-only · **Repo:** [UCL/urban-energy](https://github.com/UCL/urban-energy)

---

## 1. Project layout

```text
urban-energy/
├── README.md                  # Project pitch + theory + status (read this first)
├── CLAUDE.md                  # This file — technical brief
├── PAPER.md                   # Canonical paper (IMRaD draft, NEPI case)
├── paper/
│   ├── literature_review.md   # Thematic literature review
│   └── references.bib         # BibTeX (partial)
├── src/urban_energy/
│   ├── paths.py               # Centralised storage paths (loads URBAN_ENERGY_DATA_DIR from .env)
│   ├── pipeline.py            # Orchestrator: doctor / status / list / run
│   └── form_bias.py           # Form under-recording flags (methodology #6)
├── data/                      # Raw data acquisition + preprocessing scripts
├── processing/                # National OA pipeline
│   ├── pipeline_oa.py         # CityNetwork API, all 7,147 BUAs
│   ├── common.py              # Shared pipeline plumbing (PATHS, Stage 1, morphology constants)
│   ├── process_morphology.py  # Building shape metrics from LiDAR + OS footprints (deferred path)
│   └── archive/               # Frozen LSOA pipeline + one-offs (not imported — see archive/README.md)
├── stats/                     # Analysis, NEPI scorecard, planning tool
│   ├── proof_of_concept_oa.py # Core OA data loading + aggregation functions
│   ├── oa_figures.py          # Three-surfaces publication figures (fig1–fig8)
│   ├── build_case_oa.py       # Entry point: regenerates the three-surfaces figures
│   ├── nepi.py                # NEPI scorecard, A–G bands, surface decomposition
│   ├── access_penalty_model.py # Empirical OLS access-energy penalty
│   ├── nepi_model.py          # Train four XGBoost models (form / mobility / cars / commute)
│   ├── export_static_tool.py  # Export trained models → nepi_static/nepi_models.json
│   ├── nepi_app.py            # Streamlit interactive planning tool
│   ├── nepi_static/           # Static HTML/JS planning tool (no Python runtime)
│   ├── figures/oa/            # Three-surfaces case figures
│   ├── figures/basket_oa/     # Basket case figures
│   ├── figures/nepi/          # NEPI scorecard, bands, empirical penalty
│   ├── figures/archive_lsoa/  # Archived LSOA-era figures
│   └── archive/               # Archived LSOA analysis scripts
├── docs/                      # GitHub Pages mirror of stats/nepi_static/
├── tests/                     # pytest framework configured, tests pending
├── temp/                      # Default $URBAN_ENERGY_DATA_DIR (gitignored)
└── .claude/settings.local.json # Claude Code permissions
```

---

## 2. Storage layout

The base data directory is configured via `URBAN_ENERGY_DATA_DIR` in a `.env` file at the
repo root (gitignored). [`src/urban_energy/paths.py`](src/urban_energy/paths.py) loads it
and exports:

- `DATA_DIR` = `$URBAN_ENERGY_DATA_DIR/data` — all datasets
- `PROCESSING_DIR` = `$URBAN_ENERGY_DATA_DIR/processing` — per-BUA pipeline outputs
- `CACHE_DIR` = `$URBAN_ENERGY_DATA_DIR/cache` — download caches
- `PROJECT_DIR` — source repo root (used for `stats/figures/` outputs)

**Never hardcode `temp/` paths** — always import from `urban_energy.paths`.

```text
$URBAN_ENERGY_DATA_DIR/
├── cache/                                     ← download caches
├── processing/                                ← per-BUA pipeline outputs
│   ├── combined/oa_integrated.gpkg            ← Final integrated OA dataset (national, ~199k OAs)
│   └── {bua_name}/oa_integrated.gpkg          ← Per-BUA OA outputs
└── data/                                      ← all datasets
    ├── boundaries/built_up_areas.gpkg
    ├── lidar/building_heights.gpkg
    ├── morphology/buildings_morphology.gpkg
    ├── statistics/
    │   ├── census_oa_joined.gpkg              ← Census 2021 (10 topic tables joined to OAs)
    │   ├── census_2011_commute_oa.parquet    ← Census 2011 QS701/QS702 (pre-pandemic validation)
    │   ├── postcode_energy_consumption.parquet ← DESNZ metered (postcode level — primary)
    │   ├── postcode_oa_lookup.parquet         ← Postcode → OA21CD spatial lookup
    │   ├── oa_energy_consumption.parquet      ← Postcode energy aggregated to OA (meter-weighted)
    │   ├── lsoa_energy_consumption.parquet    ← DESNZ metered (LSOA level — legacy)
    │   ├── lsoa_imd2025.parquet               ← IoD25 (7 domains)
    │   ├── lsoa_vehicles.parquet              ← DVLA vehicle licensing
    │   ├── lsoa_scaling.parquet               ← BRES + small-area GVA
    │   ├── msoa_od_commute.parquet            ← Census 2021 OD workplace flows
    │   └── census_ts*_oa.parquet              ← Individual Census topic tables
    ├── epc/epc_domestic_spatial.parquet
    ├── fsa/fsa_establishments.gpkg
    ├── transport/naptan_england.gpkg
    ├── education/gias_schools.gpkg
    ├── health/nhs_facilities.gpkg
    └── models/nepi/                           ← trained XGBoost models + band thresholds + archetypes
```

---

## 3. Data inventory

All sources are open; CRS is EPSG:27700 throughout (EPSG:4326 only for interchange).
Scope (KEEP / DEFER / CUT) is in [ROADMAP.md](ROADMAP.md); download links and the full
rebuild recipe are in [REPRODUCTION.md](REPRODUCTION.md). The load-bearing (KEEP) sources:

| Source | Script | Output | Role |
|--------|--------|--------|------|
| Census 2021 (10 topic tables) | `download_census.py` | `census_oa_joined.gpkg` | Population, dwelling type (TS044), commute, cars, deprivation |
| DESNZ postcode energy | `download_energy_postcode.py` → `aggregate_energy_oa.py` | `oa_energy_consumption.parquet` | **Primary DV (Form)**: metered gas + electricity → OA |
| EPC domestic | `process_epc.py` | `epc_domestic_spatial.parquet` | Construction age → `median_build_year` (Form-model feature) |
| OS Open Roads | (manual) | `oproad_gpkg_gb/` | CityNetwork centrality + accessibility |
| OS Open Greenspace | (manual) | `opgrsp_gpkg_gb/` | Recreation accessibility |
| OS Open UPRN | (manual) | `osopenuprn_*/` | Property geocoding (aggregation key) |
| OS Code-Point Open | (manual) | `codepo_gpkg_gb/` | Postcode→OA + NHS geocoding |
| OS Built Up Areas | `process_boundaries.py` | `built_up_areas.gpkg` | ~7,244 English BUA processing units |
| OS Boundary Line | (manual) | `bdline_gpkg_gb/` | Atlas LAD layer |
| FSA establishments | `download_fsa.py` | `fsa_establishments.gpkg` | Food accessibility (~500k) |
| NaPTAN stops | `download_naptan.py` | `naptan_england.gpkg` | Bus/rail accessibility (~434k) |
| GIAS schools | `prepare_gias.py` | `gias_schools.gpkg` | Education accessibility (~25k) |
| NHS ODS | `prepare_nhs.py` | `nhs_facilities.gpkg` | Health accessibility (GPs/pharmacies/hospitals) |
| IoD 2025 | `download_imd.py` | `lsoa_imd2025.parquet` | Deprivation control (income domain in OLS) |
| DVLA vehicles | `download_vehicles.py` | `lsoa_vehicles.parquet` | Fleet composition (Atlas `bev_share`) |
| NESO projections | `build_projections.py` | `projections.parquet` | Atlas scenario factors |

For the per-variable derivation table see [PAPER.md §3.2](PAPER.md).

The manual-download checklist (exact target paths, EPC registration), the external
binaries (`tippecanoe`/`pmtiles`), and the deferred sources (OS Map Local, EA LiDAR)
are all in [REPRODUCTION.md](REPRODUCTION.md).

---

## 4. Pipeline architecture

Three layers:

```text
┌─────────────────────────────────────────────────────────────────┐
│  data/        Raw acquisition + initial preprocessing           │
│               Outputs to $DATA_DIR/{statistics, lidar, ...}     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  processing/  National OA pipeline (per-BUA, resumable)         │
│               pipeline_oa.py — CityNetwork API, three stages    │
│               Outputs to $PROCESSING_DIR/{bua_name, combined}/  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  stats/       Analysis, scorecard, models, figures              │
│               Outputs to stats/figures/{oa, basket_oa, nepi}/   │
│               + $DATA_DIR/models/nepi/                          │
└─────────────────────────────────────────────────────────────────┘
```

### OA pipeline stages (`processing/pipeline_oa.py`)

For each Built-Up Area:

1. **Stage 1 — Building morphology** (from cached LiDAR + momepy metrics)
2. **Stage 2 — Network analysis** (cityseer 4.25 CityNetwork: centrality + Gaussian-weighted accessibility to FSA, NaPTAN, GIAS, NHS, OS Open Greenspace)
3. **Stage 3 — OA aggregation** (transient UPRN joins → meter-weighted aggregation to OA polygons; postcode energy joined via `oa_energy_consumption.parquet`)

The pipeline is **resumable**: per-BUA outputs are written under
`$PROCESSING_DIR/{bua_name}/` and skipped on re-run if present. The merged national output
(`$PROCESSING_DIR/combined/oa_integrated.gpkg`, ~2.4 GB) is regenerated each invocation.

Run by name, GSS code, or all:

```bash
uv run python processing/pipeline_oa.py                  # all BUAs
uv run python processing/pipeline_oa.py cambridge        # by name
uv run python processing/pipeline_oa.py E63010556        # by code
```

### Building morphology output schema

`$DATA_DIR/morphology/buildings_morphology.gpkg`:

| Column | Type | Description |
| ------ | ---- | ----------- |
| `id` | string | OS building identifier |
| `geometry` | polygon | Building footprint (EPSG:27700) |
| `height_*` | float | LiDAR-derived heights |
| `footprint_area_m2` | float | Building footprint area |
| `perimeter_m` | float | Building perimeter |
| `orientation` | float | Deviation from cardinal directions (0–45°) |
| `convexity` | float | Area / convex hull area (1=simple, <1=L-shapes) |
| `compactness` | float | Circular compactness (1=circle, <1=elongated) |
| `elongation` | float | Longest / shortest axis ratio |
| `shared_wall_length_m` | float | Total shared wall length (momepy, 1.5 m tolerance) |
| `shared_wall_ratio` | float | shared_wall_length / perimeter (0=detached, ~0.5=terraced) |

Per-boundary cache lives in `$DATA_DIR/morphology/cache/` (one `.gpkg` per BUA22CD).

### NEPI planning-tool models (`stats/nepi_model.py`)

Four XGBoost models with **monotonic constraints**, predicting from planner-controllable
inputs only (no mediators):

| Model | Target | Features |
|-------|--------|----------|
| Form | `building_kwh_per_hh` | density, type mix, build year |
| Mobility | `transport_kwh_per_hh_total_est` | density, type mix, local coverage, bus + rail access |
| Cars | `cars_per_hh` | density, type mix, local coverage, bus + rail access |
| Commute | `avg_commute_km` | local coverage, bus + rail access |

Trained models live at `$DATA_DIR/models/nepi/nepi_model_{form,mobility,cars,commute}.json`,
plus `nepi_band_thresholds.json`, `nepi_archetype_profiles.json`, `nepi_feature_stats.json`.
SHAP TreeExplainer provides exact attributions in the Streamlit app.

### Static planning tool

`stats/nepi_static/index.html` + `nepi_models.json` runs the same XGBoost trees in
JavaScript with **no Python runtime**. The two files are mirrored to `docs/` for GitHub
Pages deployment.

---

## 5. Commands reference

### Development

```bash
uv sync                      # Install dependencies
uv run pytest                # Run tests
uv run ruff check .          # Lint
uv run ruff format .         # Format
uv run ty check              # Type check
```

### Data acquisition + national pipeline (via the orchestrator)

```bash
uv run python -m urban_energy.pipeline run --layer acquire  # KEEP-set downloads + OA energy
uv run python -m urban_energy.pipeline run pipeline         # national pipeline → oa_integrated.gpkg
```

Individual scripts still run standalone (e.g. `uv run python data/download_census.py`);
`pipeline list` prints the full manifest with each stage's script and outputs.
The deferred LiDAR/morphology stages run only on opt-in:

```bash
uv run python -m urban_energy.pipeline run lidar morphology --include-optional
```

### Analysis + NEPI

```bash
uv run python stats/build_case_oa.py             # Three-surface figures
uv run python stats/nepi.py                      # NEPI scorecard + bands
uv run python stats/access_penalty_model.py      # Empirical access-energy penalty
uv run python stats/nepi_model.py                # Train XGBoost planning-tool models
uv run streamlit run stats/nepi_app.py           # Launch interactive tool
```

### Static-tool export (after retraining models)

```bash
# Export the XGBoost trees → stats/nepi_static/nepi_models.json, then mirror docs/
uv run python stats/export_static_tool.py
uv run python -m urban_energy.pipeline run mirror_docs
```

The export logic (previously an inline `python -c` one-liner here) is now
[stats/export_static_tool.py](stats/export_static_tool.py), wired as the
orchestrator's `static_tool` stage so the browser models cannot drift from the
trained `$DATA_DIR/models/nepi/` artifacts.

---

## 6. Conventions

### Python code quality

1. **Type annotations:** all functions must have complete type hints. Use modern Python typing (3.10+) — `|` unions, `list[]` / `dict[]` generics.
2. **Docstrings:** NumPy style for all public functions and classes.
3. **Linting:** `ruff check` + `ruff format` (config in `pyproject.toml`).
4. **Type checking:** `uv run ty check`.
5. **Tests:** pytest for non-trivial functions; geospatial operations validated against known inputs.

### Script conventions

All `data/` scripts follow this pattern:

- Import `DATA_DIR` and `CACHE_DIR` from `urban_energy.paths` (never hardcode `temp/`)
- Use `download_and_cache()` to avoid re-downloading
- Output parquet (tabular) or GeoPackage (spatial) under `$DATA_DIR`
- Print progress with `[1/N]` step numbers and summary statistics
- Filter to England only (codes starting with `E`)
- Use `requests` with `User-Agent: urban-energy-research/1.0`

### Geospatial best practices

1. **CRS:** EPSG:27700 for analysis. EPSG:4326 only for interchange.
2. **Spatial joins:** prefer `sjoin` with explicit `predicate` parameter.
3. **Memory:** appropriate dtypes; chunk large datasets; release with `del` + `gc.collect()`.
4. **Raster:** always use context managers with rasterio; respect nodata.
5. **Reproducibility:** set random seeds; document data versions; use deterministic ordering.

### Academic writing

1. **Style:** formal, no contractions or first-person singular.
2. **Citations:** author-date (APA/Harvard); all claims supported.
3. **Precision:** quantitative claims specify units, CIs, and sources.
4. **Reproducibility:** methods sufficient for independent replication.
5. **Structure:** IMRaD.

### Git workflow

```bash
git status                   # Check changes
git diff                     # Review
git log --oneline -10        # Recent history
```

Commits use HEREDOCs with `Co-Authored-By: Claude Opus 4.8 (1M context)` trailer when
Claude assists. New commits, never `--amend` after a hook failure.

---

## 7. Dependencies

Core geospatial: **geopandas, shapely, rasterio, pyproj, momepy**
Analysis: **numpy, pandas, scipy, statsmodels** (NEPI access-penalty OLS), **xgboost + shap** (planning-tool models with monotonic constraints), **scikit-learn** (stratified splits + CV)
Network: **cityseer ≥ 4.25** (CityNetwork API used by `pipeline_oa.py`)
Spatial stats: **esda, libpysal**
Visualisation: **seaborn / matplotlib** (figures), **streamlit** (interactive tool)
I/O: **requests / aiohttp**, **openpyxl** (DESNZ XLSX), **pyarrow** (parquet)
Dev: **ruff, ty, pytest**

Full pin list in `uv.lock` (112 packages).
