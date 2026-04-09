# Urban Energy — Technical Brief

Codebase reference and Claude Code briefing. For the project pitch, headline result,
theory synopsis, and current status read [README.md](README.md). For the paper itself
read [PAPER.md](PAPER.md).

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
│   ├── references.bib         # BibTeX (partial)
│   └── archive/               # LSOA case_v1.md + stale main.tex (frozen)
├── src/urban_energy/
│   └── paths.py               # Centralised storage paths (loads URBAN_ENERGY_DATA_DIR from .env)
├── data/                      # Raw data acquisition + preprocessing scripts
├── processing/                # National OA pipeline
│   ├── pipeline_oa.py         # CityNetwork API, all 7,147 BUAs
│   ├── process_morphology.py  # Building shape metrics from LiDAR + OS footprints
│   └── archive/pipeline_lsoa.py  # LSOA pipeline (still imported for shared morphology constants)
├── stats/                     # Analysis, NEPI scorecard, planning tool
│   ├── proof_of_concept_oa.py # Core OA data loading + aggregation functions
│   ├── oa_figures.py          # Three-surfaces publication figures (fig1–fig8)
│   ├── basket_index_oa.py     # Illustrative basket case
│   ├── build_case_oa.py       # Entry point: regenerates oa_figures + basket_index_oa
│   ├── nepi.py                # NEPI scorecard, A–G bands, surface decomposition
│   ├── access_penalty_model.py # Empirical OLS access-energy penalty
│   ├── nepi_model.py          # Train four XGBoost models (form / mobility / cars / commute)
│   ├── nepi_app.py            # Streamlit interactive planning tool
│   ├── nepi_static/           # Static HTML/JS planning tool (no Python runtime)
│   ├── figures/oa/            # Three-surfaces case figures
│   ├── figures/basket_oa/     # Basket case figures
│   ├── figures/nepi/          # NEPI scorecard, bands, empirical penalty
│   ├── figures/archive_lsoa/  # Archived LSOA-era figures
│   └── archive/               # Archived LSOA analysis scripts
├── docs/                      # GitHub Pages mirror of stats/nepi_static/
├── notes/                     # Archived v0 working notes (LSOA-era, banner-marked)
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

All sources are open. EPC uses "All domestic certificates"; non-domestic EPCs are
excluded. CRS is EPSG:27700 (British National Grid) throughout; EPSG:4326 only for
interchange.

| # | Source | Granularity | Script | Output | Role |
|---|--------|------------|--------|--------|------|
| 1 | Census 2021 (10 topic tables) | OA | `data/download_census.py` | `census_oa_joined.gpkg` | Population, accommodation type (TS044), commute distance/mode, deprivation, cars, tenure |
| 2 | Census 2011 commute (QS701/QS702) | OA | `data/download_census_2011.py` | `census_2011_commute_oa.parquet` | Pre-pandemic validation of the morphology-transport gradient (case_v2 §5.5) |
| 3 | DESNZ postcode-level energy | Postcode | `data/download_energy_postcode.py` → `data/aggregate_energy_oa.py` | `oa_energy_consumption.parquet` | **Primary DV (Form surface)**: gas + electricity, gas weather-corrected, aggregated to OA via postcode→OA lookup |
| 4 | DESNZ LSOA energy | LSOA | `data/download_energy_stats.py` | `lsoa_energy_consumption.parquet` | Legacy / LSOA-level cross-check |
| 5 | EPC domestic certificates | UPRN | `data/process_epc.py` | `epc_domestic_spatial.parquet` | Building fabric, construction age, floor area (NOT the primary DV — see case_v2 §3.3 for justification) |
| 6 | Environment Agency LiDAR | 2 m raster | `data/process_lidar.py` | `building_heights.gpkg` | nDSM building heights → S/V ratio, volume |
| 7 | OS Open Map Local | Building footprint | (manual download) | `os_open_local/` | Building geometry for morphology |
| 8 | OS Open Roads | Road network | (manual) | `oproad_gpkg_gb/` | Street network for CityNetwork centrality + accessibility |
| 9 | OS Open Greenspace | Polygon | (manual) | `opgrsp_gpkg_gb/` | Recreation / restoration trophic layer |
| 10 | OS Open UPRN | Point | (manual) | `osopenuprn_*/` | Property-level geocoding |
| 11 | OS Code-Point Open | Postcode centroid | (manual) | `codepo_gpkg_gb/` | Postcode → OA join + NHS site geocoding |
| 12 | OS Built Up Areas | Polygon | `data/process_boundaries.py` | `built_up_areas.gpkg` | 7,147 English BUA processing units |
| 13 | FSA establishments | Point | `data/download_fsa.py` | `fsa_establishments.gpkg` | Food retail accessibility (~500k) |
| 14 | NaPTAN transport stops | Point | `data/download_naptan.py` | `naptan_england.gpkg` | Bus/rail accessibility (~434k stops) |
| 15 | GIAS schools (DfE) | Point | `data/prepare_gias.py` | `gias_schools.gpkg` | Education accessibility (~25k) |
| 16 | NHS ODS facilities | Point | `data/prepare_nhs.py` | `nhs_facilities.gpkg` | Health accessibility (~24k: GPs, pharmacies, hospitals) |
| 17 | IoD 2025 (IMD) | LSOA (2021) | `data/download_imd.py` | `lsoa_imd2025.parquet` | 7-domain deprivation control (income domain in OLS) |
| 18 | DVLA vehicle licensing | LSOA (2021) | `data/download_vehicles.py` | `lsoa_vehicles.parquet` | Fleet composition by fuel type, transport lock-in |
| 19 | ONS Small Area GVA + BRES | LSOA | `data/download_scaling.py` | `lsoa_scaling.parquet` | Bettencourt scaling analysis (forward work) |
| 20 | Census 2021 OD workplace flows | MSOA→MSOA | (manual) | `msoa_od_commute.parquet` | OD commute distance robustness check (case_v2 §5.6) |

For the per-variable derivation table (what each metric measures, native scale, how it's
brought to OA), see [PAPER.md §3.2](PAPER.md).

### Manual downloads (one-off, no API)

| # | Source | Save to |
|---|--------|---------|
| 1 | [OA Boundaries](https://geoportal.statistics.gov.uk/datasets/ons::output-areas-december-2021-boundaries-ew-bfe-v9/about) | `$DATA_DIR/` |
| 2 | [Built Up Areas](https://osdatahub.os.uk/downloads/open/BuiltUpAreas) | `$DATA_DIR/OS_Open_Built_Up_Areas_GeoPackage/` |
| 3 | [OS Open Map Local](https://osdatahub.os.uk/downloads/open/OpenMapLocal) | `$DATA_DIR/os_open_local/` |
| 4 | [OS Open Roads](https://osdatahub.os.uk/downloads/open/OpenRoads) | `$DATA_DIR/oproad_gpkg_gb/` |
| 5 | [OS Open Greenspace](https://osdatahub.os.uk/downloads/open/OpenGreenspace) | `$DATA_DIR/opgrsp_gpkg_gb/` |
| 6 | [OS Open UPRN](https://osdatahub.os.uk/downloads/open/OpenUPRN) | `$DATA_DIR/osopenuprn_*/` |
| 7 | [OS Code-Point Open](https://osdatahub.os.uk/downloads/open/CodePointOpen) | `$DATA_DIR/codepo_gpkg_gb/` |
| 8 | [EPC "All domestic certificates"](https://epc.opendatacommunities.org/) (registration) | `$DATA_DIR/epc/` |
| 9 | [GIAS edubasealldata CSV](https://get-information-schools.service.gov.uk/Downloads) | `$CACHE_DIR/gias/` |
| 10 | [NHS ODS](https://digital.nhs.uk/services/organisation-data-service/data-search-and-export/csv-downloads): `ets.csv`, `epraccur.csv`, `edispensary.csv` | `$CACHE_DIR/nhs_ods/` |

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
uv run pytest                # Run tests (framework configured, tests pending)
uv run ruff check .          # Lint
uv run ruff format .         # Format
uv run ty check              # Type check
```

### Data acquisition

```bash
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
uv run python data/prepare_nhs.py                # NHS ODS (after manual CSVs + Code-Point)
uv run python data/process_boundaries.py         # OS BUAs → study boundaries
uv run python data/process_lidar.py              # LiDAR → building heights
uv run python data/process_epc.py                # EPC → spatial parquet
uv run python data/build_postcode_oa_lookup.py   # Postcode → OA lookup
uv run python data/aggregate_energy_oa.py        # Postcode energy → OA
```

### National pipeline

```bash
uv run python processing/process_morphology.py   # Building morphology metrics
uv run python processing/pipeline_oa.py          # All BUAs → oa_integrated.gpkg
```

### Analysis + NEPI

```bash
uv run python stats/build_case_oa.py             # Three-surfaces + basket figures
uv run python stats/nepi.py                      # NEPI scorecard + bands
uv run python stats/access_penalty_model.py      # Empirical access-energy penalty
uv run python stats/nepi_model.py                # Train XGBoost planning-tool models
uv run streamlit run stats/nepi_app.py           # Launch interactive tool
```

### Static-tool export (after retraining models)

```bash
# 1. Export trained models to JSON for the static tool
uv run python -c "
import json, xgboost as xgb, sys
sys.path.insert(0, 'stats')
from nepi_model import MODEL_DIR, MODEL_FEATURES
from pathlib import Path

OUT = Path('stats/nepi_static/nepi_models.json')

def extract(path, features):
    m = xgb.XGBRegressor(); m.load_model(path)
    dump = m.get_booster().get_dump(dump_format='json')
    trees = []
    for t in dump:
        nodes = []; tree = json.loads(t)
        def walk(n):
            if 'leaf' in n: nodes.append({'leaf': n['leaf']})
            else:
                nd = {'f': n['split'], 't': n['split_condition'], 'y': None, 'n': None}
                nodes.append(nd)
                for c in n['children']:
                    if c['nodeid'] == n.get('yes', 0): nd['y'] = len(nodes); walk(c)
                    elif c['nodeid'] == n.get('no', 0): nd['n'] = len(nodes); walk(c)
                    else: walk(c)
        walk(tree); trees.append(nodes)
    return {'features': features, 'base_score': 0.0, 'n_trees': len(trees), 'trees': trees}

models = {n: extract(MODEL_DIR/f'nepi_model_{n}.json', MODEL_FEATURES[n]) for n in MODEL_FEATURES}
with open(MODEL_DIR/'nepi_band_thresholds.json') as f: bands = json.load(f)
with open(MODEL_DIR/'nepi_archetype_profiles.json') as f: archetypes = json.load(f)
with open(OUT, 'w') as f: json.dump({'models': models, 'band_thresholds': bands, 'archetypes': archetypes}, f, separators=(',', ':'))
print(f'Exported {OUT} ({OUT.stat().st_size/1024:.0f} KB)')
"

# 2. Mirror to docs/ for GitHub Pages
cp stats/nepi_static/index.html stats/nepi_static/nepi_models.json docs/
```

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

Commits use HEREDOCs with `Co-Authored-By: Claude Opus 4.6 (1M context)` trailer when
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
