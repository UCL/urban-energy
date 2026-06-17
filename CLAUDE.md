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
│   └── pipeline.py            # Acquisition orchestrator: doctor / status / list / run
├── data/                      # Raw data acquisition + OA aggregation scripts
├── stats/                     # Two-axis analysis (energy spent vs access gained)
│   ├── oa_data.py             # Core OA loader — assembles from primary artefacts + OLS helpers
│   ├── oa_access.py           # Straight-line KD-tree access (counts within 1,600 m) — cached
│   ├── travel_energy.py       # NTS-anchored car-travel energy (constrained disaggregation)
│   ├── access_profile.py      # Per-service access counts + ×/kWh (incl. grocery, jobs)
│   ├── lock_in.py             # Residual energy gap after best fabric + full EV
│   ├── form_size_decomposition.py # Heat vs dwelling/household-size decomposition
│   ├── argument_figures.py    # The two paper/summary.md figures
│   ├── figures/{oa,nepi}/     # Legacy three-surface PNGs (referenced only by the deferred PAPER)
│   └── archive/               # Archived LSOA analysis scripts
├── tests/                     # pytest framework configured, tests pending
├── temp/                      # Default $URBAN_ENERGY_DATA_DIR (gitignored)
└── .claude/settings.local.json # Claude Code permissions
```

---

## 2. Storage layout

The base data directory is configured via `URBAN_ENERGY_DATA_DIR` in a `.env` file at the
repo root (gitignored). [`src/urban_energy/paths.py`](src/urban_energy/paths.py) loads it
and exports:

- `DATA_DIR` = `$URBAN_ENERGY_DATA_DIR` — all datasets
- `CACHE_DIR` = `$URBAN_ENERGY_DATA_DIR/cache` — download caches
- `PROJECT_DIR` — source repo root

**Never hardcode `temp/` paths** — always import from `urban_energy.paths`.

```text
$URBAN_ENERGY_DATA_DIR/
├── cache/                                     ← download caches
├── statistics/
│   ├── census_oa_joined.gpkg                 ← Census 2021 OA tables + OA/LSOA geometry
│   ├── postcode_energy_consumption.parquet   ← DESNZ metered (postcode level — primary)
│   ├── postcode_oa_lookup.parquet            ← Postcode → OA21CD spatial lookup
│   ├── oa_energy_consumption.parquet         ← Postcode energy aggregated to OA (meter-weighted)
│   ├── oa_epc.parquet                        ← EPC floor area + best-fabric intensity + build year → OA
│   ├── oa_access.parquet                     ← straight-line access counts per OA (cached, ~6 s)
│   ├── lsoa_imd2025.parquet                  ← IoD25 income (deprivation control)
│   ├── lsoa_vehicles.parquet                 ← DVLA vehicle licensing (bev_share)
│   ├── nts_mileage_by_ruc.parquet            ← NTS9904 car miles/person by 2021 RUC (travel anchor)
│   └── oa21_ruc21.parquet                    ← OA → 2021 rural-urban class
├── epc/epc_domestic_spatial.parquet
├── fsa/fsa_establishments.gpkg               ← food service + grocery retail
├── transport/naptan_england.gpkg
├── education/gias_schools.gpkg
├── health/nhs_facilities.gpkg
├── employment/workplace_jobs.gpkg            ← Census WP101EW workplace jobs → OA points
└── opgrsp_gpkg_gb/                            ← OS Open Greenspace (access point layer)
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
| EPC domestic | `process_epc.py` → `aggregate_epc_oa.py` | `epc_domestic_spatial.parquet`, `oa_epc.parquet` | Build year + dwelling floor area + best-fabric (POTENTIAL) intensity |
| OS Open Greenspace | (manual) | `opgrsp_gpkg_gb/` | Greenspace access (straight-line) |
| OS Open UPRN | (manual) | `osopenuprn_*/` | EPC geocoding |
| OS Code-Point Open | (manual) | `codepo_gpkg_gb/` | Postcode→OA lookup |
| FSA establishments | `download_fsa.py` | `fsa_establishments.gpkg` | Food + grocery access (~299k) |
| NaPTAN stops | `download_naptan.py` | `naptan_england.gpkg` | Bus/rail access (~434k) |
| GIAS schools | `prepare_gias.py` | `gias_schools.gpkg` | Education access (~25k) |
| NHS ODS | `prepare_nhs.py` | `nhs_facilities.gpkg` | Health access (GPs/pharmacies/hospitals) |
| Census WP101EW jobs | `download_workplace.py` | `workplace_jobs.gpkg` | Employment access (jobs reachable per OA) |
| IoD 2025 | `download_imd.py` | `lsoa_imd2025.parquet` | Deprivation control (income domain) |
| DVLA vehicles | `download_vehicles.py` | `lsoa_vehicles.parquet` | Fleet composition (`bev_share` → travel-energy fleet intensity) |
| NTS9904 mileage | `download_nts_mileage.py` | `nts_mileage_by_ruc.parquet` | **Travel-energy anchor**: measured car miles/person by 2021 RUC class |
| ONS RUC 2021 | `download_ons_ruc.py` | `oa21_ruc21.parquet` | OA→rural-urban class (travel-energy disaggregation) |

For the per-variable derivation table see [PAPER.md §3.2](PAPER.md).

The manual-download checklist (exact target paths, EPC registration) and the deferred
sources (OS Map Local, EA LiDAR) are in [REPRODUCTION.md](REPRODUCTION.md).

---

## 4. Pipeline architecture

Two layers — acquire, then analyse (no heavy processing pipeline):

```text
┌─────────────────────────────────────────────────────────────────┐
│  data/    Acquire raw layers + aggregate to OA                  │
│           census · energy → oa_energy · EPC → oa_epc ·          │
│           FSA · NaPTAN · GIAS · NHS · greenspace · jobs          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  stats/   Two-axis analysis (print-only, run on demand)         │
│           oa_data assembles the frame + network access          │
│           (oa_network_access) + straight-line walkable          │
│           (oa_access) → travel_energy · access_profile ·        │
│           lock_in · form_size_decomposition                     │
└─────────────────────────────────────────────────────────────────┘
```

### Access is measured over the road network (cityseer)

[`stats/oa_network_access.py`](stats/oa_network_access.py) is the access measure. It builds the
**whole England road network into a single cityseer Rust structure once** (OS Open Roads
`road_link`, ~3.6 M nodes, ~6 min), then — because cityseer's accessibility skips non-live nodes
as origins — sets every OA's nearest node live and computes the **full amenity-vs-distance curve**
in one pass: counts at each ladder rung from **1,600 m to 25,600 m**. From that one curve per OA,
three numbers are read (`access_profile.py`): the **walkable doorstep** (1,600 m), the
**like-for-like** reach (matched distance, **4.5–9.5×**), and the **drivable rate** (each OA
interpolated at its own NTS catchment ÷ travel energy, **~2.9×**/kWh). Caches to
`statistics/oa_network_access.parquet` (~15 min total).

[`stats/oa_access.py`](stats/oa_access.py) is retained for the **straight-line** KD-tree counts
within 1,600 m (a fast cross-check; `access_profile` now uses the network walkable instead).
Cached to `statistics/oa_access.parquet` in ~6 s.

> The earlier *straight-line-only* simplification (cityseer removed) was reverted: scoping access
> to the real network is the rigorous measure. Like-for-like a flat reaches **4.5–9.5×** more at
> any distance; sprawl only matches the count by driving ~2.4× as far, for **~2.9×** the energy.

### Two-axis analysis layer (`stats/`)

The analysis is **two measured axes and a rate** (canonical statement:
[`paper/summary.md`](paper/summary.md)). All four scripts load the same core,
[`stats/oa_data.py`](stats/oa_data.py) (`load_and_aggregate` + shared OLS helpers), and
are **print-only** — they consume the built data artefacts and report to stdout, so they
are run on demand rather than wired as pipeline stages.

| Script | What it computes |
|--------|------------------|
| `oa_network_access.py` | **Network access** (cityseer over OS Open Roads): network built **once**, then the full amenity-vs-distance curve (1,600 m → 25.6 km) for every OA in one pass → `oa_network_access.parquet` (~15 min) |
| `oa_access.py` | Straight-line KD-tree counts within 1,600 m — a fast cross-check, cached |
| `travel_energy.py` | Total car-travel energy by constrained disaggregation of measured NTS9904 mileage (the `compute_travel_energy` the loader calls) |
| `access_profile.py` | Three access numbers: walkable doorstep (network 1,600 m), like-for-like **4.5–9.5×**, drivable rate **~2.9×**/kWh |
| `lock_in.py` | Energy gap surviving best-fabric + full EV (1.74× → 1.47×) |
| `form_size_decomposition.py` | Heat vs dwelling/household-size, via floor-area elasticity + a total→direct regression ladder |

> **⏸ Pending.** The earlier three-surface / A–G scorecard, the empirical access-penalty model,
> and the Atlas (XGBoost planning models + static site) were removed from the tree in the
> two-axis migration (git history holds them). The Atlas scoring + models are **pending
> reevaluation** for the two-axis frame; the paper is deferred. See
> [`paper/summary.md`](paper/summary.md) and [ROADMAP.md](ROADMAP.md).

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

### Data acquisition (via the orchestrator)

```bash
uv run python -m urban_energy.pipeline doctor    # check manual downloads + disk
uv run python -m urban_energy.pipeline run --all # all downloads + OA aggregations
```

Individual scripts also run standalone (e.g. `uv run python data/download_census.py`);
`pipeline list` prints the manifest with each stage's script and outputs.

### Analysis — two-axis (current)

```bash
uv run python stats/oa_network_access.py         # build network-access cache (cityseer, ~12 min)
uv run python stats/lock_in.py                   # energy gradient 1.74× → optimised 1.47×
uv run python stats/access_profile.py            # network ~2.9×/kWh + walkable richness ~10×
uv run python stats/form_size_decomposition.py   # heat vs dwelling/household-size decomposition
```

The analysis assembles the frame in-process from the acquired artefacts. The straight-line
walkable counts are cached by `oa_access` on first run; the **network** access rate needs the
`oa_network_access.parquet` cache (build it first with `oa_network_access.py`).

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

Core geospatial: **geopandas, shapely, pyproj**
Network access: **cityseer** (≥4.25.0b24; the road-network accessibility engine — pulls
fiona/networkx/rasterio). Requires **Python <3.14** (no cityseer wheel for 3.14 yet).
Analysis: **numpy, pandas, scipy** (KD-tree walkable access), **statsmodels** (the form/size OLS ladder)
Visualisation: **matplotlib** (the two argument figures)
I/O: **requests / aiohttp**, **openpyxl** (DESNZ XLSX), **odfpy** (NTS ODS), **pyarrow** (parquet)
Dev: **ruff, ty, pytest**

Full pin list in `uv.lock`. **cityseer is back** (network access, §4). Still pruned (two-axis
strip): xgboost, shap, streamlit, scikit-learn, seaborn, esda, fpdf2 — re-add only if the Atlas
is rebuilt. (momepy/libpysal/rasterstats stayed out — the morphology layer is not revived; only
cityseer accessibility is.)
