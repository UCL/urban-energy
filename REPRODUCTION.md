# Reproduction

How to rebuild the urban-energy datasets and outputs from scratch. Scope is the
**load-bearing KEEP set** from the 2026-06-09 consumption audit; the heavy
LiDAR/morphology path is deferred (see [ROADMAP.md](ROADMAP.md)). The pipeline is
driven by an executable orchestrator — the dependency order lives in code, not
prose.

```
acquire → process → analyse → atlas → deploy
```

## Step 0 — Environment

```bash
uv sync
echo "URBAN_ENERGY_DATA_DIR=/path/to/big/disk" > .env   # ~250 GB free recommended
brew install tippecanoe pmtiles                          # Atlas tile generation
```

## Step 1 — Manual downloads (Tier A, human-gated)

These cannot be scripted (portals / registration). `doctor` verifies each path.
**Start with EPC** — it needs an account and is the largest file.

| Dataset | Source | Save to |
| ------- | ------ | ------- |
| EPC "All domestic certificates" | EPC Open Data Communities (**registration**) | `$DATA_DIR/epc/` |
| OS Open Built Up Areas | OS Data Hub (open) | `$DATA_DIR/OS_Open_Built_Up_Areas_GeoPackage/` |
| OS Open Roads | OS Data Hub | `$DATA_DIR/oproad_gpkg_gb/Data/oproad_gb.gpkg` |
| OS Open Greenspace | OS Data Hub | `$DATA_DIR/opgrsp_gpkg_gb/Data/opgrsp_gb.gpkg` |
| OS Open UPRN | OS Data Hub | `$DATA_DIR/osopenuprn_202601_gpkg/osopenuprn_202601.gpkg` |
| OS Code-Point Open | OS Data Hub | `$DATA_DIR/codepo_gpkg_gb/` |
| OS Boundary Line (Atlas LAD layer) | OS Data Hub | `$DATA_DIR/bdline_gpkg_gb/Data/bdline_gb.gpkg` |
| OA 2021 boundaries | ONS Geoportal | `$DATA_DIR/` (`Output_Areas_2021_*`) |
| GIAS `edubasealldata` CSV | Get Information About Schools | `$CACHE_DIR/gias/` |
| NHS ODS `ets.csv` / `epraccur.csv` / `edispensary.csv` | NHS ODS | `$CACHE_DIR/nhs_ods/` |

**Gotcha:** the OS Open UPRN path is pinned to the `osopenuprn_202601` vintage in
`processing/common.py`. Match that name or update `PATHS`.

**Not needed (deferred):** OS Open Map Local (footprints) and EA LiDAR — they
only feed the morphology path.

## Step 2 — Drive the rebuild with the orchestrator

```bash
uv run python -m urban_energy.pipeline doctor     # env, manual downloads, binaries, disk
uv run python -m urban_energy.pipeline status      # what's built vs missing
uv run python -m urban_energy.pipeline list        # the full manifest

uv run python -m urban_energy.pipeline run --layer acquire   # ~1 h, scripted downloads
uv run python -m urban_energy.pipeline run pipeline          # national pipeline (heavy)
uv run python -m urban_energy.pipeline run --layer analyse   # figures + NEPI + models
uv run python -m urban_energy.pipeline run --layer atlas     # static tool + Atlas tiles
uv run python -m urban_energy.pipeline run --layer deploy     # mirror nepi_static/ → docs/
```

Stages skip when their declared outputs already exist (`--force` to rebuild).
`run --all` runs every non-optional stage in order. The national pipeline is
resumable per-BUA, so it can be interrupted and re-run.

**Travel-energy inputs.** Two small open downloads anchor household car-travel
energy and must be present before the analyse layer:
`uv run python data/download_nts_mileage.py` (NTS9904 2024 car miles/person by
2021 rural-urban class) and `uv run python data/download_ons_ruc.py` (ONS 2021
RUC of OAs). The disaggregation that consumes them is `stats/travel_energy.py`.

## Layer notes & budgets

- **acquire** (~1 h, scripted): Census 2021, DESNZ postcode energy, IMD, DVLA
  vehicles, FSA, NaPTAN, GIAS, NHS, EPC, BUAs, projections, then the postcode→OA
  lookup and the meter-weighted OA energy aggregation (primary Form DV).
- **process** (~30–50 h, resumable): `pipeline_oa.py` — CityNetwork centrality +
  accessibility, then OA aggregation → `processing/combined/oa_integrated.gpkg`.
  Includes the methodology #6 Form under-recording flags.
- **analyse** (minutes): case figures, NEPI scorecard/bands, empirical access
  penalty, four monotonic XGBoost models + SHAP.
- **atlas** (minutes; needs `tippecanoe`/`pmtiles`): export the static-tool JSON
  and the national Atlas (`summary.json` + pmtiles).
- **deploy**: mirror `stats/nepi_static/` → `docs/` for GitHub Pages. **The OA
  pmtiles upload to Cloudflare R2 is manual** (set `ATLAS_OA_TILES_URL_BASE` and
  upload `*_oa.pmtiles`); the orchestrator prints the reminder.

## Deferred path (optional)

```bash
uv run python -m urban_energy.pipeline run lidar morphology --include-optional
```

Runs the ~30–45 h LiDAR + morphology stages. Only `height_mean` (one figure
table cell) is consumed downstream; see [ROADMAP.md](ROADMAP.md).

## Verify

```bash
uv run pytest                       # incl. tests/test_form_bias.py
uv run ruff check . && uv run ty check
uv run python -m urban_energy.pipeline status   # all non-deferred stages "done"
```
