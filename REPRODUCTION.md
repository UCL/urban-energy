# Reproduction

How to rebuild the urban-energy datasets from scratch. Scope is the **load-bearing
KEEP set** from the consumption audit; the heavy LiDAR/morphology path is deferred
(see [ROADMAP.md](ROADMAP.md)). The build is driven by an executable orchestrator —
the dependency order lives in code, not prose — and produces `oa_integrated.gpkg`
plus the per-OA tables the two-axis analysis consumes.

```
acquire → process     (then run the two-axis analysis scripts on demand)
```

## Step 0 — Environment

```bash
uv sync
echo "URBAN_ENERGY_DATA_DIR=/path/to/big/disk" > .env   # ~250 GB free recommended
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
| OA 2021 boundaries | ONS Geoportal | `$DATA_DIR/` (`Output_Areas_2021_*`) |
| GIAS `edubasealldata` CSV | Get Information About Schools | `$CACHE_DIR/gias/` |
| NHS ODS `ets.csv` / `epraccur.csv` / `edispensary.csv` | NHS ODS | `$CACHE_DIR/nhs_ods/` |

**Gotcha:** the OS Open UPRN path is pinned to the `osopenuprn_202601` vintage in
`processing/common.py`. Match that name or update `PATHS`.

**Not needed (deferred):** OS Open Map Local (footprints) and EA LiDAR — they
only feed the morphology path.

## Step 2 — Drive the rebuild with the orchestrator

```bash
uv run python -m urban_energy.pipeline doctor     # env, manual downloads, disk
uv run python -m urban_energy.pipeline status      # what's built vs missing
uv run python -m urban_energy.pipeline list        # the full manifest

uv run python -m urban_energy.pipeline run --layer acquire   # ~1 h, scripted downloads
uv run python -m urban_energy.pipeline run pipeline          # national pipeline (heavy)

# Two-axis analysis (print-only; needs oa_integrated.gpkg + oa_epc.parquet)
uv run python stats/lock_in.py
uv run python stats/access_profile.py
uv run python stats/form_size_decomposition.py
```

Stages skip when their declared outputs already exist (`--force` to rebuild).
`run --all` runs every non-optional stage in order. The national pipeline is
resumable per-BUA, so it can be interrupted and re-run. The analysis scripts are
print-only and run on demand, not as pipeline stages.

**Travel-energy inputs.** Two small open downloads anchor household car-travel
energy (both are `acquire` stages): `data/download_nts_mileage.py` (NTS9904 2024 car
miles/person by 2021 rural-urban class) and `data/download_ons_ruc.py` (ONS 2021 RUC
of OAs). The disaggregation that consumes them is `stats/travel_energy.py`.

## Layer notes & budgets

- **acquire** (~1 h, scripted): Census 2021, DESNZ postcode energy, IMD, DVLA
  vehicles, NTS9904 mileage, ONS RUC, FSA, NaPTAN, GIAS, NHS, EPC, BUAs, then the
  postcode→OA lookup, the meter-weighted OA energy aggregation (primary heat DV),
  and the EPC→OA aggregation (floor area + best-fabric intensity).
- **process** (~30–50 h, resumable): `pipeline_oa.py` — CityNetwork centrality +
  accessibility, then OA aggregation → `processing/combined/oa_integrated.gpkg`.
  Includes the methodology #6 Form under-recording flags.
- **two-axis analysis** (minutes, on-demand; not pipeline stages): `lock_in.py`,
  `access_profile.py`, `form_size_decomposition.py` — print-only, reading the built
  tables via the shared `stats/oa_data.py` core.

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
