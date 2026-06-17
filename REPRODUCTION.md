# Reproduction

How to rebuild the urban-energy datasets from scratch. The build is driven by an
executable orchestrator (`urban_energy.pipeline`) — the dependency order lives in
code, not prose. There is **no heavy processing pipeline**: it is data acquisition +
OA aggregation, then the two-axis analysis runs in the stats layer on demand. The access
measure is a one-off cityseer build over the national road network (~12 min), then cached.

```
acquire  (downloads + OA aggregations)  →  analyse  (stats/, on demand)
```

## Step 0 — Environment

```bash
uv sync
echo "URBAN_ENERGY_DATA_DIR=/path/to/big/disk" > .env   # ~80 GB free (EPC raw dominates)
```

## Step 1 — Manual downloads (Tier A, human-gated)

These cannot be scripted (portals / registration). `doctor` verifies each path.
**Start with EPC** — it needs an account and is the largest file.

| Dataset | Source | Save to |
| ------- | ------ | ------- |
| EPC "All domestic certificates" | EPC Open Data Communities (**registration**) | `$DATA_DIR/epc/` |
| OS Open Greenspace | OS Data Hub (open) | `$DATA_DIR/opgrsp_gpkg_gb/Data/opgrsp_gb.gpkg` |
| OS Open Roads | OS Data Hub (open) | `$DATA_DIR/oproad_gpkg_gb/Data/oproad_gb.gpkg` |
| OS Open UPRN | OS Data Hub | `$DATA_DIR/osopenuprn_*_gpkg/` |
| OS Code-Point Open | OS Data Hub | `$DATA_DIR/codepo_gpkg_gb/` |
| OA 2021 boundaries | ONS Geoportal | `$DATA_DIR/` (`Output_Areas_2021_*`) |
| GIAS `edubasealldata` CSV | Get Information About Schools | `$CACHE_DIR/gias/` |
| NHS ODS `ets.csv` / `epraccur.csv` / `edispensary.csv` | NHS ODS | `$CACHE_DIR/nhs_ods/` |

The OS Open UPRN path is matched by vintage glob in `src/urban_energy/paths.py`
(`latest_uprn_gpkg`), so any `osopenuprn_*` vintage works.

## Step 2 — Acquire (orchestrator)

```bash
uv run python -m urban_energy.pipeline doctor    # env, manual downloads, disk
uv run python -m urban_energy.pipeline status     # what's built vs missing
uv run python -m urban_energy.pipeline run --all  # all downloads + OA aggregations
```

Stages skip when their declared outputs already exist (`--force` to rebuild).
`pipeline list` prints the manifest. Individual scripts also run standalone
(e.g. `uv run python data/download_census.py`).

**Travel-energy inputs** are two small scripted downloads (`acquire` stages):
`download_nts_mileage.py` (NTS9904 car miles/person by 2021 RUC) and
`download_ons_ruc.py` (ONS 2021 RUC of OAs); consumed by `stats/travel_energy.py`.

## Step 3 — Analyse (stats layer, on demand)

```bash
uv run python stats/oa_network_access.py        # build network-access cache (cityseer, ~12 min)
uv run python stats/lock_in.py                  # energy 1.74× → optimised 1.47×
uv run python stats/access_profile.py           # network ~2.9×/kWh + walkable richness ~10×
uv run python stats/form_size_decomposition.py  # heat vs dwelling/household size
```

The loader (`stats/oa_data.py`) assembles the per-OA frame from the acquired artefacts
in-process. The **network** access rate needs `statistics/oa_network_access.parquet`
(`oa_network_access.py` — cityseer over OS Open Roads, built once + queried per catchment
band, ~12 min). The straight-line **walkable** counts are cached by `stats/oa_access.py` on
first run (`statistics/oa_access.parquet`, ~6 s).

## Verify

```bash
uv run pytest
uv run ruff check . && uv run ty check
uv run python -m urban_energy.pipeline status   # all stages "done"
```
