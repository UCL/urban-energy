# data/archive

Acquisition scripts retired from the load-bearing pipeline by the consumption
audit (2026-06-09). Their outputs are read by **no live code** (paper figures or
Atlas). Kept for provenance — see [ROADMAP.md](../../ROADMAP.md) for status.

| Script | Why retired |
| ------ | ----------- |
| `download_census_2011.py` | Pre-pandemic commute robustness check (PAPER §5.5). Output `census_2011_commute_oa.parquet` is never read by the OA pipeline or stats. Also needs a manual `oa11_to_oa21_lookup.parquet` that was not auto-fetched. |
| `download_energy_stats.py` | Legacy LSOA-level DESNZ energy. Superseded by the postcode→OA path (`download_energy_postcode.py` → `aggregate_energy_oa.py`). Only the archived LSOA pipeline referenced it. |
| `download_scaling.py` | BRES + small-area GVA for Bettencourt scaling — forward work ("data loaded, analysis pending"). Only `lsoa_gva_millions`→`gva_per_hh` was ever consumed (descriptive); the Stage-3 join was removed. |
| `extract_gm_roads.py` | Greater-Manchester roads subset for local case-study/testing. Not part of the national pipeline. |

To revive one: `git mv` it back to `data/` and re-add its stage to
`src/urban_energy/pipeline.py`.
