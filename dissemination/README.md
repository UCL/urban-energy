# Dissemination

The dissemination inventory and its sequencing. Specs in this directory; the phase gates are in [ROADMAP.md](../ROADMAP.md).

| Item | What | When |
| ------ | ------ | ------ |
| Zenodo DOIs | Code release + per-OA data deposit | Phase 0 (submission checklist, user actions) |
| Preprint | Same manuscript on a preprint server (the journal's preprint policy permits); the citable anchor for everything below | At submission |
| NEPI score | Per-OA A–G label, current vs potential — [score_spec.md](score_spec.md) | Built 2026-07-29; publish with preprint |
| Atlas v1 | Static map + label card + lever sliders — [atlas_architecture.md](atlas_architecture.md) | Built; soft-launched 2026-07-31 under `noindex`; full launch on acceptance |
| Landing/about page | Part of the Atlas site: the argument in one page, links to paper, data, code | Built, live with the Atlas |
| Policy brief | Two pages for planners/policy (the access-first case, the lock-in result) | On acceptance |
| Press + posts | Per the press plan | On acceptance (embargo-safe) |
| Talks | Seminars/conferences off the preprint | During review |
| Paper 2 (NEPI) | Score design + Atlas as artefact | Phase 2, after acceptance |

## Build plan and status (2026-07-29)

1. **Done** — specs + decisions locked (two papers; no XGBoost; single as-lived per-dwelling mode; static zero-backend stack; LAD/LSOA/OA zoom ladder with national summaries).
2. **Done** — `stats/nepi_score.py`: 178,353 OAs scored → `oa_nepi_score.parquet`; bands frozen at [nepi_bands_2021.json](nepi_bands_2021.json); 12 tests, suite green. Potential-letter saturation is deliberate (score_spec §Label).
3. **Done** — export layer (`stats/atlas_export.py`): England/LAD/LSOA household-weighted aggregates + linear savings sums, PMTiles at three zoom levels (LAD 0.6 MB, LSOA 15.5 MB, OA under the 100 MB host cap at z12), 2,212 postcode shards. Heavy outputs gitignored (regenerable).
4. **Done** — site in `site/`: MapLibre + pmtiles (vendored), zoom-linked choropleth, England panel with live lever totals, unit/OA cards, postcode search, about page. Preview: `npx http-server site -p 8000` (Range-request support is required for PMTiles; Python's `http.server` lacks it).
5. **Done (2026-07-31)** — deployed to Cloudflare R2 under `noindex`: [launch_checklist.md](launch_checklist.md). Remaining there: Zenodo DOIs + preprint link into the about page, then the full-launch steps on acceptance.
