# Atlas v1 — static architecture (zero backend)

Status: draft for author sign-off. Constraint: set it up once, serve it statically, no server-side compute, no database, nothing to patch. The retired Atlas (git `583ff6e^`, `stats/atlas/` + `stats/nepi_static/`) already proved this stack at national scale; v1 revives the stack, not the old scoring.

## Stack

- **Tiles**: polygons + attributes → GeoJSON → tippecanoe → **PMTiles**. A single archive per level, served as a static asset over HTTP range requests. Three levels, switched by zoom: LAD (national view), LSOA (city view), OA (neighbourhood detail). Each level carries its own pre-aggregated attributes, so clicking a unit at any scale yields summary statistics without loading the level below.
- **Map**: MapLibre GL JS with the pmtiles protocol. Vendored (not CDN-pinned at runtime) to remove the one external dependency that can rot.
- **UI**: small vanilla JS or Alpine.js label card + lever controls. No framework build step beyond copying files.
- **Hosting**: GitHub Pages from the repo (or Cloudflare Pages if the PMTiles archives exceed the 100 MB file cap; OA-level England fitted before with coalescing).
- **Search**: postcode → OA via the existing `postcode_oa_lookup.parquet`, exported as small JSON shards keyed by outcode (~2,900 files, a few KB each). Static, no geocoding API.

## Interactivity without a model

The scenario levers are closed-form per OA (`scenario_energy`), so the "experiment with the inputs" requirement is met client-side with arithmetic on baked attributes. Each OA tile carries roughly eight numbers: metered gas, metered electricity, travel kWh, EPC fabric factor, catchment amenity count, on-foot amenity count, households, coverage flag.

- Controls: three uptake sliders (fabric, heat pump, EV, 0–100%), optionally COP. Defaults offer the named rungs from `scenarios.py` (status quo, CCC 2040, full rollout).
- The card recomputes energy, the rate, and the potential letters live from the frozen thresholds (shipped as a small JSON). The choropleth recolours via MapLibre feature-state, no re-tiling.
- **No model, in any phase** (decided 2026-07-29). The levers are closed-form and the score is measured, not modelled. Modelled form-change what-ifs are out of scope for the Atlas; the old XGBoost code stays in git history.

## Aggregation ladder and summaries

The landing view is England, not a blank map. It opens on the LAD choropleth with a national summary panel; zooming steps down LAD → LSOA → OA, and clicking any unit opens its card.

- **National panel (landing)**: England-wide numbers — the household-weighted median rate and average letter, the letter distribution (a small A–G bar), the paper's headlines (on-foot 27×, rate 3.9×), and national energy totals (TWh/yr, heat vs travel) with the savings under the named policy rungs (fabric only, heat pumps only, EVs only, CCC 2040, full rollout).
- **Region/LAD/LSOA cards**: the same panel per unit — median rate, letter mix, households, energy split, potential savings, and how the unit ranks nationally.
- **OA card**: the full NEPI label (current vs potential letters, both axes, lever sliders).
- **Live national/regional savings under the sliders**: `scenario_energy` is linear in the two uptake fractions, so any unit's total energy under arbitrary slider positions is exact from four precomputed sums per unit (baseline gas, transformed-heat gas, baseline travel, EV travel) plus unchanged electricity. No per-OA recomputation is needed above OA level. Letter-distribution shifts are not linear, so those are precomputed at the named rungs only.

All aggregates are household-weighted and computed once at export, at every level (England, LAD, LSOA), and shipped as a small static JSON alongside the tiles. There is no separate region level: the built artefacts carry no region key, and LAD-at-national-zoom covers the role.

## Build pipeline (one command)

`stats/nepi_score.py` → `oa_nepi_score.parquet` → an export script writes GeoJSON, shards the postcode index, runs tippecanoe/pmtiles, and copies the static site into `site/` (or a `site/` branch). Rebuild only when the score changes; the site itself never needs maintenance between rebuilds.

## Launch gating

Build now; **soft-launch with the preprint, full launch on acceptance** (ROADMAP Phase 1). Until then the site lives on a non-indexed preview URL.
