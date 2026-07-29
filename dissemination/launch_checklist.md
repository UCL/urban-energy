# Atlas soft-launch checklist (Phase 1)

The Atlas builds now and launches in two stages: soft-launch with the preprint, full launch on acceptance (ROADMAP Phase 1). Everything below the "user actions" line is prepared in-repo.

## Rebuild recipe

```bash
uv run python stats/nepi_score.py     # re-score (bands stay frozen)
uv run python stats/atlas_export.py   # aggregates + tiles + postcode shards → docs/
npx http-server docs -p 8000          # local preview at http://localhost:8000
```

The preview server must support HTTP `Range` requests (PMTiles fetches byte ranges); `npx http-server` does, Python's `http.server` does not. GitHub Pages and Cloudflare Pages both do.

```bash
```

Heavy outputs (`docs/tiles/`, `docs/data/pc/`) are gitignored and regenerable; the site source (`index.html`, `app.js`, `style.css`, `about.html`, `vendor/`) is tracked.

## Prepared

- Static site in `docs/`: map (LAD/LSOA/OA zoom ladder), England panel with live lever totals, unit cards, OA label card, postcode search, about page. No backend, no CDN dependencies (MapLibre 4.7.1 and pmtiles 3.2.1 vendored).
- `noindex` meta tags on both pages — the soft-launch guard.

## Soft-launch (with the preprint) — user actions

1. Choose hosting: GitHub Pages (repo → Settings → Pages → deploy from `docs/`, requires committing the heavy artefacts or publishing via an Actions artifact step) or Cloudflare Pages (`wrangler pages deploy docs/`; note the 25 MB per-file limit — check the OA archive size first, GitHub's limit is 100 MB).
2. Deposit code + data on Zenodo (submission checklist items 2–3) and paste the DOIs into `about.html` §The research.
3. Post the preprint; add its link to `about.html`.
4. Announce quietly if desired (the `noindex` tags stay).

## Full launch (on acceptance) — user actions

1. Remove the two `noindex` meta tags; rebuild nothing else.
2. Swap the preprint link for the published DOI.
3. Press/posts per the dissemination inventory (UCL press office, personal post).

## Basemap dependency

The map draws a Carto Positron raster basemap (OSM-derived) for orientation, with the grade fills at 55% opacity over it. This is the site's one external runtime dependency; attribution is included. For full launch, consider replacing it with a self-hosted OS Open Zoomstack layer to return to zero external dependencies.

## Known limits (state honestly if asked)

- No region level (no region key in the built artefacts); the ladder is England → LAD → LSOA → OA.
- Unit-level slider response scales medians by linear-sum ratios — exact for EV within homogeneous fleets, approximate otherwise; OA-level arithmetic is exact.
- Letter distributions under arbitrary slider positions are shown only at OA level; unit bars show as-lived and full-deployment only.
