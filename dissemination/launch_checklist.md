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

## Hosting (decided 2026-07-31): all-R2

The existing Cloudflare R2 bucket `nepi-atlas` (public) hosts the whole site: sync all of `docs/` into it. The tile URLs in `app.js` are relative, so no code change is needed. The bucket still holds the May 2026 `england_oa.pmtiles` (259.72 MB, old A–G schema) — delete it after the new upload; the new site cannot read it. GitHub Pages was rejected (would commit 153 MB of regenerable binaries per rebuild); Cloudflare Pages direct deploy is blocked by its 25 MB per-file cap (`oa.pmtiles` is 68 MB). The `r2.dev` public URL is rate-limited by Cloudflare — acceptable while the site is quiet under `noindex`; attach a custom domain to the bucket at full launch.

## Short term — user actions (no preprint needed; `noindex` stays)

1. Browser pass of the preview (`npx http-server docs -p 8000`): map loads, England → LAD → LSOA → OA ladder switches, levers update the panel totals, postcode search resolves, OA card renders, about/sources pages read correctly.
2. Deploy: sync `docs/` to the `nepi-atlas` bucket; delete the stale `england_oa.pmtiles`.

## Long term (deferred until the preprint exists) — user actions

1. Deposit code + data on Zenodo (submission checklist items 2–3) and paste the DOIs into `about.html` §The research.
2. Post the preprint; add its link to `about.html`.
3. Announce quietly if desired (the `noindex` tags stay).
4. Submission front-matter, reviewed internally before submission: co-author surnames, acknowledgements, suggested referees.

## Full launch (on acceptance) — user actions

1. Remove the two `noindex` meta tags; rebuild nothing else.
2. Swap the preprint link for the published DOI.
3. Attach a custom domain to the R2 bucket (replaces the rate-limited `r2.dev` URL).
4. Press/posts per the dissemination inventory (UCL press office, personal post).

## Basemap dependency

The map draws a Carto Positron raster basemap (OSM-derived) for orientation, with the grade fills at 55% opacity over it. This is the site's one external runtime dependency; attribution is included. For full launch, consider replacing it with a self-hosted OS Open Zoomstack layer to return to zero external dependencies.

## Known limits (state honestly if asked)

- No region level (no region key in the built artefacts); the ladder is England → LAD → LSOA → OA.
- Unit-level slider response scales medians by linear-sum ratios — exact for EV within homogeneous fleets, approximate otherwise; OA-level arithmetic is exact.
- Letter distributions under arbitrary slider positions are shown only at OA level; unit bars show as-lived and full-deployment only.
