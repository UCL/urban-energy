# Roadmap

Single source of truth for status, open work, and methodology decisions.
Updated 2026-07-31 (manuscript in editorial review; score + Atlas built and soft-launched).

> **Current status.** The manuscript ([paper/latex/main.tex](paper/latex/main.tex)) is
> drafted, compiles clean, and is under a structural editorial pass before submission.
> The NEPI score and Atlas are built and soft-launched under `noindex`
> ([dissemination/launch_checklist.md](dissemination/launch_checklist.md)); full launch
> is gated on acceptance. Remaining before submission: Zenodo DOIs, front-matter, and
> the editorial revisions.

## The publication sequence

Three self-contained stages; each stands alone.

### Phase 0 — submit

The paper stays pure: the empirical finding (two axes, the rate, lock-in), no index
proposal, no tool promises.

1. Apply the editorial revisions; final prose pass.
2. Mint DOIs: Zenodo code release + deposit of the built per-OA artefacts.
3. Front-matter: co-author details, acknowledgements, cover letter (kept local).
4. Submit.

### Phase 1 — NEPI score + Atlas v1 (done; full launch on acceptance)

Built 2026-07-29 and deployed 2026-07-31. The score is *measured, not modelled*: the
compositional regression is for inference, the score is descriptive of the place as
lived. Specs: [dissemination/score_spec.md](dissemination/score_spec.md),
[dissemination/atlas_architecture.md](dissemination/atlas_architecture.md),
inventory at [dissemination/README.md](dissemination/README.md).

- **Score** (`stats/nepi_score.py`): per-OA A–G on the rate, bands frozen at the 2021
  household-weighted distribution ([dissemination/nepi_bands_2021.json](dissemination/nepi_bands_2021.json)).
- **Current vs potential label**: each OA scored as-lived and under full technology
  deployment (deterministic, from the `scenarios.py` levers); the gap between the two
  labels is the lock-in result made visible per neighbourhood.
- **Atlas v1** (`stats/atlas_export.py` → `site/`): map + label card + the three
  deterministic technology levers; no model anywhere. Live under `noindex` on
  Cloudflare R2; de-`noindex` and custom domain on acceptance.

### Phase 2 — the NEPI paper (after acceptance)

- Paper 2: the score design, banding rationale + sensitivity, score stability across
  scales, the equity distribution of letters, current-vs-potential label, the Atlas
  as artefact; cites paper 1 as evidence base.
- **XGBoost is dropped from the plan** (decided 2026-07-29). Nothing in the score or
  the Atlas needs a model, and modelled form-change what-ifs (densification) would
  undermine the score's measured-not-modelled identity. The old code stays in git
  history; any future densification study is a separate project that picks its own
  method.

## Scope decisions (consumption audit)

The rebuild targets only what the two-axis analysis consumes:

- **KEEP** (load-bearing): Census 2021, DESNZ postcode metered energy, EPC
  (build year + dwelling floor area + best-fabric intensity), OS Greenspace/UPRN/
  Code-Point, **OS Open Roads** (network access via cityseer), FSA (food + grocery), NaPTAN,
  GIAS, NHS, Census workplace jobs, IoD 2025, DVLA vehicles (`bev_share`),
  NTS9904 mileage, ONS 2021 RUC.
- **Removed** (nothing consumed them): LiDAR/momepy **morphology** (cityseer centrality not
  revived — accessibility only) + OS Open Map Local footprints + OS Built-Up-Areas/Boundary-Line.
  In git history.
- **Removed from the tree** (in git history): the summed three-surface / A–G code
  (scorecard, bands, empirical access-penalty model, three-surface figures) and the
  old Atlas (XGBoost planning models + static site), taken out in the two-axis
  migration. The scoring has since been rebuilt on the two-axis frame
  (`stats/nepi_score.py`); XGBoost is dropped (Phase 2 note above). Plus the earlier
  archive: Census 2011, DESNZ LSOA energy, MSOA OD flows, BRES+GVA scaling, NESO
  projections, the basket index.

## Done

- **National OA dataset** — assembled in the stats layer.
- **Network access** (`stats/oa_network_access.py`): cityseer over OS Open Roads, national network
  built **once** and queried per catchment band, each OA at its own NTS car-trip catchment,
  the built-once counts matching a literal per-OA computation to ~2% (~12 min); the rate is
  **3.9× access per kWh** (access advantage × energy saving, `access_profile.py`).
- **Two-axis analysis** ([paper/summary.md](paper/summary.md)): NTS-anchored
  car-travel energy, lock-in (per dwelling 2.12× → 1.51×; at equal family size 1.71× → 1.18×), network
  access rate (3.9× access per kWh) + on-foot gap (~27×), heat-vs-size decomposition, all on the shared
  `stats/oa_data.py` core.
- **Decarbonisation scenarios** (`stats/scenarios.py` + `paper/figures/fig8_scenarios.png`): fabric, heat pumps
  and EVs as **separate** levers over the energy axis, at CCC Seventh Carbon Budget Balanced Pathway
  2040 uptakes (heat pumps 50%, EVs 75%) and full deployment. Fabric closes ~20% of the log gap,
  heat pumps ~2% wider (a delivered-energy fuel switch that unmasks travel), EVs ~18%; full rollout
  leaves 69% surviving, access unchanged in every scenario. `lock_in.py` is retained as the
  fabric+EV bound (1.51×).
- **MAUP scale check** (`stats/maup_scale.py`): the compositional energy gap re-fit household-weighted
  at OA/LSOA/MSOA — total survives re-zoning (2.12/1.88/1.72×; support-respecting dominant-type median
  1.74/1.59/1.47×); the heat sub-component's MSOA reversal is an out-of-support extrapolation.
- **Two-axis migration cleanup:** stripped the retired three-surface / A–G code and
  the old Atlas; unified the EPC→OA aggregation (`data/aggregate_epc_oa.py`); lean
  orchestrator (`urban_energy.pipeline`, acquire-only); `REPRODUCTION.md`.
- **Accessibility bands** settled on the minute-clean ladder (400/800/1600/4800/
  9600 m ≈ 5/10/20/60/120 min at ~80 m/min) — kept as-is.
- **NEPI score + Atlas + site** (2026-07-29 → 2026-07-31): `stats/nepi_score.py`
  (bands frozen), `stats/atlas_export.py` (aggregates, PMTiles, postcode shards),
  static site in `site/`, deployed to Cloudflare R2 under `noindex`.

## Open work — by layer

### Data (changing requires re-acquiring)

- None outstanding. Climate (HadUK-Grid heating-degree-days) is acquired (`oa_hdd.parquet`) and held
  as a confound in the energy ladder.

### Analyse (computed in `stats/`; cheap to revise — minutes)

These are the contestable scientific choices; none gate acquisition.

- **Per-household vs per-capita unit.** Resolved in the manuscript: per dwelling is
  canonical, with household size and floor area as freely estimated controls (elasticities
  ~0.5 and <1, so fixed denominators are rejected). The paper reports the as-lived and
  equal-household-size views side by side; per-person and per-m² appear only in the
  Discussion as the units under which the gap narrows. The NEPI score inherits the
  single as-lived per-dwelling mode (dissemination/score_spec.md §Units).
- **Lock-in end-state.** Resolved. `scenarios.py` reports the full ladder (each lever alone at 100%,
  the CCC Balanced Pathway 2040 mix, and full deployment) rather than a single ceiling; `lock_in.py`
  stays as the fabric+EV bound (1.51×).
- **Rate circularity.** Travel energy is partly the cost of low access, so the rate
  contains the inverse of its own numerator; consider rating access against heat + an
  idealised/electrified travel cost (see summary.md §7). A circularity-robust rate (2.33×) is already
  reported alongside the headline in `access_profile.py`.
- **Under-recording robustness.** Addressed. `form_size_decomposition.py` §7 reports the gas-meter
  coverage and electricity-meter denominator checks (heat gap 1.61× at coverage ≥0.9, 1.55× at
  elec-meters ≈ households) plus winsorisation; the gradient is not driven by under-recording.
- **Spatial autocorrelation & MAUP.** LAD-clustered SEs are the delivered primary inference; the MAUP
  scale check is done (`maup_scale.py`). What remains open is an optional spatial error / lag model as
  a cross-check.

### Forward work (out of current scope)

- Bettencourt scaling analysis (BRES + GVA) — source archived; revive if pursued.
- Morphology features (LiDAR/momepy network centrality) — only if a future Atlas needs them.
  (The network-distance access measure is done — `stats/oa_network_access.py`, summary.md §3.)

### Paper / repo

- Finalise `paper/references.bib` (data-source entries; NEED/EHS added).
- Pytest suite at 49 tests (compositional model, travel constraint, inference, aggregation,
  EPC bands, postcode, NEPI score/bands); scenario-ladder and access-ratio coverage still pending.
