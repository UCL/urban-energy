# Roadmap

Single source of truth for status, open work, and methodology decisions.
Updated 2026-06-15 (two-axis reframe; paper + Atlas deferred).

> **⏸ Current focus.** The live work is the **[argument](paper/summary.md)** and the
> **data + analysis pipeline**. The **paper ([PAPER.md](PAPER.md))** is drafted on the two-axis frame (finalisation pending); the **Atlas is
> pending** — its place-scoring and XGBoost planning models are to be reevaluated for the
> two-axis frame (that code lives in git history).

## Deliverables & priority

### Current focus

1. **The argument** — canonical two-axis statement in
   [paper/summary.md](paper/summary.md). Single source of truth.
2. **The data + analysis pipeline** — `oa_data` + `oa_access` → `travel_energy`,
   `access_profile`, `lock_in`, `form_size` (assembled in the stats layer, no network run).

### ⏸ Pending (next phase)

1. **The paper** ([PAPER.md](PAPER.md)): drafted on the two-axis frame and matching the canonical
   `summary.md` numbers; finalisation and submission pending. The old three-surface draft is archived
   at [paper/archive/PAPER_three_surface_deferred.md](paper/archive/PAPER_three_surface_deferred.md).
2. **The NEPI Atlas + planning tool** — pending: reevaluate the place-scoring and the XGBoost
   planning models for the two-axis frame (code in git history).

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
  migration. The Atlas scoring + models are **pending reevaluation**, not retired.
  Plus the earlier archive: Census 2011, DESNZ LSOA energy, MSOA OD flows, BRES+GVA
  scaling, NESO projections, the basket index.

## Done

- **National OA dataset** — assembled in the stats layer.
- **Network access** (`stats/oa_network_access.py`): cityseer over OS Open Roads, national network
  built **once** and queried per catchment band, each OA at its own NTS car-trip catchment,
  the built-once counts matching a literal per-OA computation to ~2% (~12 min); the rate is
  **3.6× access per kWh** (access advantage × energy saving, `access_profile.py`).
- **Two-axis analysis** ([paper/summary.md](paper/summary.md)): NTS-anchored
  car-travel energy, lock-in (per dwelling 2.12× → 1.51×; at equal family size 1.71× → 1.18×), network
  access rate (3.6× access per kWh) + on-foot gap (~24×), heat-vs-size decomposition, all on the shared
  `stats/oa_data.py` core.
- **Decarbonisation scenarios** (`stats/scenarios.py` + `scenario_ladder.png`): fabric, heat pumps
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

## Open work — by layer

### Data (changing requires re-acquiring)
- None outstanding. Climate (HadUK-Grid heating-degree-days) is acquired (`oa_hdd.parquet`) and held
  as a confound in the energy ladder.

### Analyse (computed in `stats/`; cheap to revise — minutes)
These are the contestable scientific choices; none gate acquisition.

- **Per-household vs per-capita unit.** Reported per household; household size varies
  with type (flats are smaller households than detached), so per-hh understates the
  per-capita intensity of compact types. Per-hh suits billed energy; per-capita suits
  emissions/equity. Decide: keep per-hh canonical + a per-capita view, or publish both.
- **Lock-in end-state.** Resolved. `scenarios.py` reports the full ladder (each lever alone at 100%,
  the CCC Balanced Pathway 2040 mix, and full deployment) rather than a single ceiling; `lock_in.py`
  stays as the fabric+EV bound (1.51×).
- **Rate circularity.** Travel energy is partly the cost of low access, so the rate
  contains the inverse of its own numerator; consider rating access against heat + an
  idealised/electrified travel cost (see summary.md §7). A circularity-robust rate (2.17×) is already
  reported alongside the headline in `access_profile.py`.
- **Under-recording robustness.** Addressed. `form_size_decomposition.py` §7 reports the gas-meter
  coverage and electricity-meter denominator checks (heat gap 1.61× at coverage ≥0.9, 1.55× at
  elec-meters ≈ households) plus winsorisation; the gradient is not driven by under-recording.
- **Spatial autocorrelation & MAUP.** LAD-clustered SEs are the delivered primary inference; the MAUP
  scale check is done (`maup_scale.py`). What remains open is an optional spatial error / lag model as
  a cross-check.

### Forward work (out of current scope)
- **Atlas (pending):** reevaluate the place-scoring and the XGBoost planning models for the
  two-axis frame (old code in git history).
- Bettencourt scaling analysis (BRES + GVA) — source archived; revive if pursued.
- Morphology features (LiDAR/momepy network centrality) — only if a future Atlas needs them.
  (The network-distance access measure is now done — `stats/oa_network_access.py`, summary.md §3.)

### Paper / repo
- Finalise `paper/references.bib`.
- (Re)establish the pytest suite: none currently (the earlier form-bias tests were removed); pipeline and stats coverage pending.
- Pre-submission cover-letter framing for the target journal.
