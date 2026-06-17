# Roadmap

Single source of truth for status, open work, and methodology decisions.
Updated 2026-06-15 (two-axis reframe; paper + Atlas deferred).

> **⏸ Current focus.** The live work is the **[argument](paper/summary.md)** and the
> **data + analysis pipeline**. The **paper ([PAPER.md](PAPER.md)) is deferred**; the **Atlas is
> pending** — its place-scoring and XGBoost planning models are to be reevaluated for the
> two-axis frame (that code lives in git history).

## Deliverables & priority

### Current focus

1. **The argument** — canonical two-axis statement in
   [paper/summary.md](paper/summary.md). Single source of truth.
2. **The data + analysis pipeline** — `oa_data` + `oa_access` → `travel_energy`,
   `access_profile`, `lock_in`, `form_size` (assembled in the stats layer, no network run).

### ⏸ Pending (next phase)

1. **The paper** — deferred ([PAPER.md](PAPER.md)).
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
  built **once** and queried per catchment band, each OA at its own NTS car-trip catchment;
  rate **~2.9×/kWh**, validated to ~2% of a literal per-OA computation (~12 min).
- **Two-axis analysis** ([paper/summary.md](paper/summary.md)): NTS-anchored
  car-travel energy, lock-in (1.74× → 1.47×), network access rate (~2.9×/kWh) + walkable
  richness (~10×), heat-vs-size decomposition — all on the shared `stats/oa_data.py` core.
- **Two-axis migration cleanup:** stripped the retired three-surface / A–G code and
  the old Atlas; unified the EPC→OA aggregation (`data/aggregate_epc_oa.py`); lean
  orchestrator (`urban_energy.pipeline`, acquire-only); `REPRODUCTION.md`.
- **Accessibility bands** settled on the minute-clean ladder (400/800/1600/4800/
  9600 m ≈ 5/10/20/60/120 min at ~80 m/min) — kept as-is.

## Open work — by layer

### Data (changing requires re-acquiring)
- Climate stratification (heating degree days as a control).

### Analyse (computed in `stats/`; cheap to revise — minutes)
These are the contestable scientific choices; none gate acquisition.

- **Per-household vs per-capita unit.** Reported per household; household size varies
  with type (flats are smaller households than detached), so per-hh understates the
  per-capita intensity of compact types. Per-hh suits billed energy; per-capita suits
  emissions/equity. Decide: keep per-hh canonical + a per-capita view, or publish both.
- **Lock-in end-state.** Resolve the optimisation ceiling (100% HP+EV vs a realistic
  80/80) and whether to expose per-half (heat-only / travel-only) residuals.
- **Rate circularity.** Travel energy is partly the cost of low access, so the rate
  contains the inverse of its own numerator; consider rating access against heat + an
  idealised/electrified travel cost (see summary.md §7).
- **Under-recording robustness.** Flat metered energy omits communal/bulk gas (district
  heating); detached omits off-gas. Re-add an under-recording check before publishing the
  heat gradient (the old `form_bias` flags were removed with the pipeline).
- **Spatial autocorrelation.** Consider spatial error / lag models (or spatially-clustered SEs)
  on the form/size regression.

### Forward work (out of current scope)
- **Atlas (pending):** reevaluate the place-scoring and the XGBoost planning models for the
  two-axis frame (old code in git history).
- Bettencourt scaling analysis (BRES + GVA) — source archived; revive if pursued.
- Morphology features (LiDAR/momepy network centrality) — only if a future Atlas needs them.
  (The network-distance access measure is now done — `stats/oa_network_access.py`, summary.md §3.)

### Paper / repo
- Finalise `paper/references.bib`.
- Expand the pytest suite (form-bias tests landed; pipeline/stats coverage next).
- Pre-submission cover-letter framing for the target journal.
