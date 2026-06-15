# Roadmap

Single source of truth for status, open work, and methodology decisions.
Updated 2026-06-15 (two-axis reframe; paper + Atlas deferred).

> **⏸ Current focus.** The live work is the **[argument](paper/argument.md)** and the
> **processing pipeline**. The **paper ([PAPER.md](PAPER.md)) is deferred**; the **Atlas is
> pending** — its place-scoring and XGBoost planning models are to be reevaluated for the
> two-axis frame (that code lives in git history).

## Deliverables & priority

### Current focus

1. **The argument** — canonical two-axis statement in
   [paper/argument.md](paper/argument.md). Single source of truth.
2. **The processing pipeline** + two-axis analysis layer (`stats/travel_energy.py`,
   `stats/access_profile.py`, `stats/lock_in.py`).

### ⏸ Pending (next phase)

1. **The paper** — deferred ([PAPER.md](PAPER.md)).
2. **The NEPI Atlas + planning tool** — pending: reevaluate the place-scoring and the XGBoost
   planning models for the two-axis frame (code in git history).

## Scope decisions (consumption audit)

The rebuild targets only what the two-axis analysis consumes:

- **KEEP** (load-bearing): Census 2021, DESNZ postcode metered energy, EPC
  (build year + dwelling floor area + best-fabric intensity), OS Roads/Greenspace/
  UPRN/Code-Point/BUAs, FSA, NaPTAN, GIAS, NHS, IoD 2025, DVLA vehicles (`bev_share`),
  NTS9904 mileage, ONS 2021 RUC.
- **DEFER** (heavy, unused columns): LiDAR heights + momepy morphology (~30–45 h)
  and OS Open Map Local footprints. Re-runnable via
  `pipeline run lidar morphology --include-optional`.
- **Removed from the tree** (in git history): the summed three-surface / A–G code
  (scorecard, bands, empirical access-penalty model, three-surface figures) and the
  old Atlas (XGBoost planning models + static site), taken out in the two-axis
  migration. The Atlas scoring + models are **pending reevaluation**, not retired.
  Plus the earlier archive: Census 2011, DESNZ LSOA energy, MSOA OD flows, BRES+GVA
  scaling, NESO projections, the basket index.

## Done

- **National OA pipeline** (CityNetwork API) → `oa_integrated.gpkg`.
- **Two-axis analysis** ([paper/argument.md](paper/argument.md)): NTS-anchored
  car-travel energy, lock-in (1.78× → 1.44×), per-service access profile (~10×/kWh),
  heat-vs-size decomposition — all on the shared `stats/oa_data.py` core.
- **Two-axis migration cleanup:** stripped the retired three-surface / A–G code and
  the old Atlas; unified the EPC→OA aggregation (`data/aggregate_epc_oa.py`); lean
  orchestrator (`urban_energy.pipeline`, acquire + process); `REPRODUCTION.md`.
- **Methodology #6** (Form under-recording flags) in `urban_energy.form_bias` +
  Stage 3, with tests.
- **Accessibility bands** settled on the minute-clean ladder (400/800/1600/4800/
  9600 m ≈ 5/10/20/60/120 min at ~80 m/min) — kept as-is.

## Open work — by layer

### Process / data (baked into the pipeline; changing requires a re-run)
- Climate stratification (heating degree days as a control).
- Optional: calibrate the Gaussian decay against observed travel-survey distances
  (bands themselves are settled, see Done).

### Analyse (computed in `stats/`, post-pipeline; cheap to revise — minutes)
These are the contestable scientific choices; none gate the national run.

- **Per-household vs per-capita unit.** Reported per household; household size varies
  with type (flats are smaller households than detached), so per-hh understates the
  per-capita intensity of compact types. Per-hh suits billed energy; per-capita suits
  emissions/equity. Decide: keep per-hh canonical + a per-capita view, or publish both.
- **Lock-in end-state.** Resolve the optimisation ceiling (100% HP+EV vs a realistic
  80/80) and whether to expose per-half (heat-only / travel-only) residuals.
- **Rate circularity.** Travel energy is partly the cost of low access, so the rate
  contains the inverse of its own numerator; consider rating access against heat + an
  idealised/electrified travel cost (see argument.md §7).
- **#6 follow-on.** Consider an EPC-based heat correction for the most affected OA
  classes (high-flat / off-gas); surface the `form_*` flags in the paper.
- **Spatial autocorrelation.** BUA-clustered SEs are partial; consider spatial
  error / lag models on the form/size regression.

### Forward work (out of current scope)
- **Atlas (pending):** reevaluate the place-scoring and the XGBoost planning models for the
  two-axis frame (old code in git history).
- Bettencourt scaling analysis (BRES + GVA) — source archived; revive if pursued.
- LiDAR/morphology source (LiDAR vs WALS) and sky-view-factor / shadow features —
  deferred with the morphology dimension; revisit only if it is reinstated.

### Paper / repo
- Finalise `paper/references.bib`.
- Expand the pytest suite (form-bias tests landed; pipeline/stats coverage next).
- Pre-submission cover-letter framing for the target journal.
