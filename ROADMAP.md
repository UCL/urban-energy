# Roadmap

Single source of truth for status, open work, and methodology decisions.
Updated 2026-06-15 (two-axis reframe; paper + Atlas deferred).

> **⏸ Current focus.** The live work is the **[argument](paper/argument.md)** (canonical
> two-axis statement) and the **processing pipeline** — making both watertight. **The paper
> ([PAPER.md](PAPER.md)) and the Atlas / A–G planning tool are DEFERRED** to a later phase;
> they still carry the old *three-surface / A–G* framing. Sections below describing the A–G
> scorecard, bands, surface decomposition and the empirical access-penalty model refer to that
> **deferred** legacy layer.

## Deliverables & priority

### Current focus

1. **The argument** — canonical two-axis statement in
   [paper/argument.md](paper/argument.md). Single source of truth.
2. **The processing pipeline** + two-axis analysis layer (`stats/travel_energy.py`,
   `stats/access_profile.py`, `stats/lock_in.py`).

### ⏸ Deferred (next phase — old three-surface / A–G framing)

1. **The paper** — IMRaD case in [PAPER.md](PAPER.md); rewrite to two axes.
2. **The NEPI Atlas + planning tool** — public A–G dashboard + four XGBoost models;
   migrate off the three-surface framing.

## Scope decisions (2026-06-09 consumption audit)

The rebuild targets only what the deliverables consume:

- **KEEP** (load-bearing): Census 2021, DESNZ postcode metered energy, EPC
  (build-year only), OS Roads/Greenspace/UPRN/Code-Point/Boundary-Line/BUAs,
  FSA, NaPTAN, GIAS, NHS, IoD 2025, DVLA vehicles, NESO projections.
- **DEFER** (heavy, unused columns): LiDAR heights + momepy morphology (~30–45 h)
  and OS Open Map Local footprints. Only `height_mean` is consumed (one figure
  table cell). Re-runnable via `pipeline run lidar morphology --include-optional`.
- **CUT** (archived, zero live reads): Census 2011, DESNZ LSOA energy, MSOA OD
  flows, BRES+GVA scaling, the basket index. See the `*/archive/README.md` files.

## Done

- National OA pipeline (CityNetwork API), NEPI scorecard + A–G bands + surface
  decomposition, empirical access-penalty OLS, four monotonic XGBoost models +
  SHAP, Streamlit + static HTML/JS tool, Atlas live on GitHub Pages.
- **Lean-pipeline cleanup:** consumption audit; CUT scripts archived; live
  pipeline constants extracted out of `archive/` into `processing/common.py`;
  executable orchestrator (`urban_energy.pipeline`); `REPRODUCTION.md`;
  static-tool export promoted to `stats/export_static_tool.py`.
- **Methodology #6** (Form under-recording flags) implemented in
  `urban_energy.form_bias` + Stage 3, with tests.
- **Accessibility bands** settled on the minute-clean ladder (400/800/1600/4800/
  9600 m ≈ 5/10/20/60/120 min at ~80 m/min) — kept as-is.

## Open work — by layer

### Process / data (baked into the pipeline; changing requires a re-run)
- Climate stratification (heating degree days as a control).
- Optional: calibrate the Gaussian decay against observed travel-survey distances
  (bands themselves are settled, see Done).

### Analyse (computed in `stats/`, post-pipeline; cheap to revise — minutes)
These are the contestable scientific choices; none gate the national run.

- **Per-household vs per-capita unit.** NEPI is published per household. Household
  size varies with archetype (flats are smaller households than detached), so
  per-hh understates the per-capita intensity of compact types. Per-hh is natural
  for billed energy; per-capita for emissions/equity. Decide: keep per-hh canonical
  + a per-capita toggle / switch / publish both.
- **SHAP interpretation.** SHAP attributions are conditional on the model's joint
  feature distribution, not structural causal effects — co-linear features
  (density, type, build year) share explanatory power. The workbench "vs actual"
  baseline is the mid-archetype interpolation (a convenience reference, not a
  counterfactual). Document this; consider exposing alternative baselines.
- **Lock-in surface.** Resolve audience (planners siting new build vs analysts
  targeting retrofit), end-state definition (100% HP+EV vs a realistic 80/80
  ceiling), and whether to expose per-pillar (form-only / mobility-only) residuals.
- **#6 follow-on.** Surface the new `form_*` under-recording flags on the Atlas
  About page and in the paper; consider an EPC-based Form correction for the most
  affected OA classes (high-flat / off-gas).
- **Spatial autocorrelation.** BUA-clustered SEs are partial; consider spatial
  error / lag models.

### Forward work (out of current scope)
- Bettencourt scaling analysis (BRES + GVA) — source archived; revive if pursued.
- DVLA fleet-electrification scenarios for lock-in quantification.
- LiDAR/morphology source (LiDAR vs WALS) and sky-view-factor / shadow features —
  deferred with the morphology dimension; revisit only if it is reinstated.

### Paper / repo
- Finalise `paper/references.bib`.
- Expand the pytest suite (form-bias tests landed; pipeline/stats coverage next).
- Pre-submission cover-letter framing for the target journal.
