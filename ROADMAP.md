# Roadmap

Single source of truth for status and open work. Folds the former README
Done/Open lists and cross-references the detailed methodology write-ups in
[TODO.md](TODO.md). Updated 2026-06-09 (lean-pipeline cleanup).

## Deliverables & priority

1. **The paper** — IMRaD case in [PAPER.md](PAPER.md). Primary scholarly output.
2. **The NEPI Atlas** — public, live on GitHub Pages. Primary impact artifact.
3. **The NEPI planning tool** — four XGBoost models, embedded in the Atlas
   (static HTML/JS) and available as a Streamlit app.

## Scope decisions (2026-06-09 consumption audit)

The rebuild targets only what the deliverables consume. Full evidence in the
audit; summary:

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

## Open — by layer

### Process / data
- **TODO #4 (LiDAR vs WALS)** and **#5 (sky-view-factor / shadow)** — deferred
  with the LiDAR path. Revisit only if the morphology dimension is reinstated.
- Climate stratification (heating degree days as a control).
- Calibrate Gaussian decay thresholds against observed travel-survey distances.

### Analyse (post-rebuild; no pipeline re-run needed)
- **TODO #1** — per-household vs per-capita canonical unit.
- **TODO #2** — critical interpretation of SHAP / gradient-booster outputs.
- **TODO #3** — lock-in surface: audience, definition, per-pillar unpacking.
- **TODO #6 follow-on** — surface the new `form_*` flags on the Atlas About page
  and in the paper; consider an EPC-based correction for the most affected OAs.
- Spatial autocorrelation: BUA-clustered SEs are partial; consider spatial
  error / lag models.

### Forward work (out of current scope)
- Bettencourt scaling analysis (BRES + GVA) — source archived; revive if pursued.
- DVLA fleet-electrification scenarios for lock-in quantification.

### Paper / repo
- Reconcile or retire `paper/archive/main.tex`; finalise `paper/references.bib`.
- Expand the pytest suite (form-bias tests landed; pipeline/stats coverage next).
- Pre-submission cover-letter framing for the target journal.
