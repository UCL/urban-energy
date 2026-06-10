# NEPI — Step-by-Step Robustness Plan

A sequenced, executable plan to improve the robustness of the NEPI argument,
derived from [methodology_review.md](methodology_review.md). Ordered by
**dependency** then **return-on-effort**.

**Read this first:** Phase 0 must precede everything else — those are *construction
corrections that change the headline numbers*, so any robustness test run before
them is wasted. Phases 1–2 are where most of the value is, and **most of them use
data already in the repository**. Phase 4 is text/framing only (no computation).

Legend — Return: ●●● high / ●● med / ● low · Effort: ◯ low / ◯◯ med / ◯◯◯ high ·
"in-repo" = needs no new data.

---

## Phase 0 — Construction corrections (PREREQUISITE — these move the numbers)

**Step 1 — Reconstruct Form as genuine per-household.** ●●● ◯◯ · in-repo
- *Do:* in `aggregate_energy_oa.py`, carry postcode **totals** (or mean×num_meters), and compute OA energy = (Σ gas kWh + Σ elec kWh) ÷ Census households (TS017), using a **common household denominator for both fuels**.
- *Why:* `building_kwh_per_hh` is currently kWh **per meter** (no division by households), summed across two different meter denominators; the meter→household ratio is morphology-correlated.
- *Then:* re-run the energy aggregation + Stage-3 merge, regenerate the headline. Expect the Form gradient and band cut-points to move.

**Step 2 — Reconcile the Access/Mobility double-count.** ●●● ◯ · in-repo
- *Do:* in `nepi.py` (composite at line 226), either (a) present Access as a shaded **attribution within** the Mobility bar (not summed on top), or (b) redefine Mobility as the *reference-coverage* prediction so `Mobility + Access` is non-overlapping.
- *Why:* Access is a regression slice of the Mobility variable; summing them double-counts transport energy in the composite and the 14% Access share.

**Step 3 — Make the 85% Access reference a transparent parameter; reconsider the zero-clip.** ●● ◯ · in-repo
- *Do:* report Access under references {national-median, 75th, 85th, 95th pct}; decide whether the one-sided `clip(lower=0)` stays (document it) or becomes symmetric (rewarding good access).

---

## Phase 1 — Quick, high-return robustness (in-repo data + existing code)

**Step 4 — Confound controls on Form: climate, floor area, tenure, occupancy.** ●●● ◯◯
- *Data:* HDD/CDD grid (HadUK-Grid 1 km or ERA5 — *one new download*) at OA centroid + altitude (LiDAR DTM, in-repo); EPC `TOTAL_FLOOR_AREA` (in-repo); Census TS054 tenure, TS050/TS017 occupancy (in-repo).
- *Do:* stepwise add HDD → floor area → tenure → occupancy; report the Flat/Detached Form ratio before/after; report **both** per-household and per-m² (floor-area-controlled).
- *Defends:* the two most obvious reviewer kill-shots (climate confound; floor-area OVB). Benchmark per-m² against Liddiard et al. (2021): gas EUI 162/164/157/138 by form.

**Step 5 — Self-selection proxies + upper-bound framing in the Access/Mobility OLS.** ●●● ◯ · in-repo
- *Do:* add tenure (TS054), household-size (TS017), children/age structure as covariates; state estimates are **upper bounds**. Report coefficient movement.

**Step 6 — Sensitivity-to-unobservables: Oster bounds + E-value + Cinelli–Hazlett.** ●●● ◯ · in-repo
- *Do:* from the existing M1–M5 control progression (`run_regressions` already prints β and R² movement), compute **Oster (2019) δ** and bias-adjusted β; add **E-value** (VanderWeele & Ding 2017) and **sensemakr** benchmarking vs the income covariate.
- *Defends:* this is the test reviewers actually want; demote the bootstrap CIs.

**Step 7 — Fix the flagship Access-model standard errors.** ●●● ◯◯ · in-repo
- *Do:* re-estimate the penalty OLS with BUA fixed effects (reuse the FWL demeaning at `proof_of_concept_oa.py:975`) + BUA-clustered SEs + **Conley spatial-HAC**; compute Moran's I on **residuals** (not raw variables); add LM-lag/LM-error.
- *Defends:* the headline `t = −161` magnitudes are currently computed under HC1-only, no FE, no spatial correction.

**Step 8 — BUA-block bootstrap replacing the iid bootstrap.** ●● ◯ · in-repo
- *Do:* resample **whole BUAs**; report the honest (5–20× wider) CIs.

**Step 9 — Continuous-share dose–response replacing the plurality "steepening" test.** ●● ◯ · in-repo
- *Do:* regress energy on the continuous `pct_flat`/`pct_detached` over the **full, fixed** sample.
- *Defends:* against the discard-selection / regression-to-extremes artifact in §5.2.

---

## Phase 2 — MAUP, spatial models, out-of-sample validation (mostly in-repo)

**Step 10 — MAUP: scale ladder + zonation + band stability.** ●●● ◯◯ · in-repo
- *Data:* the LSOA pipeline + `lsoa_energy_consumption.parquet` (archived but present); a regular grid.
- *Do:* recompute the gradient at OA/LSOA/MSOA; aggregate to a 1 km grid and random equal-population partitions; cross-tabulate each OA's band at OA vs LSOA and report the % shifting ≥1 band. Relabel §5.2.

**Step 11 — Out-of-sample validation of the composite and surfaces.** ●●● ◯◯ · in-repo
- *Data:* `lsoa_energy_consumption.parquet` (independent aggregation), `epc_domestic_spatial.parquet` (SAP), `lsoa_vehicles.parquet` (DVLA), NTS.
- *Do:* correlate Form vs LSOA-energy and EPC by dwelling type (also *measures* the off-gas/communal compression); validate Mobility/Access vs DVLA cars-per-LSOA and NTS area-type travel **out of sample**; spatial CV (train North / test South).
- *Defends:* currently there is zero ground-truth validation.

**Step 12 — Spatial model + spatial block CV.** ●● ◯◯
- *Do:* fit `spreg.GM_Error`/`GM_Lag` (GMM, scales to large N) or eigenvector spatial filtering on the building-energy model; switch the XGBoost tools from random `train_test_split` to **GroupKFold grouped on BUA**; report the random-vs-spatial CV gap.

**Step 13 — Accessibility decay calibration + sensitivity.** ●● ◯◯
- *Do:* calibrate β by the **half-life method** (Östh, Lyhagen & Reggiani 2016) so the modelled median walk distance matches the NTS observed median; sensitivity over decay form (exponential/Gaussian/log-normal — Chen 2015) and ±50% thresholds; report band-assignment stability. Cite a walk-speed norm for 4.8 km/h.

**Step 14 — Decomposition-share confidence intervals.** ●● ◯◯ · in-repo
- *Do:* block-bootstrap the whole pipeline (re-fitting the penalty model inside each replicate) to put 95% CIs on the Form/Mobility/Access shares (currently "45/43/14%" has none).

**Step 15 — Spatial heterogeneity.** ●● ◯◯ · in-repo
- *Do:* random-slope multilevel model (OA in BUA/region) — the random-slope variance is the heterogeneity measure; promote `run_per_city` into a figure; stratify by off-gas share.

---

## Phase 3 — Index methodology + remaining sensitivities

**Step 16 — Absolute (criterion-referenced) bands.** ●●● ◯◯ · in-repo
- *Do:* set A–G by **absolute kWh thresholds** anchored to a meaningful level (e.g. a net-zero-compatible household budget), so NEPI can track progress like an EPC; keep percentile rank as a secondary "national position" indicator. If percentile bands are retained, rename them and **freeze cut-points from a base year**. Report band stability under the Phase 0/1 fixes.

**Step 17 — Carbon- and cost-weighted composite sensitivity.** ●● ◯ · in-repo
- *Do:* recompute the composite weighting surfaces by kgCO₂ and by £ (the Atlas already carries these factors); drop the "no weighting" claim; show whether the A–G **ranking** is robust to the weighting metric (a stronger claim than "no weighting").

**Step 18 — Specification-curve analysis.** ●● ◯◯◯ · in-repo
- *Do:* report the headline gap and Access share across the full grid of defensible choices (9 decay thresholds, 85% reference, zero-clip, 6.04× NTS scalar, band cut-points, build-year bins, road intensity).

**Step 19 — Placebo / falsification tests.** ●● ◯ · in-repo
- *Do:* permute coverage within BUA (penalty should collapse to ~0); regress an implausible outcome on coverage with the same controls.

**Step 20 — Mobility refinements.** ●● ◯◯ · partly in-repo
- *Do:* source a **morphology-stratified** total/commute scalar from NTS area-type bands; report the OD-distance Mobility (already computed, §5.6) alongside the band-midpoint; verify car *passengers* aren't double-charged at road intensity; consider making the pre-pandemic (2011-mode-share) Mobility canonical.

---

## Phase 4 — Framing & positioning (text + literature; no computation)

**Step 21 — Self-selection paragraph (the single highest-value text addition).** Name it up front; state the bias is **upward** (estimates are upper bounds); cite Mokhtarian & Cao (2008), Cao-Mokhtarian-Handy (2009), the Stevens↔Ewing&Cervero (2017) exchange, and the causal-ML corroboration (Wagner 2023; Nachtigall 2023 — ~half the gap is self-selection, 73.7% of the BE effect is accessibility).

**Step 22 — Lock-in: two-tier framing.** Adopt Seto et al. (2016) infrastructural/institutional/behavioural taxonomy; cite Boeing (2021, street layout persists "for centuries") and Barrington-Leigh & Millard-Ball (2019); state the **durability hierarchy** (layout ≫ buildings ≫ vehicles); separate the durable substrate from the malleable behavioural outcome; temper "no policy addresses" → "**under-addressed** by current decarbonisation policy."

**Step 23 — Scaling: mechanism, not magnitude.** Keep Bettencourt (2007) as motivation but cite the critiques (Leitão 2016; Arcaute 2015; Cottineau 2017) and explicitly disavow any OA-level scaling exponent (OAs are not cities; exponents are city-definition/MAUP-sensitive).

**Step 24 — Add an energy-justice section.** Structure on Jenkins et al.'s (2016) three tenets (distribution / recognition / procedure); cite Bouzarovski & Simcock (2017) and Walker & Day (2012); add an **anti-stigma safeguard** (Wacquant 2007 — grades are a property of *form*, not residents; G-areas are *priority-for-support*, not *failing*); flag the **regressive distributional risk** of area labels with EPC capitalization evidence (Fuerst et al. 2015; Brounen & Kok 2011).

**Step 25 — Ecological/MAUP caveat + temper individual-level language.** Cite Robinson (1950), Openshaw (1984), Greenland & Morgenstern (1989), Subramanian et al. (2009); phrase consistently as "band-G *areas* exhibit," never "households in band G consume."

**Step 26 — Form-surface framing.** Present the 1.46× as a *realised* gradient (Liddiard 2021; Steemers 2008; Huebner 2015) contrasted with Rode et al.'s (2014) *modelled* six-fold ceiling; close the metered-over-EPC argument (Few 2023; Summerfield 2019; Firth 2024; Crawley 2019; Sunikka-Blank & Galvin 2012); position novelty against CNT's H+T Index, NEBULA (2025), Feng et al. (2025).

---

## Suggested execution order & "minimum viable for resubmission"

If the goal is a defensible resubmission with the least work, the **critical path** is:

**Phase 0 (Steps 1–2)** → **Phase 1 (Steps 4–7)** → **Phase 2 (Steps 10–11)** → **Step 16** → **Phase 4 framing (Steps 21–25)**.

That sequence fixes the numbers, controls the two killer confounds, gives the
sensitivity-to-unobservables reviewers demand, adds the missing out-of-sample
validation and the MAUP check, corrects the banding category error, and aligns
the language with the design — and **every computational step except the HDD grid
uses data already in the repository.**

The remaining steps (8–9, 12–15, 17–20) are valuable hardening that can follow.

---

## Map to the methodology review

| Plan step | Review finding |
|---|---|
| 1 | A1 (Form per-meter) |
| 2, 3 | A2, A3 (double-count, reference) |
| 4 | D1, D2 (climate, floor area) |
| 5, 21 | B2 (self-selection) |
| 6 | B5 (sensitivity-to-unobservables) |
| 7, 12 | C3 (spatial inference) |
| 8, 14 | D5 (bootstrap, decomposition CI) |
| 9 | C4 (plurality artifact) |
| 10 | C2 (MAUP) |
| 11 | D4 (validation) |
| 13 | D3 (decay calibration) |
| 15 | C5 (heterogeneity) |
| 16 | E1 (banding) |
| 17 | E2 (weighting) |
| 18, 19 | D5 (spec curve, placebo) |
| 20 | E4 (Mobility) |
| 22 | B4 (lock-in) |
| 23 | scaling strand |
| 24 | energy-justice strand |
| 25 | C1 (ecological inference) |
| 26 | Part F (positioning) |
