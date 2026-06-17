# NEPI — Methodological Rigour Review

> **⏸ Predates the two-axis migration.** This audit was written against the old
> three-surface / A–G analysis, much of which has since been **removed** (its file:line
> citations point at deleted code). Kept for the methodological concerns that carry over to
> the two-axis work (spatial autocorrelation, climate/tenure confounds, the per-household fix);
> [`summary.md`](summary.md) is the current canonical statement.

An intensive, literature-grounded audit of the NEPI methodology, identifying
opportunities to strengthen the rigour of the argument. Produced 2026-06-10 from
a fan-out of eight specialist reviews (four internal methodology critics, four
external literature reviewers). Findings that **multiple independent reviewers
reached separately** are flagged ⟂ (high confidence by triangulation).

---

## Verdict in one paragraph

The **substantive thesis is well-founded and on-trend**: that *destination
accessibility*, more than raw density, governs transport energy is the single
most robust finding in the built-environment–travel literature (Ewing & Cervero
2010; NRC SR 298 2009) and survives the harshest self-selection corrections
(Stevens 2017) and the newest causal-ML designs (Wagner et al. 2023; Nachtigall
et al. 2023). NEPI's emphasis is *vindicated* by the evidence base. The exposure
is entirely in the **gap between the descriptive/ecological design and the
causal-flavoured headline** ("penalty", "attributable to", "locked in",
"the surface technology cannot fix"). Two of the problems are **construction
errors that change the numbers, not just caveats** — fix these before anything
else. The rest are missing confound controls, un-tested robustness, and an
EPC-analogy that is, strictly, a category error. None of this sinks the paper;
all of it is addressable, much with data already in the repository.

---

## PART A — Construction issues that change the numbers (fix first)

### A1. The Form surface is kWh **per meter**, not per household ⟂ (CRITICAL)
`building_kwh_per_hh` is assigned directly from `oa_total_mean_kwh`
(`proof_of_concept_oa.py:349`) with **no division by household count**.
`oa_total_mean_kwh` is DESNZ `Mean_cons_kwh` = mean consumption *per meter*
(`download_energy_postcode.py:245`), meter-weighted to OA. So the "per household"
label is false, and worse: it sums `gas mean per gas-meter` + `elec mean per
elec-meter` — **two different denominators**. Because the meter→household ratio
is itself morphology-correlated (communal gas in flats → few domestic gas meters
serving many dwellings; off-gas detached → no gas meter), the error is structured
*along the treatment axis*, the worst case for a descriptive comparison. This is
45% of the headline gap.
**Fix:** reconstruct a genuine per-household value — `OA_total_gas_kWh /
households + OA_total_elec_kWh / households` over a *common* Census household
denominator (TS017). Requires carrying postcode *totals* (or mean×num_meters),
not means, through `aggregate_energy_oa.py`. This is a pipeline change and it
moves the largest surface.

### A2. The composite double-counts the transport channel (Access ⊂ Mobility) ⟂ (CRITICAL)
Access is, by construction, a *predicted slice of the Mobility variable*:
`compute_access_penalty` regresses `transport_kwh_per_hh_total_est` (which **is**
the Mobility surface) on coverage, then takes `pred(actual) − pred(85%)`. The
composite then computes `Form + Mobility + Access` (`nepi.py:226`). Since Mobility
already contains all transport energy, adding Access adds the low-coverage
transport increment **a second time**. The 14% Access share of the gap is, in
part, transport energy counted twice.
**Fix (choose one):** (a) present Access as a shaded *attribution within* the
Mobility bar, never summed on top (cleanest); or (b) redefine Mobility as the
reference-coverage prediction so `Mobility + Access` = total transport with no
overlap. Verify and reconcile at `nepi.py:226`.

### A3. The 85% Access reference is circular, and the penalty is one-sided
The reference coverage = the *flat-dominant median* (`nepi.py:152`), which
**guarantees flats ≈ zero penalty and detached the maximum** — the gap is baked
in by choosing the favoured group as the zero. The `clip(lower=0)` (`nepi.py:166`)
then makes Access a one-sided, non-negative term that can only widen the composite
gap, never narrow it (high-coverage OAs get 0, not a credit).
**Fix:** report Access under references {national median, 75th, 85th, 95th pct};
justify the reference behaviourally or treat it as a transparent sensitivity
parameter; reconsider the clip (a symmetric penalty would let good access *reward*
an OA).

---

## PART B — Identification & causal inference

### B1. The Access "penalty" is a non-identified OLS counterfactual in causal language ⟂
"the additional transport energy **attributable to** poor coverage… would fall by
1,543 kWh" (§4.3) is a *do*-operation on a single regressor, but the coefficient
is an ordinary partial correlation from the project's **least** rigorous
regression — the penalty model explicitly uses **"City FE: none"** and HC1 SEs
only (`access_penalty_model.py:118,123`), unlike the main regressions. The
"three converging estimates" (262 / 1,543 / ~1,700 kWh) are **not independent**:
the fleet figure reuses the same OLS-predicted excess cars, so it cannot
corroborate the OLS.
**Fix:** re-language to "the cross-sectional transport-energy *gap* associated
with lower coverage"; promote the spec to BUA-FE + clustered SEs + housing-type
controls (the FWL demeaning already exists, `proof_of_concept_oa.py:975`); drop
or relabel the "convergence" paragraph.

### B2. Residential self-selection — acknowledged, then neutralised by a non-sequitur ⟂ (the field's central threat)
Self-selection (car-preferrers sort into detached suburbs; walkers into flats) is
*the* dominant methodological preoccupation of this literature (Mokhtarian & Cao
2008; Cao, Mokhtarian & Handy 2009). The paper cites it (§6.3) then defuses it
with "the planning implication is the same… planners control morphology, not
preferences." **This inverts the logic:** if the gap is sorting, then building a
walkable neighbourhood and moving car-preferring households in does *not* transfer
the low-energy profile — which is exactly the §4.3 counterfactual. The appeal to
Stevens' "5–25% attenuation" is **misapplied**: those figures come from studies
*with individual attitudinal controls*, which NEPI has none of, so the honest
expectation for an uncontrolled cross-section is attenuation at the top of, or
beyond, that range.
- **Quantified benchmark (new, decisive):** Nachtigall et al. (2023, double-ML on
  32k Berlin diaries) find **~half the naïve neighbourhood emissions gap is
  self-selection**, and **73.7% of the *built-environment* effect is destination
  accessibility**. This both (i) tells NEPI its cross-sectional estimates are
  inflated ~2× (present as upper bounds, optionally apply a ~0.5× sensitivity)
  and (ii) corroborates the accessibility-over-density emphasis.
- **The reassurance to use:** Cao-Mokhtarian-Handy's verbatim finding — *"virtually
  all of the 38 empirical studies… found a statistically significant influence of
  the built environment remaining after self-selection was accounted for"* —
  self-selection attenuates, it rarely annihilates.
**Fix:** delete the "planning implication is the same" sentence; state the bias
direction explicitly (**upward; estimates are upper bounds**) — this single
sentence pre-empts ~80% of likely objections; add the tenure/composition proxies
already in the Census (TS054 tenure, TS017 size, children/age structure); cite
the Stevens↔Ewing&Cervero (2017) exchange as a pair; commit to a future
mover-panel design (Understanding Society / NTS) as the field-standard rebuttal.

### B3. Omitted-variable bias and endogenous coverage ⟂
Income enters only as the **IMD income-domain rank at LSOA level** (coarser than
the OA outcome) — residual income confounding remains, and income drives both car
ownership and dwelling type. **Tenure is absent.** `cars_per_hh` is treated as a
mediator (so excluded) on an *assumed* causal chain. **Coverage is endogenous:**
services locate where demand/density already is, so the causal arrow runs both
ways — the OLS treats a co-determined regressor as pre-determined.
**Fix:** replace IMD-rank with continuous small-area income (ONS MSOA net income);
add tenure + household-composition shares; report the cars-as-confounder *bracket*
(gap with and without `cars_per_hh`); lag or instrument coverage and state its
endogeneity.

### B4. The lock-in / generational claim is asserted, not evidenced ⟂
The headline elevation — Access is "locked in for generations… the surface no
policy addresses" — rests on a single durability statistic ("38% of housing
predates 1946") that speaks to *buildings*, not to the *behavioural energy gap*.
It is **internally contradicted**: §6.2 itself says land-use policy can cut the
access penalty without changing street geometry — i.e. it is *not* locked in. The
§6.2 offset timescales (10–20/10–15/50–100 yr) have no citation or model. And the
§5.5 Census-2011 comparison shows the gradient *changed* over a decade (2.00×→
1.70×), which undercuts strict lock-in.
**Fix:** adopt Seto et al. (2016, *Carbon Lock-In*, ARER) — separate *infrastructural*
(street geometry, durable, citable) from *behavioural* (the energy gap, malleability
unknown). The defensible narrowed claim: "Access responds to land-use/service policy
but not to the *building- and vehicle-technology* levers current decarbonisation
policy actually pulls" — still novel, and true. Cite or delete the timescale numbers.

### B5. No sensitivity-to-unobservables — and the inputs already exist ⟂
§5 ("Robustness") tests *analyst choices* but contains **zero** tests of robustness
to unobserved confounding — the threat that governs the causal claims. The headline
bootstrap CIs ([0.682,0.687]) are *precision theatre* (iid resampling under heavy
spatial autocorrelation).
**Fix (high return, low cost):** the progressive control sets M1–M5
(`run_regressions`) already supply the inputs for **Oster (2019) bounds** — report
δ (selection-proportionality at which the effect is zero) and the bias-adjusted β.
Add **E-values** (VanderWeele & Ding 2017) and **Cinelli–Hazlett (2020)**
benchmarking against the income covariate. Demote the bootstrap CIs.

---

## PART C — Ecological inference, MAUP, and spatial dependence

### C1. The ecological defense is sound for Form, over-extended to behaviour ⟂
Greenland (2001) / Wakefield (2008) exempt area-level designs from ecological bias
when the *outcome* is intrinsically contextual — true for **Form** (metered OA
energy is a real aggregate), **false** for **Mobility/Access** (individual
behaviours aggregated up). The §4.3 household counterfactual is exactly the
cross-level (Robinson 1950) inference the exemption does not cover; aggregation can
*amplify* the sorting confounder.
**Fix:** bifurcate the defense in the text (Form = legitimately ecological;
Mobility/Access = ecological associations subject to Robinson confounding); bound
the inflation with a partial individual-level check (Census microdata / NTS
household records — compare individual vs ecological β); for the car-ownership
claim, King's (1997) method-of-bounds is cheap.

### C2. MAUP is named but never tested ⟂ (HIGH)
The plurality-threshold test (§5.2) holds units fixed and varies only the
*classification* — it is not a MAUP test. Fotheringham & Wong (1991) define MAUP's
*scale* and *zonation* effects; **neither is tested**. Every headline number is
computed at one zonation (2021 OAs) at one scale. The A–G bands are a *percentile*
operation, so the same neighbourhood can change band purely from re-aggregation.
**Fix (nearly free — the LSOA pipeline and `lsoa_energy_consumption.parquet`
already exist):** recompute the gradient at OA/LSOA/MSOA (scale ladder) and on a
regular grid / random equal-population partitions (zonation); cross-tabulate each
OA's band at OA vs LSOA and report the fraction shifting ≥1 band. Relabel §5.2
"classification-threshold sensitivity."

### C3. Spatial autocorrelation is diagnosed but never corrected — and the flagship model is the worst case ⟂ (HIGH)
Moran's I is computed on **raw variables, not residuals**, on a 50k subsample —
so it cannot tell you whether the *regression* inference is valid. The Access
penalty (the flagship) is fit with **HC1 only, no clustering, no FE, no spatial
correction** (`access_penalty_model.py:118`), on data §5.1 admits is
autocorrelated. With ~197k OAs in ~6,700 BUAs the design effect can shrink
effective N by 1–2 orders of magnitude, so the "t = −161" magnitudes the Results
lean on are unsupported. BUA clustering only captures *between*-BUA dependence;
the dominant signal is *within*-BUA short-range (adjacent OAs).
**Fix:** Moran's I + LM-lag/LM-error tests on **residuals**; re-estimate the Access
penalty with **Conley spatial-HAC** SEs (and BUA clustering and FE — the "memory
footprint" excuse is solved by the FWL demeaning already in the code); fit a
spatial error/lag model (`spreg.GM_Error/GM_Lag`, GMM-scalable) or eigenvector
spatial filtering nationally; report whether the coverage coefficient survives.

### C4. "Steepening at stricter thresholds" may be a tail-selection artifact, not attenuation
At 60% purity, **>half the sample is discarded** and the retained OAs are
spatially purest (mono-tenure social-flat estates; affluent detached enclaves) —
so steepening is confounded with *changing sample composition*. Mechanically,
selecting the purest tails pushes the two medians apart *by construction*; the
Bound et al. (2001) attenuation story predicts the same sign, so the test **cannot
discriminate** benign attenuation from artifact.
**Fix:** replace the categorical test with a **continuous-share dose–response**
on the *full, fixed* sample (immune to discard-selection); post-stratify the
strict subsamples to match the plurality sample on income/tenure/size.

### C5. One national gradient hides near-certain spatial heterogeneity
Land price, off-gas share, transit supply and the income–morphology correlation
vary enormously by region; a single pooled β can average heterogeneous local
relationships, and the national A–G banding then mis-rates low-gradient regions.
**Fix:** random-slope multilevel model (OA in BUA/region) — the random-slope
variance *is* the heterogeneity measure and it handles part of the spatial
dependence; GWR/MGWR on a tractable subset to map where the gradient is steep;
promote the existing `run_per_city` output into a heterogeneity figure;
stratify by off-gas share.

### C6. The ML planning models use random, not spatial, CV
`nepi_model.py:265` uses a row-wise `train_test_split` → adjacent (correlated) OAs
leak across train/test, so the public tool's reported R² is **optimistic**.
**Fix:** `GroupKFold` grouped on BUA (the `city` field exists); report the
random-CV vs spatial-CV gap.

---

## PART D — Missing confounds & validation (what a top-journal reviewer will demand)

### D1. No climate / heating-degree-day control on Form ⟂ (CRITICAL)
Detached-dominant OAs are disproportionately rural/northern/upland (colder,
windier); flat-dominant concentrate in the warmer south-east. Gas is
weather-corrected to a national model, *not* normalised across space; there is no
HDD, region FE, or altitude term. A material share of the 1.46× Form gradient may
be **climate, not morphology**, and BUA-FE does not help because the flat-vs-
detached contrast is largely *between* settlements.
**Fix:** merge gridded HDD/CDD (HadUK-Grid 1 km, or ERA5) at OA centroid + altitude
(LiDAR DTM exists); re-estimate Form with HDD control and within-region FE; report
the ratio before/after. The single most obvious reviewer kill-shot.

### D2. No floor-area / tenure / occupancy control ⟂ (CRITICAL)
Floor area is the **#1 omitted variable for a metered-energy DV** — detached homes
consume more partly because they are *larger*, not only lower-S/V. EPC
`TOTAL_FLOOR_AREA` is already ingested; tenure (TS054) and occupancy (TS050/TS052)
are in-pipeline. **Huebner et al. (2015):** building factors (incl. floor area,
form) explain ~39% of domestic energy and *dominate* socio-demographics — which
both licenses a form-led surface *and* shows floor area must be in the model.
**Fix:** add floor area, tenure, occupancy stepwise; report both the per-household
gradient (current) and the floor-area-controlled / per-m² variant — the per-m²
benchmark in Liddiard et al. (2021) London metered houses (gas EUI 162/164/157/138
by form) is an external validation target.

### D3. The accessibility decay/bands are assumed, not calibrated ⟂ (HIGH)
The Gaussian decay and 80 m/min bands underpin the entire Access surface yet are
un-calibrated (the paper concedes this, §6.4) with **no sensitivity analysis**.
cityseer's own `β = 4/d_max` (Simons 2023) is a convention, not a calibration.
**Fix:** (i) **calibrate β with the half-life method (Östh, Lyhagen & Reggiani
2016)** so the modelled median walk distance matches the NTS observed median —
turns an assumption into a one-paragraph citable choice; (ii) sensitivity table
over decay form (exponential/Gaussian/log-normal — Chen 2015) and ±50% thresholds,
reporting band-assignment stability; (iii) cite Sharmin & Kamruzzaman (2018)
meta-analysis (configurational measures predict pedestrian movement; effect ~0.21
integration, ~0.48 choice) for the "accessibility predicts behaviour" claim; cite
a walk-speed norm (HCM/TRB) for 4.8 km/h.

### D4. No out-of-sample / independent validation of the composite ⟂ (CRITICAL)
The composite NEPI is never validated against any independent outcome; both DV and
"validation" come from the same DESNZ feed. **The repo already holds the validation
data and never uses it:** `lsoa_energy_consumption.parquet` (independent
aggregation), `epc_domestic_spatial.parquet` (SAP-modelled), `lsoa_vehicles.parquet`
(DVLA), and NTS.
**Fix:** (i) correlate the OA Form surface with the LSOA-DESNZ and EPC series,
reporting discrepancy by type (this also *measures* the off-gas/communal
compression the paper only asserts); (ii) validate Mobility/Access against DVLA
cars-per-LSOA and NTS area-type travel *out of sample*; (iii) spatial CV holding
out whole regions (train North/test South), checking band stability; (iv) the
Walk Score validation design (Duncan et al. 2011 — Spearman vs objective measures)
is the exact template.

### D5. Other robustness gaps reviewers will list
- **Placebo / falsification:** permute coverage within BUA → penalty should
  collapse; regress an implausible outcome on coverage. None exist.
- **BUA-block bootstrap:** the current iid bootstrap gives indefensibly narrow CIs;
  block on BUA (5–20× wider, honest).
- **Decomposition CI:** "45/43/14%" has **no uncertainty quantification** — block-
  bootstrap the whole pipeline, re-fitting the penalty model inside each replicate.
- **Penalty model fit:** R² = 0.294 (70% of transport-energy variance unexplained),
  yet the signature 1,543 kWh is read off *this* model and is an **off-support
  extrapolation** (≈no detached OA has 85% coverage; coverage⊥morphology near-
  collinear). Report prediction intervals, VIF for the penalty model (not just the
  main ones), a nonlinear coverage term, and a common-support restriction.
- **Specification-curve analysis** across the garden of forking paths (9 decay
  thresholds, 85% reference, zero-clip, 6.04× NTS scalar, band cut-points,
  build-year bins, road intensity). Given the qualitative pattern is stable in the
  partial sensitivities, a spec-curve would be persuasive.
- **Stop reporting significance stars** (N≈197k → everything significant); report
  effect sizes + CIs.
- **Pin data versions** (DESNZ edition, Census, ECUK) and **add the pending pytest
  tests** for the load-bearing geospatial joins.

---

## PART E — Banding & index methodology

### E1. Percentile (norm-referenced) bands ≠ EPC (criterion-referenced) — a category error ⟂ (HIGH)
Bands are assigned by **national percentile** (`nepi.py:37`), so exactly 8% are
always Band A and 5% always Band G — **the band distribution cannot improve even
if every neighbourhood reaches net-zero.** A building EPC uses *absolute* SAP
thresholds, so the stock *can* shift toward A. Calling percentile bands
"EPC-analogous" is a category error, and the index cannot track progress over
time (an OA that improves in absolute kWh can *worsen* in band if others improve
more). Equal-percentile widths also map to unequal kWh widths, so a one-band step
means different things in different parts of the range.
**Fix:** adopt **absolute kWh thresholds** (criterion-referenced), anchored to a
meaningful level (e.g. a net-zero-compatible household budget) so NEPI behaves like
the EPC it emulates and can track progress; keep percentile rank as a secondary
"national position" indicator. If percentile banding is retained, rename it,
drop the EPC-equivalence framing, and freeze the cut-points from a base year.
Report band-stability under the A1 per-household fix and the decay/scalar/reference
sensitivities.

### E2. "Common kWh = no weighting" is false — equal-kWh is a value choice ⟂
Summing site-energy kWh weights gas = petrol = grid-electricity equally, which is a
strong (hidden) choice: under *carbon* they differ ~3× (gas 0.18, petrol 0.24,
2024 grid ~0.07 kgCO₂/kWh) and under *cost* electricity ≈ 4× gas. Equal-kWh makes
the gas-heavy (most *decarbonisable*) Form surface dominate; a carbon-weighted
composite would up-weight Mobility/Access.
**Fix:** reframe honestly ("we weight by site-energy kWh; carbon- and cost-weighted
composites are alternatives") and provide carbon- and cost-weighted composites as
sensitivity columns; show whether the A–G *ranking* is robust to the weighting
metric (if it is — likely, given correlated gradients — that is a far stronger
claim than "no weighting"). Follow OECD/JRC (2008) composite-index practice;
indices are most sensitive to *weighting*, so it must be sensitivity-tested.

### E3. Mixing surfaces of different epistemic status without uncertainty propagation
Form (metered, but mis-denominated), Mobility (17% measured + 83% national
constant), and Access (regression-imputed, reference-dependent) are summed into one
number with **no uncertainty band**. Propagate: envelope over {Mobility scalar
1×–10×} × {Access reference 72–95%} × {Form off-gas/communal correction on/off};
flag OAs whose band flips. Label each surface by tier (measured/modelled/imputed).

### E4. Mobility construction caveats (mostly framing)
83% of Mobility is a uniform **6.04× commute→total scalar** — yet the paper's own
thesis is that *non-commute* trips scale with morphology, so a uniform multiplier
is internally inconsistent. **Fix:** source a *morphology-stratified* total/commute
ratio from NTS area-type bands. Band-midpoint distance understates ~2× (the OD
check, §5.6, already shows this) and the 80 km top-band cap is morphology-correlated;
report the OD-distance Mobility alongside. Verify car *passengers* aren't double-
charged at road intensity. Consider making the pre-pandemic (2011-mode-share)
Mobility canonical, since NEPI rates *structural* performance.

---

## PART F — Positioning & novelty (literature-grounded; mostly tailwind)

- **Accessibility-over-density is your strongest card** — foreground it. Benchmark
  NEPI's own effect sizes against the meta-analytic elasticities: Ewing & Cervero
  (2010) destination accessibility −0.20 / distance-to-downtown −0.22 vs density
  −0.04; Stevens (2017) self-selection-corrected still has accessibility dominant
  (−0.20/−0.63); NRC SR 298 (2009): doubling density ≈ −5% VMT, destination
  accessibility ≈ −20%. Showing your cross-sectional penalty is *not larger* than
  these causal-leaning benchmarks is a strong honesty signal.
- **Cite the modern causal-ML corroboration prominently:** Wagner et al. (2023,
  six cities, causal discovery + Shapley — distance-to-centre dominant) and
  Nachtigall et al. (2023, Berlin double-ML — 73.7% of BE effect is accessibility,
  ~half the gap is self-selection). "Our descriptive pattern matches what causal
  designs recover" is the most persuasive defense of an ecological index.
- **Form-surface framing:** present the 1.46× as a *realised* gradient consistent
  with the best UK metered studies (Liddiard et al. 2021; Steemers 2008; Huebner
  et al. 2015), explicitly contrasted with Rode et al.'s (2014) *modelled* six-fold
  ceiling — turning "why isn't your effect six-fold?" into a validation. Close the
  metered-over-EPC argument with Few et al. (2023, EPCs over-predict, bias *grows*
  with worsening rating: band D −20%, F/G −48%), Summerfield et al. (2019), Firth
  et al. (2024, the gap is *spatially form-correlated* — the decisive reason not to
  use EPC energy as the DV), Crawley et al. (2019), Sunikka-Blank & Galvin (2012,
  prebound). Name prebound/affordability as the intrinsic cost of metered data.
- **Novelty positioning (honest):** no consumer-facing place-based A–G rating that
  combines building operational energy *and* transport energy, validated against
  metered consumption, appears to exist. The closest structural precedent is CNT's
  **Housing + Transportation (H+T) Index** — place-based, building+transport, but in
  *dollars*, not energy, and not an A–G label. District building-energy
  classification (Mutani; NEBULA 2025) and the closest UK DESNZ-based peer (Feng et
  al. 2025, MGWR) are building-energy only. Defensible claim: *"first to combine
  building operational and transport energy into a single place-based A–G rating,
  framed as a neighbourhood EPC, validated against metered consumption."*
- **Off-gas (16–24%) and communal-heating (14–45% of flats) biases** both *compress*
  the flat-vs-detached gradient — i.e. the true Form gradient is *larger* than
  reported (the conservative direction). Quantify with the DESNZ/Nesta/Commons-
  Library figures; position NEPI against NEBULA (2025) as the contemporary
  energy-only benchmark (which validates the postcode-matching: 98.3% gas match).

---

## PART G — Prioritised action list

| # | Action | Severity | Effort | Why |
|---|--------|----------|--------|-----|
| 1 | **Reconstruct Form as genuine per-household** (totals ÷ Census households) | CRITICAL | Med | Largest surface; current unit is per-meter |
| 2 | **Reconcile the Access/Mobility double-count** (nest Access in Mobility) | CRITICAL | Low | Removes a double-count in the headline gap |
| 3 | **Climate/HDD + floor-area + tenure controls on Form** | CRITICAL | Med | The two most obvious reviewer kill-shots; data exists |
| 4 | **Out-of-sample validation** vs LSOA-energy / EPC / DVLA / NTS | CRITICAL | Med | Currently zero ground-truth; data already in repo |
| 5 | **Self-selection honesty pass** — state estimates are upper bounds; add tenure/composition; cite Nachtigall/Stevens | Serious | Low | Pre-empts ~80% of objections; one paragraph |
| 6 | **Oster bounds / E-value / Cinelli–Hazlett** from existing M1–M5 progression | Serious | Low | Inputs already computed; replaces precision-theatre CIs |
| 7 | **Fix the flagship Access model SEs** — Conley spatial-HAC + FE + clustering; residual Moran's I | Serious | Low–Med | The "t=−161" magnitudes are currently unsupported |
| 8 | **MAUP scale + zonation analysis** (LSOA pipeline exists) | Serious | Low | Removes the §2.5 over-claim |
| 9 | **Absolute (criterion-referenced) bands** or rename+freeze percentile | Serious | Med | The EPC analogy is currently a category error |
| 10 | **Calibrate the accessibility decay** (half-life method) + decay/threshold sensitivity | Serious | Med | Turns "assumptions" into a demonstrated robustness result |
| 11 | **Continuous-share dose–response** (fixed sample) replacing plurality test | Moderate | Low | Immune to discard-selection |
| 12 | **Carbon/cost-weighted composite** sensitivity; drop "no weighting" claim | Moderate | Low | "Ranking robust to weighting" is stronger than "no weighting" |
| 13 | **Random-slope multilevel / GWR** + spatial block CV for the ML tool | Moderate | Med | Heterogeneity + remove leakage |
| 14 | **Specification-curve** across the forking-paths grid; stop significance stars | Moderate | Med | Persuasive given the stable partial sensitivities |

**Bottom line.** The compact-→-lower-energy + higher-access *pattern* is almost
certainly real and survives the sensitivities already run, and the
accessibility-over-density thesis is directly corroborated by the newest causal
evidence. The work becomes publication-robust by (i) fixing the two construction
errors (A1, A2), (ii) controlling the two obvious Form confounds (climate, floor
area), (iii) validating out-of-sample against data already in the repo, and
(iv) aligning the language to the design — descriptive, estimates as upper bounds,
EPC analogy stated precisely. Most of this needs no new data collection.

---

## PART H — Framing & scope: scaling, lock-in, energy justice

### H1. Urban scaling — keep as mechanism, cite the critiques, disavow OA-level exponents
The Bettencourt (2007) N^1.15 / N^0.85 framing is defensible only as *motivation*
(the proximity mechanism), not as a law NEPI relies on. The exponents are heavily
contested: Leitão et al. (2016) show standard log-log OLS yields β≠1 conclusions
that "vary dramatically" once heavy-tailed city sizes and non-Gaussian fluctuations
are handled; Arcaute et al. (2015, England & Wales) and Cottineau et al. (2017,
~5,000 city definitions) show the exponent is **city-definition / MAUP-sensitive**
and can flip super-↔sub-linear. **Fix:** cite the critiques alongside Bettencourt;
frame scaling as direction-not-magnitude; explicitly disavow any OA-level
energy-vs-population exponent (OAs are intra-urban units, not cities). Same
areal-unit sensitivity bounds both scaling exponents and NEPI's bands (→ C2).

### H2. Lock-in — supported, but must be two-tier
The durable-substrate half is strongly supported and arguably *understated*: Boeing
(2021) finds street patterns persist "for centuries"; Seto et al. (2016) — the
canonical infrastructural/institutional/behavioural taxonomy — state the built
environment "determine[s] energy demand for decades" and "lock[s] in… mode
choices… and behaviors." But Seto et al. say *constrains*, not *determines*:
current travel energy also responds to faster-turnover fuel/vehicle/income/transit
factors. The **durability hierarchy** (layout: centuries ≫ buildings: decades–
century ≫ vehicles: ~years) is the sound empirical core of the headline. **Fix:**
adopt Seto's taxonomy; cite Boeing + Barrington-Leigh & Millard-Ball (2019);
separate durable substrate from malleable outcome; temper "no policy addresses" →
"**under-addressed** by current decarbonisation policy."

### H3. Energy justice — add a section; an A–G area label carries real distributional risk
In energy-justice terms (Jenkins et al. 2016: distribution / recognition /
procedure; Walker & Day 2012; Bouzarovski & Simcock 2017) a place-based A–G label
is both a contribution (it makes spatial energy burdens visible) and a hazard.
Because NEPI's worst grades fall on the most deprived areas, the label risks
**territorial stigmatization** (Wacquant 2007) and **regressive capitalization**
into property values/rents (EPC evidence: Fuerst et al. 2015; Brounen & Kok 2011).
**Fix:** add an energy-justice section on the three tenets; add an anti-stigma
safeguard (grades describe *form*, not residents; G-areas are priority-for-support,
not failing); flag the regressive risk and pair the label with an
investment/entitlement dimension.

### H4. Ecological inference & MAUP — corroborated (→ C1/C2)
Robinson (1950; ecological r can exceed or sign-flip the individual r), Openshaw
(1984, MAUP), Greenland & Morgenstern (1989, ecological bias from effect
modification alone), and Subramanian et al. (2009, distinguishing contextual from
compositional needs individual + ecological data) confirm C1/C2: keep all claims
ecological, temper individual-level language ("band-G *areas*"), and add the LSOA
MAUP re-run.

➡ **The step-by-step execution plan for all of the above is
[robustness_plan.md](robustness_plan.md).**

---

## Citations (for `paper/references.bib`)

Ewing & Cervero (2010) *JAPA* 76(3):265–294 · Stevens (2017) *JAPA* 83(1):7–18 ·
Ewing & Cervero (2017) *JAPA* 83(1):19–25 · NRC (2009) *Driving and the Built
Environment* SR 298 · Mokhtarian & Cao (2008) *Transp. Res. B* 42(3):204–228 ·
Cao, Mokhtarian & Handy (2009) *Transport Reviews* 29(3):359–395 · Wagner et al.
(2023) arXiv:2308.16599 · Nachtigall et al. (2023) arXiv:2312.06616 · Cervero &
Duncan (2006) *JAPA* 72(4):475–490 · Brownstone & Golob (2009) *J. Urban Econ.*
65(1):91–98 · Mindali, Raveh & Salomon (2004) *Transp. Res. A* 38(2):143–162 ·
Rode et al. (2014) *Env. Plan. B* 41(1):138–162 · Steemers (2003) *Energy &
Buildings* 35(1):3–14 · Steemers et al. (2008) *Energy Policy* · Liddiard et al.
(2021) *Buildings & Cities* 2(1):336–353 · Huebner et al. (2015) *Applied Energy*
159:589–600 · Few et al. (2023) *Energy & Buildings* 288:113024 · Summerfield et
al. (2019) *Energy Policy* 129:997–1007 · Firth, Allinson & Watson (2024) *J.
Build. Perf. Sim.* · Crawley et al. (2019) *Energies* 12(18):3523 · Sunikka-Blank
& Galvin (2012) *Build. Res. & Info.* 40(3):260–273 · Feng, Miao & Turner (2025)
*Energy, Sustainability & Society* 15(1):24 · NEBULA (2025) arXiv:2501.09407 ·
Hillier et al. (1993) *Env. Plan. B* 20(1):29–66 · Sharmin & Kamruzzaman (2018)
*Transport Reviews* 38(4):524–550 · Simons (2023) *Env. Plan. B* 50(5):1268–1289 ·
Östh, Lyhagen & Reggiani (2016) *EJTIR* 16(2):344–363 · Chen (2015) *Chaos,
Solitons & Fractals* 77:174–189 · Moreno et al. (2021) *Smart Cities* 4(1):93–111 ·
Duncan et al. (2011) *IJERPH* 8(11):4160–4179 · Hall & Ram (2018) *Transp. Res. D*
61:310–324 · OECD/JRC (2008) *Handbook on Constructing Composite Indicators* ·
Greco et al. (2019) *Soc. Indic. Res.* 141:61–94 · Oster (2019) *J. Bus. Econ.
Stat.* · VanderWeele & Ding (2017) *Ann. Intern. Med.* · Cinelli & Hazlett (2020)
*JRSS-B* · Robinson (1950) *Am. Soc. Rev.* · King (1997) *A Solution to the
Ecological Inference Problem* · Openshaw (1984) *MAUP* CATMOG 38 · Fotheringham &
Wong (1991) *Env. Plan. A* · Anselin (1988) *Spatial Econometrics* · Conley (1999)
*J. Econometrics* · Seto et al. (2016) *Ann. Rev. Env. Res.* (Carbon Lock-In) ·
Bound, Brown & Mathiowetz (2001) *Handbook of Econometrics* · Norman et al. (2006)
*J. Urban Plan. Dev.* 132(1):10 · Bettencourt et al. (2007) *PNAS* 104(17):7301 ·
Leitão et al. (2016) *R. Soc. Open Sci.* 3(7):150649 · Arcaute et al. (2015) *J. R.
Soc. Interface* 12(102):20140745 · Cottineau et al. (2017) *CEUS* 63:80 · Unruh
(2000) *Energy Policy* 28(12):817 · Boeing (2021) *JAPA* 87(1):123 · Barrington-
Leigh & Millard-Ball (2019) *PNAS* 116(6):1941 · Walker & Day (2012) *Energy Policy*
49:69 · Jenkins et al. (2016) *ERSS* 11:174 · Sovacool & Dworkin (2015) *Applied
Energy* 142:435 · Bouzarovski & Simcock (2017) *Energy Policy* 107:640 · Wacquant
(2007) *Thesis Eleven* 91(1):66 · Fuerst et al. (2015) *Energy Economics* 48:145 ·
Brounen & Kok (2011) *JEEM* 62(2):166 · Greenland & Morgenstern (1989) *IJE*
18(1):269 · Subramanian et al. (2009) *IJE* 38(2):342.
