# The Neighbourhood Energy Performance Index: Energy Spent versus Access Gained

> **STATUS: SCAFFOLD (frame only — not prose).** Bullet frame for the two-axis paper,
> to be developed section by section. Canonical numbers live in
> [`paper/summary.md`](paper/summary.md); the literature synthesis in
> [`paper/literature_review.md`](paper/literature_review.md) (esp. §2, §6); the
> methodological decisions and caveats in [`paper/method_notes.md`](paper/method_notes.md).
> The previous three-surface / A–G-scorecard draft is archived at
> [`paper/archive/PAPER_three_surface_deferred.md`](paper/archive/PAPER_three_surface_deferred.md)
> (some Background / data-source prose is reusable). IMRaD; formal academic style.

---

## Abstract

- Premise: judge a neighbourhood not by the energy it consumes but by the everyday access that energy buys (Jacobs / trophic framing — function per unit energy).
- Data: ~178,000 Census 2021 Output Areas, England. Metered household energy (DESNZ) + NTS-anchored car-travel energy; network access (cityseer over OS Open Roads).
- Method: compositional ecological regression; energy modelled **per dwelling** with family size and floor area as free controls; **metered, not SAP-modelled**.
- Headline: a detached neighbourhood spends ~2.1× a flat's energy per dwelling, yet a flat reaches ~24× the amenities on foot and returns ~6.3× the access per kWh; under best fabric + full electrification the energy gap closes only part-way while the access deficit is unchanged.
- Contribution: two *measured* axes plus an explicit rate; the access-per-energy rate at national OA scale is novel.

## 1. Introduction

- The problem: energy policy and neighbourhood assessment judge places by energy *consumed*, ignoring what the energy is *for* (the access/function it delivers).
- The reframing (Jacobs; ecological/trophic analogy): a compact, complex settlement extracts more function from each unit of energy, as a rainforest does; a dispersed one lets energy run through (the desert). State the rate as the object of interest.
- Why now: net-zero retrofit + electrification debates assume technology closes the gap; the lock-in question (what survives optimisation) is unresolved.
- Gaps in prior work (forward-reference §2): energy *or* access measured, rarely together; inconsistent functional units; modelled (SAP/EPC) rather than metered energy; coarse spatial units.
- Contribution and roadmap: two measured axes (energy spent, access gained) + a rate; metered energy; network access; OA scale; compositional method; robustness; a *place-level* estimand.

## 2. Background

> Reuse/adapt from [`literature_review.md`](paper/literature_review.md) §2 and §6.

### 2.1 Building energy and urban form
- Surface-to-volume / shared-wall mechanism; compact forms lose less heat (Rode et al. 2014, simulated; the shared-wall ratio).
- Empirical English regularity: detached ≈ 2× flat gas (NEED / EHS; Wyatt 2013; Buyuklieva et al. 2023).

### 2.2 Transport energy, density and accessibility
- Density–travel link (Newman & Kenworthy 1989); the "modest" marginal effect (Echenique et al. 2012); reframing as regional **accessibility** not density (Ewing et al. 2018); local access reduces driving (Elldér et al. 2022).

### 2.3 The EPC performance gap — why metered, not modelled
- SAP/EPC over-predicts, worst for large detached (Few et al. 2023; Summerfield et al. 2019; Firth et al. 2024; Crawley et al. 2019). Motivates the metered DV.

### 2.4 The functional-unit problem
- Per-capita vs per-m² changes the conclusion (Norman et al. 2006); household-size elasticity of energy < 1 (Huebner & Shipworth 2017; Druckman & Jackson 2008) ⇒ per-person normalisation is an artefact; argues for per-dwelling with size as a free control.

### 2.5 Ecological analysis, MAUP and self-selection
- Ecological inference and the modifiable areal unit problem at OA scale; residential self-selection of car-oriented households (Cao, Mokhtarian & Handy 2009) as the threat to causal reading.

## 3. Data and Methods

### 3.1 Unit of analysis

The unit of analysis is the 2021 Census Output Area (OA), the smallest geography at which the Census of England and Wales is released, comprising approximately 125 households and 300 residents. The analysis covers the Output Areas of England (approximately 178,000). Wales is excluded: although the energy, Census, Energy Performance Certificate and Ordnance Survey inputs all extend to it, the deprivation measure used as a control — the English Index of Multiple Deprivation — has no directly comparable Welsh equivalent, and harmonising the two is left to future work. The Output Area is preferred to the Lower-layer Super Output Area (LSOA; approximately 1,500 residents) used by much of the small-area energy literature because the finer grain reduces within-unit heterogeneity in dwelling type, which is the exposure of interest; the consequences of ecological aggregation at this scale are addressed in §5.5.

### 3.2 Data sources

All inputs are open data, processed to the Output Area on the British National Grid (EPSG:27700). Table 1 lists each source, the artefact it yields, and its role.

| Domain | Source | Role in the analysis |
| --- | --- | --- |
| Population and dwellings | Census 2021 (TS001, TS017, TS044) | Residents, households, dwelling-type shares (the exposure) |
| Tenure and socio-economic class | Census 2021 (TS054, TS062) | Tenure shares (confound); NS-SeC (self-selection robustness) |
| Car ownership and commuting | Census 2021 (TS045, TS058) | Local signals for travel-energy disaggregation |
| Workplace jobs | Census 2021 (WP101EW) | Jobs reachable (access) |
| Domestic energy | DESNZ sub-national gas and electricity | Metered household energy (the energy axis) |
| Building fabric | EPC (domestic) | Floor area, build year, current and potential intensity |
| Road network | OS Open Roads | Network routing for the access axis |
| Amenities | FSA, NaPTAN, GIAS, NHS ODS, OS Open Greenspace | Reachable everyday destinations |
| Deprivation | Indices of Deprivation 2025 | Overall IMD and income domain (confounds) |
| Vehicle fleet | DVLA licensing | Electric-vehicle share (travel-energy intensity) |
| Travel behaviour | National Travel Survey (NTS9904) | Car-miles per person by rural-urban class (travel anchor) |
| Climate | HadUK-Grid (1991–2020) | Heating-degree-days (confound) |

### 3.3 The energy axis

Household energy is measured per dwelling per year as *delivered* energy — the energy arriving at the home and the vehicle, in kilowatt-hours, with no conversion to primary energy or carbon — in two components.

The home component is metered, not modelled. The Department for Energy Security and Net Zero publishes domestic gas and electricity consumption at postcode level, each as a meter count and a mean per meter. Each fuel is aggregated to the Output Area by a meter-weighted mean, the two are summed as delivered energy, and the total is divided by the count of Census households. Gas consumption is weather-corrected; both series cover domestic meters only. The use of metered rather than SAP-modelled energy is deliberate (§2.3): Energy Performance Certificate ratings systematically over-predict consumption, and most for the largest, least efficient dwellings, so an EPC-based dependent variable would inflate the dwelling-type gradient with model error rather than measure it (Few et al., 2023; Summerfield et al., 2019; Firth et al., 2024).

The travel component is the car-travel energy associated with a neighbourhood's location, not merely its commute, which accounts for only about one-sixth of car mileage. Total local car travel is not recorded in open data and is therefore estimated by constrained disaggregation of a measured total. The National Travel Survey (NTS9904) reports car-driver miles per person by 2021 rural-urban classification of residence, rising from approximately 2,500 miles per person in dense urban areas to approximately 5,200 in rural ones. Each class total is distributed across its constituent Output Areas in proportion to two local signals — car ownership and commute distance — subject to the constraint that each class's population-weighted mean reproduces the survey figure, so that the measured class marginals are preserved exactly and only the within-class distribution is estimated. Estimated miles are converted to energy at the local fleet's energy per mile, blending an internal-combustion figure of approximately 0.93 kWh per mile and an electric figure of approximately 0.32 by the area's electric-vehicle share (DVLA). Sensitivity to the single free parameter, the elasticity of the estimate to commute distance, is reported in §4.5.

### 3.4 The access axis
- cityseer network access over OS Open Roads (built once); amenities / jobs / people reachable at a short walk (1,600 m), each area's own car catchment, and a long drive (25,600 m).

### 3.5 The rate
- Access per kWh: amenities reachable within the area's own car catchment ÷ its car-travel energy; catchment = NTS distance ÷ ~370 trips, capped [1.6, 25.6] km. (Spell out exactly as in summary.md.)

### 3.6 The compositional model
- No-intercept, household-weighted ecological regression on dwelling-type share fractions (each coefficient a pure-type mean); ratio = exp(b_detached − b_flat).
- Energy axes: log-OLS, **per dwelling**, family size (log) and floor area (log) as **free** controls (the functional-unit resolution); confounds held: build year, overall IMD + income, tenure, climate (HDD).
- Access axis: Poisson log-link (counts, zero-inflated); income-controlled, NOT density-controlled (density is the mechanism); NOT overall-IMD (its barriers domains are access measures).

### 3.7 The lock-in scenario
- Optimised energy = best-practice fabric (metered gas × EPC potential/current ratio) + full electrification (EV fleet intensity, miles unchanged); access unchanged by construction.

### 3.8 Robustness and the estimand
- Oster (2019) coefficient-stability bound on a continuous detached-share gradient; NS-SeC control; access location-intrinsic ⇒ immune. State the place-level estimand explicitly.

## 4. Results

### 4.1 The energy axis
- Heat 1.60× → family-size-held 1.27× → size-held direct 1.17× (γ ≈ 0.5); travel 3.07×; total 2.12× per dwelling. Descriptive medians + the form/size ladder. (summary.md Heat / Car travel.)

### 4.2 The access axis
- On foot ~24× amenities (52× jobs, 12× people); at own catchment near-parity (~1.2×); at 25 km ~10–14×. (summary.md Access table.)

### 4.3 The rate
- ~6.3× access per kWh; why it exceeds the on-foot count parity at catchment (same reach, ~3× the fuel).

### 4.4 Lock-in
- Per dwelling 2.12× → 1.51× (at equal family size 1.71× → 1.18×); ~45% survives; access deficit 100% unchanged — the hard, technology-immune lock-in.

### 4.5 Self-selection robustness
- Total energy robust (δ* ≈ 1, the structural travel gap); heat more confound-entangled (δ* ≈ 0.3); NS-SeC adds nothing over deprivation; access immune. ⇒ case rests on total energy + access.

## 5. Discussion

### 5.1 Relation to prior work
- Consistency synthesis (lit review §6): building energy reproduces NEED/EHS/Wyatt/Buyuklieva; metered-vs-modelled and unit-dependence reproduce Rode/Norman/Summerfield/Few/Firth; travel same direction as Echenique (extremes vs marginal); access extends Ewing/Elldér; the rate is novel.

### 5.2 What is locked in, and what technology can and cannot offset
- Uniform technology scales energy down but leaves the structural quantities (floor area, distance) intact; access is fixed in the street layout — changes only on rebuild timescales.

### 5.3 The functional unit and why it matters
- Per-dwelling vs per-capita/per-m²; the γ ≈ 0.5 finding; why per-person is a presentation lens, not an inferential unit.

### 5.4 Policy implications
- Access-based neighbourhood assessment; planning for proximity; retrofit + EV necessary but insufficient.

### 5.5 Limitations
- Self-selection / endogeneity (place-level estimand; mover-panel as the definitive future test); ecological inference / MAUP; heat axis confound-sensitivity (lean on total + access); EPC fabric-ratio assumption in the lock-in; England-only; amenity-location endogeneity on access.

## 6. Dissemination

> The measured findings (§4) stand on their own; these are the intended delivery vehicles — planned outputs, not yet built (see [`paper/summary.md`](paper/summary.md) and ROADMAP). Frame as forthcoming, not as results of this paper.

### 6.1 The NEPI scorecard
- An EPC-style rating, but for **neighbourhoods rather than buildings**: the two-axis result (energy spent vs access gained, and the rate) condensed into a single banded score per Output Area, legible like an A–G label.
- Purpose: make the access-per-energy measure usable by planners, residents and policy — a common currency for "how much everyday life this place reaches for the energy it costs."

### 6.2 The online Atlas
- An interactive web map of every English Output Area's NEPI rating: explore the two axes and the rate spatially, compare neighbourhoods, and inspect the components behind a score.
- Purpose: dissemination and scrutiny — turn the national dataset into something explorable rather than a table.

### 6.3 Predictive models and scenarios (gradient boosting)
- Gradient-boosted (XGBoost) models predicting a neighbourhood's NEPI from its form, fabric and fleet inputs, so combinations of those inputs can be simulated and re-scored.
- Carries pre-defined scenarios — full fleet electrification; buildings brought to best-practice thermal efficiency — i.e. the lock-in analysis (§4.4) generalised to user-chosen interventions, showing how far any given combination moves a neighbourhood's score (and how much of the access deficit it leaves untouched).

## 7. Conclusion
- Two measured axes and a rate; a detached neighbourhood spends more and reaches less; technology cannot move the access deficit; judge neighbourhoods by the access each unit of energy buys.

## References
- From [`paper/references.bib`](paper/references.bib).
