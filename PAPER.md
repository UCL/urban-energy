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

The unit is the 2021 Census Output Area (OA): the smallest published Census geography, about 125 households and 300 residents. The analysis covers England's ~178,000 Output Areas. Wales is excluded because the English Index of Multiple Deprivation, used as a control, has no comparable Welsh equivalent. The Output Area is finer than the Lower-layer Super Output Area (~1,500 residents) used by most small-area energy work, reducing within-unit heterogeneity in dwelling type, the exposure. Ecological aggregation is addressed in §5.5.

### 3.2 Data sources

All inputs are open data, processed to the Output Area on the British National Grid (EPSG:27700). Table 1 lists the sources.

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

Energy is delivered energy in kilowatt-hours per dwelling per year (not primary energy or carbon), in two components: home and travel.

The home component is metered (DESNZ sub-national statistics), not modelled. Postcode-level gas and electricity, each a meter count and a mean per meter, are aggregated to the Output Area by a meter-weighted mean, summed, and divided by Census households. Gas is weather-corrected; both cover domestic meters only. Metered energy is used because SAP/EPC ratings over-predict consumption, most for large detached dwellings, and would inflate the dwelling-type gradient with model error (§2.3; Few et al., 2023; Summerfield et al., 2019; Firth et al., 2024).

The travel component is total car-travel energy of a location's residents, not the commute alone (about one-sixth of car mileage). It is estimated by constrained disaggregation. The National Travel Survey (NTS9904) gives car-driver miles per person by 2021 rural-urban class, from about 2,500 (urban) to 5,200 (rural). Each class total is distributed across its Output Areas by car ownership and commute distance, constrained so each class's population-weighted mean reproduces the survey figure; only the within-class distribution is estimated. Miles are converted at the local fleet's energy per mile — 0.93 (internal combustion) and 0.32 (electric) kWh per mile, blended by electric-vehicle share (DVLA). Sensitivity to the one free parameter, the commute-distance elasticity, is reported in §4.5.

### 3.4 The access axis

Access is the count of opportunities reachable from an Output Area over the road network. The England network (OS Open Roads, ~3.6 million junctions) is built once into a routable graph (cityseer), and counts are accumulated outward from each Output Area's nearest node. Three quantities are measured: amenities (seven destination types — general practitioners, pharmacies and hospitals from the NHS; schools from the Get Information about Schools register; food outlets and supermarkets from the Food Standards Agency; greenspace from Ordnance Survey); jobs (workplace jobs, each workplace weighted by its job count, Census WP101EW); and people (resident population). Each is read at 1,600 m, the Output Area's own car catchment (§3.5), and 25,600 m. On-foot reach is a subset of the catchment, which is a subset of the drive. Access depends on location, not occupants, and is therefore invariant to the household denominator and, by construction, to residential self-selection (§3.8).

### 3.5 The rate

The rate is amenities reachable per kilowatt-hour of car-travel energy, formed per Output Area. The numerator is the amenity count within the Output Area's own car catchment: its NTS car-driver distance per person divided by 370 car trips per person per year, bounded to [1,600 m, 25,600 m]. The denominator is car-travel energy (§3.3), in kilowatt-hours per dwelling per year. The reported value is the compositional flat-to-detached estimate (§3.6).

### 3.6 The compositional model

All flat-to-detached contrasts come from one compositional, no-intercept regression on each Output Area's dwelling-type shares (flat, terraced, semi-detached, detached, residual) as fractions summing to one. Without an intercept each coefficient is the modelled outcome of a neighbourhood composed entirely of one type, and the ratio is exp(b_detached − b_flat). Regressions are household-weighted. The estimate is read at the pure-composition vertices, which few areas approach, so each ratio is a lower bound on the gap.

Energy enters in logarithms, per dwelling. Household size and floor area are covariates with freely estimated coefficients, not denominators: per-person or per-square-metre normalisation imposes an elasticity of one, but the estimated household-size elasticity of heat is 0.5, so those units are reported descriptively only (§2.4). Four confounds are held throughout: building age, deprivation (overall IMD and its income domain), tenure (social- and private-rented shares), and climate (heating-degree-days). Adding household size, then floor area, yields the total, family-size-held, and size-held gaps.

Access counts are non-negative and frequently zero, so they are fitted with a Poisson log-link, not a linear model. The access regression holds the income domain but not density — the mechanism under study — and not the overall IMD, whose barriers and living-environment sub-domains are themselves access measures.

### 3.7 The lock-in scenario

The energy axis is recomputed under best-practice fabric and full electrification, and the gap re-estimated. Fabric: metered gas is scaled by the EPC fabric-improvement ratio (modelled potential over current intensity, median 0.5); metered electricity is unchanged. Both ratio terms are EPC-modelled, so the performance gap cancels, and anchoring to the metered bill holds the scale of §3.3. Electrification: car energy is recomputed at the electric fleet's energy per mile, mileage fixed. Access is unchanged by construction.

### 3.8 Robustness and the estimand

The estimand is place-level — the energy and access profile of a neighbourhood type, conditional on the observed confounds — not a household treatment effect, because households self-select into dwelling types. Three checks bound selection on unobservables. First, access depends on location, so it is immune by construction. Second, occupational class (higher managerial and professional share, Census TS062) is added to the confounds; movement in the gap measures selection on this observable. Third, an Oster (2019) bound on a continuous detached-share gradient gives δ*, the strength of unobserved selection, relative to the observed confounds, required to nullify the gap. As §4.5 reports, total energy is robust and heat is not, so the argument rests on total energy and access.

## 4. Results

### 4.1 The energy axis

A detached neighbourhood spends 2.12× a flat's energy per dwelling; the gradient is steepest in travel (Table 2).

| kWh per dwelling/yr | Flat | Terraced | Semi | Detached | flat→detached |
| --- | --: | --: | --: | --: | --: |
| Heat (metered gas + electricity) | 10,194 | 12,995 | 13,876 | 15,020 | 1.60× |
| Car travel (NTS-anchored) | 3,240 | 5,088 | 6,660 | 9,272 | 3.07× |
| Total | 13,674 | 18,265 | 20,564 | 23,832 | 2.12× |

The type columns are observed medians; the ratio is the compositional flat-to-detached estimate (§3.6), not their quotient. The heat gap decomposes from 1.60× (total) to 1.27× holding household size to 1.17× also holding floor area: size and occupancy mediate about 71%, leaving a 17% direct fabric and exposure effect. The floor-area elasticity of heat is 0.54 and the household-size elasticity 0.5, both below one — so per-square-metre intensity falls with dwelling size and per-person energy falls with occupancy, and neither is used as the unit (§3.6).

### 4.2 The access axis

A flat neighbourhood reaches far more at every distance, the gap narrowing as distance grows (Table 3). On foot it reaches 23.9× the amenities, 52.4× the jobs and 12.5× the people of a detached one. At each area's own car catchment the raw counts nearly converge (amenities 1.2×); at a 25 km drive the flat still leads 10–14×.

| Within reach (median) | Flat | Terraced | Semi | Detached | flat:det |
| --- | --: | --: | --: | --: | --: |
| Amenities, on foot | 209 | 119 | 67 | 22 | 23.9× |
| Amenities, own catchment | 2,531 | 2,255 | 2,765 | 2,776 | 1.2× |
| Amenities, 25 km | 20,812 | 9,950 | 8,796 | 4,653 | 10.4× |
| Jobs, on foot | 6,927 | 3,790 | 2,100 | 598 | 52.4× |
| Jobs, own catchment | 102,652 | 87,077 | 107,065 | 101,215 | 1.7× |
| Jobs, 25 km | 807,658 | 382,638 | 337,938 | 173,447 | 14.3× |
| People, on foot | 17,838 | 11,861 | 8,207 | 2,766 | 12.5× |
| People, own catchment | 255,216 | 236,228 | 285,772 | 270,115 | 1.2× |
| People, 25 km | 2,343,165 | 1,032,734 | 913,638 | 472,236 | 11.1× |

Population density is 79 people per hectare in a flat neighbourhood against 14 in a detached one, a factor of 5.7.

### 4.3 The rate

A flat returns 6.3× the access per kilowatt-hour of car energy. The near-convergence of counts at the catchment (1.2×, Table 3) does not close the rate: a detached area reaches a comparable count only by driving far enough to spend about three times the car energy (Table 2), so per kilowatt-hour the flat leads 6.3×.

### 4.4 Lock-in

Best-practice fabric and full electrification close the energy gap only part-way (Table 4): per dwelling from 2.12× to 1.51×, and at equal family size from 1.71× to 1.18×. About 45% of the excess survives — a larger home still loses more heat, and electrification cuts energy per mile but not the miles. The access gap does not move: 24× on foot, before and after.

| Total energy gap (per dwelling) | now | optimised |
| --- | --: | --: |
| As-lived | 2.12× | 1.51× |
| At equal family size | 1.71× | 1.18× |

### 4.5 Robustness

The energy gaps could reflect who chooses detached rather than the form itself. Three checks (§3.8) bound this. Access depends on location and is immune by construction. Adding occupational class (NS-SeC) to the confounds leaves the gaps unchanged (Table 5). An Oster bound on the detached-share gradient gives δ* ≈ 1.1 for total energy — unobserved selection would have to be about as strong as all observed confounds combined to nullify it — but δ* ≈ 0.3 for heat, whose contrast is largely deprivation and tenure. The argument therefore rests on total energy and access, not on heat alone.

| Detached-share gradient | raw | + confounds | + NS-SeC | Oster δ* |
| --- | --: | --: | --: | --: |
| Total energy | 1.87× | 1.28× | 1.28× | 1.1 |
| Heat | 1.30× | 1.02× | 1.02× | 0.3 |

The travel estimate has one free parameter, the commute-distance elasticity (0.30). Because each rural-urban class total is held fixed (§3.3), the elasticity only redistributes miles within a class, while the flat-to-detached contrast is largely between classes; the contrast is therefore insensitive to it.

## 5. Discussion

### 5.1 Relation to prior work

The building-energy result reproduces the English literature: NEED and the English Housing Survey put detached gas at about twice a flat's, as do dwelling-level regressions (Wyatt, 2013; Buyuklieva et al., 2023). The metered 1.60× is smaller than the up-to-sixfold contrast of simulation studies (Rode et al., 2014) by exactly the performance gap — SAP over-predicts most for large detached dwellings (Summerfield et al., 2019; Few et al., 2023; Firth et al., 2024) — which is why metered energy is used. Dependence on the functional unit reproduces Norman et al. (2006), and the sub-unit household-size elasticity reproduces Huebner and Shipworth (2017). The travel gradient is the same direction as the urban-form literature but larger than its central estimates: Echenique et al. (2012) find a modest marginal effect (about 10% fewer vehicle-miles per density doubling), whereas this is a cross-sectional contrast between the extremes of the distribution, read against the observed rural-urban mileage gradient. The access axis operationalises the reframing of the density–travel relationship as one of accessibility (Ewing et al., 2018; Elldér et al., 2022); the access-per-energy rate has no direct antecedent.

### 5.2 What technology can and cannot offset

Uniform technology scales energy down without changing the structural quantities. Insulation lowers heat loss per square metre but not floor area or surface; electrification lowers energy per mile but not the miles. The detached-to-flat ratio is therefore near-invariant to uniform improvement: about 45% of the energy excess survives best fabric and full electrification (§4.4). The access deficit is fixed in the street layout, untouched by either, and changes only when places are rebuilt. Access is the binding, technology-immune component, and must be planned for directly.

### 5.3 The functional unit

Whether a detached neighbourhood is more energy-intensive depends on the unit (§3.6). Per dwelling and per capita it spends more; per square metre the gap nearly closes; per person it narrows because detached homes hold more people. Since the household-size elasticity of energy is about 0.5, per-person normalisation imposes a false elasticity of one and is reported descriptively only. The headline is per dwelling — the unit at which energy is metered and billed — with household size held as a covariate.

### 5.4 Policy implications

A neighbourhood rating that counts only consumption rewards low-density form for its low per-capita heat while ignoring its travel and access penalties. A two-axis measure — energy spent against access gained — values proximity, which retrofit and electrification cannot supply. Decarbonising buildings and vehicles is necessary but not sufficient; access must be a planned outcome, scored alongside energy.

### 5.5 Limitations

Five limitations qualify the result. First, residential self-selection: the estimand is place-level, not a household treatment effect, and the heat axis is confound-sensitive (§4.5); the definitive test would difference out fixed preferences using movers in panel data. Second, ecological inference: the analysis is at Output Area level and does not resolve within-area heterogeneity. Third, the lock-in scenario assumes the EPC potential-to-current ratio captures achievable fabric improvement. Fourth, the access counts treat amenity location as exogenous, though amenities partly follow demand. Fifth, the analysis is England-only, pending a harmonised Welsh deprivation measure.

## 6. Dissemination

The measure is intended for delivery in three forms. None is yet built; they are planned outputs, not results of this paper.

### 6.1 The NEPI scorecard

An Energy Performance Certificate-style rating, but for neighbourhoods rather than buildings: the two axes and the rate condensed into a single banded score per Output Area, legible like an A–G label, to make the measure usable by planners, residents and policy.

### 6.2 The online Atlas

An interactive web map of every English Output Area's score, allowing the two axes and the rate to be explored spatially, neighbourhoods compared, and the components behind a score inspected.

### 6.3 Predictive models and scenarios

Gradient-boosted (XGBoost) models predicting a neighbourhood's score from its form, fabric and fleet, so combinations can be simulated and re-scored. Pre-defined scenarios — full electrification, best-practice fabric — generalise the lock-in analysis (§4.4) to any intervention, showing how far it moves a score and how much of the access deficit it leaves.

## 7. Conclusion

A detached neighbourhood spends about twice a flat's energy per dwelling and reaches a fraction of its everyday destinations; a flat returns about six times the access per kilowatt-hour. Best-practice fabric and full electrification leave roughly half the energy gap and all of the access gap intact, because technology changes the efficiency of floor area and distance but not their quantity. Judged by access gained per unit of energy spent, compact form is the more efficient, and the difference is structural — fixed until places are rebuilt.

## References
- From [`paper/references.bib`](paper/references.bib).
