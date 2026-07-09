# The Neighbourhood Energy Performance Index: Energy Spent versus Access Gained

> **Status: complete first draft (two-axis frame).** All IMRaD sections are drafted in prose.
> Canonical numbers are cross-checked against [`paper/results_snapshot.txt`](paper/results_snapshot.txt)
> and [`paper/summary.md`](paper/summary.md); the literature synthesis is in
> [`paper/literature_review.md`](paper/literature_review.md) (§2, §6); the methodological
> decisions and caveats in [`paper/method_notes.md`](paper/method_notes.md). Reported ratios carry
> local-authority-clustered 95% confidence intervals. The References section and a few entries in
> [`paper/references.bib`](paper/references.bib) remain to be finalised. The earlier three-surface /
> A–G-scorecard draft is archived at
> [`paper/archive/PAPER_three_surface_deferred.md`](paper/archive/PAPER_three_surface_deferred.md).

---

## Abstract

A neighbourhood is conventionally judged by the energy it consumes, not by what that energy buys. This paper judges English neighbourhoods on two measured axes (energy spent and everyday access gained) and the rate between them, across 178,353 Census 2021 Output Areas. Household energy combines metered gas and electricity with National Travel Survey-anchored car-travel energy; access is the count of amenities, jobs and people reachable over the road network. Energy is modelled per dwelling, with household size and floor area as freely estimated controls, and is metered rather than SAP-modelled, avoiding the performance gap. A detached neighbourhood spends about 2.1× a flat's energy per dwelling, yet a flat reaches about 24× the amenities on foot and returns about 3.6× the access per kilowatt-hour. No decarbonisation scenario, up to full deployment of insulation, heat pumps and electric vehicles, closes half the energy gap, and none closes any of the access gap: the access deficit of dispersed form is structural and technology-immune.

## 1. Introduction

Energy policy and neighbourhood assessment judge places by the energy they consume. This ignores what energy is for: the everyday life a household can reach (work, shops, schools, services) for the energy it spends. A place that consumes little but reaches little is not efficient in any useful sense.

This paper judges a neighbourhood by access gained per unit of energy spent. The premise is ecological: a system extracts more function from each unit of energy when it is compact and complex, cycling energy through many close exchanges rather than letting it dissipate (Jacobs, 1969, 2000). Compact form does the former, dispersed form the latter. The object of study is therefore the rate between two measured quantities (energy spent and access gained), not energy alone.

The question is pressing because decarbonisation policy assumes technology closes the gap: insulation and electric vehicles are expected to make every form efficient. Whether that holds is unresolved. If the energy cost of dispersed form is structural (fixed in floor area and distance), technology lowers it uniformly without closing the gap between forms, and the access cost is untouched entirely.

Prior work measures energy or access, rarely together; uses inconsistent functional units; relies on modelled rather than metered energy; and works at coarse spatial units (§2). This paper measures both axes for 178,353 of England's 178,605 Output Areas, using metered household and car-travel energy against network-based access with the rate between them, modelling energy per dwelling with household size and floor area as free controls, which resolves the functional-unit ambiguity. The contribution is two measured axes and an explicit rate at national fine-grained scale; the access-per-energy rate has no precedent. The estimand is place-level, and its exposure to residential self-selection is bounded directly.

## 2. Background

### 2.1 Building energy and urban form

Compact dwelling forms lose less heat: shared walls and a lower surface-to-volume ratio cut the exposed envelope. Simulation puts the heat-demand contrast between detached and compact forms as high as sixfold (Rode et al., 2014). Empirically, English metered data place detached gas consumption at about twice a flat's (NEED; the English Housing Survey), and dwelling-level regressions confirm dwelling type and floor area as the dominant physical drivers (Wyatt, 2013; Buyuklieva et al., 2023).

### 2.2 Transport energy, density and accessibility

Density associates with lower car use (Newman and Kenworthy, 1989), but the marginal effect on travel is modest: Echenique et al. (2012) estimate about a 10% reduction in vehicle-miles per density doubling in England. The relationship is better read as one of accessibility than of density as such (Ewing et al., 2018), and local amenity access reduces driving directly (Elldér et al., 2022). The compact-city case is contested. Gaigné, Riou and Thisse (2012) show that densification need not lower emissions once firm and household relocation and the resulting commuting patterns are allowed to adjust, and Mindali, Raveh and Salomon (2004) find no simple cross-city relationship between density and transport energy. The disagreement turns on scope (household or dwelling versus whole-metropolitan) and on whether accessibility or density is the operative variable, which is why this study measures access directly rather than inferring it from density.

### 2.3 The EPC performance gap

Energy Performance Certificate ratings are SAP-modelled estimates, not measurements, and systematically over-predict consumption, most for the largest, least efficient dwellings (Crawley et al., 2019; Summerfield et al., 2019; Few et al., 2023; Firth et al., 2024). Using modelled energy as a dependent variable would inflate the dwelling-type gradient with model error, so this study uses metered energy throughout.

### 2.4 The functional-unit problem

Conclusions about density and energy depend on the functional unit: a low-density premium that is large per capita narrows or reverses per square metre (Norman et al., 2006). The cause is that energy is sub-linear in household size, since larger households use less per person (Druckman and Jackson, 2008; Huebner and Shipworth, 2017), so per-person normalisation imposes an elasticity of one that the data reject. This motivates modelling energy per dwelling with household size as a freely estimated control rather than a denominator.

### 2.5 Ecological analysis and self-selection

Two inferential cautions apply. Analysis at the Output Area is ecological: associations need not hold at the household level and are subject to the modifiable areal unit problem. And households are not assigned to dwelling types at random (car-oriented households self-select into low-density homes; Cao, Mokhtarian and Handy, 2009), so cross-sectional gradients may overstate the causal effect of form. The magnitude of that overstatement is itself debated: meta-analysing the travel–built-environment evidence, Stevens (2017) concludes that compact development does reduce driving, but that the effect is modest once residential self-selection is accounted for, while the studies he reviews correct for self-selection to varying and often partial degrees. Both cautions are addressed in §3.8 and §5.5.

## 3. Data and Methods

### 3.1 Unit of analysis

The unit is the 2021 Census Output Area (OA): the smallest published Census geography, about 125 households and 300 residents. Of England's 178,605 Output Areas, the 178,353 with usable metered energy are analysed. Wales is excluded because the English Index of Multiple Deprivation, used as a control, has no comparable Welsh equivalent. The Output Area is finer than the Lower-layer Super Output Area (~1,500 residents) used by most small-area energy work, reducing within-unit heterogeneity in dwelling type, the exposure. Ecological aggregation is addressed in §5.5.

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

The home component is metered (DESNZ sub-national statistics), not modelled. Postcode-level gas and electricity, each a meter count and a mean per meter, are aggregated to the Output Area by a meter-weighted mean, summed, and divided by Census households. Gas is weather-corrected; both cover domestic meters only. The denominator is all households, so communal- or electrically-heated dwellings, commoner among flats, are not fully captured on the gas series; restricting to well-measured areas (gas-meter coverage ≥ 0.9) leaves the heat gap unchanged (1.61× against the 1.60× headline), so this does not drive the result. The same restriction lowers the total-energy gap to 1.74×, but that is a composition effect of the subsample, which under-represents the off-gas rural detached homes that drive the most, not a weakening of the metered heat result. Metered energy is used because SAP/EPC ratings over-predict consumption, most for large detached dwellings, and would inflate the dwelling-type gradient with model error (§2.3; Few et al., 2023; Summerfield et al., 2019; Firth et al., 2024).

The travel component is total car-travel energy of a location's residents, not the commute alone (about one-sixth of car mileage). It is estimated by constrained disaggregation. The National Travel Survey (NTS9904) gives car-driver miles per person by 2021 rural-urban class, from about 2,500 (urban) to 5,200 (rural). Each class total is distributed across its Output Areas by car ownership and commute distance, constrained so each class's population-weighted mean reproduces the survey figure; only the within-class distribution is estimated. Miles are converted at the local fleet's energy per mile: 0.93 (internal combustion) and 0.32 (electric) kWh per mile, blended by electric-vehicle share (DVLA). Sensitivity to the one free parameter, the commute-distance elasticity, is reported in §4.5.

### 3.4 The access axis

Access is the count of opportunities reachable from an Output Area over the road network. The England network (OS Open Roads, ~3.6 million junctions) is built once into a routable graph (cityseer), and counts are accumulated outward from each Output Area's nearest node. Three quantities are measured: amenities (seven destination types: general practitioners, pharmacies and hospitals from the NHS; schools from the Get Information about Schools register; food outlets and supermarkets from the Food Standards Agency; greenspace from Ordnance Survey); jobs (workplace jobs, each workplace weighted by its job count, Census WP101EW); and people (resident population). Each is read at 1,600 m, the Output Area's own car catchment (§3.5), and 25,600 m. On-foot reach is a subset of the catchment, which is a subset of the drive. Access depends on location, not occupants, and is therefore invariant to the household denominator and, by construction, to residential self-selection (§3.8).

### 3.5 The rate

The rate is the access a neighbourhood reaches for the energy it spends driving: amenities reachable per kilowatt-hour of car-travel energy. Per Output Area it is the amenity count within the area's own car catchment (its NTS car-driver distance per person divided by 370 trips per person per year, bounded to [1,600 m, 25,600 m]), divided by its car-travel energy (§3.3). The flat-to-detached ratio of this rate is, by construction, the product of the two reported axes: the access advantage (catchment amenities, flat:detached) times the energy saving (car-travel energy, detached:flat). It is reconstructable from Tables 2 and 3, not estimated as a separate per-area model. The two terms are not independent (the catchment radius and the car-travel energy are both functions of how far residents drive), which is the intended reading: the rate prices the access a neighbourhood reaches against the energy it spends reaching it.

### 3.6 The compositional model

All flat-to-detached contrasts come from one compositional, no-intercept regression on each Output Area's dwelling-type shares (flat, terraced, semi-detached, detached, residual) as fractions summing to one. Without an intercept each coefficient is the modelled outcome of a neighbourhood composed entirely of one type, and the ratio is exp(b_detached − b_flat). Regressions are household-weighted. The estimate is read at the pure-composition vertices, which few areas approach, so each ratio is a lower bound on the gap.

Energy enters in logarithms, per dwelling. Household size and floor area are covariates with freely estimated coefficients, not denominators: per-person or per-square-metre normalisation imposes an elasticity of one, but the estimated household-size elasticity of heat is 0.5, so those units are reported descriptively only (§2.4). Four confounds are held throughout: building age, deprivation (overall IMD and its income domain), tenure (social- and private-rented shares), and climate (heating-degree-days). Adding household size, then floor area, yields the total, family-size-held, and size-held gaps.

Access counts are non-negative and frequently zero, so they are fitted with a Poisson log-link, not a linear model. The access regression holds the income domain but not density (the mechanism under study) and not the overall IMD, whose barriers and living-environment sub-domains are themselves access measures. Households enter as analytic weights, so the effective sample is the number of areas, not the summed household count.

Standard errors are clustered by local-authority district (about 309 in England), because neighbouring Output Areas share unobserved local conditions and are not 178,353 independent observations. Every reported ratio carries a 95% confidence interval on this basis; the intervals are wider than heteroskedasticity-robust ones but do not change the conclusions. Ratios that combine several fits (the rate, the surviving-gap share, the mediated fraction) are given cluster-bootstrap intervals.

### 3.7 Decarbonisation scenarios

The energy axis is recomputed under a ladder of technology scenarios and the gap re-estimated at each. Three levers act on the per-dwelling energy. Fabric: metered gas is scaled by the EPC fabric-improvement ratio (modelled potential over current intensity, median 0.5); metered electricity is unchanged. Heat pump: the gas heat is instead delivered as electricity at the heat pump's delivered-energy factor, boiler efficiency over seasonal coefficient of performance (0.90/2.8 = 0.32), since a heat pump draws less delivered energy than a boiler for the same useful heat. Electrification: car energy is recomputed at the electric fleet's energy per mile, mileage fixed. Both EPC ratio terms are modelled, so the performance gap cancels, and anchoring to the metered bill holds the scale of §3.3. A lever is applied at a scenario's uptake fraction, blending transformed and untransformed energy. The scenarios are: the status quo; each lever alone at full deployment; the Climate Change Committee's Seventh Carbon Budget (2025) Balanced Pathway at 2040 (heat pumps in half of homes, three-quarters of cars electric); and full deployment of all three. Access is unchanged in every scenario by construction, since no fabric, heat pump or drivetrain moves a destination closer.

### 3.8 Robustness and the estimand

The estimand is place-level, the energy and access profile of a neighbourhood type conditional on the observed confounds, not a household treatment effect, because households self-select into dwelling types. Three checks bound selection on unobservables. First, access depends on location, so it is immune by construction. Second, occupational class (higher managerial and professional share, Census TS062) is added to the confounds; movement in the gap measures selection on this observable. Third, an Oster (2019) bound on a continuous detached-share gradient gives δ*, the strength of unobserved selection, relative to the observed confounds, required to nullify the gap. As §4.5 reports, total energy is robust and heat is not, so the argument rests on total energy and access.

## 4. Results

### 4.1 The energy axis

A detached neighbourhood spends 2.12× (95% CI 2.01–2.24) a flat's energy per dwelling; the gradient is steepest in travel, 3.07× (2.82–3.34) (Table 2).

| kWh per dwelling/yr | Flat | Terraced | Semi | Detached | flat→detached |
| --- | --: | --: | --: | --: | --: |
| Heat (metered gas + electricity) | 10,194 | 12,995 | 13,876 | 15,020 | 1.60× |
| Car travel (NTS-anchored) | 3,240 | 5,088 | 6,660 | 9,272 | 3.07× |
| Total | 13,674 | 18,265 | 20,564 | 23,832 | 2.12× |

The type columns are observed medians; the ratio is the compositional flat-to-detached estimate (§3.6), not their quotient. The total is the per-type median of household total energy, and need not equal the sum of the separate heat and car medians, because a median is not additive. The heat gap decomposes from 1.60× (1.45–1.76, total) to 1.27× (1.16–1.40) holding household size to 1.17× (1.06–1.30) also holding floor area: on the log scale, size and occupancy account for about 66% (54–84%) of the gap, leaving a direct fabric and exposure effect of 1.17× (a 17% premium). This decomposition is a covariate adjustment on cross-sectional data, not an identified causal mediation, since floor area and household size are themselves selected. An unweighted intercept-OLS specification gives a parallel ladder (1.58× to 1.30× to 1.20×, 60% accounted for), so the exact split is specification-dependent; the compositional figures are primary. The floor-area elasticity of heat is 0.54 and the household-size elasticity 0.5, both below one, so per-square-metre intensity falls with dwelling size and per-person energy falls with occupancy, and neither is used as the unit (§3.6).

### 4.2 The access axis

A flat neighbourhood reaches far more at every distance, the gap narrowing as distance grows (Table 3). On foot it reaches 23.9× (95% CI 17–33) the amenities, 52.4× (30–91) the jobs and 12.5× (9–17) the people of a detached one. These on-foot ratios are modelled pure-type contrasts read at 100%-of-type compositions; the corresponding dominant-type median ratio is smaller (amenities 9.5×, jobs 11.6×), and the support-restricted median comparing areas at least half of one type is 11.7×, because few real areas approach a pure vertex and several detached medians are zero. The pure-type contrast states the gap between idealised extremes; the median states the gap between typical flat and detached areas, and both are reported. At each area's own car catchment the raw counts nearly converge (amenities 1.17×, confidence interval 0.99–1.38, spanning parity); at a 25 km drive the flat still leads 10–14×.

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

A flat returns about 3.6× (95% CI 2.9–4.4) the access per kilowatt-hour of car energy, the product of the access advantage and the energy saving. A flat and a detached home reach a comparable count at their own catchments (1.17×, confidence interval spanning parity, Table 3), but the detached spends about three times the car energy to do so (3.07×, Table 2), so per kilowatt-hour the flat leads 1.17 × 3.07 ≈ 3.6×. The figure depends on how the two axes are conditioned. Matching the access side's income-only conditioning on the energy side raises the energy saving to 4.24× and the rate to about 5.0× (95% CI 3.9–6.0); pricing access instead against an already-electrified travel cost, which removes the dependence between driving distance and catchment (§3.5), lowers the rate to about 2.2× (95% CI 1.8–2.5). The dominant-type median rate is 2.9×. The headline 3.6× sits within this range rather than at its edge.

### 4.4 Decarbonisation scenarios and lock-in

No technology scenario closes much of the energy gap, and none moves the access gap at all (Table 4; all ratios as-lived, per dwelling, compositional). The three levers act differently. Insulation closes the gap: fabric alone, fully deployed, moves it from 2.12× to 1.83×, 20% of the log-gap, because detached homes carry more fabric headroom. Heat pumps do not: fully deployed they move the gap to 2.16×, marginally wider, because a heat pump sharply cuts the low-gap heat component in delivered-energy terms and unmasks the higher-gap car travel. Electrification closes the most of any single lever, to 1.85× (18%), attacking travel where the gap is largest. The Balanced Pathway for 2040 leaves the gap at 1.89×, closing 16%; even full deployment of all three leaves 1.68×, 31% closed and 69% surviving. Fabric plus full electrification without heat pumps gives 1.51×, the conventional lower bound. On foot the access gap is 23.9× in every scenario, unchanged.

| Total energy gap (per dwelling, as-lived) | Det:Flat | log-gap closed |
| --- | --: | --: |
| Status quo | 2.12× | — |
| Fabric only (100%) | 1.83× | 20% |
| Heat pumps only (100%) | 2.16× | −2% |
| EVs only (100%) | 1.85× | 18% |
| CCC Balanced Pathway 2040 | 1.89× | 16% |
| Full rollout (100%) | 1.68× | 31% |

Heat pumps are essential to decarbonisation on the carbon axis (the electricity that drives them is increasingly low-carbon); their near-neutral effect here is specific to the delivered-energy form gap, and reflects that the gap is structural in floor area and distance, not in the heat carrier.

### 4.5 Robustness

The energy gaps could reflect who chooses detached rather than the form itself. Three checks (§3.8) bound this. Access depends on location and is immune by construction. Adding occupational class (NS-SeC) to the confounds leaves the gaps unchanged (Table 5). An Oster bound on the detached-share gradient gives δ* ≈ 1.1 for total energy, so unobserved selection would have to be about as strong as all observed confounds combined to nullify it, but δ* ≈ 0.3 for heat, whose contrast is largely deprivation and tenure. The argument therefore rests on total energy and access, not on heat alone.

| Detached-share gradient | raw | + confounds | + NS-SeC | Oster δ* |
| --- | --: | --: | --: | --: |
| Total energy | 1.87× | 1.28× | 1.28× | 1.1 |
| Heat | 1.30× | 1.02× | 1.02× | 0.3 |

The heat gap is also robust to the meter-versus-household denominator (§3.3). Restricting to Output Areas whose electricity-meter count is within half the household count, which removes areas where meters and dwellings diverge most, leaves the heat gap at 1.55× and the total at 2.08×; winsorising the top 0.5% of per-dwelling energy leaves the heat gap at 1.59×.

The travel estimate has one free parameter, the commute-distance elasticity (0.30). Because each rural-urban class total is held fixed (§3.3), the elasticity only redistributes miles within a class, while the flat-to-detached contrast is largely between classes. Varying it across 0.15, 0.30 and 0.45 moves the travel contrast only from 2.74× to 2.98× (dominant-type median), confirming the contrast is insensitive to it.

The result is not an artefact of the Output Area zonation (the modifiable areal unit problem). Re-aggregating the frame household-weighted to the two coarser census scales that nest above the OA and re-fitting the same model, the total-energy gap holds: 2.12× at OA, 1.88× at LSOA, 1.72× at MSOA on the compositional contrast, and 1.74×, 1.59×, 1.47× comparing dominant-type medians. The mild attenuation with scale is expected, because coarsening averages away within-unit dwelling-type mixing. The heat sub-component's compositional contrast falls faster and reverses at MSOA (1.60× to 1.17× to 0.91×), but this is an out-of-support extrapolation, since almost no MSOA approaches a pure dwelling-type composition; the support-respecting dominant-type median holds heat positive at every scale (1.47× to 1.32× to 1.17×), consistent with total energy being the robust axis.

## 5. Discussion

### 5.1 Relation to prior work

The building-energy result reproduces the English literature: NEED and the English Housing Survey put detached gas at about twice a flat's, as do dwelling-level regressions (Wyatt, 2013; Buyuklieva et al., 2023). The metered 1.60× is smaller than the up-to-sixfold contrast of simulation studies (Rode et al., 2014) by exactly the performance gap (SAP over-predicts most for large detached dwellings; Summerfield et al., 2019; Few et al., 2023; Firth et al., 2024), which is why metered energy is used. Dependence on the functional unit reproduces Norman et al. (2006), and the sub-unit household-size elasticity reproduces Huebner and Shipworth (2017). The travel gradient is the same direction as the urban-form literature but larger than its central estimates: Echenique et al. (2012) find a modest marginal effect (about 10% fewer vehicle-miles per density doubling), whereas this is a cross-sectional contrast between the extremes of the distribution, read against the observed rural-urban mileage gradient. The access axis operationalises the reframing of the density–travel relationship as one of accessibility (Ewing et al., 2018; Elldér et al., 2022); the access-per-energy rate has no direct antecedent. This household- and neighbourhood-scale finding does not settle the metropolitan-scale debate. Gaigné, Riou and Thisse (2012) show that whether densification lowers total emissions depends on how firms and households relocate in response, a general-equilibrium question that a cross-sectional design cannot address.

### 5.2 What technology can and cannot offset

Uniform technology scales energy down without changing the structural quantities. Insulation lowers heat loss per square metre but not floor area or surface; electrification lowers energy per mile but not the miles. The detached-to-flat ratio is therefore near-invariant to uniform improvement: between 55% (fabric and electrification) and 69% (adding heat pumps) of the energy gap survives, and heat pumps, which cut delivered heat sharply, leave it marginally wider (§4.4). The access deficit is fixed in the street layout, untouched by any of these, and changes only when places are rebuilt. Access is the binding, technology-immune component, and must be planned for directly.

### 5.3 The functional unit

Whether a detached neighbourhood is more energy-intensive depends on the unit (§3.6). Per dwelling and per capita it spends more; per square metre the gap nearly closes; per person it narrows because detached homes hold more people. Since the household-size elasticity of energy is about 0.5, per-person normalisation imposes a false elasticity of one and is reported descriptively only. The headline is per dwelling (the unit at which energy is metered and billed), with household size held as a covariate.

### 5.4 Policy implications

A neighbourhood rating that counts only consumption rewards low-density form for its low per-capita heat while ignoring its travel and access penalties. A two-axis measure (energy spent against access gained) values proximity, which retrofit and electrification cannot supply. Decarbonising buildings and vehicles is necessary but not sufficient; access must be a planned outcome, scored alongside energy. Two caveats temper this. Energy saved by efficiency is partly spent elsewhere (the rebound effect), so dwelling-level gains need not scale to economy-wide reductions. And access is capitalised into land and rent, so the advantages of compact form are not shared evenly; a measure that scores access should be read alongside distributional evidence, not as a welfare verdict on its own.

### 5.5 Limitations

Five limitations qualify the result. First, residential self-selection: the estimand is place-level, not a household treatment effect, and the heat axis is confound-sensitive (§4.5); the definitive test would difference out fixed preferences using movers in panel data. Second, ecological inference: the analysis is at Output Area level and does not resolve within-area heterogeneity. Third, the lock-in scenario assumes the EPC potential-to-current ratio captures achievable fabric improvement. Fourth, the access counts treat amenity location as exogenous, though amenities partly follow demand: if shops and services locate where population is dense, the on-foot advantage of compact areas is partly a supply response rather than a pure effect of form. The near-convergence at each area's own catchment (amenities 1.17×, confidence interval spanning parity) is consistent with the advantage being one of proximity rather than of supply, since at equal travel reach the counts are similar; the access figures should be read as what residents can currently reach, not as the causal effect of form on amenity provision. Fifth, the analysis is England-only, pending a harmonised Welsh deprivation measure.

## 6. Dissemination

The measure is intended for delivery in three forms. None is yet built; they are planned outputs, not results of this paper.

### 6.1 The NEPI scorecard

An Energy Performance Certificate-style rating, but for neighbourhoods rather than buildings: the two axes and the rate condensed into a single banded score per Output Area, legible like an A–G label, to make the measure usable by planners, residents and policy.

### 6.2 The online Atlas

An interactive web map of every English Output Area's score, allowing the two axes and the rate to be explored spatially, neighbourhoods compared, and the components behind a score inspected.

### 6.3 Predictive models and scenarios

Gradient-boosted (XGBoost) models predicting a neighbourhood's score from its form, fabric and fleet, so combinations can be simulated and re-scored. Pre-defined scenarios (full electrification, best-practice fabric) generalise the lock-in analysis (§4.4) to any intervention, showing how far it moves a score and how much of the access deficit it leaves.

## 7. Conclusion

A detached neighbourhood spends about twice a flat's energy per dwelling and reaches a fraction of its everyday destinations; a flat returns between three and four times the access per kilowatt-hour. Full deployment of insulation, heat pumps and electric vehicles leaves most of the energy gap and all of the access gap intact, because technology changes the efficiency of floor area and distance but not their quantity. Judged by access gained per unit of energy spent, compact form is the more efficient, and the difference is structural, fixed until places are rebuilt.

## References

The reference database is [`paper/references.bib`](paper/references.bib); every author-date citation above resolves against its keys, and the formatted list is rendered from it at typesetting. Three entries remain to be finalised (marked `TODO` in the file): the full citation for Ewing et al. (2018), and the author fields for the Sustainability systematic review (systematic2023) and the NEBULA dataset (nebula2025).
