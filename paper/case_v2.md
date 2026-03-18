# Morphology, Energy, and the Cost of Ordinary Access

## Introduction

Energy policy is often framed as a problem of improving the efficiency of individual
technologies: insulating buildings, replacing boilers, and electrifying vehicles. These are
necessary measures, but they do not exhaust the problem. Daily energy expenditure is
also shaped by the wider package of housing form, neighbourhood morphology, transport
dependence, and local service access. These conditions generate secondary effects and can
lock in characteristic patterns of energy use.

This note examines the relationship between urban form and energy expenditure
through two linked quantities:

- the **energy cost** of living (building + transport), and
- the **local access return** that energy secures (access to everyday amenities and services).

At city scale, this also bears on service provision efficiency. Dispersed urban form
increases not only household energy demand, but also the energy required for a city to
deliver ordinary access to services across the settlement pattern.

## Scope and Framing

The analysis covers 94 English Built-Up Areas at Output Area (OA) resolution.

- Geography: 94 English BUAs (of 7,147 total — national pipeline in progress)
- Unit: Output Area (67,263 OAs after filtering)
- Sample by dominant type: Flat 18,460, Terraced 20,941, Semi-detached 21,304, Detached 6,558
- Core typology stratification: dominant Census 2021 accommodation type (TS044)
- Access model: trip-type-specific distance-decay local-access model (nearest network distance), with wider-access bound at 4,800m
- Energy model: metered building energy (DESNZ postcode data aggregated to OA) + modelled commute energy (private/public decomposition from Census mode and distance)
- Deprivation control: Census TS011 (OA-native); IMD 2025 (via LSOA foreign key)
- Vehicle fleet: DVLA licensing statistics (via LSOA foreign key)

OA is used because Census data is native at this scale (~130 households), providing a
finer-grained stratification than LSOA (~700 households). Metered building energy is
derived from DESNZ postcode-level statistics aggregated to OA via a spatial postcode-to-OA
lookup, with meter-weighted means. This avoids the ecological aggregation bias present in
the earlier LSOA-level analysis while retaining actual metered consumption as the dependent
variable.

All results are area-level (ecological) associations between OA-level morphology and
energy/access outcomes. They describe structural patterns across neighbourhoods, not causal
effects on individual households. Residential sorting — the possibility that household
characteristics rather than morphology drive the gradient — is addressed through
deprivation stratification in Step 7.

## The Analytical Structure

The analysis proceeds in nine steps:

1. **Building typology** is associated with different building energy demand.
2. **Neighbourhood morphology** is associated with different transport energy demand.
3. **Neighbourhood morphology** is associated with different local amenity access.
4. A **trip-demand schedule** converts access into annual trip budgets by destination type.
5. **Distance allocation** assigns trips to local vs wider access and derives the land-use energy penalty.
6. **Access delivery and penalty** are compared across morphology types.
7. The morphology-access-energy pattern is shown to hold **within deprivation quintiles**.
8. The **relationship across all OAs** confirms the pattern is distribution-wide.
9. An **aggregate cost of sprawl** is estimated for the 94-BUA sample.

Dominant housing type is used as a practical OA-level proxy for broader morphology,
supported by density and network-access evidence.

### Aggregation conventions

Three levels of aggregation are used throughout. To avoid ambiguity, each is stated here
and flagged in figure notes where it applies:

1. **Node → UPRN → OA.** Cityseer accessibility metrics are computed at individual
   street-network nodes (road segment centroids). Each UPRN (address point) is assigned to
   its nearest node, and the OA value is the **mean across all UPRNs** within the OA. The
   UPRN intermediate step is intentional: it acts as a dwelling-weighted spatial aggregation,
   so that a street segment with 50 addresses contributes 50 times while one with no
   dwellings contributes zero. This ensures the OA value reflects the access experienced by
   residents, not the unweighted average across all street segments (which would be diluted
   by roads through parks, industrial areas, or other non-residential land). The same
   aggregation applies to nearest-distance metrics (`cc_*_nearest_max_4800`).

2. **OA → dominant-type group.** Each OA is assigned to a dominant housing type by
   **plurality share** of Census TS044 accommodation types (the single type with the largest
   share, even if below 50%). Where bar charts or summary tables aggregate across OAs
   within a type group, the statistic used is stated explicitly:
   - Steps 1–3 and Step 5 use **medians** (robust to skew in energy and distance
     distributions).
   - Steps 6–7 use **means** (required for trip budgets and penalties that must sum to
     interpretable totals).

3. **Type-group → compact/sprawl.** Step 9 groups Flat + Terraced as "compact" and
   Semi + Detached as "sprawl," then reports **household-weighted means** (each OA
   contributes in proportion to its household count) to produce aggregate energy figures
   that reflect the population-weighted burden.

KDE contour plots (Figures 3, 4, 8) show the **full distribution** of individual OAs
without further aggregation; contours are normalised within each type group for visual
comparability.

## Step 1: Building Typology and Building Energy

![Figure 1: Building energy by dominant housing type](../stats/figures/oa/fig1_building_energy.png)

_Figure note (data and method)._ Building energy uses DESNZ postcode-level domestic energy
statistics (2024), combining metered gas (weather-corrected) and electricity, aggregated to
OA level via meter-weighted means. Each OA is grouped by dominant Census 2021 accommodation
type (`TS044`, plurality share). Bars show median OA values by dominant type; panel B
stratifies by deprivation tercile using Census `TS011`.

Interpretation.

- Building energy is higher in detached-dominant OAs than in flat-dominant OAs.
- The building-energy gradient is steeper at OA level than at LSOA level (1.55x vs 1.28x),
  because OAs are more morphologically homogeneous — less mixing of building types within
  each unit.
- Per-person and per-household readings differ because household size varies by typology:
  average household size rises from approximately 2.1 persons in flat-dominant OAs to 2.5
  in detached-dominant OAs, which dilutes per-capita building energy in larger households.

Median OA values:

| Dwelling type | Building energy (kWh/household) | Building energy (kWh/person) |
| ------------- | ------------------------------- | ---------------------------- |
| Flat          | 10,852                          | 4,962                        |
| Terraced      | 13,601                          | 5,276                        |
| Semi-detached | 14,628                          | 5,841                        |
| Detached      | 16,865                          | 6,874                        |

> Note: Building energy at OA level uses DESNZ postcode-level data (2024) aggregated via
> a spatial postcode-to-OA lookup with meter-weighted means, rather than the LSOA-native
> DESNZ publication used in Case One. This provides finer spatial resolution at the cost
> of introducing aggregation noise from the postcode-to-OA mapping (99.3% match rate,
> median 6.3 postcodes per OA).

## Step 2: Morphology and Transport Energy (The Mobility Penalty)

![Figure 2: Building + transport energy by dominant housing type](../stats/figures/oa/fig2_mobility_penalty.png)

_Figure note (data and method)._ Figure 2 shows two panels using the same type-level
aggregation: (A) **commute-energy estimate** from Census 2021 commute distance bands (`TS058`)
and commute mode counts (`TS061`), annualised (return trips x workdays), with mode-specific
road/rail passenger-energy intensities; (B) a secondary **overall-travel scenario** that
scales panel A by the NTS 2024 total-to-commute distance ratio (`6082/1007 = 6.04x`).
Values are reported per household and summarised by median within dominant housing type.

Interpretation.

- Transport energy materially widens the cost gradient associated with neighbourhood form.
- The combined cost of ordinary living (building + transport) is substantially higher in
  more sprawling typologies.
- This is a morphology-linked energy dependency pattern, not just a building-envelope effect.

Median OA values:

| Dwelling type | Transport (commute est.) kWh/hh | Transport (overall est.) kWh/hh | Total (commute base) kWh/hh | Total (overall est.) kWh/hh |
| ------------- | ------------------------------- | ------------------------------- | --------------------------- | --------------------------- |
| Flat          | 684                             | 4,131                           | 11,536                      | 14,984                      |
| Terraced      | 981                             | 5,923                           | 14,582                      | 19,524                      |
| Semi-detached | 1,123                           | 6,786                           | 15,751                      | 21,413                      |
| Detached      | 1,251                           | 7,556                           | 18,116                      | 24,421                      |

Under the commute-only estimate, total energy rises from **11,536 kWh/hh** (flat-dominant) to
**18,116 kWh/hh** (detached-dominant), a **57%** gap. Under the overall-travel scenario, the
same comparison is **14,984 → 24,421 kWh/hh**, a **63%** gap. Both are substantially larger
than the LSOA-level gaps (30% and 38% respectively), reflecting the reduced aggregation
bias at OA resolution.

> Note: The 6.04x overall-travel scaling factor is a single national ratio applied
> uniformly across all OAs. If sprawling areas make proportionally more non-commute car
> trips (shopping, school runs, leisure) than compact areas, this uniform scalar would
> understate the sprawl transport penalty. The true ratio likely varies by morphology type;
> the uniform scalar is a known simplification.

![Figure 2b: Estimated private vs public commute energy by housing type](../stats/figures/oa/fig2b_private_public_transport.png)

_Figure note (data and method)._ Private/public mode decomposition uses Census `TS061` mode
counts and Census `TS058` distance bands. Private modes are defined as driving, passenger in
car/van, taxi, and motorcycle; public modes as bus, train, and metro/tram. Energy intensities
from ECUK 2025: road passenger = 0.399 kWh/pkm; rail passenger = 0.178 kWh/pkm.

Interpretation.

- Private commute energy rises from 464 kWh/hh (flat) to 1,192 kWh/hh (detached) — a 2.57x
  gradient.
- Public commute energy moves in the opposite direction: 220 kWh/hh (flat) to 59 kWh/hh
  (detached) — flat-dominant OAs use 3.71x more public transport energy.
- The net effect is a clear morphology gradient driven by mode choice.

![Figure 3: Density and transport energy (KDE contours)](../stats/figures/oa/fig3_density_transport.png)

_Figure note (data and method)._ Population density is derived from Census 2021 population
and OA area. Transport energy is the same commute-energy estimate described above. Filled
KDE contours show the distribution of OAs by dominant housing type in the density-transport
space; contours are normalised within type for visual comparison.

Interpretation.

- Lower-density / more sprawling OAs cluster at higher transport energy.
- With 67,263 OAs, the type-group distributions are well separated.

## Step 3: Morphology and Access to Amenities (The Return Side)

![Figure 4: Local accessibility dividend (network frontage proxy)](../stats/figures/oa/fig4_accessibility_dividend.png)

_Figure note (data and method)._ This figure uses cityseer network metrics at an 800m
pedestrian catchment (roughly a 10-minute walk). The plotted accessibility component is
street-network frontage/density (`cc_density_800`) aggregated to OA via UPRN-linked node
assignments. KDE contours show the distribution by dominant housing type.

Interpretation.

- More compact neighbourhood forms are associated with greater local network opportunity.
- This is a structural precondition for local amenity access and lower travel dependence.

![Figure 5: Accessibility components by housing type](../stats/figures/oa/fig5_access_bar.png)

_Figure note (data and method)._ Panels show median OA-level cityseer accessibility metrics
by dominant housing type, using gravity-weighted (`_wt`) counts within 800m network walk
catchments. Destination layers include FSA categories, bus, rail, green space, schools, and
GP practices. Error bars show IQRs.

Interpretation.

- The access gradient is not confined to one category; it appears across multiple everyday
  destination types.
- More sprawling typologies have lower local access while also having higher transport
  energy — the core compounding mechanism.

## Step 4: Basket as a Trip-Demand Schedule

The basket defines annual trip demand by destination type, using observed national rates.

Trip-demand anchors:

| Basket category             | Annual demand anchor (trips per person) | Included in summed destination-trip budget | Source basis                                                              |
| --------------------------- | --------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------- |
| Local food/services (proxy) | 167.0                                   | Yes                                        | DfT NTS 2024 shopping trips                                               |
| GP                          | 6.6                                     | Yes                                        | NHS England GP appointments (Apr 2024-Mar 2025), per-person               |
| Pharmacy                    | 7.2                                     | Yes                                        | NHSBSA prescription items (2024/25), assuming 3 items per collection trip |
| School                      | 59.2                                    | Yes                                        | DfE pupil headcount (2024/25) x 2 school trips/day x 190 days, per-person |
| Green space                 | 85.0                                    | Yes                                        | DfT NTS 2024 "just walk" trips                                            |
| Hospital                    | 2.0                                     | Yes                                        | NHS Digital outpatient attended appointments (2024/25), per-person        |
| Bus                         | 41.0                                    | No                                         | DfT NTS 2024 local bus trips/person (mode access variable)                |
| Rail/metro                  | 21.0                                    | No                                         | DfT NTS 2024 surface rail trips/person (mode access variable)             |

Bus and rail are excluded from the summed destination-trip budget because they are
mode-access variables rather than destination trips.

## Step 5: Distance Allocation and Land-Use Penalty

For each basket category in each OA, nearest network distance to destination is used to
allocate annual trips. Nearest-distance metrics (`cc_*_nearest_max_4800`) report the
shortest network path from each UPRN-linked street node to the closest destination of
each type, capped at 4,800m.

- `local-access share`: distance-decay share `exp(-ln(2) * (d / d_half)^2)`, where `d` is nearest distance and `d_half` is a trip-type-specific half-distance
- `local-access trips`: `base trips × local-access share`
- `wider-access trips`: `base trips` when nearest distance `<= 4800m`, else `0`
- `additional travel-required trips`: `wider-access trips − local-access trips`

Half-distances (`d_half`):

- Food/services: 400m
- GP: 700m
- Pharmacy: 650m
- School: 900m
- Green space: 1,000m
- Hospital: 700m

These half-distances represent walk-willingness thresholds scaled by trip purpose.

![Figure 6: Basket local access presence by housing type](../stats/figures/basket_oa/fig_basket_oa_category_scores_heatmap.png)

_Figure note (data and method)._ Heatmap reports the share of OAs in each dominant housing
type with local access (at least one destination reachable within the 800m network
catchment), by basket category.

## Step 6: Access Delivery and Penalty by Morphology

![Figure 7: Local trip coverage and land-use access penalty by housing type](../stats/figures/basket_oa/fig_basket_oa_by_type.png)

_Figure note (data and method)._ Panel A shows mean local trip coverage by dominant housing
type. Panel B shows mean annual land-use access penalty under the overall-travel transport
scenario.

Median OA values (land-use access penalty shown under overall-travel scenario):

| Type          | Trip budget (trips/hh/yr) | Local-access trips (trips/hh/yr) | Additional travel-required trips (trips/hh/yr) | Local trip coverage | Land-use access penalty (kWh/hh/yr, overall-travel) |
| ------------- | ------------------------- | -------------------------------- | ---------------------------------------------- | ------------------- | --------------------------------------------------- |
| Flat          | 696                       | 535                              | 130                                            | 81.4%               | 291                                                 |
| Terraced      | 833                       | 617                              | 194                                            | 77.4%               | 482                                                 |
| Semi-detached | 826                       | 531                              | 295                                            | 64.6%               | 844                                                 |
| Detached      | 817                       | 383                              | 430                                            | 46.7%               | 1,352                                               |

Interpretation.

- Required trip budgets are similar across housing types.
- The difference is how much of that budget is delivered locally.
- Detached-dominant OAs require **3.31x** more additional travel-required trips than
  flat-dominant OAs.
- The associated land-use access penalty is **4.65x** higher in detached-dominant OAs.

## Step 7: Deprivation Control

![Figure 7b: Deprivation gradient of access burden](../stats/figures/basket_oa/fig_basket_oa_deprivation_gradient.png)

_Figure note (data and method)._ Panel A shows median land-use access penalty (kWh/hh/yr,
overall-travel scenario) and Panel B shows median local trip coverage (%) by dominant
housing type, each stratified by Census deprivation quintile (`TS011`). Each line traces
one housing type across deprivation quintiles (Q1 most deprived to Q5 least deprived).

Interpretation.

- The morphology gradient in access penalty persists within deprivation quintiles.
- Detached-dominant OAs carry a higher land-use access penalty than flat-dominant OAs
  at comparable levels of deprivation, which rules out the explanation that the pattern is
  driven solely by wealth differences or residential self-selection.
- This supports interpreting the access penalty as a structural property of neighbourhood
  morphology rather than a compositional artefact of who lives there.

## Step 8: Relationship Across All OAs

![Figure 8: Energy vs local trip coverage (KDE by housing type)](../stats/figures/basket_oa/fig_basket_oa_scatter_energy_vs_basket.png)

_Figure note (data and method)._ KDE contours show all OAs in the space defined by
`total_kwh_per_hh` (x-axis) and `local trip coverage` (y-axis), coloured by dominant housing
type. Dashed lines mark medians.

Interpretation.

- The morphology pattern is distribution-wide, not confined to type medians.
- Compact typologies cluster toward higher local trip coverage and lower total energy.
- More sprawling typologies shift toward lower local trip coverage and higher total energy.
- With 67,263 OAs, the distributions are more finely resolved than at LSOA level.

## Step 9: Aggregate Cost of Sprawl (94-BUA Sample)

This analysis aggregates the premium paid by sprawling morphologies across the full
94-BUA sample (67,263 OAs).

Method.

- Compact group: Flat + Terraced dominant OAs (39,401 OAs)
- Sprawl group: Semi + Detached dominant OAs (27,862 OAs)
- Weighting: total households (`total_hh`) in each OA
- Component metrics (commute-basis accounting): `building_kwh_per_hh`,
  `transport_kwh_per_hh_est`, `land_use_access_penalty_kwh_hh_commute_est`
- Combined = sum of the three components
- Sprawl premium per household: `sprawl mean - compact mean`
- Aggregate sprawl premium: `premium per household x sprawl households`

The compounding gradient:

| Surface | Flat (kWh/hh) | Detached (kWh/hh) | Ratio |
| ------- | ------------: | -----------------: | ----: |
| Building energy | 10,852 | 16,865 | 1.55x |
| Transport energy (overall) | 4,131 | 7,556 | 1.83x |
| kWh per unit access | 3,337 | 8,964 | **2.69x** |

Each successive surface amplifies the morphology penalty. The compounding is visible:
the sprawl/compact ratio rises from 1.55x (building alone, a thermal envelope effect)
through 1.83x (adding transport, a mobility dependency effect) to 2.69x (normalising by
access return, a structural land-use effect).

## What Technology Can and Cannot Offset

Current energy policy targets three interventions: envelope retrofit (insulation), heating
electrification (heat pumps), and transport electrification (EVs). Each addresses a
different layer of the morphology penalty. The question is which layers are structurally
locked in and which can be offset by technology turnover.

| Layer | Sprawl ratio | Policy intervention | Can it close the gap? | Replacement cycle |
| ----- | ------------ | ------------------- | --------------------- | ----------------- |
| Building energy (1.55x) | Heat pump (COP 2.4–3.2) | Yes — more than eliminates premium | 10–20 yrs |
| Transport energy (1.83x) | EV fleet electrification | Partially — reduces intensity, not distance | 10–15 yrs |
| Access penalty (2.69x) | None available | **No** — structural; set by distance to destinations | 50–100+ yrs |

The pattern is clear: the morphology penalty is largest on the layer that technology
cannot reach, and smallest on the layer most amenable to retrofit. No propulsion technology
changes the distance to the nearest school; no insulation standard brings the pharmacy
closer.

Morphology does not turn over on technology timescales. 38% of English housing stock
predates 1946 (BRE Trust, 2020); the UK demolition rate implies the average dwelling
would need to last over 1,000 years at current replacement rates (LGA, 2023). A
low-density development approved today commits to its access penalty for the foreseeable
future.

## Comparison with LSOA Analysis (Case One)

The OA-level analysis strengthens the findings from the earlier LSOA-level proof of concept
(18 cities, 3,678 LSOAs) in several ways:

| Dimension | LSOA (Case One) | OA (Case Two) |
| --------- | --------------- | ------------- |
| Sample | 3,678 LSOAs, 18 cities | 67,263 OAs, 94 BUAs |
| Unit size | ~700 households | ~130 households |
| Building energy gradient | 1.28x | **1.55x** |
| Transport gradient | 1.68x | **1.83x** |
| Access gradient | 2.29x | **2.69x** |
| Energy source | DESNZ LSOA-native | DESNZ postcode-aggregated to OA |
| Deprivation data | Census TS011 only | Census TS011 + IMD 2025 (via LSOA) |
| Vehicle data | Census TS045 only | Census TS045 + DVLA fleet (via LSOA) |
| Cityseer API | Legacy (4.23) | CityNetwork (4.25) |

The steeper gradients at OA level are expected: smaller units are more morphologically
homogeneous, so the dominant-type classification captures actual building form more
precisely. The LSOA-level analysis was understating the thermal surface gradient by ~20%
due to within-unit type mixing.

## Conclusion

The results indicate a consistent morphology-energy-access relationship across 94 English
Built-Up Areas at Output Area resolution:

- more sprawling neighbourhood typologies have higher total household energy,
- lower delivery of routine trips within local walkable catchments, and
- higher additional travel-required energy to deliver the same everyday access functions.

The key result is compounding: morphology increases both baseline household energy demand
and the travel energy needed to secure ordinary access. Because these effects move in the
same direction, marginal densification carries a double dividend — reducing energy cost
while simultaneously improving local access return.

This compounding is structural, not compositional. The deprivation control (Step 7)
demonstrates that the morphology gradient persists within deprivation quintiles,
which means it cannot be explained away as a wealth effect or residential sorting artefact.

The policy implication is direct: planning decisions that permit low-density peripheral
development lock in energy penalties that persist for the life of the housing stock.
These penalties are not confined to building energy — they compound through transport
dependence and reduced local access.

The national pipeline (7,147 BUAs) is in progress. When complete, the results can be
aggregated as an **excess energy cost of sprawl**: the additional energy used above a
compact-reference benchmark to deliver comparable basket access, summed across all English
OAs. This would make the cost of sprawl visible as a system-wide energy burden, not only
a household burden.

## Limitations

- Basket uses a **distance-decay walkability proxy** with trip-type half-distance
  assumptions, not a full 20-minute multimodal travel-time model. The half-distances
  are plausible but not empirically calibrated.
- The FSA food/services component is a proxy, not a clean essential-retail layer.
- Transport energy remains modelled from Census commute-energy (`TS058` + `TS061`) with an
  overall-travel scenario sensitivity. The 6.04x overall-travel scaling factor is a single
  national ratio; the true commute-to-total ratio likely varies by morphology type.
- Building energy at OA level is derived from postcode-level DESNZ data via spatial
  aggregation, not a native OA-level publication. Postcodes with fewer than 5 meters
  are suppressed in the source data; OAs with fewer than 5 total meters are excluded.
- IMD 2025 and DVLA vehicle data are LSOA-native, joined to OAs via a foreign key. All
  OAs within the same LSOA receive identical IMD/DVLA values, which limits within-LSOA
  variation in these control variables.
- OAs outside Built-Up Area boundaries (rural areas) are not included in this analysis.
  The national pipeline processes all 7,147 English BUAs but does not cover dispersed
  rural settlement.
- Key ratios are reported as group medians without confidence intervals. Adding bootstrap
  ranges to the headline ratios would strengthen the statistical basis.
