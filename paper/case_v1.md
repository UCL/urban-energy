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

The analysis covers an 18-city English sample and an LSOA-level index formulation.

- Geography: 18 English cities
- Unit: LSOA (3,678 LSOAs)
- Sample by dominant type: Flat 712, Terraced 1,209, Semi-detached 1,473, Detached 284
- Core typology stratification: dominant Census 2021 accommodation type (TS044)
- Current access model: trip-type-specific distance-decay local-access model (nearest network distance), with wider-access bound at 4,800m
- Current energy model: metered building energy + modelled commute energy (private/public decomposition from Census mode and distance)

LSOA is used because DESNZ metered domestic energy is published at that scale and sets the
binding resolution.

All results are area-level (ecological) associations between LSOA-level morphology and
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
8. The **relationship across all LSOAs** confirms the pattern is distribution-wide.
9. An **aggregate cost of sprawl** is estimated for the 18-city sample.

Dominant housing type is used as a practical LSOA-level proxy for broader morphology,
supported by density and network-access evidence.

### Aggregation conventions

Three levels of aggregation are used throughout. To avoid ambiguity, each is stated here
and flagged in figure notes where it applies:

1. **Node → UPRN → LSOA.** Cityseer accessibility metrics are computed at individual
   street-network nodes. Each UPRN (address point) is assigned to its nearest node, and the
   LSOA value is the **mean across all UPRNs** within the LSOA. The UPRN intermediate step
   is intentional: it acts as a dwelling-weighted spatial aggregation, so that a street
   segment with 50 addresses contributes 50 times while one with no dwellings contributes
   zero. This ensures the LSOA value reflects the access experienced by residents, not the
   unweighted average across all street segments (which would be diluted by roads through
   parks, industrial areas, or other non-residential land). The same aggregation applies to
   nearest-distance metrics (`cc_*_nearest_max_4800`).

2. **LSOA → dominant-type group.** Each LSOA is assigned to a dominant housing type by
   **plurality share** of Census TS044 accommodation types (the single type with the largest
   share, even if below 50%). Where bar charts or summary tables aggregate across LSOAs
   within a type group, the statistic used is stated explicitly:
   - Steps 1–3 and Step 5 use **medians** (robust to skew in energy and distance
     distributions).
   - Steps 6–7 use **means** (required for trip budgets and penalties that must sum to
     interpretable totals).

3. **Type-group → compact/sprawl.** Step 9 groups Flat + Terraced as "compact" and
   Semi + Detached as "sprawl," then reports **household-weighted means** (each LSOA
   contributes in proportion to its household count) to produce aggregate energy figures
   that reflect the population-weighted burden.

KDE contour plots (Figures 3, 4, 8) show the **full distribution** of individual LSOAs
without further aggregation; contours are normalised within each type group for visual
comparability.

## Step 1: Building Typology and Building Energy

![Figure 1: Building energy by dominant housing type](../stats/figures/fig1_building_energy.png)

_Figure note (data and method)._ Building energy uses DESNZ sub-national domestic energy
statistics (2023), combining metered gas (weather-corrected) and electricity at LSOA level.
Each LSOA is grouped by dominant Census 2021 accommodation type (`TS044`, plurality share).
Bars show median LSOA values by dominant type; panel B stratifies by deprivation tercile
using Census `TS011`.

Interpretation.

- Building energy is higher in detached-dominant LSOAs than in flat-dominant LSOAs.
- The building-energy gradient is clear, but does not by itself explain the full structural
  gap between neighbourhood forms.
- Per-person and per-household readings differ because household size varies by typology:
  average household size rises from approximately 2.1 persons in flat-dominant LSOAs to 2.4
  in detached-dominant LSOAs, which dilutes per-capita building energy in larger households
  and reorders the per-person ranking (terraced lowest at 5,275 kWh/person).

Median LSOA values:

| Dwelling type | Building energy (kWh/household) | Building energy (kWh/person) |
| ------------- | ------------------------------- | ---------------------------- |
| Flat          | 13,034                          | 6,208                        |
| Terraced      | 13,342                          | 5,275                        |
| Semi-detached | 14,203                          | 5,807                        |
| Detached      | 16,632                          | 6,817                        |

> Note: An earlier approach attempted to derive building-physics variables (envelope
> area, party-wall sharing) at UPRN level from EPC records and LiDAR. This was abandoned
> because of partial EPC coverage (biased against long-term owner-occupied stock),
> domestic/non-domestic classification ambiguity in mixed-use buildings, and the SAP
> performance gap. DESNZ metered consumption at LSOA level avoids these issues at the
> cost of coarser spatial resolution.

## Step 2: Morphology and Transport Energy (The Mobility Penalty)

![Figure 2: Building + transport energy by dominant housing type](../stats/figures/fig2_mobility_penalty.png)

_Figure note (data and method)._ Figure 2 now shows two panels using the same type-level
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

Median LSOA values:

| Dwelling type | Transport (commute est.) kWh/hh | Transport (overall est.) kWh/hh | Total (commute base) kWh/hh | Total (overall est.) kWh/hh |
| ------------- | ------------------------------- | ------------------------------- | --------------------------- | --------------------------- |
| Flat          | 739                             | 4,464                           | 13,774                      | 17,499                      |
| Terraced      | 959                             | 5,792                           | 14,301                      | 19,134                      |
| Semi-detached | 1,099                           | 6,635                           | 15,302                      | 20,838                      |
| Detached      | 1,243                           | 7,510                           | 17,875                      | 24,141                      |

Under the commute-only estimate, total energy rises from **13,774 kWh/hh** (flat-dominant) to
**17,875 kWh/hh** (detached-dominant), a **30%** gap. Under the overall-travel scenario, the
same comparison is **17,499 → 24,141 kWh/hh**, a **38%** gap.

> Note: The 6.04x overall-travel scaling factor is a single national ratio applied
> uniformly across all LSOAs. If sprawling areas make proportionally more non-commute car
> trips (shopping, school runs, leisure) than compact areas, this uniform scalar would
> understate the sprawl transport penalty. Conversely, compact-area residents may substitute
> more non-commute trips on foot, which would not appear in the commute data. The true
> ratio likely varies by morphology type; the uniform scalar is a known simplification.

![Figure 2b: Estimated private vs public commute energy by housing type](../stats/figures/fig2b_private_public_transport.png)

_Figure note (data and method)._ Private/public mode decomposition uses Census `TS061` mode
counts and Census `TS058` distance bands. Private modes are defined as driving, passenger in
car/van, taxi, and motorcycle; public modes as bus, train, and metro/tram. Commute distance
is estimated from `TS058` travelling bands (excluding work-from-home and offshore/no-fixed-
place categories), annualised as return trips over 220 workdays, and converted using national
passenger-energy intensities from ECUK 2025: road passenger = 34.3 ktoe/billion passenger-km
and rail passenger = 15.3 ktoe/billion passenger-km (0.399 and 0.178 kWh/pkm respectively).

Interpretation.

- The private/public decomposition preserves the morphology gradient on transport demand.
- Private commute energy is substantially higher than public commute energy across all housing
  types, with the largest private burden in semi- and detached-dominant LSOAs.
- This decomposition is an estimated commute-energy split and complements the main
  building+transport totals.

![Figure 3: Density and transport energy (KDE contours)](../stats/figures/fig3_density_transport.png)

_Figure note (data and method)._ Population density is derived from Census 2021 population
and OA area aggregated to LSOA. Transport energy is the same **commute-energy estimate**
described above. Filled KDE contours show the distribution of LSOAs by dominant housing type
in the density-transport space; contours are normalised within type for visual comparison.

Interpretation.

- The transport penalty is associated with neighbourhood structure, though residential
  sorting (household self-selection into neighbourhood types) cannot be ruled out at this
  stage; the deprivation control in Step 7 addresses this partially.
- Lower-density / more sprawling LSOAs cluster at higher transport energy.
- This supports the morphology mechanism behind the combined energy gradient.

## Step 3: Morphology and Access to Amenities (The Return Side)

![Figure 4: Local accessibility dividend (network frontage proxy)](../stats/figures/fig4_accessibility_dividend.png)

_Figure note (data and method)._ This figure uses cityseer network metrics at an 800m
pedestrian catchment (roughly a 10-minute walk). The plotted accessibility component is
street-network frontage/density (`cc_density_800`) aggregated to LSOA via UPRN-linked node
assignments. KDE contours show the distribution by dominant housing type.

Interpretation.

- More compact neighbourhood forms are associated with greater local network opportunity.
- This is a structural precondition for local amenity access and lower travel dependence.

![Figure 5: Accessibility components by housing type](../stats/figures/fig5_access_bar.png)

_Figure note (data and method)._ Panels show median LSOA-level cityseer accessibility metrics
by dominant housing type, using gravity-weighted (`_wt`) counts within 800m network walk
catchments. Destination layers include FSA categories, bus, rail, green space, schools, and
GP practices. Error bars show IQRs.

Interpretation.

- The access gradient is not confined to one category; it appears across multiple everyday
  destination types.
- More sprawling typologies tend to have lower local access while also having higher
  transport energy, which is the core compounding mechanism.

## Step 4: Basket as a Trip-Demand Schedule

The basket is retained, but used only to define annual trip demand by destination type
rather than as an attainment score.

Trip-demand anchors used in Basket v1:

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
mode-access variables rather than destination trips: a bus stop or rail station is an
enabler of wider access, not itself a trip endpoint. Their presence within reach is used
as an accessibility indicator in Step 3 but does not generate a separate trip demand.

## Step 5: Distance Allocation and Land-Use Penalty

For each basket category in each LSOA, nearest network distance to destination is used to
allocate annual trips. Nearest-distance metrics (`cc_*_nearest_max_4800`) report the
shortest network path from each UPRN-linked street node to the closest destination of
each type, capped at 4,800m.

- `local-access share`: distance-decay share `exp(-ln(2) * (d / d_half)^2)`, where `d` is nearest distance and `d_half` is a trip-type-specific half-distance (the distance at which the local-access share falls to 50%)
- `local-access trips`: `base trips × local-access share`
- `wider-access trips`: `base trips` when nearest distance `<= 4800m`, else `0`
- `additional travel-required trips`: `wider-access trips − local-access trips`

Half-distances (`d_half`) used in this run:

- Food/services: 400m
- GP: 700m
- Pharmacy: 650m
- School: 900m
- Green space: 1,000m
- Hospital: 700m

These half-distances represent walk-willingness thresholds scaled by trip purpose: shorter
for high-frequency convenience trips (food/services), longer for less frequent or
child-accompanied trips (school, green space). They are explicit assumptions, not
empirically calibrated parameters; sensitivity to their values is noted in Limitations.

Land-use access energy penalty is then:

- `additional travel-required trips × LSOA trip-energy intensity (kWh/trip)`

Trip-energy intensity uses the same transport model as Step 2 (Census `TS058` + `TS061`,
with the overall-travel scaling sensitivity).

![Figure 6: Basket v1 local access presence by housing type](../stats/figures/basket_v1/fig_basket_v1_category_scores_heatmap.png)

_Figure note (data and method)._ Heatmap reports the share of LSOAs in each dominant housing
type with local access (at least one destination reachable within the 800m network
catchment), by basket category. FSA nearest distance is the minimum of
restaurant/pub/takeaway/other FSA nearest layers.

## Step 6: Access Delivery and Penalty by Morphology

![Figure 7: Local trip coverage and land-use access penalty by housing type](../stats/figures/basket_v1/fig_basket_v1_by_type.png)

_Figure note (data and method)._ Panel A shows mean local trip coverage
(`local-access trips / total basket trips`) by dominant housing type. Panel B shows mean
annual land-use access penalty (`additional travel-required trips × trip-energy intensity`)
under the overall-travel transport scenario. Local-access trips use the trip-type
distance-decay specification in Step 5 rather than a binary 800m rule.

Mean LSOA values (land-use access penalty shown under overall-travel scenario):

| Type          | Trip budget (trips/hh/yr) | Local-access trips (trips/hh/yr) | Additional travel-required trips (trips/hh/yr) | Local trip coverage | Land-use access penalty (kWh/hh/yr, overall-travel) |
| ------------- | ------------------------- | -------------------------------- | ---------------------------------------------- | ------------------- | --------------------------------------------------- |
| Flat          | 691                       | 515                              | 176                                            | 75.2%               | 460                                                 |
| Terraced      | 845                       | 625                              | 220                                            | 73.3%               | 611                                                 |
| Semi-detached | 822                       | 505                              | 317                                            | 61.1%               | 953                                                 |
| Detached      | 811                       | 367                              | 445                                            | 45.1%               | 1,611                                               |

Interpretation.

- Required trip budgets are similar in magnitude across housing types.
- The difference is how much of that budget is delivered locally under trip-type walking-distance decay.
- Detached-dominant LSOAs require **2.53x** more additional travel-required trips than
  flat-dominant LSOAs.
- The associated land-use access penalty is **3.50x** higher in detached-dominant LSOAs.
- Trip budgets vary by type because the basket defines trips per person; conversion to
  per-household uses LSOA-level household size, which differs across dominant types.

## Step 7: Deprivation Control

![Figure 7b: Deprivation gradient of access burden](../stats/figures/basket_v1/fig_basket_v1_deprivation_gradient.png)

_Figure note (data and method)._ Panel A shows median land-use access penalty (kWh/hh/yr,
overall-travel scenario) and Panel B shows median local trip coverage (%) by dominant
housing type, each stratified by Census deprivation quintile (`TS011`). Each line traces
one housing type across deprivation quintiles (Q1 most deprived to Q5 least deprived).

Interpretation.

- The morphology gradient in access penalty persists within deprivation quintiles.
- Detached-dominant LSOAs carry a higher land-use access penalty than flat-dominant LSOAs
  at comparable levels of deprivation, which rules out the explanation that the pattern is
  driven solely by wealth differences or residential self-selection.
- This supports interpreting the access penalty as a structural property of neighbourhood
  morphology rather than a compositional artefact of who lives there.

## Step 8: Relationship Across All LSOAs

![Figure 8: Energy vs local trip coverage (KDE by housing type)](../stats/figures/basket_v1/fig_basket_v1_scatter_energy_vs_basket.png)

_Figure note (data and method)._ KDE contours show all pilot LSOAs in the space defined by
`total_kwh_per_hh` (x-axis) and `local trip coverage` (y-axis), coloured by dominant housing
type. Dashed lines mark pilot medians.

Interpretation.

- The morphology pattern is distribution-wide, not confined to type medians.
- Compact typologies cluster toward higher local trip coverage and lower total energy.
- More sprawling typologies shift toward lower local trip coverage and higher total energy.

## Step 9: Aggregate Cost of Sprawl (18-City PoC)

This proof-of-concept aggregates the premium paid by sprawling morphologies across the full
18-city sample.

Method.

- Compact group: Flat + Terraced dominant LSOAs
- Sprawl group: Semi + Detached dominant LSOAs
- Weighting: total households (`total_hh`) in each LSOA
- Component metrics (commute-basis accounting): `building_kwh_per_hh`,
  `transport_kwh_per_hh_est`, `land_use_access_penalty_kwh_hh_commute_est`
- Combined = sum of the three components
- Note: Step 9 uses commute-basis trip-energy intensity for conservative aggregate
  accounting; Steps 6–7 report the overall-travel scenario for illustration of the
  penalty gradient. The morphology ratios are similar under both scenarios.
- Sprawl premium per household: `sprawl mean - compact mean`
- Aggregate sprawl premium: `premium per household x sprawl households`

Household-weighted results (18 cities):

| Category                                | Compact (kWh/hh/yr) | Sprawl (kWh/hh/yr) | Premium (kWh/hh/yr) | Sprawl / Compact | Aggregate premium over sprawl households (TWh/yr) |
| --------------------------------------- | ------------------- | ------------------ | ------------------- | ---------------- | ------------------------------------------------- |
| Housing (building)                      | 13,833              | 15,062             | 1,228               | 1.09x            | 1.36                                              |
| Transport (commute estimate)            | 909                 | 1,136              | 226                 | 1.25x            | 0.25                                              |
| Land-use access penalty (commute basis) | 735                 | 1,411              | 675                 | 1.92x            | 0.75                                              |
| Combined commute-accounting burden      | 15,478              | 17,608             | 2,130               | 1.14x            | 2.36                                              |

Interpretation.

- The compounding gradient is visible in the ratio column: the sprawl/compact ratio rises
  from 1.09x (building alone) to 1.25x (transport) to 1.92x (land-use access penalty).
  Each successive layer amplifies the morphology penalty.
- The blended combined ratio (1.14x) is modest because building energy dominates the total
  and has the smallest sprawl premium. The individual component ratios better reveal the
  compounding structure.
- The aggregate premium of 2.36 TWh/yr across 18 cities is equivalent to the total annual
  domestic energy consumption of approximately 140,000 households at the national median
  (~17,000 kWh/hh). Scaled to England's full housing stock, the implied national sprawl
  premium would be substantially larger.

## What Technology Can and Cannot Offset

Current energy policy targets three interventions: envelope retrofit (insulation), heating
electrification (heat pumps), and transport electrification (EVs). Each addresses a
different layer of the morphology penalty. The question is which layers are structurally
locked in and which can be offset by technology turnover.

| Layer                    | Sprawl premium  | Policy intervention      | Typical saving                                                                                                            | Can it close the gap?                                     | Replacement cycle |
| ------------------------ | --------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- | ----------------- |
| Building energy (1.09x)  | 1,228 kWh/hh/yr | Cavity wall insulation   | ~2,000 kWh/hh (engineering est.; real-world rebound erodes to single digits within years; Chitnis et al., 2014)           | Partially                                                 | 10–20 yrs         |
|                          |                 | Heat pump (COP 2.4–3.2)  | ~6,000–7,500 kWh/hh (replaces 11,000 kWh gas with 3,400–4,600 kWh electricity; DESNZ Electrification of Heat trial, 2022) | Yes — more than eliminates premium                        | 10–20 yrs         |
| Transport energy (1.25x) | 226 kWh/hh/yr   | EV fleet electrification | ~3x efficiency gain (0.55 → 0.17 kWh/km); premium compresses by ~⅔                                                        | Partially — reduces intensity, not distance or trip count | 10–15 yrs         |
| Access penalty (1.92x)   | 675 kWh/hh/yr   | None available           | n/a                                                                                                                       | **No** — structural; set by distance to destinations      | 50–100+ yrs       |

The pattern is clear: the morphology penalty is largest (1.92x) on the layer that
technology cannot reach, and smallest (1.09x) on the layer most amenable to retrofit.
Current energy strategy (insulate, electrify) addresses the readily fixable layers while
the structural access penalty — the additional trips generated because the GP, school,
shop, or green space is beyond walking distance — remains locked in by land-use
configuration and street layout.

Morphology does not turn over on technology timescales. 38% of English housing stock
predates 1946 (BRE Trust, 2020); the UK demolition rate implies the average dwelling
would need to last over 1,000 years at current replacement rates (LGA, 2023). A
low-density development approved today commits to its access penalty for the foreseeable
future. No propulsion technology changes the distance to the nearest school; no insulation
standard brings the pharmacy closer.

## Conclusion

The results indicate a consistent morphology-energy-access relationship:

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

This framing also provides a route to national accounting. If an England-wide version of the
index is estimated, the results can be aggregated as an **excess energy cost of sprawl**:
the additional energy used above a compact-reference benchmark to deliver comparable basket
access, summed across LSOAs. This would make the cost of sprawl visible as a system-wide
energy burden, not only a household burden.

## Limitations

- Basket v1 uses a **distance-decay walkability proxy** with trip-type half-distance
  assumptions, not a full 20-minute multimodal travel-time model. The half-distances
  (`d_half`) are plausible but not empirically calibrated; a sensitivity analysis
  varying all half-distances by ±25% would strengthen confidence in the access penalty
  magnitudes reported in Steps 6 and 9.
- The FSA food/services component is a proxy, not a clean essential-retail layer.
- Trip-demand conversion uses best-available annual counts with explicit assumptions
  (notably pharmacy collection conversion and pupil-trip conversion). The pharmacy
  anchor (3 items per collection trip) is the weakest conversion assumption and may
  warrant a ±50% sensitivity check.
- Transport energy remains modelled from Census commute-energy (`TS058` + `TS061`) with an
  overall-travel scenario sensitivity, not direct observed full-purpose LSOA travel energy.
  The 6.04x overall-travel scaling factor is a single national ratio; the true
  commute-to-total ratio likely varies by morphology type.
- Dominant housing type is assigned by plurality share of Census TS044 within each LSOA.
  Some LSOAs may have narrow plurality margins (e.g., 40% semi, 35% terraced). The
  sensitivity of results to LSOAs where the dominant type exceeds a stricter threshold
  (e.g., >50% share) has not yet been tested.
- Key ratios (1.09x, 1.25x, 1.92x) are reported as group medians or means without
  confidence intervals or bootstrap ranges. Adding IQRs or permutation-based uncertainty
  estimates to the headline ratios would strengthen the statistical basis.
