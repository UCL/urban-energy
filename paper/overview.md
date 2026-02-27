# Research Overview

## The Argument

Urban energy policy usually measures the cost side of the ledger: kilowatt-hours used in
buildings, fuels consumed in transport, and the efficiency of individual technologies. But
neighbourhoods do not only differ in how much energy they consume. They differ in the
energy required to secure ordinary urban life: housing, mobility, and access to everyday
services. That requirement is not neutral. It is shaped by urban morphology.

This project argues that housing typology and neighbourhood morphology lock in distinct
energy dependency patterns. Dwelling form affects building energy demand (through exposed
envelope area, household structure, and stock composition), while street layout, density,
and land-use mix affect transport behaviour and the feasibility of local access. The result
is a morphology-conditioned energy cost of access: some neighbourhoods require
substantially more energy to achieve everyday access, while others deliver comparable local
access at much lower energy cost.

The aim is to build an LSOA-level index for England that combines:

1. typical housing energy associated with local dwelling typology,
2. transport energy associated with local morphology and travel behaviour, and
3. attainment of a 20-minute city amenity basket grounded in UK accessibility literature.

This document is a proof of concept currently tested across 18 English cities (3,678
LSOAs), with planned expansion to all of England. The current implementation already shows
the core pattern: differences in dwelling type and neighbourhood form produce a much larger
gap in energy per unit of local access than in building energy alone.

## What This Proof of Concept Measures (Current State)

The proof of concept is already an energy-cost-of-access framework, but it is not yet the
final 20-minute city basket index.

### Cost Side (Current)

Per LSOA, the current cost metric is:

- `Building energy (kWh/hh)`: DESNZ metered domestic gas + electricity (2023), native LSOA
- `Transport energy (kWh/hh)`: **commute-energy estimate** from Census 2021 commute distance
  (TS058) and mode counts (TS061), using mode-specific passenger-energy intensities
- `Transport energy (overall scenario)`: commute estimate scaled by NTS 2024 total-to-commute
  distance ratio (`6082/1007 = 6.04x`) for a secondary full-travel approximation
- `Total energy (kWh/hh) = building + transport`

### Return Side (Current Proxy)

The current return metric is a proxy for local access, not yet a formal 20-minute basket:

- `Street frontage`: cityseer `cc_density_800` (reachable street network nodes within 800m)
- `FSA access`: sum of gravity-weighted FSA destination layers within 800m walk catchment
- `Accessibility (proxy)`: z-scored `street frontage + FSA`

This is then converted into an energy ratio:

- `access_per_kwh` (higher = better)
- `kwh_per_access` (lower = better)

Important: because the current accessibility score is a shifted z-score composite, the ratio
is useful for comparison across LSOAs but not yet a final interpretable "basket unit." That
formal unit comes in the next phase.

## Pilot Dataset and Scope

All analysis is at LSOA level because metered domestic energy (DESNZ) is only available at
LSOA. Other inputs are aggregated upward from finer scales (OA, UPRN, street node, point
features) to match.

- Coverage: 18 English cities, 3,678 LSOAs
- Stratification shown here: dominant accommodation type (Census 2021 TS044)
- Accessibility catchment in current PoC: 800m network distance (about a 10-minute walk)

Data and derivation details are documented in `paper/data.md`.

## Observations (Current PoC Results)

All values below are medians across LSOAs, stratified by dominant accommodation type (Census
2021 TS044). These are ecological comparisons at area level, not property-level estimates.

### Surface 1: Building Energy (Cost)

![Figure 1: Building Energy](../stats/figures/fig1_building_energy.png)

| Dwelling type | Building energy (kWh/household) | Building energy (kWh/person) |
| ------------- | ------------------------------- | ---------------------------- |
| Flat          | 13,034                          | 6,208                        |
| Terraced      | 13,342                          | 5,275                        |
| Semi-detached | 14,203                          | 5,807                        |
| Detached      | 16,632                          | 6,817                        |

At the household level, detached-dominated LSOAs use 28% more building energy than
flat-dominated LSOAs. Per capita, the contrast narrows because flat-dominated LSOAs have
smaller households (median 2.08 persons/hh versus 2.43 in detached-dominated LSOAs). The
building-energy gradient is real, but on its own it does not capture the full morphology
penalty.

### Surface 2: Transport Energy (Cost)

![Figure 2: Mobility Penalty](../stats/figures/fig2_mobility_penalty.png)

| Dwelling type | Transport energy (kWh/household) | Car commute share | Cars per household |
| ------------- | -------------------------------- | ----------------- | ------------------ |
| Flat          | 2,286                            | 29%               | 0.68               |
| Terraced      | 3,380                            | 40%               | 0.89               |
| Semi-detached | 4,072                            | 48%               | 1.13               |
| Detached      | 4,489                            | 45%               | 1.47               |

Transport sharply widens the gap. Detached-dominated LSOAs incur about twice the transport
energy of flat-dominated LSOAs. Total household energy (building + transport) rises from
15,764 kWh/hh (flat) to 21,416 kWh/hh (detached), a 36% gap. Car ownership and car commute
share follow the same compact-to-sprawl gradient, consistent with morphology-shaped travel
dependence.

![Figure 3: Density and Transport](../stats/figures/fig3_density_transport.png)

### Surface 3: Local Access Proxy (Return)

![Figure 4: Accessibility](../stats/figures/fig4_accessibility_dividend.png)

Surfaces 1 and 2 measure cost. Surface 3 measures return: what local, walkable access the
neighbourhood structure provides within an 800m network catchment.

The table below reports current PoC accessibility components. Values for destination layers
are gravity-weighted accessibility scores (`_wt`), not literal destination counts.

| Dwelling type | Street frontage | Restaurants (wt) | Pubs (wt) | Bus stops (wt) | Green space (wt) | GPs (wt) |
| ------------- | --------------- | ---------------- | --------- | -------------- | ---------------- | -------- |
| Flat          | 128.1           | 1.61             | 0.40      | 2.42           | 0.54             | 0.11     |
| Terraced      | 115.7           | 0.76             | 0.18      | 2.06           | 0.54             | 0.10     |
| Semi-detached | 89.5            | 0.28             | 0.08      | 1.96           | 0.36             | 0.03     |
| Detached      | 63.9            | 0.12             | 0.03      | 1.31           | 0.29             | 0.00     |

The same morphological gradient appears on the return side. Compact forms have denser
street networks and substantially higher walkable access to everyday destinations. In other
words, morphology shapes not only energy demand, but what that energy can substitute for.

![Figure 5: Accessibility by Type](../stats/figures/fig5_access_bar.png)

### The Compounding: Energy Cost of Access

The core result is the combination of these surfaces. Detached-dominated LSOAs tend to
spend more energy and receive lower local access in return. The cost is higher and the
return is lower, so the gap compounds.

| Dwelling type | Total energy (kWh/hh) | Accessibility proxy (z-sum) | kWh / access (proxy) | Access / kWh (proxy) |
| ------------- | --------------------- | --------------------------- | -------------------- | -------------------- |
| Flat          | 15,764                | 0.37                        | 4,317                | 0.000232             |
| Terraced      | 16,841                | 0.01                        | 5,315                | 0.000188             |
| Semi-detached | 18,557                | -0.76                       | 7,377                | 0.000136             |
| Detached      | 21,416                | -1.24                       | 10,616               | 0.000094             |

On the current proxy measure, flat-dominated LSOAs deliver about 2.46x more local access
per unit of household energy than detached-dominated LSOAs. This is the core proof-of-
concept result: dwelling type and neighbourhood morphology move both sides of the fraction
in opposite directions.

## What Changes in the Full 20-Minute City Basket Index

The current PoC demonstrates the morphology-and-energy mechanism. The next step is to
replace the proxy return metric with a formal 20-minute city basket score.

### Planned Change

Replace:

- shifted z-score accessibility proxy (`street frontage + FSA`)

With:

- a bounded, interpretable basket attainment score based on UK-relevant everyday functions
  (for example: grocery, GP, pharmacy, school, green space, frequent bus/rail access, and
  selected daily retail/services), measured within a 20-minute accessibility definition

### Why This Matters

This will convert the current comparative proxy into a clearer policy metric:

- `Energy cost of access` (`kWh per basket unit`)
- `Energy productivity of urban form` (`basket units per kWh`)

That makes the morphology lock-in argument easier to communicate outside research settings:
some neighbourhoods structurally require more energy to secure ordinary daily access.

## Interpretation and Caveats (Current PoC)

- The transport energy estimate is modelled from Census commute behaviour and scaled to total
  car travel; it is not directly metered transport energy.
- Census 2021 commute patterns were affected by residual COVID-era home working, which may
  understate sprawl-associated transport energy.
- The current return metric uses an 800m (roughly 10-minute walk) catchment, not a full
  20-minute multimodal basket framework.
- Household-level energy (`kWh/hh`) is used for the main cost metric; per-person values are
  reported for interpretation but not yet used as the primary denominator in the access
  ratio.
- Results shown here are area-level (LSOA) medians by dominant housing type and should be
  interpreted as structural patterns, not causal estimates for individual households.

## Working Thesis for the Project

Neighbourhood morphology determines the energy cost of ordinary access. Energy policy should
therefore evaluate not only efficiency (how much energy is used), but also the energy
productivity of urban form (how much everyday access that energy secures).
