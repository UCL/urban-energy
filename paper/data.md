# Data Strategy

## Spatial Scale

All analysis operates at **Lower Super Output Area (LSOA)** level — approximately 1,500
residents and 650 households per unit, with ~33,000 LSOAs covering England.

The LSOA is not the natural grain of most variables used here. Census data is collected at
Output Area (OA, ~125 households, ~180,000 units in England); street-network accessibility
is computed at individual street nodes; EPC records are address-level (UPRN); FSA and
NaPTAN data are point features. All of these have finer native resolution than LSOA.

**The binding constraint is metered domestic energy consumption.** DESNZ publishes gas and
electricity consumption only at LSOA — it is not disaggregated to OA and cannot be
reconstructed from available data. Because building energy is the primary dependent variable
of the analysis, every other data source is aggregated upward to match it. The analysis
scale is LSOA by necessity, not by preference.

A secondary constraint applies to the deprivation robustness check. Stratifying
simultaneously by deprivation quintile and housing type requires cross-tabulation at LSOA+
to maintain stable cell sizes; OA cells would be too small for reliable group means.

---

## Data Sources

The table below lists every variable used in the analysis: what it measures, its source,
its native spatial scale, and how it is brought to LSOA.

| Variable                                                | What it measures                                                                       | Source                                                                                       | Native scale          | Derivation to LSOA                                                                                                                                                                                                                                            |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Building energy** (kWh/household)                     | Mean total domestic energy consumption — metered gas (weather-corrected) + electricity | DESNZ Sub-national Energy Statistics, consumption year 2023 (2010–2024 release, Dec 2025)    | LSOA (native)         | Direct join on LSOA21CD                                                                                                                                                                                                                                       |
| **Transport energy** (kWh/household)                    | Estimated total car travel energy per household                                        | Derived from Census 2021 TS058 + TS061 + NTS scaling                                         | OA (Census inputs)    | OA commuter counts summed to LSOA; transport energy computed at LSOA from derived commute-km — see note                                                                                                                                                       |
| **Average commute distance** (km)                       | Weighted mean one-way commute distance                                                 | Census 2021, TS058                                                                           | OA                    | Each distance band assigned a midpoint km; weighted mean computed from OA counts summed to LSOA                                                                                                                                                               |
| **Car commute share** (%)                               | Proportion of employed residents commuting by car or van                               | Census 2021, TS061                                                                           | OA                    | Car commuter count / total employed commuters; OA counts summed to LSOA before division                                                                                                                                                                       |
| **Walk share** (%)                                      | Proportion commuting on foot                                                           | Census 2021, TS061                                                                           | OA                    | As above                                                                                                                                                                                                                                                      |
| **Cycle share** (%)                                     | Proportion commuting by bicycle                                                        | Census 2021, TS061                                                                           | OA                    | As above                                                                                                                                                                                                                                                      |
| **Cars per household**                                  | Mean vehicles available per household                                                  | Census 2021, TS045                                                                           | OA                    | Weighted mean (0 × none + 1 × one + 2 × two + 3 × three-plus) / total households; OA counts summed to LSOA                                                                                                                                                    |
| **Housing type — % detached**                           | Share of dwellings that are detached houses                                            | Census 2021, TS044                                                                           | OA                    | Type counts summed across OAs within LSOA; proportion computed at LSOA                                                                                                                                                                                        |
| **Housing type — % semi-detached**                      | Share of dwellings that are semi-detached houses                                       | Census 2021, TS044                                                                           | OA                    | As above                                                                                                                                                                                                                                                      |
| **Housing type — % terraced**                           | Share of dwellings that are terraced houses                                            | Census 2021, TS044                                                                           | OA                    | As above                                                                                                                                                                                                                                                      |
| **Housing type — % flat**                               | Share of dwellings in purpose-built flat blocks                                        | Census 2021, TS044                                                                           | OA                    | As above                                                                                                                                                                                                                                                      |
| **Dominant housing type**                               | Mode dwelling type for LSOA stratification (Flat / Terraced / Semi / Detached)         | Census 2021, TS044                                                                           | OA → LSOA             | Plurality type from LSOA-level type proportions                                                                                                                                                                                                               |
| **Population density** (persons/km²)                    | Residential population density                                                         | Census 2021, TS006 + TS001                                                                   | OA                    | OA area back-calculated from population / density; areas summed to LSOA; LSOA density = LSOA population / LSOA area                                                                                                                                           |
| **Total population**                                    | Usual residents living in households                                                   | Census 2021, TS001                                                                           | OA                    | OA counts summed to LSOA                                                                                                                                                                                                                                      |
| **Household count**                                     | Occupied household spaces                                                              | Census 2021, TS017                                                                           | OA                    | Total household spaces minus zero-person spaces; OA counts summed to LSOA                                                                                                                                                                                     |
| **Deprivation** (% not deprived)                        | Proportion of households with no deprivation dimensions                                | Census 2021, TS011                                                                           | OA                    | OA counts summed to LSOA; proportion computed at LSOA                                                                                                                                                                                                         |
| **Street network density** (cc_density_800)             | Count of reachable street network nodes within 800m network distance                   | OS Open Roads + cityseer                                                                     | Street node           | Computed per node by cityseer; each UPRN assigned to nearest node; mean across all UPRNs within LSOA                                                                                                                                                          |
| **Street network harmonic closeness** (cc_harmonic_800) | Gravity-weighted inverse-distance sum to all reachable nodes within 800m               | OS Open Roads + cityseer                                                                     | Street node           | As above                                                                                                                                                                                                                                                      |
| **Restaurant / café access** (cc_fsa_restaurant_800_wt) | Gravity-weighted count of restaurants and cafés within 800m walk                       | FSA Food Hygiene Register + cityseer                                                         | Point (establishment) | Gravity-weighted accessibility computed per node; mean across UPRNs within LSOA                                                                                                                                                                               |
| **Pub / bar access** (cc_fsa_pub_800_wt)                | Gravity-weighted count of pubs and bars within 800m walk                               | FSA Food Hygiene Register + cityseer                                                         | Point                 | As above                                                                                                                                                                                                                                                      |
| **Takeaway access** (cc_fsa_takeaway_800_wt)            | Gravity-weighted count of takeaways within 800m walk                                   | FSA Food Hygiene Register + cityseer                                                         | Point                 | As above                                                                                                                                                                                                                                                      |
| **Bus stop access** (cc_bus_800_wt)                     | Gravity-weighted count of bus and coach stops within 800m walk                         | NaPTAN (DfT) + cityseer                                                                      | Point (stop)          | As above                                                                                                                                                                                                                                                      |
| **Rail / metro access** (cc_rail_800_wt)                | Gravity-weighted count of rail and metro stations within 800m walk                     | NaPTAN (DfT) + cityseer                                                                      | Point (station)       | As above                                                                                                                                                                                                                                                      |
| **Green space access** (cc_greenspace_800_wt)           | Gravity-weighted count of designated green space sites within 800m walk                | OS Open Greenspace + cityseer                                                                | Point (site centroid) | As above                                                                                                                                                                                                                                                      |
| **Building S/V ratio** _(robustness check only)_        | Mean surface-to-volume ratio of residential buildings — physical compactness proxy     | Environment Agency LiDAR composite (2m resolution, 2000–2022) + OS Open Map Local footprints | Building polygon      | Building-level S/V computed from LiDAR-derived height × footprint area; UPRNs joined to enclosing building polygon; mean S/V across UPRNs within LSOA. Includes non-domestic buildings; used only to validate TS044 type gradient, not as a primary predictor |
| **Median construction era**                             | Median build year of EPC-registered dwellings — proxy for insulation standard          | EPC Open Data (MHCLG), certificates from November 2021 onward                                | UPRN (address)        | Most recent EPC per UPRN selected; construction age band mapped to midpoint year; median across UPRNs within LSOA                                                                                                                                             |

---

## Why Not EPCs for Building Energy?

EPCs were considered but not used as the primary energy variable, for three reasons:

1. **Coverage bias.** EPCs are required only at sale or rental, so coverage is systematically lower for long-term owner-occupied stock — predominantly detached and semi-detached houses — which are exactly the sprawling types central to this analysis.

2. **SAP is modelled, not metered.** EPC energy figures are outputs of the Standard Assessment Procedure under standardised occupancy assumptions. SAP systematically over-predicts energy use in poorly performing buildings (Few et al., 2023), and because older buildings concentrate in dense urban areas, this would artificially inflate the apparent thermal efficiency advantage of compact stock.

3. **Domestic/non-domestic ambiguity.** Mixed-use buildings — flats above shops, converted commercial premises — appear inconsistently across the domestic and non-domestic EPC registers. This ambiguity is more prevalent in compact, mixed-use areas, introducing a spatial bias in exactly the part of the housing spectrum under study.

EPC data is retained for two secondary uses only: `CONSTRUCTION_AGE_BAND` to derive median build year per LSOA, and `PROPERTY_TYPE` to cross-validate the Census TS044 type classification.

---

## Transport Energy Derivation

Transport energy is not measured directly. It is estimated from three Census tables and one
National Travel Survey scaling factor:

1. **Commute distance distribution** (TS058): each distance band is assigned a midpoint
   in kilometres (e.g., "2km to less than 5km" → 3.5 km). The weighted sum across bands,
   divided by total employed commuters, gives mean one-way commute distance per LSOA.

2. **Car mode share** (TS061): car and van commuters divided by total employed commuters
   gives the proportion travelling by car.

3. **Car-commute kilometres per LSOA**: mean commute distance × car commuter count × 2
   (return trip).

4. **Scaling to total car travel**: commuting accounts for approximately 22% of all
   car-kilometres travelled (NTS 2019, Table NTS0409). Total car-km = commute-km ÷ 0.22.

5. **Energy conversion**: 0.73 kWh/km (petrol average: ~8 litres/100 km × 9.1 kWh/litre).
   Transport energy per household = total car-km × 0.73 ÷ household count.

**Limitation:** Census 2021 was conducted during residual COVID-19 disruption. Approximately
31% of respondents recorded "works mainly from home" (zero commute kilometres). To the
extent that home-working is concentrated in knowledge-economy occupations more prevalent in
central, compact areas, this would understate the transport energy penalty of sprawl.

---

## Geographic Coverage

18 English cities are included, selected to span the compact–sprawl typological spectrum and
provide regional balance:

| City                                                                                  | Character                                   |
| ------------------------------------------------------------------------------------- | ------------------------------------------- |
| Manchester, Birmingham, Leeds, Sheffield, Liverpool, Newcastle, Nottingham, Leicester | Large provincial cities — statistical power |
| Bristol, Brighton, Southampton, Plymouth                                              | Southern / coastal — regional balance       |
| York, Cambridge, Canterbury                                                           | Historic compact cities                     |
| Milton Keynes, Stevenage                                                              | Post-war new towns — sprawl control         |
| Burnley                                                                               | Northern mill town — dense terraced control |

Analysis is restricted to built-up area extents as defined by OS Built Up Areas 2022
(BUA22). LSOAs are included only where their centroid falls within a BUA boundary.

---

## Accessibility Weighting

All cityseer accessibility metrics use **gravity weighting**: destinations closer to a node
contribute more than distant ones, following a negative exponential decay with the distance
parameter set to the analysis radius (800m). The `_wt` suffix in column names denotes this
gravity-weighted count, as opposed to a simple count of destinations within the threshold.
The 800m radius corresponds to approximately a 10-minute walk at average pedestrian speed
and represents the canonical pedestrian catchment used in walkability research.

---

## Key Limitations

| Limitation            | Nature                                                                                      | Direction of effect                                                                |
| --------------------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Transport energy modelled | Transport energy is estimated from Census commute data, not measured directly               | Uncertainty is likely greater in sprawling areas where trip purposes are more varied; any underestimate of total car travel would understate the sprawl penalty |
| COVID-19 commute data     | Census 2021 recorded unusually high home-working rates (31% works mainly from home)         | Understates the transport energy penalty of sprawl; bias favours the null         |
| LiDAR vintage         | Composite survey 2000–2022; some buildings may have changed                                 | Affects S/V robustness check only; not a primary variable                          |
| Temporal alignment    | Energy consumption year 2023; Census 2021; EPC certificates from 2021 onward                | Two-year gap between Census and energy data; structural patterns assumed stable    |
