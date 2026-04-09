# Data Strategy

## Spatial Scale

All analysis operates at **Output Area (OA)** level — the finest geography at which
Census 2021 data is published, comprising approximately 130 households and 330 residents.
England has 188,880 OAs, of which 198,779 enter the analysis after merging across all 6,687
processed Built-Up Areas (some OAs intersect multiple BUAs). OAs are designed by ONS to be
socially homogeneous and of consistent population size, making them the natural unit for
neighbourhood-level analysis.

The earlier version of this study operated at LSOA level (~1,500 residents) because the
DESNZ metered energy consumption was only published at LSOA. That binding constraint has
since been relaxed: DESNZ now publishes **postcode-level** energy data, and the project
constructs its own postcode → OA spatial lookup using OS Code-Point Open. The Form surface
is therefore an OA-level meter-weighted mean across constituent postcodes (median ~6.3
postcodes per OA, 99.3% match rate).

The LSOA-level data and analysis is preserved in archive directories
(`stats/archive/`, `paper/archive/case_v1.md`).

---

## Data Sources

The table below lists every variable used in the analysis: what it measures, its source,
its native spatial scale, and how it is brought to OA.

| Variable | What it measures | Source | Native scale | Derivation to OA |
| -------- | ---------------- | ------ | ------------ | ---------------- |
| **Form: building energy** (kWh/hh) | Mean total domestic energy consumption — metered gas (weather-corrected by Xoserve) + electricity | DESNZ Sub-national Postcode Statistics, December 2025 release (gas: mid-May 2024 to mid-May 2025; electricity: Jan–Dec 2024) | Postcode | Postcodes joined to OAs via OS Code-Point Open spatial join; OA value is the meter-weighted mean across constituent postcodes |
| **Mobility: transport energy** (kWh/hh) | Estimated annual commute energy plus a national overall-travel scalar | Census 2021 TS058 + TS061; ECUK 2025 energy intensities; NTS 2024 total/commute scalar | OA (Census inputs) | TS058 distance bands → midpoint km; TS061 mode counts split into private/public; per-household commute energy = (private × 0.399 + public × 0.178) × 220 workdays × 2 returns / household count; multiplied by 6.04× to estimate overall travel |
| **Average commute distance** (km) | Weighted mean one-way commute distance | Census 2021, TS058 | OA | Each distance band assigned a midpoint km; weighted mean from OA counts |
| **Car commute share** (%) | Proportion of employed residents commuting by car or van | Census 2021, TS061 | OA | Car commuter count / total employed commuters at OA |
| **Walk share / cycle share** (%) | Proportion commuting on foot or by bicycle | Census 2021, TS061 | OA | As above |
| **Cars per household** | Mean vehicles available per household | Census 2021, TS045 | OA | Weighted mean (0×none + 1×one + 2×two + 3×three+) / total households at OA |
| **Housing type shares** (%) | Share of dwellings that are detached / semi-detached / terraced / flat | Census 2021, TS044 | OA | Type counts / total accommodation per OA |
| **Dominant housing type** | Plurality dwelling type for OA stratification (Flat / Terraced / Semi / Detached) | Census 2021, TS044 | OA | Plurality from OA-level type counts; sensitivity to 40–60% strict thresholds reported |
| **Population density** (persons/ha) | Residential population density | Census 2021, TS006 | OA | Native at OA |
| **Total population** | Usual residents living in households | Census 2021, TS001 | OA | Native at OA |
| **Household count** | Occupied household spaces | Census 2021, TS017 | OA | Total household spaces minus zero-person spaces |
| **Deprivation indices** | Income, employment, education, health, crime, barriers, living environment | IoD 2025 (MHCLG), File 7 | LSOA (2021) | Joined OA → LSOA via OA21CD lookup; income domain used as the primary OLS control |
| **Vehicle fleet composition** | Cars by fuel type, ULEV/BEV share | DVLA Vehicle Licensing (VEH0125, VEH0135) | LSOA (2021) | Joined OA → LSOA via OA21CD lookup |
| **Local service coverage** (0–1) | Walkable coverage of nine essential services via Gaussian decay over network distance | OS Open Roads + cityseer + FSA, NaPTAN, GIAS, NHS ODS, OS Open Greenspace | Network node | Each OA receives the meter-weighted mean over constituent UPRNs of the nearest-distance Gaussian-decayed coverage at each service-specific threshold (800–2,000 m); see case_v2.md §2.5 |
| **Network centrality** | Closeness, harmonic, betweenness at 800/1600/3200/4800/9600 m | OS Open Roads + cityseer CityNetwork API | Network node | Computed per node and aggregated to OA |
| **Building S/V ratio** *(robustness)* | Mean surface-to-volume ratio of residential buildings | LiDAR (2m, 2000–2022) + OS Open Map Local | Building polygon | Building-level S/V from LiDAR-derived height × footprint area; UPRNs joined to enclosing building polygon; mean across OA |
| **Median construction era** | Median build year of EPC-registered dwellings | EPC Open Data (MHCLG), certificates from November 2021 onward | UPRN (address) | Most recent EPC per UPRN; construction age band → midpoint year; median across OA |
| **Pre-pandemic commute** (validation) | Census 2011 commute distance and mode | Nomis QS701EW, QS702EW | OA (2011) | Joined to 2021 OAs where codes are unchanged; used for §4.5 validation |
| **OD commute distance** (robustness) | Origin–destination workplace flows | Census 2021 ODWP01EW | MSOA→MSOA | Euclidean centroid distance per origin MSOA; assigned to constituent OAs; case_v2 §4.6 |

---

## Why DESNZ Postcode Energy, not EPCs, for the Form Surface

EPCs were considered but not used as the primary energy variable, for three reasons:

1. **Coverage bias.** EPCs are required only at sale or rental, so coverage is
   systematically lower for long-term owner-occupied stock — predominantly detached and
   semi-detached houses — which are exactly the sprawling types central to this analysis.

2. **SAP is modelled, not metered.** EPC energy figures are outputs of the Standard
   Assessment Procedure under standardised occupancy assumptions. Few et al. (2023)
   demonstrate systematic over-prediction of metered consumption in EPC ratings, with the
   performance gap varying by dwelling type and age. Because older buildings concentrate
   in dense urban areas, this would artificially inflate the apparent thermal efficiency
   advantage of compact stock.

3. **Domestic/non-domestic ambiguity.** Mixed-use buildings — flats above shops,
   converted commercial premises — appear inconsistently across the domestic and
   non-domestic EPC registers. This ambiguity is more prevalent in compact, mixed-use
   areas, introducing a spatial bias in exactly the part of the housing spectrum under
   study.

EPC data is retained for two secondary uses only: `CONSTRUCTION_AGE_BAND` to derive
median build year per OA (a feature in the NEPI Form model), and `PROPERTY_TYPE` to
cross-validate the Census TS044 type classification.

---

## Transport Energy Derivation

Transport energy is not measured directly. It is estimated as **commute energy** from
Census tables TS058 and TS061, then scaled to overall travel:

1. **Commute distance distribution** (TS058): distance bands are mapped to midpoint km
   (e.g., "2km to less than 5km" → 3.5 km). Work-from-home and offshore/no-fixed-place
   categories are excluded from travelling-commuter distance.

2. **Mode counts** (TS061): commuters are split into:
   - private modes = drive + passenger + taxi + motorcycle
   - public modes = bus + train + metro/tram

3. **Annual commute distance**: travelling-commuter one-way distance × 2 (return) × 220
   workdays.

4. **Mode-specific energy intensities** (ECUK 2025):
   - road passenger intensity: 0.399 kWh/pkm
   - rail passenger intensity: 0.178 kWh/pkm

5. **Per-household commute energy**:
   `(private commute energy + public commute energy) / household count`

6. **Overall travel scenario**: commute energy × 6.04 (NTS 2024 total distance / commute
   distance ratio). Sensitivity reported across 1×–10× scalars (case_v2 §4.3).

**Limitations:**

- Census 2021 was conducted on 21 March 2021 during the third national lockdown.
  Approximately 31% of respondents recorded "works mainly from home". The pandemic
  compresses the morphology gradient because work-from-home is concentrated in
  knowledge-economy occupations more prevalent in compact areas. Pre-pandemic Census 2011
  validation (case_v2 §4.5) shows the gradient was steeper before COVID (2.00× vs 1.70×).
- Band-midpoint estimates systematically understate commute distance due to top-band
  truncation; the absolute Mobility values are conservative. OD distance robustness check
  in case_v2 §4.6 confirms the gradient is preserved when MSOA-to-MSOA distances replace
  band midpoints.

---

## The Access Surface

The Access surface is computed in two stages:

1. **Local service coverage (0–1):** For each of nine essential services
   (food restaurant, food takeaway, food pub, GP, pharmacy, school, greenspace, bus stop,
   hospital), the network-based nearest distance is converted via Gaussian decay
   (`exp(-ln(2) × (d / d_half)²)`) to a coverage score. The OA's coverage is the mean
   across all nine services. Service-specific thresholds (800 m for food/bus, 1,000 m for
   pharmacy/greenspace, 1,200 m for GP/school, 2,000 m for hospital) reflect canonical
   walking-time benchmarks.

2. **Empirical access penalty (kWh/hh/yr):** Rather than assuming trip rates and decay
   parameters, the penalty is the difference between an OA's predicted transport energy
   at its observed coverage and the predicted value at a compact reference (85% coverage,
   the flat-dominant median). The OLS specification regresses transport energy on local
   coverage with controls for log population density, household size, deprivation,
   building age, and IMD income domain (HC1 robust standard errors). See `stats/access_penalty_model.py`.

The empirical penalty (~1,540 kWh/hh/yr in detached-dominant OAs) converges with two
independent estimates: a service-specific direct calculation (~262 kWh/hh) capturing only
the nine modelled trips, and a fleet-based upper bound (~1,700 kWh/hh) from the 0.19
excess cars per household predicted in detached areas at average annual mileage.

---

## Geographic Coverage

All 6,687 English Built-Up Areas (of 7,147 total) are processed at OA level,
yielding 198,779 OAs after filtering (population > 10, ≥5 UPRNs, valid metered energy).
Built-Up Area boundaries are taken from OS Open Built Up Areas 2022 (BUA22), processed
through `data/process_boundaries.py`.

---

## Accessibility Weighting

All cityseer accessibility metrics use **Gaussian decay** weighting: destinations closer
to a node contribute more than distant ones, following a Gaussian function calibrated
such that the weight is 0.5 at the service-specific threshold (e.g., 800 m for bus
stops). Network distances reflect actual walking routes via OS Open Roads, computed by
the cityseer 4.25 CityNetwork API (`from_geopandas`). The 800 m radius corresponds to
approximately a 10-minute walk at average pedestrian speed and represents the canonical
pedestrian catchment used in walkability research.

---

## Key Limitations

| Limitation | Nature | Direction of effect |
| ---------- | ------ | ------------------- |
| Postcode energy classification | Domestic/non-domestic split via 73,200 kWh/yr Annual Quantity threshold; communal heating served via single non-domestic meter | Compresses Form gradient — underestimates flat heating (communal) and detached heating (off-gas-grid, ~15% of homes) |
| Transport energy modelled | Estimated from Census commute data plus a national NTS scalar, not measured directly | Uniform scalar likely understates the sprawl penalty; sensitivity reported |
| COVID-19 commute data | Census 2021 recorded ~31% works-mainly-from-home rates | Compresses the transport gradient; 2011 validation shows true steady-state is steeper |
| Gaussian decay thresholds | 800–2,000 m service-specific cut-offs are assumptions, not calibrated parameters | Sensitivity not yet exhaustively tested |
| Spatial autocorrelation | OAs share regional context; OLS standard errors are anti-conservative | BUA-clustered SEs partially address; spatial models on the forward-work list |
| Ecological inference | Area-level associations cannot be transposed to households | Acknowledged throughout; NEPI rates places, not households |
| Temporal alignment | Energy 2024; Census 2021; EPC certificates from Nov-2021 onward | Three-year gap between Census and energy; structural patterns assumed stable |
