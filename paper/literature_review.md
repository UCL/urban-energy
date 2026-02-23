# Literature Review and Background Research

## Urban Form and Building Energy Consumption in the UK

**Research Report - February 2026**

---

## 1. Executive Summary

This report provides a comprehensive literature review on urban-scale energy models, focusing on the relationship between urban morphology and building energy consumption in the UK context. Key findings include:

- Strong theoretical and empirical support for morphology-energy relationships, but with important nuances
- Limited UK-specific quantitative research on building energy (vs transport) and urban form
- Critical methodological challenges around data sparsity, confounding, and causal inference
- Emerging datasets (NEBULA, EPC open data) create new research opportunities

---

## 2. Key Literature

### 2.1 Foundational Studies

**Rode, P., Keim, C., Robazza, G., Viejo, P., Schofield, J. (2014).** "Cities and Energy: Urban Morphology and Residential Heat-Energy Demand." _Environment and Planning B_, 41(1), 138-162.

Rode et al. used a static (steady-state) energy balance model, developed in collaboration between LSE Cities and the European Institute for Energy Research (EIFER) at the Karlsruhe Institute of Technology, to compute annual heat-energy demand as a theoretical value across neighbourhood-scale samples. They studied the four largest European cities---London, Paris, Berlin, and Istanbul---identifying the five most dominant residential building typologies in each (20 typologies total). For each typology, they selected real neighbourhood samples at a scale of 500m × 500m, then constructed idealised archetypal versions by removing non-conforming ("invasive") buildings to isolate the morphological signal. Both real and idealised samples were run through the simulation, yielding approximately 120 samples covering around 30 million sqm of floor area.

Urban morphology was operationalised through the SpaceMate framework (Berghauser Pont and Haupt), measuring floor area ratio (FAR/FSI), ground space index (GSI/site coverage), average building height, open space ratio (OSR), and surface-to-volume ratio (S/V). The simulation computed solar gains and building surface energy losses from 3D digital elevation models of urban areas. Micro-scale design parameters (U-values, glazing ratios) were held constant to isolate morphological effects.

The headline finding was that urban-morphology-induced heat-energy efficiency can produce up to a six-fold difference in heat-energy demand between the least efficient form (detached housing) and the most efficient (compact, tall urban blocks). Specific quantitative results included: at S/V ratios of 0.15, energy performance ranged from 35 to 80 kWh/m²/a, whereas at S/V ratios of 0.40 it ranged from 110 to 200 kWh/m²/a; at densities above FAR 4, all morphologies converged to a narrow performance band of 30--50 kWh/m²/a; and at FAR 1 the greatest variation was observed, ranging from 50 to 150 kWh/m²/a. Average building height served as a logarithmic proxy for heat-energy demand, with diminishing returns at greater heights.

The authors argued that urban morphology constitutes a significant but underexplored determinant of residential heating energy demand, alongside the better-studied interventions of behavioural change and technological improvement. In Europe, approximately 70% of residential energy use is heating-related, making the morphological effect on heat demand highly consequential. They concluded that urban form decisions create long-lasting physical infrastructure that locks in energy consumption patterns for decades.

Important limitations include the focus on theoretical heat-energy demand only (not cooling, lighting, appliances, embodied energy, or transport); exclusion of occupant behaviour; constant technological assumptions across comparisons; and the acknowledgement that idealised modelling could exaggerate morphological effects, with theoretical values potentially as low as 70% of observed real-world data. Only residential buildings in four European cities were studied, constraining generalisability.

**Norman, J., MacLean, H.L., Kennedy, C.A. (2006).** "Comparing High and Low Residential Density: Life-Cycle Analysis of Energy Use and Greenhouse Gas Emissions." _J. Urban Planning and Development_, 132(1), 10-21.

Norman et al. applied an Economic Input-Output Life-Cycle Assessment (EIO-LCA) model to compare the energy use and GHG emissions of high-density urban core and low-density suburban residential development in Toronto, Canada. The study was distinctive in combining three lifecycle components within a single analytical framework: (1) embodied energy in construction materials for dwellings, utilities, and roads (estimated via EIO-LCA and amortised over a 50-year building lifespan); (2) operational energy for building heating, cooling, lighting, and appliances; and (3) transportation energy for private automobile and public transit use. Operational and transport estimates drew on nationally and regionally averaged data for the Toronto area.

The primary finding was that low-density suburban development is 2.0 to 2.5 times more energy- and GHG-intensive than high-density urban core development on a per capita basis---translating to the commonly cited 40--60% lower per-capita energy for high-density settings. However, a critical methodological contribution was demonstrating that the choice of functional unit fundamentally alters conclusions: when normalised per unit of living space (per m²) rather than per capita, the advantage largely disappears (factor of 1.0 to 1.5), because low-density dwellings provide considerably more living space per person. Building operations dominated the lifecycle energy budget at 60--70%, followed by transport at 20--30%, with embodied energy in construction materials accounting for approximately 10%.

The authors argued that the most targeted measures for reducing GHG emissions should focus on transportation (due to petroleum dependence), while the most effective measures for reducing total energy use should target building operations. The lifecycle lens was crucial in demonstrating that while embodied energy is relatively small, integrating all three categories provides a more complete basis for urban planning decisions. Limitations include reliance on regionally averaged rather than site-specific data, only two case studies in one city, exclusion of some infrastructure categories, and limited treatment of behavioural and socioeconomic confounders.

**Newman, P.W.G., Kenworthy, J.R. (1989).** _Cities and Automobile Dependence: An International Sourcebook._ Gower Publishing.

Newman and Kenworthy's landmark study was the product of ten years of primary data collection (commenced in 1982), in which the authors personally visited government agencies in 32 cities across North America, Australia, Europe, and wealthy Asia to assemble standardised, comparable data on urban form, transport infrastructure, travel behaviour, and energy consumption for 1960, 1970, and 1980. To ensure like-for-like comparison, they delineated urbanised areas by physically drawing boundaries on city maps and weighing cut-out paper to measure area, rather than relying on inconsistent administrative boundaries.

The core analytical output was a cross-sectional comparison across the 32 cities, producing the now-iconic scatter plot (p. 128) of per-capita gasoline consumption against gross urban population density. This figure---described as "one of the iconic images of the urban planning field"---revealed a negative hyperbolic (power) relationship with R² = 0.86. US cities (approximately 14--15 persons/ha) clustered at the high-consumption end, with average gasoline consumption nearly twice that of Australian cities, four times higher than European cities (approximately 50 persons/ha), and ten times higher than wealthy Asian cities (150+ persons/ha, with Hong Kong at approximately 300 persons/ha). The extreme anchors were Houston (lowest density, highest consumption) and Hong Kong (highest density, lowest consumption). A density pertinence threshold of approximately 30 inhabitants per hectare was identified as the point at which public transport investment becomes economically viable.

The central argument was that physical planning decisions---particularly regarding urban density, land use patterns, and transport infrastructure---are the primary determinants of automobile dependence, not economic factors such as income or fuel price. When gasoline consumption was adjusted for US prices and incomes, the density-energy relationship persisted. Newman and Kenworthy defined automobile dependence as a structural characteristic of cities that forces reliance on private vehicles, not a matter of individual preference. They identified a self-reinforcing cycle: low-density land use leads to automobile dependence, which drives road investment, which enables further sprawl. They also identified a "transit leverage effect" whereby each kilometre of transit service replaces 5 to 7 kilometres of car travel, as transit access causes relocation, trip consolidation, and modal shift.

The work has been extensively critiqued. Gordon and Richardson (1989) objected ideologically to the planning implications, characterising the policy prescriptions as authoritarian. Gomez-Ibanez (1991) provided a substantive methodological critique, arguing that the bivariate analysis lacked control for confounders (income, fuel price). Mindali, Raveh, and Salomon (2004) applied multivariate Co-Plot analysis to the original data and found no direct impact of gross urban density when analysed multivariately, instead finding inner area employment density and CBD density more influential. Ewing et al. (2018) tested the theory across 157 US urbanised areas and concluded that it is not localised neighbourhood density but rather regional accessibility that explains lower VMT in compact areas---reframing the density argument as an accessibility argument. Despite these critiques, Newman and Kenworthy's policy prescriptions of re-urbanisation and transit prioritisation have been "largely mainstreamed" in planning practice, and the framework remains the foundational reference point for all subsequent work on urban form and transport energy.

### 2.2 Systematic Reviews

**Quan, S.J., Li, C. (2021).** "Urban form and building energy use: A systematic review of measures, mechanisms, and methodologies." _Renewable and Sustainable Energy Reviews_, 139, 110662.

Quan and Li systematically reviewed 89 articles on the relationship between urban form and building energy use, identifying a striking 365 unique metrics of urban morphology across the literature---revealing significant fragmentation in naming conventions, operational definitions, and data sources. They organised their analysis along four dimensions: measure definitions (how urban form and energy are operationalised), mechanism assumptions (what causal pathways are posited), methodologies (what analytical approaches are employed), and site context (climate zone, building function, geographic location).

The 365 metrics were classified into six broad categories: (1) urban tissue configuration (spatial arrangement of blocks and parcels); (2) street network (connectivity, geometry, canyon dimensions); (3) building-plot characteristics (height, footprint, S/V ratio, orientation, coverage ratio); (4) land use (functional mixing, diversity, zoning); (5) natural features and greenspace; and (6) urban growth patterns. The most commonly used specific measures across studies were floor area ratio (FAR), building coverage ratio (BCR), building height, surface-to-volume ratio, sky view factor, street canyon aspect ratio, and building orientation.

A key contribution was the development of pathway maps summarising the mechanisms through which urban form transmits its influence to building energy outcomes. The authors categorised mechanism complexity into three levels: Level 0 studies consider only building design and occupancy in isolation; Level 1 studies additionally account for mutual shading, daylight availability, and passive solar gains as mediated by urban geometry; and Level 2 studies further integrate microclimate effects including urban heat island, modified wind patterns, and altered ventilation potential. They found that most studies operated at Level 1 (solar radiation pathways only), with very few attempting Level 2 integration---identified as a significant gap.

The review highlighted three main debates: the magnitude of urban form's influence on building energy (with estimates ranging from modest to a six-fold difference); the inconsistent effects of densification (higher population density typically associates with lower per-capita energy but shows conflicting results for heating and cooling specifically); and the preference for particular typologies (compact and tall types consistently outperform detached housing, but results are heavily constrained by climate zone). The reviewed studies were split between simulation-based approaches (enabling parametric control but limited in realism) and empirical/statistical approaches (grounded in real-world data but constrained by data availability). Critical gaps included definitional inconsistency across studies, inadequate understanding of causal pathway contributions, insufficient integration of findings across spatial scales, and poor behavioural integration.

**Narimani Abar, S., Schulwitz, M., Faulstich, M. (2023).** "The Impact of Urban Form and Density on Residential Energy Use: A Systematic Review." _Sustainability_ (MDPI), 15(22), 15685.

Narimani Abar et al. conducted a systematic review extracting 10 urban form variables from the literature and examining their correlations with residential energy use. The distinctive methodological contribution was the application of a multinomial logistic regression as a meta-analytical tool: each reviewed study's finding was classified into one of three outcome categories (negative association, positive association, or non-significant), and the model predicted the direction of reported findings based on nine study-level characteristics---including the number of density indicators used, the type of energy examined (heating, cooling, electricity, total), the unit of measurement, methodology, data reliability, publication year, geographical location, and climate classification. The model achieved approximately 80% classification accuracy, demonstrating that the apparent "disagreement" in the literature is largely systematic and predictable from methodological and contextual choices rather than being random noise.

The review found that approximately 67% of studies report that increasing density leads to less residential energy consumption, while approximately 10% reach the opposite conclusion, with the remainder finding non-significant or mixed relationships. A particularly important finding was the climate-dependent nature of the relationship: density correlates negatively with residential energy use in cold climates (where compact forms reduce heating demand through shared walls, reduced S/V ratios, and heat island effects), but the relationship may become positive in temperate regions, potentially due to cooling demands and behavioural factors. The authors also highlighted a disciplinary gap: only 35% of reviewed publications came from urban planning journals, with the remaining 65% from energy, engineering, or other fields that may interpret urban form variables differently---leading to definitional inconsistency that complicates cross-study comparison. Recommendations included standardising density measurement, explicitly accounting for climate context, differentiating between energy types rather than treating residential energy as monolithic, and increasing research from an urban planning perspective.

### 2.3 Critical Counterpoints

**Gaigne, C., Riou, S., Thisse, J.F. (2012).** "Are compact cities environmentally friendly?" _Journal of Urban Economics_, 72(2-3), 123-136.

- Challenges simple density-emissions relationships
- Shows context-dependency of outcomes
- Essential critique to engage with

**Mindali, O., Raveh, A., Salomon, I. (2004).** "Urban density and energy consumption: A new look at old statistics." _Transportation Research Part A_, 38(2), 143-162.

- Questions Newman-Kenworthy methodology
- Highlights confounding factors

### 2.4 UK-Specific Research

**Fuerst, F., McAllister, P., Nanda, A., Wyatt, P. (2015).** "Does energy efficiency matter to home-buyers? An investigation of EPC ratings and transaction prices in England." _Energy Economics_, 48, 362-373.

- Validates EPC data use in UK research
- Shows market responses to energy ratings

**Jones, R.V., Fuerst, F., Cook, M. (2015).** "The influence of building characteristics on gas and electricity consumption." _Journal of Building Performance Simulation_, 8(5), 349-366.

- UK building characteristics and energy relationships
- Important control variables identified

**NEBULA Dataset (2025).** "A National Scale Dataset for Neighbourhood-Level Urban Building Energy Modelling for England and Wales." _arXiv_.

- Recent comprehensive UK dataset
- LSOA/MSOA mapping with EUI calculations

### 2.5 Methodological Literature

**Boeing, G. (2017).** "OSMnx: New methods for acquiring, constructing, and analyzing complex street networks." _Computers, Environment and Urban Systems_, 65, 126-139.

- Standard tool for street network analysis
- Python-based workflow

**Fleischmann, M. (2019).** "momepy: Urban Morphology Measuring Toolkit." _Journal of Open Source Software_, 4(43), 1807.

- Comprehensive morphometric toolkit
- Integration with GeoPandas

**Simons, G. (2023).** "The cityseer Python package for pedestrian-scale network-based urban analysis." _Environment and Planning B_.

- Pedestrian-scale network analysis
- Localised methods avoiding edge effects

**Dibble, J., et al. (2019).** "On the origin of spaces: Morphometric foundations of urban form evolution." _Environment and Planning B_, 46(5), 866-884.

- Current state of morphometric methods
- Theoretical foundations

### 2.6 Synthesis: Built Form Measures for This Study

The reviewed literature identifies a large and fragmented landscape of urban form measures---Quan and Li (2021) alone catalogued 365 unique metrics across 89 studies---but convergent patterns emerge when the measures are examined against the data sources available for England-wide analysis. This section maps the literature's recommended variables to the datasets assembled in this study (EPC, LiDAR-derived building heights, OS building footprints, Census 2021, NaPTAN, FSA establishments, and cityseer network analysis), identifies which measures can be operationalised, and proposes a selection framework grounded in the existing evidence base.

#### Building-Level Thermal Physics

The most direct pathway from urban form to building energy use operates through envelope thermodynamics: the ratio of exposed building surface area to enclosed volume determines heat loss per unit of habitable space. Rode et al. (2014) demonstrated that this surface-to-volume ratio (S/V) alone could account for much of the six-fold variation in heat-energy demand across European neighbourhood typologies. They found that at S/V ratios of 0.15, energy performance ranged from 35 to 80 kWh/m²/a, whereas at S/V ratios of 0.40 it ranged from 110 to 200 kWh/m²/a. This measure can be computed directly from the data available in this study: building footprint area and perimeter from OS Open Map Local, combined with LiDAR-derived building heights, yield both S/V ratio and the related form factor (envelope area / volume^(2/3)), which provides a dimensionless metric comparable across building sizes.

Closely related is the shared wall ratio---the proportion of a building's perimeter that abuts adjacent structures. This metric, computable via momepy's adjacency functions with a 1.5m tolerance to accommodate OS cartographic gaps between buildings, captures the thermal benefit of terraced and semi-detached forms over detached housing. Each shared wall eliminates a heat-loss surface, and EPC data confirms that mid-terrace dwellings exhibit substantially lower energy intensity than detached houses of equivalent age and floor area. The combination of S/V ratio and shared wall ratio operationalises Rode et al.'s core finding that compact and attached building types achieve superior thermal efficiency, while offering building-level granularity that their neighbourhood-scale simulation could not provide.

Additional shape metrics available through momepy---orientation (deviation from cardinal directions, affecting passive solar gain), compactness (circular compactness ratio), convexity (ratio of area to convex hull area), and elongation (longest to shortest axis ratio)---provide supplementary descriptors of building form. Quan and Li (2021) identified building orientation and plan depth among frequently used measures in the literature, noting that these affect solar radiation pathways (their Level 1 mechanism). However, the evidence for their independent contribution to energy outcomes, once S/V ratio and shared walls are controlled, is less well established.

#### Density and Neighbourhood-Scale Form

Population density, the central variable in the Newman and Kenworthy (1989) framework, is available from Census 2021 at Output Area level (approximately 100--200 households). The literature consistently finds that higher population density associates with lower per-capita energy use, though Narimani Abar et al. (2023) demonstrated that this finding is climate-dependent and that 10% of studies report the opposite relationship. Norman et al. (2006) showed that the choice of functional unit---per capita versus per square metre---fundamentally alters conclusions about density's effect on building energy, a distinction this study can examine directly using EPC floor area data.

Beyond population density, the literature identifies several neighbourhood-scale built form metrics. Rode et al. (2014) operationalised urban morphology through the SpaceMate framework (Berghauser Pont and Haupt, 2010), measuring floor area ratio (FAR/FSI), ground space index (GSI/site coverage), and open space ratio (OSR). These metrics require defined plot or neighbourhood boundaries for computation. In this study, cityseer's network-based spatial aggregation provides a natural alternative: rather than administrative boundaries (which introduce MAUP effects), morphological statistics can be computed within pedestrian-scale network catchments at multiple distances (400m, 800m, 1600m). This yields network-aggregated measures of building density (total footprint area or volume within catchment), mean building height, and floor area ratio approximations that are sensitive to walkable neighbourhood structure rather than arbitrary boundary placement.

Quan and Li's (2021) six-category classification provides a useful checklist for coverage. Of their categories, this study can operationalise: (1) urban tissue configuration, via network-aggregated building statistics and centrality measures; (2) street network characteristics, via cityseer's closeness and betweenness centrality at multiple distance thresholds; (3) building-plot characteristics, via the full suite of momepy shape metrics and LiDAR heights; (4) land use mixing, via FSA establishment accessibility and NaPTAN public transport node proximity; and (5) natural features, via OS OpenGreenspace accessibility. The sixth category, urban growth patterns, is not directly available but could be partially inferred from EPC construction age band distributions within catchments.

#### Accessibility and Transport Dimensions

Newman and Kenworthy (1989) established that the density-transport energy relationship is mediated by infrastructure provision and land use accessibility, not density alone. Ewing et al. (2018) further reframed this as a regional accessibility argument. This study's use of cityseer for network-based accessibility analysis---measuring proximity to bus stops, rail stations, restaurants, takeaways, pubs, and greenspace at multiple distance thresholds---directly operationalises this accessibility dimension. Census 2021 data on car ownership (TS045), commute distance (TS058), and travel mode (TS061) provide the transport energy estimates that complete the building-plus-transport lifecycle framing advocated by Norman et al. (2006).

#### Proposed Measure Selection

Given the evidence base, the measures for this study can be organised into three tiers reflecting the strength of evidence and analytical role:

**Tier 1: Core envelope physics** (strongest evidence, direct causal pathway)

- Surface-to-volume ratio (S/V) --- primary thermal efficiency metric (Rode et al., 2014; Quan and Li, 2021)
- Shared wall ratio --- captures attached/detached form effect on heat loss
- Total floor area --- controls for building size; enables per-m² normalisation (Norman et al., 2006)
- Building height --- logarithmic proxy for heat-energy demand (Rode et al., 2014); from LiDAR
- Construction age band --- controls for building regulation era and insulation standards

**Tier 2: Neighbourhood context** (consistent evidence, mediating pathways)

- Population density --- fundamental density-energy relationship (Newman and Kenworthy, 1989; Narimani Abar et al., 2023)
- Network centrality --- captures urban structure beyond simple density; multiple distance thresholds
- Building type composition --- proportion of detached/terraced/flat within catchment; mediates density effect on energy
- Public transport accessibility --- proximity to bus/rail (NaPTAN); enables transit leverage assessment

**Tier 3: Supplementary descriptors** (supporting evidence, exploratory)

- Building orientation, compactness, elongation --- shape descriptors with theoretical but less empirically established independent effects
- Land use accessibility --- FSA establishments and greenspace as walkability proxies
- Car ownership and commute distance --- transport energy estimation (Norman et al., 2006; Newman and Kenworthy, 1989)

This selection most closely aligns with Rode et al.'s (2014) analytical framework, which prioritised S/V ratio, building height, and FAR as the primary morphological determinants of heat-energy demand. However, where Rode et al. used simulated energy demand across idealised neighbourhood archetypes, this study applies equivalent measures to observed EPC energy estimates across the full English building stock, linked to actual neighbourhood context via network-based spatial analysis. The addition of transport and accessibility dimensions extends the framework toward the lifecycle perspective advocated by Norman et al. (2006), while the network-based aggregation addresses the MAUP and boundary effects that Quan and Li (2021) identified as persistent methodological weaknesses in the field.

### 2.7 The EPC Performance Gap: SAP-Modelled vs Metered Energy

A critical methodological consideration for any study using EPC energy estimates is the well-documented discrepancy between SAP-modelled energy and actual metered consumption. The literature consistently finds a systematic, asymmetric performance gap: EPCs over-predict energy use in thermally inefficient homes and under-predict (or accurately predict) in efficient homes. The practical consequence is that actual variation in energy consumption across EPC bands is far smaller than the certificates suggest.

#### The Prebound and Rebound Effects

**Sunikka-Blank, M. and Galvin, R. (2012).** "Introducing the prebound effect: the gap between performance and actual energy consumption." _Building Research & Information_, 40(3), 260-273. DOI: [10.1080/09613218.2012.690952](https://doi.org/10.1080/09613218.2012.690952)

Sunikka-Blank and Galvin introduced the conceptual framework that has structured subsequent research on this topic. Analysing data from 3,400 German homes, they plotted calculated energy performance ratings against actual measured consumption and identified two complementary phenomena. The _prebound effect_ describes how occupants of thermally inefficient homes consume on average 30% less heating energy than the calculated rating, with the gap increasing as the rated inefficiency worsens. Occupants adapt by heating fewer rooms, using lower temperatures, or heating for shorter periods---partly from economic necessity (fuel poverty) and partly from behavioural adaptation to the building's thermal characteristics. The converse _rebound effect_ describes how occupants of highly efficient homes tend to consume more than predicted, as low marginal heating costs encourage higher comfort temperatures or heating of additional rooms.

The combined effect is a compression of actual energy consumption toward the middle of the distribution: the real-world difference between the worst and best buildings is substantially smaller than energy performance certificates suggest. Sunikka-Blank and Galvin estimated that economically feasible fuel savings through comprehensive thermal retrofits would typically amount to 25--35%, rather than the 70--80% claimed by German policy-makers---because the starting-point consumption is already lower than the certificate implies. Although this study used German data, the prebound/rebound framework has since been confirmed in UK, Irish, Belgian, Swiss, and Danish contexts.

#### UK Smart Meter Evidence

**Few, J., Manouseli, D., McKenna, E., Pullinger, M., Zapata-Webborn, E., Elam, S., Shipworth, D. and Oreszczyn, T. (2023).** "The over-prediction of energy use by EPCs in Great Britain: A comparison of EPC-modelled and metered primary energy use intensity." _Energy & Buildings_, 288, 113024. DOI: [10.1016/j.enbuild.2023.113024](https://doi.org/10.1016/j.enbuild.2023.113024)

Few et al. provides the most rigorous UK evidence to date on the EPC performance gap. The study compared EPC-modelled and smart-meter measured annual energy use on a like-for-like basis in 1,374 gas-heated British households from the Smart Energy Research Lab (SERL) Observatory. Both EPC and metered data were converted to total primary energy use intensity (PEUI) using consistent SAP 2012 primary energy factors, enabling direct comparison of the same quantity for the first time.

The headline finding was a mean difference of -66 kWh/yr/m² across the sample, with EPCs significantly over-predicting metered consumption. The gap was not uniform across bands:

| EPC Band | Mean difference (kWh/yr/m²) [95% CI] | p-value | n   |
| -------- | ------------------------------------ | ------- | --- |
| A and B  | +10.6 [-10.7, +31.9]                 | 0.32    | 32  |
| C        | -25.9 [-17.9, -33.9] (-8%)           | <0.001  | 439 |
| D        | -65.8 [-58.3, -73.4] (-20%)          | <0.001  | 705 |
| E        | -161 [-144, -178] (-35%)             | <0.001  | 185 |
| F and G  | -276 [-206, -345] (-48%)             | <0.001  | 13  |

Only bands A and B showed no statistically significant difference between modelled and metered PEUI. The difference in metered PEUI between bands was much smaller than modelled: the metered difference between bands D and E was only 8.1 kWh/yr/m², compared to the modelled difference of 103 kWh/yr/m²---less than 10% of the expected differentiation.

A particularly important contribution was the analysis of homes matching the SAP model's occupancy and heating assumptions (the home is heated to the assumed set-point, the whole home is heated, and occupant numbers match the SAP estimate). Even in these 51 matching homes, the gradient of the difference against modelled PEUI remained significantly below zero (gradient -0.81 [-0.48, -1.13], p < 0.001), indicating that the discrepancy is not solely explained by occupant behaviour differing from model assumptions. The study also found that the RdSAP assessment procedure (used for most existing buildings) showed a statistically significant over-prediction pattern (gradient -0.69, p < 0.001), while new homes assessed via the full SAP procedure showed no significant gradient (-0.14, p = 0.65)---implicating the RdSAP process itself, with its reliance on default values and age-based assumptions, as a structural contributor to the gap.

**McKenna, E., Few, J., Webborn, E., Anderson, B., Elam, S., Shipworth, D., Cooper, A., Pullinger, M. and Oreszczyn, T. (2022).** "Explaining daily energy demand in British housing using linked smart meter and socio-technical data in a bottom-up statistical model." _Energy & Buildings_, 258, 111845. DOI: [10.1016/j.enbuild.2022.111845](https://doi.org/10.1016/j.enbuild.2022.111845)

McKenna et al. used the same SERL Observatory dataset to develop bottom-up statistical models of daily energy demand, achieving adjusted R² of 63--80% depending on sample size and the combination of contextual data used (smart meter, weather, building characteristics, and socio-technical survey data covering appliance ownership, demographics, behaviours, and attitudes). This study provides the methodological foundation for Few et al. (2023), demonstrating that SERL covariates explain daily energy demand variation well and that the linked smart meter plus survey dataset enables robust comparison of modelled and metered energy at the household level.

#### National-Scale Gas Consumption Evidence

**Summerfield, A.J., Oreszczyn, T., Hamilton, I.G., Lowe, R.J., Elwell, C.A. and Crawley, J. (2019).** "What do empirical findings reveal about modelled energy demand and energy ratings? Comparisons of gas consumption across the English residential sector." _Energy Policy_, 129, 997-1007. DOI: [10.1016/j.enpol.2019.02.033](https://doi.org/10.1016/j.enpol.2019.02.033)

Summerfield et al. used data from over 2.5 million gas-heated dwellings in England from the National Energy Efficiency Data-Framework (NEED) to compare metered gas consumption with estimates from the Cambridge Housing Model (CHM), a national energy stock model based on the SAP methodology. The central finding was that metered gas consumption across all EPC bands falls almost entirely within the narrow range estimated for EPC band C. The actual gradient between bands is far shallower than predicted: where the model predicts large differences between, say, band D and band F, metered consumption barely changes.

The CHM overestimated average gas consumption for all dwelling types built before 1930, most notably for large detached dwellings---precisely the building types that receive the worst EPC ratings due to high surface-to-volume ratios and assumed poor insulation. For dwellings built since 1930, model estimates were in closer agreement with NEED data. The policy implication is that savings from upgrading dwellings to at least EPC band C would be substantially lower than predicted, since actual consumption in lower-rated bands is already near the band C level.

#### EPC Measurement Error

**Crawley, J., Biddulph, P., Northrop, P.J., Wingfield, J., Oreszczyn, T. and Elwell, C. (2019).** "Quantifying the Measurement Error on England and Wales EPC Ratings." _Energies_, 12(18), 3523. DOI: [10.3390/en12183523](https://doi.org/10.3390/en12183523)

Crawley et al. analysed 1.6 million repeat EPC assessments of existing dwellings in England and Wales to estimate the measurement error inherent in the assessment process. The one standard deviation measurement error decreased with EPC rating, from approximately +/-8.0 EPC points at a rating of 35 to +/-2.4 at a rating of 85. This error exceeds the limit recommended in UK guidance for all but the most efficient buildings. A practical consequence is systematic misclassification: the study estimated that 24% of band D homes are incorrectly rated as band C. The error is larger for less efficient dwellings, compounding the over-prediction bias identified by Few et al. and Summerfield et al.---not only does the SAP model over-predict consumption for inefficient buildings, but the assessment process introduces additional noise that is largest precisely where the model bias is greatest.

#### Spatial Variation in the Performance Gap

**Firth, S.K., Allinson, D. and Watson, S. (2024).** "Quantifying the spatial variation of the energy performance gap for the existing housing stock in England and Wales." _Journal of Building Performance Simulation_. DOI: [10.1080/19401493.2024.2380309](https://doi.org/10.1080/19401493.2024.2380309)

Firth et al. compared EPC predictions against annual gas meter readings across the existing housing stock in England and Wales, finding that the performance gap varies greatly across regions, local authorities, built forms, construction ages, and levels of predicted gas consumption. The gap increases by 3.6 percentage points (absolute) for every 1,000 kWh/year increase in predicted gas consumption, suggesting a structural error in the RdSAP model that scales with predicted thermal inefficiency. This spatial heterogeneity is directly relevant to studies investigating relationships between urban form and energy consumption: if the performance gap correlates with building typology (e.g. detached houses in suburban areas have larger gaps than terraced houses in dense urban areas), then associations between urban morphology and SAP-modelled energy may partly reflect systematic model error rather than genuine differences in energy demand.

#### Synthesis and Implications for This Study

The performance gap literature carries several implications for the present analysis:

1. **Dependent variable interpretation.** EPC `ENERGY_CONSUMPTION_CURRENT` is a SAP-modelled estimate of annual energy demand under standardised assumptions, not a measure of actual energy consumption. The literature demonstrates that this metric overstates the variation in real energy use across the housing stock, particularly at the extremes. Results using this dependent variable should be framed as associations with _modelled potential energy demand_ rather than actual consumption.

2. **Systematic bias correlated with building form.** The SAP over-prediction is largest for building types with high surface-to-volume ratios (detached houses, bungalows) and oldest construction ages---precisely the building characteristics most strongly associated with low-density suburban morphologies. This creates a risk that morphology-energy associations partly reflect SAP model bias rather than genuine energy demand differences. The Few et al. finding that the RdSAP procedure itself (rather than occupant behaviour) drives the discrepancy strengthens this concern, since most existing buildings in the dataset are assessed via RdSAP.

3. **Compressed actual gradient.** Summerfield et al.'s finding that metered gas consumption across all bands falls within the range estimated for band C implies that any regression model using SAP energy as the dependent variable will overstate the predictive power of building characteristics---the true R² for actual consumption would be substantially lower, because the actual variance to be explained is smaller.

4. **Dual dependent variable validation strategy.** Rather than applying post-hoc corrections to SAP estimates---which would introduce model-on-model uncertainty from Few et al.'s high-RMSE regression (95.4 kWh/yr/m²), small samples at the band extremes (n=13 for F and G), and spatial heterogeneity in the correction (Firth et al., 2024)---this study adopts a triangulation approach using two complementary dependent variables:

   - **Model A (building-level):** SAP-modelled `ENERGY_CONSUMPTION_CURRENT` / `TOTAL_FLOOR_AREA` as energy intensity (kWh/m²). This provides individual building-level variation and rich building-specific controls, but reflects modelled potential demand with the documented systematic biases.

   - **Model B (LSOA-level):** DESNZ metered mean energy per LSOA (`lsoa_total_mean_kwh` from the sub-national energy statistics). This provides actual metered consumption aggregated to neighbourhood level (~1,600 residents per LSOA), capturing real-world energy use including behavioural adaptation, but at coarser spatial resolution and without building-specific controls.

   A coefficient comparison table across the two specifications directly addresses the Few et al. and Summerfield et al. critique. If morphological associations are consistent in sign and relative magnitude across both SAP-modelled and metered DVs, they are robust to the performance gap. If associations attenuate with the metered DV, they likely reflect SAP model bias correlated with building form rather than genuine energy demand differences. If associations strengthen, they may be masked by SAP's compression of the true gradient. This design is arguably stronger than either measure alone, and provides a natural robustness check that most EPC-based studies cannot perform.

---

## 3. UK Datasets

### 3.1 Energy Data

**Energy Performance Certificates (EPCs)**

- Coverage: ~25 million certificates since 2006
- Spatial: Address-level, linkable to UPRN
- Limitations: SAP-modelled energy under standardised assumptions, not metered consumption; systematic over-prediction for inefficient buildings (see Section 2.7); RdSAP assessment process introduces measurement error; sparse coverage in low-density areas
- Access: <https://epc.opendatacommunities.org/>

**NEED Database (National Energy Efficiency Data-Framework)**

- Coverage: ~18.2 million properties with metered gas and electricity (2025 report)
- Temporal: 2005-present, annual metered consumption
- Limitations: Available at property level but with disclosure controls; not publicly downloadable at individual level
- Use: National-scale validation of EPC estimates; the primary evidence base for Summerfield et al. (2019)
- Access: <https://www.gov.uk/government/collections/national-energy-efficiency-data-need-framework>

**Sub-national Energy Statistics (DESNZ)**

- Coverage: LSOA-level domestic gas and electricity consumption (weather-corrected for gas)
- Temporal: 2010-2024
- Spatial: ~33,000 LSOAs in England; domestic meters only (electricity profile classes 1-2; gas below 73,200 kWh/year)
- Use: Area-level validation of SAP-modelled estimates; quantifying the performance gap at neighbourhood scale
- Access: <https://www.gov.uk/government/collections/sub-national-energy-consumption-data>

### 3.2 Urban Form Data

**OS MasterMap**

- High-resolution building footprints
- 3D data available separately
- Licensing required

**OpenStreetMap**

- Free alternative for buildings and streets
- Variable completeness

**Census 2021**

- LSOA-level sociodemographic data
- Key variables: tenure, household composition, travel mode, car ownership

### 3.3 Accessibility Data

**NaPTAN** - Public transport access points
**GTFS feeds** - Transit schedules
**OSM POI data** - Retail and amenities

---

## 4. Methodological Recommendations

### 4.1 Spatial Units

Street-network level aggregation recommended because:

- Captures walkable catchments (~500m)
- Allows morphological feature analysis
- Finer-grained than LSOA
- Supported by cityseer/momepy tools

Conduct sensitivity analysis at LSOA/MSOA to address MAUP.

### 4.2 Statistical Approach

**Recommended: Multi-level regression**

```
Energy ~ urban_form + building_controls + census_controls +
         (1|street_segment) + (1|LSOA)
```

Accounts for:

- Hierarchical data structure
- Spatial clustering
- Unbalanced data

**Complementary approaches:**

- Spatial regression (lag/error models)
- Matching for causal inference
- Machine learning for feature discovery

### 4.3 Key Challenges

| Challenge                   | Recommended Solution                        |
| --------------------------- | ------------------------------------------- |
| EPC sparsity in low-density | Spatial smoothing, Bayesian partial pooling |
| Confounding                 | DAG mapping, stratified analysis            |
| Temporal misalignment       | Document versions, control for EPC year     |
| MAUP                        | Multi-scale sensitivity analysis            |

### 4.4 Morphological Metrics

**Density:** Building density, FAR/FSI, population density
**Form:** Street network density, block size, building heights
**Accessibility:** PT access, retail distance, mixed-use ratio
**Network:** Centrality measures, permeability

---

## 5. Critical Gaps This Research Addresses

1. **Limited UK building energy-morphology research** - Most work focuses on transport
2. **Methodological integration** - Few studies combine heterogeneous data at fine spatial scale
3. **Policy relevance** - Planning policy rarely cites energy-morphology evidence
4. **Street-network scale** - Under-developed compared to coarser administrative units

---

## 6. Theoretical Considerations

### 6.1 Lock-in Mechanisms

- **Infrastructure path-dependency**: Dispersed utilities constrain densification
- **Governance lock-in**: Planning regulations perpetuate existing forms
- **Social lock-in**: Cultural preferences for suburban living
- **Economic lock-in**: Property values and development economics

### 6.2 Cautions

- **Correlation vs causation**: Analysis shows association; causal claims require assumptions
- **Rebound effects**: Per-capita savings may not translate to absolute reductions
- **Equity implications**: Densification has distributional consequences
- **Embodied carbon**: Densification may require demolition with lifecycle costs

---

## 7. Policy Context

**UK Net Zero Strategy (2023)** - Technology-focused; under-develops spatial aspects
**NPPF (2023)** - Design policies could be strengthened by form-energy evidence
**Climate Change Committee** - Identifies gap between planning potential and performance
**UKGBC Roadmap** - Whole-life carbon pathway for built environment

---

## 8. Priority Reading List

### Essential (Read First)

1. Rode et al. (2014) - Morphology and heat demand
2. Norman et al. (2006) - Lifecycle analysis
3. Boeing (2017) - OSMnx methods
4. Fleischmann (2019) - momepy toolkit
5. Fuerst et al. (2015) - UK EPC validation

### High Priority

1. Winkler et al. (2023) - Transport transitions
2. Gaigne et al. (2012) - Compact city critique
3. Bibri (2020) - Comprehensive review
4. Dibble et al. (2019) - Morphometric foundations
5. Jones et al. (2015) - UK building energy

### Methods/Policy Context

1. Berghauser Pont & Haupt (2010) - Spacematrix
2. Hillier (2007) - Space syntax
3. IPCC AR5 Chapter 12 - Human settlements
