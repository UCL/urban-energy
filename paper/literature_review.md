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

**Rode, P., Keim, C., Robazza, G., Viejo, P., Schofield, J. (2014).** "Cities and Energy: Urban Morphology and Residential Heat-Energy Demand." *Environment and Planning B*, 41(1), 138-162.
- Finds morphology can cause up to 6x difference in heat-energy demand
- Compact and tall building types have greatest heat-energy efficiency at neighbourhood scale
- Essential methodological foundation for UK research

**Norman, J., MacLean, H.L., Kennedy, C.A. (2006).** "Comparing High and Low Residential Density: Life-Cycle Analysis of Energy Use and Greenhouse Gas Emissions." *J. Urban Planning and Development*, 132(1), 10-21.
- Lifecycle perspective critical for full assessment
- High-density scenarios show 40-50% lower per-capita energy
- Includes embodied energy considerations

**Newman, P.W.G., Kenworthy, J.R. (1989).** *Cities and Automobile Dependence.*
- Seminal work establishing density-transport energy relationship
- Foundation for subsequent compact city research

### 2.2 Systematic Reviews

**Quan, S.J., Li, C. (2021).** "Urban form and building energy use: A systematic review of measures, mechanisms, and methodologies." *Renewable and Sustainable Energy Reviews*.
- Comprehensive methodological framework
- Identifies 10 key urban form variables

**The Impact of Urban Form and Density on Residential Energy Use: A Systematic Review (2023).** *MDPI Sustainability*, 15(22), 15685.
- Multinomial logistic regression predicts density-energy relationships with ~80% accuracy
- Identifies parameters causing mixed effects across studies

### 2.3 Critical Counterpoints

**Gaigne, C., Riou, S., Thisse, J.F. (2012).** "Are compact cities environmentally friendly?" *Journal of Urban Economics*, 72(2-3), 123-136.
- Challenges simple density-emissions relationships
- Shows context-dependency of outcomes
- Essential critique to engage with

**Mindali, O., Raveh, A., Salomon, I. (2004).** "Urban density and energy consumption: A new look at old statistics." *Transportation Research Part A*, 38(2), 143-162.
- Questions Newman-Kenworthy methodology
- Highlights confounding factors

### 2.4 UK-Specific Research

**Fuerst, F., McAllister, P., Nanda, A., Wyatt, P. (2015).** "Does energy efficiency matter to home-buyers? An investigation of EPC ratings and transaction prices in England." *Energy Economics*, 48, 362-373.
- Validates EPC data use in UK research
- Shows market responses to energy ratings

**Jones, R.V., Fuerst, F., Cook, M. (2015).** "The influence of building characteristics on gas and electricity consumption." *Journal of Building Performance Simulation*, 8(5), 349-366.
- UK building characteristics and energy relationships
- Important control variables identified

**NEBULA Dataset (2025).** "A National Scale Dataset for Neighbourhood-Level Urban Building Energy Modelling for England and Wales." *arXiv*.
- Recent comprehensive UK dataset
- LSOA/MSOA mapping with EUI calculations

### 2.5 Methodological Literature

**Boeing, G. (2017).** "OSMnx: New methods for acquiring, constructing, and analyzing complex street networks." *Computers, Environment and Urban Systems*, 65, 126-139.
- Standard tool for street network analysis
- Python-based workflow

**Fleischmann, M. (2019).** "momepy: Urban Morphology Measuring Toolkit." *Journal of Open Source Software*, 4(43), 1807.
- Comprehensive morphometric toolkit
- Integration with GeoPandas

**Simons, G. (2023).** "The cityseer Python package for pedestrian-scale network-based urban analysis." *Environment and Planning B*.
- Pedestrian-scale network analysis
- Localised methods avoiding edge effects

**Dibble, J., et al. (2019).** "On the origin of spaces: Morphometric foundations of urban form evolution." *Environment and Planning B*, 46(5), 866-884.
- Current state of morphometric methods
- Theoretical foundations

---

## 3. UK Datasets

### 3.1 Energy Data

**Energy Performance Certificates (EPCs)**
- Coverage: ~25 million certificates since 2006
- Spatial: Address-level, linkable to UPRN
- Limitations: Estimated (SAP) not metered; sparse in low-density areas
- Access: https://epc.opendatacommunities.org/

**NEED Database (National Energy Efficiency Data)**
- Coverage: ~11m properties with metered gas + electricity
- Temporal: 2005-present, annual
- Limitations: Postcode level only; disclosure controls
- Use: Validation of EPC estimates

**Sub-national Energy Statistics (DESNZ)**
- Coverage: MSOA-level consumption
- Use: Area-level validation and benchmarking

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

| Challenge | Recommended Solution |
|-----------|---------------------|
| EPC sparsity in low-density | Spatial smoothing, Bayesian partial pooling |
| Confounding | DAG mapping, stratified analysis |
| Temporal misalignment | Document versions, control for EPC year |
| MAUP | Multi-scale sensitivity analysis |

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
6. Winkler et al. (2023) - Transport transitions
7. Gaigne et al. (2012) - Compact city critique
8. Bibri (2020) - Comprehensive review
9. Dibble et al. (2019) - Morphometric foundations
10. Jones et al. (2015) - UK building energy

### Methods/Policy Context
11. Berghauser Pont & Haupt (2010) - Spacematrix
12. Hillier (2007) - Space syntax
13. IPCC AR5 Chapter 12 - Human settlements
