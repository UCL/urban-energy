# Urban Energy: Development Status

**Last updated:** 2026-02-27

---

## Case One: Completed

The LSOA-level "three energy surfaces" proof of concept is complete for 18 English cities.

### Pipeline

- [x] Data acquisition: Census, DESNZ energy, EPC, LiDAR, FSA, NaPTAN, GIAS, NHS ODS, OS boundaries, OS roads, OS greenspace, ONS scaling
- [x] Building morphology: footprint, height, S/V ratio, form factor, shared walls, adjacency (via momepy + LiDAR)
- [x] Network analysis: cityseer centrality + gravity-weighted accessibility at 800m
- [x] LSOA aggregation: 3-stage pipeline (morphology, network, integration) across 18 cities
- [x] Three energy surfaces: building (DESNZ metered), transport (Census commute), accessibility (network + land use)
- [x] Basket index v1: illustrative trip-demand case for selected land uses (food, health, education, greenspace, transit) showing the access penalty at observed rates
- [x] Publication figures: 7 main figures + basket subfolder + summary tables
- [x] Case narrative: `paper/case_v1.md` with full 9-step analytical structure

### Key Scripts

| Script | Purpose |
|--------|---------|
| `stats/build_case.py` | Regenerate all case-one figures |
| `stats/lsoa_figures.py` | Three-surfaces publication figures |
| `stats/basket_index_v1.py` | Illustrative basket case: access penalty for selected land uses |
| `stats/proof_of_concept_lsoa.py` | Core LSOA analysis and data loading |
| `stats/diagnostic_fig1b.py` | Confounder diagnostics |

### Key Result

For the selected land-use categories at observed trip rates, a 3.5x access penalty emerges between detached-dominant and flat-dominant LSOAs. The compounding widens at each normalisation level (kWh/m2 -> kWh/capita -> kWh/capita/accessibility). This is illustrative — the basket covers a particular set of land uses, not all travel purposes.

---

## Forward: Case Two and Beyond

### Data and Methods

- [ ] Dual DV validation: compare SAP-modelled (EPC) vs DESNZ metered energy at LSOA level
- [ ] Climate stratification: add heating degree days as control or stratify by region
- [ ] Sensitivity analysis: test basket weights, distance-decay parameters, trip-demand assumptions
- [ ] Temporal validation: compare Census 2011 vs 2021 commute patterns (COVID effect)

### Analysis Extensions

- [ ] Scaling analysis: test Bettencourt superlinear/sublinear scaling with BRES employment + GVA data
- [ ] Lock-in quantification: scenario modelling across insulation and EV technology levels
- [ ] Distribution analysis: move beyond means to full distributional comparisons (quantile regression)
- [ ] Deprivation deep-dive: interaction effects between IMD quintile and urban form

### Paper

- [ ] Reconcile `paper/main.tex` with current `paper/case_v1.md` narrative
- [ ] Write formal methods section for case-one approach
- [ ] Finalise bibliography in `references.bib`

### Code Quality

- [ ] Add pytest test suite (framework configured, no tests yet)
- [ ] Complete docstrings for all public functions (NumPy style)
- [ ] Run full type check pass (`uv run ty check`)
- [ ] Consider environment variable for `STORAGE_DIR` path (currently hardcoded)
