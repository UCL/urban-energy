# Urban Energy: Development Status

**Last updated:** 2026-03-20

---

## Case Two: OA-Level National Analysis (complete)

The analysis has been overhauled from LSOA level (18 cities, 3,678 units) to OA level (national, all 7,147 English BUAs). The pipeline uses the new CityNetwork API (cityseer 4.25.0b3) with `from_geopandas`, consolidated land-use accessibility, and centroid-based sampling.

### Current state

- **6,687 BUAs processed** → 198,779 OAs (of 203,018 loaded; filtered for valid data)
- **National pipeline complete** — all figures, tables, and paper updated
- Compounding confirmed at OA level: **1.46x** (thermal) → **1.67x** (mobility) → **2.68x** (access)

### Pipeline

- [x] New data sources: IMD 2025, DESNZ postcode energy, DVLA vehicles
- [x] Postcode → OA energy aggregation (203,018 OAs nationally)
- [x] OA pipeline with CityNetwork API (`processing/pipeline_oa.py`)
- [x] OA analysis scripts (`stats/proof_of_concept_oa.py`, `stats/oa_figures.py`, `stats/basket_index_oa.py`)
- [x] Complete national pipeline (6,687 of 7,147 BUAs)
- [x] Publication figures regenerated for national dataset
- [x] NEPI scorecard and access penalty model regenerated
- [x] Write case narrative (`paper/case_v2.md`)
- [ ] Reconcile `paper/archive/main.tex` with OA results

### Key Scripts

| Script | Purpose |
|--------|---------|
| `processing/pipeline_oa.py` | National OA pipeline (CityNetwork API, all BUAs) |
| `stats/build_case_oa.py` | Regenerate all OA figures |
| `stats/nepi.py` | NEPI scorecard and band figures |
| `stats/access_penalty_model.py` | Empirical access energy penalty |
| `stats/proof_of_concept_oa.py` | Core OA analysis and data loading |
| `stats/oa_figures.py` | Three-surfaces publication figures |
| `stats/basket_index_oa.py` | Basket case: access penalty |

### Key Results (6,687 BUAs, 198,779 OAs)

| Surface | Flat | Detached | Ratio |
|---------|-----:|---------:|------:|
| Building kWh/hh | 10,755 | 15,713 | 1.46x |
| Transport kWh/hh (overall) | 4,150 | 9,185 | 2.21x |
| Total kWh/hh (overall) | 14,906 | 24,898 | 1.67x |
| kWh per unit access | 3,292 | 8,820 | **2.68x** |

Dominant type: Flat 36,502 / Terraced 50,592 / Semi 65,986 / Detached 45,699

NEPI scorecard: Flat Band A (15,982 kWh) / Detached Band F (26,897 kWh)

### Running the analysis

```bash
uv run python stats/build_case_oa.py       # regenerate all OA figures and tables
uv run python stats/nepi.py                # regenerate NEPI scorecard
uv run python stats/access_penalty_model.py # regenerate access penalty model
```

---

## Case One: LSOA Reference (archived)

The original LSOA-level proof of concept (18 cities, 3,678 LSOAs) is preserved in archive directories.

| Script | Purpose |
| ------ | ------- |
| `stats/archive/build_case.py` | Regenerate LSOA case-one figures |
| `stats/archive/lsoa_figures.py` | LSOA three-surfaces figures |
| `stats/archive/basket_index_v1.py` | LSOA basket case |
| `stats/archive/proof_of_concept_lsoa.py` | LSOA analysis |
| `stats/archive/diagnostic_fig1b.py` | LSOA confounder diagnostics |
| `processing/archive/pipeline_lsoa.py` | LSOA pipeline (old cityseer API) |
| `paper/archive/case_v1.md` | LSOA case narrative |
| `paper/archive/main.tex` | Stale LaTeX paper |
| `stats/figures/archive_lsoa/` | LSOA figures and tables |

---

## Forward Work

### Data and Methods

- [ ] Sensitivity analysis: basket weights, distance-decay parameters, trip-demand assumptions
- [ ] Climate stratification: heating degree days as control
- [ ] Temporal validation: Census 2011 vs 2021 commute patterns (COVID effect)
- [ ] Dual DV validation: SAP-modelled (EPC) vs DESNZ metered at OA level

### Analysis Extensions

- [ ] Scaling analysis: Bettencourt superlinear/sublinear with BRES + GVA
- [ ] Lock-in quantification: DVLA fleet electrification scenarios
- [ ] Distribution analysis: quantile regression beyond means
- [ ] Deprivation deep-dive: IMD 2025 domain interactions with urban form

### Paper

- [x] Write `paper/case_v2.md` (OA-level narrative)
- [ ] Reconcile `paper/archive/main.tex` with OA results
- [ ] Formal methods section
- [ ] Finalise `references.bib`

### Code Quality

- [ ] Add pytest test suite
- [ ] Complete docstrings (NumPy style)
- [ ] Run `uv run ty check`
- [x] Environment variable for `URBAN_ENERGY_DATA_DIR`
