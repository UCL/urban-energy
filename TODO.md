# Urban Energy: Development Status

**Last updated:** 2026-03-18

---

## Case Two: OA-Level National Analysis (in progress)

The analysis has been overhauled from LSOA level (18 cities, 3,678 units) to OA level (national, all 7,147 English BUAs). The pipeline uses the new CityNetwork API (cityseer 4.25.0b3) with `from_geopandas`, consolidated land-use accessibility, and centroid-based sampling.

### Current state

- **94 BUAs processed** → 67,263 OAs (of ~178k total)
- **National pipeline running** — processing largest BUAs first, skip-if-exists for restarts
- Compounding confirmed at OA level: **1.55x** (thermal) → **1.83x** (mobility) → **2.69x** (access)

### Pipeline

- [x] New data sources: IMD 2025, DESNZ postcode energy, DVLA vehicles
- [x] Postcode → OA energy aggregation (178,355 OAs nationally)
- [x] OA pipeline with CityNetwork API (`processing/pipeline_oa.py`)
- [x] OA analysis scripts (`stats/proof_of_concept_oa.py`, `stats/oa_figures.py`, `stats/basket_index_oa.py`)
- [x] Publication figures regenerated for 94-BUA sample
- [ ] Complete national pipeline (7,147 BUAs — running)
- [ ] Write case narrative (`paper/case_oa.md`)
- [ ] Reconcile `paper/main.tex` with OA results

### Key Scripts

| Script | Purpose |
|--------|---------|
| `processing/pipeline_oa.py` | National OA pipeline (CityNetwork API, all BUAs) |
| `stats/build_case_oa.py` | Regenerate all OA figures |
| `stats/proof_of_concept_oa.py` | Core OA analysis and data loading |
| `stats/oa_figures.py` | Three-surfaces publication figures |
| `stats/basket_index_oa.py` | Basket case: access penalty |

### Key Results (94 BUAs, 67,263 OAs)

| Surface | Flat | Detached | Ratio |
|---------|-----:|---------:|------:|
| Building kWh/hh | 10,852 | 16,865 | 1.55x |
| Transport kWh/hh (overall) | 4,131 | 7,556 | 1.83x |
| kWh per unit access | 3,337 | 8,964 | **2.69x** |

Dominant type: Flat 18,460 / Terraced 20,941 / Semi 21,304 / Detached 6,558

### Running the national pipeline

```bash
uv run python processing/pipeline_oa.py    # all 7,147 BUAs (skip-if-exists)
uv run python stats/build_case_oa.py       # regenerate figures from completed BUAs
```

---

## Case One: LSOA Reference (archived)

The original LSOA-level proof of concept (18 cities, 3,678 LSOAs) is preserved as a reference.

| Script | Purpose |
|--------|---------|
| `stats/build_case.py` | Regenerate LSOA case-one figures |
| `stats/lsoa_figures.py` | LSOA three-surfaces figures |
| `stats/basket_index_v1.py` | LSOA basket case |
| `stats/proof_of_concept_lsoa.py` | LSOA analysis |
| `processing/pipeline_lsoa.py` | LSOA pipeline (old cityseer API) |
| `paper/case_v1.md` | LSOA case narrative |

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

- [ ] Write `paper/case_oa.md` (OA-level narrative)
- [ ] Reconcile `paper/main.tex` with OA results
- [ ] Formal methods section
- [ ] Finalise `references.bib`

### Code Quality

- [ ] Add pytest test suite
- [ ] Complete docstrings (NumPy style)
- [ ] Run `uv run ty check`
- [ ] Environment variable for `STORAGE_DIR`
