# Urban Energy: Development Status

**Last updated:** 2026-04-09

---

## Case Two: NEPI at OA-Level (current)

The analysis pivoted from LSOA level (18 cities, 3,678 units) to a national OA-level
study and is now packaged as the **Neighbourhood Energy Performance Index (NEPI)**.
The pipeline uses the cityseer 4.25 CityNetwork API (`from_geopandas`) with consolidated
land-use accessibility and centroid-based sampling.

### Current state

- **6,687 BUAs processed** → 198,779 OAs (filtered for valid data)
- National pipeline complete; NEPI scorecard, access penalty model, and planning tool live
- Compounding confirmed at OA level: **1.46×** (Form) → **1.67×** (total) → **2.68×** (kWh per unit access)
- Median Flat NEPI: **15,982 kWh/hh/yr (Band A)** ; Median Detached: **26,897 kWh/hh/yr (Band F)**
- Gap decomposition: Form 45% / Mobility 43% / Access penalty 14%
- Static planning tool deployed at <https://UCL.github.io/urban-energy/> (mirror in `docs/`)

### Pipeline

- [x] New data sources: IMD 2025, DESNZ postcode energy, DVLA vehicles
- [x] Postcode → OA energy aggregation (`build_postcode_oa_lookup.py` + `aggregate_energy_oa.py`)
- [x] OA pipeline with CityNetwork API (`processing/pipeline_oa.py`)
- [x] OA analysis scripts (`stats/proof_of_concept_oa.py`, `stats/oa_figures.py`, `stats/basket_index_oa.py`)
- [x] Complete national pipeline (6,687 of 7,147 BUAs)
- [x] Publication figures regenerated for the national dataset
- [x] NEPI scorecard (`stats/nepi.py`) and access penalty model (`stats/access_penalty_model.py`)
- [x] NEPI planning tool: XGBoost models, Streamlit app, static HTML/JS build
- [x] Pre-pandemic Census 2011 validation (case_v2 §4.5)
- [x] OD commute distance robustness check (case_v2 §4.6)
- [x] Static tool deployed to GitHub Pages via `docs/`
- [x] Storage paths centralised behind `URBAN_ENERGY_DATA_DIR`
- [x] Case narrative `paper/case_v2.md` (full IMRaD draft, NEPI framing)
- [ ] Reconcile `paper/archive/main.tex` with OA results (or retire it)
- [ ] Finalise `paper/references.bib`

### Key Scripts

| Script | Purpose |
|--------|---------|
| `processing/pipeline_oa.py` | National OA pipeline (CityNetwork API, all BUAs) |
| `stats/build_case_oa.py` | Regenerate three-surfaces + basket figures |
| `stats/nepi.py` | NEPI scorecard, A–G bands, surface decomposition |
| `stats/access_penalty_model.py` | Empirical access-energy penalty (OLS) |
| `stats/nepi_model.py` | Train four XGBoost planning-tool models with monotonic constraints + SHAP |
| `stats/nepi_app.py` | Streamlit interactive planning tool |
| `stats/nepi_static/index.html` | Static browser planning tool (mirrored to `docs/`) |
| `stats/proof_of_concept_oa.py` | Core OA data loading and aggregation functions |
| `stats/oa_figures.py` | Three-surfaces publication figures |
| `stats/basket_index_oa.py` | Illustrative basket case: access penalty by land use |

### Key Results (6,687 BUAs, 198,779 OAs)

| Surface | Flat | Detached | Ratio |
|---------|-----:|---------:|------:|
| Form (building, kWh/hh) | 10,755 | 15,713 | 1.46× |
| Mobility (overall, kWh/hh) | 4,150 | 9,185 | 2.21× |
| Access penalty (kWh/hh) | 0 | 1,519 | — |
| **Total NEPI (kWh/hh)** | **15,982 (A)** | **26,897 (F)** | **1.68×** |
| kWh per unit access | 3,292 | 8,820 | 2.68× |

Dominant type counts: Flat 36,502 / Terraced 50,592 / Semi 65,986 / Detached 45,699.

Robustness: pre-pandemic Census 2011 transport gradient is **steeper** (2.00× vs 1.70× in COVID-affected 2021); plurality sensitivity *steepens* the gradient (1.84× at 60% purity).

### Running the analysis

```bash
uv run python stats/build_case_oa.py             # all OA case figures + tables
uv run python stats/nepi.py                      # NEPI scorecard + bands
uv run python stats/access_penalty_model.py      # empirical access penalty
uv run python stats/nepi_model.py                # train XGBoost planning models
uv run streamlit run stats/nepi_app.py           # interactive tool
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
| `processing/archive/pipeline_lsoa.py` | LSOA pipeline (still imported by `pipeline_oa.py` for shared morphology constants) |
| `paper/archive/case_v1.md` | LSOA case narrative |
| `paper/archive/main.tex` | Stale LaTeX paper |
| `stats/figures/archive_lsoa/` | LSOA figures and tables |
| `notes/` | v0 working notes (LSOA-era methodology, roadmap, log) |

---

## Forward Work

### Data and Methods

- [ ] Sensitivity analysis: basket weights, distance-decay parameters, trip-demand assumptions
- [ ] Climate stratification: heating degree days as control
- [ ] Dual DV validation: SAP-modelled (EPC) vs DESNZ metered at OA level
- [ ] Calibrate Gaussian decay thresholds against observed travel survey distances
- [ ] Spatial autocorrelation: BUA-clustered SEs are partial; consider spatial error / lag models

### Analysis Extensions

- [ ] Scaling analysis: Bettencourt superlinear/sublinear with BRES + GVA
- [ ] Lock-in quantification: DVLA fleet electrification scenarios (BEV / PHEV penetration paths)
- [ ] Distribution analysis: quantile regression beyond means
- [ ] Deprivation deep-dive: IMD 2025 domain interactions with urban form
- [ ] NEPI archetype clustering for typology-aware recommendations

### Paper

- [x] Write `paper/case_v2.md` (OA-level NEPI narrative, IMRaD)
- [ ] Reconcile `paper/archive/main.tex` with OA results, or retire in favour of pure markdown
- [ ] Formal methods section (port from case_v2 §3)
- [ ] Finalise `paper/references.bib`
- [ ] Cover-letter framing for target journal

### Code Quality

- [ ] Add pytest test suite (currently configured but empty)
- [ ] Complete docstrings (NumPy style)
- [ ] Run `uv run ty check` clean
- [x] Environment variable for `URBAN_ENERGY_DATA_DIR`
- [ ] Refactor the in-README static-tool exporter into `stats/nepi_static/build.py`
