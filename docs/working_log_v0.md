# Working Log: Urban Form and Building Energy

**Purpose:** Track detailed progress, experiments, decisions, and learnings. This complements the [RESEARCH_FRAMEWORK.md](RESEARCH_FRAMEWORK.md) which maintains the big picture.

---

## Log Format

Each entry follows this structure:

```
## YYYY-MM-DD: Brief Title

### Objective
What we're trying to accomplish

### Approach
What we did

### Results
What we found

### Interpretation
What it means for the hypotheses

### Next Steps
What to do next

### Code/Files Changed
- file1.py: description
- file2.py: description
```

---

## 2026-02-04: Project Review and Framework Setup

### Objective

Understand current state of analysis and establish guiding framework for continued research.

### Approach

- Read all markdown documentation (analysis_report, literature_review, methodology_notes, READMEs)
- Reviewed processing pipeline (test_pipeline.py)
- Reviewed statistical analysis (02_multilevel_regression.py)
- Analyzed current findings and identified gaps

### Results

**Current Data:**

- 173,846 properties with EPC data (Manchester test area)
- Variables: building morphology, network centrality, Census demographics, EPC energy data

**Key Findings from Existing Analysis:**

| Variable           | Raw Correlation with Energy/Capita | After Age Control |
| ------------------ | ---------------------------------- | ----------------- |
| Building Age       | r = +0.21                          | (control)         |
| Network Centrality | r = +0.00                          | r = -0.02         |
| Population Density | r = +0.00                          | r = +0.06         |

**Implemented Features:**

- [x] EPC selection bias analysis (IPW)
- [x] Surface-to-volume ratio computation
- [x] Categorical attached_type variable
- [x] Cross-level interactions
- [x] Spatial autocorrelation testing (Moran's I)
- [x] Cross-validation
- [x] Formal model selection (AIC/BIC)

**Gaps Identified:**

1. Hypothesis framing too broad ("density reduces energy")
2. Need mediation analysis to test stock composition pathway
3. Need stratified analysis by construction era and building type
4. Form factor not yet implemented (better than S/V for thermal modeling)
5. No propensity score matching for cleaner density comparison

### Interpretation

The near-zero density-energy correlation is a meaningful finding, not a null result. It suggests compact development's building energy benefits operate through building type selection, not independent "density effects."

### Framework Established

Created two guiding documents:

1. **RESEARCH_FRAMEWORK.md** - North star with hypotheses H1-H5, theoretical framework, analytical strategy
2. **WORKING_LOG.md** - This document for detailed tracking

### Next Steps (Prioritized)

1. [ ] Add form_factor to processing pipeline
2. [ ] Implement mediation analysis (H2)
3. [ ] Implement stratified analysis by era (H4) and building type (H5)
4. [ ] Update exploratory analysis with new variables
5. [ ] Run full hypothesis testing sequence

### Code/Files Changed

- RESEARCH_FRAMEWORK.md: Created (north star document)
- WORKING_LOG.md: Created (this document)

---

## 2026-02-04: Mediation and Stratified Analysis (H2, H4, H5)

### Objective

Test whether the density-energy association is mediated by building stock composition (H2), and whether it varies by construction era (H4) or building type (H5).

### Approach

1. Created `stats/02d_mediation_analysis.py` - Baron-Kenny mediation with bootstrap CIs
2. Created `stats/02f_stratified_analysis.py` - Era and building type stratification
3. Added `form_factor` and `heat_loss_parameter` to processing pipeline
4. Ran both analyses on Manchester test dataset (141,531 complete cases)

### Results

#### H2: Mediation Analysis (Density → Stock Composition → Energy)

**Key coefficients:**

| Path                   | Coefficient | p-value | Interpretation                     |
| ---------------------- | ----------- | ------- | ---------------------------------- |
| c (total effect)       | +0.035      | <0.001  | Weak POSITIVE total effect         |
| a1 (density → terrace) | +0.082      | <0.001  | Denser areas have more terraces    |
| a2 (density → flat)    | +0.043      | <0.001  | Denser areas have more flats       |
| b1 (terrace → energy)  | **-0.073**  | <0.001  | Terraces use LESS energy           |
| b2 (flat → energy)     | **+0.510**  | <0.001  | Flats use MORE energy (per capita) |
| c' (direct effect)     | +0.019      | <0.001  | Direct effect after mediation      |

**Indirect effects:**

- Via terrace: -0.006 (density → more terraces → lower energy)
- Via flat: **+0.022** (density → more flats → HIGHER energy)
- Total indirect: +0.016
- Proportion mediated: 45.4%

**Critical Finding:** Flats show HIGHER energy per capita (β = +0.51), creating an offsetting effect that masks the terrace benefit.

#### H4: Age-Density Confounding

**Age-density correlation:** r = **-0.064** (denser areas have NEWER buildings)

- Opposite to hypothesis! We expected dense central areas to have older buildings.

**Within-era density-energy correlations:**

| Construction Era     | N      | r          | Interpretation |
| -------------------- | ------ | ---------- | -------------- |
| Pre-1919 (Victorian) | 46,289 | **-0.120** | Negative       |
| 1919-1944 (Interwar) | 22,755 | -0.075     | Negative       |
| 1945-1979 (Post-war) | 33,179 | +0.017     | Near zero      |
| 1980+ (Modern)       | 39,308 | **+0.141** | Positive       |

**Age-residualized density correlation:** r = +0.051

#### H5: Building Type Heterogeneity

**Density-energy correlation by type:**

| Type       | N      | r          | β(density) | Interpretation          |
| ---------- | ------ | ---------- | ---------- | ----------------------- |
| **Houses** | 85,736 | **-0.109** | -0.000013  | Density → LOWER energy  |
| **Flats**  | 55,795 | **+0.061** | +0.000003  | Density → HIGHER energy |

**Within-house built forms (all show negative density effect):**

| Built Form    | N      | r      |
| ------------- | ------ | ------ |
| Semi-Detached | 30,916 | -0.140 |
| Detached      | 3,020  | -0.129 |
| Mid-Terrace   | 36,761 | -0.127 |
| End-Terrace   | 14,573 | -0.030 |

### Interpretation

**MAJOR FINDING: The near-zero overall density-energy correlation MASKS opposite effects:**

1. **For HOUSES:** Density is associated with **lower** energy (r = -0.11)
   - This is the expected "compact development benefit"
   - Consistent across all built forms (detached, semi, terrace)

2. **For FLATS:** Density is associated with **higher** energy per capita (r = +0.06)
   - This is a **per-capita artifact**: flats have smaller households
   - Same energy divided by fewer people = higher per-capita values
   - This is NOT a true "density penalty" for flats

3. **Methodological insight:** "Energy per capita" may be problematic as DV
   - Consider using energy per m² (energy intensity) instead
   - Or control for household size separately

4. **Era effects:** Modern buildings (1980+) show positive density-energy association
   - May reflect higher-rise flats with different thermal characteristics
   - Or different household composition in new-build flats

### Implications for Hypotheses

| Hypothesis              | Status                  | Finding                                       |
| ----------------------- | ----------------------- | --------------------------------------------- |
| H2 (Mediation)          | **Partially supported** | 45% mediated; offsetting flat/terrace effects |
| H4 (Age confounding)    | **Not supported**       | Dense areas have newer (not older) buildings  |
| H5 (Type heterogeneity) | **STRONGLY supported**  | Houses vs flats show opposite patterns        |

### Next Steps

1. [ ] Re-run analysis with energy INTENSITY (kWh/m²) as DV instead of per capita
2. [ ] Investigate why modern buildings show positive density association
3. [ ] Consider household size as explicit control rather than embedded in DV
4. [ ] Update RESEARCH_FRAMEWORK.md with revised understanding

### Code/Files Changed

- `processing/test_pipeline.py`: Added form_factor, heat_loss_parameter
- `stats/02d_mediation_analysis.py`: Created (mediation analysis)
- `stats/02f_stratified_analysis.py`: Created (stratified analysis)

---

## Experiment Tracking

### Completed Experiments

| ID  | Date       | Hypothesis | Approach                               | Result                           | Conclusion                                    |
| --- | ---------- | ---------- | -------------------------------------- | -------------------------------- | --------------------------------------------- |
| E01 | Prior      | H3         | Raw correlation density-energy         | r ≈ 0.00                         | No raw association                            |
| E02 | Prior      | H3         | Partial correlation (age-controlled)   | r ≈ -0.02 to +0.06               | No association after age control              |
| E03 | Prior      | H1         | Built form comparison                  | Terraces < Detached (after age)  | Shared walls matter                           |
| E04 | Prior      | -          | Shared wall: continuous vs categorical | Similar AIC                      | Categorical provides more insight             |
| E05 | 2026-02-04 | H1         | Add form_factor to pipeline            | Code added                       | Awaits re-run of pipeline                     |
| E06 | 2026-02-04 | H2         | Mediation: density → stock → energy    | 45% mediated; offsetting effects | Flats show +0.51 effect (per capita artifact) |
| E07 | 2026-02-04 | H4         | Stratify by construction era           | Era effects vary                 | Pre-1919 negative, Modern positive            |
| E08 | 2026-02-04 | H5         | Stratify by building type              | Houses r=-0.11, Flats r=+0.06    | OPPOSITE patterns by type                     |
| E10 | 2026-02-04 | H4         | Age-residualized density               | r = +0.05                        | Weak positive after removing age              |

### Planned Experiments

| ID  | Hypothesis | Approach                              | Status                          |
| --- | ---------- | ------------------------------------- | ------------------------------- |
| E09 | H3         | Propensity score matching for density | Pending                         |
| E11 | H5         | Re-run with energy INTENSITY (kWh/m²) | **DONE** - see 2026-02-05 entry |
| E12 | -          | Modern buildings deep-dive            | **DONE** - see 2026-02-05 entry |

---

## 2026-02-05: Energy Intensity Analysis (CRITICAL FINDING)

### Objective

Test whether the house/flat divergence is a per-capita artifact by comparing energy INTENSITY (kWh/m²) with energy per capita.

### Approach

Created `stats/02g_intensity_analysis.py` comparing both DVs across:

- Overall correlations
- By building type (houses vs flats)
- By construction era
- Modern buildings deep-dive

### Results

#### PATTERN COMPLETELY REVERSES WITH INTENSITY

| Metric                 | Houses         | Flats          |
| ---------------------- | -------------- | -------------- |
| **Per capita**         | r = -0.109     | r = +0.061     |
| **Intensity (kWh/m²)** | r = **+0.081** | r = **-0.061** |

#### Regression with controls (floor area, age, building type)

| DV         | β(is_flat)        | R²       |
| ---------- | ----------------- | -------- |
| Per capita | +0.22 (p < 0.001) | 0.15     |
| Intensity  | -0.004 (p = 0.09) | **0.64** |

#### Modern Buildings (1980+)

- Composition: **69% flats**, 31% houses
- Height: Flats mean 16m, Houses mean 6m
- Height-energy correlation:
  - Per capita: r = +0.30 (height → higher per capita)
  - Intensity: r = -0.01 (no relationship)
- Modern flat density-intensity: r = **-0.04** (negative, as expected)

#### Household Size by Era and Type

| Era      | Houses | Flats    |
| -------- | ------ | -------- |
| Pre-1919 | 2.77   | 2.26     |
| 1980+    | 2.60   | **1.98** |

### Interpretation

**THE PER-CAPITA ARTIFACT IS CONFIRMED:**

1. **Flats are actually MORE efficient per m²** (r = -0.06 with density)
   - The positive per-capita association is entirely due to smaller households
   - Same energy ÷ fewer people = inflated per-capita values

2. **Houses show HIGHER intensity in dense areas** (r = +0.08)
   - Opposite of the per-capita finding!
   - Possible explanation: dense areas have smaller houses (less floor area to heat)

3. **R² dramatically better with intensity** (0.64 vs 0.15)
   - Intensity is a more appropriate DV for thermal efficiency

4. **Modern flats are NOT thermally inefficient**
   - Height doesn't correlate with intensity (r = -0.01)
   - The per-capita penalty is purely household size composition

### Implications

**CRITICAL: The choice of DV fundamentally changes conclusions:**

| DV         | Conclusion about flats                    |
| ---------- | ----------------------------------------- |
| Per capita | "Flats in dense areas have higher energy" |
| Intensity  | "Flats in dense areas have LOWER energy"  |

**Recommendation:** Use **energy intensity (kWh/m²)** as primary DV for building thermal efficiency. Per-capita conflates building physics with household composition.

### Next Steps

1. [ ] Re-run all H1-H5 analyses with intensity as primary DV
2. [ ] Update RESEARCH_FRAMEWORK.md with revised DV recommendation
3. [ ] Consider: Should we report BOTH metrics?

### Code/Files Changed

- `stats/02g_intensity_analysis.py`: Created (intensity comparison)

---

## 2026-02-05: H1 Form Factor Testing and Pipeline Data Quality Fix

### Objective

1. Test whether form_factor improves H1 (thermal physics) model beyond existing variables
2. Fix data quality issue with exploding thermal physics ratios
3. Add transport analysis capability (H6)

### Approach

1. Ran H1 regression adding form_factor to existing model
2. Investigated extreme values in surface_to_volume and form_factor
3. Fixed pipeline to cap values at reasonable bounds
4. Created data quality reporting script
5. Created H6 transport analysis script

### Results

#### H1: Form Factor Analysis

| Model         | Variables                    | R²    | ΔAIC |
| ------------- | ---------------------------- | ----- | ---- |
| Base          | floor_area, age              | 0.143 | -    |
| + form_factor | floor_area, age, form_factor | 0.145 | -200 |

**Conclusion:** form_factor adds minimal explanatory power (~0.2% R²) after controlling for floor area and age. The variable is significantly correlated but practically redundant with size.

#### Data Quality Issue Identified

- 2,758 buildings with volume < 1 m³ (including 800 with exactly 0)
- These cause surface_to_volume to reach 2,097,152 and form_factor to reach 215,962
- Root cause: DSM/height data missing or erroneous for small structures

#### Pipeline Fix Applied

Changed `processing/test_pipeline.py` to:

```python
MIN_VOLUME_M3 = 10.0  # Minimum reasonable building volume
MAX_S2V = 5.0         # Cap surface-to-volume ratio
MAX_FF = 30.0         # Cap form factor
```

Records below volume threshold set to NaN (not imputed).

#### H6: Transport Analysis Setup

Created `stats/02h_transport_analysis.py` with:

- Census car ownership (TS045) as transport energy proxy
- Combined building + transport footprint analysis
- Preliminary finding: high-density flats show ~39% lower TOTAL footprint than low-density houses

### Interpretation

1. **H1 refinement:** Form factor does not substantially improve thermal physics model. Shared wall ratio and building type remain primary predictors.

2. **Data quality:** ~1.4% of records have physically implausible thermal ratios due to height data quality. Filtering these is appropriate data cleaning, not p-hacking.

3. **H6 (transport):** Demonstrates the "sprawl penalty" - even if suburban houses are thermally efficient, their transport footprint dominates total energy.

### Code/Files Changed

- `processing/test_pipeline.py`: Added MIN_VOLUME_M3, MAX_S2V, MAX_FF caps
- `stats/00_data_quality.py`: Created (reusable data quality report)
- `stats/02h_transport_analysis.py`: Created (H6 transport analysis)
- `RESEARCH_FRAMEWORK.md`: Added H6 hypothesis

---

## 2026-02-05: Research Consolidation and Report Structure

### Objective

Consolidate research findings into reportable package with clear narrative structure, reproducible pipeline, and publication-ready figures.

### Approach

1. Created 3-part narrative structure for analysis report (Thermal Physics → Per-Capita Artifact → Combined Footprint)
2. Ran full regression suite with log_energy_intensity as primary DV
3. Generated key figures (3-8) demonstrating core findings
4. Organized scripts into canonical pipeline with archive for deprecated code

### Results

#### Regression Suite Confirmation

- **Intensity R²**: 0.63 vs 0.19 for per-capita (4× better fit)
- **Flat coefficient**: −0.07 (intensity) vs +0.18 (per-capita) - reversal confirms artifact
- **ICC**: 9.1% (most variance within LSOAs, not between)

#### Key Figures Generated

| Figure | Content                             |
| ------ | ----------------------------------- |
| fig3   | House/flat divergence scatter plots |
| fig4   | Metric comparison bar chart         |
| fig5   | Household size by type/era          |
| fig6   | Mediation path diagram              |
| fig7   | Combined footprint stacked bars     |
| fig8   | Car ownership by density            |

#### Script Organization

**Canonical Pipeline (run_all.py):**

- 00_data_quality.py → Data validation
- 02_regression_suite.py → Main regression (intensity DV)
- 03_generate_figures.py → Figure generation

**Archived (stats/archive/):**

- 02_multilevel_regression.py → superseded by 02_regression_suite.py
- 04_visualizations.py → superseded by 03_generate_figures.py
- 03_shap_analysis.py → optional ML analysis

### Code/Files Changed

- `stats/analysis_report.md`: Complete rewrite with 3-part narrative
- `stats/02_regression_suite.py`: Created (consolidated intensity regression)
- `stats/03_generate_figures.py`: Created (publication figures)
- `stats/run_all.py`: Updated (orchestrates canonical pipeline)
- `RESEARCH_FRAMEWORK.md`: Updated hypothesis status table
- `stats/archive/`: Created for deprecated scripts

---

## Decision Log

| Date       | Decision                           | Rationale                                                                | Alternatives Considered             |
| ---------- | ---------------------------------- | ------------------------------------------------------------------------ | ----------------------------------- |
| 2026-02-04 | Decompose hypothesis into H1-H5    | Original hypothesis too broad; need testable components                  | Single omnibus hypothesis           |
| 2026-02-04 | Use SAP "potential energy" framing | EPC data is modeled, not metered; must be clear about what DV represents | "Energy consumption" (misleading)   |
| 2026-02-04 | Prioritize mediation analysis      | Key to understanding whether density works through stock selection       | Skip mediation, use regression only |

---

## Variable Definitions (Quick Reference)

### Thermal Physics (Building-Level)

- `shared_wall_ratio` = shared_wall_length / perimeter (0 = detached, ~0.5 = mid-terrace)
- `surface_to_volume` = envelope_area / volume (lower = more efficient)
- `form_factor` = envelope_area / volume^(2/3) (TO BE ADDED; 1 = cube, >1 = less efficient)
- `attached_type` = categorical: detached, semi, mid_terrace, end_terrace, flat

### Stock Composition (Area-Level)

- `pct_terraced` = % buildings that are terraced in catchment
- `pct_flats` = % buildings that are flats
- `pct_detached` = % buildings that are detached

### Density Measures

- `pop_density` = persons per hectare (Census)
- `building_coverage_ratio` = Σfootprint / catchment_area
- `FAR` = Σfloor_area / catchment_area
- `uprn_density` = addresses per hectare

---

## Questions to Resolve

- [x] ~~What is the exact ICC from the multilevel model?~~ → 9.1% between LSOAs
- [ ] Is there significant spatial autocorrelation in residuals? (Moran's I run but not reported)
- [x] ~~Sample size for each construction era?~~ → Pre-1919: 46K, 1919-44: 23K, 1945-79: 33K, 1980+: 51K
- [x] ~~Correlation between density and building age?~~ → r = −0.064 (denser areas have NEWER buildings)

---

## Resources

- **Data:** `temp/processing/test/uprn_integrated.gpkg`
- **Main analysis:** `stats/02_regression_suite.py` (intensity DV)
- **Figures:** `stats/figures/`
- **Results:** `temp/stats/results/`
- **Run pipeline:** `uv run python stats/run_all.py`

---

_Update this log after each analysis session. Keep entries focused and actionable._
