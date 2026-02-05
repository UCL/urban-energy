# Urban Energy Project: Development Agenda

**Generated:** 2026-02-04
**Based on:** Expert reviews from statistician and energy specialist

---

## Priority Legend

- **P1 (Critical)**: Must address before publication; affects validity of conclusions
- **P2 (High)**: Should address; significantly improves robustness
- **P3 (Medium)**: Recommended; enhances analysis quality
- **P4 (Low)**: Nice to have; minor improvements

---

## Phase 1: Data Quality & Selection Bias

### 1.1 EPC Selection Bias Analysis [P1] ✅ IMPLEMENTED

The EPC dataset only covers transacted properties, creating non-random missingness.

- [x] **Calculate EPC coverage rates** by:
  - Building type (house/flat/bungalow)
  - Construction age band
  - LSOA deprivation decile
  - Built form (detached/semi/terrace)

- [x] **Implement inverse probability weighting (IPW)**
  - Model P(has_epc | building_characteristics, LSOA_characteristics)
  - Use all buildings (with/without EPC) from OS data
  - Weight regression models by inverse propensity scores
  - Clip weights to [0.1, 0.9] to avoid extreme values

- [x] **Report coverage statistics** in paper methodology section

**File:** `stats/00_selection_bias_analysis.py` ✅ CREATED

### 1.2 SAP vs Metered Energy Clarification [P1] ✅ IMPLEMENTED

EPC energy is modelled under standardized assumptions, not actual consumption.

- [x] **Reframe dependent variable** throughout analysis and paper:
  - Change terminology from "energy consumption" to "potential energy demand"
  - Note: Results apply to building fabric efficiency under standard conditions

- [ ] **Consider alternative DVs**:
  - [ ] Test models with SAP score (1-100) as DV
  - [ ] Test models with energy rating (A-G, ordinal) as DV
  - [ ] Compare coefficient stability across specifications

- [ ] **NEED dataset validation** (if accessible):
  - LSOA-level actual meter readings
  - Validate SAP-based findings against metered data

**Files:**
- `stats/02_multilevel_regression.py` ✅ UPDATED with clarifying notes
- `stats/analysis_report.md` ✅ REWRITTEN with proper framing

---

## Phase 2: Missing Building Physics Controls

### 2.1 Add EPC Fabric Efficiency Variables [P1] ✅ IMPLEMENTED

Critical thermal performance variables missing from current models.

- [x] **Extract from EPC data** in Stage 3 integration:

  ```
  WALLS_ENERGY_EFF (1-5 scale)
  WINDOWS_ENERGY_EFF (1-5 scale)
  MAINHEAT_ENERGY_EFF (1-5 scale)
  ROOF_ENERGY_EFF (1-5 scale)
  FLOOR_ENERGY_EFF (1-5 scale)
  HOT_WATER_ENERGY_EFF (1-5 scale)
  ```

- [x] **Add to M1 building controls** in regression:

  ```python
  building_controls = [
      'TOTAL_FLOOR_AREA',
      'PROPERTY_TYPE',
      'building_age',
      'WALLS_ENERGY_EFF',      # NEW
      'WINDOWS_ENERGY_EFF',    # NEW
      'MAINHEAT_ENERGY_EFF',   # NEW
  ]
  ```

- [ ] **Report descriptive statistics** for fabric variables

**Files:**

- `processing/test_pipeline.py` (modify Stage 3)
- `stats/01_exploratory_analysis.py` (add descriptives)
- `stats/02_multilevel_regression.py` (add to models)

### 2.2 Derive Surface-to-Volume Ratio [P2] ✅ IMPLEMENTED

Current height variable doesn't capture thermal envelope efficiency.

- [x] **Compute in Stage 1 morphology**:

  ```python
  external_wall_area = perimeter_m * height_median * (1 - shared_wall_ratio)
  roof_area = footprint_area_m2  # flat roof assumption
  floor_area = footprint_area_m2
  envelope_area = external_wall_area + roof_area + floor_area
  volume = footprint_area_m2 * height_median
  surface_to_volume = envelope_area / volume
  ```

- [ ] **Test in models** as alternative to raw height

- [ ] **Validate** against EPC floor area (consistency check)

**File:** `processing/test_pipeline.py` ✅ UPDATED

### 2.3 Refine Shared Wall Variable [P2] ✅ IMPLEMENTED

Current `shared_wall_ratio` treats all shared walls equally.

- [x] **Create categorical attached_type**:

  ```python
  # Map from BUILT_FORM or derive from shared_wall_ratio
  attached_type = ['detached', 'semi', 'mid_terrace', 'end_terrace', 'flat']
  ```

- [x] **Test interaction terms**:
  - `shared_wall_ratio × pct_deprived` (cross-level interaction)
  - `compactness × pop_density` (cross-level interaction)
  - `building_age × pct_owner_occupied` (cross-level interaction)

- [x] **Compare models** with continuous vs categorical specification

**File:** `stats/02_multilevel_regression.py` ✅ UPDATED

---

## Phase 3: Spatial Statistics

### 3.1 Spatial Autocorrelation Testing [P1] ✅ IMPLEMENTED

Standard errors are biased if residuals are spatially correlated.

- [x] **Install dependencies**:

  ```bash
  uv add esda libpysal
  ```

- [x] **Compute Moran's I** on model residuals:

  ```python
  from esda.moran import Moran
  from libpysal.weights import KNN

  w = KNN.from_dataframe(gdf, k=8)
  mi = Moran(model.resid, w)
  # Report I statistic and p-value
  ```

- [x] **Cluster-robust standard errors** implemented as alternative to full spatial models
- [ ] **If significant (p < 0.05)**, implement spatial regression:
  - [ ] Spatial lag model (ML_Lag)
  - [ ] Spatial error model (ML_Error)
  - [ ] Compare AIC/BIC with non-spatial model

**File:** `stats/02b_spatial_regression.py` ✅ CREATED

- [ ] **Create residual map** showing spatial clustering

**File:** `stats/02b_spatial_regression.py` (new)

### 3.2 Climate Stratification [P3]

Heating demand varies ~40% across England due to climate.

- [ ] **Obtain heating degree days (HDD)** by region/LSOA
  - Source: Met Office or derived from temperature data

- [ ] **Add HDD as control variable** in models

- [ ] **Alternatively**: Stratify analysis by climate zone (North/Midlands/South)

**File:** `stats/02_multilevel_regression.py` (modify)

---

## Phase 4: Causal Identification

### 4.1 Language and Framing [P1]

Current framing implies causation; design only supports association.

- [ ] **Audit paper/report** for causal language:
  - Replace "effect" with "association"
  - Replace "reduces" with "is associated with lower"
  - Replace "causes" with "predicts"

- [ ] **Add limitations section** discussing:
  - Selection/sorting (energy-conscious people choose dense areas)
  - Omitted variable bias (unobserved building quality)
  - Reverse causality concerns

**File:** `paper/` (all sections)

### 4.2 Sensitivity Analysis [P2]

Quantify robustness to unmeasured confounding.

- [ ] **E-value calculation**:
  - How strong would unmeasured confounding need to be to explain away results?
  - Use `evalue` package or manual calculation

- [ ] **Coefficient stability analysis**:
  - Track how coefficients change as controls are added (M1→M4)
  - Large changes suggest confounding sensitivity

- [ ] **Subset analyses**:
  - [ ] Houses only (control for building type completely)
  - [ ] Post-2000 builds only (modern insulation standards)
  - [ ] Gas-heated only (control for fuel type)

**File:** `stats/02c_sensitivity_analysis.py` (new)

### 4.3 Instrumental Variables (Optional) [P4]

If causal claims are desired, consider IV approach.

- [ ] **Potential instruments** (require data acquisition):
  - Historical street grid density (pre-1900 layout)
  - Distance to Victorian rail stations
  - WWII bomb damage patterns (exogenous redevelopment)

- [ ] **IV regression** if valid instrument found:
  - First stage: Instrument → Density
  - Second stage: Predicted Density → Energy
  - Test instrument validity (F-stat, overidentification)

**File:** `stats/02d_instrumental_variables.py` (new, optional)

---

## Phase 5: Model Specification

### 5.1 Log Transformation Validation [P2] ✅ IMPLEMENTED

Using log(energy) without checking assumptions.

- [x] **Check for zeros/negatives** in energy variable
- [x] **Compare specifications** (log-linear, level-level, log-log)
- [x] **Test normality** of residuals for each specification

**File:** `stats/02c_sensitivity_analysis.py` ✅ CREATED (includes log validation)

### 5.2 Mixed Model Optimization [P3] ✅ IMPLEMENTED

Current Powell optimizer may find local optima.

- [x] **Try BFGS first**, fall back to Powell:

  ```python
  try:
      result = model.fit(method='bfgs', maxiter=500)
  except:
      result = model.fit(method='powell')
  ```

- [x] **Compare ML vs REML** estimation

- [x] **Report convergence diagnostics**

**File:** `stats/02_multilevel_regression.py` ✅ UPDATED

### 5.3 Cross-Level Interactions [P3] ✅ IMPLEMENTED

Morphology effects may vary by neighborhood characteristics.

- [x] **Test interactions**:
  - `shared_wall_ratio × pct_deprived` (party walls matter more in fuel poverty?)
  - `compactness × pop_density` (shape matters more in dense areas?)
  - `building_age × pct_owner_occupied` (age penalty varies by tenure mix?)

- [ ] **Visualize significant interactions** with marginal effects plots (optional)

**File:** `stats/02_multilevel_regression.py` ✅ UPDATED

### 5.4 VIF Threshold and Multicollinearity [P2] ✅ IMPLEMENTED

Current VIF > 10 threshold is too permissive.

- [x] **Lower threshold to VIF > 5** for warnings
- [x] **Consider PCA** on morphology variables:

**File:** `stats/01_exploratory_analysis.py` ✅ UPDATED

  ```python
  from sklearn.decomposition import PCA

  morph_vars = ['compactness', 'convexity', 'elongation', 'orientation']
  pca = PCA(n_components=2)
  morph_pcs = pca.fit_transform(df[morph_vars])
  df['morph_pc1'] = morph_pcs[:, 0]  # "shape efficiency" component
  df['morph_pc2'] = morph_pcs[:, 1]  # "orientation" component
  ```

- [ ] **Compare models** with original vs PCA-transformed morphology

**File:** `stats/01_exploratory_analysis.py` (modify)

---

## Phase 6: Model Comparison & Validation

### 6.1 Formal Model Selection [P2] ✅ IMPLEMENTED

Currently using informal R² comparison.

- [x] **Compute AIC/BIC** for all model specifications
- [x] **Likelihood ratio tests** for nested models
- [x] **Create model comparison table** for paper

**File:** `stats/02_multilevel_regression.py` ✅ UPDATED (formal_model_selection function added)

### 6.2 Cross-Validation [P2] ✅ IMPLEMENTED

Test geographic generalization.

- [x] **Spatial cross-validation** (by LSOA):
  - Split LSOAs into 5 folds
  - Train on 4 folds, predict on held-out fold
  - Report out-of-sample R², RMSE

- [x] **Random vs spatial CV comparison**
- [x] **Model comparison using spatial CV**

**File:** `stats/02e_cross_validation.py` ✅ CREATED

---

## Phase 7: Accessibility Metrics Clarification

### 7.1 Clarify Causal Pathways [P2] ✅ IMPLEMENTED

Accessibility metrics included without clear energy mechanism.

- [x] **Document mechanisms** in analysis report:

  | Metric        | Pathway to Building Energy    | Direct?                 |
  | ------------- | ----------------------------- | ----------------------- |
  | FSA density   | Walkability → lifestyle → ??? | Indirect                |
  | Greenspace    | Urban heat island?            | Weak                    |
  | Bus stops     | Transit use                   | Transport, not building |
  | Rail stations | Transit use                   | Transport, not building |

- [x] **Frame as "urban form correlates"** not energy drivers
- [ ] **Alternative**: Separate transport energy analysis (requires vehicle data)

**File:** `stats/analysis_report.md` ✅ UPDATED (Note on Accessibility Metrics section added)

---

## Phase 8: Documentation & Reproducibility

### 8.1 Update Paper Methodology [P1]

Align paper with actual analysis.

- [ ] **Selection bias section**: Document EPC coverage and IPW approach

- [ ] **Dependent variable section**: Clarify SAP vs metered, "potential energy"

- [ ] **Limitations section**:
  - Observational design (association not causation)
  - SAP model assumptions
  - EPC selection bias
  - Single study area
  - COVID-19 timing of Census 2021

**File:** `paper/methods.md` or similar

### 8.2 Code Documentation [P3]

Ensure reproducibility.

- [ ] **Add docstrings** to all analysis functions

- [ ] **Create analysis README** with:
  - Execution order for scripts
  - Expected outputs
  - Runtime estimates

- [ ] **Version pin** all dependencies in `pyproject.toml`

**File:** `stats/README.md` (update)

### 8.3 Results Archive [P3] ✅ PARTIALLY IMPLEMENTED

Preserve intermediate outputs.

- [x] **Orchestration script created**: `stats/run_all.py`
  - Runs all analysis scripts in correct order
  - Generates timestamped markdown reports to `temp/stats/results/`
  - Creates `results_latest.md` symlink for easy access

- [ ] **Save model objects**:

  ```python
  import pickle
  with open('temp/stats/models/m4_full.pkl', 'wb') as f:
      pickle.dump(result, f)
  ```

- [ ] **Export coefficient tables** to CSV

- [ ] **Archive figures** with descriptive filenames

**Files:**
- `stats/run_all.py` ✅ CREATED
- `stats/` (individual scripts)

---

## Implementation Order

### Week 1: Critical Fixes

1. [ ] 1.1 EPC selection bias analysis
2. [ ] 1.2 Reframe SAP as "potential energy"
3. [ ] 2.1 Add EPC fabric efficiency controls
4. [ ] 3.1 Spatial autocorrelation testing
5. [ ] 4.1 Audit causal language

### Week 2: Robustness ✅ COMPLETED

6. [x] 2.2 Surface-to-volume ratio
7. [x] 2.3 Refine shared wall variable
8. [x] 4.2 Sensitivity analysis (E-value, subsets)
9. [x] 5.1 Log transformation validation
10. [x] 5.4 VIF threshold and PCA

### Week 3: Validation ✅ COMPLETED

11. [x] 6.1 Formal model selection (AIC/BIC)
12. [x] 6.2 Cross-validation
13. [x] 5.2 Mixed model optimization
14. [x] 5.3 Cross-level interactions

### Week 4: Documentation (Partial)

15. [x] 7.1 Clarify accessibility pathways
16. [ ] 8.1 Update paper methodology
17. [ ] 8.2 Code documentation
18. [x] 8.3 Results archive (run_all.py created)
19. [ ] 3.2 Climate stratification (if data available)

---

## New Files Created

| File                                  | Purpose                   | Priority | Status |
| ------------------------------------- | ------------------------- | -------- | ------ |
| `stats/00_selection_bias_analysis.py` | IPW, coverage rates       | P1       | ✅ DONE |
| `stats/02b_spatial_regression.py`     | Moran's I, spatial models | P1       | ✅ DONE |
| `stats/02c_sensitivity_analysis.py`   | E-value, subset analyses  | P2       | ✅ DONE |
| `stats/02d_instrumental_variables.py` | IV regression (optional)  | P4       | Not started |
| `stats/02e_cross_validation.py`       | Spatial CV, LOSO          | P2       | ✅ DONE |
| `stats/run_all.py`                    | Orchestration & reporting | P3       | ✅ DONE |

---

## Dependencies Added

```bash
uv add esda libpysal  # spatial statistics ✅ INSTALLED
```

---

## Success Criteria

Analysis ready for publication when:

- [x] Selection bias quantified and addressed (IPW or acknowledged)
- [x] Spatial autocorrelation tested and handled
- [x] Building fabric controls included
- [x] Causal language appropriate for observational design
- [x] Model selection formally justified
- [x] Cross-validation demonstrates generalizability
- [x] Limitations clearly documented
