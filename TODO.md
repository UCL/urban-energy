# Urban Energy Project: Development Agenda

**Generated:** 2026-02-04
**Based on:** Expert reviews from statistician and energy specialist

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

- [x] **DESNZ metered energy data** ✅ DOWNLOADED:
  - LSOA-level actual domestic gas + electricity meter readings (2023)
  - Weather-corrected gas; ~33,000 LSOAs in England
  - Output: `temp/statistics/lsoa_energy_consumption.parquet`
  - Joined to UPRN pipeline via `LSOA21CD` in `processing/test_pipeline.py`

- [ ] **Dual DV validation** (SAP-modelled vs DESNZ metered):
  - [ ] Model A: Building-level regression with SAP energy intensity (kWh/m²) as DV
  - [ ] Model B: LSOA-level regression with DESNZ metered mean energy as DV
  - [ ] Aggregate building-level morphology predictors to LSOA means
  - [ ] Coefficient comparison table: same sign? similar magnitude?
  - [ ] If associations attenuate with metered DV → likely SAP model bias
  - [ ] If associations strengthen → SAP compresses true gradient
  - [ ] Report as robustness check in paper methodology
  - [ ] Add validation step to `stats/proof_of_concept.py`
  - [ ] Add validation step to `stats/01_regression_analysis.py`

**Files:**

- `stats/02_multilevel_regression.py` ✅ UPDATED with clarifying notes
- `stats/analysis_report.md` ✅ REWRITTEN with proper framing
- `data/download_energy_stats.py` ✅ CREATED (DESNZ download/parse)
- `data/process_epc.py` ✅ REWRITTEN (disk-backed, expanded columns)
- `paper/literature_review.md` ✅ UPDATED (Section 2.7: EPC performance gap)

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

1. [x] 2.2 Surface-to-volume ratio
2. [x] 2.3 Refine shared wall variable
3. [x] 4.2 Sensitivity analysis (E-value, subsets)
4. [x] 5.1 Log transformation validation
5. [x] 5.4 VIF threshold and PCA

### Week 3: Validation ✅ COMPLETED

1. [x] 6.1 Formal model selection (AIC/BIC)
2. [x] 6.2 Cross-validation
3. [x] 5.2 Mixed model optimization
4. [x] 5.3 Cross-level interactions

### Week 4: Documentation (Partial)

1. [x] 7.1 Clarify accessibility pathways
2. [ ] 8.1 Update paper methodology
3. [ ] 8.2 Code documentation
4. [x] 8.3 Results archive (run_all.py created)
5. [ ] 3.2 Climate stratification (if data available)

---

## New Files Created

| File                                  | Purpose                   | Priority | Status      |
| ------------------------------------- | ------------------------- | -------- | ----------- |
| `stats/00_selection_bias_analysis.py` | IPW, coverage rates       | P1       | ✅ DONE     |
| `stats/02b_spatial_regression.py`     | Moran's I, spatial models | P1       | ✅ DONE     |
| `stats/02c_sensitivity_analysis.py`   | E-value, subset analyses  | P2       | ✅ DONE     |
| `stats/02d_instrumental_variables.py` | IV regression (optional)  | P4       | Not started |
| `stats/02e_cross_validation.py`       | Spatial CV, LOSO          | P2       | ✅ DONE     |
| `stats/run_all.py`                    | Orchestration & reporting | P3       | ✅ DONE     |

---

## Dependencies Added

```bash
uv add esda libpysal  # spatial statistics ✅ INSTALLED
```

---

## Phase 9: Built Form Measures Audit (Literature Review §2.6)

Audit of measures proposed in the literature synthesis against current implementation status.
Maps each measure to its data source, processing stage, and implementation status.

### Tier 1: Core Envelope Physics

#### 9.1 Surface-to-Volume Ratio (S/V) — ✅ IMPLEMENTED

**Status:** Now computed in `processing/process_morphology.py` via `compute_thermal_metrics()`.
Duplicate code removed from `processing/test_pipeline.py` Stage 1 (now validates presence only).

- [x] **Move S/V computation into `process_morphology.py`**
  - `compute_thermal_metrics()` function added after shared wall computation
  - Computes: volume_m3, external_wall_area_m2, envelope_area_m2, surface_to_volume, form_factor
  - Filters: min volume 10m³, cap S/V at 5.0, cap form_factor at 30.0
  - Stored in `buildings_morphology.gpkg` for all downstream uses

- [ ] **Validate S/V against EPC floor area** (consistency check between LiDAR-derived volume and EPC-reported floor area)

**Data sources:** OS building footprints (footprint_area, perimeter) + LiDAR (height) + momepy (shared_wall_ratio)
**Lit reference:** Rode et al. (2014) — primary thermal efficiency metric; Quan & Li (2021) — among most commonly used measures

#### 9.2 Shared Wall Ratio — ✅ IMPLEMENTED

**Status:** Fully implemented in `processing/process_morphology.py` via `momepy.shared_walls()` with 1.5m tolerance for OS cartographic gaps.

- [x] `shared_wall_length_m` — absolute shared wall length
- [x] `shared_wall_ratio` — shared_wall_length / perimeter, clamped [0, 1]
- [x] Used in regression models (stats/01, stats/05, stats/advanced/)

**No further action required.**

#### 9.3 Total Floor Area — ✅ AVAILABLE

**Status:** Available directly from EPC data as `TOTAL_FLOOR_AREA`. Joined to UPRNs in Stage 3 of test_pipeline.py.

- [x] Used as control variable (`log_floor_area`) in all regression models
- [x] Used to compute `energy_intensity` (kWh/m²)

**No further action required.**

#### 9.4 Building Height — ✅ IMPLEMENTED

**Status:** LiDAR-derived in `data/process_lidar.py`. Statistics per building: `height_min`, `height_max`, `height_mean`, `height_median`, `height_std`. Stored in `building_heights.gpkg`.

- [x] Used in test_pipeline.py for S/V computation
- [x] Used in regression as `building_height` (from `height_mean`)

**No further action required.**

#### 9.5 Construction Age Band — ✅ AVAILABLE

**Status:** Available from EPC data as `CONSTRUCTION_AGE_BAND`. Mapped to midpoint year and used as `building_age` and `era` bins.

- [x] Used in all regression models as key control

**No further action required.**

---

### Tier 2: Neighbourhood Context

#### 9.6 Population Density — ✅ AVAILABLE

**Status:** Available from Census 2021 TS006 at Output Area level. Joined to UPRNs via spatial join in Stage 3.

- [x] Used as `pop_density` in regression models
- [x] Used to create `density_quartile` for stratified analyses (stats/03, stats/05)

**No further action required.**

#### 9.7 Network Centrality — ✅ IMPLEMENTED

**Status:** Computed in `processing/test_pipeline.py` Stage 2 via cityseer.
Closeness (beta, harmonic) and betweenness centrality at multiple distances.

- [x] Full mode distances: 800, 1600, 3200, 4800, 9600m
- [x] `cc_harmonic_800` used in regression models (stats/advanced/)
- [x] Joined to UPRNs via nearest-segment lookup

**No further action required.**

#### 9.8 Building Type Composition Within Catchment — ✅ IMPLEMENTED

**Status:** Now computed in `processing/test_pipeline.py` Stage 2 via both approaches.

- [x] **Morphology-derived classification for all OS buildings**
  - `is_detached` (shared_wall_ratio == 0), `is_semi` (0 < ratio < 0.3), `is_terraced` (ratio ≥ 0.3)
  - Aggregated via `compute_stats()` at distances 400, 800, 1600, 4800m
  - Mean values within catchment = proportion (e.g., `cc_is_detached_800_mean`)

- [x] **EPC labels available at UPRN level** for comparison (joined in Stage 3)

**Data sources:** momepy shared_wall_ratio + cityseer network catchments; EPC BUILT_FORM at UPRN level
**Lit reference:** Mediation analysis (stats/02) tests density → building type → energy; this provides continuous catchment-level versions

#### 9.9 Public Transport Accessibility — ✅ IMPLEMENTED

**Status:** Computed in `processing/test_pipeline.py` Stage 2 via cityseer accessibility.

- [x] Bus stop accessibility at 400, 800, 1600, 4800m (from NaPTAN)
- [x] Rail station accessibility at same distances
- [x] Used in exploratory analysis (stats/archive/01, 03)

**No further action required.**

---

### Tier 3: Supplementary Descriptors

#### 9.10 Building Orientation — ✅ IMPLEMENTED

**Status:** Computed in `process_morphology.py` via `momepy.orientation()`. Deviation from cardinal directions (0-45°).

- [x] Available in morphology output
- [ ] **Not yet used in any regression model** — consider adding to exploratory analysis to test independent contribution

**Data sources:** OS building footprints via momepy

#### 9.11 Compactness — ✅ IMPLEMENTED

**Status:** Computed in `process_morphology.py` via `momepy.circular_compactness()`.

- [x] Used in regression models (stats/advanced/cross_validation, sensitivity_analysis, spatial_regression)
- [x] Used in exploratory PCA (stats/archive/01)

**No further action required.**

#### 9.12 Elongation — ✅ IMPLEMENTED

**Status:** Computed in `process_morphology.py` via `momepy.elongation()`.

- [x] Available in morphology output
- [x] Used in exploratory analysis (PCA on morphology variables)

**No further action required.**

#### 9.13 Land Use Accessibility — ✅ IMPLEMENTED

**Status:** Computed in `processing/test_pipeline.py` Stage 2 via cityseer accessibility.

- [x] FSA restaurant, takeaway, pub, other accessibility (400, 800, 1600, 4800m)
- [x] Greenspace accessibility at same distances
- [x] Used in exploratory and SHAP analyses (stats/archive/)

**No further action required.**

#### 9.14 Car Ownership — ✅ IMPLEMENTED

**Status:** Derived from Census TS045 in `stats/05_lockin_analysis.py`.

- [x] `avg_cars_per_hh` — weighted mean cars per household per OA
- [x] `pct_no_car` — percentage of households with no car
- [x] Used to compute `transport_energy_ice` and `transport_energy_ev`
- [x] Stratified by density quartile

**No further action required.**

#### 9.15 Commute Distance — ✅ IMPLEMENTED

**Status:** Derived from Census TS058 in `stats/03_transport_analysis.py`.

- [x] `avg_commute_km` — weighted mean one-way commute distance
- [x] Used in combined building + transport energy analysis

**No further action required.**

---

### Additional Measures from Literature (Not in Current Tier Selection)

These are measures identified in the literature review (particularly Quan & Li 2021 and Rode et al. 2014) that are **not currently computed** and would require new implementation if desired.

#### 9.16 Floor Area Ratio (FAR/FSI) at Catchment Level — ✅ IMPLEMENTED

**Status:** Now computed in `processing/test_pipeline.py` Stage 2.

- [x] **Network-catchment FAR approximation**
  - Floor count estimated from LiDAR height: `estimated_floors = floor(height / 3m)`, min 1
  - `gross_floor_area_m2 = footprint_area × estimated_floors`
  - Aggregated via `compute_stats()` at 400, 800, 1600, 4800m
  - `far_{dist} = sum(gross_floor_area) / (π × dist²)`

**Lit reference:** Rode et al. (2014), SpaceMate framework (Berghauser Pont & Haupt, 2010)

#### 9.17 Building Coverage Ratio (BCR/GSI) at Catchment Level — ✅ IMPLEMENTED

**Status:** Now computed in `processing/test_pipeline.py` Stage 2.

- [x] **Network-catchment BCR**
  - `bcr_{dist} = sum(footprint_area) / (π × dist²)`
  - Computed at 400, 800, 1600, 4800m
  - Catchment area approximated as π×d² (circular)

**Lit reference:** Quan & Li (2021), SpaceMate x-axis variable

#### 9.18 Sky View Factor (SVF) — ❌ NOT IMPLEMENTED

**Status:** Identified by Quan & Li (2021) as commonly used. Requires either LiDAR-based computation or geometric approximation from building heights and spacing.

- [ ] **Compute SVF from LiDAR DSM** (if pursued)
  - Standard hemispherical projection method on 2m DSM
  - Computationally expensive at national scale
  - Could use `scikit-image` or dedicated tools

- [ ] **Alternative: approximate from street canyon geometry**
  - Building height / street width ratio as proxy
  - Heights available from LiDAR; street widths estimable from OS roads

**Complexity:** High (full computation) or Medium (proxy)
**Priority:** Low — Quan & Li note it primarily affects cooling demand; UK is heating-dominated

#### 9.19 Street Canyon Aspect Ratio — ❌ NOT IMPLEMENTED

**Status:** Identified by Quan & Li (2021) as commonly used. Related to SVF.

- [ ] **Estimate from building heights and road widths**
  - Average building height along street edges / road width
  - Heights: available from LiDAR
  - Road widths: not directly available from OS Open Roads (no width attribute); would need estimation from parallel building facades or road class lookup

**Complexity:** High — road width estimation is non-trivial
**Priority:** Low — same rationale as SVF

#### 9.20 Open Space Ratio (OSR) — ❌ NOT IMPLEMENTED

**Status:** Part of SpaceMate framework (Berghauser Pont & Haupt). Defined as (1 - GSI) / FAR.

- [ ] **Computable if BCR and FAR are implemented** (9.16, 9.17)
  - OSR = (1 - BCR) / FAR
  - Captures "pressure on non-built space"

**Complexity:** Low (once FAR and BCR exist)
**Priority:** Low — derivative of other measures

#### 9.21 Land Use Diversity/Mixing Index — ❌ NOT IMPLEMENTED

**Status:** Quan & Li (2021) category 4 (land use). Currently only accessibility distances are computed, not diversity/entropy measures.

- [ ] **Compute Shannon entropy or Simpson diversity** of land uses within catchment
  - Uses FSA categories + greenspace + transport as land use types
  - Measures functional mixing rather than just proximity

**Complexity:** Low-medium — data already available, just needs aggregation logic
**Priority:** Low — exploratory; mechanism linking mixing to building energy is indirect

---

### Summary: Implementation Priority

| Item | Measure | Status | Priority | Effort |
| ---- | ------- | ------ | -------- | ------ |
| 9.1  | S/V ratio in process_morphology.py   | ✅ Done | **High** | Low    |
| 9.8  | Building type composition (catchment) | ✅ Done | **High** | Medium |
| 9.10 | Orientation in regression models      | Unused  | Low      | Low    |
| 9.16 | FAR at catchment level                | ✅ Done | Medium   | Medium |
| 9.17 | BCR/GSI at catchment level            | ✅ Done | Low-Med  | Medium |
| 9.18 | Sky View Factor                       | Missing | Low      | High   |
| 9.19 | Street canyon aspect ratio            | Missing | Low      | High   |
| 9.20 | Open Space Ratio                     | Missing | Low      | Low*   |
| 9.21 | Land use diversity index             | Missing | Low      | Low-Med |

\* OSR is now computable from FAR and BCR: `OSR = (1 - BCR) / FAR`.

**Remaining items** (9.18--9.21) are low priority and can be deferred
unless analysis reveals gaps. 9.20 is trivially derivable from
the now-implemented FAR and BCR.

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
