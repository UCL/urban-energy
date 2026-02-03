# Statistical Analysis

Research design and analysis plan for quantifying morphology-energy relationships.

## Research Questions

1. **Primary:** Do urban morphological characteristics (density, compactness, connectivity) correlate with building energy consumption after controlling for building fabric and socio-economics?

2. **Secondary:**
   - Which morphological features have the largest effect sizes?
   - At what spatial scale are associations strongest (building, 400m, LSOA)?
   - Do relationships differ by building type (houses vs flats)?

---

## Analytical Framework

### Units of Analysis

| Level | Unit | N (approx) | Role |
|-------|------|------------|------|
| 1 | UPRN (property) | ~15M with EPC | Observation unit |
| 2 | Building | ~8M | Morphology computed here |
| 3 | Street segment | ~500K | Neighbourhood aggregation |
| 4 | LSOA | ~35K | Socio-demographic controls |

**Nesting structure:** UPRNs → Buildings → Segments → LSOAs

### Variables

#### Dependent Variable

| Variable | Source | Notes |
|----------|--------|-------|
| `energy_consumption_kwh` | EPC | SAP-modelled annual consumption |
| `energy_intensity_kwh_m2` | EPC | Consumption / floor area |
| `energy_efficiency_sap` | EPC | SAP rating (1-100) |

**Preferred:** `energy_intensity_kwh_m2` — normalises for dwelling size, focuses on efficiency.

#### Independent Variables (Morphology)

**Building-level** (inherited by UPRN):

| Variable | Description | Hypothesis |
|----------|-------------|------------|
| `height_median` | LiDAR building height | Taller = more wind exposure but less roof loss per unit |
| `shared_wall_ratio` | Party wall proportion | Higher = less heat loss |
| `compactness` | Circular compactness | Higher = less surface/volume |
| `elongation` | Shape elongation | Higher = more surface/volume |
| `orientation` | Cardinal deviation | Affects passive solar |
| `convexity` | Envelope complexity | Lower = more thermal bridges |

**Neighbourhood-level** (400m catchment aggregates):

| Variable | Aggregation | Hypothesis |
|----------|-------------|------------|
| `building_density` | count / ha | Higher = urban heat island, shelter |
| `far` | Σ floor area / catchment area | Development intensity |
| `mean_height` | mean(height_median) | Shelter effects |
| `mean_shared_wall_ratio` | mean(shared_wall_ratio) | Terraced prevalence |
| `pct_terraced` | % buildings with ratio > 0.3 | Housing stock mix |
| `pct_detached` | % buildings with ratio = 0 | Suburban character |

#### Control Variables

**Building characteristics** (EPC):

| Variable | Role |
|----------|------|
| `total_floor_area` | Size control |
| `property_type` | House/Flat/Bungalow |
| `built_form` | Detached/Semi/Terrace/Flat position |
| `construction_age_band` | Vintage (insulation proxy) |
| `main_fuel` | Gas/Electric/Oil |
| `walls_description` | Solid/Cavity/Insulated |

**Socio-demographics** (Census → LSOA):

| Variable | Role |
|----------|------|
| `imd_decile` | Deprivation (income, investment) |
| `tenure_owned_pct` | Ownership (maintenance incentive) |
| `household_size_mean` | Occupancy intensity |
| `age_65plus_pct` | Heating needs |

---

## Statistical Approach

### Primary Model: Multi-level Regression

**Rationale:** Data has hierarchical structure. Ignoring this leads to:
- Underestimated standard errors (clustered observations)
- Inability to partition variance across scales
- Ecological fallacy if only using aggregate data

**Model specification:**

```
Level 1 (UPRN):
  energy_ij = β₀j + β₁(floor_area) + β₂(age) + β₃(building_type) + ... + ε_ij

Level 2 (Segment):
  β₀j = γ₀₀ + γ₀₁(building_density) + γ₀₂(mean_height) + γ₀₃(far) + ... + u₀j

Level 3 (LSOA):
  γ₀₀ = δ₀₀₀ + δ₀₀₁(imd_decile) + δ₀₀₂(tenure) + ... + v₀₀k
```

**Key outputs:**
- Intraclass correlation (ICC) at each level — how much variance is between vs within clusters
- Fixed effects for morphology variables — effect sizes
- Random intercepts — unexplained spatial variation

### Alternative/Complementary Approaches

#### 1. Spatial Regression (Spatial Lag / Spatial Error)

**When:** Significant spatial autocorrelation in residuals (Moran's I test)

**Models:**
- **Spatial lag:** Neighbours' energy use affects own energy use (behavioural spillovers)
- **Spatial error:** Unobserved spatially-correlated factors

**Implementation:** `pysal` / `spreg`

#### 2. Propensity Score Matching

**Purpose:** Causal inference approximation

**Approach:**
1. Define "treatment" = high-density neighbourhood
2. Match high-density properties to similar low-density properties on building characteristics
3. Compare energy use between matched pairs

**Advantage:** Reduces selection bias (e.g., flats cluster in dense areas)

#### 3. Instrumental Variables

**Challenge:** Endogeneity — building type and location are jointly determined by historical/economic factors that also affect energy use

**Potential instruments:**
- Historical street layout (pre-automobile grid patterns)
- Distance to Victorian-era rail stations
- Bomb damage patterns (WWII reconstruction areas)

**Validity requirements:** Instrument affects morphology but only affects energy through morphology

#### 4. SHAP Values (Shapley Additive Explanations)

**Purpose:** Model-agnostic feature importance with theoretical guarantees

**Background:** Shapley values originate from cooperative game theory. They fairly distribute a "payout" (prediction) among "players" (features) based on their marginal contributions across all possible coalitions.

**Why SHAP for this research:**

| Advantage | Relevance |
| --- | --- |
| **Handles multicollinearity** | Morphology variables are correlated — Shapley values fairly attribute even with collinearity |
| **Captures interactions** | Building type × neighbourhood context interactions emerge naturally |
| **Local + global** | Can explain individual predictions AND aggregate to feature importance |
| **Model-agnostic** | Works with gradient boosting, random forests, or any model |

**Approach:**

```
1. Fit gradient boosting model (XGBoost/LightGBM)
   - Target: energy_intensity_kwh_m2
   - Features: building + morphology + socio-demographics

2. Compute SHAP values
   - TreeSHAP for tree-based models (fast, exact)
   - Sample if dataset too large (~100K observations sufficient)

3. Analyse contributions
   - Global feature importance (mean |SHAP|)
   - Dependence plots (SHAP vs feature value)
   - Interaction effects (SHAP interaction values)
```

**Key outputs:**

| Output | Interpretation |
|--------|----------------|
| **Summary plot** | Rank features by importance, show direction of effects |
| **Dependence plots** | How energy changes as density/height/compactness varies |
| **Interaction plots** | Does density effect depend on building type? |
| **Force plots** | Why does this specific property have high/low energy? |

**Comparison with regression:**

| Aspect | Multi-level Regression | SHAP |
|--------|------------------------|------|
| **Interpretation** | Coefficients (marginal effects) | Contributions (fair attribution) |
| **Inference** | p-values, confidence intervals | Descriptive importance |
| **Linearity** | Assumes (or transforms) | Captures non-linear effects |
| **Interactions** | Must specify explicitly | Detected automatically |
| **Multicollinearity** | Inflates SEs, unstable coefficients | Handles naturally |
| **Theory** | Statistical inference | Game-theoretic fairness |

**Recommendation:** Use SHAP as complementary to regression:

- SHAP for exploratory importance and detecting non-linearities
- Regression for hypothesis testing with proper controls and inference

**Implementation:** `shap` package with `xgboost` or `lightgbm`

```python
import shap
import xgboost as xgb

# Fit model
model = xgb.XGBRegressor(n_estimators=500, max_depth=6)
model.fit(X_train, y_train)

# Compute SHAP values (TreeSHAP is fast)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Visualise
shap.summary_plot(shap_values, X_sample)
shap.dependence_plot("building_density", shap_values, X_sample)
```

**Computational considerations:**

- Dataset: ~15M UPRNs with EPCs
- TreeSHAP: O(TLD²) per prediction (T=trees, L=leaves, D=depth)
- Strategy: Sample ~100K observations stratified by region/building type
- Or: Aggregate to segment level (~500K) for full analysis

---

## Implementation Plan

### Phase 1: Data Preparation

```
[ ] Complete morphology processing (running)
[ ] Build UPRN integrated dataset (process_uprn.py)
[ ] Compute segment-level aggregations (process_network.py)
[ ] Create analysis-ready table with all variables
```

### Phase 2: Exploratory Analysis

```
[ ] Descriptive statistics by building type, region
[ ] Correlation matrix (morphology vs energy)
[ ] Spatial distribution maps
[ ] Check for multicollinearity (VIF)
```

### Phase 3: Model Building

```
[ ] Null model (empty multi-level) — partition variance
[ ] Add building-level controls — baseline
[ ] Add neighbourhood morphology — key test
[ ] Add socio-demographic controls — robustness
[ ] Test interactions (e.g., morphology × building type)
```

### Phase 4: Robustness Checks

```
[ ] Different catchment sizes (200m, 400m, 800m)
[ ] Subset by property type (houses only, flats only)
[ ] Spatial regression comparison
[ ] Cross-validation (train/test split by LSOA)
```

---

## Key Methodological Considerations

### 1. Selection Bias in EPCs

**Issue:** EPCs are only required at sale/rental — biased toward transacted properties

**Mitigation:**
- Acknowledge limitation
- Compare EPC coverage rates across morphology types
- Weight by inverse probability of having EPC (if predictable)

### 2. SAP vs Metered Consumption

**Issue:** EPC energy is modelled (SAP algorithm), not metered

**Implications:**
- SAP uses standardised occupancy assumptions
- Doesn't capture behavioural differences
- May systematically over/underestimate by building type

**Mitigation:**
- Treat as "potential consumption under standard conditions"
- Compare SAP assumptions to actual behaviour where possible
- Note that morphology effects on SAP are still policy-relevant (building regulations)

### 3. Ecological Fallacy

**Issue:** Relationships at aggregate level may not hold at individual level

**Mitigation:**
- Use multi-level models that explicitly partition effects
- Report ICC to show how much variation is within vs between areas
- Interpret neighbourhood effects as contextual, not individual

### 4. Endogeneity

**Issue:** Unobserved factors affect both morphology and energy use

**Examples:**
- Wealth → larger detached houses + higher energy use
- Building age → solid walls + poor insulation + terraced form

**Mitigation:**
- Control for building characteristics and socio-demographics
- Sensitivity analysis with different control sets
- Acknowledge causal claims are limited

---

## Software Stack

| Task | Package |
|------|---------|
| Data manipulation | `pandas`, `geopandas` |
| Multi-level models | `statsmodels` (MixedLM), `pymer4` |
| Spatial regression | `pysal`, `spreg` |
| Visualisation | `matplotlib`, `seaborn`, `folium` |
| Diagnostics | `scipy`, `statsmodels` |

---

## Expected Outputs

1. **Variance decomposition:** What % of energy variation is at building vs neighbourhood vs LSOA level?

2. **Effect sizes:** Standardised coefficients for morphology variables

3. **Maps:** Spatial distribution of:
   - Energy intensity
   - Morphological variables
   - Model residuals (unexplained variation)

4. **Tables:** Regression results with progressive model building

5. **Robustness:** Sensitivity to scale, property type, model specification

---

## References

Key methodological references to review:

- Ewing & Rong (2008) — residential energy and urban form
- Steadman et al. (2014) — UK building stock modelling
- Rode et al. (2014) — cities and energy, morphology effects
- Wilson (2013) — spatial regression in urban energy
- Goldstein (2011) — multilevel statistical models
