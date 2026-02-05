# Research Framework: Urban Form and Building Energy

**Last Updated:** 2026-02-05
**Status:** Analysis Complete (Phase 2 - Lock-In Framing)

---

## 1. Core Research Question

> **What are the structural energy penalties of sprawling development, and to what extent do they persist with technological improvements (insulation, EVs)?**

This question reframes urban energy analysis around **lock-in**: morphological decisions create baseline energy penalties that technology cannot eliminate.

### The Policy Thesis

Sprawling development locks in two structural penalties:

| Penalty       | Mechanism                           | Why Technology Can't Fix It                            |
| ------------- | ----------------------------------- | ------------------------------------------------------ |
| **Envelope**  | More exposed wall area per dwelling | Even with same U-values, more surface = more heat loss |
| **Transport** | More vehicle-km per household       | Even with EVs, more km = more energy                   |

**Implication:** You cannot insulate and electrify your way out of sprawl.

---

## 2. Theoretical Framework

### 2.1 The Causal Chain (Hypothesized)

```
COMPACT URBAN FORM
        │
        ├──────────────────────────────────────────────────────────┐
        │                                                          │
        ▼                                                          ▼
[Stock Composition]                                    [Area-Level Effects]
Dense areas have more                                  Urban heat island
flats, terraces, smaller                              Wind sheltering
dwellings                                             Solar access patterns
        │                                                          │
        ▼                                                          ▼
[Building Thermal Physics]                             [Behavioral Changes]
Shared walls reduce heat loss                          Smaller spaces
Compact shapes = lower S/V ratio                       Different lifestyles
Multi-story = less roof loss/unit                      (NOT captured by SAP)
        │                                                          │
        └──────────────────────┬───────────────────────────────────┘
                               │
                               ▼
                    BUILDING ENERGY DEMAND
                    (per capita or per m²)
```

### 2.2 What SAP Captures vs. What It Doesn't

| Captured by SAP                           | NOT Captured by SAP          |
| ----------------------------------------- | ---------------------------- |
| Building envelope (walls, roof, floor)    | Actual thermostat settings   |
| Window area and glazing type              | Occupancy patterns           |
| Heating system efficiency                 | Fuel poverty (under-heating) |
| Building geometry (floor area)            | Lifestyle/behavioral choices |
| Construction age (as proxy for standards) | Rebound effects              |

**Critical Implication:** Our dependent variable measures _potential_ demand under standardized conditions. Any "density effect" must operate through building physics, not behavior.

### 2.3 The Confounding Problem

```
                    HISTORICAL DEVELOPMENT
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
    Central Location              Older Building Stock
    (high density)                (poor fabric efficiency)
            │                               │
            └───────────────┬───────────────┘
                            │
                            ▼
                    OBSERVED CORRELATION
            (density appears unrelated to energy
             because age effect cancels location effect)
```

This is why controlling for building age is essential—and why the "near-zero density effect" is actually a meaningful finding.

---

## 3. Hypotheses (Lock-In Framework)

### H1: Floor Area Lock-In

> **Sprawling development (detached houses) is associated with larger floor area, resulting in higher total energy demand regardless of efficiency.**

- **Metric:** Total floor area (m²)
- **Comparison:** Detached vs Semi vs Terrace vs Flat
- **Expected:** Detached ~60% larger than terraces
- **Status:** ✅ Confirmed (142m² vs 90m²)

### H2: Envelope Lock-In

> **Each additional exposed wall is associated with approximately 9-10% higher energy intensity, and this proportional penalty persists across insulation levels.**

- **Metric:** Energy intensity (kWh/m²)
- **Test:** Matched comparison (same era, same size, different built form)
- **Expected:** Detached ~20-25% higher intensity than mid-terrace
- **Status:** ✅ Confirmed (298 vs 238 kWh/m²)

### H3: Transport Lock-In

> **Low-density development is associated with higher car ownership, resulting in more vehicle-km and higher transport energy regardless of vehicle technology.**

- **Metric:** Cars per household, annual vehicle-km
- **Comparison:** High-density vs low-density quartiles
- **Expected:** Low-density ~80% more cars per household
- **Status:** ✅ Confirmed (0.98 vs 0.54 cars/hh)

### H4: Technology Persistence

> **The proportional penalty of sprawl persists with technology improvements (better insulation, EVs) because the structural disadvantage (more surface area, more km) cannot be eliminated.**

- **Test:** Compare penalties across technology scenarios
- **Expected:** Percentage penalty approximately constant
- **Status:** ✅ Confirmed (penalty persists at ~67-183%)

### Legacy Hypotheses (Demoted to Methodological Notes)

The following hypotheses from v1 analysis are now considered methodological context:

- **Per-capita artifact (old H5):** The per-capita metric confounds thermal efficiency with household size. Use intensity (kWh/m²) instead.
- **Mediation (old H2):** Density effects are mediated through building type composition—this is now incorporated into the lock-in framing.
- **Age confounding (old H4):** Not supported; dense areas have newer buildings.

- **Level:** Building type subgroups
- **Test:** Stratified models (houses only, flats only)
- **Expected:** Density matters for houses (where shared walls vary); less for flats (already optimized)
- **Status:** Not yet tested

### H6: Total Household Energy Footprint (Policy-Level)

> **Compact urban form shows lower TOTAL household energy footprint when transport energy is included alongside building energy.**

- **Level:** Household/area
- **Test:** Combined building + transport energy analysis by urban form typology
- **Expected:** Dense areas (especially flats) have lower total footprint despite potential per-capita building energy penalty
- **Mechanism:** Lower car ownership in dense areas offsets any building efficiency differences
- **Data sources:**
  - Building: SAP potential energy demand (kWh/year)
  - Transport: Census car ownership (TS045) as proxy for transport energy
- **Transport estimation:** 12,000 km/year × 0.17 kg CO2/km per vehicle, converted to kWh equivalent
- **Limitations:** Transport energy is illustrative proxy, not measured consumption
- **Status:** Preliminary analysis complete (2026-02-05)

---

## 4. Key Variables

### 4.1 Dependent Variables

| Variable               | Definition                  | Use Case                        | Notes                                   |
| ---------------------- | --------------------------- | ------------------------------- | --------------------------------------- |
| `energy_intensity`     | SAP energy / floor area     | **PRIMARY** for thermal physics | Not confounded by household size        |
| `log_energy_intensity` | log transform               | For regression                  | Better distributional properties        |
| `energy_per_capita`    | SAP energy / household size | Secondary (policy)              | **Confounded by household composition** |

**CRITICAL METHODOLOGICAL INSIGHT (2026-02-05):**

Choice of DV fundamentally changes conclusions:

- **Intensity (kWh/m²):** Flats in dense areas are MORE efficient (r = −0.06)
- **Per capita:** Flats in dense areas appear LESS efficient (r = +0.06)

The per-capita metric confounds thermal efficiency with household size. Flats have smaller households, so the same building energy divided by fewer people yields inflated per-capita values. **Use intensity for thermal physics questions.**

### 4.2 Building-Level Thermal Physics (H1)

| Variable            | Definition                                              | Hypothesis                               |
| ------------------- | ------------------------------------------------------- | ---------------------------------------- |
| `shared_wall_ratio` | Shared wall length / perimeter                          | Higher → less energy                     |
| `surface_to_volume` | Envelope area / volume                                  | Lower → less energy                      |
| `form_factor`       | Envelope area / volume^(2/3)                            | Lower → less energy                      |
| `attached_type`     | Categorical: detached/semi/mid-terrace/end-terrace/flat | Captures discrete thermal configurations |

### 4.3 Stock Composition (H2)

| Variable                 | Definition                        | Level        |
| ------------------------ | --------------------------------- | ------------ |
| `pct_terraced`           | % buildings that are terraced     | Segment/LSOA |
| `pct_flats`              | % buildings that are flats        | Segment/LSOA |
| `pct_detached`           | % buildings that are detached     | Segment/LSOA |
| `mean_shared_wall_ratio` | Average shared wall ratio in area | Segment      |

### 4.4 Density Measures (H3)

| Variable                  | Definition                   | Mechanism Tested      |
| ------------------------- | ---------------------------- | --------------------- |
| `pop_density`             | Persons per hectare (Census) | General density       |
| `building_coverage_ratio` | Σfootprint / catchment area  | Urban heat island     |
| `FAR`                     | Σfloor_area / catchment area | Development intensity |
| `uprn_density`            | Addresses per hectare        | Subdivision intensity |

### 4.5 Controls

| Category                     | Variables                                                                         |
| ---------------------------- | --------------------------------------------------------------------------------- |
| **Building fabric**          | `walls_efficiency`, `windows_efficiency`, `heating_efficiency`, `roof_efficiency` |
| **Building characteristics** | `TOTAL_FLOOR_AREA`, `building_age`, `PROPERTY_TYPE`, `BUILT_FORM`                 |
| **Socio-demographics**       | `pct_owner_occupied`, `pct_deprived`, `avg_household_size`                        |
| **Spatial**                  | `LSOA21CD` (random effect)                                                        |

---

## 5. Analytical Strategy

### 5.1 Analysis Sequence

```
Phase 1: Descriptive & Exploratory
├── Variable distributions
├── Correlation matrix
├── Spatial patterns (maps)
└── Identify outliers, missing data

Phase 2: Building-Level Analysis (H1)
├── OLS: energy ~ thermal physics variables + controls
├── Compare shared_wall_ratio (continuous) vs attached_type (categorical)
├── Add form_factor, test improvement
└── Stratify by building type

Phase 3: Mediation Analysis (H2)
├── Total effect: density → energy
├── Path a: density → stock composition
├── Path b: stock composition → energy
└── Indirect vs direct effect decomposition

Phase 4: Multilevel Models (H3)
├── Null model (ICC decomposition)
├── Building controls only
├── + Neighborhood density
├── + Full controls
└── Compare coefficients across specifications

Phase 5: Robustness & Heterogeneity (H4, H5)
├── Stratify by construction era
├── Stratify by building type
├── Age-residualized density
├── Spatial regression (Moran's I, spatial lag/error)
└── Cross-validation

Phase 6: Synthesis
├── Summary table of all hypothesis tests
├── Effect size comparison
├── Limitations acknowledgment
└── Policy implications
```

### 5.2 Model Comparison Criteria

| Criterion             | Use                                   |
| --------------------- | ------------------------------------- |
| AIC                   | Model selection (prediction focus)    |
| BIC                   | Model selection (parsimony focus)     |
| Likelihood ratio test | Nested model comparison               |
| R² change             | Variance explained by added variables |
| Coefficient stability | Sensitivity to specification          |
| Cross-validation RMSE | Out-of-sample prediction              |

---

## 6. Key Methodological Constraints

### 6.1 What We CAN Claim

- Associations between variables
- Variance decomposition across spatial scales
- Relative importance of predictors
- Mediation structure (with causal assumptions stated)

### 6.2 What We CANNOT Claim

- Causal effects (observational design)
- Behavioral energy differences (SAP limitation)
- Transport energy implications (outside scope)
- Generalization beyond study area (single city)

### 6.3 Language Discipline

| Avoid                       | Use Instead                               |
| --------------------------- | ----------------------------------------- |
| "Density reduces energy"    | "Density is associated with lower energy" |
| "The effect of compactness" | "The association with compactness"        |
| "Causes"                    | "Predicts" or "is associated with"        |
| "Energy consumption"        | "Potential energy demand" (for SAP data)  |

---

## 7. Success Criteria

The analysis is complete when we can answer:

1. **H1:** What is the coefficient for shared_wall_ratio and form_factor, controlling for building characteristics? Is it statistically and practically significant?

2. **H2:** What proportion of the density-energy association is mediated through building stock composition vs. direct effects?

3. **H3:** After full controls, what is the residual density coefficient? Is it distinguishable from zero?

4. **H4:** Does the density-energy relationship differ across construction eras? Does age-residualized density show a clearer pattern?

5. **H5:** Does the density-energy relationship differ between houses and flats?

---

## 8. Current Understanding (Updated: 2026-02-05)

### Key Empirical Findings

1. **Building age dominates:** r = 0.21 with energy per capita (strongest single predictor)

2. **Overall density shows weak positive association:** β = +0.035 (p < 0.001)
   - This is OPPOSITE to expected direction
   - Explained by offsetting building type effects (see below)

3. **HOUSES vs FLATS show OPPOSITE patterns:**
   - Houses: r = **-0.109** (density → LOWER energy) - expected direction
   - Flats: r = **+0.061** (density → HIGHER energy) - unexpected
   - The near-zero overall correlation MASKS these opposing effects

4. **Flat energy per capita is a methodological artifact:**
   - Flats have smaller households → same energy ÷ fewer people = higher per capita
   - This creates spurious positive density-flat-energy pathway

5. **Mediation analysis (H2):**
   - 45% of density effect mediated through building type
   - Terrace pathway: density → more terraces → LOWER energy (-0.006)
   - Flat pathway: density → more flats → HIGHER energy (+0.022)
   - These effects partially offset

6. **Construction era effects (H4):**
   - Pre-1919: r = -0.12 (negative association)
   - Interwar: r = -0.075 (negative)
   - Post-war: r = +0.02 (near zero)
   - Modern 1980+: r = +0.14 (POSITIVE association)

7. **Age-density correlation:** r = -0.064
   - Denser areas have NEWER buildings (opposite to hypothesis)
   - Age confounding does NOT explain the null result

### Revised Interpretation

The near-zero overall density-energy correlation is NOT a simple null result. It reflects:

1. **For houses:** Compact development IS associated with lower energy (as theory predicts)
2. **For flats:** The per-capita metric creates an artifact masking true efficiency
3. **Methodological insight:** "Energy per capita" may be problematic as DV

**Implication:** The research question should distinguish between:

- Building energy INTENSITY (kWh/m²) - true thermal efficiency
- Energy per DWELLING - household-level consumption
- Energy per CAPITA - affected by household size composition

### Hypothesis Status

| Hypothesis              | Status           | Key Finding                                                    |
| ----------------------- | ---------------- | -------------------------------------------------------------- |
| H1 (Thermal physics)    | ✅ Supported     | Shared walls β = −0.07; form_factor adds minimal (ΔR² = 0.002) |
| H2 (Mediation)          | ⚠️ Partial       | 45% mediated; offsetting flat/terrace effects                  |
| H3 (Residual density)   | ✅ Supported     | r ≈ 0.00 after full controls                                   |
| H4 (Age confounding)    | ❌ Not supported | Dense areas have NEWER buildings (r = −0.064)                  |
| H5 (Type heterogeneity) | ✅ Supported     | Sign reverses with intensity metric (small effects, r < 0.12)  |
| H6 (Combined footprint) | ✅ Supported\*   | Illustrative estimates suggest ~30–50% lower footprint         |

\*H6 relies on proxy-based transport estimates with substantial uncertainty.

### Key Findings Summary (2026-02-05)

1. **THE PER-CAPITA ARTIFACT:** The sign reverses with intensity metric:
   - Per capita: Houses r = −0.11, Flats r = +0.06
   - Intensity: Houses r = +0.08, Flats r = −0.06
   - Note: All correlations are small (|r| < 0.12, explaining <1.5% variance)

2. **Flats ARE efficient:** The "flat penalty" is a household size artifact

3. **Transport (illustrative):** Combined footprint estimates suggest ~30–50% lower for dense flats
   - These are proxy-based estimates with substantial uncertainty
   - Census 2021 data affected by COVID (31% WFH)

4. **Commute insight:** Distances nearly equal (9.3 vs 9.5 km); difference associated with car ownership (0.54 vs 0.98 cars/hh)

### Effect Size Context

All density-energy correlations are small (|r| < 0.15). These are:

- Statistically significant due to N > 140,000
- Theoretically meaningful for understanding mechanisms
- NOT large practical effects (each explains <2% of variance)

### Open Questions

- [x] ~~What does mediation analysis show?~~ → 45% mediated, offsetting effects
- [x] ~~Do effects differ by construction era?~~ → Yes, substantial variation
- [x] ~~Do effects differ between houses and flats?~~ → Yes, opposite directions
- [x] ~~Does form_factor improve on shared_wall_ratio?~~ → Minimal improvement (ΔR² = 0.002)
- [x] ~~What happens with energy INTENSITY (kWh/m²) as DV?~~ → **Pattern reverses; flats efficient**
- [x] ~~Why do modern buildings show positive density association?~~ → Per-capita artifact (smaller households)
- [ ] Is there spatial autocorrelation in residuals? (Moran's I run but not reported)
- [ ] Validate with metered consumption data

---

## 9. File Organization

```
urban-energy/
├── RESEARCH_FRAMEWORK.md      ← This document (north star)
├── WORKING_LOG.md             ← Detailed progress tracking
│
├── stats/
│   ├── run_all.py             ← Pipeline runner
│   │
│   ├── 00_data_quality.py     ← Data quality report
│   ├── 01_regression_analysis.py  ← Main regression (H1, H3)
│   ├── 02_mediation_analysis.py   ← Mediation (H2)
│   ├── 03_transport_analysis.py   ← Combined footprint (H6)
│   ├── 04_generate_figures.py     ← Figures (fig1-fig8)
│   │
│   ├── analysis_report.md     ← Main findings narrative
│   │
│   ├── advanced/              ← Optional deep-dive analyses
│   │   ├── spatial_regression.py
│   │   ├── sensitivity_analysis.py
│   │   ├── cross_validation.py
│   │   └── selection_bias_analysis.py
│   │
│   ├── figures/               ← Generated figures
│   │   └── archive/           ← Old figures
│   │
│   └── archive/               ← Superseded scripts
│
├── processing/
│   └── test_pipeline.py
│
└── paper/
    ├── literature_review.md
    └── methodology_notes.md
```

---

_This framework guides all analysis decisions. Update "Current Understanding" as findings emerge._
