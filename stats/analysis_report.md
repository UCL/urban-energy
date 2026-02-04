# Does Compact Urban Development Reduce Energy Use?

## Controlling for Building Age and Other Confounders

**Generated:** 2026-02-04 15:30

**Dataset:** 173846 properties with EPC data from test area

---

## Research Question

> **After controlling for building age and other confounders, is there a relationship between compact urban development and energy use per capita?**

---

## Executive Summary

| Metric | Raw Correlation | After Controlling for Age |
|--------|-----------------|---------------------------|
| Network Centrality | r = +0.00 | r = -0.02 |
| Population Density | r = +0.00 | r = +0.06 |
| Building Age | r = +0.21 | (control variable) |

**Summary:** This table compares raw correlations with partial correlations after controlling for building age.

- Raw correlations between centrality/density and energy are near zero
- After controlling for building age, correlations remain small
- Building age (r = 0.21) shows the strongest association with energy per capita

### Sample Statistics

| Metric | Value |
|--------|-------|
| Sample Size | 173846 properties |
| Mean Energy per Capita | 96 kWh/person/year |
| Mean Building Age | 66 years |

---

## 1. The Key Result: Controlling for Building Age

![Controlled Effects](stats/08_controlled_effects.png)

### What This Figure Shows

- **Panels A & C (left)**: Raw correlations between urban form variables (centrality, density) and energy per capita
- **Panels B & D (right)**: Partial correlations after regressing out building age from both variables
- **Method**: OLS residuals used to remove linear effect of building age before computing correlations

---

## 2. Building Age and Energy

![Building Age](stats/04_building_age_effect.png)

Building age has a correlation of **r = 0.21** with energy per capita.

- Older buildings tend to have lower thermal efficiency (insulation, glazing, heating systems)
- Central urban areas often have older building stock
- Building age may correlate with both urban form and energy consumption

---

## 3. Relationship Between Centrality and Age

![Centrality vs Age](stats/07_centrality_age_confounding.png)

**Left panel**: Scatter plot of network centrality vs building age

**Right panel**: Energy vs centrality, stratified by building age category

---

## 4. Other Variables

![Other Confounders](stats/09_other_confounders.png)

Additional relationships between variables:
- **Property type**: Distribution of building age by property type
- **Built form**: Distribution of building age by built form
- **Tenure**: Energy by tenure category
- **Bottom-right**: Correlation matrix of key variables

---

## 5. Built Form: Raw vs Controlled

![Built Form Controlled](stats/03_built_form_controlled.png)

### Description

- **Panel A (Raw)**: Mean energy per capita by built form category
- **Panel B (Controlled)**: Mean age-adjusted residuals by built form category
  - Residuals computed by regressing energy per capita on building age
  - Positive residuals indicate higher-than-expected energy given building age
  - Negative residuals indicate lower-than-expected energy given building age

### Raw Built Form Distribution

![Built Form Raw](stats/02_energy_by_built_form.png)

---

## 6. Machine Learning Feature Importance (SHAP)

![SHAP Importance](stats/10_shap_importance.png)

SHAP feature importance from Gradient Boosting model.

![SHAP Summary](stats/11_shap_summary.png)

**Reading this plot**: Each dot is one observation. Red = high feature value, Blue = low.
Position on x-axis shows impact on prediction.

---

## 7. Supporting Figures

### Energy Distribution

![Energy Distribution](stats/01_energy_distribution.png)

Distribution of energy per capita by property type.
Flats: 106 kWh mean, Houses: 87 kWh mean.

### Density (Raw)

![Density Effect](stats/05_density_effect.png)

Raw correlation between population density and energy per capita.

### Correlation Matrix

![Correlation Heatmap](stats/06_correlation_heatmap.png)

### SHAP Dependence

![SHAP Age Dependence](stats/12_shap_dependence_age.png)

SHAP dependence plot for building age.

---

## Summary of Results

### Correlations by Variable

| Variable | Raw Correlation | After Controlling for Age |
|----------|-----------------|---------------------------|
| **Shared Walls (Built Form)** | See Figure 5 | Terraces show lower residuals |
| **Building Height** | Negative | Negative |
| **Network Centrality** | r = +0.00 | r = -0.02 |
| **Population Density** | r = +0.00 | r = +0.06 |
| **Building Age** | r = +0.21 | (control variable) |

### Observations

1. Building age shows the strongest correlation with energy per capita (r = 0.21)
2. Network centrality and population density show near-zero correlations with energy
3. After controlling for building age, correlations remain small
4. This analysis covers building energy only; transport energy is not included

### Limitations

- Sample: 173846 properties
- Single study area may not generalize
- EPC energy excludes transport/behaviour
- Multi-level models may provide additional insights

---

## Appendix: Data Sources

| Source | Variables |
|--------|-----------|
| Energy Performance Certificates | Energy consumption, floor area, building age |
| Census 2021 | Household size, tenure, travel to work |
| OS Open Map Local | Building footprints |
| Environment Agency LiDAR | Building heights |
| cityseer | Network centrality, accessibility metrics |

---

*Report generated automatically by stats/04_visualizations.py*
