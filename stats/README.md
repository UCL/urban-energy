# Statistical Analysis

Research design and methodology for quantifying the structural energy penalties of urban sprawl.

**Scope:** Domestic (residential) buildings only.

For key findings, see the [analysis report](analysis_report_v3.md).

---

## Prerequisites

Complete the data pipeline before running analysis:

```text
1. DATA ACQUISITION    → data/README.md
2. PROCESSING          → processing/README.md
3. STATISTICAL ANALYSIS → (this document)
```

**Required input:** Processed EPC data with Census linkage

---

## Running the Analysis

```bash
# Run complete analysis pipeline
uv run python stats/run_all.py

# Or run individual scripts
uv run python stats/05_lockin_analysis.py  # Core lock-in analysis
uv run python stats/06_generate_report.py  # Generate report from results
```

### Scripts

| Script                      | Purpose                                     | Output                                   |
| --------------------------- | ------------------------------------------- | ---------------------------------------- |
| `00_data_quality.py`        | Data quality assessment, sample description | `temp/stats/data_quality/`               |
| `01_regression_analysis.py` | Regression models, effect decomposition     | Coefficients, variance decomposition     |
| `02_mediation_analysis.py`  | Mediation through building type composition | Direct/indirect effect estimates         |
| `03_transport_analysis.py`  | Transport energy estimation from Census     | Combined building + transport footprints |
| `04_generate_figures.py`    | Publication-ready figures                   | `stats/figures/`                         |
| `05_lockin_analysis.py`     | **Core lock-in quantification**             | `temp/stats/results/lockin_summary.json` |
| `06_generate_report.py`     | Generate narrative report from JSON         | `stats/analysis_report_v3.md`            |
| `run_all.py`                | Pipeline orchestrator                       | Runs all scripts in sequence             |

### Output

| File                                     | Description                                    |
| ---------------------------------------- | ---------------------------------------------- |
| `stats/analysis_report_v3.md`            | Main findings narrative (generated)            |
| `temp/stats/results/lockin_summary.json` | Machine-readable results for report generation |
| `stats/figures/*.png`                    | Publication figures                            |

---

## Research Hypotheses

### H1: Floor Area Lock-In

> Sprawling development (detached houses) is associated with larger floor area, resulting in higher total energy demand regardless of efficiency.

- **Metric:** Total floor area (m²)
- **Expected:** Detached ~60% larger than terraces

### H2: Envelope Lock-In

> Detached houses have higher energy intensity per square metre than attached dwellings due to greater exposed wall area. This proportional penalty cannot be eliminated through insulation improvements.

- **Metric:** Energy intensity (kWh/m²) in matched samples
- **Expected:** Detached higher intensity than mid-terrace (controlling for age, size)

### H3: Transport Lock-In

> Low-density development is associated with higher car ownership, resulting in more vehicle-km and higher transport energy regardless of vehicle technology.

- **Metric:** Cars per household (Census)
- **Expected:** Low-density ~20% more cars per household

### H4: Technology Persistence

> The proportional penalty of sprawl persists with technology improvements because the structural disadvantage (more surface area, more km) cannot be eliminated.

- **Test:** Compare penalties across insulation and vehicle scenarios
- **Expected:** Percentage penalty approximately constant

---

## Analytical Approach

### Primary Method: Matched Comparison

The core analysis uses **matched comparisons** to isolate the shared-wall effect from confounders:

**Problem:** Raw data shows detached houses with _lower_ energy intensity than terraces because detached homes tend to be newer (better insulation).

**Solution:** Match on construction era and floor area to compare like with like.

```text
Matched Sample Criteria:
- Construction era: 1945-1979 (same building regulations)
- Floor area: 80-100 m² (removes size confounding)
```

### Complementary Methods

| Method                 | Purpose                                          | Implementation           |
| ---------------------- | ------------------------------------------------ | ------------------------ |
| **OLS Regression**     | Effect sizes with controls                       | `statsmodels`            |
| **Mediation Analysis** | Decompose density → building type → energy paths | Baron-Kenny approach     |
| **Scenario Modelling** | Technology persistence testing                   | Manual calculation       |
| **Spatial Regression** | Account for spatial autocorrelation              | `pysal` (in `advanced/`) |

---

## Key Variables

### Dependent Variable

| Variable           | Definition                       | Use Case                      |
| ------------------ | -------------------------------- | ----------------------------- |
| `energy_intensity` | SAP energy / floor area (kWh/m²) | **PRIMARY** - thermal physics |
| `total_energy`     | SAP energy (kWh/year)            | Combined lock-in analysis     |

**Critical:** Use intensity (kWh/m²) for thermal efficiency questions. Per-capita metrics confound efficiency with household size.

### Independent Variables

| Variable                | Description                | Source       |
| ----------------------- | -------------------------- | ------------ |
| `BUILT_FORM`            | Detached/Semi/Terrace/Flat | EPC          |
| `TOTAL_FLOOR_AREA`      | Floor area (m²)            | EPC          |
| `CONSTRUCTION_AGE_BAND` | Construction era           | EPC          |
| `pop_density`           | Persons per hectare        | Census       |
| `cars_per_hh`           | Cars per household         | Census TS045 |

---

## Methodological Considerations

### 1. SAP vs Metered Consumption

EPC energy is SAP-modelled, not metered.

| Captured by SAP                 | NOT Captured by SAP           |
| ------------------------------- | ----------------------------- |
| Building envelope (walls, roof) | Actual thermostat settings    |
| Window area and glazing type    | Occupancy patterns            |
| Heating system efficiency       | Fuel poverty (under-heating)  |
| Building geometry (floor area)  | Lifestyle/behavioural choices |

**Advantage:** SAP isolates building physics from behaviour—exactly what we need for envelope lock-in.

**Language:** Use "potential energy demand" not "consumption".

### 2. Selection Bias in EPCs

EPCs required only at sale/rental—biased toward transacted properties.

### 3. Causal Inference

Observational design cannot establish causation.

- Use "associated with" not "causes"
- Use "lock-in" for structural constraints, not causal mechanisms

### 4. Transport Energy Estimation

Derived from Census car ownership (12,000 km/year per car assumed).

**Limitation:** Census 2021 affected by COVID (31% WFH).

---

## References

- Ewing & Rong (2008) — Residential energy and urban form
- Steadman et al. (2014) — UK building stock modelling
- Rode et al. (2014) — Cities and energy, morphology effects
