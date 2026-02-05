# Analysis Report: Structural Energy Penalties of Urban Sprawl

**Generated:** 2026-02-05
**Dataset:** 173,907 domestic properties (Greater Manchester)
**Methodology:** [stats/README.md](README.md)

---

## Summary

Sprawling development locks in energy penalties that technology cannot eliminate:

| Lock-In        | Mechanism            | Magnitude      | Persists with Best Tech? |
| -------------- | -------------------- | -------------- | ------------------------ |
| **Floor area** | Larger homes         | +59%           | Yes                      |
| **Envelope**   | More exposed walls   | +35% kWh/m²/yr | Yes (proportionally)     |
| **Transport**  | Car dependence       | +22%           | Yes (proportionally)     |
| **Combined**   | Building + transport | **+50%**       | **Yes**                  |

---

## 1. Floor Area Lock-In

Detached houses are systematically larger:

| Built Form    | Mean Floor Area | vs Mid-Terrace |
| ------------- | --------------- | -------------- |
| Detached      | 136 m²          | **+59%**       |
| Semi-Detached | 97 m²           | +13%           |
| Mid-Terrace   | 86 m²           | baseline       |
| Flat          | 59 m²           | -31%           |

---

## 2. Envelope Lock-In (Matched Comparison)

Controlling for construction era (1945-1979) and floor area (80-100 m²):

| Built Form    | Exposed Walls | Energy Intensity (kWh/m²/year) | vs Detached |
| ------------- | ------------- | ------------------------------ | ----------- |
| Detached      | 4             | 283                            | baseline    |
| Semi-Detached | 3             | 234                            | **-17%**    |
| Mid-Terrace   | 2             | 185                            | **-35%**    |

Each shared wall eliminates ~17% of heat loss.

**Why matched?** Raw data shows detached with _lower_ intensity (212 vs 230 kWh/m²) because detached homes are newer on average. Matching reveals the true thermal penalty.

---

## 3. Transport Lock-In

Car ownership by density quartile:

| Density          | Cars/Household | Transport Energy (ICE) |
| ---------------- | -------------- | ---------------------- |
| High (top 25%)   | 0.63           | 5,496 kWh-eq/year      |
| Low (bottom 25%) | 0.76           | 6,681 kWh-eq/year      |
| **Difference**   | **+22%**       | **+22%**               |

---

## 4. Combined Lock-In

Total energy footprint (building + transport):

| Scenario                    | High-Density Flat | Low-Density Detached | Penalty  |
| --------------------------- | ----------------- | -------------------- | -------- |
| Current (avg stock, ICE)    | 15,883 kWh        | 23,896 kWh           | **+50%** |
| Best tech (Passivhaus + EV) | 2,274 kWh         | 3,400 kWh            | **+51%** |

Technology reduces absolute demand but the proportional penalty persists.

---

## 5. Practical Translation

For a 90 m² home:

- **Matched intensity difference:** 98 kWh/m²/year (detached vs mid-terrace)
- **Annual energy penalty:** 8,820 kWh/year
- **Annual cost penalty:** ~£617/year at current prices

---

## Appendix: Limitations

| Issue                    | Implication                                       |
| ------------------------ | ------------------------------------------------- |
| SAP not metered          | Results reflect potential demand, not consumption |
| Single city              | Findings need multi-city validation               |
| Observational design     | Association, not causation                        |
| Transport from car proxy | Illustrative, not measured                        |

## Appendix: Sample Sizes

**Matched comparison (1945-1979, 80-100 m²):**

- Detached: 151
- Semi-Detached: 2,450
- Mid-Terrace: 3,443

**Transport analysis:**

- High-density quartile: 43,396
- Low-density quartile: 43,499

---

_Generated from `temp/stats/results/lockin_summary.json`_
_Pipeline: `uv run python stats/run_all.py`_
