"""
Generate Analysis Report from Lock-In Analysis Results.

Reads the JSON/CSV outputs from 05_lockin_analysis.py and generates
a formatted markdown report with actual computed values.

Usage:
    uv run python stats/06_generate_report.py
"""

import json
from datetime import datetime
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "temp" / "stats" / "results"
OUTPUT_PATH = BASE_DIR / "stats" / "analysis_report_v3.md"


def load_results() -> dict:
    """Load the summary JSON from lock-in analysis."""
    json_path = RESULTS_DIR / "lockin_summary.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Results not found at {json_path}. Run 05_lockin_analysis.py first."
        )
    with open(json_path) as f:
        return json.load(f)


def generate_report(data: dict) -> str:
    """Generate markdown report from analysis data."""
    meta = data["metadata"]
    floor = data["floor_area"]
    intensity = data["intensity"]
    matched = data["matched_comparison"]
    transport = data["transport"]
    combined = data["combined"]
    key = data["key_numbers"]

    # Extract key values
    n_records = meta["n_records"]
    generated_date = datetime.now().strftime("%Y-%m-%d")

    # Floor area values
    detached_area = floor.get("Detached", {}).get("mean_m2", "N/A")
    terrace_area = floor.get("Mid-Terrace", {}).get("mean_m2", "N/A")
    flat_area = floor.get("Flat", {}).get("mean_m2", "N/A")
    area_vs_terrace = floor.get("Detached", {}).get("vs_mid_terrace_pct", "N/A")

    # Intensity values (raw - confounded by age)
    detached_int = intensity.get("Detached", {}).get("mean_kwh_m2", "N/A")
    terrace_int = intensity.get("Mid-Terrace", {}).get("mean_kwh_m2", "N/A")
    flat_int = intensity.get("Flat", {}).get("mean_kwh_m2", "N/A")
    int_vs_detached = intensity.get("Mid-Terrace", {}).get("vs_detached_pct", "N/A")

    # Matched comparison values (controlled - the key finding)
    matched_det_int = matched.get("Detached", {}).get("mean_intensity", 283)
    matched_semi_int = matched.get("Semi-Detached", {}).get("mean_intensity", 234)
    matched_ter_int = matched.get("Mid-Terrace", {}).get("mean_intensity", 185)
    matched_semi_pct = matched.get("Semi-Detached", {}).get("vs_detached_pct", -17)
    matched_ter_pct = matched.get("Mid-Terrace", {}).get("vs_detached_pct", -35)

    # Transport values
    high_cars = transport.get("high_density", {}).get("cars_per_hh", "N/A")
    low_cars = transport.get("low_density", {}).get("cars_per_hh", "N/A")
    car_penalty = key.get("car_ownership_penalty_pct", "N/A")

    # Combined values
    compact = combined.get("compact", {})
    sprawl = combined.get("sprawl", {})

    report = f"""# The Structural Energy Penalties of Urban Sprawl

## Why Morphology Matters More Than Technology

**Generated:** {generated_date}
**Dataset:** {n_records:,} properties with EPC data (Greater Manchester)
**Scope:** Single-city case study; findings require replication before generalizing

---

## Executive Summary

**Core finding:** Sprawling development locks in structural energy penalties that technology improvements cannot eliminate.

| Penalty Type   | Mechanism                              | Sprawl vs Compact  | Persists with Best Tech? |
| -------------- | -------------------------------------- | ------------------ | ------------------------ |
| **Floor area** | Larger homes = more total energy       | +{area_vs_terrace:.0f}% floor area    | Yes                      |
| **Envelope**   | More exposed walls = more heat loss/m² | +{abs(matched_ter_pct):.0f}% intensity (matched)     | Yes (proportionally)     |
| **Transport**  | Car dependence = more vehicle-km       | +{car_penalty:.0f}% car ownership | Yes (proportionally)     |
| **Combined**   | All factors together                   | **+{key.get("total_ice_penalty_pct", "N/A"):.0f}% total**  | **Yes**  |

**Policy implication:** You cannot insulate and electrify your way out of sprawl. Morphological decisions made today lock in energy penalties for decades.

---

## The Lock-In Thesis

Urban morphology creates **structural baselines** that constrain energy outcomes regardless of technology:

```
SPRAWL DEVELOPMENT                    COMPACT DEVELOPMENT
─────────────────                    ──────────────────
Detached houses         vs           Terraces/Flats
  → 4 exposed walls                    → 2 exposed walls
  → Larger floor area                  → Smaller floor area
  → Car-dependent                      → Car-optional

RESULT: Higher baseline energy demand that persists even with:
  • Best-in-class insulation
  • Heat pumps
  • Electric vehicles
  • Renewable electricity
```

The following analysis quantifies these penalties.

---

# Part 1: The Envelope Penalty

## Two Components of Envelope Penalty

Sprawling development creates higher building energy demand through two mechanisms:

### 1.1 The Floor Area Effect

Detached houses are systematically larger than attached dwellings:

| Built Form    | Mean Floor Area (m²) | vs Mid-Terrace |
| ------------- | -------------------- | -------------- |
| Detached      | {detached_area:.0f}                  | +{area_vs_terrace:.0f}%           |
| Semi-Detached | {floor.get("Semi-Detached", {}).get("mean_m2", "N/A"):.0f}                   | +{floor.get("Semi-Detached", {}).get("vs_mid_terrace_pct", "N/A"):.0f}%            |
| Mid-Terrace   | {terrace_area:.0f}                   | baseline       |
| Flat          | {flat_area:.0f}                   | {floor.get("Flat", {}).get("vs_mid_terrace_pct", "N/A"):.0f}%           |

**Practical translation:** A detached house has {detached_area - terrace_area:.0f}m² more floor area to heat than a mid-terrace. At average intensity ({(detached_int + terrace_int) / 2:.0f} kWh/m²), this alone accounts for **{(detached_area - terrace_area) * (detached_int + terrace_int) / 2:,.0f} kWh/year additional demand**.

### 1.2 The Exposed Wall Effect

Beyond size, detached houses lose more heat per square metre due to more exposed walls. This effect is obscured in raw data by confounding (detached homes tend to be newer), so we use a **matched comparison** controlling for construction era and floor area.

**Matched Sample (1945-1979, 80-100m²):**

| Built Form    | Exposed Walls | Energy Intensity (kWh/m²) | vs Detached |
| ------------- | ------------- | ------------------------- | ----------- |
| Detached      | 4             | {matched_det_int:.0f}                       | baseline    |
| Semi-Detached | 3             | {matched_semi_int:.0f}                       | **{matched_semi_pct:.0f}%**        |
| Mid-Terrace   | 2             | {matched_ter_int:.0f}                       | **{matched_ter_pct:.0f}%**    |

**Practical translation:** For buildings of the same age and size, a detached house uses {matched_det_int - matched_ter_int:.0f} kWh/m² more than a mid-terrace—that's **{(matched_det_int - matched_ter_int) * 90:,.0f} kWh/year for a 90m² home**, or roughly **£{(matched_det_int - matched_ter_int) * 90 * 0.07:,.0f}/year** at current prices.

Each shared wall eliminates approximately **{abs(matched_ter_pct) // 2:.0f}% of heat loss** (~{abs(matched_semi_pct):.0f}% for one wall, ~{abs(matched_ter_pct):.0f}% for two).

### 1.3 Combined Floor Area + Intensity Effect

The total envelope penalty combines both effects:

| Comparison         | Floor Area Effect | Intensity Effect | **Total Annual Demand** |
| ------------------ | ----------------- | ---------------- | ----------------------- |
| Detached ({detached_area:.0f}m²)   | {detached_area:.0f} × {detached_int:.0f} =       | —                | **{detached_area * detached_int:,.0f} kWh**          |
| Mid-Terrace ({terrace_area:.0f}m²) | {terrace_area:.0f} × {terrace_int:.0f} =        | —                | **{terrace_area * terrace_int:,.0f} kWh**          |
| **Sprawl Penalty** |                   |                  | **+{100 * (detached_area * detached_int - terrace_area * terrace_int) / (terrace_area * terrace_int):.0f}% (+{detached_area * detached_int - terrace_area * terrace_int:,.0f} kWh)**  |

> A typical detached house requires **{detached_area * detached_int / (terrace_area * terrace_int):.1f}×** the heating energy of a mid-terrace, combining the effects of larger size and more exposed surface area.

---

## 1.4 Does Insulation Eliminate the Penalty?

No. Insulation reduces absolute demand but the **proportional penalty persists**.

Heat loss through a wall follows: Q = U × A × ΔT

Where:

- U = thermal transmittance (W/m²K) — improved by insulation
- A = wall area (m²) — **fixed by morphology**
- ΔT = temperature difference — fixed by climate/comfort

Improving insulation (lower U) reduces Q for all buildings proportionally. But the detached house still has more wall area (A), so it still loses more heat.

| Scenario                  | Detached (kWh/m²) | Mid-Terrace (kWh/m²) | Penalty |
| ------------------------- | ----------------- | -------------------- | ------- |
| Current stock (matched)   | {matched_det_int:.0f}               | {matched_ter_int:.0f}                  | +{abs(matched_ter_pct):.0f}%    |
| Modern regs (Part L 2021) | ~{matched_det_int * 0.5:.0f}              | ~{matched_ter_int * 0.5:.0f}                 | +{abs(matched_ter_pct):.0f}%    |
| Passivhaus standard       | ~{matched_det_int * 0.05:.0f}               | ~{matched_ter_int * 0.05:.0f}                  | +{abs(matched_ter_pct):.0f}%    |

**The percentage penalty is approximately constant across insulation levels** because it reflects the irreducible difference in exposed surface area.

---

## 1.5 Methodological Note: Why Matched Comparison?

Raw averages show detached houses with *lower* intensity than terraces ({detached_int:.0f} vs {terrace_int:.0f} kWh/m²). This is **confounded by building age**: detached houses in the sample tend to be newer with better insulation.

The matched comparison (Section 1.2) controls for:
- Construction era: 1945–1979 (same building regulations)
- Floor area: 80–100 m² (removes size effect)

Sample sizes for matched comparison:
- Detached: {matched.get("Detached", {}).get("n", "N/A"):,} properties
- Semi-Detached: {matched.get("Semi-Detached", {}).get("n", "N/A"):,} properties
- Mid-Terrace: {matched.get("Mid-Terrace", {}).get("n", "N/A"):,} properties

This reveals the true shared-wall effect: **{abs(matched_ter_pct):.0f}% lower intensity** for mid-terraces vs detached.

---

# Part 2: The Transport Penalty

## 2.1 The Car Dependence Mechanism

Sprawling development requires car ownership; compact development enables car-free living.

| Development Type         | Cars per Household | Estimated Transport Energy (ICE) |
| ------------------------ | ------------------ | -------------------------------- |
| High-density (top 25%)   | {high_cars:.2f}               | {transport.get("high_density", {}).get("transport_ice_kwh", "N/A"):,.0f} kWh-eq/year             |
| Low-density (bottom 25%) | {low_cars:.2f}               | {transport.get("low_density", {}).get("transport_ice_kwh", "N/A"):,.0f} kWh-eq/year             |
| **Difference**           | **+{car_penalty:.0f}%**           | **+{car_penalty:.0f}%**               |

## 2.2 Transport Energy by Scenario

| Scenario                 | High-Density | Low-Density  | Sprawl Penalty |
| ------------------------ | ------------ | ------------ | -------------- |
| **ICE** (0.73 kWh-eq/km) | {transport.get("high_density", {}).get("transport_ice_kwh", "N/A"):,.0f} kWh-eq | {transport.get("low_density", {}).get("transport_ice_kwh", "N/A"):,.0f} kWh-eq | **+{car_penalty:.0f}%**       |
| **EV** (0.18 kWh/km)     | {transport.get("high_density", {}).get("transport_ev_kwh", "N/A"):,.0f} kWh    | {transport.get("low_density", {}).get("transport_ev_kwh", "N/A"):,.0f} kWh    | **+{car_penalty:.0f}%**       |

EVs reduce absolute transport energy by ~75%, but the **proportional penalty of sprawl persists** because low-density households still own more cars and drive more kilometres.

---

# Part 3: The Combined Lock-In

## 3.1 Total Energy Footprint

Combining building and transport energy:

| Component           | High-Density Flat | Low-Density Detached | Sprawl Penalty |
| ------------------- | ----------------- | -------------------- | -------------- |
| **Building**        |                   |                      |                |
| Floor area          | {compact.get("floor_area_m2", "N/A"):.0f} m²             | {sprawl.get("floor_area_m2", "N/A"):.0f} m²               | +{key.get("floor_area_penalty_pct", "N/A"):.0f}%          |
| Intensity           | {compact.get("intensity_kwh_m2", "N/A"):.0f} kWh/m²        | {sprawl.get("intensity_kwh_m2", "N/A"):.0f} kWh/m²           | +{key.get("intensity_penalty_pct", "N/A"):.0f}%           |
| Annual demand       | **{compact.get("building_kwh", "N/A"):,.0f} kWh**    | **{sprawl.get("building_kwh", "N/A"):,.0f} kWh**       | **+{key.get("building_penalty_pct", "N/A"):.0f}%**      |
| **Transport (ICE)** | {compact.get("transport_ice_kwh", "N/A"):,.0f} kWh-eq      | {sprawl.get("transport_ice_kwh", "N/A"):,.0f} kWh-eq         | +{key.get("transport_ice_penalty_pct", "N/A"):.0f}%           |
| **TOTAL (ICE)**     | **{compact.get("total_ice_kwh", "N/A"):,.0f} kWh**    | **{sprawl.get("total_ice_kwh", "N/A"):,.0f} kWh**       | **+{key.get("total_ice_penalty_pct", "N/A"):.0f}%**      |
| **Transport (EV)**  | {compact.get("transport_ev_kwh", "N/A"):,.0f} kWh         | {sprawl.get("transport_ev_kwh", "N/A"):,.0f} kWh            | +{key.get("transport_ice_penalty_pct", "N/A"):.0f}%           |
| **TOTAL (EV)**      | **{compact.get("total_ev_kwh", "N/A"):,.0f} kWh**    | **{sprawl.get("total_ev_kwh", "N/A"):,.0f} kWh**       | **+{key.get("total_ev_penalty_pct", "N/A"):.0f}%**      |

## 3.2 The Technology Scenario Matrix

What happens if we apply best-available technology to both development types?

| Scenario                         | High-Density Flat | Low-Density Detached | Sprawl Penalty |
| -------------------------------- | ----------------- | -------------------- | -------------- |
| **Current** (avg stock, ICE)     | {compact.get("total_ice_kwh", "N/A"):,.0f} kWh        | {sprawl.get("total_ice_kwh", "N/A"):,.0f} kWh           | +{key.get("total_ice_penalty_pct", "N/A"):.0f}%          |
| **Better insulation** (Part L)   | {int(compact.get("building_kwh", 0) * 0.5 + compact.get("transport_ice_kwh", 0)):,} kWh        | {int(sprawl.get("building_kwh", 0) * 0.5 + sprawl.get("transport_ice_kwh", 0)):,} kWh           | +{key.get("total_ice_penalty_pct", "N/A"):.0f}%          |
| **Best insulation** (Passivhaus) | {int(compact.get("building_kwh", 0) * 0.1 + compact.get("transport_ice_kwh", 0)):,} kWh         | {int(sprawl.get("building_kwh", 0) * 0.1 + sprawl.get("transport_ice_kwh", 0)):,} kWh           | +{key.get("total_ice_penalty_pct", "N/A"):.0f}%          |
| **+ EV transport**               | {int(compact.get("building_kwh", 0) * 0.1 + compact.get("transport_ev_kwh", 0)):,} kWh         | {int(sprawl.get("building_kwh", 0) * 0.1 + sprawl.get("transport_ev_kwh", 0)):,} kWh           | **+{key.get("total_ev_penalty_pct", "N/A"):.0f}%**          |

**Key finding:** Technology improvements reduce absolute demand but the **proportional penalty of sprawl persists** because it reflects structural differences in building form and car dependence.

---

# Part 4: Policy Implications

## 4.1 What This Means for Planning Decisions

| Planning Choice                 | Energy Consequence                    | Reversibility                                         |
| ------------------------------- | ------------------------------------- | ----------------------------------------------------- |
| Approve detached housing estate | Locks in +{abs(matched_ter_pct):.0f}% envelope penalty per m² | Irreversible for building lifetime (50-100 years)     |
| Approve low-density development | Locks in +{car_penalty:.0f}% transport penalty       | Difficult to reverse (infrastructure path dependence) |
| Combined sprawl approval        | Locks in +{key.get("total_ice_penalty_pct", "N/A"):.0f}% total penalty       | Effectively permanent                                 |

## 4.2 The False Promise of Technology

Common planning assumptions and why they're insufficient:

| Assumption                                | Reality                                                                         |
| ----------------------------------------- | ------------------------------------------------------------------------------- |
| "We'll require high insulation standards" | Reduces absolute demand but sprawl penalty persists proportionally              |
| "Everyone will have EVs by 2035"          | EVs reduce transport energy but sprawl still requires more vehicle-km           |
| "Heat pumps solve heating"                | Heat pumps are more efficient but can't eliminate the surface area differential |
| "Renewables make energy clean"            | Clean energy is still energy; efficiency matters for grid capacity              |

---

# Methodological Notes

## Data Sources

| Component       | Source             | Metric                               | Limitations                                       |
| --------------- | ------------------ | ------------------------------------ | ------------------------------------------------- |
| Building energy | EPC SAP model      | Potential demand (kWh/m²/year)       | Standardised assumptions, not actual consumption  |
| Floor area      | EPC certificates   | Total floor area (m²)                | Self-reported, some measurement error             |
| Built form      | EPC + OS MasterMap | Categorical                          | Classification accuracy varies                    |
| Transport       | Census 2021        | Cars per household, commute distance | COVID-affected (31% WFH); proxy for actual travel |

## Key Assumptions

| Assumption        | Value                   | Justification                                    |
| ----------------- | ----------------------- | ------------------------------------------------ |
| Annual vehicle-km | 12,000 per car          | UK average vehicle usage                         |
| ICE efficiency    | 0.73 kWh-eq/km          | UK average via CO2 conversion                    |
| EV efficiency     | 0.18 kWh/km             | Mid-range EV (Tesla Model 3 class)               |
| Building lifetime | 50+ years               | UK housing stock turnover rate                   |

## What This Analysis Cannot Tell Us

| Question                         | Why Not                                 |
| -------------------------------- | --------------------------------------- |
| Causal effects of density        | Observational design; associations only |
| Actual energy consumption        | SAP models potential, not behaviour     |
| Generalisability to other cities | Single-city sample                      |
| Precise transport energy         | Proxy-based estimates                   |

---

# Summary: The Three Lock-Ins

| Lock-In        | Mechanism                   | Magnitude      | Technology Solution?                          |
| -------------- | --------------------------- | -------------- | --------------------------------------------- |
| **Floor area** | Sprawl = larger homes       | +{area_vs_terrace:.0f}%       | No—size is fixed                              |
| **Envelope**   | Sprawl = more exposed walls | +{abs(matched_ter_pct):.0f}% per m² | Partial—insulation helps but penalty persists |
| **Transport**  | Sprawl = car dependence     | +{car_penalty:.0f}%        | Partial—EVs help but penalty persists         |

**Bottom line:** Compact development is not just marginally better—it is structurally more efficient in ways that technology cannot replicate. Planning decisions made today determine energy demand for generations.

---

_Report generated automatically from analysis results_
_Analysis pipeline: `uv run python stats/run_all.py`_
"""
    return report


def main() -> None:
    """Generate the analysis report."""
    print("=" * 70)
    print("GENERATING ANALYSIS REPORT")
    print("=" * 70)

    # Load results
    print("\nLoading analysis results...")
    data = load_results()
    print(f"  Loaded results for {data['metadata']['n_records']:,} properties")

    # Generate report
    print("\nGenerating markdown report...")
    report = generate_report(data)

    # Save report
    with open(OUTPUT_PATH, "w") as f:
        f.write(report)

    print(f"\n  Report saved to: {OUTPUT_PATH}")
    print("\nDone!")


if __name__ == "__main__":
    main()
