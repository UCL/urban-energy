# Urban Energy — Argument & Analysis Cheat Sheet

## Core Thesis

**Urban morphological types have characteristic building physics signatures that predict energy performance — all else being equal. Transport energy compounds the picture. Morphology choice at the planning stage is an energy infrastructure decision, locked in for the building lifetime.**

---

## The Argument Chain

```
MORPHOLOGY TYPE → PHYSICS SIGNATURE → ENERGY PERFORMANCE
   (planning)       (geometry)          (outcome)

   detached     →  high S/V, 0 shared walls  →  highest kWh/m²
   semi         →  moderate S/V, partial      →  moderate
   end-terrace  →  lower S/V, one shared      →  lower
   mid-terrace  →  low S/V, two shared walls  →  lower still
   flat/block   →  lowest S/V, max shared     →  lowest kWh/m²

   + each type has a transport profile:
   detached suburb → high car ownership → high transport kWh/cap
   compact urban   → low car ownership  → low transport kWh/cap
```

---

## Key Indicators & What They Tell Us

### Tier 1: The Physics Signature (why morphology matters)

| Indicator | What it measures | Why it matters | Source |
|---|---|---|---|
| **S/V ratio** | Exposed envelope area ÷ volume | Heat loss surface per unit habitable space. THE primary mechanism | OS footprint + LiDAR |
| **Shared wall ratio** | Shared perimeter ÷ total perimeter | Each shared wall = eliminated heat loss surface | momepy adjacency |
| **Building height** | Mean LiDAR height | Log proxy for S/V (diminishing returns at height) | LiDAR DSM |
| **Form factor** | Envelope area ÷ volume^(2/3) | Dimensionless S/V — comparable across building sizes | Derived |

These four indicators define the physics signature of a morphological type. They don't vary independently — they're bundled by geometry.

### Tier 2: The Morphology Context (what creates the signatures)

| Indicator | What it measures | Why it matters | Source |
|---|---|---|---|
| **Building type** (attached_type) | Detached/semi/terrace/flat | The morphological classification itself | EPC BUILT_FORM + momepy |
| **FAR** (floor area ratio) | Gross floor area ÷ catchment area | Neighbourhood-scale intensity; converges at high values (Rode) | cityseer catchment |
| **BCR** (building coverage ratio) | Footprint area ÷ catchment area | Ground-level compactness | cityseer catchment |
| **Pop density** | Persons per km² | Proxy for morphology at area level; mediates through form | Census TS006 |

These define *which* morphological type you're in. They predict the physics signature, which predicts energy.

### Tier 3: Transport Extension (compounding effect)

| Indicator | What it measures | Why it matters | Source |
|---|---|---|---|
| **Car ownership** | Avg cars per household | Direct → transport energy (kWh per vehicle) | Census TS045 |
| **Commute distance** | Avg one-way km | Scales transport energy | Census TS058 |
| **PT accessibility** | Bus/rail stops within catchment | Alternative to car dependence | NaPTAN + cityseer |

Transport energy tracks morphology: detached suburbs → more cars → more kWh/capita. This compounds the building energy disadvantage.

### Controls (all else being equal)

| Indicator | What it controls for | Source |
|---|---|---|
| **Construction age** | Building regulations era, insulation standards | EPC |
| **Floor area** | Building size (larger = more energy, but lower intensity) | EPC |
| **Fabric efficiency** | Wall/window/roof/heating ratings (1–5) | EPC |
| **Tenure, deprivation** | Socioeconomic confounders | Census |

The "all else being equal" claim depends on these controls. A 2020 detached house still has worse S/V than a 2020 terrace — that's the morphological signal surviving after controlling for technology era.

---

## How Each Analysis Serves the Argument

| Script | Role in argument | What it establishes |
|---|---|---|
| **00 data_quality** | Foundation | Data is fit for purpose; coverage and bias quantified |
| **01 regression** | Core evidence | Morphology indicators predict energy intensity after controls |
| **02 mediation** | The mechanism | Density → building form → energy. Form mediates density |
| **03 transport** | Extension | Transport energy compounds the morphology effect |
| **04 figures** | Communication | Visualises the argument |
| **05 lock-in** | Implication | Morphology choices persist for decades |
| **06 report** | Synthesis | Narrative assembly |
| **07 replication** | Validation | Rode and Norman findings hold in observed English data |
| **advanced/** | Robustness | Spatial autocorrelation, sensitivity, cross-validation, selection bias |

---

## The Three Studies & Their Role

| Study | What they showed | What we add |
|---|---|---|
| **Rode et al. (2014)** | Morphology → up to 6× heat demand variation (simulated, idealised archetypes, 4 European cities) | Same physics signature in *observed* EPC data across real English buildings. If it holds despite measurement noise and tech variation, that's stronger |
| **Norman et al. (2006)** | Per-capita vs per-m² reverses conclusions; building ops 60-70%, transport 20-30% | We can decompose both building + transport by morphological type, and show *why* per-capita differs (floor area per person varies by type) |
| **Newman & Kenworthy (1989)** | Density → automobile dependence (macro, 32 cities) | We show the same pattern at building level: morphological types that are energy-inefficient for buildings are also car-dependent. The physics and transport penalties are bundled |

---

## Key Scales

| Scale | Unit | Use |
|---|---|---|
| **Building** | Individual UPRN | Physics signature (S/V, shared walls, height) |
| **Catchment 400m** | π×0.4²≈0.5 km² | Closest to Rode's 500m grid. Primary for FAR/BCR |
| **Catchment 800m** | π×0.8²≈2.0 km² | Standard pedestrian catchment |
| **Output Area** | ~100-200 households | Census demographics, pop density |
| **LSOA** | ~1500 people | Cluster unit for robust SEs |

---

## Proof of Concept Plan

**Goal:** Run the full argument chain on Manchester (E63008401) to test whether the signal exists before scaling to more cities.

**Data prerequisite:** `uprn_integrated.gpkg` must exist (run `processing/test_pipeline.py` first).

### Step 1: Descriptive — Types bundle different physics

Table of mean S/V, shared wall ratio, height, floor area by building type (detached/semi/terrace/flat). This is the "morphological types have characteristic physics signatures" claim. If types don't separate on physics, there's no argument.

- [ ] Status: existing partial coverage in `07_replication_analysis.py` (`rode_typology_hierarchy`)

### Step 2: Significance — Types differ in energy outcome

ANOVA / Kruskal-Wallis on energy intensity by type. Then pairwise comparisons. This establishes that the outcome actually varies by morphology.

- [ ] Status: not yet as a formal test with effect sizes

### Step 3: Regression — Physics predicts energy, all else equal

OLS: `energy_intensity ~ S/V + shared_wall_ratio + controls` (age, floor area, fabric efficiency). The physics signature must survive after controlling for building technology era. This is the core evidence.

- [ ] Status: existing partial coverage in `01_regression_analysis.py` and `07_replication_analysis.py`

### Step 4: Mediation — Type effect operates through physics

Two regressions:
- (a) `energy ~ type_dummies + controls` → type coefficients
- (b) `energy ~ type_dummies + S/V + shared_walls + controls` → type coefficients shrink

If type coefficients attenuate when physics is added, the type effect operates *through* the physics signature — not through some unmeasured confounder correlated with type.

- [ ] Status: existing partial coverage in `02_mediation_analysis.py` and `07_replication_analysis.py`, but framed as density mediation, not type mediation

### Step 5: Transport extension — Morphology compounds

Car ownership and transport energy by building type / density quintile. Shows that the same morphologies that are energy-inefficient for buildings are also car-dependent.

- [ ] Status: existing in `03_transport_analysis.py` and `07_replication_analysis.py`

### Step 6: Interpretation — Lock-in and policy

Quantify: how much of the building stock is locked into high-S/V morphologies? What's the aggregate energy penalty? How long do these buildings last?

- [ ] Status: existing in `05_lockin_analysis.py`

### Decision gates

- **After Step 1:** If types don't separate on physics → investigate data quality / S/V computation
- **After Step 3:** If S/V and shared walls are not significant → the morphological argument fails; pivot to density or accessibility framing
- **After Step 4:** If type coefficients don't attenuate → physics doesn't explain the type effect; something else about type matters (e.g., occupant sorting)

### Future extensions (not proof of concept)

- Clustering on morphological variables to discover data-driven sub-types beyond the 5 EPC categories
- Multi-city replication (scale beyond Manchester)
- Spatial regression (lag/error models)
- Instrumental variables for causal identification

---

## One-Sentence Version

> Different urban morphologies bundle characteristic building physics (S/V ratio, shared walls) and transport profiles (car ownership) that predict energy performance after controlling for building age, size, and technology — making morphology choice a long-term energy infrastructure decision.
