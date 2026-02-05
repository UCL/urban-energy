# Methodology Notes

Working notes on data processing decisions and morphology metric computation.

## Theoretical Framework

### The Causal Chain (Hypothesised)

```text
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
[Building Thermal Physics]                             [Behavioural Changes]
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

### The Confounding Problem

Raw data shows detached houses with _lower_ energy intensity than terraces. This counterintuitive result arises from age confounding:

```text
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

**Solution:** The matched comparison design controls for construction era and floor area, isolating the true shared-wall effect. This reveals the +53% intensity penalty for detached houses.

---

## Analysis Framework

### Units of Analysis

**Primary spatial unit:** OS Open Map Local **building** footprints

**Analysis unit:** **UPRN** (Unique Property Reference Number), disaggregated from buildings

**Aggregation unit:** **Street network segments** (400m pedestrian catchments)

### Hierarchy

```text
Building (OS Open Map Local)
    │
    ├── morphology computed here (height, footprint, adjacency)
    │
    ↓ spatial join (UPRN point ∈ building polygon)
UPRN
    │
    ├── inherits building morphology
    ├── + EPC data (dwelling-level characteristics)
    ├── + Census data (area-level socio-demographics)
    │
    ↓ cityseer network assignment
Street segment (400m catchment)
    │
    └── aggregated metrics for regression
```

### Design Decision: Buildings vs Flats

**Decision:** Compute morphology at building level, not individual dwelling level.

**Rationale:**

1. **Thermal boundary** - The building envelope determines external heat loss. Party walls between flats within a building are less thermally significant than external walls.

2. **Data availability** - We have building footprints (OS) and heights (LiDAR), but not floor plans. EPC already captures within-building position via `BUILT_FORM` and `FLOOR_LEVEL`.

3. **Shared walls** - Compute adjacency between **buildings** (party walls with neighbouring buildings). Don't attempt to model party walls between flats - EPC's `BUILT_FORM` (Mid-Floor Flat, Top-Floor Flat, etc.) already captures this.

4. **Multiple UPRNs per building** - All UPRNs in a building inherit the same morphology metrics. The `n_uprns_in_building` count indicates dwelling density within the envelope.

**What EPC captures (don't duplicate):**

- `BUILT_FORM`: Mid-Terrace, End-Terrace, Mid-Floor Flat, Top-Floor Flat, Ground-Floor Flat
- `FLOOR_LEVEL`: Vertical position in building
- `TOTAL_FLOOR_AREA`: Individual dwelling floor area

**What morphology adds (building context):**

- Height (from LiDAR)
- Footprint area
- Adjacency to other buildings
- Shared wall ratio with other buildings
- Building type classification

### Variable Categories

| Category                 | Source     | Level           | Role in Model             |
| ------------------------ | ---------- | --------------- | ------------------------- |
| Energy consumption       | EPC        | UPRN            | Dependent variable        |
| Building characteristics | EPC        | UPRN            | Controls                  |
| Building morphology      | OS + LiDAR | Building → UPRN | Contextual                |
| Neighbourhood morphology | Aggregated | Segment         | **Independent variables** |
| Socio-demographics       | Census     | OA → UPRN       | Controls                  |
| Network metrics          | cityseer   | Segment         | Independent variables     |

---

## Building-Level Metrics

These are computed on building footprints and inherited by UPRNs via spatial join.

### Geometry Metrics

| Metric                | Formula                              | Notes                                |
| --------------------- | ------------------------------------ | ------------------------------------ |
| `footprint_area_m2`   | `geometry.area`                      | Direct from polygon                  |
| `perimeter_m`         | `geometry.length`                    | For compactness/shared wall          |
| `height_m`            | `height_median` from LiDAR           | Less affected by roof peaks than max |
| `estimated_floors`    | `max(1, round(height_median / 2.8))` | 2.8m typical UK floor-to-floor       |
| `gross_floor_area_m2` | `footprint_area × estimated_floors`  | For FAR calculation                  |

### Building Typology (Adjacency Analysis)

**Method:**

1. Buffer building polygons by 0.3m
2. Find intersections with neighbouring buildings
3. Measure shared boundary length

| Metric                 | Formula                          | Interpretation                 |
| ---------------------- | -------------------------------- | ------------------------------ |
| `n_adjacent_buildings` | Count of touching neighbours     | 0=detached, 1=semi, 2+=terrace |
| `shared_wall_length_m` | Sum of intersection lengths      | Party wall exposure            |
| `shared_wall_ratio`    | `shared_wall_length / perimeter` | 0=detached, ~0.5=terrace       |

**Classification Logic:**

```python
def classify_building(n_adjacent, shared_wall_ratio, n_uprns, has_flats):
    if n_uprns > 4 and has_flats:
        return 'flat_block'
    elif n_adjacent == 0:
        return 'detached'
    elif n_adjacent == 1 and shared_wall_ratio < 0.3:
        return 'semi_detached'
    elif n_adjacent >= 2 or shared_wall_ratio > 0.3:
        return 'terraced'
    else:
        return 'unknown'
```

**EPC Cross-Validation:**

Where EPCs exist for a building, compare:

- `BUILT_FORM`: Detached, Semi-Detached, Mid-Terrace, End-Terrace
- `PROPERTY_TYPE`: House, Flat, Bungalow, Maisonette

Modal EPC classification can validate geometry-based inference.

---

## Floor Area and FAR

### The Problem

FAR (Floor Area Ratio) = Total Floor Area / Land Area

We need total floor area per building, but:

- EPC `TOTAL_FLOOR_AREA` is internal usable area per dwelling
- `footprint × floors` is gross floor area for the building
- These differ systematically (walls, stairs, common areas)

### Approach

**For buildings WITH EPCs:**

- Sum `TOTAL_FLOOR_AREA` across all EPCs in building
- This gives actual internal floor area (more accurate for energy)

**For buildings WITHOUT EPCs:**

- Use `gross_floor_area = footprint × estimated_floors`
- Accept systematic overestimate vs internal area

**At segment level:**

```
FAR = Σ(building_floor_area) / catchment_area
```

### Floor Estimation Considerations

| Factor              | Issue                       | Approach                                |
| ------------------- | --------------------------- | --------------------------------------- |
| Pitched roofs       | Add height without floors   | Use `height_median` not `height_max`    |
| Victorian buildings | Higher ceilings (~3.0-3.2m) | Could adjust by `CONSTRUCTION_AGE_BAND` |
| Ground floor retail | ~4-5m floor height          | Not applicable (domestic EPCs only)     |
| Loft conversions    | Partial floor               | Captured in height but overestimates    |

**Pragmatic choice:** Use 2.8m divisor for all residential. Accept error.

---

## Segment-Level Morphology Variables

These are the actual independent variables for regression, computed by aggregating building metrics within 400m network catchments.

### Density Metrics

| Metric                    | Formula                                  | Hypothesis                                |
| ------------------------- | ---------------------------------------- | ----------------------------------------- |
| `building_coverage_ratio` | `Σ footprint_area / catchment_area`      | Higher → urban heat island → less heating |
| `far`                     | `Σ floor_area / catchment_area`          | Development intensity                     |
| `building_density_per_ha` | `n_buildings / (catchment_area / 10000)` | General density                           |
| `uprn_density_per_ha`     | `n_uprns / (catchment_area / 10000)`     | Address density                           |

### Form Metrics

| Metric                   | Formula                   | Hypothesis                    |
| ------------------------ | ------------------------- | ----------------------------- |
| `mean_building_height_m` | `mean(height_median)`     | Taller context → more shelter |
| `height_std_m`           | `std(height_median)`      | Height variation              |
| `mean_footprint_m2`      | `mean(footprint_area)`    | Building grain                |
| `mean_shared_wall_ratio` | `mean(shared_wall_ratio)` | Party wall prevalence         |

### Stock Composition

| Metric              | Formula                     | Interpretation            |
| ------------------- | --------------------------- | ------------------------- |
| `pct_terraced`      | `count(terraced) / total`   | Traditional dense housing |
| `pct_detached`      | `count(detached) / total`   | Suburban character        |
| `pct_flats`         | `count(flat_block) / total` | Apartment buildings       |
| `pct_semi_detached` | `count(semi) / total`       | Interwar suburbs          |

---

## EPC Variables (Controls)

These are building-level controls from EPC, not morphology variables.

| Variable                    | Role               | Notes                           |
| --------------------------- | ------------------ | ------------------------------- |
| `TOTAL_FLOOR_AREA`          | Size control       | Energy scales with floor area   |
| `PROPERTY_TYPE`             | Form control       | Detached/Semi/Terrace/Flat      |
| `BUILT_FORM`                | Detailed form      | Mid vs End terrace matters      |
| `CONSTRUCTION_AGE_BAND`     | Age/efficiency     | Proxy for insulation standards  |
| `MAIN_FUEL`                 | Heating system     | Gas vs electric vs oil          |
| `WALLS_DESCRIPTION`         | Construction       | Solid vs cavity walls           |
| `CURRENT_ENERGY_EFFICIENCY` | Overall efficiency | SAP score (alternative outcome) |

**Dependent Variable:**

- `ENERGY_CONSUMPTION_CURRENT`: Estimated annual kWh (SAP modelled)

---

## Causal Pathways

How might neighbourhood morphology affect building energy consumption?

### Direct Thermal Effects

| Pathway           | Mechanism            | Captured by                      |
| ----------------- | -------------------- | -------------------------------- |
| Urban heat island | Dense areas warmer   | `building_coverage_ratio`, `far` |
| Wind shelter      | Buildings block wind | `mean_building_height`, density  |
| Solar shading     | Shadows reduce gain  | Height, density (complex)        |

### Stock Composition Effects

| Pathway           | Mechanism                      | Captured by                              |
| ----------------- | ------------------------------ | ---------------------------------------- |
| Building type mix | Terraces more efficient        | `pct_terraced`, `mean_shared_wall_ratio` |
| Age distribution  | Areas have consistent age      | Correlated with morphology               |
| Size distribution | Dense areas have smaller units | `mean_footprint`, UPRN density           |

### Confounders to Control

| Factor                   | Why it matters                | How to address                      |
| ------------------------ | ----------------------------- | ----------------------------------- |
| Building characteristics | Direct efficiency             | EPC controls                        |
| Socio-economics          | Behaviour, investment         | Census controls                     |
| Tenure                   | Maintenance, split incentives | EPC + Census                        |
| Climate zone             | Heating degree days           | Regional controls or national focus |

---

## Open Questions

1. **Height estimation refinement:**
   - Should we adjust floor height by building age?
   - Can we detect roof type from height_std within building?

2. **Mixed-use buildings:**
   - OS Open Map Local includes commercial buildings
   - Should we filter to residential only, or include commercial as "context"?

3. **Building-UPRN mismatch:**
   - Some UPRNs may not fall within any building footprint
   - How to handle? Nearest building? Exclude?

4. **Temporal alignment:**
   - LiDAR: 2000-2022 composite
   - EPCs: 2007-present (using most recent)
   - Census: 2021
   - Buildings may have changed

5. **Scale sensitivity:**
   - 400m catchment is one choice
   - Should test 200m, 800m, LSOA for robustness

---

## Implementation Sequence

1. **`process_morphology.py`**
   - Input: `building_heights.gpkg`
   - Compute: footprint area, estimated floors, gross floor area
   - Compute: adjacency analysis, shared wall ratio, building type
   - Output: `buildings_morphology.gpkg`

2. **`process_uprn.py`**
   - Input: buildings, OS UPRN, Census, EPCs
   - Spatial join: UPRNs → buildings
   - Spatial join: UPRNs → Census OAs
   - Attribute join: UPRNs → EPCs
   - Count: UPRNs per building (for flat detection)
   - Output: `uprn_integrated.gpkg`

3. **`process_network.py`**
   - Input: UPRN dataset, OS Open Roads
   - Build network graph via cityseer
   - Assign UPRNs to nearest street segments
   - Compute 400m catchment aggregations
   - Output: `segments_analysis.gpkg`
