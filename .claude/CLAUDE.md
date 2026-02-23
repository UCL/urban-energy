# Urban Energy Project Guidelines

## Project Overview

This project investigates the relationship between urban form (morphology, density, accessibility) and per-capita energy consumption in England. It combines geospatial data processing with academic research on sustainable urban development.

**Author:** Gareth Simons
**License:** GPL-3.0-only

## Project Structure

```
urban-energy/
├── src/urban_energy/    # Python package source code
├── paper/               # Academic paper and literature
├── tests/               # Test suite
├── data/                # Data files (not in version control)
└── .claude/             # Claude Code configuration
```

## Development Standards

### Python Code Quality

1. **Type Annotations:** All functions must have complete type hints. Use modern Python typing (3.10+) including `|` union syntax and `list[]`/`dict[]` generics.

2. **Docstrings:** Use NumPy-style docstrings for all public functions and classes:

   ```python
   def calculate_density(geometries: gpd.GeoSeries, population: pd.Series) -> pd.Series:
       """
       Calculate population density per unit area.

       Parameters
       ----------
       geometries : gpd.GeoSeries
           Polygon geometries with projected CRS (units in meters).
       population : pd.Series
           Population counts aligned with geometries index.

       Returns
       -------
       pd.Series
           Population density in persons per square kilometer.

       Raises
       ------
       ValueError
           If CRS is geographic (unprojected) or series lengths mismatch.
       """
   ```

3. **Linting:** Code must pass `ruff check` and `ruff format`. Configuration in pyproject.toml.

4. **Type Checking:** Code must pass `ty check` (or `uv run ty check`).

5. **Testing:** Write pytest tests for all non-trivial functions. Tests should cover edge cases and validate geospatial operations with known inputs.

### Geospatial Best Practices

1. **Coordinate Reference Systems (CRS):**
   - Always explicitly set and validate CRS
   - Use EPSG:27700 (British National Grid) for UK analyses requiring metric distances/areas
   - Use EPSG:4326 (WGS84) only for data interchange
   - Transform early, validate often: `gdf = gdf.to_crs(epsg=27700)`

2. **Spatial Joins and Operations:**
   - Prefer `sjoin` with explicit `predicate` parameter
   - Use spatial indices: ensure `.sindex` is built for large datasets
   - Validate geometries before operations: `gdf.geometry.is_valid`

3. **Memory Efficiency:**
   - Use appropriate dtypes (e.g., `float32` vs `float64`, categorical for string columns)
   - Process large datasets in chunks when necessary
   - Release memory explicitly with `del` and `gc.collect()` for large intermediates

4. **Raster Operations:**
   - Always use context managers with rasterio: `with rasterio.open(path) as src:`
   - Respect nodata values in calculations
   - Match raster and vector CRS before extraction

5. **Reproducibility:**
   - Set random seeds where stochastic processes are involved
   - Log or document data source versions and access dates
   - Use deterministic ordering in spatial operations where possible

### Data Handling

1. **Data Provenance:** Document all data sources, licenses, and access dates in code comments or metadata files.

2. **Sensitive Data:** Never commit raw data containing personal information. Use aggregated or anonymized derivatives.

3. **File Paths:** Use `pathlib.Path` for all file operations. Prefer relative paths from project root.

4. **Data Validation:** Validate inputs at function boundaries. Check for:
   - Expected columns present
   - CRS matches expectations
   - No unexpected null values in critical fields
   - Geometry validity

## Academic Writing Standards

### Paper Guidelines

1. **Style:** Follow formal academic conventions. Avoid contractions, colloquialisms, and first-person singular.

2. **Citations:** Use author-date format (APA/Harvard style). All claims require supporting citations.

3. **Precision:** Quantitative claims must specify units, confidence intervals where appropriate, and data sources.

4. **Reproducibility:** Methods sections should provide sufficient detail for independent replication.

5. **Structure:** Follow standard IMRaD structure (Introduction, Methods, Results, and Discussion) where applicable.

### Literature Review

- Critically evaluate sources; do not merely summarize
- Identify gaps in existing research
- Connect sources thematically rather than serially
- Prioritize peer-reviewed sources; note preprints explicitly

## Commands Reference

```bash
# Development
uv sync                      # Install dependencies
uv run pytest                # Run tests
uv run ruff check .          # Lint code
uv run ruff format .         # Format code
uv run ty check              # Type check

# Git workflow
git status                   # Check changes
git diff                     # Review changes
git log --oneline -10        # Recent history
```

## Code Review Checklist

Before committing, verify:

- [ ] All functions have type annotations
- [ ] Public functions have docstrings
- [ ] `ruff check .` passes
- [ ] `ruff format --check .` passes
- [ ] `ty check` passes (or type errors are intentional/documented)
- [ ] Tests pass: `uv run pytest`
- [ ] CRS handling is explicit and correct
- [ ] No hardcoded absolute paths
- [ ] No sensitive data included
- [ ] Commit message is descriptive and follows conventional format

## Core Thesis: The Trophic Layers Framework

### The Argument

Cities are conduits that capture energy and recycle it through layers of human interaction
(Jacobs, 2000, "The Nature of Economies"). The measure of urban efficiency is not how much
energy a neighbourhood consumes, but how many transactions, connections, and transformations
that energy enables before it dissipates. This connects to Bettencourt et al.'s (2007)
urban scaling laws: cities scale superlinearly in socioeconomic output (~N^1.15) and
sublinearly in infrastructure (~N^0.85). Proximity is the mechanism.

### The Rainforest Analogy

A rainforest and a desert receive the same solar radiation per m². The difference is
**trophic depth** — how many times energy is captured and re-used before it dissipates.

- **Rainforest** (dense urban neighbourhood): Energy passes through dozens of layers —
  canopy, understorey, epiphytes, soil biome, mycorrhizal networks. Each layer captures
  energy from the one above. Thousands of species, millions of interactions.
- **Desert** (suburban sprawl): Energy hits the ground and radiates straight back out.
  One trophic level. Minimal recycling.

The urban equivalent: a 1km × 1km inner-city neighbourhood has thousands of land uses,
shops, schools, transit stops, green spaces — each a layer in the conduit. A 1km × 1km
suburban plot has a handful of destinations. Same energy input (building + transport),
radically different interaction depth.

### The Trophic Layers (mapped to our data)

Each layer captures a different dimension of how energy is recycled through the urban system:

| Layer | Ecological equivalent | Urban function | Our metric |
|-------|----------------------|----------------|------------|
| **Physical substrate** | Soil/root network | Street network connectivity | `cc_harmonic_800`, `cc_density_800` |
| **Commercial exchange** | Canopy photosynthesis | Places where economic transactions happen | `cc_fsa_restaurant_800_wt`, `cc_fsa_pub_800_wt`, `cc_fsa_takeaway_800_wt` |
| **Mobility** | Seed dispersal/pollinators | Connections to wider city network | `cc_bus_800_wt`, `cc_rail_800_wt` |
| **Recreation/restoration** | Water cycle/shade | Green space — regenerative capacity | `cc_greenspace_800_wt` |

All metrics at 800m (~10 min walk), the pedestrian catchment. The `_wt` suffix means
gravity-weighted count (more establishments closer = higher score).

### The Compounding Effect

The proof of concept demonstrates that each successive normalisation widens the efficiency
gap between compact and sprawling morphologies:

1. **kWh/m²** — Building physics alone. Modest difference (~1.04x detached vs flat).
2. **kWh/capita** — Add transport, normalise per person. Gap widens (~1.76x).
3. **kWh/capita/accessibility** — Ask what you GET for the energy. Gap widens further (~2.79x).

This is the compounding: sprawl is not just energy-costly, it delivers less city per unit
of energy consumed. The energy pours through the conduit and is not recoverable.

### The PoC Structure (stats/proof_of_concept.py)

| Step | Test | Role |
|------|------|------|
| 1. Physics signatures | Types have distinct thermal envelopes | Foundation |
| 2. Physics → energy | Physics predicts SAP + metered energy | Foundation |
| 3. Accessibility signatures | Types differ across ALL trophic layers | Sets up compounding |
| 4. The compounding | Three normalisations, gap widens at each level | **Centerpiece** |
| 5. Deprivation control | Compounding holds within deprivation quintiles | Rules out wealth |
| 6. Lock-in | Stock composition locks inefficiency in for decades | Policy implication |

### Key References

- Jacobs, J. (2000). *The Nature of Economies*. Random House. — Cities as ecosystems
- Bettencourt, L.M.A. et al. (2007). "Growth, innovation, scaling, and the pace of life in cities." *PNAS*, 104(17). — Superlinear/sublinear scaling
- Norman, J. et al. (2006). "Comparing high and low residential density." *J. Urban Planning*. — Functional unit matters (per m² vs per capita)
- Newman, P. & Kenworthy, J. (1989). *Cities and Automobile Dependence*. — Density-transport energy
- Rode, P. et al. (2014). "Cities and energy: urban morphology and residential heat-energy demand." *Env. & Planning B*. — S/V ratio and building physics
- Few, J. et al. (2023). "The over-prediction of energy use by EPCs." *Energy & Buildings*. — SAP performance gap
- Ewing, R. & Cervero, R. (2010). "Travel and the built environment: A meta-analysis." *JAPA*. — Destination accessibility > density

## Dependencies

Core geospatial stack:

- **geopandas:** Vector data operations
- **shapely:** Geometry primitives
- **rasterio:** Raster I/O and operations
- **pyproj:** CRS transformations (via geopandas)

Analysis:

- **numpy:** Numerical operations
- **pandas:** Tabular data
- **scipy:** Statistical methods

Development:

- **ruff:** Linting and formatting
- **ty:** Type checking
- **pytest:** Testing framework
