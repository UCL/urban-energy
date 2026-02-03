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
