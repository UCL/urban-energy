# Urban Form as Energy Infrastructure

**Quantifying the Systemic Efficiency of Compact Morphologies in the United Kingdom**

## Project Overview

This project investigates the relationship between urban morphology and building energy consumption across England and Wales. While transport-energy relationships with urban form are well-established, building energy remains under-quantified in the UK context. We address this gap by constructing a high-resolution spatial dataset integrating Energy Performance Certificates, street network analysis, and socio-demographic controls.

## Documents

- **[main.tex](main.tex)** - Full academic paper (LaTeX)
- **[references.bib](references.bib)** - BibTeX bibliography
- **[literature_review.md](literature_review.md)** - Detailed literature review and methodology notes

## Research Questions

1. How strongly do urban morphological characteristics correlate with per-capita building energy consumption in UK cities?
2. Which morphological features (density, compactness, mixed-use, connectivity) have the largest effect sizes?
3. Do these relationships persist after controlling for building fabric, tenure, and socio-economic factors?
4. At what spatial scales are morphology-energy associations most pronounced?

## Data Sources

| Dataset                         | Coverage        | Resolution     | Source                                                        |
| ------------------------------- | --------------- | -------------- | ------------------------------------------------------------- |
| Energy Performance Certificates | England & Wales | Address (UPRN) | [EPC Open Data](https://epc.opendatacommunities.org/)         |
| Census 2021                     | England & Wales | Output Area    | [ONS](https://www.ons.gov.uk/census)                          |
| Building Footprints             | England         | Building-level | [OS Open Map Local](https://osdatahub.os.uk/downloads/open)   |
| Building Heights                | England         | 1m raster      | [Environment Agency LiDAR](https://environment.data.gov.uk/)  |
| Street Network                  | Great Britain   | Segment-level  | [OS Open Roads](https://osdatahub.os.uk/downloads/open)       |
| Address Coordinates             | Great Britain   | Point (UPRN)   | [OS Open UPRN](https://osdatahub.os.uk/downloads/open)        |

## Methodology Summary

**Analysis Unit:** UPRN (Unique Property Reference Number), aggregated to street segments

**Data Integration Workflow:**

1. **Building-level processing:** Compute morphology metrics on OS Open Map Local building footprints with LiDAR-derived heights
2. **UPRN linkage:** Spatial join buildings → UPRNs; interpolate Census from OAs; join EPCs on UPRN field
3. **Network aggregation:** Assign UPRNs to street segments via cityseer; compute 400m pedestrian catchment metrics

**Statistical Approach:** Multi-level regression controlling for:

- Building characteristics (type, age, construction, floor area)
- Socio-demographics (tenure, household composition, deprivation)
- Spatial clustering (random effects for segments and LSOAs)

**Key Morphological Metrics:**

- Density: Building density, FAR, population density
- Form: Building heights (LiDAR-derived), compactness, footprint area
- Network: Street density, connectivity, centrality
- Accessibility: Service proximity, mixed-use intensity

## Key Limitations

This research explicitly acknowledges:

1. **Correlation vs causation** - Observational analysis cannot establish causal effects
2. **EPC data quality** - Modelled estimates, not metered consumption; non-random coverage
3. **Embodied carbon** - Analysis focuses on operational energy; densification has lifecycle implications
4. **Equity considerations** - Densification has distributional consequences not captured here
5. **Rebound effects** - Per-capita savings may not translate to absolute emissions reductions

## Tools

- [cityseer](https://cityseer.benchmarkurbanism.com/) - Pedestrian-scale network analysis
- [momepy](https://momepy.org/) - Urban morphology metrics
- [OSMnx](https://osmnx.readthedocs.io/) - Street network data

## Building the Paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Citation

*[To be added upon publication]*

## License

GPL-3.0
