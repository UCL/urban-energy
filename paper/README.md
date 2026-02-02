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

| Dataset | Coverage | Resolution | Source |
|---------|----------|------------|--------|
| Energy Performance Certificates | England & Wales | Address-level | [EPC Open Data](https://epc.opendatacommunities.org/) |
| Census 2021 | England & Wales | LSOA | [ONS](https://www.ons.gov.uk/census) |
| Street Network | National | Segment-level | OpenStreetMap via cityseer |
| Building Footprints | National | Building-level | OS MasterMap / OSM |
| Public Transport | National | Stop-level | NaPTAN |

## Methodology Summary

**Spatial Unit:** Street network segments (400m pedestrian catchment)

**Statistical Approach:** Multi-level regression controlling for:
- Building characteristics (type, age, construction)
- Socio-demographics (tenure, household composition, deprivation)
- Spatial clustering (random effects for segments and LSOAs)

**Key Morphological Metrics:**
- Density: Building density, FAR, population density
- Form: Building heights, compactness, block size
- Network: Street density, connectivity, centrality
- Accessibility: PT access, retail proximity, mixed-use intensity

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
