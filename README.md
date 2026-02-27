# Urban Energy

Investigating the relationship between urban form and per-capita energy consumption across 18 English cities at LSOA level.

## Thesis

Cities are conduits that capture energy and recycle it through layers of human interaction. The measure of urban efficiency is not how much energy a neighbourhood consumes, but how many transactions, connections, and transformations that energy enables before it dissipates. Sprawling morphologies consume more energy per capita and deliver less city per unit of energy consumed.

**Key result:** For selected land-use categories (food, health, education, greenspace, transit) at observed trip rates, a 3.5x access penalty emerges between detached-dominant and flat-dominant LSOAs. The efficiency gap widens at each normalisation level (building -> transport -> accessibility). The basket is illustrative — it covers particular land uses, not all travel purposes.

## Project Structure

| Folder | Purpose |
| ------ | ------- |
| [data/](data/README.md) | Data acquisition (Census, DESNZ, EPC, LiDAR, FSA, NaPTAN, GIAS, NHS ODS) |
| [processing/](processing/README.md) | Building morphology + LSOA aggregation pipeline |
| [stats/](stats/README.md) | Statistical analysis and figure generation |
| [paper/](paper/README.md) | Academic paper and literature review |
| [docs/](docs/) | Archived working notes from earlier phases |

## Quick Start

```bash
# Regenerate all case-one figures and tables
uv run python stats/build_case.py
```

Output: figures in `stats/figures/`, narrative in `paper/case_v1.md`.

## License

GPL-3.0-only
