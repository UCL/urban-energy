# Urban Energy

Investigating the relationship between urban form and per-capita energy consumption across English Built-Up Areas at Output Area (OA) level.

## Thesis

Cities are conduits that capture energy and recycle it through layers of human interaction. The measure of urban efficiency is not how much energy a neighbourhood consumes, but how many transactions, connections, and transformations that energy enables before it dissipates. Sprawling morphologies consume more energy per capita and deliver less city per unit of energy consumed.

**Key result (OA level, 94 BUAs, 67,263 OAs):** The compounding efficiency gap widens at each normalisation level: 1.55x (building) to 1.83x (transport) to 2.69x (accessibility) between detached-dominant and flat-dominant OAs.

## Project Structure

| Folder | Purpose |
| ------ | ------- |
| [data/](data/README.md) | Data acquisition (Census, DESNZ, EPC, LiDAR, FSA, NaPTAN, GIAS, NHS ODS) |
| [processing/](processing/README.md) | Building morphology + OA aggregation pipeline |
| [stats/](stats/README.md) | Statistical analysis and figure generation (OA level) |
| [paper/](paper/README.md) | Academic paper and literature review |
| [docs/](docs/) | Archived working notes from earlier phases |

## Quick Start

```bash
# Regenerate all OA-level figures and tables
uv run python stats/build_case_oa.py

# Run the national OA pipeline (all 7,147 BUAs, skip-if-exists)
uv run python processing/pipeline_oa.py
```

Output: figures in `stats/figures/oa/` and `stats/figures/basket_oa/`, narrative in `paper/case_v2.md`.

## Archived Work

The original LSOA-level analysis (18 cities, 3,678 LSOAs) is preserved in archive directories:

- `stats/archive/` -- LSOA analysis scripts
- `stats/figures/archive_lsoa/` -- LSOA figures
- `processing/archive/` -- LSOA pipeline
- `paper/archive/` -- LSOA case narrative and stale LaTeX

## License

GPL-3.0-only
