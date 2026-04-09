# Urban Energy

Investigating the relationship between urban form and per-capita energy consumption across English Built-Up Areas at Output Area (OA) level. The output is the **Neighbourhood Energy Performance Index (NEPI)** — a place-level rating analogous to a building EPC, computed entirely from open data.

## Thesis

Cities are conduits that capture energy and recycle it through layers of human interaction. The measure of urban efficiency is not how much energy a neighbourhood consumes, but how many transactions, connections, and transformations that energy enables before it dissipates. Sprawling morphologies consume more energy per capita and deliver less city per unit of energy consumed.

**Key result (198,779 OAs across 6,687 BUAs):** The median flat-dominant OA scores **Band A** (15,982 kWh/hh/yr); the median detached-dominant OA scores **Band F** (26,897 kWh/hh/yr). The 10,915 kWh/hh/yr gap decomposes as Form 45% / Mobility 43% / Access 14%. The compounding gradient widens at each surface — 1.46x (building) → 1.67x (total energy) → 2.68x (energy per unit access).

The full case is in [paper/case_v2.md](paper/case_v2.md). Development status and forward work in [TODO.md](TODO.md).

## NEPI Planning Tool

The NEPI is also delivered as an interactive tool that predicts household energy costs from neighbourhood morphology. Four XGBoost models with monotonic constraints predict Form (building energy), Mobility (transport energy), car ownership, and commute distance from planner-controllable inputs (density, built form mix, walkable amenity access, transit access, building era).

### Live demo

**[https://UCL.github.io/urban-energy/](https://UCL.github.io/urban-energy/)** — static build deployed via GitHub Pages from [docs/](docs/).

### Static version (no server required)

Two files (`index.html` + `nepi_models.json`) — runs in any browser, no Python:

```bash
cd stats/nepi_static
python3 -m http.server 8501
# Open http://localhost:8501
```

The same files are mirrored in [docs/](docs/) for GitHub Pages hosting. Host on any static file server, or open locally.

### Streamlit version (interactive, with SHAP explanations)

```bash
uv run python stats/nepi_model.py          # train models (one-time, ~2 min)
uv run streamlit run stats/nepi_app.py     # launch interactive app
```

### Reproducing from scratch

```bash
# 0. Install + configure
uv sync
echo "URBAN_ENERGY_DATA_DIR=$(pwd)/temp" > .env

# 1. Acquire and prepare raw data (see data/README.md for full setup)
uv run python data/download_census.py
uv run python data/download_energy_postcode.py
uv run python data/download_imd.py
uv run python data/download_vehicles.py
uv run python data/download_fsa.py
uv run python data/download_naptan.py
uv run python data/prepare_gias.py
uv run python data/prepare_nhs.py
uv run python data/process_boundaries.py
uv run python data/process_lidar.py
uv run python data/build_postcode_oa_lookup.py
uv run python data/aggregate_energy_oa.py

# 2. Run the national OA pipeline (all 7,147 BUAs, skip-if-exists)
uv run python processing/pipeline_oa.py

# 3. Regenerate all OA figures, tables, NEPI scorecard, and access penalty model
uv run python stats/build_case_oa.py
uv run python stats/nepi.py
uv run python stats/access_penalty_model.py

# 4. Train the NEPI planning-tool XGBoost models
uv run python stats/nepi_model.py

# 5. Export trained models to JSON for the static tool
uv run python -c "
import json, xgboost as xgb, sys
sys.path.insert(0, 'stats')
from nepi_model import MODEL_DIR, MODEL_FEATURES
from pathlib import Path

OUT = Path('stats/nepi_static/nepi_models.json')

def extract(path, features):
    m = xgb.XGBRegressor(); m.load_model(path)
    dump = m.get_booster().get_dump(dump_format='json')
    trees = []
    for t in dump:
        nodes = []; tree = json.loads(t)
        def walk(n):
            if 'leaf' in n: nodes.append({'leaf': n['leaf']})
            else:
                nd = {'f': n['split'], 't': n['split_condition'], 'y': None, 'n': None}
                nodes.append(nd)
                for c in n['children']:
                    if c['nodeid'] == n.get('yes', 0): nd['y'] = len(nodes); walk(c)
                    elif c['nodeid'] == n.get('no', 0): nd['n'] = len(nodes); walk(c)
                    else: walk(c)
        walk(tree); trees.append(nodes)
    return {'features': features, 'base_score': 0.0, 'n_trees': len(trees), 'trees': trees}

models = {n: extract(MODEL_DIR/f'nepi_model_{n}.json', MODEL_FEATURES[n]) for n in MODEL_FEATURES}
with open(MODEL_DIR/'nepi_band_thresholds.json') as f: bands = json.load(f)
with open(MODEL_DIR/'nepi_archetype_profiles.json') as f: archetypes = json.load(f)
with open(OUT, 'w') as f: json.dump({'models': models, 'band_thresholds': bands, 'archetypes': archetypes}, f, separators=(',', ':'))
print(f'Exported {OUT} ({OUT.stat().st_size/1024:.0f} KB)')
"

# 6. Mirror the static build into docs/ for GitHub Pages
cp stats/nepi_static/index.html stats/nepi_static/nepi_models.json docs/

# 7. Launch the static tool locally
cd stats/nepi_static && python3 -m http.server 8501
```

## Project Structure

| Folder | Purpose |
| ------ | ------- |
| [data/](data/README.md) | Data acquisition (Census 2021/2011, DESNZ postcode energy, EPC, LiDAR, FSA, NaPTAN, GIAS, NHS ODS, IMD25, DVLA) |
| [processing/](processing/README.md) | Building morphology + national OA pipeline (CityNetwork API) |
| [stats/](stats/README.md) | Statistical analysis, NEPI scorecard, access penalty model, planning-tool models |
| [stats/nepi.py](stats/nepi.py) | NEPI scorecard (Form / Mobility / Access) and band figures |
| [stats/access_penalty_model.py](stats/access_penalty_model.py) | Empirical OLS access-energy penalty |
| [stats/nepi_model.py](stats/nepi_model.py) | XGBoost training (form / mobility / cars / commute) with SHAP |
| [stats/nepi_app.py](stats/nepi_app.py) | Streamlit interactive planning tool |
| [stats/nepi_static/](stats/nepi_static/) | Static HTML planning tool (mirrored to [docs/](docs/)) |
| [paper/](paper/README.md) | Case narrative ([case_v2.md](paper/case_v2.md)), data methodology, literature review |
| [notes/](notes/) | Archived v0 working notes (LSOA-era methodology snapshots) |

## Quick Start

```bash
# Install dependencies
uv sync

# Configure data directory (create .env in repo root)
echo "URBAN_ENERGY_DATA_DIR=$(pwd)/temp" > .env

# Regenerate all OA-level figures and tables
uv run python stats/build_case_oa.py

# Run the national OA pipeline (all 7,147 BUAs, skip-if-exists)
uv run python processing/pipeline_oa.py
```

The `.env` file (gitignored) sets `URBAN_ENERGY_DATA_DIR` to the base of your data storage. The repo expects this layout underneath it:

```text
$URBAN_ENERGY_DATA_DIR/
├── data/         # all datasets (statistics, boundaries, lidar, epc, …)
├── processing/   # per-BUA pipeline outputs
└── cache/        # download caches
```

Output: figures in [stats/figures/oa/](stats/figures/oa/) and [stats/figures/basket_oa/](stats/figures/basket_oa/), NEPI in [stats/figures/nepi/](stats/figures/nepi/), narrative in [paper/case_v2.md](paper/case_v2.md).

## Archived Work

The original LSOA-level analysis (18 cities, 3,678 LSOAs) is preserved in archive directories:

- [stats/archive/](stats/archive/) — LSOA analysis scripts
- [stats/figures/archive_lsoa/](stats/figures/archive_lsoa/) — LSOA figures
- [processing/archive/](processing/archive/) — LSOA pipeline (still imported by `pipeline_oa.py` for shared morphology constants)
- [paper/archive/](paper/archive/) — LSOA case narrative and stale LaTeX
- [notes/](notes/) — v0 working notes, methodology, roadmap

## License

GPL-3.0-only
