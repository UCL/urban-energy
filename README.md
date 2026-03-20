# Urban Energy

Investigating the relationship between urban form and per-capita energy consumption across English Built-Up Areas at Output Area (OA) level.

## Thesis

Cities are conduits that capture energy and recycle it through layers of human interaction. The measure of urban efficiency is not how much energy a neighbourhood consumes, but how many transactions, connections, and transformations that energy enables before it dissipates. Sprawling morphologies consume more energy per capita and deliver less city per unit of energy consumed.

**Key result (198,779 OAs across 6,687 BUAs):** The compounding efficiency gap widens at each surface: 1.46x (building) to 1.67x (total energy) to 2.68x (energy per unit access) between detached-dominant and flat-dominant OAs.

## NEPI Planning Tool

The Neighbourhood Energy Performance Index (NEPI) is an interactive tool that predicts household energy costs from neighbourhood morphology. Four XGBoost models predict Form (building energy), Mobility (transport energy), car ownership, and commute distance from planner-controllable inputs.

### Static version (no server required)

Two files, runs in any browser:

```bash
cd stats/nepi_static
python3 -m http.server 8501
# Open http://localhost:8501
```

Host on GitHub Pages, any static file server, or open locally.

### Streamlit version (interactive, with SHAP explanations)

```bash
uv run python stats/nepi_model.py          # train models (one-time, ~2 min)
uv run streamlit run stats/nepi_app.py     # launch interactive app
```

### Reproducing from scratch

```bash
# 1. Process OA data (requires external storage + raw data)
uv run python processing/pipeline_oa.py

# 2. Regenerate all figures, tables, and NEPI scores
uv run python stats/build_case_oa.py
uv run python stats/nepi.py
uv run python stats/access_penalty_model.py

# 3. Train NEPI planning tool models
uv run python stats/nepi_model.py

# 4. Export models for static tool
uv run python -c "
import json, xgboost as xgb, sys
sys.path.insert(0, 'stats')
from nepi_model import MODEL_FEATURES
from pathlib import Path

MODEL_DIR = Path('/Volumes/1TB/urban-energy/temp/models/nepi')
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

# 5. Launch static tool
cd stats/nepi_static && python3 -m http.server 8501
```

## Project Structure

| Folder | Purpose |
| ------ | ------- |
| [data/](data/README.md) | Data acquisition (Census, DESNZ, EPC, LiDAR, FSA, NaPTAN, GIAS, NHS ODS) |
| [processing/](processing/README.md) | Building morphology + OA aggregation pipeline |
| [stats/](stats/README.md) | Statistical analysis and figure generation (OA level) |
| [stats/nepi_model.py](stats/nepi_model.py) | XGBoost model training, SHAP, prediction API |
| [stats/nepi_app.py](stats/nepi_app.py) | Streamlit interactive planning tool |
| [stats/nepi_static/](stats/nepi_static/) | Static HTML planning tool (hostable anywhere) |
| [paper/](paper/README.md) | Academic paper and literature review |

## Quick Start

```bash
# Install dependencies
uv sync

# Regenerate all OA-level figures and tables
uv run python stats/build_case_oa.py

# Run the national OA pipeline (all 7,147 BUAs, skip-if-exists)
uv run python processing/pipeline_oa.py
```

Output: figures in `stats/figures/oa/` and `stats/figures/basket_oa/`, NEPI in `stats/figures/nepi/`, narrative in `paper/case_v2.md`.

## Archived Work

The original LSOA-level analysis (18 cities, 3,678 LSOAs) is preserved in archive directories:

- `stats/archive/` -- LSOA analysis scripts
- `stats/figures/archive_lsoa/` -- LSOA figures
- `processing/archive/` -- LSOA pipeline
- `paper/archive/` -- LSOA case narrative and stale LaTeX

## License

GPL-3.0-only
