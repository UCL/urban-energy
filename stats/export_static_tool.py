"""
Export the trained NEPI models to the static planning tool's JSON.

Serialises the four XGBoost models (form / mobility / cars / commute) into the
compact tree encoding that ``stats/nepi_static/index.html`` evaluates in the
browser (no Python runtime), bundled with the band thresholds and archetype
profiles. This replaces the inline one-liner previously kept in CLAUDE.md §5 so
the export is a first-class, orchestrated stage (``pipeline … run static_tool``)
and can never silently drift from the trained models.

Run after ``stats/nepi_model.py``. The orchestrator's ``mirror_docs`` stage then
copies the result into ``docs/`` for GitHub Pages.

Output: ``stats/nepi_static/nepi_models.json``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))

from nepi_model import MODEL_DIR, MODEL_FEATURES  # noqa: E402

from urban_energy.paths import PROJECT_DIR  # noqa: E402

OUT = PROJECT_DIR / "stats" / "nepi_static" / "nepi_models.json"


def _extract(path: Path, features: list[str]) -> dict[str, Any]:
    """
    Serialise one XGBoost model into the browser tree encoding.

    Each tree becomes a flat node list; internal nodes carry the split feature
    ``f``, threshold ``t``, and the indices of their yes/no children (``y``/``n``
    point at the position the child's subtree begins). Leaves carry ``leaf``.
    """
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    dump = model.get_booster().get_dump(dump_format="json")

    trees: list[list[dict[str, Any]]] = []
    for raw in dump:
        nodes: list[dict[str, Any]] = []
        tree = json.loads(raw)

        def walk(node: dict[str, Any]) -> None:
            if "leaf" in node:
                nodes.append({"leaf": node["leaf"]})
                return
            nd: dict[str, Any] = {
                "f": node["split"],
                "t": node["split_condition"],
                "y": None,
                "n": None,
            }
            nodes.append(nd)
            for child in node["children"]:
                if child["nodeid"] == node.get("yes", 0):
                    nd["y"] = len(nodes)
                    walk(child)
                elif child["nodeid"] == node.get("no", 0):
                    nd["n"] = len(nodes)
                    walk(child)
                else:
                    walk(child)

        walk(tree)
        trees.append(nodes)

    return {
        "features": features,
        "base_score": 0.0,
        "n_trees": len(trees),
        "trees": trees,
    }


def main() -> None:
    models = {
        name: _extract(MODEL_DIR / f"nepi_model_{name}.json", MODEL_FEATURES[name])
        for name in MODEL_FEATURES
    }
    with open(MODEL_DIR / "nepi_band_thresholds.json") as f:
        bands = json.load(f)
    with open(MODEL_DIR / "nepi_archetype_profiles.json") as f:
        archetypes = json.load(f)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(
            {"models": models, "band_thresholds": bands, "archetypes": archetypes},
            f,
            separators=(",", ":"),
        )
    print(f"Exported {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
