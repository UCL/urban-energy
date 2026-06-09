"""
Single entrypoint to regenerate the OA-level case figures.

The illustrative basket index (stats/archive/basket_index_oa.py) was retired
from the case build: it is a cross-check whose outputs feed neither the paper's
headline result nor the Atlas. See ROADMAP.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import oa_figures  # noqa: E402


def main(cities: list[str] | None = None) -> None:
    print("=" * 70)
    print("BUILD OA CASE (THREE SURFACES)")
    print("=" * 70)

    print("\n[1/1] Generating three-surfaces figures (OA)...")
    oa_figures.main(cities=cities)


if __name__ == "__main__":
    _cities = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(cities=_cities or None)
