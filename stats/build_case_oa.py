"""
Single entrypoint to regenerate the OA-level case figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import basket_index_oa  # noqa: E402
import oa_figures  # noqa: E402


def main(cities: list[str] | None = None) -> None:
    print("=" * 70)
    print("BUILD OA CASE (THREE SURFACES + BASKET)")
    print("=" * 70)

    print("\n[1/2] Generating three-surfaces figures (OA)...")
    oa_figures.main(cities=cities)

    print("\n[2/2] Generating basket figures (OA)...")
    basket_index_oa.main(cities=cities)


if __name__ == "__main__":
    _cities = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(cities=_cities or None)
