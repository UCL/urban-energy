"""
Single entrypoint to regenerate the pilot case figures for `paper/case_v1.md`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import basket_index_v1  # noqa: E402
import lsoa_figures  # noqa: E402


def main(cities: list[str] | None = None) -> None:
    print("=" * 70)
    print("BUILD CASE V1 (THREE SURFACES + BASKET V1)")
    print("=" * 70)

    print("\n[1/2] Generating three-surfaces figures and summary tables...")
    lsoa_figures.main(cities=cities)

    print("\n[2/2] Generating basket-v1 figures and tables...")
    basket_index_v1.main(cities=cities)

    print("\nOutput")
    print("  paper/case_v1.md")


if __name__ == "__main__":
    _cities = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(cities=_cities or None)
