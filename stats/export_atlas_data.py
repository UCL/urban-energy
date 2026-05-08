"""
CLI entry point: export NEPI Atlas data for a region.

Usage:
    uv run python stats/export_atlas_data.py                      # default region
    uv run python stats/export_atlas_data.py greater_manchester
    uv run python stats/export_atlas_data.py manchester --no-cache
"""

import argparse

from atlas import REGIONS, export_region

DEFAULT_REGION = "greater_manchester"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "region", nargs="?", default=DEFAULT_REGION,
        help=f"Region slug (default: {DEFAULT_REGION}). "
             f"Available: {', '.join(REGIONS)}",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Recompute national NEPI from scratch (slow)",
    )
    args = parser.parse_args()

    if args.region not in REGIONS:
        parser.error(
            f"Unknown region {args.region!r}. "
            f"Available: {', '.join(REGIONS)}"
        )

    export_region(args.region, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
