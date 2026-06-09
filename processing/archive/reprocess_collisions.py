"""
Reprocess BUAs whose sanitised names collided with another BUA pre-fix.

Until 2026-05-07 the OA pipeline keyed per-BUA outputs by `_sanitise_name(BUA22NM)`.
329 sanitised names collide across 771 BUA22CDs — colliding BUAs shared an
output folder so each overwrote the others as they processed in turn.

The path-key fix landed in pipeline_oa.py (`_bua_dir` now uses BUA22CD as a
prefix). This script identifies the 771 affected BUA22CDs and reprocesses
them under the new scheme, then re-runs the combine step. Existing per-BUA
caches are skipped where present at the new path.

Usage:
    uv run python processing/reprocess_collisions.py
    uv run python processing/reprocess_collisions.py --dry-run
"""

from __future__ import annotations

import argparse
import collections
import gc
import sys
import traceback
from pathlib import Path

# Allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline_oa import (  # noqa: E402
    _bua_dir,
    combine_oa_cities,
    load_all_buas,
    process_city,
)


def _collisions(buas: dict[str, str]) -> dict[str, str]:
    """Return BUA22CD → name for codes whose sanitised name collides."""
    name_to_codes: dict[str, list[str]] = collections.defaultdict(list)
    for code, name in buas.items():
        name_to_codes[name].append(code)
    affected: dict[str, str] = {}
    for name, codes in name_to_codes.items():
        if len(codes) > 1:
            for code in codes:
                affected[code] = name
    return affected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List affected BUAs without reprocessing",
    )
    parser.add_argument(
        "--skip-combine", action="store_true",
        help="Skip the combine step at the end (run later via pipeline_oa.py)",
    )
    args = parser.parse_args()

    buas = load_all_buas()
    affected = _collisions(buas)
    print(f"Total BUAs:                {len(buas):,}")
    print(f"BUAs with name collisions: {len(affected):,}")
    print()

    # Filter to those that don't already have new-scheme output
    todo = {
        code: name for code, name in affected.items()
        if not (_bua_dir(code, name) / "oa_integrated.gpkg").exists()
    }
    print(f"To reprocess (missing new-scheme output): {len(todo):,}")
    if args.dry_run:
        for i, (code, name) in enumerate(sorted(todo.items())):
            print(f"  [{i + 1}/{len(todo)}] {code} / {name}")
        return

    if not todo:
        print("Nothing to do.")
        return

    print()
    print("=" * 60)
    print("REPROCESSING")
    print("=" * 60)
    failed: list[tuple[str, str]] = []
    for i, (code, name) in enumerate(sorted(todo.items())):
        print(f"\n[{i + 1}/{len(todo)}] {code} / {name}")
        try:
            process_city(code, name)
        except Exception:
            print(f"  FAILED — {code} / {name}")
            traceback.print_exc()
            failed.append((code, name))
        gc.collect()

    print()
    print("=" * 60)
    print(f"COMPLETE — {len(todo) - len(failed)} reprocessed, {len(failed)} failed")
    print("=" * 60)
    if failed:
        print("\nFailed BUAs:")
        for code, name in failed:
            print(f"  {code} / {name}")

    if args.skip_combine:
        print("\nSkipping combine step (--skip-combine).")
        print("Run later: uv run python -c \"from pipeline_oa import "
              "combine_oa_cities, load_all_buas; "
              "combine_oa_cities(load_all_buas())\"")
        return

    print("\nRebuilding combined GeoPackage...")
    combine_oa_cities(buas)
    print("\nDone. Re-run `uv run python stats/export_atlas_data.py` to refresh "
          "the Atlas cache.")


if __name__ == "__main__":
    main()
