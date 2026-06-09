"""Centralized path configuration for the urban-energy project."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Source repo root (for stats/figures/ outputs)
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

# Load .env from repo root
load_dotenv(PROJECT_DIR / ".env")

_base = os.environ.get("URBAN_ENERGY_DATA_DIR")
if _base is None:
    raise EnvironmentError(
        "URBAN_ENERGY_DATA_DIR is not set. "
        "Create a .env file in the project root with: "
        "URBAN_ENERGY_DATA_DIR=/path/to/temp"
    )
_base = Path(_base)

# All datasets live directly under URBAN_ENERGY_DATA_DIR (no nested "data/").
DATA_DIR = _base

# Per-BUA pipeline processing outputs
PROCESSING_DIR = _base / "processing"

# Download caches
CACHE_DIR = _base / "cache"


def latest_uprn_gpkg() -> Path | None:
    """
    Return the OS Open UPRN GeoPackage under DATA_DIR, whatever vintage.

    The OS release is date-stamped (e.g. ``osopenuprn_202605_gpkg``); globbing
    means the pinned vintage never has to be edited when a newer one is used.
    """
    matches = sorted(DATA_DIR.glob("osopenuprn_*_gpkg/osopenuprn_*.gpkg"))
    return matches[-1] if matches else None


def epc_input_dir() -> Path | None:
    """
    Return the raw domestic-EPC input directory, supporting both packagings.

    Accepts the per-year ``domestic-csv/`` bulk export or the per-local-authority
    ``all-domestic-certificates/`` download, whichever is present.
    """
    for name in ("domestic-csv", "all-domestic-certificates"):
        candidate = DATA_DIR / name
        if candidate.exists():
            return candidate
    return None
