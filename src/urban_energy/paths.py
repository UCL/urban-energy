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

# All datasets
DATA_DIR = _base / "data"

# Per-BUA pipeline processing outputs
PROCESSING_DIR = _base / "processing"

# Download caches
CACHE_DIR = _base / "cache"
