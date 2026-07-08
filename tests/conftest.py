"""
Pytest bootstrap: make the data layer importable with no downloaded data.

Two things must be true before any ``urban_energy`` or ``data/`` import runs:

1. ``urban_energy.paths`` raises unless ``URBAN_ENERGY_DATA_DIR`` is set. In CI
   there is no ``.env`` and no data, so a throwaway placeholder is supplied. On a
   developer machine with a real ``.env`` the placeholder is NOT set, so
   python-dotenv still populates the real path.
2. The ``data/`` scripts are a flat directory, not an installed package, so the
   directory is added to ``sys.path`` for direct import by the tests.
"""

import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Only fill the env var when it is genuinely unset AND no .env will supply it,
# so a real local configuration always wins.
if "URBAN_ENERGY_DATA_DIR" not in os.environ and not (_REPO_ROOT / ".env").exists():
    os.environ["URBAN_ENERGY_DATA_DIR"] = tempfile.gettempdir()

for _extra in (_REPO_ROOT / "data", _REPO_ROOT / "src", _REPO_ROOT / "stats"):
    _path = str(_extra)
    if _path not in sys.path:
        sys.path.insert(0, _path)
