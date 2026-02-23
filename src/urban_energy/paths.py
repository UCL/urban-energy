"""Centralized path configuration for the urban-energy project."""

from pathlib import Path

# Project root (source code repository)
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

# External storage for large data, cache, and temporary files
STORAGE_DIR = Path("/Volumes/1TB/urban-energy")

# Temporary processing outputs
TEMP_DIR = STORAGE_DIR / "temp"

# Download caches
CACHE_DIR = STORAGE_DIR / "cache"
