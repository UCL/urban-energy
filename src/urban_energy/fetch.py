"""
Shared HTTP download + cache helpers for the data-acquisition scripts.

Every ``data/download_*.py`` script previously carried its own copy of the same
"download a URL, stream to disk, cache it" logic. This module holds the single
canonical implementation so the User-Agent, timeout defaults, and progress-bar
behaviour stay consistent across sources.

``download_and_cache`` additionally verifies a SHA-256 digest when one is
recorded for the file in the data-source manifest (``data/manifest.json``), so a
corrupt or truncated cache is caught rather than silently reused.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import requests
from tqdm import tqdm

#: Standard User-Agent for every request the pipeline makes.
USER_AGENT = "urban-energy-research/1.0"
DEFAULT_HEADERS: dict[str, str] = {"User-Agent": USER_AGENT}
DEFAULT_TIMEOUT = 600
_HASH_CHUNK = 1 << 20  # 1 MiB read blocks when hashing.

#: Data-source manifest (URL / filename / SHA-256 per source). Resolved from the
#: package location so importing this module never requires the data-dir env var.
_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "data" / "manifest.json"


def sha256_file(path: Path) -> str:
    """
    Return the hex SHA-256 digest of a file, read in fixed-size chunks.

    Parameters
    ----------
    path : Path
        File to digest.

    Returns
    -------
    str
        Lower-case hexadecimal SHA-256 digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_HASH_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, dict[str, object]]:
    """
    Load the data-source manifest, or an empty mapping if it is absent.

    Returns
    -------
    dict[str, dict[str, object]]
        The ``sources`` mapping from ``data/manifest.json`` keyed by source id.
    """
    if not _MANIFEST_PATH.exists():
        return {}
    with open(_MANIFEST_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    sources = data.get("sources", {})
    return sources if isinstance(sources, dict) else {}


def manifest_sha256(filename: str) -> str | None:
    """
    Return the recorded SHA-256 for a cache filename, if the manifest has one.

    Parameters
    ----------
    filename : str
        Cache filename (basename) to look up.

    Returns
    -------
    str | None
        The recorded digest, or None when the file is absent from the manifest
        or its hash slot is still blank.
    """
    for entry in load_manifest().values():
        if entry.get("filename") == filename:
            digest = entry.get("sha256")
            return digest if isinstance(digest, str) and digest else None
    return None


def verify_sha256(path: Path, expected: str) -> None:
    """
    Raise if a file's SHA-256 does not match the expected digest.

    Parameters
    ----------
    path : Path
        File to verify.
    expected : str
        Expected hexadecimal SHA-256 digest.

    Raises
    ------
    ValueError
        If the computed digest does not match ``expected``.
    """
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}. "
            "The cached file is corrupt or out of date; delete it and re-download."
        )


def download_file(
    url: str,
    dest: Path,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    headers: dict[str, str] | None = None,
    show_progress: bool = True,
) -> Path:
    """
    Stream a URL to ``dest`` on disk, with an optional tqdm progress bar.

    The download is written to a sibling ``.part`` file and atomically renamed
    on success, so an interrupted download never leaves a truncated file that a
    later run would mistake for a complete cache entry.

    Parameters
    ----------
    url : str
        URL to download.
    dest : Path
        Destination file path.
    timeout : int
        Request timeout in seconds.
    headers : dict[str, str] | None
        Request headers; defaults to the standard research User-Agent.
    show_progress : bool
        Whether to render a tqdm progress bar.

    Returns
    -------
    Path
        The written ``dest`` path.
    """
    headers = headers or DEFAULT_HEADERS
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / (dest.name + ".part")

    response = requests.get(url, stream=True, timeout=timeout, headers=headers)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(tmp, "wb") as fh:
        if show_progress:
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc=dest.name
            ) as pbar:
                for chunk in response.iter_content(chunk_size=65536):
                    fh.write(chunk)
                    pbar.update(len(chunk))
        else:
            for chunk in response.iter_content(chunk_size=65536):
                fh.write(chunk)

    os.replace(tmp, dest)
    return dest


def download_bytes(
    url: str,
    *,
    desc: str = "",
    timeout: int = DEFAULT_TIMEOUT,
    headers: dict[str, str] | None = None,
    show_progress: bool = True,
) -> bytes:
    """
    Download a URL fully into memory, with an optional tqdm progress bar.

    For callers that post-process the bytes (extract a zip member, parse a CSV)
    before deciding how to cache them.

    Parameters
    ----------
    url : str
        URL to download.
    desc : str
        Progress-bar description.
    timeout : int
        Request timeout in seconds.
    headers : dict[str, str] | None
        Request headers; defaults to the standard research User-Agent.
    show_progress : bool
        Whether to render a tqdm progress bar.

    Returns
    -------
    bytes
        The downloaded content.
    """
    headers = headers or DEFAULT_HEADERS
    response = requests.get(url, stream=True, timeout=timeout, headers=headers)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    chunks: list[bytes] = []
    if show_progress:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                chunks.append(chunk)
                pbar.update(len(chunk))
    else:
        for chunk in response.iter_content(chunk_size=8192):
            chunks.append(chunk)
    return b"".join(chunks)


def download_and_cache(
    url: str,
    cache_path: Path,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    headers: dict[str, str] | None = None,
    expected_sha256: str | None = None,
    show_progress: bool = True,
) -> Path:
    """
    Return a cached copy of ``url``, downloading to ``cache_path`` if absent.

    When a SHA-256 is supplied (or recorded for this filename in the data-source
    manifest), the cached file is verified and a mismatch raises. Hashes in the
    manifest are optional; if none is recorded, no verification is performed.

    Parameters
    ----------
    url : str
        URL to download.
    cache_path : Path
        Full path the file is cached at (its basename is the cache filename).
    timeout : int
        Request timeout in seconds.
    headers : dict[str, str] | None
        Request headers; defaults to the standard research User-Agent.
    expected_sha256 : str | None
        Explicit expected digest. When None, the manifest is consulted.
    show_progress : bool
        Whether to render a tqdm progress bar.

    Returns
    -------
    Path
        Path to the cached file.
    """
    if expected_sha256 is None:
        expected_sha256 = manifest_sha256(cache_path.name)

    if cache_path.exists():
        size_mb = cache_path.stat().st_size / 1e6
        print(f"  Loading cached {cache_path.name} ({size_mb:.1f} MB)")
        if expected_sha256:
            verify_sha256(cache_path, expected_sha256)
        return cache_path

    print(f"  Downloading {cache_path.name}...")
    download_file(
        url,
        cache_path,
        timeout=timeout,
        headers=headers,
        show_progress=show_progress,
    )
    print(f"  Cached to {cache_path}")
    if expected_sha256:
        verify_sha256(cache_path, expected_sha256)
    return cache_path
