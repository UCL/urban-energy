"""
Lean orchestrator for the urban-energy data acquisition.

Encodes the load-bearing downloads + OA aggregations as an executable, resumable
stage manifest. Each stage wraps a script and declares its output files, so the
runner skips stages whose outputs already exist. The products are the per-OA
tables the two-axis analysis assembles in the stats layer (``oa_data``). The one
heavy analysis artefact — the national road-network access curve
(``stats/oa_network_access.py``, cityseer over OS Open Roads, ~15 min) — is
built on demand from the stats layer, not here; the straight-line KD-tree
(``stats/oa_access.py``) remains as a fast cross-check.

Usage::

    uv run python -m urban_energy.pipeline doctor    # preflight checks
    uv run python -m urban_energy.pipeline status     # what's built
    uv run python -m urban_energy.pipeline list       # the manifest
    uv run python -m urban_energy.pipeline run --all  # build everything missing
    uv run python -m urban_energy.pipeline run census energy_oa

Then run the analysis on demand (prints to stdout)::

    uv run python stats/lock_in.py
    uv run python stats/access_profile.py
    uv run python stats/form_size_decomposition.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

MIN_FREE_GB = 80  # EPC raw (~55 GB) + UPRN dominate the footprint

# Size floor below which an output file is treated as truncated / not built. The
# smallest real product (a per-OA parquet) is comfortably above this; the floor
# only catches zero-byte or grossly-truncated writes from an interrupted run.
MIN_OUTPUT_BYTES = 1024


@dataclass(frozen=True)
class Stage:
    """One acquisition step: a wrapped script with declared outputs."""

    name: str
    outputs: tuple[Path, ...]
    argv: tuple[str, ...]
    note: str = ""

    def is_done(self) -> bool:
        """
        True only if EVERY declared output exists and clears the size floor.

        Existence alone is insufficient: an interrupted or failed write can
        leave a zero-byte or truncated file that must not count as built. All
        outputs of a multi-output stage are checked, not just the first.
        """
        if not self.outputs:
            return False
        for out in self.outputs:
            if not out.exists():
                return False
            if out.is_dir():
                # A directory output counts as built only if it is non-empty.
                if not any(out.iterdir()):
                    return False
                continue
            try:
                if out.stat().st_size < MIN_OUTPUT_BYTES:
                    return False
            except OSError:
                return False
        return True


@dataclass(frozen=True)
class Paths:
    project: Path
    data: Path
    cache: Path


def load_paths() -> Paths:
    """Resolve storage paths, raising a friendly error if the env var is unset."""
    from urban_energy.paths import CACHE_DIR, DATA_DIR, PROJECT_DIR

    return Paths(project=PROJECT_DIR, data=DATA_DIR, cache=CACHE_DIR)


def build_stages(p: Paths) -> list[Stage]:
    """Ordered acquisition manifest (downloads → OA aggregations)."""
    s = p.data / "statistics"
    return [
        Stage(
            "census",
            (s / "census_oa_joined.gpkg",),
            ("data/download_census.py",),
            note="Census 2021 OA tables + OA/LSOA geometry",
        ),
        Stage(
            "energy_postcode",
            (s / "postcode_energy_consumption.parquet",),
            ("data/download_energy_postcode.py",),
        ),
        Stage(
            "imd",
            (s / "lsoa_imd2025.parquet",),
            ("data/download_imd.py",),
            note="IoD2025 income domain (deprivation control)",
        ),
        Stage(
            "vehicles",
            (s / "lsoa_vehicles.parquet",),
            ("data/download_vehicles.py",),
            note="DVLA fleet → BEV share",
        ),
        Stage(
            "nts_mileage",
            (s / "nts_mileage_by_ruc.parquet",),
            ("data/download_nts_mileage.py",),
            note="travel-energy anchor: car miles/person by 2021 RUC",
        ),
        Stage(
            "ruc",
            (s / "oa21_ruc21.parquet",),
            ("data/download_ons_ruc.py",),
            note="OA → 2021 rural-urban class",
        ),
        Stage(
            "fsa",
            (p.data / "fsa" / "fsa_establishments.gpkg",),
            ("data/download_fsa.py",),
            note="food service + grocery retail",
        ),
        Stage(
            "naptan",
            (p.data / "transport" / "naptan_england.gpkg",),
            ("data/download_naptan.py",),
        ),
        Stage(
            "gias",
            (p.data / "education" / "gias_schools.gpkg",),
            ("data/prepare_gias.py",),
        ),
        Stage(
            "nhs",
            (p.data / "health" / "nhs_facilities.gpkg",),
            ("data/prepare_nhs.py",),
        ),
        Stage(
            "epc",
            (p.data / "epc" / "epc_domestic_spatial.parquet",),
            ("data/process_epc.py",),
            note="~55 GB raw → build year + floor area + best-fabric intensity",
        ),
        Stage(
            "workplace",
            (p.data / "employment" / "workplace_jobs.gpkg",),
            ("data/download_workplace.py",),
            note="Census 2021 WP101EW workplace jobs → OA points",
        ),
        Stage(
            "climate",
            (s / "oa_hdd.parquet",),
            ("data/process_climate.py",),
            note="HadUK-Grid → annual HDD per OA (climate confound); "
            "needs the manual CEDA .nc (see doctor)",
        ),
        # dependent aggregations
        Stage(
            "postcode_oa_lookup",
            (s / "postcode_oa_lookup.parquet",),
            ("data/build_postcode_oa_lookup.py",),
            note="needs census + Code-Point",
        ),
        Stage(
            "energy_oa",
            (s / "oa_energy_consumption.parquet",),
            ("data/aggregate_energy_oa.py",),
            note="primary heat DV; needs energy_postcode + postcode_oa_lookup",
        ),
        Stage(
            "epc_oa",
            (s / "oa_epc.parquet",),
            ("data/aggregate_epc_oa.py",),
            note="EPC floor area + best-fabric intensity + build year → OA",
        ),
    ]


# ---------------------------------------------------------------------------
# Preflight (doctor)
# ---------------------------------------------------------------------------


def _manual_prereqs(p: Paths) -> list[tuple[str, bool]]:
    """Manual downloads the acquisition needs (portals / registration)."""
    from urban_energy.paths import epc_input_dir, latest_uprn_gpkg

    d = p.data
    oa = list(d.glob("Output_Areas_*")) if d.exists() else []
    nhs_ok = any(
        (x / "epraccur.csv").exists() for x in (d, d / "nhs_ods", p.cache / "nhs_ods")
    )
    haduk_ok = (d / "climate").exists() and any((d / "climate").glob("tas*.nc"))
    return [
        ("OA 2021 boundaries (Output_Areas_*)  ←census", bool(oa)),
        (
            "OS Open Greenspace  ←access",
            (d / "opgrsp_gpkg_gb" / "Data" / "opgrsp_gb.gpkg").exists(),
        ),
        (
            "OS Open Roads  ←network access (stats/oa_network_access.py)",
            (d / "oproad_gpkg_gb" / "Data" / "oproad_gb.gpkg").exists(),
        ),
        ("OS Code-Point Open  ←postcode_oa_lookup", (d / "codepo_gpkg_gb").exists()),
        ("OS Open UPRN (any vintage)  ←epc", latest_uprn_gpkg() is not None),
        ("EPC domestic certificates  ←epc", epc_input_dir() is not None),
        ("NHS ODS (epraccur/ets/edispensary)  ←nhs", nhs_ok),
        ("HadUK-Grid tas .nc (CEDA)  ←climate", haduk_ok),
    ]


def cmd_doctor(args: argparse.Namespace) -> int:
    print("=" * 68)
    print("PREFLIGHT  (urban-energy pipeline doctor)")
    print("=" * 68)
    try:
        p = load_paths()
    except Exception as exc:
        print(f"  ✗ URBAN_ENERGY_DATA_DIR: {exc}")
        print("\nFATAL: set URBAN_ENERGY_DATA_DIR in a .env at the repo root.")
        return 1
    print(f"  ✓ data dir: {p.data}")

    print("\nManual downloads:")
    missing = 0
    for label, ok in _manual_prereqs(p):
        print(f"  {'✓' if ok else '✗'} {label}")
        missing += not ok

    base = p.data if p.data.exists() else p.data.parent
    try:
        free_gb = shutil.disk_usage(base).free / 1e9
        print(
            f"\n  {'✓' if free_gb >= MIN_FREE_GB else '⚠'} free disk: "
            f"{free_gb:,.0f} GB (wants ~{MIN_FREE_GB}+)"
        )
    except OSError:
        print(f"\n  ⚠ could not stat free disk on {base}")

    print("\n" + "-" * 68)
    print(
        f"{missing} manual download(s) missing."
        if missing
        else "All manual downloads present."
    )
    print("Env OK. Run `… pipeline status` to see what's built.")
    return 0


# ---------------------------------------------------------------------------
# status / list / run
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    p = load_paths()
    print(f"{'STAGE':<20} {'STATUS':<9} OUTPUT")
    print("-" * 68)
    for st in build_stages(p):
        out = st.outputs[0]
        try:
            out = out.relative_to(p.project)
        except ValueError:
            pass
        done = st.is_done()
        print(
            f"{'✓' if done else '✗'} {st.name:<18} "
            f"{'done' if done else 'missing':<9} {out}"
        )
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    p = load_paths()
    for st in build_stages(p):
        print(f"{st.name:<18} run : {' '.join(st.argv)}")
        for o in st.outputs:
            print(f"{'':<18} out : {o}")
        if st.note:
            print(f"{'':<18} note: {st.note}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    p = load_paths()
    by_name = {s.name: s for s in build_stages(p)}
    if args.names:
        unknown = [n for n in args.names if n not in by_name]
        if unknown:
            raise SystemExit(f"unknown stage(s): {', '.join(unknown)}")
        chosen = [by_name[n] for n in args.names]
    elif args.all:
        chosen = list(by_name.values())
    else:
        raise SystemExit("nothing selected: pass stage name(s) or --all")

    for st in chosen:
        if st.is_done() and not args.force:
            print(f"[skip] {st.name} — outputs present")
            continue
        cmd = " ".join(st.argv)
        if args.dry_run:
            print(f"[would run] {st.name}: {cmd}")
            continue
        print(f"\n{'=' * 68}\n[run] {st.name}: {cmd}\n{'=' * 68}")
        result = subprocess.run([sys.executable, *st.argv], cwd=p.project, check=False)
        if result.returncode != 0:
            print(f"\n[FAIL] {st.name} exited {result.returncode}; stopping.")
            return result.returncode
    print("\n[done]")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="urban_energy.pipeline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="preflight: env, manual downloads, disk")
    sub.add_parser("status", help="show which stage outputs exist")
    sub.add_parser("list", help="print the stage manifest")
    run = sub.add_parser("run", help="run stages (skipping those already built)")
    run.add_argument("names", nargs="*", help="explicit stage name(s)")
    run.add_argument("--all", action="store_true", help="run every stage")
    run.add_argument("--force", action="store_true", help="rebuild even if built")
    run.add_argument("--dry-run", action="store_true", help="print plan, run nothing")

    args = parser.parse_args(argv)
    dispatch: dict[str, Callable[[argparse.Namespace], int]] = {
        "doctor": cmd_doctor,
        "status": cmd_status,
        "list": cmd_list,
        "run": cmd_run,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
