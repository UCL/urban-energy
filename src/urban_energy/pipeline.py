"""
Lean orchestrator for the urban-energy rebuild.

Encodes the load-bearing data pipeline as an executable, resumable stage
manifest. Each stage wraps an existing script and declares its output files, so
the runner can skip stages whose outputs already exist and report exactly what
is built. The end product is ``oa_integrated.gpkg`` plus the per-OA tables the
two-axis analysis layer (``stats/travel_energy.py``, ``stats/access_profile.py``,
``stats/lock_in.py``, ``stats/form_size_decomposition.py``) consumes.

Scope is the **KEEP set** from the consumption audit. The heavy LiDAR/morphology
path is included but tagged ``optional`` and skipped by default (nothing consumes
its columns; see ROADMAP.md). The two-axis analyses are print-only and run
on-demand (not build artefacts), so they are not stages here.

Usage::

    uv run python -m urban_energy.pipeline doctor          # preflight checks
    uv run python -m urban_energy.pipeline status          # what's built
    uv run python -m urban_energy.pipeline list            # the manifest
    uv run python -m urban_energy.pipeline run --all       # build everything missing
    uv run python -m urban_energy.pipeline run --layer acquire
    uv run python -m urban_energy.pipeline run census energy_oa
    uv run python -m urban_energy.pipeline run --from pipeline
    uv run python -m urban_energy.pipeline run lidar --include-optional   # opt in

Flags: ``--force`` (rebuild even if outputs exist), ``--dry-run`` (print only),
``--include-optional`` (allow the deferred LiDAR/morphology stages).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

LAYERS = ["acquire", "process"]

# Heads-up threshold for free disk on the data volume (GB).
MIN_FREE_GB = 100


@dataclass(frozen=True)
class Stage:
    """One pipeline step: a wrapped script with declared outputs."""

    name: str
    layer: str
    outputs: tuple[Path, ...]
    argv: tuple[str, ...] = ()  # script + args, run under sys.executable from repo root
    optional: bool = False
    note: str = ""

    def is_done(self) -> bool:
        return bool(self.outputs) and all(o.exists() for o in self.outputs)


@dataclass(frozen=True)
class Paths:
    project: Path
    data: Path
    processing: Path
    cache: Path


def load_paths() -> Paths:
    """Resolve storage paths, raising a friendly error if the env var is unset."""
    from urban_energy.paths import (  # local import: may raise EnvironmentError
        CACHE_DIR,
        DATA_DIR,
        PROCESSING_DIR,
        PROJECT_DIR,
    )

    return Paths(
        project=PROJECT_DIR,
        data=DATA_DIR,
        processing=PROCESSING_DIR,
        cache=CACHE_DIR,
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_stages(p: Paths) -> list[Stage]:
    """Construct the ordered stage manifest for the KEEP-set rebuild."""
    stats = p.data / "statistics"

    return [
        # --- acquire ---
        Stage(
            "census",
            "acquire",
            (stats / "census_oa_joined.gpkg",),
            ("data/download_census.py",),
        ),
        Stage(
            "energy_postcode",
            "acquire",
            (stats / "postcode_energy_consumption.parquet",),
            ("data/download_energy_postcode.py",),
        ),
        Stage(
            "imd",
            "acquire",
            (stats / "lsoa_imd2025.parquet",),
            ("data/download_imd.py",),
        ),
        Stage(
            "vehicles",
            "acquire",
            (stats / "lsoa_vehicles.parquet",),
            ("data/download_vehicles.py",),
        ),
        Stage(
            "nts_mileage",
            "acquire",
            (stats / "nts_mileage_by_ruc.parquet",),
            ("data/download_nts_mileage.py",),
            note="travel-energy anchor: car miles/person by 2021 RUC",
        ),
        Stage(
            "ruc",
            "acquire",
            (stats / "oa21_ruc21.parquet",),
            ("data/download_ons_ruc.py",),
            note="OA→2021 rural-urban class",
        ),
        Stage(
            "fsa",
            "acquire",
            (p.data / "fsa" / "fsa_establishments.gpkg",),
            ("data/download_fsa.py",),
        ),
        Stage(
            "naptan",
            "acquire",
            (p.data / "transport" / "naptan_england.gpkg",),
            ("data/download_naptan.py",),
        ),
        Stage(
            "gias",
            "acquire",
            (p.data / "education" / "gias_schools.gpkg",),
            ("data/prepare_gias.py",),
        ),
        Stage(
            "nhs",
            "acquire",
            (p.data / "health" / "nhs_facilities.gpkg",),
            ("data/prepare_nhs.py",),
        ),
        Stage(
            "epc",
            "acquire",
            (p.data / "epc" / "epc_domestic_spatial.parquet",),
            ("data/process_epc.py",),
            note="~8 GB raw; yields build year + floor area + best-fabric intensity",
        ),
        Stage(
            "boundaries",
            "acquire",
            (p.data / "boundaries" / "built_up_areas.gpkg",),
            ("data/process_boundaries.py",),
        ),
        # dependent aggregations
        Stage(
            "postcode_oa_lookup",
            "acquire",
            (stats / "postcode_oa_lookup.parquet",),
            ("data/build_postcode_oa_lookup.py",),
            note="needs census + Code-Point",
        ),
        Stage(
            "energy_oa",
            "acquire",
            (stats / "oa_energy_consumption.parquet",),
            ("data/aggregate_energy_oa.py",),
            note="primary heat DV; needs energy_postcode + postcode_oa_lookup",
        ),
        Stage(
            "epc_oa",
            "acquire",
            (stats / "oa_epc.parquet",),
            ("data/aggregate_epc_oa.py",),
            note="EPC floor area + best-fabric intensity → OA; size decomposition "
            "+ lock_in; needs epc + postcode_oa_lookup",
        ),
        # --- process ---
        Stage(
            "lidar",
            "process",
            (p.data / "lidar" / "building_heights.gpkg",),
            ("data/process_lidar.py",),
            optional=True,
            note="DEFERRED ~20-30h; only height_mean is consumed (one table cell)",
        ),
        Stage(
            "morphology",
            "process",
            (p.data / "morphology" / "buildings_morphology.gpkg",),
            ("processing/process_morphology.py",),
            optional=True,
            note="DEFERRED ~10-15h; momepy metrics consumed by nothing",
        ),
        Stage(
            "pipeline",
            "process",
            (p.processing / "combined" / "oa_integrated.gpkg",),
            ("processing/pipeline_oa.py",),
            note="national CityNetwork pipeline ~30-50h; resumable per-BUA",
        ),
    ]


# ---------------------------------------------------------------------------
# Preflight (doctor)
# ---------------------------------------------------------------------------


@dataclass
class Check:
    label: str
    ok: bool
    detail: str = ""
    hard: bool = False  # hard failures gate the run (exit nonzero)


def _manual_prereqs(p: Paths) -> list[Check]:
    from urban_energy.paths import epc_input_dir, latest_uprn_gpkg

    d = p.data
    checks: list[Check] = []

    # OA 2021 boundaries (glob — the manual download is date/version stamped).
    oa = list(d.glob("Output_Areas_*")) if d.exists() else []
    checks.append(Check("OA 2021 boundaries (Output_Areas_*)  ←census", bool(oa)))

    for label, path, needed in [
        ("OS Built Up Areas", d / "OS_Open_Built_Up_Areas_GeoPackage", "boundaries"),
        ("OS Open Roads", d / "oproad_gpkg_gb" / "Data" / "oproad_gb.gpkg", "pipeline"),
        (
            "OS Open Greenspace",
            d / "opgrsp_gpkg_gb" / "Data" / "opgrsp_gb.gpkg",
            "pipeline",
        ),
        ("OS Code-Point Open", d / "codepo_gpkg_gb", "postcode_oa_lookup"),
    ]:
        checks.append(Check(f"{label}  ←{needed}", path.exists()))

    # Vintage- / packaging-agnostic checks via the path resolvers.
    checks.append(
        Check(
            "OS Open UPRN (any vintage)  ←pipeline/epc",
            latest_uprn_gpkg() is not None,
        )
    )
    checks.append(Check("EPC domestic certificates  ←epc", epc_input_dir() is not None))
    nhs_ok = any(
        (directory / "epraccur.csv").exists()
        for directory in (p.data, p.data / "nhs_ods", p.cache / "nhs_ods")
    )
    checks.append(Check("NHS ODS (epraccur/ets/edispensary)  ←nhs", nhs_ok))
    return checks


def cmd_doctor(args: argparse.Namespace) -> int:
    print("=" * 68)
    print("PREFLIGHT  (urban-energy pipeline doctor)")
    print("=" * 68)

    # 1. Environment / storage paths
    try:
        p = load_paths()
    except Exception as exc:  # EnvironmentError or import failure
        print(f"  ✗ URBAN_ENERGY_DATA_DIR: {exc}")
        print("\nFATAL: set URBAN_ENERGY_DATA_DIR in a .env at the repo root.")
        return 1
    print(f"  ✓ data dir: {p.data}")

    # 2. Manual prerequisites (warn-only: needed for the rebuild, not for analyse)
    print("\nManual downloads (Tier A):")
    prereqs = _manual_prereqs(p)
    for c in prereqs:
        print(f"  {'✓' if c.ok else '✗'} {c.label}")
    print("  (GIAS schools downloads automatically; NHS ODS is a manual DSE export)")
    missing = [c for c in prereqs if not c.ok]

    # 3. Disk
    base = p.data if p.data.exists() else p.data.parent
    try:
        free_gb = shutil.disk_usage(base).free / 1e9
        icon = "✓" if free_gb >= MIN_FREE_GB else "⚠"
        print(
            f"\n  {icon} free disk on {base}: {free_gb:,.0f} GB "
            f"(rebuild wants ~{MIN_FREE_GB}+ GB)"
        )
    except OSError:
        print(f"\n  ⚠ could not stat free disk on {base}")

    # Summary
    print("\n" + "-" * 68)
    if missing:
        print(
            f"{len(missing)} manual download(s) missing — acquire/pipeline stages "
            "that need them will fail until provided."
        )
    else:
        print("All manual downloads present.")
    print("Env OK. Run `… pipeline status` to see what's built.")
    return 0  # env is the only hard gate; missing downloads are warnings


# ---------------------------------------------------------------------------
# status / list
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    p = load_paths()
    stages = build_stages(p)
    print(f"{'LAYER':<9} {'STAGE':<20} {'STATUS':<9} OUTPUT")
    print("-" * 68)
    for s in stages:
        done = s.is_done()
        status = "done" if done else ("optional" if s.optional else "missing")
        out = s.outputs[0] if s.outputs else Path("-")
        try:
            out = out.relative_to(p.project)
        except ValueError:
            pass
        mark = "✓" if done else ("·" if s.optional else "✗")
        print(f"{s.layer:<9} {mark} {s.name:<18} {status:<9} {out}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    p = load_paths()
    for s in build_stages(p):
        tag = " [optional]" if s.optional else ""
        cmd = " ".join(s.argv) if s.argv else "(in-process)"
        print(f"{s.layer:<9} {s.name:<18}{tag}")
        print(f"            run : {cmd}")
        for o in s.outputs:
            print(f"            out : {o}")
        if s.note:
            print(f"            note: {s.note}")
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _select(stages: list[Stage], args: argparse.Namespace) -> list[Stage]:
    by_name = {s.name: s for s in stages}
    if args.names:
        unknown = [n for n in args.names if n not in by_name]
        if unknown:
            raise SystemExit(f"unknown stage(s): {', '.join(unknown)}")
        return [by_name[n] for n in args.names]  # explicit: optional allowed
    if args.layer:
        if args.layer not in LAYERS:
            raise SystemExit(f"unknown layer {args.layer!r}; choose from {LAYERS}")
        chosen = [s for s in stages if s.layer == args.layer]
    elif args.from_stage:
        if args.from_stage not in by_name:
            raise SystemExit(f"unknown stage {args.from_stage!r}")
        idx = next(i for i, s in enumerate(stages) if s.name == args.from_stage)
        chosen = stages[idx:]
    elif args.all:
        chosen = list(stages)
    else:
        raise SystemExit(
            "nothing selected: pass stage name(s), --layer, --from, or --all"
        )
    # implicitly-selected optional stages are skipped unless opted in
    if not args.include_optional:
        chosen = [s for s in chosen if not s.optional]
    return chosen


def cmd_run(args: argparse.Namespace) -> int:
    p = load_paths()
    stages = build_stages(p)
    selected = _select(stages, args)
    if not selected:
        print("nothing to run (optional stages need --include-optional).")
        return 0

    for s in selected:
        if s.is_done() and not args.force:
            print(f"[skip] {s.name} — outputs present")
            continue
        cmd = " ".join(s.argv) if s.argv else "(in-process)"
        if args.dry_run:
            print(f"[would run] {s.name}: {cmd}")
            continue
        print(f"\n{'=' * 68}\n[run] {s.name}: {cmd}\n{'=' * 68}")
        result = subprocess.run([sys.executable, *s.argv], cwd=p.project, check=False)
        if result.returncode != 0:
            print(f"\n[FAIL] {s.name} exited {result.returncode}; stopping.")
            return result.returncode
    print("\n[done]")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
    run.add_argument("--layer", help=f"run a whole layer {LAYERS}")
    run.add_argument("--from", dest="from_stage", help="run from this stage to the end")
    run.add_argument("--all", action="store_true", help="run every non-optional stage")
    run.add_argument("--force", action="store_true", help="rebuild even if built")
    run.add_argument("--dry-run", action="store_true", help="print plan, run nothing")
    run.add_argument(
        "--include-optional",
        action="store_true",
        help="allow the deferred LiDAR/morphology stages",
    )

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
