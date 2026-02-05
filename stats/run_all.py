"""
Run Canonical Analysis Pipeline for Urban Energy Project.

Executes the core analysis sequence:
1. Data quality report (00_data_quality.py)
2. Regression analysis with intensity DV (01_regression_analysis.py)
3. Mediation analysis (02_mediation_analysis.py)
4. Transport analysis (03_transport_analysis.py)
5. Generate figures (04_generate_figures.py)
6. Lock-in analysis (05_lockin_analysis.py) - Main quantitative findings
7. Generate report (06_generate_report.py) - Produces analysis_report_v3.md

Advanced analyses (in stats/advanced/):
- spatial_regression.py
- sensitivity_analysis.py
- cross_validation.py
- selection_bias_analysis.py

Usage:
    uv run python stats/run_all.py              # Core pipeline
    uv run python stats/run_all.py --full       # Include advanced analyses

Archived scripts (in stats/archive/):
    01_exploratory_analysis.py    # Superseded by 00_data_quality.py
    02f_stratified_analysis.py    # Integrated into main analysis
    02g_intensity_analysis.py     # Integrated into 01_regression_analysis.py
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
STATS_DIR = BASE_DIR / "stats"
ADVANCED_DIR = STATS_DIR / "advanced"

# Core pipeline scripts (run in order)
CORE_PIPELINE = [
    "00_data_quality.py",
    "01_regression_analysis.py",
    "02_mediation_analysis.py",
    "03_transport_analysis.py",
    "04_generate_figures.py",
    "05_lockin_analysis.py",
    "06_generate_report.py",
]

# Advanced analyses (optional)
ADVANCED_ANALYSES = [
    "advanced/spatial_regression.py",
    "advanced/sensitivity_analysis.py",
    "advanced/cross_validation.py",
    "advanced/selection_bias_analysis.py",
]


def run_script(script_name: str) -> bool:
    """Run a Python script and return success status."""
    script_path = STATS_DIR / script_name

    if not script_path.exists():
        print(f"  WARNING: Script not found: {script_name}")
        return False

    print(f"\n{'=' * 60}")
    print(f"RUNNING: {script_name}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR),
            capture_output=False,
            text=True,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    """Run the analysis pipeline."""
    print("=" * 60)
    print("URBAN ENERGY ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base directory: {BASE_DIR}")

    # Check for --full flag
    run_full = "--full" in sys.argv

    if run_full:
        print("\nMode: FULL (core + advanced analyses)")
    else:
        print("\nMode: CORE ONLY (use --full for advanced analyses)")

    # Track results
    results = {"success": [], "failed": [], "skipped": []}

    # Run core pipeline
    print("\n" + "=" * 60)
    print("CORE PIPELINE")
    print("=" * 60)

    for script in CORE_PIPELINE:
        if run_script(script):
            results["success"].append(script)
        else:
            results["failed"].append(script)

    # Run advanced analyses if --full
    if run_full:
        print("\n" + "=" * 60)
        print("ADVANCED ANALYSES")
        print("=" * 60)

        for script in ADVANCED_ANALYSES:
            if run_script(script):
                results["success"].append(script)
            else:
                results["failed"].append(script)
    else:
        results["skipped"] = ADVANCED_ANALYSES

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nSuccessful: {len(results['success'])}")
    for s in results["success"]:
        print(f"  ✓ {s}")

    if results["failed"]:
        print(f"\nFailed: {len(results['failed'])}")
        for s in results["failed"]:
            print(f"  ✗ {s}")

    if results["skipped"]:
        print(f"\nSkipped (use --full): {len(results['skipped'])}")
        for s in results["skipped"]:
            print(f"  - {s}")

    print("\n" + "=" * 60)
    print("OUTPUT LOCATIONS")
    print("=" * 60)
    print("  Data quality report: temp/stats/data_quality/")
    print("  Regression summary:  temp/stats/results/")
    print("  Lock-in results:     temp/stats/results/lockin_*.csv")
    print("  Summary JSON:        temp/stats/results/lockin_summary.json")
    print("  Figures:             stats/figures/")
    print("  Analysis report:     stats/analysis_report_v3.md")


if __name__ == "__main__":
    main()
