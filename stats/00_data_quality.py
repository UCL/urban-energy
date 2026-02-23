"""
Data Quality Report for Urban Energy Pipeline Output.

Generates a comprehensive data quality assessment of the integrated UPRN dataset,
including distribution summaries, missing data patterns, and filtering recommendations.

Run after test_pipeline.py to validate processed data before analysis.
"""

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# Paths
from urban_energy.paths import TEMP_DIR

DATA_PATH = TEMP_DIR / "processing" / "test" / "uprn_integrated.gpkg"
OUTPUT_DIR = TEMP_DIR / "stats" / "data_quality"


def load_data() -> gpd.GeoDataFrame:
    """Load the integrated UPRN dataset."""
    print(f"Loading data from {DATA_PATH}")
    gdf = gpd.read_file(DATA_PATH)
    print(f"Loaded {len(gdf):,} records with {len(gdf.columns)} columns")
    return gdf


def summarize_column(series: pd.Series, name: str) -> dict:
    """Generate summary statistics for a numeric column."""
    # Convert to numeric, coercing errors to NaN
    numeric = pd.to_numeric(series, errors="coerce")
    clean = numeric.dropna()

    if len(clean) == 0:
        return {"name": name, "valid": 0, "missing": len(series)}

    # Convert to numpy for reliable quantile calculation
    arr = clean.to_numpy()

    return {
        "name": name,
        "valid": len(clean),
        "missing": numeric.isna().sum(),
        "missing_pct": 100 * numeric.isna().sum() / len(series),
        "min": float(np.min(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _to_numeric_array(series: pd.Series) -> np.ndarray:
    """Convert series to numeric numpy array, dropping NaN."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.dropna().to_numpy()


def check_extreme_values(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Check for extreme/exploding values in key variables."""
    checks = []

    # Thermal physics extremes
    if "form_factor" in gdf.columns:
        ff = _to_numeric_array(gdf["form_factor"])
        checks.extend([
            ("form_factor > 20", int((ff > 20).sum()), len(ff)),
            ("form_factor > 50", int((ff > 50).sum()), len(ff)),
            ("form_factor > 100", int((ff > 100).sum()), len(ff)),
        ])

    if "surface_to_volume" in gdf.columns:
        s2v = _to_numeric_array(gdf["surface_to_volume"])
        checks.extend([
            ("surface_to_volume > 1.0", int((s2v > 1.0).sum()), len(s2v)),
            ("surface_to_volume > 2.0", int((s2v > 2.0).sum()), len(s2v)),
            ("surface_to_volume > 5.0", int((s2v > 5.0).sum()), len(s2v)),
        ])

    if "volume_m3" in gdf.columns:
        vol = _to_numeric_array(gdf["volume_m3"])
        checks.extend([
            ("volume_m3 < 10", int((vol < 10).sum()), len(vol)),
            ("volume_m3 < 1", int((vol < 1).sum()), len(vol)),
            ("volume_m3 == 0", int((vol == 0).sum()), len(vol)),
        ])

    if "height_mean" in gdf.columns:
        h = _to_numeric_array(gdf["height_mean"])
        checks.extend([
            ("height_mean < 1.0m", int((h < 1.0).sum()), len(h)),
            ("height_mean < 2.0m", int((h < 2.0).sum()), len(h)),
            ("height_mean > 50m", int((h > 50).sum()), len(h)),
        ])

    # Energy extremes
    if "ENERGY_CONSUMPTION_CURRENT" in gdf.columns:
        e = _to_numeric_array(gdf["ENERGY_CONSUMPTION_CURRENT"])
        checks.extend([
            ("ENERGY_CONSUMPTION <= 0", int((e <= 0).sum()), len(e)),
            ("ENERGY_CONSUMPTION > 50000", int((e > 50000).sum()), len(e)),
        ])

    if "TOTAL_FLOOR_AREA" in gdf.columns:
        fa = _to_numeric_array(gdf["TOTAL_FLOOR_AREA"])
        checks.extend([
            ("TOTAL_FLOOR_AREA < 10", int((fa < 10).sum()), len(fa)),
            ("TOTAL_FLOOR_AREA > 1000", int((fa > 1000).sum()), len(fa)),
        ])

    return pd.DataFrame(checks, columns=["check", "count", "total"])


def check_missing_patterns(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Analyze missing data patterns by data source."""
    patterns = []

    # Morphology columns
    morph_cols = ["footprint_area_m2", "volume_m3", "form_factor", "shared_wall_ratio"]
    morph_cols = [c for c in morph_cols if c in gdf.columns]
    if morph_cols:
        missing = gdf[morph_cols[0]].isna().sum()
        patterns.append(("Building Morphology", missing, len(gdf)))

    # EPC columns
    epc_cols = ["ENERGY_CONSUMPTION_CURRENT", "TOTAL_FLOOR_AREA", "PROPERTY_TYPE"]
    epc_cols = [c for c in epc_cols if c in gdf.columns]
    if epc_cols:
        missing = gdf[epc_cols[0]].isna().sum()
        patterns.append(("EPC Data", missing, len(gdf)))

    # Census geography
    if "OA21CD" in gdf.columns:
        missing = gdf["OA21CD"].isna().sum()
        patterns.append(("Census Geography", missing, len(gdf)))

    # Network metrics
    network_cols = [c for c in gdf.columns if c.startswith("cc_metric")]
    if network_cols:
        missing = gdf[network_cols[0]].isna().sum()
        patterns.append(("Network Metrics", missing, len(gdf)))

    df = pd.DataFrame(patterns, columns=["source", "missing", "total"])
    df["missing_pct"] = 100 * df["missing"] / df["total"]
    return df


def compute_analysis_sample(gdf: gpd.GeoDataFrame) -> dict:
    """Compute the clean analysis sample with recommended filters."""
    filters = {}

    # Base: has morphology
    if "footprint_area_m2" in gdf.columns:
        filters["has_morphology"] = pd.to_numeric(
            gdf["footprint_area_m2"], errors="coerce"
        ).notna()

    # Reasonable volume
    if "volume_m3" in gdf.columns:
        vol = pd.to_numeric(gdf["volume_m3"], errors="coerce")
        filters["volume_ok"] = vol >= 10

    # Reasonable form factor (if not already capped in pipeline)
    if "form_factor" in gdf.columns:
        ff = pd.to_numeric(gdf["form_factor"], errors="coerce")
        filters["form_factor_ok"] = ff <= 50

    # Has EPC
    if "ENERGY_CONSUMPTION_CURRENT" in gdf.columns:
        energy = pd.to_numeric(gdf["ENERGY_CONSUMPTION_CURRENT"], errors="coerce")
        filters["has_epc"] = energy.notna()
        filters["energy_positive"] = energy > 0

    # Has geography
    if "OA21CD" in gdf.columns:
        filters["has_geography"] = gdf["OA21CD"].notna()

    # Combine filters
    combined = pd.Series(True, index=gdf.index)
    for name, mask in filters.items():
        combined = combined & mask.fillna(False)

    return {
        "total_records": len(gdf),
        "filter_counts": {k: int(v.sum()) for k, v in filters.items()},
        "clean_sample_size": int(combined.sum()),
        "clean_sample_pct": 100 * combined.sum() / len(gdf),
        "mask": combined,
    }


def generate_report(gdf: gpd.GeoDataFrame) -> str:
    """Generate the full data quality report as markdown."""
    lines = []
    lines.append("# Data Quality Report: UPRN Integrated Dataset")
    lines.append("")
    lines.append(f"**File:** `{DATA_PATH}`")
    lines.append(f"**Total Records:** {len(gdf):,}")
    lines.append(f"**Total Columns:** {len(gdf.columns)}")
    lines.append("")

    # Key variable summaries
    lines.append("## 1. Key Variable Distributions")
    lines.append("")

    key_vars = [
        "form_factor",
        "surface_to_volume",
        "volume_m3",
        "shared_wall_ratio",
        "height_mean",
        "footprint_area_m2",
        "ENERGY_CONSUMPTION_CURRENT",
        "TOTAL_FLOOR_AREA",
    ]
    key_vars = [v for v in key_vars if v in gdf.columns]

    lines.append("| Variable | Valid | Missing% | Min | P01 | Median | P99 | Max |")
    lines.append("|----------|-------|----------|-----|-----|--------|-----|-----|")

    for var in key_vars:
        s = summarize_column(gdf[var], var)
        if "min" in s:
            lines.append(
                f"| {s['name']} | {s['valid']:,} | {s['missing_pct']:.1f}% | "
                f"{s['min']:.2f} | {s['p01']:.2f} | {s['median']:.2f} | "
                f"{s['p99']:.2f} | {s['max']:.2f} |"
            )
        else:
            lines.append(f"| {s['name']} | {s['valid']:,} | 100% | - | - | - | - | - |")

    lines.append("")

    # Extreme values
    lines.append("## 2. Extreme Value Checks")
    lines.append("")
    extremes = check_extreme_values(gdf)
    lines.append("| Check | Count | % of Valid |")
    lines.append("|-------|-------|------------|")
    for _, row in extremes.iterrows():
        pct = 100 * row["count"] / row["total"] if row["total"] > 0 else 0
        flag = "⚠️" if pct > 1 else ""
        lines.append(f"| {row['check']} | {row['count']:,} | {pct:.2f}% {flag} |")
    lines.append("")

    # Missing data patterns
    lines.append("## 3. Missing Data by Source")
    lines.append("")
    missing = check_missing_patterns(gdf)
    lines.append("| Data Source | Missing | % Missing |")
    lines.append("|-------------|---------|-----------|")
    for _, row in missing.iterrows():
        lines.append(f"| {row['source']} | {row['missing']:,} | {row['missing_pct']:.1f}% |")
    lines.append("")

    # Analysis sample
    lines.append("## 4. Clean Analysis Sample")
    lines.append("")
    sample = compute_analysis_sample(gdf)
    lines.append(f"**Total records:** {sample['total_records']:,}")
    lines.append("")
    lines.append("**Filter application:**")
    lines.append("")
    for name, count in sample["filter_counts"].items():
        pct = 100 * count / sample["total_records"]
        lines.append(f"- `{name}`: {count:,} ({pct:.1f}%)")
    lines.append("")
    lines.append(
        f"**Clean sample (all filters):** {sample['clean_sample_size']:,} "
        f"({sample['clean_sample_pct']:.1f}%)"
    )
    lines.append("")

    # Recommendations
    lines.append("## 5. Recommendations")
    lines.append("")
    lines.append("### Recommended Analysis Filters")
    lines.append("")
    lines.append("```python")
    lines.append("clean_mask = (")
    lines.append("    gdf['footprint_area_m2'].notna() &")
    lines.append("    (gdf['volume_m3'] >= 10) &           # Exclude tiny fragments")
    lines.append("    (gdf['form_factor'] <= 50) &         # Exclude implausible ratios")
    lines.append("    (gdf['ENERGY_CONSUMPTION_CURRENT'] > 0) &  # Exclude errors")
    lines.append("    gdf['OA21CD'].notna()                # Require geography")
    lines.append(")")
    lines.append("```")
    lines.append("")

    # Flag any critical issues
    lines.append("### Data Quality Flags")
    lines.append("")

    issues = []
    for _, row in extremes.iterrows():
        pct = 100 * row["count"] / row["total"] if row["total"] > 0 else 0
        if pct > 1:
            issues.append(f"- ⚠️ {row['check']}: {row['count']:,} records ({pct:.1f}%)")

    if issues:
        lines.extend(issues)
    else:
        lines.append("✓ No critical data quality issues detected.")

    lines.append("")
    lines.append("---")
    lines.append("*Report generated by stats/00_data_quality.py*")

    return "\n".join(lines)


def main():
    """Run data quality assessment and save report."""
    # Load data
    gdf = load_data()

    # Generate report
    print("\nGenerating data quality report...")
    report = generate_report(gdf)

    # Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "data_quality_report.md"
    report_path.write_text(report)
    print(f"\nReport saved to: {report_path}")

    # Also print to console
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    # Return sample info for programmatic use
    sample = compute_analysis_sample(gdf)
    return sample


if __name__ == "__main__":
    main()
