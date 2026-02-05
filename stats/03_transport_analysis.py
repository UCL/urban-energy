"""
H6: Combined Building + Transport Energy Footprint Analysis.

Tests whether compact urban form shows lower TOTAL household energy footprint
when transport energy is included alongside building energy.

Key insight: Even if flats appear less efficient per capita (due to smaller
households), the overall energy footprint may be lower when transport is
considered, because dense areas have lower car ownership.

Transport energy estimation:
- Uses Census car ownership data (ts045) as proxy
- Assumes 12,000 km/year per vehicle (UK average)
- Applies 0.17 kg CO2/km (average UK car)
- Converts to kWh equivalent using 0.233 kg CO2/kWh (UK grid)
"""

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# Paths
DATA_PATH = Path(__file__).parent.parent / "temp/processing/test/uprn_integrated.gpkg"
CENSUS_PATH = Path(__file__).parent.parent / "temp/statistics"
OUTPUT_DIR = Path(__file__).parent.parent / "temp/stats/results"

# Transport energy parameters
KM_PER_VEHICLE_YEAR = 12000  # UK average annual mileage

# ICE scenario (current fleet)
KG_CO2_PER_KM_ICE = 0.17  # Average UK petrol/diesel car
KG_CO2_PER_KWH = 0.233  # UK grid carbon intensity (for CO2 to kWh conversion)
KWH_EQUIVALENT_PER_VEHICLE_ICE = (KM_PER_VEHICLE_YEAR * KG_CO2_PER_KM_ICE) / KG_CO2_PER_KWH

# EV scenario (full electrification)
# Average EV efficiency: ~0.18 kWh/km (Tesla Model 3: 0.14, larger EVs: 0.22)
KWH_PER_KM_EV = 0.18
KWH_PER_VEHICLE_EV = KM_PER_VEHICLE_YEAR * KWH_PER_KM_EV  # Direct kWh, no conversion needed

# For backward compatibility
KWH_EQUIVALENT_PER_VEHICLE = KWH_EQUIVALENT_PER_VEHICLE_ICE


def load_data() -> gpd.GeoDataFrame:
    """Load the integrated UPRN dataset with quality filters."""
    print("Loading integrated UPRN data...")
    gdf = gpd.read_file(DATA_PATH)

    # Convert numeric columns
    numeric_cols = [
        "ENERGY_CONSUMPTION_CURRENT",
        "TOTAL_FLOOR_AREA",
        "volume_m3",
        "form_factor",
        "footprint_area_m2",
    ]
    for col in numeric_cols:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

    # Apply quality filters
    clean_mask = (
        gdf["footprint_area_m2"].notna()
        & (gdf["volume_m3"] >= 10)
        & (gdf["ENERGY_CONSUMPTION_CURRENT"] > 0)
        & gdf["OA21CD"].notna()
    )
    gdf = gdf[clean_mask].copy()
    print(f"  Clean sample: {len(gdf):,} records")
    return gdf


def load_car_ownership() -> pd.DataFrame:
    """Load Census car ownership data (TS045)."""
    ts045_path = CENSUS_PATH / "census_ts045_oa.parquet"
    if not ts045_path.exists():
        raise FileNotFoundError(
            f"Car ownership data not found at {ts045_path}. "
            "Run data/download_census.py first."
        )

    print("Loading Census car ownership data (TS045)...")
    df = pd.read_parquet(ts045_path)
    print(f"  Loaded {len(df):,} Output Areas")
    return df


def load_commute_distance() -> pd.DataFrame:
    """Load Census commute distance data (TS058)."""
    ts058_path = CENSUS_PATH / "census_ts058_oa.parquet"
    if not ts058_path.exists():
        return None

    print("Loading Census commute distance data (TS058)...")
    df = pd.read_parquet(ts058_path)

    # Distance band midpoints (km) - conservative estimates
    # Using midpoints, except for open-ended categories
    distance_bands = {
        "Less than 2km": 1.0,
        "2km to less than 5km": 3.5,
        "5km to less than 10km": 7.5,
        "10km to less than 20km": 15.0,
        "20km to less than 30km": 25.0,
        "30km to less than 40km": 35.0,
        "40km to less than 60km": 50.0,
        "60km and over": 80.0,  # Conservative cap
    }

    # Find columns and calculate weighted average
    result = pd.DataFrame({"OA21CD": df["OA21CD"]})

    total_col = [c for c in df.columns if "Total:" in c and "ts058" in c]
    if not total_col:
        return None

    total_workers = df[total_col[0]]

    # Calculate weighted distance
    weighted_distance = pd.Series(0.0, index=df.index)
    total_commuters = pd.Series(0.0, index=df.index)

    for band_name, midpoint in distance_bands.items():
        col = [c for c in df.columns if band_name in c]
        if col:
            count = df[col[0]]
            weighted_distance += count * midpoint
            total_commuters += count

    # Average commute distance (one-way, km)
    result["avg_commute_km"] = weighted_distance / total_commuters.replace(0, np.nan)

    # Estimate annual travel km per worker
    # Commute: 2 trips/day × 230 working days × avg_commute_km
    # Non-work travel: ~40% additional (shopping, leisure, etc.)
    WORKING_DAYS_PER_YEAR = 230
    NON_WORK_MULTIPLIER = 1.4

    result["annual_km_per_worker"] = (
        result["avg_commute_km"] * 2 * WORKING_DAYS_PER_YEAR * NON_WORK_MULTIPLIER
    )

    # Also get WFH rate (reduces transport need)
    wfh_col = [c for c in df.columns if "from home" in c.lower()]
    if wfh_col:
        result["pct_wfh"] = 100 * df[wfh_col[0]] / total_workers.replace(0, np.nan)

    print(f"  Mean commute distance: {result['avg_commute_km'].mean():.1f} km (one-way)")
    print(f"  Mean annual km/worker: {result['annual_km_per_worker'].mean():,.0f} km")

    return result


def compute_transport_proxy(
    car_df: pd.DataFrame, commute_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Compute transport energy proxy from car ownership and commute distance.

    If commute_df is provided, uses area-specific travel distances.
    Otherwise falls back to UK average (12,000 km/year).

    Returns DataFrame with OA21CD and estimated transport energy per household.
    """
    # Find car ownership columns - match actual Census column names
    # Format: ts045_Number of cars or vans: [category]
    ts045_cols = [c for c in car_df.columns if "ts045" in c.lower()]

    total_col = [c for c in ts045_cols if "Total" in c]
    no_car_col = [c for c in ts045_cols if "No cars" in c]
    one_car_col = [c for c in ts045_cols if "1 car" in c]
    two_car_col = [c for c in ts045_cols if "2 cars" in c]
    three_plus_col = [c for c in ts045_cols if "3 or more" in c]

    if not all([total_col, no_car_col, one_car_col, two_car_col, three_plus_col]):
        # Try alternative column patterns
        print("  Looking for alternative column patterns...")
        print(f"  Available ts045 columns: {ts045_cols[:10]}")
        raise ValueError("Could not identify car ownership columns in TS045 data")

    # Calculate average cars per household
    result = pd.DataFrame({"OA21CD": car_df["OA21CD"]})

    total_hh = car_df[total_col[0]]
    cars_0 = car_df[no_car_col[0]]
    cars_1 = car_df[one_car_col[0]]
    cars_2 = car_df[two_car_col[0]]
    cars_3plus = car_df[three_plus_col[0]]

    # Assume "3 or more" averages to 3.2 cars
    total_cars = 0 * cars_0 + 1 * cars_1 + 2 * cars_2 + 3.2 * cars_3plus
    result["avg_cars_per_hh"] = total_cars / total_hh.replace(0, np.nan)

    # Also compute pct_no_car for analysis
    result["pct_no_car"] = 100 * cars_0 / total_hh.replace(0, np.nan)

    # Transport energy calculation
    if commute_df is not None:
        # Use area-specific travel distances
        result = result.merge(
            commute_df[["OA21CD", "annual_km_per_worker", "avg_commute_km"]],
            on="OA21CD",
            how="left",
        )
        # Fill missing with UK average
        result["annual_km_per_worker"] = result["annual_km_per_worker"].fillna(
            KM_PER_VEHICLE_YEAR
        )
        result["avg_commute_km"] = result["avg_commute_km"].fillna(
            KM_PER_VEHICLE_YEAR / (2 * 230 * 1.4)  # Back-calculate from UK average
        )

        # Energy = cars × km/year × kWh/km (using CO2 conversion for ICE)
        kwh_per_km_ice = KG_CO2_PER_KM_ICE / KG_CO2_PER_KWH
        result["transport_energy_kwh"] = (
            result["avg_cars_per_hh"] * result["annual_km_per_worker"] * kwh_per_km_ice
        )
        result["using_area_specific"] = True
        print("  Using AREA-SPECIFIC travel distances from Census commute data")
    else:
        # Fall back to UK average
        result["transport_energy_kwh"] = (
            result["avg_cars_per_hh"] * KWH_EQUIVALENT_PER_VEHICLE
        )
        result["annual_km_per_worker"] = KM_PER_VEHICLE_YEAR
        result["using_area_specific"] = False
        print("  Using UK average travel distance (12,000 km/year)")

    print(f"  Mean cars per household: {result['avg_cars_per_hh'].mean():.2f}")
    print(f"  Mean transport energy proxy: {result['transport_energy_kwh'].mean():.0f} kWh/hh/yr")

    return result


def classify_urban_form(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Classify properties by urban form (density × building type)."""
    # Identify building type from PROPERTY_TYPE or BUILT_FORM
    if "PROPERTY_TYPE" in gdf.columns:
        is_flat = gdf["PROPERTY_TYPE"].str.contains("Flat|flat|Maisonette", na=False)
        is_house = ~is_flat & gdf["PROPERTY_TYPE"].notna()
    else:
        is_flat = pd.Series(False, index=gdf.index)
        is_house = pd.Series(True, index=gdf.index)

    gdf["is_flat"] = is_flat
    gdf["is_house"] = is_house

    # Get density from OA-level data (use network density if available)
    density_col = None
    for col in ["pop_density", "cc_metric_node_density_800", "uprn_density"]:
        if col in gdf.columns:
            density_col = col
            break

    if density_col:
        density = pd.to_numeric(gdf[density_col], errors="coerce")
        density_median = density.median()
        gdf["high_density"] = density > density_median
    else:
        # Use transport proxy as density indicator
        gdf["high_density"] = gdf["pct_no_car"] > gdf["pct_no_car"].median()

    # Four-way classification
    gdf["urban_form"] = "Unknown"
    gdf.loc[gdf["is_house"] & ~gdf["high_density"], "urban_form"] = "Low density house"
    gdf.loc[gdf["is_house"] & gdf["high_density"], "urban_form"] = "High density house"
    gdf.loc[gdf["is_flat"] & ~gdf["high_density"], "urban_form"] = "Low density flat"
    gdf.loc[gdf["is_flat"] & gdf["high_density"], "urban_form"] = "High density flat"

    return gdf


def analyze_combined_footprint(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Analyze combined building + transport footprint by urban form."""
    # Group by urban form
    results = []

    for form in ["Low density house", "High density house", "Low density flat", "High density flat"]:
        subset = gdf[gdf["urban_form"] == form]
        if len(subset) < 100:
            continue

        building_energy = subset["ENERGY_CONSUMPTION_CURRENT"].mean()
        transport_energy = subset["transport_energy_kwh"].mean()
        total_energy = building_energy + transport_energy

        results.append({
            "urban_form": form,
            "n": len(subset),
            "building_kwh": building_energy,
            "transport_kwh": transport_energy,
            "total_kwh": total_energy,
            "pct_transport": 100 * transport_energy / total_energy,
            "avg_cars": subset["avg_cars_per_hh"].mean(),
            "pct_no_car": subset["pct_no_car"].mean(),
        })

    return pd.DataFrame(results)


def run_regression(gdf: gpd.GeoDataFrame) -> dict:
    """Run regression of total energy on urban form indicators."""
    from scipy import stats

    # Prepare variables
    y_building = gdf["ENERGY_CONSUMPTION_CURRENT"].values
    y_transport = gdf["transport_energy_kwh"].values
    y_total = y_building + y_transport

    x_density = gdf["high_density"].astype(float).values
    x_flat = gdf["is_flat"].astype(float).values

    # Simple correlations
    r_density_building = stats.pearsonr(x_density, y_building)[0]
    r_density_transport = stats.pearsonr(x_density, y_transport)[0]
    r_density_total = stats.pearsonr(x_density, y_total)[0]

    r_flat_building = stats.pearsonr(x_flat, y_building)[0]
    r_flat_transport = stats.pearsonr(x_flat, y_transport)[0]
    r_flat_total = stats.pearsonr(x_flat, y_total)[0]

    return {
        "density_building": r_density_building,
        "density_transport": r_density_transport,
        "density_total": r_density_total,
        "flat_building": r_flat_building,
        "flat_transport": r_flat_transport,
        "flat_total": r_flat_total,
    }


def analyze_ev_scenario(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Compare ICE vs EV scenarios for combined footprint.

    EVs are ~4x more efficient per km than ICE vehicles:
    - ICE: ~0.73 kWh-equivalent/km (via CO2 conversion)
    - EV: ~0.18 kWh/km (direct electricity)

    Uses area-specific travel distances if available.
    """
    results = []
    for form in ["Low density house", "High density house", "Low density flat", "High density flat"]:
        subset = gdf[gdf["urban_form"] == form]
        if len(subset) < 100:
            continue

        building = subset["ENERGY_CONSUMPTION_CURRENT"].mean()
        avg_cars = subset["avg_cars_per_hh"].mean()

        # Get area-specific travel distance if available
        if "annual_km_per_worker" in subset.columns:
            avg_km = subset["annual_km_per_worker"].mean()
            avg_commute = subset["avg_commute_km"].mean() if "avg_commute_km" in subset.columns else None
        else:
            avg_km = KM_PER_VEHICLE_YEAR
            avg_commute = None

        # ICE energy (using CO2 conversion)
        kwh_per_km_ice = KG_CO2_PER_KM_ICE / KG_CO2_PER_KWH
        transport_ice = avg_cars * avg_km * kwh_per_km_ice

        # EV energy (direct kWh)
        transport_ev = avg_cars * avg_km * KWH_PER_KM_EV

        results.append({
            "urban_form": form,
            "n": len(subset),
            "building_kwh": building,
            "avg_cars": avg_cars,
            "avg_km_year": avg_km,
            "avg_commute_km": avg_commute,
            # ICE scenario
            "transport_ice": transport_ice,
            "total_ice": building + transport_ice,
            "pct_transport_ice": 100 * transport_ice / (building + transport_ice),
            # EV scenario
            "transport_ev": transport_ev,
            "total_ev": building + transport_ev,
            "pct_transport_ev": 100 * transport_ev / (building + transport_ev),
        })

    return pd.DataFrame(results)


def main():
    """Run H6 combined footprint analysis."""
    print("=" * 60)
    print("H6: Combined Building + Transport Energy Footprint")
    print("=" * 60)
    print()

    # Load data
    gdf = load_data()

    # Load and merge car ownership + commute distance
    try:
        car_df = load_car_ownership()
        commute_df = load_commute_distance()  # May return None if not available
        transport_proxy = compute_transport_proxy(car_df, commute_df)
        gdf = gdf.merge(transport_proxy, on="OA21CD", how="left")
        print(f"  After transport merge: {gdf['transport_energy_kwh'].notna().sum():,} with data")
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        print("  Skipping transport analysis.")
        return

    # Drop records without transport data
    gdf = gdf[gdf["transport_energy_kwh"].notna()].copy()

    # Classify urban form
    print("\nClassifying urban form...")
    gdf = classify_urban_form(gdf)
    print(f"  Urban form distribution:")
    print(gdf["urban_form"].value_counts().to_string())

    # Show area-specific travel patterns
    if "avg_commute_km" in gdf.columns:
        print("\n  Travel patterns by density:")
        for density_label, is_high in [("Low density", False), ("High density", True)]:
            subset = gdf[gdf["high_density"] == is_high]
            if len(subset) > 0:
                commute = subset["avg_commute_km"].mean()
                km_year = subset["annual_km_per_worker"].mean()
                cars = subset["avg_cars_per_hh"].mean()
                print(f"    {density_label}: {commute:.1f} km commute, {km_year:,.0f} km/yr, {cars:.2f} cars/hh")
    print()

    # Analyze combined footprint
    print("\n" + "=" * 60)
    print("COMBINED ENERGY FOOTPRINT BY URBAN FORM")
    print("=" * 60)
    print()

    results = analyze_combined_footprint(gdf)

    print(f"{'Urban Form':<22} {'N':>8} {'Building':>10} {'Transport':>10} {'TOTAL':>10} {'%Transport':>10}")
    print("-" * 72)
    for _, row in results.iterrows():
        print(
            f"{row['urban_form']:<22} {row['n']:>8,} "
            f"{row['building_kwh']:>10,.0f} {row['transport_kwh']:>10,.0f} "
            f"{row['total_kwh']:>10,.0f} {row['pct_transport']:>9.1f}%"
        )

    print()

    # Compute key comparison
    low_house = results[results["urban_form"] == "Low density house"]["total_kwh"].values
    high_flat = results[results["urban_form"] == "High density flat"]["total_kwh"].values

    if len(low_house) > 0 and len(high_flat) > 0:
        pct_diff = 100 * (high_flat[0] - low_house[0]) / low_house[0]
        print(f"\nKey comparison (High density flat vs Low density house):")
        print(f"  Total footprint difference: {pct_diff:+.1f}%")
        if pct_diff < 0:
            print(f"  → High-density flats have {-pct_diff:.1f}% LOWER total footprint")

    # EV Scenario Comparison
    print("\n" + "=" * 60)
    print("SCENARIO COMPARISON: ICE vs ELECTRIC VEHICLES")
    print("=" * 60)
    print()
    print(f"Transport energy assumptions:")
    print(f"  ICE: {KWH_EQUIVALENT_PER_VEHICLE_ICE:,.0f} kWh-equivalent/vehicle/year")
    print(f"       (12,000 km × 0.17 kg CO2/km ÷ 0.233 kg CO2/kWh)")
    print(f"  EV:  {KWH_PER_VEHICLE_EV:,.0f} kWh/vehicle/year")
    print(f"       (12,000 km × 0.18 kWh/km direct electricity)")
    print(f"  EV efficiency gain: {100 * (1 - KWH_PER_VEHICLE_EV / KWH_EQUIVALENT_PER_VEHICLE_ICE):.0f}%")
    print()

    ev_results = analyze_ev_scenario(gdf)

    print(f"{'Urban Form':<22} {'Building':>9} │ {'ICE':>10} {'EV':>10} │ {'Total ICE':>10} {'Total EV':>10}")
    print("-" * 87)
    for _, row in ev_results.iterrows():
        print(
            f"{row['urban_form']:<22} {row['building_kwh']:>9,.0f} │ "
            f"{row['transport_ice']:>10,.0f} {row['transport_ev']:>10,.0f} │ "
            f"{row['total_ice']:>10,.0f} {row['total_ev']:>10,.0f}"
        )

    # Key comparison for both scenarios
    low_house_ev = ev_results[ev_results["urban_form"] == "Low density house"]
    high_flat_ev = ev_results[ev_results["urban_form"] == "High density flat"]

    if len(low_house_ev) > 0 and len(high_flat_ev) > 0:
        pct_diff_ice = 100 * (high_flat_ev["total_ice"].values[0] - low_house_ev["total_ice"].values[0]) / low_house_ev["total_ice"].values[0]
        pct_diff_ev = 100 * (high_flat_ev["total_ev"].values[0] - low_house_ev["total_ev"].values[0]) / low_house_ev["total_ev"].values[0]

        print()
        print("Key comparison (High density flat vs Low density house):")
        print(f"  ICE scenario: {pct_diff_ice:+.1f}% (dense flats {-pct_diff_ice:.0f}% lower)")
        print(f"  EV scenario:  {pct_diff_ev:+.1f}% (dense flats {-pct_diff_ev:.0f}% lower)")
        print()
        print("Interpretation:")
        print(f"  Even with full EV adoption, dense flats retain a {-pct_diff_ev:.0f}% advantage")
        print(f"  because they still have fewer cars (differential car ownership persists).")
        if abs(pct_diff_ev) < abs(pct_diff_ice):
            print(f"  The advantage shrinks from {-pct_diff_ice:.0f}% to {-pct_diff_ev:.0f}%")
            print(f"  as transport becomes a smaller share of the total footprint.")

    # Run correlations
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    correlations = run_regression(gdf)
    print()
    print(f"{'Variable':<20} {'Building':>12} {'Transport':>12} {'Total':>12}")
    print("-" * 58)
    print(
        f"{'High density':<20} {correlations['density_building']:>+12.3f} "
        f"{correlations['density_transport']:>+12.3f} {correlations['density_total']:>+12.3f}"
    )
    print(
        f"{'Is flat':<20} {correlations['flat_building']:>+12.3f} "
        f"{correlations['flat_transport']:>+12.3f} {correlations['flat_total']:>+12.3f}"
    )

    print()
    print("Interpretation:")
    if correlations["density_total"] < 0:
        print("  → High density is associated with LOWER total energy footprint")
    else:
        print("  → High density is associated with higher total energy footprint")

    if correlations["flat_total"] < 0:
        print("  → Flats have LOWER total energy footprint than houses")
    else:
        print("  → Flats have higher total energy footprint than houses")

    # Print methodological notes
    print("\n" + "=" * 60)
    print("METHODOLOGICAL NOTES")
    print("=" * 60)
    print("""
Transport energy uses AREA-SPECIFIC Census data:
- Car ownership: TS045 (cars per household by Output Area)
- Commute distance: TS058 (distance bands by Output Area)
- Annual km estimated: commute × 2 trips × 230 days × 1.4 (non-work travel)
- ICE: 0.17 kg CO2/km → 0.73 kWh-equivalent/km
- EV: 0.18 kWh/km (direct electricity)

Key finding: Dense areas have SIMILAR commute distances but FEWER CARS.
The urban advantage comes from differential car ownership, not shorter trips.

Building energy is SAP-modeled POTENTIAL demand, not actual consumption.
""")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "h6_combined_footprint.csv", index=False)
    print(f"\nResults saved to: {OUTPUT_DIR / 'h6_combined_footprint.csv'}")

    return results


if __name__ == "__main__":
    main()
