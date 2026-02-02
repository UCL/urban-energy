# Data Processing

This folder contains scripts for downloading and preprocessing Census 2021 and geographic data for England and Wales at Output Area (OA) level.

## Data Sources

| Dataset                       | Source                                                                                                                                                   | Format       | Size    |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ------- |
| OA Boundaries (2021)          | [ONS Geoportal](https://geoportal.statistics.gov.uk/datasets/ons::output-areas-december-2021-boundaries-ew-bfe-v9/about)                                 | GeoPackage   | ~150 MB |
| OA → LSOA → MSOA → LAD Lookup | [ONS Geoportal](https://geoportal.statistics.gov.uk/datasets/ons::output-area-2021-to-lsoa-to-msoa-to-lad-december-2021-exact-fit-lookup-in-ew-v3/about) | CSV          | ~15 MB  |
| Census Topic Summaries        | [Nomisweb Bulk](https://www.nomisweb.co.uk/sources/census_2021_bulk)                                                                                     | CSV (zipped) | Varies  |

## Topic Summary Tables Downloaded

| Code  | Description               | Key Variables                      |
| ----- | ------------------------- | ---------------------------------- |
| TS001 | Usual resident population | Total population                   |
| TS006 | Population density        | Persons per hectare                |
| TS007 | Age by single year        | Age distribution (LSOA+ only)      |
| TS011 | Households by deprivation | Deprivation dimensions             |
| TS017 | Household size            | Number of persons                  |
| TS044 | Accommodation type        | Detached, semi, flat, etc.         |
| TS054 | Tenure                    | Owned, rented, social              |
| TS061 | Method of travel to work  | Car, public transport, walk, cycle |
| TS062 | NS-SeC                    | Socio-economic classification      |

## Prerequisites

```bash
# Install required packages
uv add requests tqdm
# or
pip install requests tqdm
```

## Usage

### Download and Process All Data

```bash
cd data
python download_census.py
```

This will:

1. Download OA boundaries as GeoPackage
2. Download the OA → LAD lookup table
3. Download specified Census topic summary tables
4. Extract OA-level data from each table
5. Join all data to the OA boundaries
6. Save the final joined dataset to `temp/statistics/`

### Output Files

```
temp/statistics/
├── oa_boundaries.gpkg          # Raw OA boundaries
├── oa_lookup.parquet           # OA → LSOA → MSOA → LAD lookup
├── census_ts001_oa.parquet     # Individual topic tables
├── census_ts006_oa.parquet
├── ...
└── census_oa_joined.gpkg       # Final joined dataset with all variables
```

### Memory Considerations

The full OA dataset for England & Wales contains ~188,000 polygons. The joined GeoPackage may be 500MB+. For analysis, consider:

- Working with a subset (e.g., single LAD or region)
- Using Parquet files for tabular analysis (faster than GeoPackage)
- Loading geometry only when needed for spatial operations

## Data Dictionary

### Geographic Identifiers

| Column   | Description                   | Example             |
| -------- | ----------------------------- | ------------------- |
| OA21CD   | Output Area code              | E00000001           |
| LSOA21CD | Lower Super Output Area code  | E01000001           |
| LSOA21NM | LSOA name                     | City of London 001A |
| MSOA21CD | Middle Super Output Area code | E02000001           |
| MSOA21NM | MSOA name                     | City of London 001  |
| LAD22CD  | Local Authority District code | E09000001           |
| LAD22NM  | LAD name                      | City of London      |

### Census Variables

Each topic summary table adds columns following the pattern:

- `ts0XX_<category>` - Count for that category
- `ts0XX_total` - Total for the table (denominator for percentages)

## Notes

- **COVID-19 Impact**: Census 2021 was conducted in March 2021 during lockdown. Travel-to-work data shows 31% working from home, which is not representative of normal patterns.

- **Disclosure Control**: OA-level data is univariate only. Cross-tabulations are available at LSOA or MSOA level.

- **Scotland & Northern Ireland**: This script covers England & Wales only. Scotland uses different geographic codes and releases data separately via NRS.

## Troubleshooting

### Download fails with timeout

The ONS Geoportal can be slow. The script uses retries with backoff. If it still fails, try downloading manually from the links above.

### Memory errors when joining

Process in chunks by LAD:

```python
for lad in lookup['LAD22CD'].unique():
    subset = oa_gdf[oa_gdf['LAD22CD'] == lad]
    # process subset
```

### CSV parsing errors

Some Nomisweb CSVs have BOM markers or inconsistent encodings. The script handles this with `encoding='utf-8-sig'`.

---

## Ordnance Survey Open Data Products

OS Open Data products are freely available from the [OS Data Hub](https://osdatahub.os.uk/downloads/open). These provide high-quality geographic reference data for Great Britain.

### Quick Reference

| Dataset                    | Records | Key Fields                           | Use Case                         |
| -------------------------- | ------- | ------------------------------------ | -------------------------------- |
| **OS Open UPRN**           | 41.4M   | UPRN, X/Y coords, Lat/Long           | Property locations → EPC linkage |
| **OS Open Roads**          | -       | road_link (geometry, classification) | Street network analysis          |
| **OS Open Built Up Areas** | -       | gsscode, name, area_hectares         | Study area definition            |
| **OS Open Greenspace**     | -       | function, access_points              | Accessibility metrics            |
| **Code-Point Open**        | 1.7M    | postcode, admin codes                | Geocoding fallback               |
| **Boundary-Line**          | -       | Administrative boundaries            | LSOA/MSOA/LAD polygons           |

### Downloaded Products

| Product                | Description                                                      | Download Link                                                           | Local Path                                | Format     |
| ---------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------- | ---------- |
| OS Open UPRN           | Unique Property Reference Numbers for every addressable location | [OpenUPRN](https://osdatahub.os.uk/downloads/open/OpenUPRN)             | `temp/osopenuprn_202601_gpkg/`            | GeoPackage |
| Code-Point Open        | Postcode centroids with coordinates                              | [CodePointOpen](https://osdatahub.os.uk/downloads/open/CodePointOpen)   | `temp/codepo_gpkg_gb/`                    | GeoPackage |
| Boundary-Line          | Administrative and electoral boundaries                          | [BoundaryLine](https://osdatahub.os.uk/downloads/open/BoundaryLine)     | `temp/bdline_gpkg_gb/`                    | GeoPackage |
| OS Open Greenspace     | Public parks, playing fields, allotments, cemeteries             | [OpenGreenspace](https://osdatahub.os.uk/downloads/open/OpenGreenspace) | `temp/opgrsp_gpkg_gb/`                    | GeoPackage |
| OS Open Roads          | Connected road network with attributes                           | [OpenRoads](https://osdatahub.os.uk/downloads/open/OpenRoads)           | `temp/oproad_gpkg_gb/`                    | GeoPackage |
| OS Open Built Up Areas | Urban extent boundaries                                          | [BuiltUpAreas](https://osdatahub.os.uk/downloads/open/BuiltUpAreas)     | `temp/OS_Open_Built_Up_Areas_GeoPackage/` | GeoPackage |

### Product Details

#### OS Open UPRN

[Dataset page](https://osdatahub.os.uk/downloads/open/OpenUPRN)

Unique Property Reference Numbers (UPRNs) are persistent 12-digit identifiers for every addressable location in Great Britain.

```python
import geopandas as gpd

uprn = gpd.read_file("temp/osopenuprn_202601_gpkg/osopenuprn_202601.gpkg")
# Columns: UPRN, X_COORDINATE, Y_COORDINATE, LATITUDE, LONGITUDE
```

**Use cases:**

- Linking datasets via property-level identifiers
- Geocoding addresses
- Joining EPC data to spatial locations

#### Code-Point Open

[Dataset page](https://osdatahub.os.uk/downloads/open/CodePointOpen)

Postcode centroids for all ~1.7 million postcodes in Great Britain.

```python
postcodes = gpd.read_file("temp/codepo_gpkg_gb/Data/codepo_gb.gpkg")
# Columns: Postcode, Positional_quality_indicator, Eastings, Northings, Country_code,
#          NHS_Regional_HA_code, NHS_HA_code, Admin_county_code, Admin_district_code,
#          Admin_ward_code, geometry
```

**Use cases:**

- Postcode lookups and geocoding
- Aggregating data to postcode level
- Linking to census via postcode → OA lookups

#### Boundary-Line

[Dataset page](https://osdatahub.os.uk/downloads/open/BoundaryLine)

Administrative boundaries at multiple levels including parishes, wards, districts, counties, and countries.

```python
boundaries = gpd.read_file("temp/bdline_gpkg_gb/Data/bdline_gb.gpkg", layer="district_borough_unitary")
# Available layers: country, county, district_borough_unitary, parish,
#                   polling_districts_england, westminster_const, etc.
```

**Use cases:**

- Administrative aggregation
- Choropleth mapping
- Spatial filtering by authority

#### OS Open Greenspace

[Dataset page](https://osdatahub.os.uk/downloads/open/OpenGreenspace)

Public parks, playing fields, play spaces, sports facilities, allotments, and cemeteries across Great Britain.

```python
greenspace = gpd.read_file("temp/opgrsp_gpkg_gb/Data/opgrsp_gb.gpkg")
# Key columns: id, function (e.g., "Public Park Or Garden"), distName
```

**Use cases:**

- Green space accessibility analysis
- Urban morphology metrics
- Environmental quality indicators

#### OS Open Roads

[Dataset page](https://osdatahub.os.uk/downloads/open/OpenRoads)

Routable road network for Great Britain with road classifications and names.

```python
roads = gpd.read_file("temp/oproad_gpkg_gb/Data/oproad_gb.gpkg", layer="road_link")
# Key columns: identifier, class (Motorway, A Road, B Road, etc.),
#              roadFunction, formOfWay, name1, length
```

**Use cases:**

- Network analysis and routing
- Street density calculations
- Connectivity metrics

#### OS Open Built Up Areas

[Dataset page](https://osdatahub.os.uk/downloads/open/BuiltUpAreas)

Urban extent boundaries defining built-up areas across Great Britain.

```python
bua = gpd.read_file("temp/OS_Open_Built_Up_Areas_GeoPackage/os_open_built_up_areas.gpkg")
# Key columns: BUA22CD (GSS code), BUA22NM (name), geometry
```

**Use cases:**

- Filtering analysis to urban areas only
- Urban/rural classification
- Defining study area extents

### Merging Built Up Areas into Conurbations

Individual built-up areas may represent parts of a larger conurbation (e.g., Greater Manchester comprises multiple BUA polygons). To merge adjacent areas:

```python
import geopandas as gpd

bua = gpd.read_file("temp/OS_Open_Built_Up_Areas_GeoPackage/os_open_built_up_areas.gpkg")

# Option 1: Merge all touching polygons with buffer/dissolve
buffer_distance = 100  # metres - adjust based on gap tolerance
merged = bua.copy()
merged["geometry"] = merged.geometry.buffer(buffer_distance)
merged = merged.dissolve()
merged["geometry"] = merged.geometry.buffer(-buffer_distance)

# Option 2: Merge specific areas by name pattern
manchester = bua[bua["BUA22NM"].str.contains("Manchester|Salford|Stockport|Bolton")]
manchester_conurbation = manchester.dissolve()
```

### Spatial Filtering with Built Up Areas

Use built-up areas to subset other datasets:

```python
streets = gpd.read_parquet("street_network.parquet")
bua = gpd.read_file("temp/OS_Open_Built_Up_Areas_GeoPackage/os_open_built_up_areas.gpkg")
bua = bua.to_crs(streets.crs)

# Keep only streets within built-up areas
urban_streets = streets.sjoin(bua, predicate="within")

# Or clip to a specific city
london = bua[bua["BUA22NM"] == "Greater London"]
london_streets = streets.clip(london.union_all())
```

### Coordinate Reference Systems

All OS Open products use **EPSG:27700** (British National Grid) by default. Convert to WGS84 for web mapping:

```python
gdf_wgs84 = gdf.to_crs(epsg=4326)
```

### Downloading OS Open Data

Products can be downloaded manually from the [OS Data Hub](https://osdatahub.os.uk/downloads/open) or programmatically using the `osdatahub` Python package:

```bash
pip install osdatahub
```

```python
from osdatahub import OpenDataDownload

# Download Code-Point Open
OpenDataDownload.download(
    product="CodePointOpen",
    format="GeoPackage",
    output_dir="temp/"
)
```

---

## Energy Performance Certificates (EPCs)

Energy Performance Certificates provide standardised assessments of building energy efficiency for England and Wales.

### Data Source

| Dataset       | Source                                                | Records     | Update Frequency |
| ------------- | ----------------------------------------------------- | ----------- | ---------------- |
| Domestic EPCs | [EPC Open Data](https://epc.opendatacommunities.org/) | ~30 million | Monthly          |

### Key Features

- **UPRN field available** since November 2021 - enables direct linkage to OS Open UPRN
- **Bulk downloads** available by Local Authority or as complete dataset
- **API access** with filtering by UPRN, postcode, or address
- **OpenAPI v3 schemas** available since December 2023

### Key Fields

| Field                        | Description                      | Use Case              |
| ---------------------------- | -------------------------------- | --------------------- |
| `UPRN`                       | Unique Property Reference Number | Spatial linkage       |
| `CURRENT_ENERGY_EFFICIENCY`  | SAP rating (1-100)               | Primary energy metric |
| `CURRENT_ENERGY_RATING`      | EPC band (A-G)                   | Categorical rating    |
| `ENERGY_CONSUMPTION_CURRENT` | Estimated annual kWh             | Energy consumption    |
| `TOTAL_FLOOR_AREA`           | Floor area in m²                 | Normalisation         |
| `PROPERTY_TYPE`              | Detached, Semi, Terraced, Flat   | Building typology     |
| `BUILT_FORM`                 | Construction form                | Morphology proxy      |
| `CONSTRUCTION_AGE_BAND`      | Age of construction              | Vintage control       |
| `TENURE`                     | Owner-occupied, Rental, Social   | Tenure control        |
| `MAIN_FUEL`                  | Primary heating fuel             | Fuel type             |
| `HEATING_COST_CURRENT`       | Estimated heating cost           | Cost metric           |

### Downloading EPC Data

#### Option 1: Bulk Download (Recommended for Analysis)

Download complete datasets from [EPC Open Data](https://epc.opendatacommunities.org/):

1. Register for a free account
2. Navigate to Downloads → Domestic
3. Choose "All results" or filter by Local Authority
4. Download as CSV (zipped)

#### Option 2: API Access

```python
import requests

# Register at https://epc.opendatacommunities.org/ to get API key
API_KEY = "your-api-key"
BASE_URL = "https://epc.opendatacommunities.org/api/v1"

# Search by UPRN
response = requests.get(
    f"{BASE_URL}/domestic/search",
    params={"uprn": "100012345678"},
    headers={
        "Authorization": f"Basic {API_KEY}",
        "Accept": "application/json"
    }
)
```

API documentation: <https://epc.opendatacommunities.org/docs/api/domestic>

### Linking EPCs to OS Open UPRN

Only EPC records with valid UPRNs (post-November 2021) are used for spatial analysis.

```python
import pandas as pd
import geopandas as gpd

# Load EPC data
epc = pd.read_csv("domestic-certificates.csv")

# Filter to records with UPRN and convert to int64
epc["UPRN"] = pd.to_numeric(epc["UPRN"], errors="coerce")
epc = epc.dropna(subset=["UPRN"])
epc["UPRN"] = epc["UPRN"].astype("int64")

# Load UPRN points
uprn = gpd.read_file("temp/osopenuprn_202601_gpkg/osopenuprn_202601.gpkg")
uprn["UPRN"] = uprn["UPRN"].astype("int64")

# Join on UPRN
epc_spatial = uprn.merge(epc, on="UPRN", how="inner")
print(f"Matched {len(epc_spatial):,} EPCs with geometry")
```

### Data Quality Considerations

- **SAP vs Metered**: EPC estimates are modelled using the Standard Assessment Procedure (SAP), not actual metered consumption. SAP assumes standardised occupancy and heating patterns.

- **Coverage Bias**: EPCs are mandatory only for properties sold or rented. Long-term owner-occupied properties are under-represented.

- **Multiple Certificates**: Properties may have multiple EPCs from different dates. Use the most recent certificate or filter by `LODGEMENT_DATE`.

- **UPRN Coverage**: UPRNs are available from November 2021 onwards. Earlier certificates lack UPRNs and are excluded from spatial analysis (no free address data available for matching).

### Validation Against Metered Data

For validation, compare EPC estimates against:

- **NEED Database**: National Energy Efficiency Data (~11M properties with metered consumption)
- **Sub-national Energy Statistics**: MSOA-level gas/electricity consumption from DESNZ

```python
# Aggregate EPC estimates to LSOA
epc_lsoa = epc_spatial.groupby("LSOA21CD").agg({
    "ENERGY_CONSUMPTION_CURRENT": "mean",
    "UPRN": "count"
}).rename(columns={"UPRN": "n_properties"})

# Compare to sub-national statistics
subnational = pd.read_csv("subnational_gas_consumption_lsoa.csv")
validation = epc_lsoa.merge(subnational, on="LSOA21CD")
correlation = validation["ENERGY_CONSUMPTION_CURRENT"].corr(validation["metered_consumption"])
```

### References

- [EPC Open Data Portal](https://epc.opendatacommunities.org/)
- [MHCLG Blog: Changes to EPC Open Data Service](https://mhclgdigital.blog.gov.uk/2024/01/29/changes-to-the-energy-performance-certificates-open-data-service/)
- [ONS EPC Statistics](https://www.ons.gov.uk/peoplepopulationandcommunity/housing/datasets/energyperformancecertificateepcbandcoraboveenglandandwales)
- [Spatial Analysis of EPC Data Workshop](https://adamdennett.github.io/EPC_Analysis_Website/_site/EPCSpatialAnalysis.html)
