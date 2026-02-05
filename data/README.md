# Data Sources

Download and initial processing scripts for raw data sources.

**Scope:** Domestic (residential) buildings only. EPC data uses the "All domestic certificates" download; non-domestic EPCs are a separate dataset not included.

## Setup

1. **Download OA Boundaries:** Go to [ONS Geoportal](https://geoportal.statistics.gov.uk/datasets/ons::output-areas-december-2021-boundaries-ew-bfe-v9/about), click Download → GeoPackage, save to `temp/`
2. **Download Census topics:** Run `uv run python data/download_census.py`
3. **Download Built Up Areas:** Go to [OS Data Hub](https://osdatahub.os.uk/downloads/open/BuiltUpAreas), download GeoPackage, extract to `temp/`
4. **Process boundaries:** Run `uv run python data/process_boundaries.py`
5. **Download OS Open Map Local:** Go to [OS Data Hub](https://osdatahub.os.uk/downloads/open/OpenMapLocal), download GeoPackage (GB), extract to `temp/os_open_local/`
6. **Download EPC data:** Register at [epc.opendatacommunities.org](https://epc.opendatacommunities.org/), download "All domestic certificates", extract to `temp/epc/`
7. **Process EPC data:** Run `uv run python data/process_epc.py`
8. **Process LiDAR building heights:** Run `uv run python data/process_lidar.py`
9. **Download FSA establishments:** Run `uv run python data/download_fsa.py`
10. **Download NaPTAN transport stops:** Run `uv run python data/download_naptan.py`

## Pipeline Outputs

| Script                  | Output                                       | Description                                            |
| ----------------------- | -------------------------------------------- | ------------------------------------------------------ |
| `download_census.py`    | `temp/census_oa_joined.gpkg`                 | OA boundaries with 8 Census topic tables joined        |
| `process_boundaries.py` | `temp/boundaries/built_up_areas.gpkg`        | Cleaned individual built-up area polygons              |
|                         | `temp/boundaries/built_up_areas_merged.gpkg` | Merged conurbations (adjacent areas combined)          |
| `process_epc.py`        | `temp/epc_domestic_cleaned.parquet`          | Deduplicated EPC records (tabular, no geometry)        |
|                         | `temp/epc_domestic_spatial.gpkg`             | EPC records with UPRN point geometry                   |
| `process_lidar.py`      | `temp/lidar/building_heights.gpkg`           | Building polygons with LiDAR-derived height statistics |
| `download_fsa.py`       | `temp/fsa/fsa_establishments.gpkg`           | Eating/drinking establishments as walkability proxy    |
| `download_naptan.py`    | `temp/transport/naptan_england.gpkg`         | Public transport access points (bus, rail, metro, etc) |

---

## Reference

All CRS is EPSG:27700 (British National Grid) unless noted.

## Census 2021

| Dataset                       | Source                                                                                                                                                   | Format       |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| OA Boundaries (2021)          | [ONS Geoportal](https://geoportal.statistics.gov.uk/datasets/ons::output-areas-december-2021-boundaries-ew-bfe-v9/about)                                 | GeoPackage   |
| OA → LSOA → MSOA → LAD Lookup | [ONS Geoportal](https://geoportal.statistics.gov.uk/datasets/ons::output-area-2021-to-lsoa-to-msoa-to-lad-december-2021-exact-fit-lookup-in-ew-v3/about) | CSV          |
| Census Topic Summaries        | [Nomisweb Bulk](https://www.nomisweb.co.uk/sources/census_2021_bulk)                                                                                     | CSV (zipped) |

### Topic Summary Tables

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

### Geographic Identifiers

| Column   | Description                   | Example   |
| -------- | ----------------------------- | --------- |
| OA21CD   | Output Area code              | E00000001 |
| LSOA21CD | Lower Super Output Area code  | E01000001 |
| MSOA21CD | Middle Super Output Area code | E02000001 |
| LAD22CD  | Local Authority District code | E09000001 |

### Notes

- **COVID-19 Impact**: Census 2021 conducted during lockdown; travel-to-work shows 31% WFH.
- **Disclosure Control**: OA-level data is univariate only. Cross-tabulations at LSOA+.
- **Coverage**: England only.

---

## Ordnance Survey Open Data

All products from [OS Data Hub](https://osdatahub.os.uk/downloads/open).

| Product            | Download Link                                                           | Local Path                                | Key Columns / Layers                                             |
| ------------------ | ----------------------------------------------------------------------- | ----------------------------------------- | ---------------------------------------------------------------- |
| OS Open UPRN       | [OpenUPRN](https://osdatahub.os.uk/downloads/open/OpenUPRN)             | `temp/osopenuprn_202601_gpkg/`            | UPRN, X_COORDINATE, Y_COORDINATE, LATITUDE, LONGITUDE            |
| Code-Point Open    | [CodePointOpen](https://osdatahub.os.uk/downloads/open/CodePointOpen)   | `temp/codepo_gpkg_gb/`                    | Postcode, Eastings, Northings, Admin_district_code               |
| Boundary-Line      | [BoundaryLine](https://osdatahub.os.uk/downloads/open/BoundaryLine)     | `temp/bdline_gpkg_gb/`                    | **Layers:** country, county, district_borough_unitary, parish    |
| OS Open Greenspace | [OpenGreenspace](https://osdatahub.os.uk/downloads/open/OpenGreenspace) | `temp/opgrsp_gpkg_gb/`                    | id, function, distName                                           |
| OS Open Roads      | [OpenRoads](https://osdatahub.os.uk/downloads/open/OpenRoads)           | `temp/oproad_gpkg_gb/`                    | **Layers:** road_link, road_node; class, roadFunction, formOfWay |
| OS Open Map Local  | [OpenMapLocal](https://osdatahub.os.uk/downloads/open/OpenMapLocal)     | `temp/os_open_local/opmplc_gb.gpkg`       | **Layers:** building, road, woodland, surface_water_area         |
| Built Up Areas     | [BuiltUpAreas](https://osdatahub.os.uk/downloads/open/BuiltUpAreas)     | `temp/OS_Open_Built_Up_Areas_GeoPackage/` | BUA22CD (GSS code), BUA22NM (name)                               |

---

## Energy Performance Certificates (EPCs)

| Dataset       | Source                                                | Records     | Update  |
| ------------- | ----------------------------------------------------- | ----------- | ------- |
| Domestic EPCs | [EPC Open Data](https://epc.opendatacommunities.org/) | ~30 million | Monthly |

### Key Fields

| Field                        | Description                                  |
| ---------------------------- | -------------------------------------------- |
| `UPRN`                       | Unique Property Reference Number             |
| `CURRENT_ENERGY_EFFICIENCY`  | SAP rating (1-100)                           |
| `CURRENT_ENERGY_RATING`      | EPC band (A-G)                               |
| `ENERGY_CONSUMPTION_CURRENT` | Estimated annual kWh                         |
| `TOTAL_FLOOR_AREA`           | Total internal floor area in m² (all storeys)|
| `PROPERTY_TYPE`              | Detached, Semi, Terraced, Flat               |
| `BUILT_FORM`                 | More granular form (e.g., Mid-Terrace)       |
| `CONSTRUCTION_AGE_BAND`      | Age of construction                          |
| `TENURE`                     | Owner-occupied, Rental, Social               |
| `MAIN_FUEL`                  | Primary heating fuel                         |
| `FLOOR_LEVEL`                | Floor level for flats (0=ground, 1=first...) |
| `FLAT_TOP_STOREY`            | Whether flat is on top floor (Y/N)           |
| `FLOOR_HEIGHT`               | Average storey height (metres)               |

### Data Quality Notes

- **UPRN linkage**: UPRN coordinates included in EPC downloads since November 2021. Earlier certificates lack coordinates and cannot be spatially joined, so the analysis uses certificates from November 2021 onwards.
- **SAP vs Metered**: Estimates are modelled (SAP), not actual consumption.
- **Coverage Bias**: Only properties sold/rented; long-term owner-occupied under-represented.
- **Multiple Certificates**: Filter by `LODGEMENT_DATE` for most recent.

### Validation Sources

- [NEED Database](https://www.gov.uk/government/collections/national-energy-efficiency-data-need-framework) - ~11M properties with metered consumption
- [Sub-national Energy Statistics](https://www.gov.uk/government/collections/sub-national-electricity-consumption-data) - MSOA-level gas/electricity from DESNZ

### References

- [EPC Open Data Portal](https://epc.opendatacommunities.org/)
- [EPC API Documentation](https://epc.opendatacommunities.org/docs/api/domestic)
- [ONS EPC Statistics](https://www.ons.gov.uk/peoplepopulationandcommunity/housing/datasets/energyperformancecertificateepcbandcoraboveenglandandwales)

---

## Environment Agency LiDAR

| Dataset       | Source                                                                                     | Resolution | Coverage |
| ------------- | ------------------------------------------------------------------------------------------ | ---------- | -------- |
| Composite DSM | [DEFRA Data](https://environment.data.gov.uk/dataset/9ba4d5ac-d596-445a-9056-dae3ddec0178) | 2m         | ~99% Eng |
| Composite DTM | [DEFRA Data](https://environment.data.gov.uk/dataset/13787b9a-26a4-4775-8523-806d13af58fc) | 2m         | ~99% Eng |

### Key Details

- **Vintage**: 2022 composite (surveys 2000-2022)
- **Accuracy**: ±5-15cm RMSE vertical
- **Tiles**: 5km GeoTIFF, EPSG:27700
- **Building heights**: `nDSM = DSM - DTM`

### Processing

```bash
uv run python data/process_lidar.py
```

**Output:** `temp/lidar/building_heights.gpkg` — Building footprints with height statistics (min, max, mean, median, std) derived from nDSM.

See [processing/README.md](../processing/README.md) for derived morphology metrics.

---

## Food Standards Agency (FSA) Establishments

| Dataset                    | Source                                                 | Records | Update |
| -------------------------- | ------------------------------------------------------ | ------- | ------ |
| Food Hygiene Rating Scheme | [FSA Open Data](https://ratings.food.gov.uk/open-data) | ~500k   | Daily  |

### Purpose

Eating and drinking establishments serve as a proxy for **walkability** and **destination accessibility**. Higher densities indicate mixed-use, walkable neighbourhoods.

### Key Fields

| Field              | Description                                |
| ------------------ | ------------------------------------------ |
| `fhrs_id`          | Unique FSA establishment identifier        |
| `business_name`    | Establishment name                         |
| `business_type`    | Category (Restaurant, Pub, Takeaway, etc.) |
| `business_type_id` | Numeric type code                          |
| `postcode`         | UK postcode                                |
| `latitude`         | WGS84 latitude (source)                    |
| `longitude`        | WGS84 longitude (source)                   |
| `geometry`         | Point geometry (EPSG:27700 in output)      |

### Business Types Included

- Restaurant/Cafe/Canteen
- Pub/bar/nightclub
- Takeaway/sandwich shop
- Mobile caterer
- Hotel/bed & breakfast/guest house

### Processing

```bash
uv run python data/download_fsa.py
```

### References

- [FSA Open Data Portal](https://ratings.food.gov.uk/open-data)
- [FSA API Documentation](https://api.ratings.food.gov.uk/help)

---

## NaPTAN (National Public Transport Access Nodes)

| Dataset | Source                                              | Records | Update |
| ------- | --------------------------------------------------- | ------- | ------ |
| NaPTAN  | [DfT NaPTAN Portal](https://beta-naptan.dft.gov.uk) | ~434k   | Daily  |

### Purpose

Public transport access points enable analysis of **transport accessibility** - a key factor in sustainable urban form. Proximity to transit reduces car dependency and associated energy consumption.

### Coverage

England only (filtered from GB dataset). Includes:

| Stop Type         | Code        | Count |
| ----------------- | ----------- | ----- |
| Bus/Coach stops   | BCT/BCS     | ~308k |
| Rail stations     | RLY/RSE/RPL | ~5.7k |
| Underground/Metro | MET/PLT/TMU | ~4k   |
| Tram              | TMU/BST     | ~1.5k |
| Ferry terminals   | FER/FTD     | ~570  |
| Airport entrances | GAT/AIR     | ~100  |

### Key Fields

| Field       | Description                             |
| ----------- | --------------------------------------- |
| `atco_code` | Unique stop identifier                  |
| `atco_area` | 3-digit area code (010-499 for England) |
| `name`      | Common name of stop                     |
| `locality`  | Locality name                           |
| `stop_type` | Stop type code (BCT, RLY, MET, etc.)    |
| `status`    | Active/inactive status                  |
| `easting`   | OS National Grid easting                |
| `northing`  | OS National Grid northing               |
| `geometry`  | Point geometry (EPSG:27700)             |

### ATCO Area Codes

| Range   | Country                              |
| ------- | ------------------------------------ |
| 010-499 | England                              |
| 511-582 | Wales                                |
| 601-690 | Scotland                             |
| 910-940 | National services (rail, air, ferry) |

### Processing

```bash
uv run python data/download_naptan.py
```

### Links

- [NaPTAN Download Portal](https://beta-naptan.dft.gov.uk/download)
- [NaPTAN User Guide](https://www.gov.uk/government/publications/national-public-transport-access-node-schema/html-version-of-schema)
- [ATCO Codes Reference](https://beta-naptan.dft.gov.uk/article/atco-codes-in-use)
- [data.gov.uk NaPTAN](https://www.data.gov.uk/dataset/ff93ffc1-6656-47d8-9155-85ea0b8f2251/naptan)

---

## Next Steps

Once all data sources have been downloaded and processed:

1. **Proceed to [processing/README.md](../processing/README.md)** - Compute derived metrics:
   - Building morphology (shape, compactness, shared walls)

2. **Then to [stats/README.md](../stats/README.md)** - Lock-in analysis:
   - Matched comparison of built forms
   - Transport energy estimation
   - Combined penalty quantification
