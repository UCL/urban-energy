# Data Setup

1. **Download OA Boundaries:** Go to [ONS Geoportal](https://geoportal.statistics.gov.uk/datasets/ons::output-areas-december-2021-boundaries-ew-bfe-v9/about), click Download → GeoPackage, save to `temp/`
2. **Download Census topics:** Run `uv run python data/download_census.py`
3. **Download Built Up Areas:** Go to [OS Data Hub](https://osdatahub.os.uk/downloads/open/BuiltUpAreas), download GeoPackage, extract to `temp/`
4. **Process boundaries:** Run `uv run python data/process_boundaries.py`
5. **Download OS Open Map Local:** Go to [OS Data Hub](https://osdatahub.os.uk/downloads/open/OpenMapLocal), download GeoPackage (GB), extract to `temp/os_open_local/`
6. **Download EPC data:** Register at [epc.opendatacommunities.org](https://epc.opendatacommunities.org/), download "All domestic certificates", extract to `temp/epc/`
7. **Process EPC data:** Run `uv run python data/process_epc.py`
8. **Process LiDAR building heights:** Run `uv run python data/process_lidar.py`

## Pipeline Outputs

| Script                  | Output                                       | Description                                            |
| ----------------------- | -------------------------------------------- | ------------------------------------------------------ |
| `download_census.py`    | `temp/census_oa_joined.gpkg`                 | OA boundaries with 8 Census topic tables joined        |
| `process_boundaries.py` | `temp/boundaries/built_up_areas.gpkg`        | Cleaned individual built-up area polygons              |
|                         | `temp/boundaries/built_up_areas_merged.gpkg` | Merged conurbations (adjacent areas combined)          |
| `process_epc.py`        | `temp/epc_domestic_cleaned.parquet`          | Deduplicated EPC records (tabular, no geometry)        |
|                         | `temp/epc_domestic_spatial.gpkg`             | EPC records with UPRN point geometry                   |
| `process_lidar.py`      | `temp/lidar/building_heights.gpkg`           | Building polygons with LiDAR-derived height statistics |

## Data Integration Workflow

The analysis unit is the **UPRN** (Unique Property Reference Number). All data sources are linked to UPRNs, which are then aggregated to street network segments via cityseer.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PREPARATION                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OS Open Map Local (buildings)                                              │
│        │                                                                    │
│        ├── heights attached via process_lidar.py                            │
│        │                                                                    │
│        ↓                                                                    │
│  Buildings with heights                                                     │
│        │                                                                    │
│        ↓ compute morphology metrics (process_morphology.py)                 │
│        │                                                                    │
│  Buildings with heights + morphology                                        │
│        │                                                                    │
│        ↓ spatial join (UPRN point ∈ building polygon)                       │
│        │                                                                    │
│  UPRNs inherit building attributes ──────────────────────┐                  │
│                                                          │                  │
│  Census (OA-level)                                       │                  │
│        │                                                 │                  │
│        ↓ spatial interpolation (UPRN point ∈ OA)         │                  │
│        │                                                 ↓                  │
│  UPRNs with census attributes ─────────────────────► UPRN Dataset           │
│                                                          ↑                  │
│  EPCs                                                    │                  │
│        │                                                 │                  │
│        ↓ direct join on UPRN field                       │                  │
│        │                                                 │                  │
│  UPRNs with EPC attributes ──────────────────────────────┘                  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                           NETWORK AGGREGATION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  UPRN Dataset (morphology + census + EPC)                                   │
│        │                                                                    │
│        ↓ cityseer network assignment                                        │
│        │                                                                    │
│  Street segments with aggregated metrics                                    │
│        │                                                                    │
│        ↓ 400m network catchment computation                                 │
│        │                                                                    │
│  Analysis-ready dataset for regression modelling                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Linkage Methods

| Source → Target         | Method                          | Notes                                                |
| ----------------------- | ------------------------------- | ---------------------------------------------------- |
| Buildings → UPRNs       | Spatial join (point-in-polygon) | Multiple UPRNs per building (flats) share morphology |
| Census → UPRNs          | Spatial interpolation           | UPRN inherits OA-level attributes                    |
| EPCs → UPRNs            | Direct join on `UPRN` field     | Post-Nov 2021 records only                           |
| UPRNs → Street segments | cityseer network assignment     | Automated via pedestrian network analysis            |

### Processing Stages

| Stage | Script (planned)        | Input                             | Output                            |
| ----- | ----------------------- | --------------------------------- | --------------------------------- |
| 1     | `process_lidar.py` ✓    | Buildings + LiDAR tiles           | Buildings with heights            |
| 2     | `process_morphology.py` | Buildings with heights            | Buildings with morphology metrics |
| 3     | `process_uprn.py`       | Buildings, Census, EPCs, OS UPRNs | UPRN-level integrated dataset     |
| 4     | `process_network.py`    | UPRN dataset, OS Open Roads       | Street segment aggregations       |

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
- **Coverage**: England & Wales only.

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

| Field                        | Description                      |
| ---------------------------- | -------------------------------- |
| `UPRN`                       | Unique Property Reference Number |
| `CURRENT_ENERGY_EFFICIENCY`  | SAP rating (1-100)               |
| `CURRENT_ENERGY_RATING`      | EPC band (A-G)                   |
| `ENERGY_CONSUMPTION_CURRENT` | Estimated annual kWh             |
| `TOTAL_FLOOR_AREA`           | Floor area in m²                 |
| `PROPERTY_TYPE`              | Detached, Semi, Terraced, Flat   |
| `CONSTRUCTION_AGE_BAND`      | Age of construction              |
| `TENURE`                     | Owner-occupied, Rental, Social   |
| `MAIN_FUEL`                  | Primary heating fuel             |

### Data Quality Notes

- **UPRN linkage**: Available since November 2021; earlier certificates excluded from spatial analysis.
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
| Composite DSM | [DEFRA Data](https://environment.data.gov.uk/dataset/9ba4d5ac-d596-445a-9056-dae3ddec0178) | 1m         | ~99% Eng |
| Composite DTM | [DEFRA Data](https://environment.data.gov.uk/dataset/13787b9a-26a4-4775-8523-806d13af58fc) | 1m         | ~99% Eng |

### Key Details

- **Vintage**: 2022 composite (surveys 2000-2022)
- **Accuracy**: ±5-15cm RMSE vertical
- **Tiles**: 5km GeoTIFF, EPSG:27700
- **Building heights**: `nDSM = DSM - DTM`

### Processing

Use `process_lidar.py` which streams tiles via the DEFRA WCS API:

```bash
uv run python data/process_lidar.py
```

**Prerequisites:** Requires OS Open Map Local building footprints (step 5 in setup).

For each built-up area boundary, the script:

1. Loads building footprints from OS Open Map Local
2. Downloads required DSM/DTM tiles via WCS (async, max 8 concurrent)
3. Computes nDSM (DSM - DTM) in memory
4. Extracts building heights via zonal statistics
5. Caches per-boundary results for resumability
6. Deletes tiles and moves to next boundary

### Output Schema

`temp/lidar/building_heights.gpkg`:

| Column               | Type    | Description                         |
| -------------------- | ------- | ----------------------------------- |
| `id`                 | string  | OS building identifier              |
| `geometry`           | polygon | Building footprint (EPSG:27700)     |
| `height_min`         | float   | Minimum nDSM value within footprint |
| `height_max`         | float   | Maximum nDSM value (approx. ridge)  |
| `height_mean`        | float   | Mean height across footprint        |
| `height_median`      | float   | Median height across footprint      |
| `height_std`         | float   | Height standard deviation           |
| `height_pixel_count` | int     | Number of LiDAR pixels in footprint |

**Note:** Buildings outside LiDAR coverage have null height values. LiDAR-derived continuous heights provide finer resolution than EPC storey counts.
