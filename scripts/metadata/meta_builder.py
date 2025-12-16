"""
DuckDB Metadata Builder for FloodSQL_Bench
------------------------------------------

This script scans all Parquet files in the `data/` directory, 
normalizes table names (removes `_tx_fl_la` suffix), 
and generates a metadata JSON file with schema, descriptions, 
row counts, and sample rows. It also attaches global metadata 
about cross-table relations, rules, and hints.

Output: ./db/metadata_parquet.json
"""

import duckdb
import os
import json
import math
import pandas as pd

# ============================================================
# 1. Field Descriptions
# ============================================================
FIELD_DESCRIPTIONS = {
    # Table-specific descriptions
    # U.S. Census tracts' meta and entries
    "census_tracts": {
        "layer_category": "polygon + tract-level",
        "key_identifier": ["GEOID"],
        "spatial_identifier": ["geometry"],
        "_meta": (
            "Polygon-based spatial layer representing U.S. Census tracts, "
            "Each record corresponds to a single tract area defined by the U.S. Census Bureau. "
            "This table supports key-based joins via the 11-digit GEOID and spatial joins with polygon-based layers "
            "and point-based layers through its geometry field. "
        ),

        "GEOID": (
            "11-digit census tract identifier composed of STATEFP, COUNTYFP, and TRACTCODE. "
            "This field serves as the primary join key for connecting tract-level datasets such as NRI, SVI, CRE, and NFIP claims, "
            "and county-level datasets such as county and schools."
        ),

        "STATEFP": (
            "2-digit state FIPS code identifying the U.S. state containing the tract. "
            "This field is mainly used for filtering or grouping results by state "
            "(e.g., restricting queries to Texas, Florida, or Louisiana), "
            "but it does not participate in cross-table joins."
        ),

        "COUNTYFP": (
            "3-digit county FIPS code identifying the county within a state. "
            "Used primarily for grouping or summarizing results at the county level, "
            "rather than for direct joins across tables."
        ),

        "NAME": (
            "Human-readable census tract label (e.g., 'Tract 1234.02'). "
            "This field provides a descriptive name for reporting or visualization purposes, "
            "but is not used for joins or aggregations."
        ),

        "geometry": (
            "Polygon geometry representing the spatial boundary of each census tract, stored in OGC:CRS84. "
            "This field serves as the spatial join key for overlay operations with polygon layers (floodplain, county, ZCTA) "
            "and point layers (hospitals, schools). "
        )
    },

    # U.S. counties
    "county": {
        "layer_category": "polygon + county-level",
        "key_identifier": ["GEOID"],
        "spatial_identifier": ["geometry"],
        "_meta": (
            "County-level administrative polygon layer representing U.S. counties. "
            "Each record corresponds to a single county area within a state. "
            "This table supports both key-based joins via the 5-digit GEOID and spatial joins via the geometry field. "
        ),

        "GEOID": (
            "5-digit county FIPS code composed of the 2-digit STATEFP and 3-digit COUNTYFP. "
            "Serves as the primary join key for county-level aggregations and joins with other tables " 
            "such as NFIP claims, hospitals, census tracts, NRI, SIV, and CRE. "
            "Used in both direct joins and prefix-based joins derived from tract GEOIDs."
        ),

        "STATEFP": (
            "2-digit state FIPS code identifying the U.S. state containing the county. "
            "Mainly used for filtering or grouping results by state (e.g., Texas, Florida, or Louisiana), "
            "but not for cross-table joins."
        ),

        "COUNTYFP": (
            "3-digit county FIPS code identifying the county within its state. "
            "Used primarily for summarization or grouping within a state, "
            "but not for key-based joins across tables."
        ),

        "NAME": (
            "Official county name (e.g., 'Harris', 'Orleans', 'Palm Beach'). "
            "Used as a human-readable label in reporting or visualization outputs, "
            "but not as a join key."
        ),

        "geometry": (
            "Polygon geometry representing county boundaries, stored in OGC:CRS84. "
            "Serves as the spatial join key for overlay operations with polygon layers (floodplain, census tracts, ZCTA) "
            "and point layers (hospitals, schools). "
        )
    },


    "zcta": {
        "layer_category": "polygon",
        "key_identifier": [],
        "spatial_identifier": ["geometry"],
        "_meta": (
            "ZIP Code Tabulation Area (ZCTA) polygon layer representing U.S. postal ZIP code regions. "
            "ZCTAs are spatial constructs derived from census data and are not identical to USPS ZIP codes. "
            "This table supports spatial joins via its geometry field but is not used for key-based joins."
        ),

        "GEOID": (
            "5-digit ZCTA identifier that usually corresponds to a USPS ZIP code "
            "(e.g., GEOID='77845' represents the area approximating ZIP code 77845). "
            "Used for display, filtering, or spatial selection, but not for key-based joins."
        ),

        "STATEFP": (
            "2-digit state FIPS code identifying the state containing this ZCTA polygon. "
            "Used mainly for filtering or grouping by state, but not for cross-table joins."
        ),

        "geometry": (
            "Polygon geometry representing the ZCTA boundary, stored in OGC:CRS84. "
            "Serves as the spatial join key for polygon-based overlays with floodplain, census tracts, and county layers, "
            "as well as for point-based spatial joins with hospitals and schools. "
            "ZCTA polygons are derived from census tract boundaries and may align or slightly differ along tract edges."
        )
    },


    "floodplain": {
        "layer_category": "polygon",
        "key_identifier": [],
        "spatial_identifier": ["geometry"],
        "_meta": (
            "FEMA Flood Hazard polygon layer representing physical flood risk zones. "
            "Each record corresponds to a FEMA-defined flood hazard area. "
            "This table is spatial-only and supports spatial joins through its geometry field, "
            "but is not used for key-based joins."
        ),

        "GFID": (
            "Unique FEMA floodplain polygon identifier. "
            "Serves as an internal region identifier within the floodplain dataset, "
            "but is not used for joins across tables."
        ),

        "FLD_ZONE": (
            "FEMA Flood Hazard Zone classification (e.g., 'AE', 'VE', 'X', 'AO'). "
            "Indicates the level and type of flood risk associated with each polygon. "
            "Commonly used for filtering or categorizing flood exposure in spatial analyses."
        ),

        "STATEFP": (
            "2-digit state FIPS code identifying the state containing the floodplain polygon. "
            "Used mainly for filtering or grouping results by state, "
            "but not for cross-table joins."
        ),

        "geometry": (
            "Polygon geometry representing FEMA flood hazard boundaries, stored in OGC:CRS84. "
            "Serves as the spatial join key for overlay operations with polygon layers "
            "(census tracts, ZCTA, county) and point layers (hospitals, schools). "
            "Used to determine which geographic regions or facilities fall within flood hazard areas."
        )
    },


    "claims": {
        "layer_category": "tract-level",
        "key_identifier": ["GEOID"],
        "spatial_identifier": [],
        "_meta": (
            "Key-based table containing National Flood Insurance Program (NFIP) flood claim records. "
            "Each record represents a single insurance claim event linked to a census tract. "
            "All joins are key-based using the 11-digit GEOID; spatial joins are not applicable. "
        ),

        "id": (
            "Unique NFIP claim record identifier. "
            "Each entry corresponds to a single insurance claim in the dataset."
        ),

        "GEOID": (
            "11-digit census tract GEOID composed of STATEFP, COUNTYFP, and TRACTCODE. "
            "Serves as the primary join key for connecting to tract-level datasets such as census_tracts, NRI, SVI, and CRE, "
            "and can also be used for county-level aggregations through GEOID prefix joins."
        ),

        "STATEFP": (
            "2-digit state FIPS code identifying the U.S. state where the claim originated. "
            "Used for filtering or grouping results by state, but not for cross-table joins."
        ),

        "dateOfLoss": (
            "Date of the flood insurance claim loss event, recorded in ISO format (e.g., '2000-10-03T00:00:00'). "
            "Indicates when the flood damage occurred."
        ),

        "amountPaidOnBuildingClaim": (
            "Insurance payout amount (in USD) for structural building damage. "
            "Used in assessing physical exposure and structural loss severity."
        ),

        "amountPaidOnContentsClaim": (
            "Insurance payout amount (in USD) for damage to building contents. "
            "Represents internal property losses within the insured structure."
        ),

        "amountPaidOnIncreasedCostOfComplianceClaim": (
            "Insurance payout amount (in USD) for Increased Cost of Compliance (ICC). "
            "Covers expenses related to meeting post-flood elevation, relocation, or reconstruction requirements."
        ),

        "geometry": (
            "Point geometry representing the claim location, stored in OGC:CRS84. "
            "Although geometry is available, this table is not used for spatial joins; "
            "its joins are purely key-based via the 11-digit GEOID."
        )
    },

    "schools": {
        "layer_category": "point + ZIP-level + county-level",
        "key_identifier": ["ZIP"],
        "spatial_identifier": ["LAT", "LON"],
        "_meta": (
            "Point-based layer representing school facilities across Texas, Florida, and Louisiana. "
            "Each record corresponds to one educational institution, including colleges, private, and public schools. "
            "This table supports spatial joins through coordinates (LON, LAT) for point-in-polygon operations "
            "with polygon layers. "
            "ZIP serves as the primary key for key-based joins with hospitals. "
            "Geometry can be constructed dynamically using ST_Point(LON, LAT) in WGS84 (EPSG:4326)."
        ),

        "SCHOOL_ID": (
            "Unique identifier for each school, derived from IPEDS or NCES datasets. "
            "Used to uniquely distinguish school records within the dataset."
        ),

        "NAME": (
            "Official name of the school. "
            "Serves as a descriptive label for reporting and visualization."
        ),

        "ADDRESS": (
            "Street address of the school. "
            "Provides textual location information for reference or display purposes."
        ),

        "CITY": (
            "City in which the school is located, represented in uppercase. "
            "Used as an administrative attribute for filtering or grouping by city."
        ),

        "STATE": (
            "Two-letter U.S. state abbreviation (e.g., 'TX', 'FL', 'LA'). "
            "Used to identify the state associated with each school record."
        ),

        "ZIP": (
            "5-digit postal ZIP code. "
            "Serves as the primary key for key-based joins with hospitals "
            "and can be used for regional aggregation or filtering."
        ),

        "LAT": (
            "Latitude coordinate in decimal degrees (WGS84). "
            "Used with LON to construct geometry for spatial joins in EPSG:4326 coordinate reference system."
        ),

        "LON": (
            "Longitude coordinate in decimal degrees (WGS84). "
            "Used with LAT to construct spatial points for spatial joins in EPSG:4326."
        ),

        "TYPE": (
            "Categorical type of school (e.g., 'COLLEGE', 'PRIVATE_SCHOOL', 'PUBLIC_SCHOOL'). "
            "Used for classification and filtering by educational institution type."
        ),

        "STATEFP": (
            "2-digit state FIPS code identifying the U.S. state where the school is located. "
            "Used for filtering or aggregation by state, but not for cross-table joins."
        ),

        "UNIQUE_ID": (
            "Synthetic unique identifier generated by concatenating STATEFP, TYPE, and SCHOOL_ID. "
            "Ensures record-level uniqueness across the combined dataset."
        ),

        "geometry": (
            "Optional geometry field representing the spatial point location of each school. "
            "Although geometry may be present, this table primarily uses LAT and LON coordinates for spatial joins."
        )
    },


    "hospitals": {
        "layer_category": "point + ZIP-level",
        "key_identifier": ["COUNTYFIPS", "ZIP"],
        "spatial_identifier": ["LAT", "LON"],
        "_meta": (
            "Point-based layer representing hospital facilities across Texas, Florida, and Louisiana. "
            "Each record corresponds to a single healthcare facility. "
            "This table supports both spatial joins through coordinates (LON, LAT) "
            "and key-based joins with schools via ZIP, with county-level and tract-level datasets via the 5-digit COUNTYFIPS code, "
            "Geometry can be constructed dynamically using ST_Point(LON, LAT) in WGS84 (EPSG:4326)."
        ),

        "HOSPITAL_ID": (
            "Unique identifier for each hospital, assigned by the source dataset. "
            "Used to uniquely identify individual healthcare facilities."
        ),

        "NAME": (
            "Official name of the hospital. "
            "Serves as a descriptive label for reporting and visualization."
        ),

        "ADDRESS": (
            "Street address of the hospital. "
            "Provides location context for reference or display purposes."
        ),

        "CITY": (
            "City in which the hospital is located, represented in uppercase. "
            "Used as an administrative attribute for filtering or grouping by city."
        ),

        "STATE": (
            "Two-letter U.S. state abbreviation (e.g., 'TX', 'FL', 'LA'). "
            "Used to identify the state in which the hospital operates."
        ),

        "ZIP": (
            "5-digit postal ZIP code (USPS). "
            "Used for filtering or display purposes but not for spatial or key-based joins with ZCTA, "
            "since USPS ZIP codes do not align perfectly with Census ZCTA boundaries. "
            "ZIP-level relationships should instead be established through spatial joins "
            "using hospital coordinates (ST_Point(LON, LAT)) and ZCTA geometry."
        ),

        "COUNTY": (
            "Official name of the county where the hospital is located. "
            "Used for descriptive or reporting purposes but not for joins."
        ),

        "COUNTYFIPS": (
            "5-digit county FIPS code composed of STATEFP and COUNTYFP, "
            "identifying the county associated with each hospital. "
            "Serves as the primary key for key-based joins with county-level datasets" 
            "and tract-level datasets."
        ),

        "LAT": (
            "Latitude coordinate in decimal degrees (WGS84). "
            "Used with LON to construct spatial points for spatial joins in EPSG:4326."
        ),

        "LON": (
            "Longitude coordinate in decimal degrees (WGS84). "
            "Used with LAT to construct geometry for spatial joins in EPSG:4326."
        ),

        "TYPE": (
            "Categorical type of hospital (e.g., 'General Acute Care', 'Psychiatric', 'Critical Access', 'Military'). "
            "Used for classification or filtering by hospital category."
        ),

        "STATEFP": (
            "2-digit state FIPS code identifying the U.S. state in which the hospital is located. "
            "Used for filtering or aggregation by state, but not for cross-table joins."
        ),

        "UNIQUE_ID": (
            "Synthetic unique identifier generated by concatenating STATEFP, 'hospital', and HOSPITAL_ID. "
            "Ensures record-level uniqueness across the combined dataset."
        ),

        "geometry": (
            "Optional geometry field representing the hospital's location. "
            "Although geometry may exist, this table primarily uses LAT and LON coordinates for spatial joins."
        )
    },

    
    # -------------------------------
    # CRE dataset (Community Risk Estimates, Census tracts)
    # -------------------------------
    "cre": {
        "layer_category": "tract-level",
        "key_identifier": ["GEOID"],
        "spatial_identifier": [],
        "_meta": (
            "Key-based table representing the Community Resilience Estimates (CRE) dataset published by the U.S. Census Bureau. "
            "Provides tract-level measures of population resilience to natural hazards, estimating how well communities can withstand and recover from disasters. "
            "This dataset contains no geometry field; all joins are key-based via the 11-digit census tract GEOID."
        ),

        "GEO_ID": (
            "Full Census geographic identifier including a prefix (e.g., '1400000US12001000201'). "
            "Provided for reference only and not used in joins."
        ),

        "GEOID": (
            "11-digit census tract GEOID composed of STATEFP, COUNTYFP, and TRACT code. "
            "Serves as the primary join key linking CRE data to tract-level datasets such as census_tracts, NRI, SVI, and NFIP claims."
        ),

        "TRACT": (
            "6-digit census tract code within a county. "
            "Not unique at the national level; the full GEOID should be used for joins."
        ),

        "STATE": (
            "Full state name corresponding to the tract. "
            "Used as a descriptive field for filtering or reporting."
        ),

        "COUNTY": (
            "County name corresponding to the tract. "
            "Used as a descriptive attribute, not for joins."
        ),

        "POPUNI": (
            "Population universe used in tract-level resilience estimation. "
            "Represents the total number of individuals included in the CRE modeling population."
        ),

        "WATER_TRACT": (
            "Flag indicating whether a census tract is composed entirely of water. "
            "A value of 1 denotes a tract with no resident population; 0 denotes tracts with land area and population."
        ),

        "PRED0_E": (
            "Estimated number of individuals with zero components of social vulnerability."
        ),
        "PRED0_M": (
            "Margin of error for PRED0_E, representing uncertainty in the estimated number of individuals with zero vulnerability components."
        ),
        "PRED0_PE": (
            "Percentage of individuals with zero components of social vulnerability."
        ),
        "PRED0_PM": (
            "Margin of error (percentage points) for the percentage of individuals with zero components of social vulnerability."
        ),

        "PRED12_E": (
            "Estimated number of individuals with one or two components of social vulnerability."
        ),
        "PRED12_M": (
            "Margin of error for PRED12_E, representing uncertainty in the estimated number of individuals with one or two vulnerability components."
        ),
        "PRED12_PE": (
            "Percentage of individuals with one or two components of social vulnerability."
        ),
        "PRED12_PM": (
            "Margin of error (percentage points) for the percentage of individuals with one or two components of social vulnerability."
        ),

        "PRED3_E": (
            "Estimated number of individuals with three or more components of social vulnerability."
        ),
        "PRED3_M": (
            "Margin of error for PRED3_E, representing uncertainty in the estimated number of individuals with three or more vulnerability components."
        ),
        "PRED3_PE": (
            "Percentage of individuals with three or more components of social vulnerability."
        ),
        "PRED3_PM": (
            "Margin of error (percentage points) for the percentage of individuals with three or more components of social vulnerability."
        ),

        "geometry": (
            "No geometry field is included in this dataset; spatial joins are not applicable."
        )
    },


    # -------------------------------
    # NRI Flood dataset (National Risk Index)
    # -------------------------------
    "nri": {
        "layer_category": "tract-level",
        "key_identifier": ["GEOID"],
        "spatial_identifier": [],
        "_meta": (
            "Key-based table representing the National Risk Index (NRI) dataset developed by FEMA. "
            "Provides tract-level estimates of natural hazard risk across the United States, quantifying expected losses, exposure, and composite risk scores "
            "for multiple hazard types, including Coastal Flood (CFLD) and Riverine Flood (RFLD). "
            "This dataset contains no geometry field; all joins are key-based via the 11-digit census tract GEOID."
        ),

        # ------------------------------------------------------------
        # Keys
        # ------------------------------------------------------------
        "GEOID": (
            "11-digit census tract GEOID composed of STATEFP, COUNTYFP, and TRACT code. "
            "Serves as the primary join key linking NRI data to tract-level datasets such as census_tracts, SVI, CRE, and NFIP claims."
        ),

        "STATE": (
            "Two-letter state abbreviation (e.g., 'TX', 'FL', 'LA'). "
            "Used primarily for filtering or summarization by state."
        ),

        # ------------------------------------------------------------
        # Coastal Flood (CFLD)
        # ------------------------------------------------------------
        "CFLD_EVNTS": (
            "Number of observed or modeled coastal flood events within the tract, derived from FEMA NRI event records."
        ),
        "CFLD_AFREQ": (
            "Average annual frequency of coastal flood events estimated from historical and modeled data."
        ),
        "CFLD_EXPPE": (
            "Expected population exposure to coastal flooding, measured as the number of residents affected."
        ),
        "CFLD_EXPT": (
            "Expected transportation asset exposure to coastal flooding, representing potential impact on roads and bridges."
        ),
        "CFLD_EXP_AREA": (
            "Estimated land area (in square meters) expected to be affected by coastal flooding."
        ),
        "CFLD_HLRB": (
            "Historical loss ratio for buildings due to coastal flooding, representing the fraction of asset value lost."
        ),
        "CFLD_HLRP": (
            "Historical loss ratio for population impacts (e.g., fatalities, injuries) due to coastal flooding."
        ),
        "CFLD_HLRF": (
            "Historical loss ratio for agricultural crops affected by coastal flooding."
        ),
        "CFLD_HLRA": (
            "Historical loss ratio for land or area damage due to coastal flooding."
        ),
        "CFLD_EXPB": (
            "Expected exposure value of buildings to coastal flooding, expressed in U.S. dollars."
        ),
        "CFLD_EXPP": (
            "Expected exposure value of population to coastal flooding, represented as a monetized equivalent."
        ),
        "CFLD_EXPF": (
            "Expected crop exposure value under coastal flooding, expressed in U.S. dollars."
        ),
        "CFLD_EXPA": (
            "Expected land or area exposure to coastal flooding, measured in square meters."
        ),
        "CFLD_EALB": (
            "Expected annual loss for buildings due to coastal flooding, expressed in U.S. dollars."
        ),
        "CFLD_EALP": (
            "Expected annual loss for population due to coastal flooding, incorporating fatalities, injuries, or monetized equivalents."
        ),
        "CFLD_EALF": (
            "Expected annual loss for crops caused by coastal flooding, in U.S. dollars."
        ),
        "CFLD_EALA": (
            "Expected annual loss for land or area value due to coastal flooding, in U.S. dollars."
        ),
        "CFLD_EALS": (
            "Expected annual loss adjusted for social vulnerability under coastal flooding scenarios."
        ),
        "CFLD_EALR": (
            "Expected annual loss adjusted for community resilience under coastal flooding scenarios."
        ),
        "CFLD_RISKR": (
            "Categorical coastal flood risk rating (e.g., Very Low, Relatively Low, Moderate, Relatively High, Very High)."
        ),
        "CFLD_RISKS": (
            "Normalized coastal flood risk score ranging from 0 to 100; higher values indicate greater relative risk."
        ),
        "CFLD_RISKV": (
            "Continuous numeric risk value underlying the rating and score for coastal flooding."
        ),

        # ------------------------------------------------------------
        # Riverine Flood (RFLD)
        # ------------------------------------------------------------
        "RFLD_EVNTS": (
            "Number of observed or modeled riverine flood events within the tract."
        ),
        "RFLD_AFREQ": (
            "Average annual frequency of riverine flood events expected per year."
        ),
        "RFLD_EXPPE": (
            "Expected population exposure to riverine flooding, measured as the number of people affected."
        ),
        "RFLD_EXPT": (
            "Expected transportation infrastructure exposure to riverine flooding, such as roads and bridges."
        ),
        "RFLD_EXP_AREA": (
            "Expected affected land area due to riverine flooding, measured in square meters."
        ),
        "RFLD_HLRB": (
            "Historical loss ratio for buildings resulting from riverine flooding."
        ),
        "RFLD_HLRP": (
            "Historical loss ratio for population impacts caused by riverine flooding."
        ),
        "RFLD_HLRF": (
            "Historical loss ratio for agricultural crops affected by riverine flooding."
        ),
        "RFLD_HLRA": (
            "Historical loss ratio for land or area damage due to riverine flooding."
        ),
        "RFLD_EXPB": (
            "Expected exposure value of buildings under riverine flooding scenarios (USD)."
        ),
        "RFLD_EXPP": (
            "Expected exposure value of population under riverine flooding scenarios, monetized as equivalent value of statistical life."
        ),
        "RFLD_EXPF": (
            "Expected crop exposure value under riverine flooding scenarios (USD)."
        ),
        "RFLD_EXPA": (
            "Expected land or area exposure to riverine flooding (square meters)."
        ),
        "RFLD_EALB": (
            "Expected annual loss for buildings caused by riverine flooding (USD)."
        ),
        "RFLD_EALP": (
            "Expected annual loss for population impacts (fatalities, injuries, or VSL-adjusted) due to riverine flooding."
        ),
        "RFLD_EALF": (
            "Expected annual loss for crops caused by riverine flooding (USD)."
        ),
        "RFLD_EALA": (
            "Expected annual loss for land or area value due to riverine flooding (USD)."
        ),
        "RFLD_EALS": (
            "Expected annual loss adjusted for social vulnerability under riverine flooding scenarios."
        ),
        "RFLD_EALR": (
            "Expected annual loss adjusted for community resilience under riverine flooding scenarios."
        ),
        "RFLD_EALT": (
            "Total expected annual loss for riverine flooding (USD)."
        ),
        "RFLD_RISKR": (
            "Categorical riverine flood risk rating (Very Low, Low, Moderate, High, Very High)."
        ),
        "RFLD_RISKS": (
            "Normalized riverine flood risk score ranging from 0 to 100, representing percentile rank across tracts."
        ),
        "RFLD_RISKV": (
            "Underlying continuous numeric riverine flood risk value used to derive the categorical rating and normalized score."
        ),

        # ------------------------------------------------------------
        # Geometry / Join note
        # ------------------------------------------------------------
        "geometry": (
            "No geometry field is available in this dataset; spatial joins are not applicable."
        )
    },

    # -------------------------------
    # SVI dataset (Social Vulnerability Index, Census tracts)
    # -------------------------------
    "svi": {
        "layer_category": "tract-level",
        "key_identifier": ["GEOID"],
        "spatial_identifier": [],
        "_meta": (
            "Key-based table representing the Social Vulnerability Index (SVI) dataset published by the CDC/ATSDR. "
            "Provides tract-level demographic, socioeconomic, housing, and mobility indicators "
            "used to measure a community's vulnerability to natural hazards and disasters. "
            "This dataset contains no geometry field; all joins are key-based via the 11-digit census tract GEOID. "
            "A FIPS field identical to GEOID exists in the source data, but GEOID is used as the canonical key in FloodSQL-Bench."
        ),

        # ------------------------------------------------------------
        # Keys and basic geographic identifiers
        # ------------------------------------------------------------
        "GEOID": (
            "11-digit census tract GEOID composed of STATEFP, COUNTYFP, and TRACT code. "
            "Serves as the primary join key linking SVI to tract-level datasets such as census_tracts, NRI, CRE, and NFIP claims."
        ),

        "FIPS": (
            "11-digit census tract identifier equivalent to GEOID. "
            "Included for completeness but not used for joins; GEOID is preferred."
        ),

        "ST": "State FIPS code (numeric).",
        "STATE": "Full state name.",
        "ST_ABBR": "Two-letter state abbreviation (e.g., 'TX', 'FL', 'LA').",
        "STCNTY": "Combined state + county FIPS code (5 digits).",
        "COUNTY": "County name.",
        "LOCATION": "Text label describing tract's county and state.",
        "AREA_SQMI": "Tract area in square miles (approximate).",

        # ------------------------------------------------------------
        # Core demographic and housing estimates
        # ------------------------------------------------------------
        "E_TOTPOP": "Estimated total population.",
        "M_TOTPOP": "Margin of error for total population estimate.",
        "E_HU": "Estimated number of housing units.",
        "M_HU": "Margin of error for housing unit estimate.",
        "E_HH": "Estimated number of households.",
        "M_HH": "Margin of error for household estimate.",

        # ------------------------------------------------------------
        # Social vulnerability variables (estimates + Margin of error)
        # ------------------------------------------------------------
        "E_POV150": "Estimated population below 150% poverty line.",
        "M_POV150": "Margin of error for poverty estimate.",
        "E_UNEMP": "Estimated unemployed population.",
        "M_UNEMP": "Margin of error for unemployment estimate.",
        "E_HBURD": "Estimated households with high housing cost burden.",
        "M_HBURD": "Margin of error for housing cost burden estimate.",
        "E_NOHSDP": "Estimated population without high school diploma.",
        "M_NOHSDP": "Margin of error for education estimate.",
        "E_UNINSUR": "Estimated population without health insurance.",
        "M_UNINSUR": "Margin of error for health insurance estimate.",
        "E_AGE65": "Estimated population aged 65 and older.",
        "M_AGE65": "Margin of error for elderly population estimate.",
        "E_AGE17": "Estimated population aged 17 and younger.",
        "M_AGE17": "Margin of error for youth population estimate.",
        "E_DISABL": "Estimated population with disabilities.",
        "M_DISABL": "Margin of error for disability estimate.",
        "E_SNGPNT": "Estimated single-parent households.",
        "M_SNGPNT": "Margin of error for single-parent household estimate.",
        "E_LIMENG": "Estimated population with limited English proficiency.",
        "M_LIMENG": "Margin of error for limited English estimate.",
        "E_MINRTY": "Estimated minority population.",
        "M_MINRTY": "Margin of error for minority population estimate.",
        "E_NOVEH": "Estimated households with no vehicle available.",
        "M_NOVEH": "Margin of error for no-vehicle household estimate.",

        # ------------------------------------------------------------
        # Derived percentages (EP_xx) and margins of error (MP_xx)
        # ------------------------------------------------------------
        "EP_POV150": "Estimated percent of population below 150% poverty line.",
        "MP_POV150": "Margin of error for poverty percentage.",
        "EP_UNEMP": "Estimated percent unemployed.",
        "MP_UNEMP": "Margin of error for unemployment percentage.",
        "EP_HBURD": "Estimated percent of households with high housing cost burden.",
        "MP_HBURD": "Margin of error for housing cost burden percentage.",
        "EP_NOHSDP": "Estimated percent without high school diploma.",
        "MP_NOHSDP": "Margin of error for education percentage.",
        "EP_UNINSUR": "Estimated percent without health insurance.",
        "MP_UNINSUR": "Margin of error for health insurance percentage.",
        "EP_AGE65": "Estimated percent aged 65 and older.",
        "MP_AGE65": "Margin of error for elderly percentage.",
        "EP_AGE17": "Estimated percent aged 17 and younger.",
        "MP_AGE17": "Margin of error for youth percentage.",
        "EP_DISABL": "Estimated percent with disabilities.",
        "MP_DISABL": "Margin of error for disability percentage.",
        "EP_SNGPNT": "Estimated percent of single-parent households.",
        "MP_SNGPNT": "Margin of error for single-parent households percentage.",
        "EP_LIMENG": "Estimated percent with limited English.",
        "MP_LIMENG": "Margin of error for limited English percentage.",
        "EP_MINRTY": "Estimated percent of minority population.",
        "MP_MINRTY": "Margin of error for minority percentage.",
        "EP_NOVEH": "Estimated percent of households with no vehicle.",
        "MP_NOVEH": "Margin of error for no-vehicle percentage.",

        # ------------------------------------------------------------
        # Thematic indices (summary percentiles and relative ranks)
        # ------------------------------------------------------------
        "SPL_THEME1": "Summary percentile rank for Theme 1: Socioeconomic Status.",
        "RPL_THEME1": "Relative percentile for Theme 1 (Socioeconomic).",
        "SPL_THEME2": "Summary percentile rank for Theme 2: Household Composition & Disability.",
        "RPL_THEME2": "Relative percentile for Theme 2 (Household Composition & Disability).",
        "SPL_THEME3": "Summary percentile rank for Theme 3: Minority Status & Language.",
        "RPL_THEME3": "Relative percentile for Theme 3 (Minority Status & Language).",
        "SPL_THEME4": "Summary percentile rank for Theme 4: Housing & Transportation.",
        "RPL_THEME4": "Relative percentile for Theme 4 (Housing & Transportation).",
        "SPL_THEMES": "Overall summary percentile rank for all four SVI themes combined.",
        "RPL_THEMES": "Overall relative vulnerability percentile across all themes.",

        # ------------------------------------------------------------
        # Flags (binary vulnerability indicators)
        # ------------------------------------------------------------
        "F_POV150": "Flag (1 = high poverty vulnerability, 0 = otherwise).",
        "F_UNEMP": "Flag (1 = high unemployment vulnerability).",
        "F_HBURD": "Flag (1 = high housing cost burden).",
        "F_NOHSDP": "Flag (1 = low education vulnerability).",
        "F_UNINSUR": "Flag (1 = high insurance vulnerability).",
        "F_THEME1": "Flag (1 = Theme 1 Socioeconomic high vulnerability).",
        "F_AGE65": "Flag (1 = elderly population vulnerability).",
        "F_AGE17": "Flag (1 = youth population vulnerability).",
        "F_DISABL": "Flag (1 = disability vulnerability).",
        "F_SNGPNT": "Flag (1 = single-parent households vulnerability).",
        "F_LIMENG": "Flag (1 = limited English vulnerability).",
        "F_THEME2": "Flag (1 = Theme 2 Household & Disability high vulnerability).",
        "F_MINRTY": "Flag (1 = minority vulnerability).",
        "F_THEME3": "Flag (1 = Theme 3 Minority & Language high vulnerability).",
        "F_MUNIT": "Flag (1 = mobile housing vulnerability).",
        "F_MOBILE": "Flag (1 = mobile housing population vulnerability).",
        "F_CROWD": "Flag (1 = crowded housing vulnerability).",
        "F_NOVEH": "Flag (1 = no-vehicle households vulnerability).",
        "F_GROUPQ": "Flag (1 = group quarters vulnerability).",
        "F_THEME4": "Flag (1 = Theme 4 Housing & Transportation high vulnerability).",
        "F_TOTAL": "Flag (1 = overall high vulnerability).",

        # ------------------------------------------------------------
        # Additional indicators (Internet access, race/ethnicity)
        # ------------------------------------------------------------
        "E_NOINT": "Estimated population with no Internet access.",
        "M_NOINT": "Margin of error for no Internet estimate.",
        "EP_NOINT": "Percent of population with no Internet access.",
        "MP_NOINT": "Margin of error for no Internet percentage.",
        "E_AFAM": "Estimated African-American population.",
        "M_AFAM": "Margin of error for African-American population.",
        "E_HISP": "Estimated Hispanic population.",
        "M_HISP": "Margin of error for Hispanic population.",
        "E_ASIAN": "Estimated Asian population.",
        "M_ASIAN": "Margin of error for Asian population.",
        "E_AIAN": "Estimated American Indian and Alaska Native population.",
        "M_AIAN": "Margin of error for AIAN population.",
        "E_NHPI": "Estimated Native Hawaiian and Pacific Islander population.",
        "M_NHPI": "Margin of error for NHPI population.",
        "E_TWOMORE": "Estimated population of two or more races.",
        "M_TWOMORE": "Margin of error for two-or-more races population.",
        "E_OTHERRACE": "Estimated population of other races.",
        "M_OTHERRACE": "Margin of error for other races population.",
        "EP_AFAM": "Estimated percent African-American.",
        "MP_AFAM": "Margin of error for African-American percentage.",
        "EP_HISP": "Percent Hispanic.",
        "MP_HISP": "Margin of error for Hispanic percentage.",
        "EP_ASIAN": "Estimated percent Asian.",
        "MP_ASIAN": "Margin of error for Asian percentage.",
        "EP_AIAN": "Estimated percent American Indian and Alaska Native.",
        "MP_AIAN": "Margin of error for AIAN percentage.",
        "EP_NHPI": "Estimated percent Native Hawaiian and Pacific Islander.",
        "MP_NHPI": "Margin of error for NHPI percentage.",
        "EP_TWOMORE": "Estimated percent of population of two or more races.",
        "MP_TWOMORE": "Margin of error for two-or-more races percentage.",
        "EP_OTHERRACE": "Estimated percent of population of other races.",
        "MP_OTHERRACE": "Margin of error for other races percentage.",

        # ------------------------------------------------------------
        # Geometry / join information
        # ------------------------------------------------------------
        "geometry": (
            "No geometry field is available in this dataset; spatial joins are not applicable."
        )
    },
}

GLOBAL_METADATA = {
    # ============================================================
    # 1. JOIN RULES (28 total)
    # ============================================================
    "join_rules": {
        # -------------------------------
        # (1) KEY-BASED JOINS (14 total)
        # -------------------------------
        "key_based": {
            "direct": [
                # ===== Claims joins =====
                {"pair": ["claims.GEOID", "census_tracts.GEOID"]},
                {"pair": ["claims.GEOID", "nri.GEOID"]},
                {"pair": ["claims.GEOID", "svi.GEOID"]},
                {"pair": ["claims.GEOID", "cre.GEOID"]},

                # ===== Census_tracts joins =====
                {"pair": ["census_tracts.GEOID", "nri.GEOID"]},
                {"pair": ["census_tracts.GEOID", "svi.GEOID"]},
                {"pair": ["census_tracts.GEOID", "cre.GEOID"]},

                # ===== Hospital and school joins =====
                {"pair": ["hospitals.COUNTYFIPS", "county.GEOID"]},  
                {"pair": ["hospitals.ZIP", "schools.ZIP"]},                       
            ],

            "concat": [
                # ===== GEOID prefix (tract-to-county) joins =====
                {"pair": ["LEFT(claims.GEOID,5)", "county.GEOID"]},
                {"pair": ["LEFT(census_tracts.GEOID,5)", "county.GEOID"]},
                {"pair": ["LEFT(nri.GEOID,5)", "county.GEOID"]},
                {"pair": ["LEFT(svi.GEOID,5)", "county.GEOID"]},
                {"pair": ["LEFT(cre.GEOID,5)", "county.GEOID"]}
            ]
        },

        # -------------------------------
        # (2) SPATIAL JOINS (14 total)
        # -------------------------------
        "spatial": {
            "point_polygon": [
                {"pair": ["ST_Point(schools.LON, schools.LAT)", "census_tracts.geometry"]},
                {"pair": ["ST_Point(schools.LON, schools.LAT)", "floodplain.geometry"]},
                {"pair": ["ST_Point(schools.LON, schools.LAT)", "zcta.geometry"]},
                {"pair": ["ST_Point(schools.LON, schools.LAT)", "county.geometry"]},         
                {"pair": ["ST_Point(hospitals.LON, hospitals.LAT)", "census_tracts.geometry"]},
                {"pair": ["ST_Point(hospitals.LON, hospitals.LAT)", "floodplain.geometry"]},
                {"pair": ["ST_Point(hospitals.LON, hospitals.LAT)", "zcta.geometry"]},
                {"pair": ["ST_Point(hospitals.LON, hospitals.LAT)", "county.geometry"]}   
            ],
            "polygon_polygon": [
                {"pair": ["census_tracts.geometry", "floodplain.geometry"]},
                {"pair": ["census_tracts.geometry", "zcta.geometry"]},
                {"pair": ["floodplain.geometry", "zcta.geometry"]},
                {"pair": ["county.geometry", "floodplain.geometry"]},
                {"pair": ["county.geometry", "zcta.geometry"]},
                {"pair": ["county.geometry", "census_tracts.geometry"]}
            ]
        }
    },

    # ============================================================
    # 2. JOIN RULE DEFINITIONS AND NOTES
    # ============================================================
    "rules": {
        "COUNTYFIPS": (
            "Derived 5-digit county FIPS code = 2-digit STATEFP + 3-digit COUNTYFP, "
            "e.g., '48' + '201' = '48201'."
        ),
        "TRACT_GEOID": (
            "11-digit census tract code = 2-digit STATEFP + 3-digit COUNTYFP + 6-digit TRACT. "
            "Used as the tract-level join key for NRI, SVI, CRE, and Claims datasets."
        ),
        "geometry": (
            "All polygon geometries stored as DuckDB Spatial GEOMETRY type "
            "(WKB format, CRS=EPSG:4326, WGS84)."
        ),
        "hospitals_geometry": (
            "Constructed dynamically with ST_Point(LON, LAT) for spatial joins (EPSG:4326). "
            "Supports both point-in-polygon spatial joins and county-level key-based joins."
        ),
        "schools_geometry": (
            "Constructed dynamically with ST_Point(LON, LAT) for spatial joins (EPSG:4326). "
            "Used only for spatial joins with polygon layers such as census_tracts, ZCTA, and floodplain."
        ),
        "claims_geometry": (
            "Claims records contain tract-level GEOIDs and are used primarily in key-based joins "
            "(event-to-indicator analysis), not in direct spatial joins."
        ),
        "spatial_predicates": (
            "Spatial joins use ST_Contains or ST_Intersects depending on layer semantics: "
            "point layers (schools, hospitals) vs polygon layers (tracts, floodplain, ZCTA)."
        ),
        "NULL_values": (
            "NULL indicates missing data, not zero (e.g., missing claim payouts or incomplete attributes)."
        )
    },

    # ============================================================
    # 3. GENERAL NOTES
    # ============================================================
    "notes": [
        "Both GEOID LIKE '48201%' and LEFT(GEOID, 5) = '48201' can identify records whose GEOID starts with 48201. The latter is generally faster in DuckDB.",
        "For any percentage-type fields (e.g., EP_* from SVI, PRED*_PE from CRE), add filters to remove invalid values: field IS NOT NULL AND field BETWEEN 0 AND 100.",
        "For area-based questions, please use the original geometry data rather than data transformed by SQL functions such as ST_Transform().",
        "RATIO vs. PERCENTAGE: A Ratio or Fraction result is typically expressed as a number between 0 and 1, while a Percentage is the ratio multiplied by 100 (e.g., Ratio = 0.25; Percentage = 25.0).",
        "References to 'NFIP' in question text correspond to the 'claims' table, which stores National Flood Insurance Program claim-level records with tract-level GEOIDs.",
        "For 'Top N' or 'Most common' queries, results return only the primary identifying column (e.g., county_name, tract_geoid, or RFLD_RISKR) rather than a full table with counts or values.",
        "This design ensures list-type outputs (output_type='list(column=...)') where only the top entity names or categories are returned.",
        "All 28 joins are explicitly enumerated under key-based (14) and spatial (14) categories.",
        "Claims, SVI, NRI, and CRE share tract-level GEOIDs, ensuring consistent relational linkage across tract-based datasets.",
        "Hospitals serve as point-based facility layers supporting both key-based (via COUNTYFIPS) and spatial joins. Schools, on the other hand, lack COUNTYFIPS and therefore only participate in spatial joins.",
        "The CITY field in both datasets is stored in uppercase (e.g., 'HOUSTON', 'MIAMI', 'BATON ROUGE'), and queries must match this capitalization when filtering by city name.",
        "ZCTA joins rely on polygon overlays rather than ZIP code text matching.",
        "County joins are performed through derived 5-digit COUNTYFIPS identifiers (STATEFP + COUNTYFP).",
        "Floodplain joins represent physical flood-risk overlays, while census_tracts, ZCTA, and county serve as administrative or statistical boundary layers.",
        "Indicator tables (NRI, SVI, CRE) connect through census_tracts for consistent multi-layer analysis.",
        "STATEFP codes follow FIPS standards: TX=48, FL=12, LA=22.",
        "Common county FIPS examples: Harris (TX)=201, Travis (TX)=453, Bexar (TX)=029, Broward (FL)=011, Hillsborough (FL)=057, Duval (FL)=031, Jefferson (LA)=051, Caddo (LA)=017, St. Tammany (LA)=103.",
        "When question text refers to 'In <County>, <State>', use STATEFP and COUNTYFP filters consistent with these FIPS codes.",
        "FIPS and GEOID fields in SVI differ in type: FIPS (in SVI) is stored as BIGINT, while GEOID is stored as VARCHAR.",
        "Always use GEOID in SVI before performing joins or applying string predicates such as LIKE '22%' to preserve leading zeros and ensure proper key matching.",
        "Avoid casting GEOID to BIGINT, as doing so will drop leading zeros and cause mismatched joins.",
        "Temporal filters (e.g., dateOfLoss < DATE '1990-01-01') are treated as independent single-table conditions, not as join keys. They map records along a time axis for temporal trend analysis.",
        "Temporal averages (e.g., average claims per year) are computed by dividing the total record count by the inclusive number of years between the minimum and maximum dateOfLoss values (MAX - MIN + 1), treating the period as a continuous span rather than discrete years.",
        "Numeric DECIMAL, BIGINT, or VARCHAR fields used in SUM, AVG, MIN, or MAX must be explicitly CAST AS DOUBLE before aggregation to ensure numeric correctness.",
        "Columns already typed as DOUBLE (e.g., PRED0_PE, PRED3_PE, RFLD_RISKS, CFLD_RISKS, PRED12_PE, LAT, LON) should not be cast again.",
        "DATE fields such as dateOfLoss should remain uncast and be handled natively for temporal comparison and range operations. Use DATE 'YYYY-MM-DD' literals for date filters.",
        "For aggregation functions (SUM, AVG, MIN, MAX) applied to nullable fields, include explicit 'IS NOT NULL' filters to avoid propagation of NULLs and ensure numeric stability.",
        "COUNT(*) is safe without NULL filtering since it counts all rows, but when measuring data completeness (e.g., missing coordinates or payouts), explicit 'IS NULL' or 'IS NOT NULL' conditions must be used.",
        "The FLD_ZONE field includes non-flood areas such as 'X' and 'AREA NOT INCLUDED'. Queries use the condition FLD_ZONE NOT IN ('X', 'AREA NOT INCLUDED') to exclude these zones from flood-related calculations."
    ],


    # ============================================================
    # 3. TRIPLE TABLE NOTES
    # ============================================================
    "triple_table_notes": [
        "Triple-table spatial-spatial queries combine three polygonal layers (e.g., floodplain, census_tracts, zcta, county) using pairwise spatial predicates. These queries are designed to test complex spatial reasoning rather than simple containment.",
        "When multiple polygons from one layer intersect a single geometry in another layer (e.g., multiple floodplain polygons overlapping one census tract), duplication may occur in area or count-based aggregations. COUNT(DISTINCT ...) or GROUP BY keys are used to mitigate this.",
        "Unless otherwise specified, intersection areas in triple-table queries are not geometrically unioned across multiple overlapping polygons. The results therefore represent cumulative intersection magnitudes rather than de-duplicated geometric areas.",
        "Each table's geometry is validated prior to joining. The benchmark assumes that all geometry columns (geometry) are stored in OGC:CRS84 coordinates and are directly comparable across layers without transformation.",
        "The floodplain layer (floodplain) serves as the primary hazard overlay, while census_tracts, zcta, and county geometries serve as administrative or contextual boundaries for multi-level spatial filtering.",
        "Triple-table queries are classified by predicate combinations: (1) spatial–spatial–spatial (e.g., Intersects + Overlaps + Touches), (2) spatial–spatial–aggregation (e.g., area or count summaries), or (3) spatial–spatial–percentage (ratio-based metrics).",
        "Each spatial join in a triple-table query is symmetric — reversing the order of operands (e.g., ST_Intersects(a,b) vs ST_Intersects(b,a)) produces identical results, except for directional functions like ST_Contains and ST_Within.",
        "All triple-table spatial joins are inner joins. Rows failing any spatial predicate in the chain are excluded from the result set. No LEFT or RIGHT spatial joins are used in the benchmark.",
        "For tract-level filtering, LEFT(t.GEOID,5)=<FIPS> ensures that only tracts belonging to the target county are included. This condition is applied after spatial predicates to minimize unnecessary join operations."
    ],

    # ============================================================
    # 4. FUNCTION NOTES
    # ============================================================
    "spatial_function_notes": [
        "All area-based spatial joins assume no overlaps or boundary gaps exist in the same table, ensuring that pairwise intersections yield non-overlapping areas without double counting.",
        "All geometries in this benchmark are first validated with ST_IsValid() and, if necessary, repaired using ST_MakeValid() before participating in any spatial operation. This ensures topological consistency and prevents execution errors such as self-intersection or ring closure issues.",
        "ST_IsValid(geometry): Returns TRUE if the geometry is topologically valid; used to pre-filter or safeguard inputs in spatial queries.",
        "ST_Intersects(geomA, geomB): Returns TRUE if two geometries share any portion of space — including overlapping interiors, containment, or touching boundaries. This is the most general predicate for detecting any form of spatial intersection.",
        "ST_Touches(geomA, geomB): Returns TRUE only when two geometries share a boundary but do not overlap or contain one another. Commonly used for adjacency analysis (e.g., neighboring tracts that share edges but have no overlapping area).",
        "ST_Overlaps(geomA, geomB): Returns TRUE when two geometries partially overlap — that is, they share some interior area but neither fully contains the other. Often used for area-based polygon-polygon relationships.",
        "ST_Contains(geomA, geomB): Returns TRUE if geometry A completely contains geometry B (B's entire area lies inside A). Typically used for polygon-point or polygon-polygon containment checks.",
        "ST_Within(geomA, geomB): Returns TRUE if geometry A is entirely within geometry B. It is the logical inverse of ST_Contains; both are treated as equivalent in this benchmark depending on argument order.",
        "ST_Point(x, y): Constructs a geometry point from coordinate pairs (longitude, latitude), converting tabular features (e.g., schools, hospitals) into spatial objects for containment testing.",
        "ST_Intersection(geomA, geomB): Produces a new geometry representing the common area or boundary shared by geomA and geomB. When combined with ST_Area, it quantifies overlap magnitude.",
        "ST_Area(geometry): Computes the two-dimensional area of a geometry in projected coordinate units (e.g., square meters). Often applied to the result of ST_Intersection for overlap analysis.",
        "SUM(...): Aggregates spatial measures (e.g., intersection areas or object counts) over grouped spatial entities.",
        "GROUP BY: Aggregates results by relational or spatial keys (e.g., GEOID, FLD_ZONE, county_name) following spatial join operations.",
        "ORDER BY ... DESC LIMIT N: Ranks spatial entities by computed metrics (e.g., area overlap, feature count) and selects the top-N results for reporting.",
        "When multiple spatial joins are chained (e.g., f JOIN t JOIN z), each join applies independently on the intermediate geometry results. The spatial predicates do not compose automatically — a record must satisfy all join predicates simultaneously to be retained.",
        "For triple-table queries, spatial predicates such as ST_Intersects, ST_Touches, or ST_Overlaps are applied pairwise. The final result represents the intersection of all true predicate conditions.",
        "When computing ST_Area(ST_Intersection(...)) in a multi-table join, overlapping polygons from one layer (e.g., multiple floodplain geometries intersecting a single tract) can cause area duplication unless explicitly aggregated or unified. Use GROUP BY on the tract identifier before aggregation if duplicate areas must be avoided.",
        "ST_Overlaps, ST_Touches, and ST_Intersects are mutually exclusive in a strict topological sense but can produce non-disjoint results when used sequentially across multiple layers. The benchmark treats them as logical filters rather than disjoint spatial categories.",
        "Spatial joins involving point-based tables (schools, hospitals) are typically independent of polygon-polygon joins; the points must satisfy all containment predicates across polygon layers to be retained.",
        "For mixed predicates (e.g., ST_Touches + ST_Intersects), the benchmark enforces conjunction (logical AND) semantics — a record must both touch one geometry and intersect another to qualify.",
        "ST_XMin(geometry) / ST_XMax(geometry): Return the minimum and maximum X coordinates (longitudes) of a geometry’s bounding box. Commonly used for longitudinal filtering — for example, ST_XMax(geom) < -82 ensures the geometry lies entirely west of longitude -82.",
        "ST_YMin(geometry) / ST_YMax(geometry): Return the minimum and maximum Y coordinates (latitudes) of a geometry’s bounding box. Useful for latitudinal filtering — for example, ST_YMin(geom) > 30 ensures the geometry lies entirely north of latitude 30.",
        "Coordinate Reference System (CRS) note for X/Y functions: The numeric values and meaning of X/Y depend on the geometry’s CRS. When performing geographic longitude/latitude comparisons, ensure that the geometries use a geographic CRS such as OGC:CRS84 or EPSG:4326, where X represents longitude and Y represents latitude. Validate with ST_IsValid() and repair with ST_MakeValid(), and apply ST_Transform() if CRS conversion is required.",
        "ST_Extent(geometry) / ST_Envelope(geometry): Compute the bounding rectangle of a geometry or a geometry collection. Often used together with XMin/XMax/YMin/YMax for boundary evaluation, spatial pre-filtering, or quick envelope-based comparisons."
    ],

    # ============================================================
    # 5. BASIC / HELPER FUNCTION NOTES
    # ============================================================
    "basic_function_notes": [
        # ============================================================
        # 1) CORE QUERY COMMANDS
        # ============================================================
        "SELECT: The core SQL command used to retrieve data. Specifies which columns, expressions, or aggregations to return from one or more tables.",
        "FROM: Defines the source table(s) for the query. Multiple tables can be combined using JOINs.",
        "WHERE: Filters individual rows before aggregation, based on logical conditions (e.g., STATEFP = '48', dateOfLoss >= DATE '2000-01-01').",
        "GROUP BY: Groups rows with identical values in specified columns into summary rows. Required for aggregate computations such as COUNT, SUM, and AVG.",
        "HAVING: Filters grouped results after aggregation (unlike WHERE, which filters raw rows). Commonly used to retain groups meeting certain thresholds (e.g., HAVING COUNT(*) > 10).",
        "ORDER BY ... DESC LIMIT N: Sorts the query results by one or more columns and returns the top N rows (e.g., top 5 counties by payout).",
        "OFFSET N: Skips the first N rows of an ordered result set (e.g., OFFSET 1 to get the second-highest value).",

        # ============================================================
        # 2) AGGREGATION & NUMERIC OPERATIONS
        # ============================================================
        "COUNT(*): Returns the total number of rows that satisfy the query conditions; frequently used for record counts or entity totals.",
        "COUNT(DISTINCT column): Counts unique values, avoiding duplicates (e.g., unique GEOIDs or unique floodplain polygons).",
        "SUM(column): Computes the total sum of a numeric column (e.g., total payouts, exposure values, or loss estimates).",
        "AVG(column): Computes the arithmetic mean of a numeric column (e.g., average risk score, mean payout).",
        "MIN(column), MAX(column): Return the smallest or largest value within a column, often used to find earliest/largest values or geographic extremes.",
        "Nested aggregation: Combines multiple levels of aggregation (e.g., outer AVG of county-level ratios). Useful for multi-scale statistical summaries.",

        # ============================================================
        # 3) CONDITIONAL & LOGICAL EXPRESSIONS
        # ============================================================
        "CASE WHEN condition THEN result [ELSE else_result] END: Implements conditional logic inside SELECT or aggregation statements (similar to if/else). Commonly used for ratio or flag computations (e.g., proportion of water-only tracts).",
        "COALESCE(a, b): Returns the first non-null value between a and b. Useful for replacing missing numeric or string fields.",

        # ============================================================
        # 4) DATA TYPE HANDLING
        # ============================================================
        "CAST(expression AS TYPE): Converts a value to a specific SQL data type. For example, CAST(amountPaidOnBuildingClaim AS DOUBLE) ensures floating-point precision for accurate numeric operations.",
        "ROUND(value, n): Rounds a numeric expression to n decimal places for presentation or normalization purposes.",

        # ============================================================
        # 5) STRING & PATTERN FUNCTIONS
        # ============================================================
        "DISTINCT: Ensures only unique values are returned in SELECT results or counted in aggregates.",
        "LEFT(column, N): Extracts the leftmost N characters from a string. Often used to derive STATEFP (2 digits) or COUNTYFP (5 digits) from GEOID.",
        "LIKE 'pattern%': Performs pattern matching for text fields (e.g., GEOID LIKE '22%' to filter Louisiana records).",
        "CONCAT(a, b): Concatenates two strings. Useful for constructing GEOIDs or composite keys.",

        # ============================================================
        # 6) DATE & TIME FUNCTIONS
        # ============================================================
        "DATE 'YYYY-MM-DD': Declares a literal date constant, typically used in WHERE filters (e.g., dateOfLoss >= DATE '2020-01-01').",
        "STRFTIME('%Y', date_column): Extracts the year portion from a date column. Used to group or rank results by year.",
        "EXTRACT(field FROM date): Extracts a specific component such as YEAR, MONTH, or DAY from a date expression.",
        "BETWEEN date1 AND date2: Filters rows with date values within an inclusive range.",

        # ============================================================
        # 7) SUBQUERIES & CTEs
        # ============================================================
        "WITH alias AS (subquery): Defines a Common Table Expression (CTE), a temporary named subquery that can be reused in the main query. Enhances readability and modular structure.",
        "Subqueries in FROM: Allows nested aggregation or filtering before the outer query (e.g., computing county-level ratios and then averaging them).",
        "Derived tables: Temporary result sets defined in parentheses and referenced by alias, often used for intermediate grouping logic.",

        # ============================================================
        # 8) SPATIAL & GEOMETRY-SAFE FILTERS
        # ============================================================
        "IS NULL / IS NOT NULL: Tests whether a field contains missing data. Often applied to geometry fields, coordinates, or payout amounts to ensure valid aggregation.",
        "geometry validity checks (conceptual): Although not used directly in key-based joins, valid geometries are required for spatial join tasks (e.g., ST_IsValid, ST_MakeValid, ST_Intersects).",
        "Key-based approximations: For simplified tasks, state-level (STATEFP) or county-level (LEFT(GEOID,5)) joins are used instead of full spatial joins.",

        # ============================================================
        # 9) GENERAL QUERY PATTERNS
        # ============================================================
        "Prefix-based filtering: GEOID LIKE '48%' to identify all Texas tracts or counties.",
        "State-level joins: Matching by STATEFP for coarse spatial associations.",
        "County-level joins: LEFT(GEOID, 5) = COUNTYFP for finer administrative joins.",
        "Temporal aggregation: Grouping by STRFTIME('%Y', dateOfLoss) to summarize annual claim frequencies.",
        "Conditional proportion computation: SUM(CASE WHEN condition THEN 1 ELSE 0 END) / COUNT(*) to derive ratios of flagged records.",
        "Top-N ranking: ORDER BY ... DESC LIMIT N to extract the most extreme values or categories.",
        "Multi-layer derived aggregation: e.g., averaging per-county water-only tract ratios, then averaging across the state."
    ]

}

# ============================================================
# 3. Helpers
# ============================================================
def safe_value(val):
    """Convert special values (bytes, NaN, timestamps, numpy types) into JSON-safe form."""
    import numpy as np
    import pandas as pd
    import math
    if isinstance(val, (bytes, bytearray)):
        return f"BLOB({len(val)} bytes)"
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    if isinstance(val, (np.generic,)):
        return val.item()
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


def normalize_table_name(fname: str) -> str:
    """Normalize parquet file name to logical table name (remove _tx_fl_la suffix)."""
    import os
    base = os.path.splitext(fname)[0]
    for suffix in ["_tx_fl_la", "_TX_FL_LA"]:
        if base.endswith(suffix):
            return base[:-len(suffix)]
    return base


def get_description(table_name, col_name):
    """Return field description if available, otherwise 'unknown'."""
    if col_name == "geom":
        col_name = "geometry"

    # --- Special case: hospitals & schools geometry derived from LAT/LON ---
    if table_name in ("hospitals", "schools") and col_name == "geometry":
        return "Geometry not stored directly; construct via ST_Point(LON, LAT) (EPSG:4326)."

    return FIELD_DESCRIPTIONS.get(table_name, {}).get(
        col_name,
        FIELD_DESCRIPTIONS.get("common", {}).get(col_name, "unknown")
    )


# ============================================================
# 4. Main Metadata Builder
# ============================================================
def main():
    import os, json, duckdb

    data_dir = "data"
    output_file = os.path.join(data_dir, "metadata_parquet.json")

    con = duckdb.connect()
    metadata = {}

    # ------------------------------------------------
    # Iterate all parquet tables
    # ------------------------------------------------
    for fname in os.listdir(data_dir):
        if not fname.endswith(".parquet"):
            continue

        table_name = normalize_table_name(fname)
        fpath = os.path.join(data_dir, fname)

        # ------------------------------------------------
        # 1. Collect schema and attach descriptions
        # ------------------------------------------------
        schema_df = con.execute(f"DESCRIBE SELECT * FROM '{fpath}'").fetchdf()
        schema = []
        for row in schema_df.to_dict(orient="records"):
            col_name = row["column_name"]
            row["description"] = get_description(table_name, col_name)

            # Mark important columns
            if col_name in ("GEOID", "STATEFP", "COUNTYFP", "COUNTYFIPS"):
                row["indexed"] = True
            elif col_name == "geometry":
                row["indexed"] = True
                row["avoid_select_star"] = True
            else:
                row["indexed"] = False

            schema.append(row)

        # ------------------------------------------------
        # 2. Count number of rows
        # ------------------------------------------------
        count = con.execute(f"SELECT COUNT(*) AS cnt FROM '{fpath}'").fetchone()[0]

        # 3. Collect sample rows (random 3 records)
        sample_df = con.execute(f"SELECT * FROM '{fpath}' USING SAMPLE 3 ROWS").fetchdf()
        samples = []
        for _, row in sample_df.iterrows():
            samples.append({col: safe_value(val) for col, val in row.items()})

        # ------------------------------------------------
        # 4. Table-level metadata (with key/spatial identifiers and _meta note)
        # ------------------------------------------------
        table_desc = FIELD_DESCRIPTIONS.get(table_name, {})
        metadata[table_name] = {
            "file": os.path.basename(fpath),
            "row_count": int(count),
            "schema": schema,
            "sample_rows": samples,
        }

        # Optional enrichment fields
        for k in ("_meta", "key_identifier", "spatial_identifier", "layer_category"):
            if k in table_desc:
                metadata[table_name][k] = table_desc[k]

        print(f"[OK] {table_name:20s} → {count:8d} rows, {len(schema):2d} cols")


    # ------------------------------------------------
    # 5. Add global metadata (relations, rules, hints)
    # ------------------------------------------------
    metadata["_global"] = GLOBAL_METADATA

    os.makedirs(data_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Enhanced metadata written to {output_file}")


if __name__ == "__main__":
    main()
