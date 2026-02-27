"""
Download and process Food Standards Agency (FSA) Food Hygiene Rating data.

Downloads establishment data from all England local authorities
and filters for eating/drinking establishments as a walkability proxy.

Output:
    - temp/fsa/fsa_establishments.gpkg (all establishments in England)
        Columns: fhrs_id, business_name, business_type, business_type_id,
                 address, postcode, latitude, longitude, rating_value,
                 rating_date, local_authority, geometry
"""

import xml.etree.ElementTree as ET
from io import BytesIO

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point
from tqdm import tqdm

# Configuration
from urban_energy.paths import CACHE_DIR as _CACHE_ROOT
from urban_energy.paths import TEMP_DIR

OUTPUT_DIR = TEMP_DIR / "fsa"
CACHE_DIR = _CACHE_ROOT / "fsa"

# FSA API configuration
FSA_BASE_URL = "https://ratings.food.gov.uk"

# Business types to include (eating/drinking establishments)
# These are the FSA BusinessType values for food service establishments
INCLUDE_BUSINESS_TYPES = {
    "Restaurant/Cafe/Canteen",
    "Pub/bar/nightclub",
    "Takeaway/sandwich shop",
    "Mobile caterer",
    "Hotel/bed & breakfast/guest house",
}

# England local authority XML files
# Excludes Wales (FHRS550-571), Scotland (FHRS760-791),
# and Northern Ireland (FHRS801-816)
AUTHORITY_FILES: dict[str, str] = {
    # East Counties
    "Babergh": "/api/open-data-files/FHRS297en-GB.xml",
    "Basildon": "/api/open-data-files/FHRS109en-GB.xml",
    "Bedford": "/api/open-data-files/FHRS701en-GB.xml",
    "Braintree": "/api/open-data-files/FHRS110en-GB.xml",
    "Breckland": "/api/open-data-files/FHRS227en-GB.xml",
    "Brentwood": "/api/open-data-files/FHRS111en-GB.xml",
    "Broadland": "/api/open-data-files/FHRS228en-GB.xml",
    "Broxbourne": "/api/open-data-files/FHRS155en-GB.xml",
    "Cambridge City": "/api/open-data-files/FHRS027en-GB.xml",
    "Castle Point": "/api/open-data-files/FHRS112en-GB.xml",
    "Central Bedfordshire": "/api/open-data-files/FHRS702en-GB.xml",
    "Chelmsford": "/api/open-data-files/FHRS113en-GB.xml",
    "Colchester": "/api/open-data-files/FHRS114en-GB.xml",
    "Dacorum": "/api/open-data-files/FHRS156en-GB.xml",
    "East Cambridgeshire": "/api/open-data-files/FHRS028en-GB.xml",
    "East Hertfordshire": "/api/open-data-files/FHRS157en-GB.xml",
    "East Suffolk": "/api/open-data-files/FHRS302en-GB.xml",
    "Epping Forest": "/api/open-data-files/FHRS115en-GB.xml",
    "Fenland": "/api/open-data-files/FHRS029en-GB.xml",
    "Great Yarmouth": "/api/open-data-files/FHRS229en-GB.xml",
    "Harlow": "/api/open-data-files/FHRS116en-GB.xml",
    "Hertsmere": "/api/open-data-files/FHRS158en-GB.xml",
    "Huntingdonshire": "/api/open-data-files/FHRS030en-GB.xml",
    "Ipswich": "/api/open-data-files/FHRS299en-GB.xml",
    "King's Lynn and West Norfolk": "/api/open-data-files/FHRS230en-GB.xml",
    "Luton": "/api/open-data-files/FHRS869en-GB.xml",
    "Maldon": "/api/open-data-files/FHRS117en-GB.xml",
    "Mid Suffolk": "/api/open-data-files/FHRS300en-GB.xml",
    "North Hertfordshire": "/api/open-data-files/FHRS159en-GB.xml",
    "North Norfolk": "/api/open-data-files/FHRS231en-GB.xml",
    "Norwich City": "/api/open-data-files/FHRS232en-GB.xml",
    "Peterborough City": "/api/open-data-files/FHRS888en-GB.xml",
    "Rochford": "/api/open-data-files/FHRS118en-GB.xml",
    "South Cambridgeshire": "/api/open-data-files/FHRS032en-GB.xml",
    "South Norfolk": "/api/open-data-files/FHRS233en-GB.xml",
    "Southend-On-Sea": "/api/open-data-files/FHRS893en-GB.xml",
    "St Albans City": "/api/open-data-files/FHRS160en-GB.xml",
    "Stevenage": "/api/open-data-files/FHRS161en-GB.xml",
    "Tendring": "/api/open-data-files/FHRS120en-GB.xml",
    "Three Rivers": "/api/open-data-files/FHRS162en-GB.xml",
    "Thurrock": "/api/open-data-files/FHRS894en-GB.xml",
    "Uttlesford": "/api/open-data-files/FHRS122en-GB.xml",
    "Watford": "/api/open-data-files/FHRS163en-GB.xml",
    "Welwyn Hatfield": "/api/open-data-files/FHRS164en-GB.xml",
    "West Suffolk": "/api/open-data-files/FHRS298en-GB.xml",
    # East Midlands
    "Amber Valley": "/api/open-data-files/FHRS062en-GB.xml",
    "Ashfield": "/api/open-data-files/FHRS259en-GB.xml",
    "Bassetlaw": "/api/open-data-files/FHRS260en-GB.xml",
    "Blaby": "/api/open-data-files/FHRS209en-GB.xml",
    "Bolsover": "/api/open-data-files/FHRS063en-GB.xml",
    "Boston": "/api/open-data-files/FHRS219en-GB.xml",
    "Broxtowe": "/api/open-data-files/FHRS261en-GB.xml",
    "Charnwood": "/api/open-data-files/FHRS210en-GB.xml",
    "Chesterfield": "/api/open-data-files/FHRS064en-GB.xml",
    "Derby City": "/api/open-data-files/FHRS871en-GB.xml",
    "Derbyshire Dales": "/api/open-data-files/FHRS070en-GB.xml",
    "East Lindsey": "/api/open-data-files/FHRS220en-GB.xml",
    "Erewash": "/api/open-data-files/FHRS066en-GB.xml",
    "Gedling": "/api/open-data-files/FHRS262en-GB.xml",
    "Harborough": "/api/open-data-files/FHRS211en-GB.xml",
    "High Peak": "/api/open-data-files/FHRS067en-GB.xml",
    "Hinckley and Bosworth": "/api/open-data-files/FHRS212en-GB.xml",
    "Leicester City": "/api/open-data-files/FHRS878en-GB.xml",
    "Lincoln City": "/api/open-data-files/FHRS221en-GB.xml",
    "Mansfield": "/api/open-data-files/FHRS263en-GB.xml",
    "Melton": "/api/open-data-files/FHRS214en-GB.xml",
    "Newark and Sherwood": "/api/open-data-files/FHRS264en-GB.xml",
    "North East Derbyshire": "/api/open-data-files/FHRS068en-GB.xml",
    "North Kesteven": "/api/open-data-files/FHRS222en-GB.xml",
    "North Northamptonshire": "/api/open-data-files/FHRS235en-GB.xml",
    "North West Leicestershire": "/api/open-data-files/FHRS215en-GB.xml",
    "Nottingham City": "/api/open-data-files/FHRS899en-GB.xml",
    "Oadby and Wigston": "/api/open-data-files/FHRS216en-GB.xml",
    "Rushcliffe": "/api/open-data-files/FHRS266en-GB.xml",
    "Rutland": "/api/open-data-files/FHRS879en-GB.xml",
    "South Derbyshire": "/api/open-data-files/FHRS069en-GB.xml",
    "South Holland": "/api/open-data-files/FHRS223en-GB.xml",
    "South Kesteven": "/api/open-data-files/FHRS224en-GB.xml",
    "West Lindsey": "/api/open-data-files/FHRS225en-GB.xml",
    "West Northamptonshire": "/api/open-data-files/FHRS236en-GB.xml",
    # London
    "Barking and Dagenham": "/api/open-data-files/FHRS501en-GB.xml",
    "Barnet": "/api/open-data-files/FHRS502en-GB.xml",
    "Bexley": "/api/open-data-files/FHRS503en-GB.xml",
    "Brent": "/api/open-data-files/FHRS504en-GB.xml",
    "Bromley": "/api/open-data-files/FHRS505en-GB.xml",
    "Camden": "/api/open-data-files/FHRS506en-GB.xml",
    "City of London Corporation": "/api/open-data-files/FHRS508en-GB.xml",
    "Croydon": "/api/open-data-files/FHRS507en-GB.xml",
    "Ealing": "/api/open-data-files/FHRS509en-GB.xml",
    "Enfield": "/api/open-data-files/FHRS510en-GB.xml",
    "Greenwich": "/api/open-data-files/FHRS511en-GB.xml",
    "Hackney": "/api/open-data-files/FHRS512en-GB.xml",
    "Hammersmith and Fulham": "/api/open-data-files/FHRS513en-GB.xml",
    "Haringey": "/api/open-data-files/FHRS514en-GB.xml",
    "Harrow": "/api/open-data-files/FHRS515en-GB.xml",
    "Havering": "/api/open-data-files/FHRS516en-GB.xml",
    "Hillingdon": "/api/open-data-files/FHRS517en-GB.xml",
    "Hounslow": "/api/open-data-files/FHRS518en-GB.xml",
    "Islington": "/api/open-data-files/FHRS519en-GB.xml",
    "Kensington and Chelsea": "/api/open-data-files/FHRS520en-GB.xml",
    "Kingston-Upon-Thames": "/api/open-data-files/FHRS521en-GB.xml",
    "Lambeth": "/api/open-data-files/FHRS522en-GB.xml",
    "Lewisham": "/api/open-data-files/FHRS523en-GB.xml",
    "Merton": "/api/open-data-files/FHRS524en-GB.xml",
    "Newham": "/api/open-data-files/FHRS525en-GB.xml",
    "Redbridge": "/api/open-data-files/FHRS526en-GB.xml",
    "Richmond-Upon-Thames": "/api/open-data-files/FHRS527en-GB.xml",
    "Southwark": "/api/open-data-files/FHRS528en-GB.xml",
    "Sutton": "/api/open-data-files/FHRS529en-GB.xml",
    "Tower Hamlets": "/api/open-data-files/FHRS530en-GB.xml",
    "Waltham Forest": "/api/open-data-files/FHRS531en-GB.xml",
    "Wandsworth": "/api/open-data-files/FHRS532en-GB.xml",
    "Westminster": "/api/open-data-files/FHRS533en-GB.xml",
    # North East
    "Darlington": "/api/open-data-files/FHRS874en-GB.xml",
    "Durham": "/api/open-data-files/FHRS706en-GB.xml",
    "Gateshead": "/api/open-data-files/FHRS410en-GB.xml",
    "Hartlepool": "/api/open-data-files/FHRS859en-GB.xml",
    "Middlesbrough": "/api/open-data-files/FHRS861en-GB.xml",
    "Newcastle Upon Tyne": "/api/open-data-files/FHRS416en-GB.xml",
    "North Tyneside": "/api/open-data-files/FHRS417en-GB.xml",
    "Northumberland": "/api/open-data-files/FHRS707en-GB.xml",
    "Redcar and Cleveland": "/api/open-data-files/FHRS860en-GB.xml",
    "South Tyneside": "/api/open-data-files/FHRS427en-GB.xml",
    "Stockton On Tees": "/api/open-data-files/FHRS862en-GB.xml",
    "Sunderland": "/api/open-data-files/FHRS429en-GB.xml",
    # North West
    "Blackburn": "/api/open-data-files/FHRS897en-GB.xml",
    "Blackpool": "/api/open-data-files/FHRS898en-GB.xml",
    "Bolton": "/api/open-data-files/FHRS403en-GB.xml",
    "Burnley": "/api/open-data-files/FHRS196en-GB.xml",
    "Bury": "/api/open-data-files/FHRS405en-GB.xml",
    "Cheshire East": "/api/open-data-files/FHRS703en-GB.xml",
    "Cheshire West and Chester": "/api/open-data-files/FHRS704en-GB.xml",
    "Chorley": "/api/open-data-files/FHRS197en-GB.xml",
    "Cumberland": "/api/open-data-files/FHRS055en-GB.xml",
    "Fylde": "/api/open-data-files/FHRS198en-GB.xml",
    "Halton": "/api/open-data-files/FHRS889en-GB.xml",
    "Hyndburn": "/api/open-data-files/FHRS199en-GB.xml",
    "Knowsley": "/api/open-data-files/FHRS412en-GB.xml",
    "Lancaster City": "/api/open-data-files/FHRS200en-GB.xml",
    "Liverpool": "/api/open-data-files/FHRS414en-GB.xml",
    "Manchester": "/api/open-data-files/FHRS415en-GB.xml",
    "Oldham": "/api/open-data-files/FHRS418en-GB.xml",
    "Pendle": "/api/open-data-files/FHRS201en-GB.xml",
    "Preston": "/api/open-data-files/FHRS202en-GB.xml",
    "Ribble Valley": "/api/open-data-files/FHRS203en-GB.xml",
    "Rochdale": "/api/open-data-files/FHRS419en-GB.xml",
    "Rossendale": "/api/open-data-files/FHRS204en-GB.xml",
    "Salford": "/api/open-data-files/FHRS422en-GB.xml",
    "Sefton": "/api/open-data-files/FHRS424en-GB.xml",
    "South Ribble": "/api/open-data-files/FHRS205en-GB.xml",
    "St Helens": "/api/open-data-files/FHRS421en-GB.xml",
    "Stockport": "/api/open-data-files/FHRS428en-GB.xml",
    "Tameside": "/api/open-data-files/FHRS430en-GB.xml",
    "Trafford": "/api/open-data-files/FHRS431en-GB.xml",
    "Warrington": "/api/open-data-files/FHRS890en-GB.xml",
    "West Lancashire": "/api/open-data-files/FHRS206en-GB.xml",
    "Westmorland and Furness": "/api/open-data-files/FHRS056en-GB.xml",
    "Wigan": "/api/open-data-files/FHRS434en-GB.xml",
    "Wirral": "/api/open-data-files/FHRS435en-GB.xml",
    "Wyre": "/api/open-data-files/FHRS207en-GB.xml",
    # South East
    "Adur": "/api/open-data-files/FHRS323en-GB.xml",
    "Arun": "/api/open-data-files/FHRS324en-GB.xml",
    "Ashford": "/api/open-data-files/FHRS179en-GB.xml",
    "Basingstoke and Deane": "/api/open-data-files/FHRS131en-GB.xml",
    "Bracknell Forest": "/api/open-data-files/FHRS882en-GB.xml",
    "Brighton and Hove": "/api/open-data-files/FHRS875en-GB.xml",
    "Buckinghamshire": "/api/open-data-files/FHRS021en-GB.xml",
    "Canterbury City": "/api/open-data-files/FHRS180en-GB.xml",
    "Cherwell": "/api/open-data-files/FHRS268en-GB.xml",
    "Chichester": "/api/open-data-files/FHRS325en-GB.xml",
    "Crawley": "/api/open-data-files/FHRS326en-GB.xml",
    "Dartford": "/api/open-data-files/FHRS181en-GB.xml",
    "Dover": "/api/open-data-files/FHRS182en-GB.xml",
    "East Hampshire": "/api/open-data-files/FHRS132en-GB.xml",
    "Eastbourne": "/api/open-data-files/FHRS102en-GB.xml",
    "Eastleigh": "/api/open-data-files/FHRS133en-GB.xml",
    "Elmbridge": "/api/open-data-files/FHRS305en-GB.xml",
    "Epsom and Ewell": "/api/open-data-files/FHRS306en-GB.xml",
    "Fareham": "/api/open-data-files/FHRS134en-GB.xml",
    "Folkestone and Hythe": "/api/open-data-files/FHRS188en-GB.xml",
    "Gosport": "/api/open-data-files/FHRS135en-GB.xml",
    "Gravesham": "/api/open-data-files/FHRS184en-GB.xml",
    "Guildford": "/api/open-data-files/FHRS307en-GB.xml",
    "Hart": "/api/open-data-files/FHRS136en-GB.xml",
    "Hastings": "/api/open-data-files/FHRS103en-GB.xml",
    "Havant": "/api/open-data-files/FHRS137en-GB.xml",
    "Horsham": "/api/open-data-files/FHRS327en-GB.xml",
    "Isle of Wight": "/api/open-data-files/FHRS867en-GB.xml",
    "Lewes": "/api/open-data-files/FHRS105en-GB.xml",
    "Maidstone": "/api/open-data-files/FHRS185en-GB.xml",
    "Medway": "/api/open-data-files/FHRS896en-GB.xml",
    "Mid Sussex": "/api/open-data-files/FHRS328en-GB.xml",
    "Milton Keynes": "/api/open-data-files/FHRS870en-GB.xml",
    "Mole Valley": "/api/open-data-files/FHRS308en-GB.xml",
    "New Forest": "/api/open-data-files/FHRS138en-GB.xml",
    "Oxford City": "/api/open-data-files/FHRS269en-GB.xml",
    "Portsmouth": "/api/open-data-files/FHRS876en-GB.xml",
    "Reading": "/api/open-data-files/FHRS884en-GB.xml",
    "Reigate and Banstead": "/api/open-data-files/FHRS309en-GB.xml",
    "Rother": "/api/open-data-files/FHRS106en-GB.xml",
    "Runnymede": "/api/open-data-files/FHRS310en-GB.xml",
    "Rushmoor": "/api/open-data-files/FHRS140en-GB.xml",
    "Sevenoaks": "/api/open-data-files/FHRS187en-GB.xml",
    "Slough": "/api/open-data-files/FHRS885en-GB.xml",
    "South Oxfordshire": "/api/open-data-files/FHRS270en-GB.xml",
    "Southampton": "/api/open-data-files/FHRS877en-GB.xml",
    "Spelthorne": "/api/open-data-files/FHRS311en-GB.xml",
    "Surrey Heath": "/api/open-data-files/FHRS312en-GB.xml",
    "Swale": "/api/open-data-files/FHRS189en-GB.xml",
    "Tandridge": "/api/open-data-files/FHRS313en-GB.xml",
    "Test Valley": "/api/open-data-files/FHRS142en-GB.xml",
    "Thanet": "/api/open-data-files/FHRS190en-GB.xml",
    "Tonbridge and Malling": "/api/open-data-files/FHRS191en-GB.xml",
    "Tunbridge Wells": "/api/open-data-files/FHRS192en-GB.xml",
    "Vale of White Horse": "/api/open-data-files/FHRS271en-GB.xml",
    "Waverley": "/api/open-data-files/FHRS314en-GB.xml",
    "Wealden": "/api/open-data-files/FHRS107en-GB.xml",
    "West Berkshire": "/api/open-data-files/FHRS883en-GB.xml",
    "West Oxfordshire": "/api/open-data-files/FHRS272en-GB.xml",
    "Winchester City": "/api/open-data-files/FHRS143en-GB.xml",
    "Windsor and Maidenhead": "/api/open-data-files/FHRS886en-GB.xml",
    "Woking": "/api/open-data-files/FHRS315en-GB.xml",
    "Wokingham": "/api/open-data-files/FHRS887en-GB.xml",
    "Worthing": "/api/open-data-files/FHRS329en-GB.xml",
    # South West
    "Bath and North East Somerset": "/api/open-data-files/FHRS857en-GB.xml",
    "Bournemouth, Christchurch and Poole": "/api/open-data-files/FHRS872en-GB.xml",
    "Bristol": "/api/open-data-files/FHRS855en-GB.xml",
    "Cheltenham": "/api/open-data-files/FHRS124en-GB.xml",
    "Cornwall": "/api/open-data-files/FHRS705en-GB.xml",
    "Cotswold": "/api/open-data-files/FHRS125en-GB.xml",
    "Dorset": "/api/open-data-files/FHRS085en-GB.xml",
    "East Devon": "/api/open-data-files/FHRS072en-GB.xml",
    "Exeter City": "/api/open-data-files/FHRS073en-GB.xml",
    "Forest of Dean": "/api/open-data-files/FHRS126en-GB.xml",
    "Gloucester City": "/api/open-data-files/FHRS127en-GB.xml",
    "Isles of Scilly": "/api/open-data-files/FHRS336en-GB.xml",
    "Mid Devon": "/api/open-data-files/FHRS074en-GB.xml",
    "North Devon": "/api/open-data-files/FHRS075en-GB.xml",
    "North Somerset": "/api/open-data-files/FHRS858en-GB.xml",
    "Plymouth City": "/api/open-data-files/FHRS891en-GB.xml",
    "Somerset": "/api/open-data-files/FHRS281en-GB.xml",
    "South Gloucestershire": "/api/open-data-files/FHRS856en-GB.xml",
    "South Hams": "/api/open-data-files/FHRS077en-GB.xml",
    "Stroud": "/api/open-data-files/FHRS128en-GB.xml",
    "Swindon": "/api/open-data-files/FHRS881en-GB.xml",
    "Teignbridge": "/api/open-data-files/FHRS078en-GB.xml",
    "Tewkesbury": "/api/open-data-files/FHRS129en-GB.xml",
    "Torbay": "/api/open-data-files/FHRS892en-GB.xml",
    "Torridge": "/api/open-data-files/FHRS080en-GB.xml",
    "West Devon": "/api/open-data-files/FHRS081en-GB.xml",
    "Wiltshire": "/api/open-data-files/FHRS709en-GB.xml",
    # West Midlands
    "Birmingham": "/api/open-data-files/FHRS402en-GB.xml",
    "Bromsgrove": "/api/open-data-files/FHRS145en-GB.xml",
    "Cannock Chase": "/api/open-data-files/FHRS287en-GB.xml",
    "Coventry": "/api/open-data-files/FHRS407en-GB.xml",
    "Dudley": "/api/open-data-files/FHRS409en-GB.xml",
    "East Staffordshire": "/api/open-data-files/FHRS288en-GB.xml",
    "Herefordshire": "/api/open-data-files/FHRS895en-GB.xml",
    "Lichfield": "/api/open-data-files/FHRS289en-GB.xml",
    "Malvern Hills": "/api/open-data-files/FHRS148en-GB.xml",
    "Newcastle-Under-Lyme": "/api/open-data-files/FHRS290en-GB.xml",
    "North Warwickshire": "/api/open-data-files/FHRS317en-GB.xml",
    "Nuneaton and Bedworth": "/api/open-data-files/FHRS318en-GB.xml",
    "Redditch": "/api/open-data-files/FHRS149en-GB.xml",
    "Rugby": "/api/open-data-files/FHRS319en-GB.xml",
    "Sandwell": "/api/open-data-files/FHRS423en-GB.xml",
    "Shropshire": "/api/open-data-files/FHRS708en-GB.xml",
    "Solihull": "/api/open-data-files/FHRS426en-GB.xml",
    "South Staffordshire": "/api/open-data-files/FHRS291en-GB.xml",
    "Stafford": "/api/open-data-files/FHRS292en-GB.xml",
    "Staffordshire Moorlands": "/api/open-data-files/FHRS293en-GB.xml",
    "Stoke-On-Trent": "/api/open-data-files/FHRS880en-GB.xml",
    "Stratford-on-Avon": "/api/open-data-files/FHRS320en-GB.xml",
    "Tamworth": "/api/open-data-files/FHRS295en-GB.xml",
    "Telford and Wrekin Council": "/api/open-data-files/FHRS900en-GB.xml",
    "Walsall": "/api/open-data-files/FHRS433en-GB.xml",
    "Warwick": "/api/open-data-files/FHRS321en-GB.xml",
    "Wolverhampton": "/api/open-data-files/FHRS436en-GB.xml",
    "Worcester City": "/api/open-data-files/FHRS151en-GB.xml",
    "Wychavon": "/api/open-data-files/FHRS152en-GB.xml",
    "Wyre Forest": "/api/open-data-files/FHRS153en-GB.xml",
    # Yorkshire and Humberside
    "Barnsley": "/api/open-data-files/FHRS401en-GB.xml",
    "Bradford": "/api/open-data-files/FHRS404en-GB.xml",
    "Calderdale": "/api/open-data-files/FHRS406en-GB.xml",
    "Doncaster": "/api/open-data-files/FHRS408en-GB.xml",
    "East Riding of Yorkshire": "/api/open-data-files/FHRS863en-GB.xml",
    "Hull City": "/api/open-data-files/FHRS866en-GB.xml",
    "Kirklees": "/api/open-data-files/FHRS411en-GB.xml",
    "Leeds": "/api/open-data-files/FHRS413en-GB.xml",
    "North East Lincolnshire": "/api/open-data-files/FHRS865en-GB.xml",
    "North Lincolnshire": "/api/open-data-files/FHRS864en-GB.xml",
    "North Yorkshire": "/api/open-data-files/FHRS250en-GB.xml",
    "Rotherham": "/api/open-data-files/FHRS420en-GB.xml",
    "Sheffield": "/api/open-data-files/FHRS425en-GB.xml",
    "Wakefield": "/api/open-data-files/FHRS432en-GB.xml",
    "York": "/api/open-data-files/FHRS868en-GB.xml",
}


def download_xml(url: str, authority_name: str) -> bytes | None:
    """
    Download XML file from FSA API.

    Parameters
    ----------
    url : str
        Full URL to the XML file.
    authority_name : str
        Name of the local authority (for cache filename).

    Returns
    -------
    bytes | None
        XML content as bytes, or None if download fails.
    """
    # Check cache first
    cache_file = CACHE_DIR / f"{authority_name.replace(' ', '_')}.xml"
    if cache_file.exists():
        return cache_file.read_bytes()

    headers = {"User-Agent": "urban-energy-research/1.0"}
    try:
        response = requests.get(url, timeout=60, headers=headers)
        response.raise_for_status()
        # Cache the response
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(response.content)
        return response.content
    except requests.RequestException as e:
        print(f"  Warning: Failed to download {authority_name}: {e}")
        return None


def parse_establishments(xml_content: bytes) -> list[dict]:
    """
    Parse FSA XML to extract establishment records.

    Parameters
    ----------
    xml_content : bytes
        Raw XML content from FSA API.

    Returns
    -------
    list[dict]
        List of establishment dictionaries with relevant fields.
    """
    establishments = []
    try:
        root = ET.parse(BytesIO(xml_content)).getroot()
    except ET.ParseError as e:
        print(f"  Warning: XML parse error: {e}")
        return establishments

    for est in root.findall(".//EstablishmentDetail"):
        business_type = est.findtext("BusinessType", "")

        # Filter by business type
        if business_type not in INCLUDE_BUSINESS_TYPES:
            continue

        # Extract coordinates
        lat_text = est.findtext("Geocode/Latitude", "")
        lon_text = est.findtext("Geocode/Longitude", "")

        # Skip if no coordinates
        if not lat_text or not lon_text:
            continue

        try:
            lat = float(lat_text)
            lon = float(lon_text)
        except ValueError:
            continue

        # Skip invalid coordinates
        if lat == 0 or lon == 0:
            continue

        # Build address string
        address_parts = []
        for i in range(1, 5):
            line = est.findtext(f"AddressLine{i}", "").strip()
            if line:
                address_parts.append(line)
        address = ", ".join(address_parts)

        establishments.append(
            {
                "fhrs_id": est.findtext("FHRSID", ""),
                "business_name": est.findtext("BusinessName", ""),
                "business_type": business_type,
                "business_type_id": est.findtext("BusinessTypeID", ""),
                "address": address,
                "postcode": est.findtext("PostCode", ""),
                "latitude": lat,
                "longitude": lon,
                "rating_value": est.findtext("RatingValue", ""),
                "rating_date": est.findtext("RatingDate", ""),
                "local_authority": est.findtext("LocalAuthorityName", ""),
            }
        )

    return establishments


def download_all_establishments() -> pd.DataFrame:
    """
    Download and parse establishments from all local authorities.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame of all establishments.
    """
    all_establishments = []

    print(f"Downloading FSA data from {len(AUTHORITY_FILES)} local authorities...")

    for authority_name, xml_path in tqdm(
        AUTHORITY_FILES.items(), desc="Downloading authorities"
    ):
        url = f"{FSA_BASE_URL}{xml_path}"
        xml_content = download_xml(url, authority_name)

        if xml_content is None:
            continue

        establishments = parse_establishments(xml_content)
        all_establishments.extend(establishments)

    print(f"Downloaded {len(all_establishments):,} eating/drinking establishments")
    return pd.DataFrame(all_establishments)


def create_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Convert DataFrame to GeoDataFrame with point geometries.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with latitude and longitude columns.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame in EPSG:27700 (British National Grid).
    """
    # Create point geometries from WGS84 coordinates
    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]

    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Transform to British National Grid
    gdf = gdf.to_crs(epsg=27700)

    return gdf


def main() -> None:
    """Main download and processing pipeline."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Download all FSA data
    df = download_all_establishments()

    if df.empty:
        print("No establishments downloaded. Check network connection.")
        return

    # Convert to GeoDataFrame
    print("Creating spatial data...")
    gdf = create_geodataframe(df)
    print(f"Created {len(gdf):,} establishment points")

    # Summary by business type
    type_counts = gdf.groupby("business_type").size().sort_values(ascending=False)
    print("\nEstablishments by type:")
    for btype, count in type_counts.items():
        print(f"  {btype}: {count:,}")

    # Save to GeoPackage
    output_path = OUTPUT_DIR / "fsa_establishments.gpkg"
    print(f"\nSaving to {output_path}...")
    gdf.to_file(output_path, driver="GPKG")
    print("Done.")


if __name__ == "__main__":
    main()
