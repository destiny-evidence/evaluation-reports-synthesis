from scripts.utils import clean_text, parse_md_sections, detect_language
import os
from pathlib import Path
import re
from langdetect import detect, detect_langs, DetectorFactory
from deep_translator import GoogleTranslator
from deep_translator.exceptions import RequestError, TooManyRequests
import pandas as pd # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
from chromadb import PersistentClient # type: ignore
from chromadb.utils import embedding_functions # type: ignore
import numpy as np # type: ignore
from sklearn.cluster import KMeans, MiniBatchKMeans # type: ignore
import matplotlib.pyplot as plt # type: ignore
import logging
from rapidfuzz import fuzz
import json


health_terms = ['health', '"well being"', 'wellbeing', '"quality of life"', 'qol', 'hrqol', 'mortalit', 'fatalit', 'disabilit', 'disorder', 'daly', 'morbidit', 'mental', 'death', 'disease', 'illness', 'injur', 'accident', 'wound', 'medical', 'medicine', 'clinic', 'hospital', 'emergenc', '"intensive care"', 'ICU', 'doctor', 'physician', 'surgery', 'surgical', 'surgeon', 'nurs', '"primary care"', '"secondary care"', '"tertiary care"', 'healthcare', 'comorbidit', 'burns', 'syndrome', 'infect', 'diet', '"food secur"', '"food insecur"', '"food shortage"', 'nutrition', 'underweight', 'malnutrition', 'undernutrition', 'undernourish', '"water quality"', '"water availability"', '"water secur"', '"water shortage"', '"water supply"', '"water stress"', 'diarrhoea', '"thermal comfort"', '"hazard incidence"', '"hazard severity"', 'pollut', 'contamina', 'wildfire', '"wild fire"', 'bushfire', '"bush fire"', '"forest fire"', '"urban temperature"', '"temperature regulation"', 'heat', 'cooling', 'vulnerable', 'vulnerability', 'drought', 'floods', 'flooding', 'storm', 'landslide', 'mudslide'
]

nbs_terms = ['afforestation', '"agro pastoral"', 'agroforest', '"agroecological farming"', '"agroecological management"', '"agroecological practices"', 'agroecology', '"agroecological restoration"', '"agroforestry system"', '"agro silvopastoral"', 'agropastoral', '"algal restoration"', '"alley cropping"', '"aquifer recharge"', '"assisted forest regeneration"', '"assisted natural regeneration"', '"bank stabilisation"', '"bank stabilization"', '"barrier removal"', '"beaver based restoration"', '"beaver reintroduction"', 'biochar', '"biological soil crust"', 'biocorridor', 'bioengineering', 'biocrust', '"bioretention system"', 'bioswale', '"blue infrastructure"', '"blue space"', '"blue green"', '"blue grey"', '"blue gray"', '"blue carbon restoration"', '"blue carbon conservation"', '"blue carbon ecosystem restoration"', '"buffer strip"', '"buffer zone"', '"canopy expansion"', '"carbon farming"', '"catchment protection"', '"catchment restoration"', '"coastal defence"', '"coastal defense"', '"coastal ecosystem restoration"', '"coastal habitat restoration"', '"coastal resilience"', '"coastal resilience solution"', '"coastal restoration"', '"coastal stabili"', '"coastal wetland"', '"coastal zone management"', '"communal conservanc"', '"community based forest management"', '"community based natural resource management"', '"community forest"', '"conservation agriculture"', '"conservation farming"', '"conservation grazing"', '"conservation tillage"', '"constructed wetland"', '"contour farming"', '"controlled burning"', '"coral reef restoration"', '"corridor restoration"', '"cover crop"', '"cover cropping"', '"crop diversification"', '"cross slope vegetation"', '"dam decommissioning"', '"dam removal"', 'deculverting', '"deforestation control"', '"deforestation reduction"', '"demand side water management"', '"detention pond"', '"dryland restoration"', '"dune protection"', '"dune restoration"', '"dune stabili"', '"eco DRR"', '"ecological connectivity"', '"ecological corridor"', '"ecological engineering"', '"ecological flood management"', '"ecological flow restoration"', '"ecological infrastructure"', '"ecological restoration"', '"ecosystem adaptation"', '"ecosystem based adaptation"', '"ecosystem based approach"', '"ecosystem based climate adaptation"', '"ecosystem based climate change adaptation"', '"ecosystem based coastal adaptation"', '"ecosystem based coastal management"', '"ecosystem based disaster risk reduction"', '"ecosystem based fisheries management"', '"ecosystem based flood management"', '"ecosystem based infrastructure"', '"ecosystem based intervention"', '"ecosystem based management"', '"ecosystem based measure"', '"ecosystem based mitigation"', '"ecosystem based resilience"', '"ecosystem based risk reduction"', '"ecosystem based solution"', '"ecosystem based water management"', '"ecosystem enhancement"', '"ecosystem restoration"', '"environmental flow"', '"environmental service payments"', '"ecosystem service payments"', '"engineering with nature"', '"field margin"', '"fish passage restoration"', 'fishway', '"floodplain rehabilitation"', '"floodplain reconnection"', '"floodplain restoration"', '"forest conservation"', '"forest ecosystem restoration"', '"forest regeneration"', '"forest rehabilitation"', '"forest restoration"', '"forest thinning"', '"green adaptation"', '"green belt"', '"green corridor"', '"green facade"', '"green infrastructure"', '"green roof"', '"green space"', '"green street"', '"green wall"', 'greenway', '"greenway corridor"', '"green grey"', '"green gray"', '"habitat banking"', '"habitat connectivity"', '"habitat creation"', '"habitat enhancement"', '"habitat restoration"', '"headwater restoration"', 'hedgerow', '"hedgerow planting"', '"hybrid green infrastructure"', '"hybrid infrastructure"', '"hydrological restoration"', 'intercropping', '"integrated watershed management"', '"intertidal restoration"', '"kelp restoration"', '"landscape connectivity"', '"landscape restoration"', '"landscape scale connectivity"', '"landscape scale restoration"', '"levee vegetation"', '"living breakwater"', '"living coast"', '"living dam"', '"living dike"', '"living dyke"', '"living infrastructure"', '"living roof"', '"living shoreline"', '"living wall"', '"mangrove conservation"', '"mangrove restoration"', '"marine restoration"', '"marine ecosystem restoration"', '"marine habitat restoration"', '"marsh restoration"', '"meadow restoration"', '"mixed cropping"', '"multifunctional landscape"', '"natural bank stabilisation"', '"natural bank stabilization"', '"natural climate solution"', '"natural flood management"', '"natural flood retention"', '"natural flood risk management"', '"natural water retention measure"', '"nature based adaptation"', '"nature based approach"', '"nature based coastal defence"', '"nature based coastal defense"', '"nature based coastal management"', '"nature based coastal protection"', '"nature based erosion control"', '"nature based flood management"', '"nature based flood risk management"', '"nature based flood risk reduction"', '"nature based intervention"', '"nature based infrastructure"', '"nature based solution"', '"nature based measure"', '"nature based mitigation"', '"nature based resilience"', '"nature based risk reduction"', '"nature based stormwater management"', '"nature based water management"', '"ocean based climate solution"', '"organic soil management"', '"oyster reef"', '"oyster reef restoration"', '"payment for ecosystem services"', '"payments for ecosystem services"', '"payment for environmental services"', '"payments for environmental services"', '"peatland conservation"', '"peatland rehabilitation"', '"peatland restoration"', '"peatland rewetting"', '"permeable landscape"', '"permeable pavement"', 'phytoremediation', '"pollinator habitat creation"', '"protected area"', '"rain garden"', '"rangeland restoration"', '"re-vegetat"', '"reconnection of rivers"', '"reducing deforestation"', '"reduced deforestation"', '"regenerative agriculture"', '"regenerative farming"', '"regenerative land management"', 'reforestation', '"reef rehabilitation"', '"reef restoration"', '"re-naturalisation"', 'renaturalisation', '"re-naturalization"', 'renaturalization', '"restoration ecology"', '"restoration intervention"', 'revegetat', 'rewilding', '"retention pond"', '"riparian buffer"', '"riparian buffer restoration"', '"riparian buffer zone"', '"riparian conservation"', '"riparian regeneration"', '"riparian restoration"', '"riparian revegetation"', '"riparian zone restoration"', '"riparian zone protection"', '"river daylighting"', '"river rehabilitation"', '"river re-naturalisation"', '"river renaturation"', '"river restoration"', '"river remeandering"', '"room for the river"', '"rotational grazing"', '"salt marsh"', '"salt marsh restoration"', '"saltmarsh restoration"', '"seagrass meadow"', '"seagrass meadow restoration"', '"seagrass restoration"', '"shellfish reef"', 'silvoagriculture', 'silvopastoral', 'silvopasture', '"slope stabili"', '"soil carbon management"', '"soil carbon sequestration"', '"soil health improvement"', '"soil restoration"', '"soil stabili"', '"sponge city"', '"sponge cities"', '"stormwater management"', '"stormwater retention"', '"stream daylighting"', '"stream rehabilitation"', '"stream restoration"', '"street tree"', '"sustainable drainage system"', '"sustainable forest management"', '"sustainable grazing"', 'terracing', '"tree based solution"', '"tree canopy"', '"tree planting"', '"urban agriculture"', '"urban afforestation"', '"urban agroforestry"', '"urban biodiversity enhancement"', '"urban blue infrastructure"', '"urban ecosystem restoration"', '"urban forest restoration"', '"urban forest"', '"urban green infrastructure"', '"urban green"', '"urban greening"', '"urban natural infrastructure"', '"urban nature restoration"', '"urban park"', '"urban river restoration"', '"urban tree planting"', '"urban vegetation"', '"urban wetland restoration"', '"urban wetland"', '"urban woodland"', '"vegetated embankment"', '"vegetated roof"', '"vegetated swale"', '"vegetation based erosion control"', '"vegetation restoration"', '"vegetation thinning"', '"vertical green"', '"vertical garden"', '"water sensitive urban design"', '"watershed restoration"', '"wetland conservation"', '"wetland creation"', '"wetland enhancement"', '"wetland rehabilitation"', '"wetland re-creation"', '"wetland reestablishment"', '"wetland restoration"', '"wildlife corridor"', '"wildflower meadow creation"', '"woodland creation"', '"woodland restoration"', '"working with natural processes"'
]

if __name__ == "__main__":

    DetectorFactory.seed = 0  # make langdetect deterministic
    FOLDER = "/home/lt1412/Desktop/evaluation-reports-synthesis/data/"
    detect_language(folder=FOLDER)
    #FILERESULT = "/home/lt1412/Desktop/evaluation-reports-synthesis/search_results/search_result_20260824IC6_query.json"
    #FILEDATABASE = "/home/lt1412/Desktop/UNEG-repository/data/member_publications_afterdownload.csv"
    #SEARCHTITLE = "20260824IC6_query"

    
    #FILELANGUAGES = detect_language(folder=FOLDER)
    #QUERY = build_and_query([nbs_terms, health_terms])
    #print(f"Running AND query: {QUERY}\n")
    #RESULT = boolean_search_markdown_section(folder=FOLDER, query=QUERY, file_results="search_results/reportlanguage.json")
    #print(f"User query: {RESULT["query"]}")
    #print(f"Total files matching query (inside section search): {RESULT["matched_doc_count"]}")
    #print(f"Total sections matching query: {RESULT["total_section_matches"]}")
    #with open(f'search_result_{SEARCHTITLE}.json', 'w') as f:
    #    json.dump(RESULT,f)

    #match_search_results(fileresults=FILERESULT, filedatabase=FILEDATABASE, search_title=SEARCHTITLE)

    #RESULTS = []
    # for term in nbs_terms + climate_risk_terms + health_terms:
    #     RESULT = search_markdown_folder_freq(folder=FOLDER, keyword=term)
    #     with open(f'search_result_{term}.json', 'w') as f:
    #         json.dump(RESULT,f)
    #     #RESULTS.append(RESULT)
    #     visualise_file_frequencies(result=RESULT)

    # for term_list in [nbs_terms, climate_risk_terms, health_terms]:
    #     QUERY = build_or_query(term_list)
    #     print(f"Running OR query: {QUERY}\n")
    #     RESULT = boolean_search_markdown_section(folder=FOLDER, query=QUERY)
    #     print(f"User query: {RESULT["query"]}")
    #     print(f"Total files matching query (inside section search): {RESULT["matched_doc_count"]}")
    #     print(f"Total sections matching query: {RESULT["total_section_matches"]}")
    
    # KEYWORD = "executive summary"
    # result = search_section_titles(folder=FOLDER, keyword=KEYWORD)
    # print(f"{result['total_files']} total files searched,  {result['files_with_exact_matches']} files exactly matching  {KEYWORD}")
    # print(f"{result['total_files']} total files searched,  {result['files_with_fuzzy_matches']} files fuzzily matching  {KEYWORD}")



