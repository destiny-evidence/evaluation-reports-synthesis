from scripts.translate_ollama import translate_nonen_mds
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
from collections import defaultdict


health_terms = ['health', '"well being"', 'wellbeing', '"quality of life"', 'qol', 'hrqol', 'mortalit', 'fatalit', 'disabilit', 'disorder', 'daly', 'morbidit', 'mental', 'death', 'disease', 'illness', 'injur', 'accident', 'wound', 'medical', 'medicine', 'clinic', 'hospital', 'emergenc', '"intensive care"', 'ICU', 'doctor', 'physician', 'surgery', 'surgical', 'surgeon', 'nurs', '"primary care"', '"secondary care"', '"tertiary care"', 'healthcare', 'comorbidit', 'burns', 'syndrome', 'infect', 'diet', '"food secur"', '"food insecur"', '"food shortage"', 'nutrition', 'underweight', 'malnutrition', 'undernutrition', 'undernourish', '"water quality"', '"water availability"', '"water secur"', '"water shortage"', '"water supply"', '"water stress"', 'diarrhoea', '"thermal comfort"', '"hazard incidence"', '"hazard severity"', 'pollut', 'contamina', 'wildfire', '"wild fire"', 'bushfire', '"bush fire"', '"forest fire"', '"urban temperature"', '"temperature regulation"', 'heat', 'cooling', 'vulnerable', 'vulnerability', 'drought', 'floods', 'flooding', 'storm', 'landslide', 'mudslide'
]

nbs_terms = ['afforestation', '"agro pastoral"', 'agroforest', '"agroecological farming"', '"agroecological management"', '"agroecological practices"', 'agroecology', '"agroecological restoration"', '"agroforestry system"', '"agro silvopastoral"', 'agropastoral', '"algal restoration"', '"alley cropping"', '"aquifer recharge"', '"assisted forest regeneration"', '"assisted natural regeneration"', '"bank stabilisation"', '"bank stabilization"', '"barrier removal"', '"beaver based restoration"', '"beaver reintroduction"', 'biochar', '"biological soil crust"', 'biocorridor', 'bioengineering', 'biocrust', '"bioretention system"', 'bioswale', '"blue infrastructure"', '"blue space"', '"blue green"', '"blue grey"', '"blue gray"', '"blue carbon restoration"', '"blue carbon conservation"', '"blue carbon ecosystem restoration"', '"buffer strip"', '"buffer zone"', '"canopy expansion"', '"carbon farming"', '"catchment protection"', '"catchment restoration"', '"coastal defence"', '"coastal defense"', '"coastal ecosystem restoration"', '"coastal habitat restoration"', '"coastal resilience"', '"coastal resilience solution"', '"coastal restoration"', '"coastal stabili"', '"coastal wetland"', '"coastal zone management"', '"communal conservanc"', '"community based forest management"', '"community based natural resource management"', '"community forest"', '"conservation agriculture"', '"conservation farming"', '"conservation grazing"', '"conservation tillage"', '"constructed wetland"', '"contour farming"', '"controlled burning"', '"coral reef restoration"', '"corridor restoration"', '"cover crop"', '"cover cropping"', '"crop diversification"', '"cross slope vegetation"', '"dam decommissioning"', '"dam removal"', 'deculverting', '"deforestation control"', '"deforestation reduction"', '"demand side water management"', '"detention pond"', '"dryland restoration"', '"dune protection"', '"dune restoration"', '"dune stabili"', '"eco DRR"', '"ecological connectivity"', '"ecological corridor"', '"ecological engineering"', '"ecological flood management"', '"ecological flow restoration"', '"ecological infrastructure"', '"ecological restoration"', '"ecosystem adaptation"', '"ecosystem based adaptation"', '"ecosystem based approach"', '"ecosystem based climate adaptation"', '"ecosystem based climate change adaptation"', '"ecosystem based coastal adaptation"', '"ecosystem based coastal management"', '"ecosystem based disaster risk reduction"', '"ecosystem based fisheries management"', '"ecosystem based flood management"', '"ecosystem based infrastructure"', '"ecosystem based intervention"', '"ecosystem based management"', '"ecosystem based measure"', '"ecosystem based mitigation"', '"ecosystem based resilience"', '"ecosystem based risk reduction"', '"ecosystem based solution"', '"ecosystem based water management"', '"ecosystem enhancement"', '"ecosystem restoration"', '"environmental flow"', '"environmental service payments"', '"ecosystem service payments"', '"engineering with nature"', '"field margin"', '"fish passage restoration"', 'fishway', '"floodplain rehabilitation"', '"floodplain reconnection"', '"floodplain restoration"', '"forest conservation"', '"forest ecosystem restoration"', '"forest regeneration"', '"forest rehabilitation"', '"forest restoration"', '"forest thinning"', '"green adaptation"', '"green belt"', '"green corridor"', '"green facade"', '"green infrastructure"', '"green roof"', '"green space"', '"green street"', '"green wall"', 'greenway', '"greenway corridor"', '"green grey"', '"green gray"', '"habitat banking"', '"habitat connectivity"', '"habitat creation"', '"habitat enhancement"', '"habitat restoration"', '"headwater restoration"', 'hedgerow', '"hedgerow planting"', '"hybrid green infrastructure"', '"hybrid infrastructure"', '"hydrological restoration"', 'intercropping', '"integrated watershed management"', '"intertidal restoration"', '"kelp restoration"', '"landscape connectivity"', '"landscape restoration"', '"landscape scale connectivity"', '"landscape scale restoration"', '"levee vegetation"', '"living breakwater"', '"living coast"', '"living dam"', '"living dike"', '"living dyke"', '"living infrastructure"', '"living roof"', '"living shoreline"', '"living wall"', '"mangrove conservation"', '"mangrove restoration"', '"marine restoration"', '"marine ecosystem restoration"', '"marine habitat restoration"', '"marsh restoration"', '"meadow restoration"', '"mixed cropping"', '"multifunctional landscape"', '"natural bank stabilisation"', '"natural bank stabilization"', '"natural climate solution"', '"natural flood management"', '"natural flood retention"', '"natural flood risk management"', '"natural water retention measure"', '"nature based adaptation"', '"nature based approach"', '"nature based coastal defence"', '"nature based coastal defense"', '"nature based coastal management"', '"nature based coastal protection"', '"nature based erosion control"', '"nature based flood management"', '"nature based flood risk management"', '"nature based flood risk reduction"', '"nature based intervention"', '"nature based infrastructure"', '"nature based solution"', '"nature based measure"', '"nature based mitigation"', '"nature based resilience"', '"nature based risk reduction"', '"nature based stormwater management"', '"nature based water management"', '"ocean based climate solution"', '"organic soil management"', '"oyster reef"', '"oyster reef restoration"', '"payment for ecosystem services"', '"payments for ecosystem services"', '"payment for environmental services"', '"payments for environmental services"', '"peatland conservation"', '"peatland rehabilitation"', '"peatland restoration"', '"peatland rewetting"', '"permeable landscape"', '"permeable pavement"', 'phytoremediation', '"pollinator habitat creation"', '"protected area"', '"rain garden"', '"rangeland restoration"', '"re-vegetat"', '"reconnection of rivers"', '"reducing deforestation"', '"reduced deforestation"', '"regenerative agriculture"', '"regenerative farming"', '"regenerative land management"', 'reforestation', '"reef rehabilitation"', '"reef restoration"', '"re-naturalisation"', 'renaturalisation', '"re-naturalization"', 'renaturalization', '"restoration ecology"', '"restoration intervention"', 'revegetat', 'rewilding', '"retention pond"', '"riparian buffer"', '"riparian buffer restoration"', '"riparian buffer zone"', '"riparian conservation"', '"riparian regeneration"', '"riparian restoration"', '"riparian revegetation"', '"riparian zone restoration"', '"riparian zone protection"', '"river daylighting"', '"river rehabilitation"', '"river re-naturalisation"', '"river renaturation"', '"river restoration"', '"river remeandering"', '"room for the river"', '"rotational grazing"', '"salt marsh"', '"salt marsh restoration"', '"saltmarsh restoration"', '"seagrass meadow"', '"seagrass meadow restoration"', '"seagrass restoration"', '"shellfish reef"', 'silvoagriculture', 'silvopastoral', 'silvopasture', '"slope stabili"', '"soil carbon management"', '"soil carbon sequestration"', '"soil health improvement"', '"soil restoration"', '"soil stabili"', '"sponge city"', '"sponge cities"', '"stormwater management"', '"stormwater retention"', '"stream daylighting"', '"stream rehabilitation"', '"stream restoration"', '"street tree"', '"sustainable drainage system"', '"sustainable forest management"', '"sustainable grazing"', 'terracing', '"tree based solution"', '"tree canopy"', '"tree planting"', '"urban agriculture"', '"urban afforestation"', '"urban agroforestry"', '"urban biodiversity enhancement"', '"urban blue infrastructure"', '"urban ecosystem restoration"', '"urban forest restoration"', '"urban forest"', '"urban green infrastructure"', '"urban green"', '"urban greening"', '"urban natural infrastructure"', '"urban nature restoration"', '"urban park"', '"urban river restoration"', '"urban tree planting"', '"urban vegetation"', '"urban wetland restoration"', '"urban wetland"', '"urban woodland"', '"vegetated embankment"', '"vegetated roof"', '"vegetated swale"', '"vegetation based erosion control"', '"vegetation restoration"', '"vegetation thinning"', '"vertical green"', '"vertical garden"', '"water sensitive urban design"', '"watershed restoration"', '"wetland conservation"', '"wetland creation"', '"wetland enhancement"', '"wetland rehabilitation"', '"wetland re-creation"', '"wetland reestablishment"', '"wetland restoration"', '"wildlife corridor"', '"wildflower meadow creation"', '"woodland creation"', '"woodland restoration"', '"working with natural processes"'
]

if __name__ == "__main__":
    # import re
    # import json
    # import torch
    # from collections import defaultdict
    # from pathlib import Path
    # from transformers import T5ForConditionalGeneration, T5Tokenizer

    # MODEL_NAME = 'jbochi/madlad400-3b-mt'
    # model = T5ForConditionalGeneration.from_pretrained(
    #     MODEL_NAME, device_map="auto",  dtype=torch.bfloat16
    #     )
    # tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    # model.eval()

    # MAX_SEGMENT_TOKENS = 400


    import re
    import json
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import ollama

    # Setup (once):

    MODEL_NAME = "translategemma"
    OLLAMA_HOST = "http://localhost:11434"
    client = ollama.Client(host=OLLAMA_HOST)

    # No HF tokenizer here, so segments are limited by characters (~4 chars per token).
    MAX_SEGMENT_CHARS = 1500
    WORKERS = 1  # >1 only helps if the Ollama server runs with OLLAMA_NUM_PARALLEL set

    TARGET_LANG = ("en", "English")

    # Language code -> name, used in the TranslateGemma prompt.
    LANG_NAMES = {
        "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali", "cs": "Czech", "da": "Danish",
        "de": "German", "el": "Greek", "en": "English", "es": "Spanish", "et": "Estonian",
        "fa": "Persian", "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi",
        "hr": "Croatian", "hu": "Hungarian", "id": "Indonesian", "it": "Italian",
        "ja": "Japanese", "ko": "Korean", "lt": "Lithuanian", "lv": "Latvian",
        "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese",
        "ro": "Romanian", "ru": "Russian", "sk": "Slovak", "sl": "Slovenian",
        "sr": "Serbian", "sv": "Swedish", "th": "Thai", "tr": "Turkish",
        "uk": "Ukrainian", "vi": "Vietnamese", "zh": "Chinese",
    }


    def lang_name(code):
        base = code.split("-")[0].lower()
        if base in LANG_NAMES:
            return LANG_NAMES[base]
        try:  # optional: pip install langcodes language-data
            import langcodes
            return langcodes.get(code).display_name()
        except Exception:
            return code


    PROMPT = (
        "You are a professional {src_name} ({src_code}) to {tgt_name} ({tgt_code}) translator. "
        "Your goal is to accurately convey the meaning and nuances of the original {src_name} text "
        "while adhering to {tgt_name} grammar, vocabulary, and cultural sensitivities.\n"
        "Produce only the {tgt_name} translation, without any additional explanations or commentary. "
        "Please translate the following {src_name} text into {tgt_name}:\n\n\n{text}"
    )

    FOLDER = "/home/lt1412/Desktop/evaluation-reports-synthesis/data/test"
    translate_nonen_mds(folder=FOLDER, file_results="reportlanguage.json")
