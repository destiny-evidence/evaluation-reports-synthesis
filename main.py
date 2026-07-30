from scripts.utils import match_search_results
from scripts.plots import embeddings_map2d

health_terms = ['health',
    '"mental health"',
    'cardiovascular',
    '"physical health"',
    'respiratory',
    'mortality',
    '"infectious diseases"',
    'injuries',
    '"food-borne"',
    '"water-borne"',
    '"vector-borne"',
    'morbidity',
    'deaths',
    'illnesses'
]

climate_risk_terms = ['"climate change"',
    'disaster',
    'hazard',
    'mudslides',
    'storms',
    'landslides',
    'droughts',
    'floods',
    'heat',
    'erosion',
    '"extreme heat"'
]

nbs_terms = [
    '"nature-based"',
    '"nature-based solutions"',
    '"ecosystem-based"',
    '"ecosystem-based solutions"',
    '"ecosystem-based adaptation"',
    '"ecosystem-based disaster risk reduction"',
    '"green infrastructure"',
    'rewilding',
    '"natural flood management"',
    '"green roofs"',
    '"green walls"',
    'agroforestry',
    'wetlands',
    '"urban green"',
    'afforestation',
    'reforestation',
    '"blue-green"',
    '"urban forests"',
    'floodplains',
    'mangroves',
    'watershed',
    '"slope stabilisation"'
]

air_pollution_terms = [
    '"air pollution"',
    '"air quality"',
    '"PM2.5"',
    '"PM10"',
    '"particulate matter"',
]

intervention_terms = [ 
    # relevant for air pollution
    '"low emission zone"',
    '"low-emission zone"',
    '"pollution control"'
]

if __name__ == "__main__":

    FILERESULT = "/home/lt1412/Desktop/evaluation-reports-synthesis/search_results/search_result_nbs_health_climate.json"
    FILEDATABASE = "/home/lt1412/Desktop/UNEG-repository/member_publications_afterdownload.csv"
    SEARCHTITLE = "nbs_climate_health"

    match_search_results(fileresults=FILERESULT, filedatabase=FILEDATABASE, search_title=SEARCHTITLE)
    embeddings_map2d()

    # FOLDER = "/home/lt1412/Desktop/UNEG-repository/markdown"
    # QUERY = build_and_query([nbs_terms, climate_risk_terms, health_terms])
    # print(f"Running AND query: {QUERY}\n")
    # RESULT = boolean_search_markdown_section(folder=FOLDER, query=QUERY)
    # print(f"User query: {RESULT["query"]}")
    # print(f"Total files matching query (inside section search): {RESULT["matched_doc_count"]}")
    # print(f"Total sections matching query: {RESULT["total_section_matches"]}")
    # with open(f'search_result_{search_title]}.json', 'w') as f:
    #     json.dump(RESULT,f)

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



