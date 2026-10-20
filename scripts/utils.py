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

MD_DIR = Path("marker-parsed/markdown")

CHROMA_PATH = "chroma_db"
# Configure maximum chunk size for chroma.add via env var or default to 4000
CHROMA_MAX_BATCH = int(os.getenv("CHROMA_MAX_BATCH", "4000"))  # safe batch size for Chromadb adds

BOOL_OPS = {"AND", "OR", "NOT"}

logger = logging.getLogger(__name__)

# TODO: update cause it finds 0 invalid files but some other functions clearly have to skip some files
# TODO: use ignore_index in all following calls

# find empty files in the folder, create ignore_index list
def find_empty_files(folder):
    folder = Path(folder)
    ignore_index = []
    
    for path in folder.rglob("*.md"):
        try:
            text = path.read_text(encoding="utf-8")
            if not text.strip():  # empty or whitespace only
                print(f"Empty: {path.name}")
                ignore_index.append(path.stem.split("_")[0] )
        except Exception as e:
            print(f"Corrupt/unreadable: {path.name} — {e}")
            ignore_index.append(path.stem.split("_")[0] )
    
    print(f"\nFound {len(ignore_index)} invalid files")
    return ignore_index

# describe how many files we have per doc_id, how many sections and pages, do they include an executive summary?
# for evaluations with multiple documents, find which one is the main document?


# parse markdown files into hierarchical sections
# TODO: does the content also include section title? otherwise Im not sure how the section titles got translated...
def parse_md_sections(md_path: Path):
    content = md_path.read_text()
    sections = []
    current_section = {'level': 0, 'title': 'default title', 'content': ''}

    # split the document into lines and identify titles and content
    for line in content.split('\n'):
        title_match = re.match(r'^(#{1,6})\s+(.+)', line) # up to 6 levels
        if title_match:
            if current_section['content'].strip():
                sections.append(current_section)
            level = len(title_match.group(1))
            current_section = {
                'level': level,
                'title': title_match.group(2),
                'content': ''
            }
        else:
            current_section['content'] += line + '\n'
    if current_section['content'].strip():
        sections.append(current_section)
    return sections



def clean_text(text):
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)         # fenced code blocks
    text = re.sub(r"`[^`]+`", "", text)                             # inline code
    text = re.sub(r"https?://\S+", "", text)                        # URLs
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)      # headers
    text = re.sub(r"[*_~|>\-]+", " ", text)                         # markdown symbols
    return text.strip()


def detect_language(folder, out_path="reportlanguage.json"):
    """Detect the language of every .md file (one result per document, not per doc_id)."""
    folder = Path(folder)
    file_results = []

    for path in sorted(folder.rglob("*.md")):  # recursive search
        try:
            sections = parse_md_sections(path)
        except Exception as e:
            print(f"Could not read {path}: {e}")
            continue

        doc_id = re.sub(r"_\d+$", "", path.stem)  # e.g. "abc_3" -> "abc"

        # prefer the 3rd section, then fall back to the others in order;
        # use the first one that still has text after cleaning
        ordered = ([sections[2]] if len(sections) > 2 else []) + \
                  [s for i, s in enumerate(sections) if i != 2]
        text = ""
        for s in ordered:
            cleaned = clean_text(s.get("content") or "")
            if cleaned:
                text = cleaned
                break

        language, confidence = "unknown", None
        if text:
            try:
                best = detect_langs(text)[0]          # most probable language
                language, confidence = best.lang, round(best.prob, 3)
            except Exception:
                pass

        file_results.append({
            "doc_id": doc_id,        # the parent document
            "file": path.name,       # e.g. abc_3.md -> one row per file
            "language": language,
            "confidence": confidence,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(file_results, f, ensure_ascii=False, indent=2)

    return file_results


# break the document into pieces for translation
def translate_long(text, source_lang):
    if not text or not text.strip():
        return ""  # detect empty input before hitting the API
    
    chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
    translated_chunks = []
    for c in chunks:
        if not c.strip():
            # skip whitespace-only chunks (empty sequence fix)
            continue
        result = GoogleTranslator(source=source_lang, target='en').translate(c)
        if result:
            translated_chunks.append(result)
    return " ".join(translated_chunks)

# translate non-English markdown files into English
def translate_nonen_mds(folder, file_results):
    folder = Path(folder)
    ignore_index = find_empty_files(folder)
    file_results = [f for f in file_results if f['file_id'] not in ignore_index]
    failed_paths = []  

    for item in file_results:
            if item['language'] == 'en':
                continue

            # find the file or files associated with this document id
            matches = list(folder.rglob(f"{item['file_id']}_*.md"))

            if not matches:
                continue

            for path in matches:
                
                if "_en_translation" in path.stem:
                    continue

                out_path = path.with_name(path.stem + "_en_translation.md")

                if out_path.exists():
                    print(f"{out_path} already exists - skipping!")
                    continue

                try:
                    text = path.read_text(encoding="utf-8")
                    print(f"Beginning to translate file {path.stem} now.")
                    translated = translate_long(text, item['language'])
                    if not translated or not translated.strip():
                        # Empty sequence: source text was blank or all chunks came back empty
                        print(f"Skipping {path.stem}: translation returned empty content.")
                        continue
                    print(f"Translated file {path.stem}, saving it now.")
                    # save alongside original
                    out_path.write_text(translated, encoding="utf-8")
                except (RequestError, TooManyRequests, ConnectionError, OSError) as e:
                    # API / network failure — log the path so we can retry later
                    print(f"Connection error translating {path}: {e}. Queued for retry.")
                    failed_paths.append((path, item['language'], e))

                except Exception as e:
                    print(f"Could not read {path}: {e}")
                    continue
    with open('filestoretry.json', 'w') as f:
        json.dump(failed_paths, f)
    return failed_paths
# TODO: save failed_paths 

# add embeddings to Chroma DB in chunks to avoid crashing
def chunked_add(collection, embeddings, documents, metadatas, ids, max_batch=CHROMA_MAX_BATCH):
    """Add items to a chroma collection in chunks to avoid hitting max batch size limits.

    embeddings: numpy array (n, dim) or list of lists
    documents: list[str]
    metadatas: list[dict]
    ids: list[str]
    """
    n = len(ids)
    # ensure embeddings is a list of lists for chroma
    # If embeddings is numpy array, slice and call .tolist() per chunk to avoid large memory spikes
    for i in range(0, n, max_batch):
        j = min(i + max_batch, n)
        emb_chunk = embeddings[i:j]
        # if numpy array, convert to list
        try:
            emb_chunk_list = emb_chunk.tolist()
        except Exception:
            emb_chunk_list = list(emb_chunk)

        collection.add(
            embeddings=emb_chunk_list,
            documents=documents[i:j],
            metadatas=metadatas[i:j],
            ids=ids[i:j],
        )

# create embeddings for all section titles
def title_embeddings():
    rows = []
    for md_path in sorted(MD_DIR.glob("*.md")):
        doc_id = md_path.stem  # e.g. "491_1"
        sections = parse_md_sections(md_path)
        for i, s in enumerate(sections):
            sec_id = f"{doc_id}_sec_{i}"
            rows.append(
                {
                    "doc_id": doc_id,
                    "sec_id": sec_id,
                    "level": s["level"],
                    "title": s["title"],
                    "content": s["content"],
                    "text": f"{s['title']} {s['content']}",
                }
            )

    df = pd.DataFrame(rows)

    # prepare data for streaming encode + add
    model = SentenceTransformer('all-MiniLM-L6-v2')
    BATCH_SIZE = 256

    texts = df["title"].tolist()
    metadatas = [
        {
            "doc_id": r["doc_id"],
            "sec_id": r["sec_id"],
            "level": r["level"],
            "title": r["title"],
        }
        for r in df.to_dict(orient="records")
    ]
    ids = df["sec_id"].tolist()

    # store in ChromaDB with metadata (create collection once)
    chroma_client = PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')
    collection = chroma_client.get_or_create_collection(
        name="md_sections",
        embedding_function=ef
    )

    # incremental clustering with MiniBatchKMeans to avoid storing all embeddings
    n_samples = len(texts)
    clusters_n = min(N_CLUSTERS, max(1, n_samples))
    mbk = MiniBatchKMeans(n_clusters=clusters_n, random_state=16, batch_size=BATCH_SIZE)
    all_clusters = []
    all_embeddings = []

    # encode in batches and add in chunks as we go
    for i in range(0, len(texts), BATCH_SIZE):
        j = min(i + BATCH_SIZE, len(texts))
        batch_texts = texts[i:j]
        print(f"Encoding titles {i}:{j} / {len(texts)}")
        emb = model.encode(batch_texts, show_progress_bar=False)
        all_embeddings.extend(emb.tolist())

        # update clustering model incrementally and predict cluster for this batch
        try:
            mbk.partial_fit(emb)
            batch_clusters = mbk.predict(emb)
        except Exception:
            # fallback: if partial_fit not applicable (very small data), assign zeros
            batch_clusters = [0] * len(batch_texts)

        all_clusters.extend(batch_clusters)

        # add this batch to chroma (chunked_add will further split if needed)
        chunked_add(collection, emb, batch_texts, metadatas[i:j], ids[i:j])

    df['cluster'] = all_clusters
    df['embedding'] = all_embeddings

    # save structured data
    os.makedirs("title_embeddings", exist_ok=True)
    df.to_parquet('title_embeddings/all_md_section_titles.parquet')

# create embeddings for all sections
def section_embeddings(): # TODO: add a similar function for paragraph embeddings
    rows = []
    for md_path in sorted(MD_DIR.glob("*.md")):
        doc_id = md_path.stem  # e.g. "491_1"
        sections = parse_md_sections(md_path)
        for i, s in enumerate(sections):
            sec_id = f"{doc_id}_sec_{i}"
            rows.append(
                {
                    "doc_id": doc_id,
                    "sec_id": sec_id,
                    "level": s["level"],
                    "title": s["title"],
                    "content": s["content"],
                    "text": f"{s['title']} {s['content']}",
                }
            )

    df = pd.DataFrame(rows)

    # generate embeddings in streaming fashion and add to chroma incrementally
    model = SentenceTransformer('all-MiniLM-L6-v2')
    BATCH_SIZE = 256

    texts = df["text"].tolist()
    metadatas = [
        {
            "doc_id": r["doc_id"],
            "sec_id": r["sec_id"],
            "level": r["level"],
            "title": r["title"],
        }
        for r in df.to_dict(orient="records")
    ]
    ids = df["sec_id"].tolist()

    chroma_client = PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')
    collection = chroma_client.get_or_create_collection(
        name="md_sections",
        embedding_function=ef
    )

    # incremental clustering
    n_samples = len(texts)
    clusters_n = min(N_CLUSTERS, max(1, n_samples))
    mbk = MiniBatchKMeans(n_clusters=clusters_n, random_state=16, batch_size=BATCH_SIZE)
    all_clusters = []
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        j = min(i + BATCH_SIZE, len(texts))
        batch_texts = texts[i:j]
        print(f"Encoding sections {i}:{j} / {len(texts)}")
        emb = model.encode(batch_texts, show_progress_bar=False)
        all_embeddings.extend(emb.tolist())

        try:
            mbk.partial_fit(emb)
            batch_clusters = mbk.predict(emb)
        except Exception:
            batch_clusters = [0] * len(batch_texts)

        all_clusters.extend(batch_clusters)

        chunked_add(collection, emb, batch_texts, metadatas[i:j], ids[i:j])

    df['cluster'] = all_clusters
    df['embedding'] = all_embeddings

    # save structured data
    os.makedirs("section_embeddings", exist_ok=True)
    df.to_parquet('section_embeddings/all_md_sections.parquet')

# search section titles (exact and fuzzy)
# TODO: update to load all texts once and then search them all
def search_section_titles(folder, keyword, case_sensitive=False, threshold=80): 
 
    folder = Path(folder)
    if not case_sensitive:
        keyword = keyword.lower()

    total_files = 0
    files_with_exact_matches = 0
    file_ids_exact = []
    files_with_fuzzy_matches = 0
    file_ids_fuzzy = []

    for path in folder.rglob("*.md"):
        total_files += 1
        exact_found = False  # track per file
        fuzzy_found = False  # track per file

        try:
            sections = parse_md_sections(path)
        except Exception as e:
            print(f"Could not read {path}: {e}")
            continue

        for section in sections:
            # search title 
            text = section['title']
            haystack = text if case_sensitive else text.lower()
            kw = keyword if case_sensitive else keyword.lower()

            if haystack.count(kw) > 0:
                exact_found = True  # don't increment here

            for i in range(len(haystack) - len(kw) + 1):
                window = haystack[i:i+len(kw)]
                if fuzz.ratio(window, kw) >= threshold:
                    fuzzy_found = True  # don't increment here
                    break  # no need to keep scanning once found

        # increment once per file, after all sections checked
        if exact_found:
            files_with_exact_matches += 1
            file_ids_exact.append(path.stem)
        if fuzzy_found:
            files_with_fuzzy_matches += 1
            file_ids_fuzzy.append(path.stem)

    return {
        "keyword": keyword,
        "total_files": total_files,
        "files_with_exact_matches": files_with_exact_matches,
        "files_with_fuzzy_matches": files_with_fuzzy_matches,
        "file_ids_exact": file_ids_exact,
        "file_ids_fuzzy": file_ids_fuzzy
    }

def search_markdown_folder(folder, keyword, case_sensitive=False): # update to load all texts once and then search them all 
    folder = Path(folder)
    if not case_sensitive:
        keyword = keyword.lower()

    total_matches = 0
    files_with_matches = 0
    total_files = 0
    file_ids = []

    for path in folder.rglob("*.md"):  # recursively search for .md files
        total_files += 1
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"Could not read {path}: {e}")
            continue

        haystack = text if case_sensitive else text.lower()
        count = haystack.count(keyword)
        if count > 0:
            files_with_matches += 1
            total_matches += count
            file_ids.append(path.stem)  # store the file ID (stem) for matched files

    return {
        "keyword": keyword,
        "total_files": total_files,
        "files_with_matches": files_with_matches,
        "total_matches": total_matches
    }

def search_markdown_folder_freq(folder, keyword, case_sensitive=False, max_words=3): # includes per file frequencies
    folder = Path(folder)
    if not case_sensitive:
        keyword = keyword.lower()
    
    keyword = keyword.strip('"') 
    
    #pre-tokenise keyword to know its length
    #words = keyword.split()
    #keyword_len = len(keyword_words)
    
    total_matches = 0
    files_with_matches = 0
    total_files = 0
    file_results = []
    matched_ids = set() 
    
    for path in folder.rglob("*.md"):
        total_files += 1
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"Could not read {path}: {e}")
            continue
        
        haystack = text if case_sensitive else text.lower()
        words = re.findall(r'\b\w+\b', haystack)
        
        #if keyword_len == 1:
            # single word: use fast count
            #count = haystack.count(keyword)
        #else:
            # multi-word phrase: sliding window
            #count = 0
            #for i in range(len(words) - keyword_len + 1):
                #ngram = ' '.join(words[i:i+keyword_len])
                #if ngram == keyword:
                    #count += 1
        count = haystack.count(keyword)

        if count > 0:
            files_with_matches += 1
            total_matches += count
            doc_id = re.sub(r'_\d+$', '', path.stem)
            matched_ids.add(doc_id)
            file_results.append({
                "file_id": path.stem,
                "frequency": count
            })

    return {
        "keyword": keyword,
        "total_files": total_files,
        "files_with_matches": files_with_matches,
        "total_matches": total_matches,
        "matched_doc_ids": sorted(matched_ids),
        "matched_doc_count": len(matched_ids),
        "file_frequencies": file_results  # New: list of dicts with per-file counts
    }

def visualise_file_frequencies(result):
    file_data = result['file_frequencies']
    if not file_data:
        print("No matches found to visualise.")
        return

    frequencies = [f['frequency'] for f in file_data if 1 < f['frequency'] < 50]

    freq_counter = {}
    for freq in frequencies:
        freq_counter[freq] = freq_counter.get(freq, 0) + 1
    
    # sort by frequency value
    sorted_freqs = sorted(freq_counter.items())
    freq_values = [f[0] for f in sorted_freqs]
    file_counts = [f[1] for f in sorted_freqs]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(freq_values, file_counts, color='skyblue', edgecolor='navy', alpha=0.8)
    
    plt.title(f'"{result["keyword"]}" Frequency Distribution Across Files\n'
              f'(Total: {result["total_matches"]} matches in {result["files_with_matches"]}/{result["matched_doc_count"]} files/docs)')
    plt.xlabel('Keyword Occurrences per File')
    plt.ylabel('Number of Files')
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for bar, freq_val in zip(bars, freq_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                str(freq_val), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save with safe filename
    safe_keyword = result["keyword"].replace(" ", "_").replace("/", "_")
    os.makedirs("keyword_frequency_histograms", exist_ok=True)
    save_path = f'keyword_frequency_histograms/keyword_frequency_histogram_{safe_keyword}_v2.png' #v2 has the 2-word counter commented out
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def file_term_presence(text, terms, case_sensitive=False):
    if not case_sensitive:
        text = text.lower() # text is the entire content of the file
    presence = {}
    for t in terms:
        key = t if case_sensitive else t.lower()
        presence[t] = (key in text)
    return presence

def parse_boolean_query(raw_query):
    """
    Convert a user string like:
        "apple AND (banana OR NOT cherry)"
    into a Python expression like:
        "vars['apple'] and (vars['banana'] or (not vars['cherry']))"
    and return (expr_string, set_of_terms).
    """
    # Normalize spaces
    q = raw_query.strip()

    # Tokenize: words and parentheses
    # quoted phrase becomes a single token: "air pollution"
    tokens = re.findall(r'"[^"]+"|\w+|\(|\)', q)

    terms = set()
    py_tokens = []

    for token in tokens:
        upper = token.upper()
        if upper in BOOL_OPS:
            # map to Python boolean operators
            if upper == "AND":
                py_tokens.append("and")
            elif upper == "OR":
                py_tokens.append("or")
            elif upper == "NOT":
                py_tokens.append("not")
        elif token in ("(", ")"):
            py_tokens.append(token)
        else:
            # strip quotes for the actual search term
            clean = token.strip('"')        # "air pollution" → air pollution
            terms.add(clean)
            py_tokens.append(f"vars[{clean!r}]")  # vars['air pollution']

    expr = " ".join(py_tokens)
    return expr, terms

def doc_id_from_path(path):
    stem = re.sub(r'_en_translation$', '', Path(path).stem)
    return re.sub(r'_\d+$', '', stem)

# detect_language may store 'en', 'unknown', a langdetect Language object,
# or a list of them ([de:0.99, en:0.01]). Reduce all of those to a plain code.
def normalise_language(language):
    if isinstance(language, (list, tuple)):
        language = language[0] if language else ""
    lang = getattr(language, "lang", language)   # Language object → its .lang
    lang = str(lang).strip().strip("[]")         # "[de:0.99]" → "de:0.99"
    return lang.split(":")[0].strip().lower()    # "de:0.99"  → "de"

def select_search_paths(folder, file_results=None):
    folder = Path(folder)
    paths = sorted(folder.rglob("*.md"))

    if file_results is None:          # keep old behaviour when no language info given
        return paths

    if isinstance(file_results, (str, Path)):   # allow passing reportlanguage.json
        with open(file_results) as f:
            file_results = json.load(f)

    nonen_ids = {item['file_id'] for item in file_results
                 if normalise_language(item.get('language')) != 'en'}
    selected = []
    for path in paths:
        is_translation = path.stem.endswith("_en_translation")
        if doc_id_from_path(path) in nonen_ids:
            if is_translation:
                selected.append(path)
                print(f"Using English translation for {path.stem}.")                 # use the translation
            else:
                translation = path.with_name(path.stem + "_en_translation.md")
                if not translation.exists():          # nothing translated yet
                    print(f"No English translation for {path.stem}, searching the original.")
                    selected.append(path)
        elif not is_translation:
            selected.append(path)                     # English report: original only

    return selected

def boolean_search_markdown_section(folder, query, case_sensitive=False, file_results=None): # search matches inside sections rather than across the full file
    folder = Path(folder)

    # build Python expression and term set from the query
    expr, terms = parse_boolean_query(query)

    # validate expr once before the file loop
    dummy = {t: False for t in terms}
    try:
        eval(expr, {"__builtins__": {}}, {"vars": dummy})
    except SyntaxError as e:
        raise ValueError(f"Invalid query: {query!r}") from e

    total_files = 0
    files_matching_query = 0
    total_section_matches = 0
    matched_ids = set()
    file_frequencies = []

    paths = select_search_paths(folder, file_results)

    for path in paths:
        total_files += 1
        if total_files % 200 == 0:
            print(f"searched {total_files} files, {files_matching_query} matching so far", flush=True)
        try:
            sections = parse_md_sections(path)
        except Exception as e:
            print(f"Could not read {path}: {e}")
            continue

        section_match_count = 0  # sections matching in this file
        for section in sections:
            # search title + content together so a term in the heading counts
            section_text = section['title'] + '\n' + section['content']
            vars_for_section = file_term_presence(
                section_text, terms, case_sensitive=case_sensitive
            )
            try:
                match = bool(eval(expr, {"__builtins__": {}}, {"vars": vars_for_section}))
            except Exception as e:
                print(f"Error evaluating query on {path} section '{section['title']}': {e}")
                continue
            if match:
                section_match_count += 1

        if section_match_count > 0:
            files_matching_query += 1
            total_section_matches += section_match_count
            # strip the _1 / _2 suffix: "FILEID_1" → "FILEID"
            doc_id = doc_id_from_path(path)
            matched_ids.add(doc_id)
            file_frequencies.append({
                "file_id": doc_id,
                "frequency": section_match_count
            })

# file frequencies are counted multiple times if we have _2 files and so on
    return {
        "query": query,
        "total_files": total_files,
        "files_matching_query": files_matching_query,
        "total_section_matches": total_section_matches,
        "matched_doc_ids": sorted(matched_ids),
        "matched_doc_count": len(matched_ids),
        "file_frequencies": file_frequencies    
    }

def split_paragraphs(text):
    """Split text into non-empty paragraphs on blank lines."""
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

def boolean_search_markdown_para(folder, query, case_sensitive=False):
    folder = Path(folder)
    expr, terms = parse_boolean_query(query)

    dummy = {t: False for t in terms}
    try:
        eval(expr, {"__builtins__": {}}, {"vars": dummy})
    except SyntaxError as e:
        raise ValueError(f"Invalid query: {query!r}") from e

    total_files = 0
    files_matching_query = 0
    total_paragraph_matches = 0
    matched_ids = set()
    file_results = []

    for path in folder.rglob("*.md"):
        total_files += 1
        try:
            sections = parse_md_sections(path)
        except Exception as e:
            print(f"Could not read {path}: {e}")
            continue

        file_has_match = False

        paragraph_match_count = 0
        for section in sections:
            # -- paragraph-level match --
            for para in split_paragraphs(section['content']):
                vars_for_para = file_term_presence(para, terms, case_sensitive=case_sensitive)
                try:
                    para_match = bool(eval(expr, {"__builtins__": {}}, {"vars": vars_for_para}))
                except Exception as e:
                    print(f"Error on {path} paragraph: {e}")
                    continue
                if para_match:
                    paragraph_match_count += 1
                    file_has_match = True

        if file_has_match:
            total_paragraph_matches += paragraph_match_count
            files_matching_query += 1
            doc_id = re.sub(r'_\d+$', '', path.stem)
            matched_ids.add(doc_id)
            file_results.append({
                "file_id": doc_id,
                "frequency": paragraph_match_count # how many times total is the query matched inside a paragraph 
                # different from total appearances of the terms per document (for creating histograms)
            })

    return {
        "query": query,
        "total_files": total_files,
        "files_matching_query": files_matching_query,
        "total_paragraph_matches": total_paragraph_matches,
        "matched_doc_ids": sorted(matched_ids),
        "matched_doc_count": len(matched_ids),
        "file_fequency": file_results
    }

def build_or_query(terms):
    """
    Connect all terms in a list with OR.
    E.g. ['health', '"mental health"'] → 'health OR "mental health"'
    """
    # Filter out empty strings in case a list has blank entries
    filtered = [t for t in terms if t.strip()]
    return " OR ".join(filtered)

def build_and_query(term_lists):
    """
    Connect multiple synonym lists with AND, each group wrapped in parentheses.
    E.g. (['a', 'b'], ['x', 'y']) → '(a OR b) AND (x OR y)'
    
    Skips empty lists entirely so you don't get dangling ANDs.
    """
    group_queries = []
    for term_list in term_lists:
        filtered = [t for t in term_list if t.strip()]
        if not filtered:
            continue  # skip empty lists like wellbeing_terms
        group_query = " OR ".join(filtered)
        # Wrap in parentheses only if more than one term
        if len(filtered) > 1:
            group_queries.append(f"({group_query})")
        else:
            group_queries.append(group_query)
    return " AND ".join(group_queries)

# save the search results including metadata associated with the file_id
def match_search_results(fileresults, filedatabase, search_title):
    with open(fileresults, 'r') as f:
        data = json.load(f)
    matched_ids = data['matched_doc_ids']
    matched_ids = [str(id) for id in matched_ids]
    print(f"your search results contain {len(matched_ids)} unique file IDs")


    df = pd.read_csv(filedatabase)
    df['item-ID'] = df['item-ID'].astype(str)
    print(f"looking for matching data for your search results in the database of {df.shape[0]} rows")
    
    # filter the dataframe
    filtered_df = df[df['item-ID'].isin(matched_ids)]
    print(f"saving the data now, including {filtered_df.shape[0]} rows")
    filtered_df.to_csv(f"search_results/{search_title}.csv", index=False)

# TODO: double check the visualise_file_frequencies -- seems like file count vs bar chart is sometimes off
# TODO: if there is no translation available - translate the query into the language of the file and search
# TODO: implement instead semantic search in the section embeddings / paragraph embeddings