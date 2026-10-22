import re
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ollama

# Setup (once):

MODEL_NAME = "translategemma:4b"
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


# find empty files in the folder, create ignore_index list
def find_empty_files(folder):
    folder = Path(folder)
    ignore_index = []

    for path in folder.rglob("*.md"):
        try:
            text = path.read_text(encoding="utf-8")
            if not text.strip():  # empty or whitespace only
                print(f"Empty: {path.name}")
                ignore_index.append(path.stem)
        except Exception as e:
            print(f"Corrupt/unreadable: {path.name} — {e}")
            ignore_index.append(path.stem)

    print(f"\nFound {len(ignore_index)} invalid files")
    return ignore_index


def split_line(line, max_chars=MAX_SEGMENT_CHARS):
    """Split one long line into pieces under max_chars: sentences first, then hard cuts."""
    if len(line) <= max_chars:
        return [line]

    sentences = re.split(r"(?<=[.!?。！？])\s+", line)
    parts, buf = [], ""
    for s in sentences:
        candidate = f"{buf} {s}".strip()
        if buf and len(candidate) > max_chars:
            parts.append(buf)
            buf = s
        else:
            buf = candidate
    if buf:
        parts.append(buf)

    final = []
    for p in parts:
        if len(p) <= max_chars:
            final.append(p)
        else:  # a single "sentence" that is still too long: hard split
            final.extend(p[i:i + max_chars] for i in range(0, len(p), max_chars))
    return final


def translate_chunk(chunk, source_lang="auto", target=TARGET_LANG):
    """Translate one segment with the local Ollama model."""
    prompt = PROMPT.format(
        src_name=lang_name(source_lang),
        src_code=source_lang,
        tgt_name=target[1],
        tgt_code=target[0],
        text=chunk,
    )
    resp = client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return resp["message"]["content"].strip()

# TODO: is the structure of the document preserved?
# TODO: how long and expensive is one document?

def translate_long(text, source_lang="auto", workers=WORKERS):
    if not text or not text.strip():
        return ""

    lines = text.split("\n")

    # (line_index, segment) pairs; blank lines are left alone
    segments = []
    for i, line in enumerate(lines):
        if line.strip():
            for seg in split_line(line):
                segments.append((i, seg))
    print(f"Total {len(segments)} segments in the document")

    texts = [s for _, s in segments]
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            translated = list(ex.map(lambda s: translate_chunk(s, source_lang), texts))
    else:
        translated = []
        for idx, seg in enumerate(texts, start=1):
            print(f"Translating chunk {idx}/{len(texts)}")
            result = translate_chunk(seg, source_lang)
            print(f"Translated chunk {idx}/{len(texts)}")
            translated.append(result)

    by_line = defaultdict(list)
    for (i, _), t in zip(segments, translated):
        if t and t.strip():
            by_line[i].append(t.strip())

    out_lines = [" ".join(by_line[i]) if i in by_line else line
                 for i, line in enumerate(lines)]
    return "\n".join(out_lines)


# translate non-English markdown files into English
def translate_nonen_mds(folder, file_results):
    folder = Path(folder)
    failed_paths = []

    if isinstance(file_results, (str, Path)):
        with open(file_results, encoding="utf-8") as f:
            file_results = json.load(f)

    ignore_index = set(find_empty_files(folder))
    file_results = [f for f in file_results if f["file"] not in ignore_index]

    # index the folder once: stem -> list of paths (a list in case of duplicate stems in subfolders)
    by_stem = defaultdict(list)
    for p in folder.rglob("*.md"):
        by_stem[p.stem].append(p)

    for item in file_results:
        lang = item["language"]
        if lang in ("en", "unknown", None):
            continue

        stem = Path(item["file"]).stem  # works whether the JSON has "10832_2" or "10832_2.md"
        matches = by_stem.get(stem, [])

        if stem == "11129_1":
            if not matches:
                print(f"No .md file found for {item['file']} - skipping.")
                continue

            for path in matches:
                if "_en_translation" in path.stem:
                    continue

                out_path = path.with_name(path.stem + "_en_translation_ollama.md")

                if out_path.exists():
                    print(f"{out_path} already exists - skipping!")
                    continue

                try:
                    text = path.read_text(encoding="utf-8")
                    print(f"Beginning to translate file {path.stem} from {lang} to english now.")
                    translated = translate_long(text, source_lang=lang)
                    if not translated or not translated.strip():
                        print(f"Skipping {path.stem}: translation returned empty content.")
                        continue
                    print(f"Translated file {path.stem}, saving it now.")
                    out_path.write_text(translated, encoding="utf-8")
                except (ollama.ResponseError, ConnectionError, RuntimeError, OSError) as e:
                    print(f"Error translating {path}: {e}. Queued for retry.")
                    failed_paths.append((str(path), item["language"], str(e)))
                except Exception as e:
                    print(f"Unexpected error on {path}: {e}")
                    continue

    with open("filestoretry.json", "w", encoding="utf-8") as f:
        json.dump(failed_paths, f, ensure_ascii=False, indent=2)
    return failed_paths
