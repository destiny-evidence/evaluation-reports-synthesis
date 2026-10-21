import re
import json
import torch
from collections import defaultdict
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_NAME = 'jbochi/madlad400-3b-mt'
model = T5ForConditionalGeneration.from_pretrained(
    MODEL_NAME, device_map="auto",  dtype=torch.bfloat16
    )
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

model.eval()

MAX_SEGMENT_TOKENS = 400

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

def n_tokens(s):
    return len(tokenizer(s, add_special_tokens=False).input_ids)


def split_line(line, max_tokens=MAX_SEGMENT_TOKENS):
    """Split one long line into pieces under max_tokens: sentences first, then raw tokens."""
    if n_tokens(line) <= max_tokens:
        return [line]

    sentences = re.split(r'(?<=[.!?。！？])\s+', line)
    parts, buf = [], ""
    for s in sentences:
        candidate = f"{buf} {s}".strip()
        if buf and n_tokens(candidate) > max_tokens:
            parts.append(buf)
            buf = s
        else:
            buf = candidate
    if buf:
        parts.append(buf)

    final = []
    for p in parts:
        if n_tokens(p) <= max_tokens:
            final.append(p)
        else:  # a single "sentence" that is still too long: hard split on tokens
            ids = tokenizer(p, add_special_tokens=False).input_ids
            for i in range(0, len(ids), max_tokens):
                final.append(tokenizer.decode(ids[i:i + max_tokens]))
    return final


@torch.inference_mode()
def translate_batch(segments, target_lang="en", max_new_tokens=512):
    prompts = [f"<2{target_lang}> {s}" for s in segments]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def translate_chunk(chunk, source_lang=None, target_lang="en"):
    """Single-segment version, same signature as before."""
    return translate_batch([chunk], target_lang)[0]


def translate_long(text, source_lang=None, batch_size=8):
    if not text or not text.strip():
        return ""

    lines = text.split("\n")

    # (line_index, segment) pairs; blank lines are left alone
    segments = []
    for i, line in enumerate(lines):
        if line.strip():
            for seg in split_line(line):
                segments.append((i, seg))

    translated = []
    texts = [s for _, s in segments]
    for b in range(0, len(texts), batch_size):
        translated.extend(translate_batch(texts[b:b + batch_size]))

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

    ignore_index = find_empty_files(folder)
    ignore_index = set(ignore_index)
    file_results = [f for f in file_results if f['file'] not in ignore_index]

    # index the folder once: stem -> list of paths (a list in case of duplicate stems in subfolders)
    by_stem = defaultdict(list)
    for p in folder.rglob("*.md"):
        by_stem[p.stem].append(p)


    for item in file_results:
            lang = item['language']
            if lang in ('en', 'unknown', None):
                continue

            stem = Path(item['file']).stem   # works whether the JSON has "10832_2" or "10832_2.md"
            matches = by_stem.get(stem, [])

            if not matches:
                print(f"No .md file found for {item['file']} - skipping.")
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
                    translated = translate_long(text)
                    if not translated or not translated.strip():
                        # Empty sequence: source text was blank or all chunks came back empty
                        print(f"Skipping {path.stem}: translation returned empty content.")
                        continue
                    print(f"Translated file {path.stem}, saving it now.")
                    # save alongside original
                    out_path.write_text(translated, encoding="utf-8")
                except (torch.cuda.OutOfMemoryError, RuntimeError, OSError) as e:
                    if isinstance(e, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                    print(f"Error translating {path}: {e}. Queued for retry.")
                    failed_paths.append((str(path), item['language'], str(e)))

                except Exception as e:
                    print(f"Unexpected error on {path}: {e}")
                    continue
    with open('filestoretry.json', 'w', encoding="utf-8") as f:
        json.dump(failed_paths, f, ensure_ascii=False, indent=2)
    return failed_paths
