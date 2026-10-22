import re
import unicodedata
from typing import Any, Dict, List, Optional

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
KEEP_LABELS = {"executive_summary", "introduction", "project_description", "methods", "results"}
END_LABELS = {"references", "appendices"}

# TODO: maybe we want to exclude introduction if it doesn't include project description? (assuming can be hidden in text without any subtitle identation)
# TODO: add synonyms and translations to each category

# exclude in-word matches
SECTION_PATTERNS = {
    "executive_summary": [
        r"\bexecutive summary\b",
        r"\bresumen ejecutivo\b",
        r"\bresume executif\b",
        r"\bsommaire executif\b",
        r"\bresumo executivo\b",
    ],
    "introduction": [
        r"\bintroduction\b",
        r"\bbackground\b",
        r"\bcontext\b",
        r"\bintroduccion\b",
        r"\bantecedentes\b",
        r"\bcontexte\b",
        r"\bcadre\b",
        r"\bintroducao\b",
        r"\bcontexto\b",
    ],
    "project_information": [
        r"\bintroduction\b",
        r"\btheory of change\b",
        r"\bcountry programme\b",
        r"\bproject background and context\b",
        r"\bdescription of .+\b",
        r"\breview of implementation\b",
        r"\bthe context of .+\b",
    ],
    "methods": [
        r"\bmethodology\b",
        r"\bmethods?\b",
        r"\bevaluation design\b",
        r"\bevaluation approach\b",
        r"\bmethodologie\b",
        r"\bmethode?s?\b",
        r"\bmetodologia\b",
        r"\bmetodologic\w*\b",
        r"\bdiseno de la evaluacion\b",
        r"\bevaluation purpose\b",
        r"\bevaluation object\b",
        r"\bevaluation object and context\b",
        r"\bmethodological approach\b",
        r"\bevaluation methodology and limitations\b",
        r"\bmethodology of evaluation\b",
    ],
    "results": [
        r"\bresults\b",
        r"\bfindings\b",
        r"\bevaluation findings\b",
        r"\blessons learnt\b",
        r"\blessons learned\b",
        r"\bfindings tables\b",
        r"\bspecific findings\b",
        r"\bpresentation of findings regarding project performance\b",
        r"\bcoverage and quality outcomes\b",
    ],
    "references": [
        r"\breferences\b",
        r"\bbibliography\b",
        r"\bworks cited\b",
        r"\breferencias\b",
        r"\bbibliografia\b",
        r"\bbibliographie\b",
    ],
    "appendices": [
        r"\bappendix\b",
        r"\bappendices\b",
        r"\bannex\b",
        r"\bannexes\b",
        r"\banexo\b",
        r"\banexos\b",
        r"\bapendice\b",
        r"\bapendices\b",
    ],
}

COMPILED_PATTERNS = {
    section: [re.compile(p, re.IGNORECASE) for p in patterns]
    for section, patterns in SECTION_PATTERNS.items()
}

def normalize_heading(title: str) -> str:
    """Light normalization for heading text only (no whitespace flattening of the document)."""
    title = re.sub(r"[*_`]", "", title)                    # strip bold/italic/code markers
    title = unicodedata.normalize("NFKD", title)
    title = "".join(ch for ch in title if not unicodedata.combining(ch))
    title = re.sub(r"\s+", " ", title).strip().lower()
    return NUMBERING_RE.sub("", title)                     # "1. Executive Summary" -> "executive summary"


def classify_heading(title: str) -> Optional[str]:
    norm = normalize_heading(title)
    for section_name, patterns in COMPILED_PATTERNS.items():
        if any(p.search(norm) for p in patterns):
            return section_name
    return None

# parse markdown files into hierarchical sections
def parse_md_sections(md_path: Path):
    content = md_path.read_text()
    sections = []
    current = {'level': 0, 'title': 'default title', 'content': ''}

    # split the document into lines and identify titles and content
    for line in content.split('\n'):
        m = HEADING_RE.match(line)
        if m:
            if current['level'] > 0 or current['content'].strip():
                sections.append(current)
            current = {'level': len(m.group(1)), 'title': m.group(2), 'content':''}
        else:
            current['content'] += line + '\n' 

    if current['level'] > 0 or current['content'].strip():
        sections.append(current)
    return sections

def build_reduced_document(sections):
    labels = [classify_heading(s["title"]) if s["level"] > 0 else None for s in sections]

    exec_i = next((i for i, l in enumerate(labels) if l == "executive_summary"), None)
    if exec_i is None:
        return {"status": "no_executive_summary_found",
                "outline": [(s["level"], s["title"]) for s in sections], "text": None}

    # end: first references/appendices heading after the executive summary
    end_i = next((i for i in range(exec_i + 1, len(sections))
                  if labels[i] in {"references", "appendices"}), len(sections))

    kept, removed = [], []
    active_level = None   # level of the kept section we are currently inside

    for i in range(exec_i, end_i):
        s = sections[i]

        # a heading at the same or higher level closes the current kept section
        if active_level is not None and s["level"] <= active_level:
            active_level = None

        # a whitelisted heading opens a new kept section
        if active_level is None and labels[i] in KEEP_LABELS:
            active_level = s["level"]

        if active_level is not None:
            kept.append(f"{'#' * s['level']} {s['title']}\n{s['content']}")
        else:
            removed.append(s["title"])

    return {
        "status": "ok",
        "text": "\n".join(kept).strip(),
        "removed_before_exec_summary": [s["title"] for s in sections[:exec_i]],
        "removed_between": removed,
        "removed_tail": [s["title"] for s in sections[end_i:]],
    }
