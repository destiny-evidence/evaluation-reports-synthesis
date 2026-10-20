import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz          
from utils import parse_md_sections

# ---------- synonyms ----------
EN_TERMS = [
    "executive summary", "exec summary", "summary for policymakers",
    "summary for decision makers", "management summary", "summary",
    "abstract", "overview", "synopsis", "key findings", "main findings",
    "key messages", "highlights", "at a glance", "in brief", "brief summary",
    "short summary", "report summary", "project summary", "résumé",
    "executive overview", "executive brief", "key takeaways", "tl;dr",
]

# Review these translations; extend for the languages you actually have.
TRANSLATED_TERMS = {
    "de": ["zusammenfassung", "kurzfassung", "management summary", "kurzzusammenfassung",
           "überblick", "zusammenfassung für entscheidungsträger", "executive summary"],
    "fr": ["résumé exécutif", "résumé", "synthèse", "sommaire exécutif", "résumé analytique",
           "résumé de gestion", "points clés", "aperçu", "en bref"],
    "es": ["resumen ejecutivo", "resumen", "síntesis", "sumario ejecutivo",
           "resumen general", "puntos clave", "panorama general"],
    "it": ["sintesi", "riassunto", "sommario esecutivo", "sintesi esecutiva",
           "riepilogo", "panoramica", "punti chiave"],
    "pt": ["resumo executivo", "resumo", "sumário executivo", "sinopse",
           "síntese", "visão geral", "pontos principais"],
    "nl": ["samenvatting", "managementsamenvatting", "kort overzicht", "overzicht",
           "beknopte samenvatting", "kernpunten"],
    "sv": ["sammanfattning", "sammandrag", "översikt", "huvudpunkter"],
    "da": ["resumé", "sammenfatning", "oversigt", "hovedpunkter", "kort fortalt"],
    "no": ["sammendrag", "oppsummering", "oversikt", "hovedpunkter"],
    "fi": ["tiivistelmä", "yhteenveto", "johdon tiivistelmä", "katsaus"],
    "pl": ["streszczenie", "podsumowanie", "streszczenie wykonawcze", "przegląd", "najważniejsze wnioski"],
    "cs": ["shrnutí", "souhrn", "manažerské shrnutí", "přehled"],
    "ro": ["rezumat", "sumar executiv", "rezumat executiv", "prezentare generală"],
    "hu": ["összefoglaló", "vezetői összefoglaló", "áttekintés"],
    "el": ["περίληψη", "εκτελεστική περίληψη", "σύνοψη", "επισκόπηση"],
    "ru": ["резюме", "краткое содержание", "обзор", "аннотация", "краткий обзор", "основные выводы"],
    "tr": ["yönetici özeti", "özet", "genel bakış", "kısa özet", "ana bulgular"],
}

# ---------- page counting ----------
# ADAPT THIS to however pages are marked in your .md files.
PAGE_MARKER = re.compile(
    r"<!--\s*page\s*\d+\s*-->|^\s*#*\s*page\s+\d+\s*$|\f|^-{3,}\s*page\s*\d+",
    re.IGNORECASE | re.MULTILINE,
)


def normalize(s):
    s = unicodedata.normalize("NFKD", (s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))   # strip accents
    s = re.sub(r"[^\w\s]", " ", s)                              # strip punctuation
    s = re.sub(r"^\s*(\d+|[ivxlc]+)\s+", "", s)                 # leading numbering
    return re.sub(r"\s+", " ", s).strip()


def get_terms(lang):
    terms = list(EN_TERMS)
    if lang and lang not in ("en", "unknown"):
        terms += TRANSLATED_TERMS.get(lang, [])
    return sorted({normalize(t) for t in terms}, key=len, reverse=True)


def find_exec_summary(titles, terms, threshold=85):
    """Return (match_type, matched_title, matched_term) or (None, None, None)."""
    fuzzy_hit = None
    for title in titles:
        t = normalize(title)
        if not t:
            continue
        for term in terms:
            if term in t:
                return "exact", title, term
            if fuzzy_hit is None and len(term) >= 6 and fuzz.partial_ratio(term, t) >= threshold:
                fuzzy_hit = ("fuzzy", title, term)
    return fuzzy_hit or (None, None, None)


def describe_documents(folder, lang_json="reportlanguage.json", threshold=75):
    folder = Path(folder)

    with open(lang_json, encoding="utf-8") as f:
        lang_by_file = {r["file"]: r["language"] for r in json.load(f)}

    rows = []
    for path in sorted(folder.rglob("*.md")):
        try:
            sections = parse_md_sections(path)
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"Could not read {path}: {e}")
            continue

        lang = lang_by_file.get(path.name, "unknown")
        match_type, title, term = find_exec_summary(
            [s.get("title", "") for s in sections], get_terms(lang), threshold
        )
        pages = len(PAGE_MARKER.findall(raw))

        rows.append({
            "doc_id": re.sub(r"_\d+$", "", path.stem),
            "file": path.name,
            "language": lang,
            "n_sections": len(sections),
            "n_pages": pages if pages else None,    # None = no page markers found
            "has_exec_summary": match_type is not None,
            "match_type": match_type,
            "matched_title": title,
            "matched_term": term,
        })

    per_file = pd.DataFrame(rows)

    per_doc = per_file.groupby("doc_id").agg(
        n_files=("file", "count"),
        total_sections=("n_sections", "sum"),
        total_pages=("n_pages", "sum"),
        files_with_exec_summary=("has_exec_summary", "sum"),
        has_exec_summary=("has_exec_summary", "any"),
    ).reset_index()

    return per_file, per_doc


# ---------- usage ----------
FOLDER = "/home/lt1412/Desktop/evaluation-reports-synthesis/data/"
per_file, per_doc = describe_documents(folder=FOLDER)

print(per_doc["n_files"].describe())
print(per_doc["n_files"].value_counts().sort_index())      # how many docs have 1, 2, 3... files
print(per_doc["has_exec_summary"].value_counts(dropna=False))
print(per_file["match_type"].value_counts(dropna=False))   # exact vs fuzzy vs none