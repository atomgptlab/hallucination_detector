"""
LEGACY PROTOTYPE: Crossref + arXiv reference checker.

This early prototype is kept for reference only. It is NOT the method
evaluated in the paper; see ``hallucination.py`` for that.

Given a PDF research paper, this script:

1) Extracts the references section text from the PDF.
2) Splits it into individual reference entries.
3) For each reference entry:
   - Queries Crossref (REST API) and arXiv (API)
   - Picks the best candidate match
   - Computes a 0–100% match score (title + year + author hints)
4) Produces:
   - A JSON report
   - A CSV report
   - A matplotlib “colorbar” style figure (0–100%)

Notes / reality check:
- Reference parsing from PDFs is inherently messy. This does a solid
  heuristic job without external services like GROBID.
- Crossref works well for DOI/journal references; arXiv for preprints.
- If you want “production-grade” reference parsing, integrate GROBID later.

Dependencies:
  pip install requests matplotlib pandas
  pip install pdfplumber   # recommended PDF text extraction
  # Optional (improves title similarity):
  pip install rapidfuzz

Usage:
  python legacy/crossref_arxiv_prototype.py path/to/paper.pdf --outdir out

Also includes a no-PDF demo:
  python legacy/crossref_arxiv_prototype.py --demo
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import requests

# Optional: better/faster similarity
try:
    from rapidfuzz.fuzz import ratio as fuzz_ratio  # type: ignore
except Exception:
    fuzz_ratio = None

# Optional: if user wants plotting
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore


CROSSREF_API = "https://api.crossref.org/works"
ARXIV_API = "http://export.arxiv.org/api/query"


# ----------------------------
# Utilities: normalization & similarity
# ----------------------------


def normalize_text(s: str) -> str:
    s = s.strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    # keep letters/numbers/basic punctuation for titles
    s = re.sub(r"[^a-z0-9\s\-\:\,\.\(\)\/]", "", s)
    return s.strip()


def title_similarity(a: str, b: str) -> float:
    """
    Returns similarity in [0, 1].
    Uses rapidfuzz if available; else difflib.
    """
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if not a_n or not b_n:
        return 0.0

    if fuzz_ratio is not None:
        return float(fuzz_ratio(a_n, b_n)) / 100.0

    import difflib

    return difflib.SequenceMatcher(None, a_n, b_n).ratio()


def safe_int_year(s: str) -> Optional[int]:
    m = re.search(r"\b(19\d{2}|20\d{2})\b", s)
    if not m:
        return None
    y = int(m.group(1))
    if 1900 <= y <= 2100:
        return y
    return None


def extract_doi(ref: str) -> Optional[str]:
    # Common DOI patterns
    m = re.search(r"\b10\.\d{4,9}\/[-._;()\/:A-Z0-9]+\b", ref, flags=re.I)
    if m:
        return m.group(0).rstrip(".")
    return None


def extract_arxiv_id(ref: str) -> Optional[str]:
    # New style: 1707.08567 ; Old style: hep-th/9901001
    m = re.search(
        r"\barxiv:\s*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)\b", ref, flags=re.I
    )
    if m:
        return m.group(1)
    m = re.search(r"\b([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)\b", ref)
    if m:
        return m.group(1)
    m = re.search(r"\b([a-z\-]+\/\d{7}(?:v\d+)?)\b", ref, flags=re.I)
    if m:
        return m.group(1)
    return None


def guess_title_from_reference(ref: str) -> Optional[str]:
    """
    Heuristic title guess from a raw reference string.

    Strategies:
      1) If something is in quotes “...”, "...", or ‘...’: treat as title.
      2) Otherwise, remove DOI/arXiv, then try to capture the chunk
         between authors and venue/year.
         Common pattern: Authors. Title. Journal ...
    """
    ref_clean = ref.strip()

    # Remove DOI/arxiv tokens for cleaner parsing
    ref_clean = re.sub(r"\barxiv:\s*\S+", "", ref_clean, flags=re.I)
    ref_clean = re.sub(r"\bdoi:\s*\S+", "", ref_clean, flags=re.I)

    # 1) Quoted title
    for pattern in [
        r"“([^”]{8,})”",
        r"\"([^\"]{8,})\"",
        r"‘([^’]{8,})’",
        r"'([^']{8,})'",
    ]:
        m = re.search(pattern, ref_clean)
        if m:
            return m.group(1).strip(" .,")

    # 2) Split by periods and look for a plausible title segment
    # Example: "Smith J., Doe A. Deep learning for X. Nature 2020..."
    parts = [p.strip() for p in ref_clean.split(".") if p.strip()]
    if len(parts) >= 2:
        # Often: [authors] [title] [venue...]
        # Choose a "title-like" part: not too short and free of obvious
        # venue tokens
        candidates = []
        for p in parts[
            1:3
        ]:  # typically the next one or two chunks after authors
            if 8 <= len(p) <= 240 and not re.search(
                r"\b(nature|science|proc|journal|phys|rev|letters|conf"
                r"|vol|no|pp)\b",
                p,
                flags=re.I,
            ):
                candidates.append(p)
        if candidates:
            # pick the longest candidate
            return max(candidates, key=len).strip(" ,;")

    # 3) Fallback: find something before the year that looks like a title
    y = safe_int_year(ref_clean)
    if y:
        before_year = ref_clean.split(str(y))[0]
        # Remove leading author list-ish text (many commas)
        before_year = re.sub(r"^[^\.]{0,220}\.\s*", "", before_year)
        before_year = before_year.strip(" .,:;")
        if 8 <= len(before_year) <= 240:
            return before_year

    return None


def guess_first_author_lastname(ref: str) -> Optional[str]:
    """
    Very rough: take the first token before a comma, or the first
    surname-like token.
    """
    s = ref.strip()
    # "Smith, J." or "Smith J."
    m = re.match(r"^\s*([A-Z][A-Za-z\-\']{1,40})\s*,", s)
    if m:
        return m.group(1)
    m = re.match(r"^\s*([A-Z][A-Za-z\-\']{1,40})\s+[A-Z]\.", s)
    if m:
        return m.group(1)
    # If it starts with [1] Smith ...
    m = re.match(r"^\s*\[\d+\]\s*([A-Z][A-Za-z\-\']{1,40})", s)
    if m:
        return m.group(1)
    return None


# ----------------------------
# PDF extraction & reference splitting
# ----------------------------


def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text from PDF using pdfplumber if available; else try PyMuPDF.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    # Try pdfplumber
    try:
        import pdfplumber  # type: ignore

        text_chunks = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                if txt:
                    text_chunks.append(txt)
        return "\n".join(text_chunks)
    except Exception:
        pass

    # Fallback to PyMuPDF
    try:
        import fitz  # type: ignore

        doc = fitz.open(pdf_path)
        text_chunks = []
        for page in doc:
            text_chunks.append(page.get_text("text"))
        doc.close()
        return "\n".join(text_chunks)
    except Exception as e:
        raise RuntimeError(
            "Could not extract text. Install pdfplumber (recommended) "
            "or pymupdf.\n"
            "pip install pdfplumber\n"
            "or\n"
            "pip install pymupdf"
        ) from e


def find_references_section(full_text: str) -> str:
    """
    Heuristic: locate a 'References' / 'Bibliography' / 'Works Cited'
    header and return the text after it.
    """
    text = full_text

    # Common headers
    headers = [
        r"\nreferences\s*\n",
        r"\nbibliography\s*\n",
        r"\nworks cited\s*\n",
        r"\nreference\s*\n",
    ]

    idx = None
    for h in headers:
        m = re.search(h, text, flags=re.I)
        if m:
            idx = m.end()
            break

    # If we can't find a header, return full text (still might parse refs poorly)
    if idx is None:
        return text

    return text[idx:]


def split_references(ref_text: str) -> List[str]:
    """
    Split the references section into individual entries.

    Handles common formats:
      - [1] ...  [2] ...
      - 1. ...   2. ...
      - (1) ...
      - Newline-separated blocks

    Returns non-empty entries.
    """
    t = ref_text.strip()

    # Normalize line breaks
    t = re.sub(r"\r\n", "\n", t)

    # If bracketed numbering exists, use it
    if re.search(r"\n\s*\[\d+\]\s*", t):
        parts = re.split(r"\n\s*(?=\[\d+\]\s*)", t)
        refs = [p.strip() for p in parts if p.strip()]
        # Filter out tiny fragments
        return [r for r in refs if len(r) > 30]

    # If numeric dot numbering exists
    if re.search(r"\n\s*\d+\.\s+", t):
        parts = re.split(r"\n\s*(?=\d+\.\s+)", t)
        refs = [p.strip() for p in parts if p.strip()]
        return [r for r in refs if len(r) > 30]

    # If parenthesized numbering exists
    if re.search(r"\n\s*\(\d+\)\s+", t):
        parts = re.split(r"\n\s*(?=\(\d+\)\s+)", t)
        refs = [p.strip() for p in parts if p.strip()]
        return [r for r in refs if len(r) > 30]

    # Fallback: split by blank lines (common in some PDFs)
    parts = re.split(r"\n\s*\n", t)
    refs = [p.strip() for p in parts if p.strip()]
    # Some PDFs wrap each reference across multiple lines without blank lines:
    # attempt to merge short lines into larger blocks by heuristic.
    merged = []
    buf = ""
    for r in refs:
        if len(buf) == 0:
            buf = r
        elif len(r) < 80 and not re.match(r"^\s*(19\d{2}|20\d{2})\b", r):
            buf += " " + r
        else:
            merged.append(buf.strip())
            buf = r
    if buf.strip():
        merged.append(buf.strip())
    return [r for r in merged if len(r) > 30]


# ----------------------------
# API querying
# ----------------------------


def crossref_lookup(
    ref: str, timeout: float = 20.0
) -> Optional[Dict[str, Any]]:
    """
    Query Crossref for the best match.
    Strategy:
      - If DOI found: direct query /works/{doi}
      - else: use /works?query.bibliographic=...&rows=1
    """
    doi = extract_doi(ref)
    headers = {
        "User-Agent": "HallucinationFinder/0.1 (mailto:you@example.com)"
    }

    try:
        if doi:
            url = f"{CROSSREF_API}/{doi}"
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200:
                msg = r.json().get("message")
                if msg:
                    return msg
            return None

        params = {"query.bibliographic": ref, "rows": 1}
        r = requests.get(
            CROSSREF_API, params=params, headers=headers, timeout=timeout
        )
        if r.status_code != 200:
            return None
        items = r.json().get("message", {}).get("items", [])
        if not items:
            return None
        return items[0]
    except Exception:
        return None


def arxiv_lookup(ref: str, timeout: float = 20.0) -> Optional[Dict[str, Any]]:
    """
    Query arXiv for the best match.
    Strategy:
      - If arXiv id found: search by id
      - else: guess title and search all fields with that title

    Returns a dict with keys: title, authors(list), year(optional), id(optional), published(optional), link(optional)
    """
    arxiv_id = extract_arxiv_id(ref)
    guessed_title = guess_title_from_reference(ref)

    # arXiv API is Atom XML; we'll parse without extra dependencies
    def parse_atom(xml: str) -> Optional[Dict[str, Any]]:
        # Very lightweight parsing (regex). For robust parsing you could use feedparser.
        # We'll extract first <entry> ... </entry>.
        m = re.search(r"<entry>(.*?)</entry>", xml, flags=re.S)
        if not m:
            return None
        entry = m.group(1)

        def tag(name: str) -> Optional[str]:
            mm = re.search(rf"<{name}>(.*?)</{name}>", entry, flags=re.S)
            if mm:
                return re.sub(r"\s+", " ", mm.group(1)).strip()
            return None

        title = tag("title")
        published = tag("published")
        # Authors appear as <name> inside <author>
        authors = re.findall(
            r"<author>\s*<name>(.*?)</name>", entry, flags=re.S
        )
        authors = [re.sub(r"\s+", " ", a).strip() for a in authors]

        # id is a URL
        mm = re.search(r"<id>(.*?)</id>", entry, flags=re.S)
        _id = mm.group(1).strip() if mm else None

        year = None
        if published:
            ym = re.match(r"(\d{4})-\d{2}-\d{2}", published)
            if ym:
                year = int(ym.group(1))

        return {
            "title": title,
            "authors": authors,
            "published": published,
            "year": year,
            "id": _id,
            "link": _id,
        }

    try:
        if arxiv_id:
            # Search by id in all fields; arXiv doesn't offer a direct JSON for ID in this endpoint
            q = f"id:{arxiv_id}"
        else:
            if not guessed_title:
                return None
            # Quote for phrase match-ish; arXiv uses its own query syntax
            q = f'all:"{guessed_title}"'

        params = {
            "search_query": q,
            "start": 0,
            "max_results": 1,
        }
        r = requests.get(ARXIV_API, params=params, timeout=timeout)
        if r.status_code != 200:
            return None
        return parse_atom(r.text)
    except Exception:
        return None


# ----------------------------
# Scoring
# ----------------------------


@dataclass
class MatchResult:
    ref_raw: str
    ref_index: int

    guessed_title: Optional[str]
    guessed_year: Optional[int]
    guessed_first_author: Optional[str]

    crossref_title: Optional[str]
    crossref_year: Optional[int]
    crossref_doi: Optional[str]
    crossref_first_author: Optional[str]
    crossref_score: Optional[float]

    arxiv_title: Optional[str]
    arxiv_year: Optional[int]
    arxiv_id: Optional[str]
    arxiv_first_author: Optional[str]
    arxiv_score: Optional[float]

    best_source: Optional[str]
    best_score: float
    best_title: Optional[str]
    best_id: Optional[str]


def crossref_fields(
    item: Dict[str, Any],
) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[str]]:
    title = None
    if isinstance(item.get("title"), list) and item.get("title"):
        title = item["title"][0]
    elif isinstance(item.get("title"), str):
        title = item["title"]

    year = None
    # Crossref often has issued.date-parts [[YYYY, MM, DD]]
    issued = item.get("issued", {}).get("date-parts")
    if (
        issued
        and isinstance(issued, list)
        and issued
        and isinstance(issued[0], list)
        and issued[0]
    ):
        try:
            year = int(issued[0][0])
        except Exception:
            year = None

    doi = item.get("DOI") or item.get("doi")

    first_author = None
    authors = item.get("author")
    if isinstance(authors, list) and authors:
        # author is dict with family/given
        fam = authors[0].get("family")
        if fam:
            first_author = str(fam)

    return title, year, doi, first_author


def score_candidate(
    ref_title: Optional[str],
    ref_year: Optional[int],
    ref_first_author: Optional[str],
    cand_title: Optional[str],
    cand_year: Optional[int],
    cand_first_author: Optional[str],
) -> float:
    """
    Score in [0,1] based on title similarity + year match + author match.
    """
    # Weights (tune as you like)
    w_title = 0.75
    w_year = 0.15
    w_author = 0.10

    s_title = title_similarity(ref_title or "", cand_title or "")

    s_year = 0.0
    if ref_year and cand_year:
        # exact match best, within 1 year partial
        if ref_year == cand_year:
            s_year = 1.0
        elif abs(ref_year - cand_year) == 1:
            s_year = 0.5
        else:
            s_year = 0.0
    elif ref_year is None or cand_year is None:
        # unknown year: neutral-ish but not full credit
        s_year = 0.35

    s_author = 0.0
    if ref_first_author and cand_first_author:
        s_author = (
            1.0
            if normalize_text(ref_first_author)
            == normalize_text(cand_first_author)
            else 0.0
        )
    elif ref_first_author is None or cand_first_author is None:
        s_author = 0.35

    score = (w_title * s_title) + (w_year * s_year) + (w_author * s_author)
    # clamp
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    return score


# ----------------------------
# Main pipeline
# ----------------------------


def analyze_pdf(
    pdf_path: str,
    polite_delay_s: float = 0.25,
    max_refs: Optional[int] = None,
) -> List[MatchResult]:
    full_text = extract_pdf_text(pdf_path)
    refs_text = find_references_section(full_text)
    refs = split_references(refs_text)

    if max_refs is not None:
        refs = refs[:max_refs]

    results: List[MatchResult] = []

    for i, ref in enumerate(refs, start=1):
        guessed_title = guess_title_from_reference(ref)
        guessed_year = safe_int_year(ref)
        guessed_first_author = guess_first_author_lastname(ref)

        # Query Crossref
        cr_item = crossref_lookup(ref)
        cr_title = cr_year = cr_doi = cr_first_author = None
        cr_score = None
        if cr_item:
            cr_title, cr_year, cr_doi, cr_first_author = crossref_fields(
                cr_item
            )
            cr_score = score_candidate(
                guessed_title,
                guessed_year,
                guessed_first_author,
                cr_title,
                cr_year,
                cr_first_author,
            )

        time.sleep(polite_delay_s)

        # Query arXiv
        ax_item = arxiv_lookup(ref)
        ax_title = ax_year = ax_id = ax_first_author = None
        ax_score = None
        if ax_item:
            ax_title = ax_item.get("title")
            ax_year = ax_item.get("year")
            ax_id = ax_item.get("id")
            authors = ax_item.get("authors") or []
            if authors and isinstance(authors, list):
                # arXiv author list is typically "First Last" - take last token
                first = str(authors[0])
                ax_first_author = first.split()[-1] if first.split() else first
            ax_score = score_candidate(
                guessed_title,
                guessed_year,
                guessed_first_author,
                ax_title,
                ax_year,
                ax_first_author,
            )

        # Choose best
        best_source = None
        best_score = 0.0
        best_title = None
        best_id = None

        if cr_score is not None and cr_score >= best_score:
            best_score = cr_score
            best_source = "crossref"
            best_title = cr_title
            best_id = cr_doi

        if ax_score is not None and ax_score >= best_score:
            best_score = ax_score
            best_source = "arxiv"
            best_title = ax_title
            best_id = ax_id

        results.append(
            MatchResult(
                ref_raw=ref,
                ref_index=i,
                guessed_title=guessed_title,
                guessed_year=guessed_year,
                guessed_first_author=guessed_first_author,
                crossref_title=cr_title,
                crossref_year=cr_year,
                crossref_doi=cr_doi,
                crossref_first_author=cr_first_author,
                crossref_score=cr_score,
                arxiv_title=ax_title,
                arxiv_year=ax_year,
                arxiv_id=ax_id,
                arxiv_first_author=ax_first_author,
                arxiv_score=ax_score,
                best_source=best_source,
                best_score=best_score,
                best_title=best_title,
                best_id=best_id,
            )
        )

        time.sleep(polite_delay_s)

    return results


def save_outputs(
    results: List[MatchResult], outdir: str
) -> Tuple[str, str, str]:
    os.makedirs(outdir, exist_ok=True)

    # JSON
    json_path = os.path.join(outdir, "hallucination_finder_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(r) for r in results], f, indent=2, ensure_ascii=False
        )

    # CSV
    rows = []
    for r in results:
        rows.append(
            {
                "ref_index": r.ref_index,
                "best_score_pct": round(100.0 * r.best_score, 2),
                "best_source": r.best_source,
                "best_title": r.best_title,
                "best_id": r.best_id,
                "guessed_title": r.guessed_title,
                "guessed_year": r.guessed_year,
                "guessed_first_author": r.guessed_first_author,
                "crossref_score_pct": (
                    None
                    if r.crossref_score is None
                    else round(100.0 * r.crossref_score, 2)
                ),
                "arxiv_score_pct": (
                    None
                    if r.arxiv_score is None
                    else round(100.0 * r.arxiv_score, 2)
                ),
                "ref_raw": r.ref_raw,
            }
        )
    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "hallucination_finder_report.csv")
    df.to_csv(csv_path, index=False)

    # Plot: “colorbar” style result for each reference (horizontal heat strip)
    fig_path = os.path.join(outdir, "hallucination_finder_colorbar.png")
    plot_colorbar(results, fig_path)

    return json_path, csv_path, fig_path


def plot_colorbar(results: List[MatchResult], fig_path: str) -> None:
    """
    Creates a simple heatmap strip: each reference is a row, colored by score (0..100).
    """
    scores = [r.best_score for r in results]
    if not scores:
        scores = [0.0]

    data = [[s] for s in scores]  # Nx1 heatmap
    fig, ax = plt.subplots(figsize=(6, max(2, 0.25 * len(scores))))
    im = ax.imshow(data, aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([str(r.ref_index) for r in results])
    ax.set_xticks([0])
    ax.set_xticklabels(["Score"])

    ax.set_ylabel("Reference #")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Match score (0–1)")

    ax.set_title("Hallucination Finder: reference match scores")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)


# ----------------------------
# Demo without PDFs
# ----------------------------


def demo_run() -> None:
    demo_refs = [
        "[1] A. Vaswani, N. Shazeer, N. Parmar, et al. Attention Is All You Need. NeurIPS, 2017.",
        "[2] K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. CVPR, 2016. arXiv:1512.03385",
        "[3] Totally Fake Author. A Made Up Paper Title That Does Not Exist. Journal of Imaginary Results, 2099.",
        "[4] J. Jumper et al. Highly accurate protein structure prediction with AlphaFold. Nature (2021). doi:10.1038/s41586-021-03819-2",
    ]

    # Treat these as already-split references and run API checks.
    results: List[MatchResult] = []
    for i, ref in enumerate(demo_refs, start=1):
        guessed_title = guess_title_from_reference(ref)
        guessed_year = safe_int_year(ref)
        guessed_first_author = guess_first_author_lastname(ref)

        cr_item = crossref_lookup(ref)
        cr_title = cr_year = cr_doi = cr_first_author = None
        cr_score = None
        if cr_item:
            cr_title, cr_year, cr_doi, cr_first_author = crossref_fields(
                cr_item
            )
            cr_score = score_candidate(
                guessed_title,
                guessed_year,
                guessed_first_author,
                cr_title,
                cr_year,
                cr_first_author,
            )

        time.sleep(0.25)

        ax_item = arxiv_lookup(ref)
        ax_title = ax_year = ax_id = ax_first_author = None
        ax_score = None
        if ax_item:
            ax_title = ax_item.get("title")
            ax_year = ax_item.get("year")
            ax_id = ax_item.get("id")
            authors = ax_item.get("authors") or []
            if authors and isinstance(authors, list) and authors:
                first = str(authors[0])
                ax_first_author = first.split()[-1] if first.split() else first
            ax_score = score_candidate(
                guessed_title,
                guessed_year,
                guessed_first_author,
                ax_title,
                ax_year,
                ax_first_author,
            )

        best_source = None
        best_score = 0.0
        best_title = None
        best_id = None
        if cr_score is not None and cr_score >= best_score:
            best_score = cr_score
            best_source = "crossref"
            best_title = cr_title
            best_id = cr_doi
        if ax_score is not None and ax_score >= best_score:
            best_score = ax_score
            best_source = "arxiv"
            best_title = ax_title
            best_id = ax_id

        results.append(
            MatchResult(
                ref_raw=ref,
                ref_index=i,
                guessed_title=guessed_title,
                guessed_year=guessed_year,
                guessed_first_author=guessed_first_author,
                crossref_title=cr_title,
                crossref_year=cr_year,
                crossref_doi=cr_doi,
                crossref_first_author=cr_first_author,
                crossref_score=cr_score,
                arxiv_title=ax_title,
                arxiv_year=ax_year,
                arxiv_id=ax_id,
                arxiv_first_author=ax_first_author,
                arxiv_score=ax_score,
                best_source=best_source,
                best_score=best_score,
                best_title=best_title,
                best_id=best_id,
            )
        )

    # Print a small table
    rows = []
    for r in results:
        rows.append(
            {
                "ref_index": r.ref_index,
                "best_score_%": round(100 * r.best_score, 1),
                "best_source": r.best_source,
                "guessed_title": (r.guessed_title or "")[:70],
                "best_title": (r.best_title or "")[:70],
            }
        )
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


# ----------------------------
# CLI
# ----------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Hallucination Finder: verify references in a PDF via Crossref + arXiv."
    )
    parser.add_argument("pdf", nargs="?", help="Path to PDF file.")
    parser.add_argument(
        "--outdir",
        default="out_hallucination_finder",
        help="Output directory.",
    )
    parser.add_argument(
        "--max-refs",
        type=int,
        default=None,
        help="Limit number of references analyzed.",
    )
    parser.add_argument(
        "--polite-delay",
        type=float,
        default=0.25,
        help="Delay between API calls (seconds).",
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run a demo without a PDF."
    )
    args = parser.parse_args()

    if args.demo:
        demo_run()
        return

    if not args.pdf:
        raise SystemExit("Provide a PDF path, or use --demo.")

    results = analyze_pdf(
        args.pdf, polite_delay_s=args.polite_delay, max_refs=args.max_refs
    )
    json_path, csv_path, fig_path = save_outputs(results, args.outdir)

    # Print summary
    scores_pct = [100.0 * r.best_score for r in results] if results else []
    avg = sum(scores_pct) / len(scores_pct) if scores_pct else 0.0
    print(f"Analyzed {len(results)} references.")
    print(f"Average best score: {avg:.2f}%")
    print(
        f"Saved:\n  JSON: {json_path}\n  CSV:  {csv_path}\n  FIG:  {fig_path}"
    )

    # Show top likely hallucinations (low scores)
    low = sorted(results, key=lambda r: r.best_score)[: min(10, len(results))]
    print("\nLowest-score references (potential issues):")
    for r in low:
        print(
            f"  #{r.ref_index:03d}  {100.0*r.best_score:6.2f}%  {r.guessed_title or '(no title guessed)'}"
        )


if __name__ == "__main__":
    main()
