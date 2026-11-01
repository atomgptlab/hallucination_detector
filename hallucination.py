"""
hallucination.py: reference hallucination detector.

Extracts the references from a manuscript (text or PDF), retrieves the
closest real papers for each one from Semantic Scholar, and scores the
agreement on title, authors and venue to label every citation as
``verified``, ``partial`` or ``likely hallucinated``.

Pipeline
--------
1. Extraction: an OpenAI-compatible LLM parses the reference list into
   structured fields (title, authors, journal, year), with deterministic
   decoding (temperature 0). If no LLM is configured or the call fails, a
   regular-expression parser is used as a fallback.
2. Retrieval: the cleaned title is sent to the Semantic Scholar paper-search
   API and the top-k candidates are returned.
3. Scoring: every candidate is scored with three similarities on a 0-100
   scale, combined into a weighted composite; the best candidate wins.
4. Classification: the composite is thresholded into three status bands.

Steps 3 and 4 are fully deterministic and are implemented in
:func:`verify_reference`; see the README for the exact definitions.

Usage
-----
    python hallucination.py                  # built-in sample references
    python hallucination.py paper.pdf        # check a PDF
    python hallucination.py refs.txt -o out.json
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import time

import httpx

log = logging.getLogger("hallucination")

# ---------------------------------------------------------------------------
# Configuration (environment variables; nothing is hard-coded)
# ---------------------------------------------------------------------------

# Semantic Scholar API key (optional, raises the rate limit).
# https://www.semanticscholar.org/product/api
SS_API_KEY = os.environ.get("SS_API_KEY", "")
SS_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Any OpenAI-compatible chat-completions endpoint used for extraction.
LLM_URL = os.environ.get("LLM_URL", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "60"))

# Scoring defaults (change them through the function arguments).
WEIGHTS = (0.60, 0.25, 0.15)  # title, authors, journal
VERIFIED_THRESHOLD = 80.0
PARTIAL_THRESHOLD = 60.0

# Two versions of the scoring rule are supported:
#   "v2" (default, current): a missing author/journal field is dropped and
#        the remaining weights are renormalized; an author matches on
#        surname AND first initial.
#   "v1" (original, until July 2026): a missing field is replaced by the
#        title score; an author matches on surname only.
SCORING_VERSIONS = ("v1", "v2")
DEFAULT_SCORING = "v2"

_CACHE: dict = {}

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    from rapidfuzz import fuzz as _fuzz

    _HAS_FUZZ = True
except ImportError:
    _HAS_FUZZ = False

try:
    import pdfplumber

    _HAS_PDF = True
except ImportError:
    _HAS_PDF = False

try:
    from openai import OpenAI

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


_LLM_PROMPT = """\
You are a precise academic reference extractor.
Given text with a References/Bibliography section, extract each reference.

Return ONLY a JSON array of objects with keys:
- "title": string
- "authors": list of strings
- "journal": string (empty if unknown)
- "year": string (empty if unknown)

No commentary. If no references found, return []
"""


# ---------------------------------------------------------------------------
# Text helpers and similarities
# ---------------------------------------------------------------------------


def extract_pdf_text(source) -> str:
    """Return the text of a PDF given a file path or raw bytes."""
    if not _HAS_PDF:
        raise RuntimeError("pdfplumber is not installed")
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    with pdfplumber.open(source) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)


def fuzzy_score(a: str, b: str) -> float:
    """Similarity in [0, 100] used for titles and venues.

    Both strings are lower-cased and compared with RapidFuzz
    ``token_set_ratio``. No other normalization is applied (no Unicode
    folding, no punctuation stripping, no venue-abbreviation expansion).
    Without RapidFuzz, falls back to word-level Jaccard similarity x 100.
    """
    if not a or not b:
        return 0.0
    if _HAS_FUZZ:
        return _fuzz.token_set_ratio(a.lower(), b.lower())
    s1, s2 = set(a.lower().split()), set(b.lower().split())
    if not s1 or not s2:
        return 0.0
    return 100.0 * len(s1 & s2) / len(s1 | s2)


def author_score_v1(cited: list[str], candidate: list[str]) -> float:
    """v1: percentage of cited authors whose surname is in the candidate.

    The surname of a cited author is the last whitespace token of the name,
    lower-cased. It matches if it equals any whitespace token of any
    lower-cased candidate author name. Returns -1 if either list is empty
    (the field is then treated as missing by :func:`composite_score`).
    """
    if not cited or not candidate:
        return -1.0
    names = [n.lower() for n in candidate if n]
    matched = sum(
        1
        for ea in cited
        if ea.strip()
        and any(ea.lower().split()[-1] in n.split() for n in names)
    )
    return 100.0 * matched / len(cited)


def name_key(name: str) -> tuple[str, str]:
    """(surname, first initial): lower-cased, periods/commas removed."""
    parts = re.sub(r"[.,]", " ", name.lower()).split()
    if not parts:
        return ("", "")
    return (parts[-1], parts[0][0])


def author_score_v2(cited: list[str], candidate: list[str]) -> float:
    """v2: percentage of cited authors matching on surname AND initial.

    Both names are reduced with :func:`name_key`. A cited author matches if
    some candidate author has the same surname and, when both initials are
    known, the same first initial. Requiring the initial stops a fabricated
    reference from inheriting credit from a real paper by an author with a
    common surname. Returns -1 if either list has no usable name.
    """
    keys = [name_key(n) for n in candidate if n]
    keys = [k for k in keys if k[0]]
    if not cited or not keys:
        return -1.0
    matched = 0
    for ea in cited:
        if not ea.strip():
            continue
        sur, init = name_key(ea)
        if sur and any(
            sur == s2 and (not init or not i2 or init == i2) for s2, i2 in keys
        ):
            matched += 1
    return 100.0 * matched / len(cited)


def author_score(
    cited: list[str], candidate: list[str], scoring: str = DEFAULT_SCORING
) -> float:
    """Author similarity in [0, 100], or -1 if missing (see versions)."""
    if scoring == "v1":
        return author_score_v1(cited, candidate)
    return author_score_v2(cited, candidate)


def composite_score(
    title: float,
    authors: float,
    journal: float,
    weights: tuple[float, float, float] = WEIGHTS,
    scoring: str = DEFAULT_SCORING,
) -> float:
    """Weighted composite of the field scores (negative = missing).

    v2: missing fields are dropped and the weights renormalized.
    v1: a missing field takes the title score.
    """
    wt, wa, wj = weights
    if scoring == "v1":
        comp = wt * title
        comp += wa * (authors if authors >= 0 else title)
        comp += wj * (journal if journal >= 0 else title)
        return comp
    parts = [(title, wt)]
    if authors >= 0:
        parts.append((authors, wa))
    if journal >= 0:
        parts.append((journal, wj))
    wsum = sum(w for _, w in parts)
    return sum(x * w for x, w in parts) / wsum if wsum else 0.0


def classify(
    score: float,
    verified: float = VERIFIED_THRESHOLD,
    partial: float = PARTIAL_THRESHOLD,
) -> str:
    if score >= verified:
        return "verified"
    if score >= partial:
        return "partial"
    return "likely hallucinated"


def clean_query(ref: str) -> str:
    ref = re.sub(r"^\s*[\[\(]?\d+[\]\)\.]\s*", "", ref)
    ref = re.sub(r"https?://\S+", "", ref)
    ref = re.sub(r"\bdoi:\S+", "", ref, flags=re.IGNORECASE)
    ref = re.sub(r"\s+", " ", ref).strip()
    return ref[:300]


# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------


def _empty_ref(title: str, raw: str) -> dict:
    return {
        "title": title,
        "authors": [],
        "journal": "",
        "year": "",
        "raw": raw,
    }


def extract_refs_regex(text: str) -> list[dict]:
    """Fallback parser: split the reference list and guess each title."""
    match = re.search(r"\bReferences\b", text, re.IGNORECASE)
    if not match:
        return []
    parts = re.split(r"\n\s*\[\d+\]|\n\s*\d+\.", text[match.end() :])
    refs = []
    for r in parts:
        r = r.strip().replace("\n", " ")
        if len(r) < 40:
            continue
        title = ""
        for s in re.split(r"\.\s+", r):
            s = s.strip()
            if len(s) < 15:
                continue
            if re.match(r"^[A-Z][a-z]+ [A-Z][\.,]", s):  # author-like
                continue
            if "et al" in s and len(s) < 40:
                continue
            if re.search(r"\d+\(\d+\)|\d+:\d+|pages \d+|arXiv preprint", s):
                continue  # venue / volume / pages
            title = s
            break
        refs.append(_empty_ref(title or r[:120], r))
    return refs


def _normalize_llm_items(items: list) -> list[dict]:
    results = []
    for item in items:
        if isinstance(item, str) and item.strip():
            results.append(_empty_ref(item.strip(), item.strip()))
            continue
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        authors = item.get("authors", [])
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(",") if a.strip()]
        elif isinstance(authors, list):
            authors = [str(a).strip() for a in authors if str(a).strip()]
        else:
            authors = []
        journal = str(item.get("journal", "")).strip()
        year = str(item.get("year", "")).strip()
        raw_parts = [", ".join(authors) + "."] if authors else []
        raw_parts.append(title + ".")
        raw_parts += [x + "." for x in (journal, year) if x]
        results.append(
            {
                "title": title,
                "authors": authors,
                "journal": journal,
                "year": year,
                "raw": " ".join(raw_parts),
            }
        )
    return results


def extract_refs_llm(text: str) -> list[dict]:
    """Extract references with an OpenAI-compatible LLM endpoint."""
    if not (LLM_URL and LLM_MODEL):
        log.info("LLM not configured (LLM_URL/LLM_MODEL); skipping")
        return []
    if not _HAS_OPENAI:
        log.info("openai package not installed; skipping LLM extraction")
        return []

    match = re.search(
        r"\b(References|Bibliography|Works Cited)\b", text, re.IGNORECASE
    )
    refs_text = text[match.start() :] if match else text
    messages = [
        {"role": "system", "content": _LLM_PROMPT},
        {
            "role": "user",
            "content": "Extract all references as structured JSON from this "
            f"text:\n\n{refs_text}",
        },
    ]

    raw = None
    try:
        client = OpenAI(
            base_url=LLM_URL.rstrip("/"),
            api_key=LLM_API_KEY or "none",
            timeout=LLM_TIMEOUT,
        )
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            reasoning_effort="low",
            temperature=0.0,
            max_tokens=8192,
        )
        if not response.choices:
            return []
        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            return []
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        items = []
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            for key in ("references", "refs", "results", "data"):
                if isinstance(parsed.get(key), list):
                    items = parsed[key]
                    break
        return _normalize_llm_items(items)
    except json.JSONDecodeError as e:
        log.warning("LLM JSON parse failed: %s", e)
        titles = re.findall(r'"title"\s*:\s*"([^"]{10,})"', raw or "")
        return [_empty_ref(t.strip(), t.strip()) for t in titles]
    except Exception as e:  # network, auth, server errors
        log.warning("LLM extraction failed: %s: %s", type(e).__name__, e)
        return []


def extract_references(text: str) -> tuple[list[dict], str]:
    """Try LLM extraction first, fall back to the regex parser."""
    refs = extract_refs_llm(text)
    if refs:
        return refs, "llm"
    return extract_refs_regex(text), "regex"


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def search_semantic_scholar(
    query: str, api_key: str = "", top_k: int = 5
) -> tuple[list, int]:
    """Top-k Semantic Scholar candidates for ``query`` (cached, retried)."""
    if query in _CACHE:
        return _CACHE[query]
    key = api_key or SS_API_KEY
    headers = {"x-api-key": key} if key else {}
    params = {
        "query": query,
        "limit": top_k,
        "fields": "title,year,authors,venue,journal,externalIds",
    }
    for attempt in range(3):
        try:
            resp = httpx.get(
                SS_SEARCH_URL, params=params, headers=headers, timeout=30.0
            )
            if resp.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            if resp.status_code != 200:
                log.warning("Semantic Scholar HTTP %s", resp.status_code)
                return [], 0
            data = resp.json()
            papers = data.get("data", [])
            _CACHE[query] = (papers, data.get("total", len(papers)))
            return _CACHE[query]
        except Exception as e:
            log.warning("Semantic Scholar request failed: %s", e)
            if attempt < 2:
                time.sleep(5)
    return [], 0


# ---------------------------------------------------------------------------
# Verification of one reference
# ---------------------------------------------------------------------------


def _venue(paper: dict) -> str:
    venue = paper.get("venue", "") or ""
    jinfo = paper.get("journal", {}) or {}
    jname = jinfo.get("name", "") if isinstance(jinfo, dict) else str(jinfo)
    jname = jname or ""
    return jname if len(jname) > len(venue) else venue


def verify_reference(
    ref: dict,
    ss_key: str = "",
    top_k: int = 5,
    threshold: float = VERIFIED_THRESHOLD,
    partial_threshold: float = PARTIAL_THRESHOLD,
    weights: tuple[float, float, float] = WEIGHTS,
    author_veto: bool = False,
    papers: list | None = None,
    scoring: str = DEFAULT_SCORING,
) -> dict:
    """Score one structured reference against Semantic Scholar.

    Parameters
    ----------
    ref : dict
        ``{"title", "authors", "journal", "year", "raw"}``.
    threshold, partial_threshold : float
        Lower bounds of the ``verified`` and ``partial`` bands.
    weights : (title, authors, journal)
        Composite weights. Default 0.60 / 0.25 / 0.15.
    author_veto : bool
        If True, a citation whose extracted authors match none of the best
        candidate's authors (author score 0) is labelled
        ``likely hallucinated`` regardless of its composite. Off by default.
    papers : list, optional
        Pre-fetched candidates (Semantic Scholar JSON); skips the API call.
    scoring : {"v2", "v1"}
        Scoring-rule version (see ``SCORING_VERSIONS``). Default "v2".
    """
    if scoring not in SCORING_VERSIONS:
        raise ValueError(f"scoring must be one of {SCORING_VERSIONS}")
    title = ref.get("title", "")
    target = title or ref.get("raw", "")
    query = clean_query(target)
    if papers is None:
        papers, total = search_semantic_scholar(
            query, api_key=ss_key, top_k=top_k
        )
    else:
        total = len(papers)

    cited_authors = ref.get("authors", [])
    cited_journal = ref.get("journal", "")
    cited_year = ref.get("year", "")

    best = {
        "score": 0.0,
        "title": "",
        "author_score": -1.0,
        "journal_score": -1.0,
        "year_match": None,
        "authors": [],
        "venue": "",
        "year": "",
    }
    for p in papers:
        cand_title = p.get("title", "")
        if not cand_title:
            continue
        ts = fuzzy_score(cand_title, target)
        cand_authors = [a.get("name", "") for a in p.get("authors", []) or []]
        a_s = author_score(cited_authors, cand_authors, scoring)
        venue = _venue(p)
        j_s = fuzzy_score(cited_journal, venue) if cited_journal else -1.0
        cand_year = str(p.get("year", ""))
        comp = composite_score(ts, a_s, j_s, weights, scoring)
        if comp > best["score"]:
            best = {
                "score": comp,
                "title": cand_title,
                "author_score": a_s,
                "journal_score": j_s,
                "year_match": (
                    (cited_year == cand_year) if cited_year else None
                ),
                "authors": cand_authors,
                "venue": venue,
                "year": cand_year,
            }

    score = best["score"]
    status = classify(score, threshold, partial_threshold)
    if author_veto and best["author_score"] == 0:
        status = "likely hallucinated"

    def _opt(x):
        return round(x, 1) if x >= 0 else None

    return {
        "query": query,
        "best_match": best["title"],
        "score": round(score, 1),
        "title_score": round(fuzzy_score(best["title"], target), 1),
        "author_score": _opt(best["author_score"]),
        "journal_score": _opt(best["journal_score"]),
        "year_match": best["year_match"],
        "status": status,
        "total_hits": total,
        "extracted_authors": cited_authors,
        "extracted_journal": cited_journal,
        "extracted_year": cited_year,
        "matched_authors": best["authors"],
        "matched_venue": best["venue"],
        "matched_year": best["year"],
    }


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def check_references(
    refs: list[dict], sleep_sec: float = 2.0, **kwargs
) -> list[dict]:
    """Verify structured references (see :func:`verify_reference`)."""
    results = []
    for i, ref in enumerate(refs, 1):
        result = verify_reference(ref, **kwargs)
        result["index"] = i
        result["reference"] = ref.get("raw", ref.get("title", ""))
        log.info(
            "[%d/%d] %-19s %5.1f  %s",
            i,
            len(refs),
            result["status"],
            result["score"],
            result["reference"][:70],
        )
        results.append(result)
        if i < len(refs):
            time.sleep(sleep_sec)
    return results


def check_text(text: str, sleep_sec: float = 2.0, **kwargs) -> list[dict]:
    """Extract and verify every reference in ``text``."""
    refs, method = extract_references(text)
    log.info("Found %d references via %s", len(refs), method)
    return check_references(refs, sleep_sec=sleep_sec, **kwargs)


SAMPLE_TEXT = """
References
[1] Daniil A. Boiko, Robert MacKnight, Ben Kline, Gabe Gomes. Autonomous \
chemical research with large language models. Nature, 624(7992):570-578, 2023.
[2] Kamal Choudhary, Brian DeCost, and Francesca Tavazza. Benchmarking graph \
neural networks for materials chemistry. npj Computational Materials, 2020.
[3] Kamal Choudhary. AtomGPT: Atomistic generative pretrained transformer \
for forward and inverse materials design. The Journal of Physical Chemistry \
Letters, 15(27):6909-6917, 2024.
[4] Fake Q. Author. Quantum blockchain neural networks for perpetual motion \
machines. Nature of Nonsense, 99:1-99, 2099.
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input", nargs="?", help="PDF or text file")
    ap.add_argument("-o", "--output", help="write results as JSON")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--author-veto", action="store_true")
    ap.add_argument("--scoring", choices=SCORING_VERSIONS, default="v2")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.input:
        text = SAMPLE_TEXT
    elif args.input.lower().endswith(".pdf"):
        text = extract_pdf_text(args.input)
    else:
        with open(args.input, encoding="utf-8") as f:
            text = f.read()

    results = check_text(
        text,
        sleep_sec=args.sleep,
        top_k=args.top_k,
        author_veto=args.author_veto,
        scoring=args.scoring,
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        json.dump(results, sys.stdout, indent=2, ensure_ascii=False)
        print()


if __name__ == "__main__":
    main()
