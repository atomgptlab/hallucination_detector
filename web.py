"""
hallucination.py — Sequential hallucination detector for academic references.

Extracts references from text/PDF, then verifies each one against Semantic
Scholar to flag likely-hallucinated citations.

Usage:
    python hallucination.py                  # run with built-in sample text
    python hallucination.py paper.pdf        # test a PDF
    python hallucination.py paper.txt        # test a text file
"""

import io
import json
import os
import re
import sys
import time

import httpx


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG — replace placeholders or set the matching environment variables
# ═══════════════════════════════════════════════════════════════════════════

# Semantic Scholar API key (optional, but raises rate limits).
# Get one at: https://www.semanticscholar.org/product/api
_SS_API_KEY = os.environ.get("SS_API_KEY", "<YOUR_SEMANTIC_SCHOLAR_API_KEY>")

# LLM endpoint used for reference extraction (OpenAI-compatible API).
_LLM_URL     = os.environ.get("LLM_URL",     "<YOUR_LLM_BASE_URL>")
_LLM_MODEL   = os.environ.get("LLM_MODEL",   "<YOUR_LLM_MODEL_NAME>")
_LLM_API_KEY = os.environ.get("LLM_API_KEY", "<YOUR_LLM_API_KEY>")
_LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "60"))

_CACHE: dict = {}


# ── Optional deps ───────────────────────────────────────────────────────────
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


# ── LLM system prompt ──────────────────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def extract_pdf_text(content: bytes) -> str:
    if not _HAS_PDF:
        raise RuntimeError("pdfplumber not installed")
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)


def fuzzy_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if _HAS_FUZZ:
        return _fuzz.token_set_ratio(a.lower(), b.lower())
    s1, s2 = set(a.lower().split()), set(b.lower().split())
    if not s1 or not s2:
        return 0.0
    return 100.0 * len(s1 & s2) / len(s1 | s2)


def clean_query(ref: str) -> str:
    ref = re.sub(r"^\s*[\[\(]?\d+[\]\)\.]\s*", "", ref)
    ref = re.sub(r"https?://\S+", "", ref)
    ref = re.sub(r"\bdoi:\S+", "", ref, flags=re.IGNORECASE)
    ref = re.sub(r"\s+", " ", ref).strip()
    return ref[:300]


# ═══════════════════════════════════════════════════════════════════════════
# EXTRACT REFERENCES
# ═══════════════════════════════════════════════════════════════════════════

def extract_refs_regex(text: str) -> list[dict]:
    match = re.search(r"\bReferences\b", text, re.IGNORECASE)
    if not match:
        return []
    parts = re.split(r"\n\s*\[\d+\]|\n\s*\d+\.", text[match.end():])
    refs = []
    for r in parts:
        r = r.strip().replace("\n", " ")
        if len(r) < 40:
            continue
        # Extract title: skip "Authors. " then grab text before "Journal/arXiv"
        sentences = re.split(r'\.\s+', r)
        title = ""
        for s in sentences:
            s = s.strip()
            if len(s) < 15:
                continue
            # Skip author-like segments (short names with commas)
            if re.match(r'^[A-Z][a-z]+ [A-Z][\.,]', s):
                continue
            if 'et al' in s and len(s) < 40:
                continue
            # Skip journal/venue segments (volume numbers, pages)
            if re.search(r'\d+\(\d+\)|\d+:\d+|pages \d+|arXiv preprint', s):
                continue
            title = s
            break
        if not title:
            title = r[:120]  # fallback
        refs.append({"title": title, "authors": [], "journal": "", "year": "", "raw": r})
    return refs


def extract_refs_llm(text: str) -> list[dict]:
    """Extract references using an OpenAI-compatible LLM endpoint."""
    if not _LLM_URL or not _LLM_MODEL or _LLM_URL.startswith("<"):
        print("LLM not configured, skipping")
        return []
    if not _HAS_OPENAI:
        print("openai package not installed, skipping")
        return []

    match = re.search(r"\b(References|Bibliography|Works Cited)\b", text, re.IGNORECASE)
    refs_text = text[match.start():] if match else text

    messages = [
        {"role": "system", "content": _LLM_PROMPT},
        {"role": "user",   "content": f"Extract all references as structured JSON from this text:\n\n{refs_text}"},
    ]

    raw = None
    try:
        client = OpenAI(
            base_url=_LLM_URL.rstrip("/"),
            api_key=_LLM_API_KEY,
            timeout=_LLM_TIMEOUT,
        )
        response = client.chat.completions.create(
            model=_LLM_MODEL,
            messages=messages,
            reasoning_effort="low",
            temperature=0.0,
            max_tokens=8192,
        )
        if not response.choices:
            print("LLM: empty choices")
            return []

        raw = response.choices[0].message.content
        if not raw:
            print(f"LLM: empty content, finish_reason={response.choices[0].finish_reason}")
            return []

        raw = raw.strip()
        print(f"LLM raw (first 300): {raw[:300]}")

        # Strip markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)

        # Handle list or dict wrapper
        items = []
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            for key in ("references", "refs", "results", "data"):
                if key in parsed and isinstance(parsed[key], list):
                    items = parsed[key]
                    break

        results = []
        for item in items:
            if isinstance(item, str) and item.strip():
                results.append({
                    "title": item.strip(), "authors": [], "journal": "",
                    "year": "", "raw": item.strip(),
                })
            elif isinstance(item, dict):
                title = str(item.get("title", "")).strip()
                if not title:
                    continue

                authors = item.get("authors", [])
                if isinstance(authors, str):
                    authors = [a.strip() for a in authors.split(",") if a.strip()]
                elif not isinstance(authors, list):
                    authors = []
                else:
                    authors = [str(a).strip() for a in authors if str(a).strip()]

                journal = str(item.get("journal", "")).strip()
                year    = str(item.get("year", "")).strip()

                raw_parts = []
                if authors:
                    raw_parts.append(", ".join(authors) + ".")
                raw_parts.append(title + ".")
                if journal:
                    raw_parts.append(journal + ".")
                if year:
                    raw_parts.append(year + ".")

                results.append({
                    "title": title, "authors": authors, "journal": journal,
                    "year": year, "raw": " ".join(raw_parts),
                })
        return results

    except json.JSONDecodeError as e:
        print(f"LLM JSON parse failed: {e}\nRaw: {raw!r}")
        if raw:
            titles = re.findall(r'"title"\s*:\s*"([^"]{10,})"', raw)
            if titles:
                print(f"Regex fallback: {len(titles)} titles")
                return [{"title": t.strip(), "authors": [], "journal": "",
                         "year": "", "raw": t.strip()} for t in titles]
        return []
    except Exception as e:
        print(f"LLM extraction failed: {type(e).__name__}: {e}")
        return []


def extract_references(text: str) -> tuple[list[dict], str]:
    """Try LLM first (clean titles), fall back to regex."""
    llm_refs = extract_refs_llm(text)
    if llm_refs:
        return llm_refs, "llm"
    print("LLM failed, falling back to regex...")
    regex_refs = extract_refs_regex(text)
    return regex_refs, "regex"


# ═══════════════════════════════════════════════════════════════════════════
# SEMANTIC SCHOLAR SEARCH
# ═══════════════════════════════════════════════════════════════════════════

def search_semantic_scholar(query: str, api_key: str = "",
                            top_k: int = 5) -> tuple[list, int]:
    if query in _CACHE:
        return _CACHE[query]

    ss_key = api_key or _SS_API_KEY
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": ss_key} if ss_key and not ss_key.startswith("<") else {}
    params = {
        "query": query, "limit": top_k,
        "fields": "title,year,authors,venue,journal,externalIds",
    }

    for attempt in range(3):
        try:
            resp = httpx.get(url, params=params, headers=headers, timeout=30.0)
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  SS 429, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"  SS error: {resp.status_code}")
                return [], 0
            data = resp.json()
            papers = data.get("data", [])
            total = data.get("total", len(papers))
            _CACHE[query] = (papers, total)
            return papers, total
        except Exception as e:
            print(f"  SS exception: {e}")
            if attempt < 2:
                time.sleep(5)
    return [], 0


# ═══════════════════════════════════════════════════════════════════════════
# VERIFY ONE REFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def verify_reference(ref: dict, ss_key: str = "", top_k: int = 5,
                     threshold: int = 80) -> dict:
    title = ref.get("title", "")
    query = clean_query(title) if title else clean_query(ref.get("raw", ""))
    papers, total = search_semantic_scholar(query, api_key=ss_key, top_k=top_k)

    extracted_authors = ref.get("authors", [])
    extracted_journal = ref.get("journal", "")
    extracted_year    = ref.get("year", "")

    best = {
        "score": 0.0, "title": "", "author_score": -1.0,
        "journal_score": -1.0, "year_match": None,
        "authors": [], "venue": "", "year": "",
    }

    for p in papers:
        ct = p.get("title", "")
        if not ct:
            continue

        ts = fuzzy_score(ct, title or ref.get("raw", ""))

        # Author score
        ss_authors = p.get("authors", [])
        if extracted_authors and ss_authors:
            ss_names = [a.get("name", "").lower() for a in ss_authors if a.get("name")]
            matched = sum(
                1 for ea in extracted_authors
                if ea.strip() and any(
                    ea.lower().split()[-1] in sn.split()
                    for sn in ss_names
                )
            )
            a_score = 100.0 * matched / len(extracted_authors)
        else:
            a_score = -1

        # Journal score
        ss_venue = p.get("venue", "") or ""
        ss_jinfo = p.get("journal", {}) or {}
        ss_jname = ss_jinfo.get("name", "") if isinstance(ss_jinfo, dict) else str(ss_jinfo)
        venue_str = ss_jname if len(ss_jname) > len(ss_venue) else ss_venue
        j_score = fuzzy_score(extracted_journal, venue_str) if extracted_journal else -1

        # Year
        ss_yr = str(p.get("year", ""))
        yr_match = (extracted_year == ss_yr) if extracted_year else None

        # Composite: title=60%, authors=25%, journal=15%
        comp = ts * 0.60
        comp += (a_score * 0.25) if a_score >= 0 else (ts * 0.25)
        comp += (j_score * 0.15) if j_score >= 0 else (ts * 0.15)

        if comp > best["score"]:
            best = {
                "score": comp, "title": ct, "author_score": a_score,
                "journal_score": j_score, "year_match": yr_match,
                "authors": [a.get("name", "") for a in ss_authors],
                "venue": venue_str, "year": ss_yr,
            }

    score = best["score"]
    status = ("verified" if score >= threshold
              else "partial" if score >= 60
              else "likely hallucinated")

    print(f"  [{status}] score={score:.1f} → '{best['title'][:60]}'")

    return {
        "query":             query,
        "best_match":        best["title"],
        "score":             round(score, 1),
        "title_score":       round(fuzzy_score(best["title"], title or ref.get("raw", "")), 1),
        "author_score":      round(best["author_score"], 1) if best["author_score"] >= 0 else None,
        "journal_score":     round(best["journal_score"], 1) if best["journal_score"] >= 0 else None,
        "year_match":        best["year_match"],
        "status":            status,
        "total_hits":        total,
        "extracted_authors": extracted_authors,
        "extracted_journal": extracted_journal,
        "extracted_year":    extracted_year,
        "matched_authors":   best["authors"],
        "matched_venue":     best["venue"],
        "matched_year":      best["year"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# BATCH RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def check_text(text: str, ss_key: str = "", top_k: int = 5,
               threshold: int = 80, sleep_sec: float = 2.0) -> list[dict]:
    """Extract and verify every reference in `text`. Returns list of results."""
    refs, method = extract_references(text)
    if not refs:
        print("No references found.")
        return []

    print(f"Found {len(refs)} references via {method}.")
    results = []
    for i, ref in enumerate(refs, 1):
        preview = ref.get("title", ref.get("raw", ""))[:80]
        print(f"\n  Checking [{i}/{len(refs)}]: {preview}")
        result = verify_reference(ref, ss_key=ss_key, top_k=top_k, threshold=threshold)
        result["index"] = i
        result["reference"] = ref.get("raw", ref.get("title", ""))
        results.append(result)
        if i < len(refs):
            time.sleep(sleep_sec)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# LOCAL TESTING
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    SAMPLE_TEXT = """
This paper discusses materials discovery using AI.

References
[1] Daniil A. Boiko, Robert MacKnight, Ben Kline, Gabe Gomes, et al. Autonomous chemical research with large language models. Nature, 624(7992):570-578, 2023.
[2] Kamal Choudhary, Brian DeCost, and Francesca Tavazza. Benchmarking graph neural networks for materials chemistry. npj Computational Materials, 6(1), 2020.
[3] Kamal Choudhary. AtomGPT: Atomistic generative pretrained transformer for forward and inverse materials design. The Journal of Physical Chemistry Letters, 15(27):6909-6917, 2024.
[4] Fake Q. Author. Quantum blockchain neural networks for perpetual motion machines. Nature of Nonsense, 99:1-99, 2099.
"""

    def run_test(text: str, label: str = "test"):
        print(f"\n{'='*70}\n  {label}\n{'='*70}")
        print(f"Text length: {len(text)} chars\n")
        results = check_text(text)
        print(f"\n{'='*70}\n  Done! {len(results)} references checked.\n{'='*70}\n")

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"Loading file: {filepath}")
        if filepath.endswith(".pdf"):
            with open(filepath, "rb") as f:
                text = extract_pdf_text(f.read())
            run_test(text, label=f"PDF: {filepath}")
        else:
            with open(filepath, "r") as f:
                text = f.read()
            run_test(text, label=f"File: {filepath}")
    else:
        print("Usage:")
        print("  python hallucination.py                  # run with sample text")
        print("  python hallucination.py paper.pdf        # test a PDF")
        print("  python hallucination.py paper.txt        # test a text file")
        print("\nRunning with sample text...\n")
        run_test(SAMPLE_TEXT, label="Sample Text")
