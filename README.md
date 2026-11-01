# Hallucination Detector

Reference-hallucination detector for scientific manuscripts. Each citation is
parsed into bibliographic fields by an LLM, matched against
[Semantic Scholar](https://www.semanticscholar.org/product/api), and labelled
**verified**, **partial** or **likely hallucinated** from a per-field
similarity score. This is the scoring code behind the AtomGPT.org reference
checker evaluated in:

> H. Neralla, J. Lee, A. H. Romero, K. Choudhary, *Hallucination Detector: A
> hybrid LLM and Semantic Scholar tool calling for detecting hallucination in
> scientific literature on AtomGPT.org* (under review).

Benchmark data, per-citation outputs and the analysis notebook for the paper
are in
[harineralla/AtomGPT-Hallucination-Detector](https://github.com/harineralla/AtomGPT-Hallucination-Detector).

## Repository layout

| Path | Contents |
|------|----------|
| `hallucination.py` | Extraction, retrieval, scoring (`verify_reference`) and CLI |
| `scripts/rescore_sensitivity.py` | Re-scores released outputs under other weights / author veto / thresholds |
| `scripts/materials_pilot.py` | Reproduces the materials-science pilot |
| `tests/test_scoring.py` | Offline tests of the deterministic scoring stage |
| `legacy/crossref_arxiv_prototype.py` | Early Crossref + arXiv prototype (not used in the paper) |

## Install

```bash
pip install -r requirements.txt
```

## Configuration

All settings come from environment variables; nothing is hard-coded.

| Variable | Purpose |
|----------|---------|
| `SS_API_KEY` | Semantic Scholar API key (optional; raises the rate limit) |
| `LLM_URL` | Base URL of any OpenAI-compatible chat-completions endpoint |
| `LLM_MODEL` | Model name for extraction (the paper used `gpt-oss-20b`) |
| `LLM_API_KEY` | API key for that endpoint, if required |
| `LLM_TIMEOUT` | Request timeout in seconds (default 60) |

Without an LLM endpoint the tool falls back to a regular-expression parser,
which extracts titles only (author and venue terms are then treated as
missing; see below).

## Usage

```bash
python hallucination.py                    # built-in sample references
python hallucination.py paper.pdf -o out.json
python hallucination.py refs.txt --author-veto
```

```python
from hallucination import verify_reference

ref = {
    "title": "Benchmarking graph neural networks for materials chemistry",
    "authors": ["Victor Fung", "Jiaxin Zhang", "Eric Juarez", "Bobby G. Sumpter"],
    "journal": "npj Computational Materials",
    "year": "2021",
}
print(verify_reference(ref))
```

## Scoring (exact definitions)

For every one of the top-k (default 5) Semantic Scholar candidates:

* **Title similarity** `s_t = token_set_ratio(lower(cited_title), lower(candidate_title))`
  using [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz), on a 0-100 scale.
  No other normalization (no Unicode folding, punctuation removal or
  stop-word handling).
* **Author similarity** `s_a = 100 * (matched cited authors) / (number of cited authors)`.
  Each name is lower-cased and stripped of periods and commas; its surname
  is the last token and its initial the first letter of the first token. A
  cited author matches if some candidate author has the same surname and,
  when both initials are available, the same initial.
* **Journal similarity** `s_j = token_set_ratio(lower(cited_venue), lower(candidate_venue))`,
  where `candidate_venue` is the longer of Semantic Scholar's `journal.name`
  and `venue`. Abbreviations are not expanded.
* **Composite** `S = (0.60 s_t + 0.25 s_a + 0.15 s_j) / (sum of the weights used)`.
  If authors or venue were not extracted, that term is dropped and the
  remaining weights are renormalized. The year is reported as a match flag
  but is not scored.

The citation receives the maximum `S` over the candidates and is labelled
**verified** if `S >= 80`, **partial** if `60 <= S < 80`, and **likely
hallucinated** otherwise (including when no candidate is returned).
Weights, thresholds and an optional author veto (`author_veto=True`: an
author score of 0 forces *likely hallucinated*) are arguments of
`verify_reference`.

### Scoring versions

`verify_reference(..., scoring="v2")` (default) is the current rule above.
`scoring="v1"` reproduces the original rule, used until July 2026 and for
the NeurIPS 2025 benchmark in the paper. It differs in two ways:

| | v1 (original) | v2 (current) |
|---|---|---|
| Missing author / venue | term replaced by the title score | term dropped, weights renormalized |
| Author match | surname only (last token, exact) | surname and first initial |

Both changes stop a fabricated reference from scoring well on the strength
of a plausible title or a common surname alone. The CiteAudit and materials
analyses in the paper use v2.

### Known limitations of the similarity functions

* Surnames with diacritics do not match their ASCII spelling
  (`Grazulis` vs `Gražulis`).
* A literal `et al.` returned by the extractor counts as an unmatched author,
  so a truncated list such as `A, B, C, et al.` scores at most 75.
* Hyphenated surnames match only if hyphenated identically in the index.
  Surname-first strings (`Smith, J.`) take the initial as the surname.
* If Semantic Scholar returns no candidate (not indexed, or rate-limited),
  the score is 0 and the citation is flagged.
* Venue abbreviations score partially against full names
  (`Phys. Rev. Lett.` vs `Physical Review Letters` = 66.7).
* Data, software and database entries that have no indexed paper cannot be
  verified by literature matching.

## Reproducing the paper analyses

```bash
# weight / veto / threshold sensitivity on the released per-citation outputs
python scripts/rescore_sensitivity.py --scoring v1 ui_all_cols_results.csv
python scripts/rescore_sensitivity.py --scoring v2 citeaudit_all300_combined.csv

# materials pilot (needs network access to Semantic Scholar)
python scripts/materials_pilot.py fabricated_materials_citations.json

# offline tests
python -m pytest tests
```

## License

See [LICENSE](LICENSE).
