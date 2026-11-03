# Hallucination Detector

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/atomgptlab/hallucination_detector/blob/main/notebooks/quickstart.ipynb)

Detect hallucinated (fabricated) citations in scientific manuscripts.
Each reference is parsed into bibliographic fields by an LLM, matched
against [Semantic Scholar](https://www.semanticscholar.org/product/api),
and labelled **verified**, **partial** or **likely hallucinated** from an
interpretable per-field similarity score.

This is the code and benchmark data behind the
[AtomGPT.org reference checker](https://atomgpt.org/hallucination_detector)
described in:

> H. Neralla, J. Lee, A. H. Romero, K. Choudhary, *Hallucination Detector: A
> hybrid LLM and Semantic Scholar tool calling for detecting hallucination in
> scientific literature on AtomGPT.org* (under review).

**Quick start:** try it in the browser with the
[Colab notebook](https://colab.research.google.com/github/atomgptlab/hallucination_detector/blob/main/notebooks/quickstart.ipynb): check a few references, a pasted bibliography
or a PDF, run a small benchmark slice, and explore the released results.

## How it works

```
citation text ──► 1. LLM extraction ──► {title, authors, journal, year}
                     (regex fallback)              │
                                                   ▼
                  2. Semantic Scholar search (top-k candidates by title)
                                                   │
                                                   ▼
                  3. per-field similarity   title 60% · authors 25% · journal 15%
                                                   │
                                                   ▼
                  4. composite score  ≥ 80 verified · 60–80 partial · < 60 likely hallucinated
```

Steps 3 and 4 are deterministic and fully specified [below](#scoring).
Fabricated references are caught mostly through the **author** term: an
LLM can produce a plausible title and a real venue, but rarely the right
author list for a paper that exists.

## Layout

```
hallucination_detector/
├── config.py               # reads API keys / endpoints from env vars
├── hallucination.py        # core pipeline: extract → retrieve → score → label
├── run_benchmark.py        # CLI runner (concurrent, resumable, fail-fast)
├── requirements.txt
├── data/                   # benchmarks used in the paper
│   ├── neurips2025_hallucinated.json   # 100 confirmed fabrications
│   ├── citeaudit_300.json              # 150 real + 150 fake
│   ├── materials_pilot.json            # 26 genuine materials references
│   ├── materials_pilot_genuine.json    # 21 references harvested from 3 seed papers
│   └── materials_pilot_fabricated.json # 16 fabricated materials citations
├── results/                # per-citation outputs reported in the paper
│   ├── neurips2025_v1.json
│   ├── citeaudit_300_v2.json
│   ├── materials_pilot_v2.json
│   └── materials_pilot_harvested_fabricated.csv   # harvested-vs-fabricated pilot
├── analysis/               # notebook, raw tool exports and NeurIPS figures
├── notebooks/
│   └── quickstart.ipynb    # Colab quick start
├── scripts/
│   ├── rescore_sensitivity.py   # re-score outputs under other weights/thresholds
│   └── materials_pilot.py       # harvested-vs-fabricated materials pilot
├── tests/                  # offline tests (no network)
└── legacy/                 # early Crossref + arXiv prototype (not used)
```

## Setup

```bash
pip install -r requirements.txt

# LLM used for field extraction: the AtomGPT API (as in the paper)
export LLM_URL="https://atomgpt.org/api"
export LLM_MODEL="openai/gpt-oss-20b"  # the model used in the paper
export LLM_API_KEY="sk-..."            # your AtomGPT API key (see below)

# optional
export SS_API_KEY="..."                # Semantic Scholar key (higher rate limit)
export SS_MIN_INTERVAL=1.0             # seconds between Semantic Scholar calls
export LLM_TIMEOUT=60
```

To get an AtomGPT API key, sign in at [atomgpt.org](https://atomgpt.org)
and open Profile → Settings → Account → API Keys. Any other
OpenAI-compatible endpoint also works: set `LLM_URL` to its base URL (for
example `https://your-endpoint/v1`) and `LLM_MODEL` to a model it serves.
Nothing is set by default, so no citation text is sent anywhere unless you
configure an endpoint.

Without an LLM endpoint the tool falls back to a regular-expression parser
that extracts titles only, so the author and venue terms are treated as
missing. Use an LLM for meaningful results. A free
[Semantic Scholar API key](https://www.semanticscholar.org/product/api#api-key-form)
is strongly recommended for benchmark runs; unauthenticated requests are
rate-limited heavily.

## Usage

### Check a manuscript

```bash
python hallucination.py paper.pdf -o report.json   # PDF
python hallucination.py refs.txt                   # text with a References section
python hallucination.py                            # built-in sample
```

```python
from hallucination import verify_reference

ref = {
    "title": "Benchmarking graph neural networks for materials chemistry",
    "authors": ["Victor Fung", "Jiaxin Zhang", "Eric Juarez", "Bobby G. Sumpter"],
    "journal": "npj Computational Materials",
    "year": "2021",
}
print(verify_reference(ref))   # {'status': 'verified', 'score': 100.0, ...}
```

### Run a benchmark

```bash
python run_benchmark.py \
    --file data/citeaudit_300.json \
    --workers 2 \
    --output results/citeaudit_300_run.json
```

| Flag | Description |
|---|---|
| `--workers N` | Concurrent worker threads (default 2); they share one Semantic Scholar rate limiter |
| `--gt all\|true\|false` | Keep all, only real, or only fake citations |
| `--source NAME` | Keep one `source_type` (e.g. `realworld`, `generated`) |
| `--start N --limit M` | Run a slice of the dataset |
| `--scoring v1\|v2` | Scoring rule (default `v2`, see [versions](#scoring-versions)) |
| `--threshold T` | Lower bound of *verified* (default 80) |
| `--top-k K` | Semantic Scholar candidates per citation (default 5) |
| `--author-veto` | Author score 0 forces *likely hallucinated* |
| `--lenient` | Score API failures 0, as the web tool does, instead of stopping |
| `--no-resume` | Start fresh, ignoring an existing output file |

The runner is **resumable**: results are saved after every citation, and
re-running the same command picks up where it left off. It is
**fail-fast**: an authentication, quota or persistent API error stops the
run with exit code 2 (safe to resume) rather than silently scoring the
citation as a hallucination.

Any file in the same record format works, including the full
[CiteAudit](https://github.com/shiiiikw/CiteAudit) `benchmark.json`, which
is read as is.

## Benchmarks

| Dataset | n | Real / fake | Source |
|---|---:|---|---|
| `neurips2025_hallucinated.json` | 100 | 0 / 100 | Confirmed hallucinated citations in accepted NeurIPS 2025 papers ([GPTZero audit](https://gptzero.me/news/neurips/)), verbatim as submitted |
| `citeaudit_300.json` | 300 | 150 / 150 | Balanced 300-citation sample of the [CiteAudit](https://github.com/shiiiikw/CiteAudit) benchmark (91 real-world, 209 generated) |
| `materials_pilot.json` | 26 | 26 / 0 | Genuine materials references in five types: articles, database papers, software, crystallographic databases, bare data entries |
| `materials_pilot_genuine.json` + `materials_pilot_fabricated.json` | 37 | 21 / 16 | Pilot: references harvested from the JARVIS-DFT, ALIGNN and DiffractGPT papers (7 each, via the Semantic Scholar Graph API) and fabricated materials citations (plausible title, real venue, invented authors). The 16 titles, venues and years are those of the paper's Table S2; the original invented author lists were not retained, so the author lists here are a reconstruction |

### Results (default weights 60/25/15, verified ≥ 80)

| Dataset | Scoring | Verified / partial / likely | Precision | Recall | FPR |
|---|---|---|---:|---:|---:|
| NeurIPS 2025 | v1 | 6 / 28 / 66 | n/a | 0.94 | n/a |
| CiteAudit 300 | v2 | 207 / 36 / 56 | 0.80 | 0.49 | 0.12 |
| Materials (21 literature-backed) | v2 | 19 / 2 / 0 | n/a | n/a | 0.10 |
| Materials (5 bare data entries) | v2 | 0 / 3 / 2 | n/a | n/a | 1.00 |
| Materials pilot, harvested genuine | v2 | 21 / 0 / 0 | n/a | n/a | 0.00 |
| Materials pilot, fabricated | v2 | 0 / 0 / 16 | n/a | 1.00 | n/a |

A citation counts as flagged when it is not *verified*. One real CiteAudit
citation was not run and is omitted (n = 299). On CiteAudit, generated
fakes are caught with precision 0.95, while most false positives come from
real-world citations that Semantic Scholar does not index (textbooks,
datasets, reports) or returned no candidate for. Bare data entries (a
Materials Project ID, a dataset DOI) have no paper to match and are
outside what literature matching can verify.

Shifting weight to authors (40/45/15) raises CiteAudit recall to 0.59 at
the same false-positive rate; see `scripts/rescore_sensitivity.py`.

## Data format

Each record in a `data/*.json` file (under the `data` key):

```json
{
  "source_type": "realworld" | "generated" | "neurips2025" | "materials",
  "citation": "<full citation string>",
  "GT": true | false
}
```

`GT = true` means a real citation and `false` a fabricated one. The materials
set also has `category` and `label`. The runner also accepts CSV files, and
the labels `GT_label` (`real` / `fake`) or `truth` (`genuine` / `fabricated`).

## Output format

```json
{
  "config": { "scoring_version": "v2", "weights": {...}, "extraction_model": "...", ... },
  "results": [
    {
      "index": 0,
      "source_type": "generated",
      "citation": "...",
      "GT": false,
      "Pred": true,
      "Correct": false,
      "status": "verified",
      "score": 94.1,
      "title_score": 100.0,
      "author_score": 80.0,
      "journal_score": null,
      "year_match": true,
      "best_match": "<closest Semantic Scholar title>",
      "matched_authors": [...],
      "matched_venue": "..."
    }
  ],
  "summary": {
    "total": 299, "status": {...},
    "TP": 74, "FP": 18, "TN": 131, "FN": 76,
    "accuracy": 0.6856, "precision": 0.8043, "recall": 0.4933,
    "f1": 0.6115, "false_positive_rate": 0.1208,
    "by_source_type": {...}
  }
}
```

`Pred = true` means predicted real (status *verified*). A field score is
`null` when that field was not extracted.

## Evaluation

A fake citation is the positive class:

- **TP**: fake and flagged (not *verified*)
- **FN**: fake but *verified* (missed)
- **TN**: real and *verified*
- **FP**: real but flagged (over-flagged)

`Accuracy = (TP+TN)/N`, `Precision = TP/(TP+FP)`, `Recall = TP/(TP+FN)`,
`F1 = 2·P·R/(P+R)`, `FPR = FP/(FP+TN)`.

## Scoring

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
Weights, thresholds and an optional author veto are arguments of
`verify_reference`.

### Scoring versions

`scoring="v2"` (default) is the current rule above. `scoring="v1"`
reproduces the original rule, used until July 2026 and for the NeurIPS 2025
benchmark:

| | v1 (original) | v2 (current) |
|---|---|---|
| Missing author / venue | term replaced by the title score | term dropped, weights renormalized |
| Author match | surname only (last token, exact) | surname and first initial |

Both changes stop a fabricated reference from scoring well on the strength
of a plausible title or a common surname alone. The CiteAudit and materials
results use v2.

### Known limitations

* Surnames with diacritics do not match their ASCII spelling
  (`Grazulis` vs `Gražulis`).
* A literal `et al.` returned by the extractor counts as an unmatched author,
  so a truncated list such as `A, B, C, et al.` scores at most 75.
* Hyphenated surnames match only if hyphenated identically in the index.
  Surname-first strings (`Smith, J.`) take the initial as the surname.
* Venue abbreviations score partially against full names
  (`Phys. Rev. Lett.` vs `Physical Review Letters` = 66.7).
* Works Semantic Scholar does not index (books, reports, datasets, software
  without a paper) cannot be verified and are flagged.

## Reproducing the paper

```bash
# sensitivity table: re-score the released outputs (no API calls)
python scripts/rescore_sensitivity.py --scoring v1 results/neurips2025_v1.json
python scripts/rescore_sensitivity.py --scoring v2 results/citeaudit_300_v2.json
python scripts/rescore_sensitivity.py --scoring v2 results/materials_pilot_v2.json

# re-run a benchmark end to end (needs LLM + Semantic Scholar access)
python run_benchmark.py --file data/citeaudit_300.json --output results/citeaudit_300_run.json
python run_benchmark.py --file data/neurips2025_hallucinated.json --scoring v1 \
    --output results/neurips2025_run.json

# harvested-vs-fabricated materials pilot
python scripts/materials_pilot.py data/materials_pilot_fabricated.json \
    --genuine data/materials_pilot_genuine.json -o results/materials_pilot_harvested_fabricated.csv

# offline tests
python -m pytest tests
```

Re-running end to end can differ slightly from the released outputs:
Semantic Scholar's index changes over time and LLM extraction is not
bit-reproducible across model versions. The analysis notebook, the raw
web-tool exports and the NeurIPS figures are in [`analysis/`](analysis).

## Citation

```bibtex
@article{neralla2026hallucination,
  title   = {Hallucination Detector: A hybrid LLM and Semantic Scholar tool calling
             for detecting hallucination in scientific literature on AtomGPT.org},
  author  = {Neralla, Harichandana and Lee, Jaehyung and Romero, Aldo H. and Choudhary, Kamal},
  note    = {Under review},
  year    = {2026}
}
```

The CiteAudit subset is derived from the CiteAudit benchmark; please cite
that work when using `data/citeaudit_300.json`.

## License

Apache 2.0, see [LICENSE](LICENSE).
