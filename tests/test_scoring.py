"""Offline tests of the deterministic scoring stage (no network calls)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import hallucination as h  # noqa: E402

PAPER = {
    "title": "Benchmarking graph neural networks for materials chemistry",
    "authors": [
        {"name": "Victor Fung"},
        {"name": "Jiaxin Zhang"},
        {"name": "Eric Juarez"},
        {"name": "Bobby G. Sumpter"},
    ],
    "venue": "npj Computational Materials",
    "journal": {"name": "npj Computational Materials"},
    "year": 2021,
}


def ref(**kw):
    base = {
        "title": PAPER["title"],
        "authors": ["Victor Fung", "Jiaxin Zhang"],
        "journal": "npj Computational Materials",
        "year": "2021",
        "raw": "",
    }
    base.update(kw)
    return base


def test_exact_match_is_verified():
    r = h.verify_reference(ref(), papers=[PAPER])
    assert r["score"] == 100.0
    assert r["status"] == "verified"
    assert r["year_match"] is True


def test_composite_formula():
    assert h.composite_score(100, 0, 50) == pytest.approx(67.5)
    assert h.composite_score(80, -1, -1) == pytest.approx(80.0)
    assert h.composite_score(80, 0, -1) == pytest.approx(0.75 * 80)


def test_fabricated_authors_drop_score():
    r = h.verify_reference(
        ref(authors=["John Doe", "Jane Smith"]), papers=[PAPER]
    )
    assert r["author_score"] == 0.0
    assert r["score"] == pytest.approx(75.0)
    assert r["status"] == "partial"
    vetoed = h.verify_reference(
        ref(authors=["John Doe", "Jane Smith"]),
        papers=[PAPER],
        author_veto=True,
    )
    assert vetoed["status"] == "likely hallucinated"


def test_missing_fields_substitute_title_score():
    r = h.verify_reference(ref(authors=[], journal=""), papers=[PAPER])
    assert r["author_score"] is None and r["journal_score"] is None
    assert r["score"] == r["title_score"] == 100.0


def test_author_matching_rules():
    # surname = last whitespace token, compared case-insensitively
    assert h.author_score(["V. Fung"], ["Victor Fung"]) == 100.0
    # documented limitations: no diacritic folding, "et al." counts
    assert h.author_score(["S. Grazulis"], ["Saulius Gražulis"]) == 0.0
    assert h.author_score(["V. Fung", "et al."], ["Victor Fung"]) == 50.0


def test_thresholds():
    assert h.classify(80) == "verified"
    assert h.classify(79.9) == "partial"
    assert h.classify(60) == "partial"
    assert h.classify(59.9) == "likely hallucinated"


def test_no_candidates_is_hallucinated():
    r = h.verify_reference(ref(), papers=[])
    assert r["score"] == 0.0
    assert r["status"] == "likely hallucinated"


def test_regex_fallback_extracts_titles():
    refs = h.extract_refs_regex(h.SAMPLE_TEXT)
    assert len(refs) == 4
    assert "AtomGPT" in refs[2]["raw"]
    assert all(r["title"] for r in refs)
