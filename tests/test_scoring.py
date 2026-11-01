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


@pytest.mark.parametrize("v", h.SCORING_VERSIONS)
def test_exact_match_is_verified(v):
    r = h.verify_reference(ref(), papers=[PAPER], scoring=v)
    assert r["score"] == 100.0
    assert r["status"] == "verified"
    assert r["year_match"] is True


def test_composite_formula():
    for v in h.SCORING_VERSIONS:
        assert h.composite_score(100, 0, 50, scoring=v) == pytest.approx(67.5)
        assert h.composite_score(80, -1, -1, scoring=v) == pytest.approx(80)
    # missing author: v1 substitutes the title score, v2 renormalizes
    assert h.composite_score(80, -1, 0, scoring="v1") == pytest.approx(68)
    assert h.composite_score(80, -1, 0, scoring="v2") == pytest.approx(64)


@pytest.mark.parametrize("v", h.SCORING_VERSIONS)
def test_fabricated_authors_drop_score(v):
    fake = ref(authors=["John Doe", "Jane Smith"])
    r = h.verify_reference(fake, papers=[PAPER], scoring=v)
    assert r["author_score"] == 0.0
    assert r["score"] == pytest.approx(75.0)
    assert r["status"] == "partial"
    r = h.verify_reference(fake, papers=[PAPER], scoring=v, author_veto=True)
    assert r["status"] == "likely hallucinated"


@pytest.mark.parametrize("v", h.SCORING_VERSIONS)
def test_missing_fields(v):
    blank = ref(authors=[], journal="")
    r = h.verify_reference(blank, papers=[PAPER], scoring=v)
    assert r["author_score"] is None and r["journal_score"] is None
    assert r["score"] == r["title_score"] == 100.0


def test_author_matching_rules():
    gr = ["Saulius Gražulis"]
    for v in h.SCORING_VERSIONS:
        assert h.author_score(["V. Fung"], ["Victor Fung"], v) == 100.0
        # documented limitations: no diacritic folding, "et al." counts
        assert h.author_score(["S. Grazulis"], gr, v) == 0.0
        etal = ["V. Fung", "et al."]
        assert h.author_score(etal, ["Victor Fung"], v) == 50.0
    # v2 also requires the first initial; v1 does not
    assert h.author_score(["H. Zhang"], ["Dinghuai Zhang"], "v1") == 100.0
    assert h.author_score(["H. Zhang"], ["Dinghuai Zhang"], "v2") == 0.0


def test_thresholds():
    assert h.classify(80) == "verified"
    assert h.classify(79.9) == "partial"
    assert h.classify(60) == "partial"
    assert h.classify(59.9) == "likely hallucinated"


def test_no_candidates_is_hallucinated():
    r = h.verify_reference(ref(), papers=[])
    assert r["score"] == 0.0
    assert r["status"] == "likely hallucinated"


def test_unknown_scoring_version():
    with pytest.raises(ValueError):
        h.verify_reference(ref(), papers=[PAPER], scoring="v3")


def test_regex_fallback_extracts_references():
    refs = h.extract_refs_regex(h.SAMPLE_TEXT)
    assert len(refs) == 4
    assert "AtomGPT" in refs[2]["raw"]
    assert all(r["title"] for r in refs)
