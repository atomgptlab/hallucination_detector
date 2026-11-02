"""Offline tests of the benchmark runner (network calls are stubbed)."""

import json
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import hallucination as h  # noqa: E402
import run_benchmark as rb  # noqa: E402
from test_scoring import PAPER  # noqa: E402

REAL = (
    "Victor Fung, Jiaxin Zhang, Eric Juarez. Benchmarking "
    "graph neural networks for materials chemistry. npj Computational "
    "Materials, 2021."
)
FAKE = (
    "John Doe, Jane Smith. Benchmarking graph neural networks for materials "
    "chemistry. npj Computational Materials, 2021."
)


@pytest.fixture
def offline(monkeypatch):
    def fake_llm(text, strict=False):
        authors, title, venue = text.split(". ")[:3]
        return [
            {
                "title": title,
                "authors": [a.strip() for a in authors.split(",")],
                "journal": venue.split(",")[0],
                "year": "2021",
                "raw": text,
            }
        ]

    def fake_search(query, api_key="", top_k=5, strict=False):
        return [PAPER], 1

    monkeypatch.setattr(h, "extract_refs_llm", fake_llm)
    monkeypatch.setattr(h, "search_semantic_scholar", fake_search)


def test_runner_end_to_end_and_resume(tmp_path, offline):
    data = tmp_path / "bench.json"
    data.write_text(
        json.dumps(
            {
                "data": [
                    {"citation": REAL, "GT": True, "source_type": "a"},
                    {"citation": FAKE, "GT": False, "source_type": "b"},
                ]
            }
        )
    )
    out = tmp_path / "out.json"
    args = ["--file", str(data), "--output", str(out), "--workers", "1"]
    assert rb.main(args + ["--limit", "1"]) == 0
    assert len(json.loads(out.read_text())["results"]) == 1
    assert rb.main(args) == 0  # resumes, checks only the second citation
    doc = json.loads(out.read_text())
    real, fake = doc["results"]
    assert real["status"] == "verified" and real["Correct"]
    assert fake["status"] == "partial" and fake["Correct"]
    s = doc["summary"]
    assert (s["TP"], s["FP"], s["TN"], s["FN"]) == (1, 0, 1, 0)
    assert set(s["by_source_type"]) == {"a", "b"}


def test_runner_fails_fast(tmp_path, monkeypatch, offline):
    def down(query, api_key="", top_k=5, strict=False):
        raise h.APIError("Semantic Scholar: HTTP 403")

    monkeypatch.setattr(h, "search_semantic_scholar", down)
    data = tmp_path / "bench.json"
    data.write_text(json.dumps([{"citation": REAL, "GT": True}]))
    out = tmp_path / "out.json"
    assert rb.main(["--file", str(data), "--output", str(out)]) == 2
    assert json.loads(out.read_text())["results"] == []


@pytest.mark.parametrize(
    "rec, gt",
    [
        ({"GT": True}, True),
        ({"GT": "false"}, False),
        ({"GT_label": "fake"}, False),
        ({"truth": "genuine"}, True),
        ({}, None),
    ],
)
def test_label_formats(rec, gt):
    assert rb._to_gt(rec) is gt


@pytest.mark.parametrize(
    "name, n",
    [
        ("neurips2025_hallucinated.json", 100),
        ("citeaudit_300.json", 300),
        ("materials_pilot.json", 26),
    ],
)
def test_bundled_datasets(name, n):
    recs = rb.load_records(os.path.join(ROOT, "data", name))
    assert len(recs) == n
    assert all(r["citation"] and r["GT"] is not None for r in recs)


def test_released_results_match_paper():
    def summary(name):
        with open(os.path.join(ROOT, "results", name)) as f:
            return json.load(f)["summary"]

    s = summary("citeaudit_300_v2.json")
    assert (s["TP"], s["FP"], s["precision"], s["recall"]) == (
        74,
        18,
        0.8043,
        0.4933,
    )
    s = summary("neurips2025_v1.json")
    assert s["status"] == {
        "verified": 6,
        "partial": 28,
        "likely hallucinated": 66,
    }
