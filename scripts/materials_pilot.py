"""
Reproduce the materials-science pilot (genuine vs fabricated citations).

1. Harvest genuine references from three seed materials manuscripts through
   the Semantic Scholar Graph API.
2. Load the fabricated materials citations (JSON list of objects with
   ``title``, ``authors``, ``journal``, ``year``).
3. Score every reference with :func:`hallucination.verify_reference` using
   the default weights and thresholds.

Usage
-----
    export SS_API_KEY=...   # optional, raises the rate limit
    python scripts/materials_pilot.py data/materials_pilot_fabricated.json \\
        --genuine data/materials_pilot_genuine.json \\
        -o results/materials_pilot_harvested_fabricated.csv
"""

import argparse
import csv
import json
import os
import sys
import time

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hallucination import SS_API_KEY, verify_reference  # noqa: E402

S2 = "https://api.semanticscholar.org/graph/v1"
SEEDS = {
    "arXiv:2007.01831": "JARVIS-DFT",
    "arXiv:2106.01829": "ALIGNN",
    "arXiv:2508.08349": "DiffractGPT",
}


def harvest(seed, per_seed=7):
    headers = {"x-api-key": SS_API_KEY} if SS_API_KEY else {}
    r = httpx.get(
        f"{S2}/paper/{seed}/references",
        params={"fields": "title,year,authors,venue", "limit": 60},
        headers=headers,
        timeout=40,
    )
    r.raise_for_status()
    refs = []
    for item in r.json().get("data", []):
        p = item.get("citedPaper") or {}
        authors = [a["name"] for a in (p.get("authors") or [])][:6]
        if p.get("title") and authors and p.get("venue"):
            refs.append(
                {
                    "title": p["title"],
                    "authors": authors,
                    "journal": p["venue"],
                    "year": str(p.get("year") or ""),
                }
            )
    return refs[:per_seed]


def evaluate(refs, truth, sleep=1.4):
    rows = []
    for x in refs:
        ref = {
            "title": x["title"],
            "authors": x.get("authors", []),
            "journal": x.get("journal", ""),
            "year": str(x.get("year", "")),
            "raw": x["title"],
        }
        v = verify_reference(ref, top_k=5)
        rows.append(
            {
                "truth": truth,
                "title": x["title"],
                "status": v["status"],
                "score": v["score"],
                "title_score": v["title_score"],
                "author_score": v["author_score"],
                "journal_score": v["journal_score"],
                "best_match": v["best_match"],
            }
        )
        time.sleep(sleep)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("fabricated", help="JSON file of fabricated citations")
    ap.add_argument("-o", "--output", default="materials_pilot_results.csv")
    ap.add_argument(
        "--genuine",
        help="JSON of genuine references: read if it exists, otherwise "
        "harvested from the seed manuscripts and written here",
    )
    args = ap.parse_args(argv)

    if args.genuine and os.path.exists(args.genuine):
        with open(args.genuine, encoding="utf-8") as f:
            genuine = json.load(f)
    else:
        genuine = []
        for seed, name in SEEDS.items():
            for ref in harvest(seed):
                genuine.append(dict(ref, seed=name))
            time.sleep(2)
        if args.genuine:
            with open(args.genuine, "w", encoding="utf-8") as f:
                json.dump(genuine, f, indent=2, ensure_ascii=False)
    with open(args.fabricated, encoding="utf-8") as f:
        fabricated = json.load(f)

    rows = evaluate(genuine, "genuine") + evaluate(fabricated, "fabricated")
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def flagged(r):
        return r["status"] != "verified"

    tp = sum(r["truth"] == "fabricated" and flagged(r) for r in rows)
    tn = sum(r["truth"] == "genuine" and not flagged(r) for r in rows)
    print(f"sensitivity = {tp}/{len(fabricated)}")
    print(f"specificity = {tn}/{len(genuine)}")
    print(f"per-citation results written to {args.output}")


if __name__ == "__main__":
    main()
