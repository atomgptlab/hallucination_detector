"""
Run the detector on a labelled list of citations and score it.

Each citation string is parsed into fields (LLM, or the regex fallback),
matched against Semantic Scholar and labelled. A citation counts as
*flagged* (predicted fake) if its status is not ``verified``.

The runner is

* **concurrent**: ``--workers N`` threads share one Semantic Scholar
  rate limiter (``SS_MIN_INTERVAL`` seconds between calls);
* **resumable**: results are written after every citation, and re-running
  the same command skips citations already in the output file;
* **fail-fast**: an authentication, quota or persistent API failure stops
  the run with exit code 2 instead of being scored as a hallucination
  (use ``--lenient`` to reproduce the web tool, which scores such
  citations 0).

Input: a JSON file (a list, or an object with a ``data`` list) or a CSV.
Each record needs a citation string (``citation`` or ``reference``) and,
for evaluation, a label: ``GT`` (true = real), ``GT_label`` (real / fake)
or ``truth`` (genuine / fabricated). The CiteAudit ``benchmark.json``
format is read as is.

Usage
-----
    python run_benchmark.py --file data/materials_pilot.json \\
        --output results/materials_pilot.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import hallucination as h

log = logging.getLogger("run_benchmark")

KEEP = ("source_type", "category", "label")


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def _to_gt(rec: dict):
    """Ground truth as a bool (True = real citation) or None."""
    if "GT" in rec and rec["GT"] not in (None, ""):
        v = rec["GT"]
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "real", "yes")
        return bool(v)
    for key, real in (("GT_label", "real"), ("truth", "genuine")):
        if rec.get(key) not in (None, ""):
            return str(rec[key]).strip().lower() == real
    return None


def load_records(path: str) -> list[dict]:
    if path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            raw = list(csv.DictReader(f))
    else:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw = raw.get("data", [])
    records = []
    for i, rec in enumerate(raw):
        citation = rec.get("citation") or rec.get("reference") or ""
        out = {"index": i, "citation": str(citation).strip()}
        out.update({k: rec[k] for k in KEEP if rec.get(k) not in (None, "")})
        out["GT"] = _to_gt(rec)
        records.append(out)
    return records


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _ratio(a: int, b: int):
    return round(a / b, 4) if b else None


def confusion(results: list[dict]) -> dict:
    """Fake citation = positive class; flagged = status != verified."""
    lab = [r for r in results if r.get("GT") is not None]
    tp = sum(1 for r in lab if not r["GT"] and not r["Pred"])
    fn = sum(1 for r in lab if not r["GT"] and r["Pred"])
    tn = sum(1 for r in lab if r["GT"] and r["Pred"])
    fp = sum(1 for r in lab if r["GT"] and not r["Pred"])
    p, r = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = round(2 * p * r / (p + r), 4) if p and r else None
    return {
        "labelled": len(lab),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "accuracy": _ratio(tp + tn, len(lab)),
        "precision": p,
        "recall": r,
        "f1": f1,
        "false_positive_rate": _ratio(fp, fp + tn),
    }


def summarise(results: list[dict]) -> dict:
    statuses = ("verified", "partial", "likely hallucinated")
    summary = {
        "total": len(results),
        "status": {
            s: sum(1 for r in results if r["status"] == s) for s in statuses
        },
    }
    summary.update(confusion(results))
    groups = sorted({r.get("source_type") for r in results} - {None})
    if len(groups) > 1:
        summary["by_source_type"] = {
            g: confusion([r for r in results if r.get("source_type") == g])
            for g in groups
        }
    return summary


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def check_one(rec: dict, args) -> dict:
    ref, method = h.parse_citation(rec["citation"], strict=not args.lenient)
    res = h.verify_reference(
        ref,
        top_k=args.top_k,
        threshold=args.threshold,
        author_veto=args.author_veto,
        scoring=args.scoring,
        strict=not args.lenient,
    )
    pred = res["status"] == "verified"
    out = dict(rec)
    out.update(
        {
            "Pred": pred,
            "Correct": None if rec["GT"] is None else pred == rec["GT"],
            "extraction_method": method,
            "extracted_title": ref.get("title", ""),
        }
    )
    out.update(res)
    return out


def select(records: list[dict], args) -> list[dict]:
    if args.gt != "all":
        want = args.gt == "true"
        records = [r for r in records if r["GT"] is want]
    if args.source:
        records = [r for r in records if r.get("source_type") == args.source]
    end = None if args.limit is None else args.start + args.limit
    return records[args.start : end]


def write(path: str, results: list[dict], meta: dict) -> None:
    results = sorted(results, key=lambda r: r["index"])
    doc = {"config": meta, "results": results, "summary": summarise(results)}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[1].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--file", required=True, help="benchmark JSON or CSV")
    ap.add_argument("--output", required=True, help="results JSON")
    ap.add_argument("--gt", choices=("all", "true", "false"), default="all")
    ap.add_argument("--source", help="keep one source_type only")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=h.VERIFIED_THRESHOLD)
    ap.add_argument("--scoring", choices=h.SCORING_VERSIONS, default="v2")
    ap.add_argument("--author-veto", action="store_true")
    ap.add_argument(
        "--lenient",
        action="store_true",
        help="score API failures 0 (web-tool behaviour) instead of stopping",
    )
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    todo = select(load_records(args.file), args)
    done = []
    if not args.no_resume and os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            done = json.load(f).get("results", [])
        seen = {r["index"] for r in done}
        todo = [r for r in todo if r["index"] not in seen]
        log.info("resuming: %d done, %d to go", len(done), len(todo))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    meta = {
        "file": os.path.basename(args.file),
        "scoring_version": args.scoring,
        "weights": dict(zip(("title", "authors", "journal"), h.WEIGHTS)),
        "verified_threshold": args.threshold,
        "partial_threshold": h.PARTIAL_THRESHOLD,
        "top_k": args.top_k,
        "author_veto": args.author_veto,
        "extraction_model": h.LLM_MODEL or "regex fallback",
        "updated_utc": datetime.now(timezone.utc).isoformat("T", "seconds"),
    }
    lock = threading.Lock()
    failed = None
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(check_one, r, args): r for r in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            rec = futures[fut]
            try:
                res = fut.result()
            except h.APIError as e:
                failed = e
                for f in futures:
                    f.cancel()
                break
            with lock:
                done.append(res)
                write(args.output, done, meta)
            log.info(
                "[%d/%d] #%d %-19s %5.1f  GT=%s",
                n,
                len(todo),
                rec["index"],
                res["status"],
                res["score"],
                {True: "real", False: "fake", None: "?"}[rec["GT"]],
            )
    write(args.output, done, meta)
    if failed:
        log.error("stopped: %s (re-run the same command to resume)", failed)
        return 2
    s = summarise(done)
    log.info(json.dumps({k: v for k, v in s.items() if k != "by_source_type"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
