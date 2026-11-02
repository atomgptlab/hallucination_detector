"""
Re-score released per-citation outputs under alternative composite weights,
an author veto, and alternative thresholds.

The scoring stage is deterministic, so any per-citation table with the
columns ``title_score``, ``author_score`` and ``journal_score`` (blank for a
field that was not extracted) can be re-scored without new API calls.
Re-scoring uses the per-field scores of the candidate selected under the
default weights; under other weights a different top-k candidate could in
principle win, so the result is exact for that candidate and a close
approximation for the tool as a whole. The author scores themselves are
taken as stored, so ``--scoring`` changes only the missing-field rule.

Usage
-----
    python scripts/rescore_sensitivity.py --scoring v1 \
        results/neurips2025_v1.json
    python scripts/rescore_sensitivity.py --scoring v2 \
        results/citeaudit_300_v2.json

Input is a results JSON written by ``run_benchmark.py`` or a CSV. If it has a
ground-truth column (``GT`` with true = real, ``truth`` with
genuine/fabricated, or ``GT_label`` with real/fake), precision, recall and
false-positive rate are reported; a citation counts as flagged if it is not
``verified``. Rows without a score (e.g. not run) are dropped.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hallucination import SCORING_VERSIONS, classify  # noqa: E402

VARIANTS = [
    ("default 60/25/15", (0.60, 0.25, 0.15), False),
    ("40/45/15", (0.40, 0.45, 0.15), False),
    ("30/60/10", (0.30, 0.60, 0.10), False),
    ("40/60/0 (no journal)", (0.40, 0.60, 0.00), False),
    ("60/25/15 + author veto", (0.60, 0.25, 0.15), True),
]
THRESHOLDS = (75, 80, 85, 90)


def rescore(df, weights, veto=False, scoring="v2"):
    wt, wa, wj = weights
    t = df["title_score"].astype(float)
    a = df["author_score"].astype(float)
    j = df["journal_score"].astype(float)
    if scoring == "v1":
        comp = wt * t + wa * a.fillna(t) + wj * j.fillna(t)
    else:
        num = wt * t + wa * a.fillna(0) + wj * j.fillna(0)
        den = wt + wa * a.notna() + wj * j.notna()
        comp = (num / den).where(den > 0, 0.0)
    status = comp.map(classify)
    if veto:
        status = status.where(~(a == 0), "likely hallucinated")
    return comp, status


def truth_of(df):
    if "GT" in df.columns and df["GT"].notna().all():
        return ~df["GT"].astype(bool)
    if "truth" in df.columns:
        return df["truth"].eq("fabricated")
    if "GT_label" in df.columns:
        return df["GT_label"].eq("fake")
    return None


def metrics(flag, fab):
    tp = int((flag & fab).sum())
    fp = int((flag & ~fab).sum())
    n_fab, n_gen = int(fab.sum()), int((~fab).sum())
    prec = tp / max(tp + fp, 1)
    return prec, tp / max(n_fab, 1), fp / max(n_gen, 1), tp, fp


def summarise(df, name, scoring):
    fab = truth_of(df)
    print(f"\n== {name} (n={len(df)}, scoring={scoring})")
    hdr = f"{'variant':24s} {'verif':>6s} {'part':>6s} {'likely':>6s}"
    if fab is not None:
        hdr += (
            f" {'prec':>6s} {'recall':>6s} {'FPR':>6s} {'TP':>4s} {'FP':>4s}"
        )
    print(hdr)
    for label, w, veto in VARIANTS:
        _, st = rescore(df, w, veto, scoring)
        c = [int((st == s).sum()) for s in ("verified", "partial")]
        c.append(int((st == "likely hallucinated").sum()))
        line = f"{label:24s} " + " ".join(f"{x:6d}" for x in c)
        if fab is not None:
            p, r, f, tp, fp = metrics(st != "verified", fab)
            line += f" {p:6.2f} {r:6.2f} {f:6.2f} {tp:4d} {fp:4d}"
        print(line)
    comp, _ = rescore(df, (0.60, 0.25, 0.15), scoring=scoring)
    print("verified-threshold sweep (default weights):")
    for thr in THRESHOLDS:
        line = f"  S >= {thr}: {int((comp >= thr).sum())} verified"
        if fab is not None:
            p, r, f, tp, fp = metrics(comp < thr, fab)
            line += f"  prec {p:.2f} recall {r:.2f} FPR {f:.2f}"
        print(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--scoring", choices=SCORING_VERSIONS, default="v2")
    args = ap.parse_args(argv)
    for path in args.csv:
        if path.lower().endswith(".json"):
            with open(path, encoding="utf-8") as f:
                df = pd.DataFrame(json.load(f)["results"])
        else:
            df = pd.read_csv(path)
        score_col = next(
            (c for c in ("score", "composite_score") if c in df.columns), None
        )
        if score_col:
            df = df[df[score_col].notna()].copy()
            comp, _ = rescore(df, (0.60, 0.25, 0.15), scoring=args.scoring)
            err = np.abs(comp - df[score_col]).max()
            print(f"{path}: max |recomputed - reported| = {err:.2f}")
        summarise(df, os.path.basename(path), args.scoring)


if __name__ == "__main__":
    main()
