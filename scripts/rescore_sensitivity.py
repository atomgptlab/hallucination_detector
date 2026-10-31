"""
Re-score released per-citation outputs under alternative composite weights,
an author veto, and alternative thresholds.

The scoring stage is deterministic, so any per-citation table with the
columns ``title_score``, ``author_score`` and ``journal_score`` (blank for a
field that was not extracted) can be re-scored without new API calls.
Re-scoring uses the per-field scores of the candidate selected under the
default weights; under other weights a different top-k candidate could in
principle win, so the result is exact for that candidate and a close
approximation for the tool as a whole.

Usage
-----
    python scripts/rescore_sensitivity.py results.csv [more.csv ...]

Each file is summarised separately. If a file has a ``truth`` column with
values ``genuine``/``fabricated``, precision, recall and false-positive rate
are also reported (a citation is "flagged" if it is not ``verified``).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hallucination import classify  # noqa: E402

VARIANTS = [
    ("default 60/25/15", (0.60, 0.25, 0.15), False),
    ("40/45/15", (0.40, 0.45, 0.15), False),
    ("40/60/0 (no journal)", (0.40, 0.60, 0.00), False),
    ("30/60/10", (0.30, 0.60, 0.10), False),
    ("60/25/15 + author veto", (0.60, 0.25, 0.15), True),
    ("40/45/15 + author veto", (0.40, 0.45, 0.15), True),
]


def rescore(df, weights, veto=False):
    wt, wa, wj = weights
    t = df["title_score"].astype(float)
    a = df["author_score"].astype(float)
    j = df["journal_score"].astype(float)
    comp = wt * t + wa * a.fillna(t) + wj * j.fillna(t)
    status = comp.map(classify)
    if veto:
        status = status.where(~(a == 0), "likely hallucinated")
    return comp, status


def summarise(df, name):
    print(f"\n== {name} (n={len(df)})")
    has_truth = "truth" in df.columns
    hdr = f"{'variant':26s} {'verified':>8s} {'partial':>8s} {'likely':>8s}"
    if has_truth:
        hdr += f" {'prec':>6s} {'recall':>6s} {'FPR':>6s}"
    print(hdr)
    for label, w, veto in VARIANTS:
        _, st = rescore(df, w, veto)
        counts = [int((st == s).sum()) for s in ("verified", "partial")]
        counts.append(int((st == "likely hallucinated").sum()))
        line = f"{label:26s} " + " ".join(f"{c:8d}" for c in counts)
        if has_truth:
            flag = st != "verified"
            fab = df["truth"].eq("fabricated")
            tp = int((flag & fab).sum())
            fp = int((flag & ~fab).sum())
            prec = tp / max(tp + fp, 1)
            rec = tp / max(int(fab.sum()), 1)
            fpr = fp / max(int((~fab).sum()), 1)
            line += f" {prec:6.2f} {rec:6.2f} {fpr:6.2f}"
        print(line)

    comp, _ = rescore(df, (0.60, 0.25, 0.15))
    print("verified-threshold sweep (default weights):")
    for thr in (70, 75, 80, 85, 90):
        print(f"  S >= {thr}: {int((comp >= thr).sum())} verified")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", nargs="+")
    args = ap.parse_args(argv)
    for path in args.csv:
        df = pd.read_csv(path)
        if "score" in df.columns:
            comp, _ = rescore(df, (0.60, 0.25, 0.15))
            err = np.abs(comp - df["score"]).max()
            print(f"{path}: max |recomputed - reported| = {err:.2f}")
        summarise(df, os.path.basename(path))


if __name__ == "__main__":
    main()
