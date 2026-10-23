"""Summarize setup_timeline_mpi.py runs: python summarize_timeline.py DIR1 DIR2 ... [--top N] [--min SECONDS]

Each DIR holds rank_*.json from one run (a rank count, or a set of independent serial jobs). Prints, per
run, the wall setup time over ranks and its spread, then the functions whose inclusive time (mean over
ranks, seconds) grows most between the first run and the last, with the max-over-ranks beside it.
"""
import glob
import json
import os
import sys

import numpy as np

args = sys.argv[1:]
top, floor = 25, 0.05
if '--top' in args:
    i = args.index('--top'); top = int(args[i + 1]); del args[i:i + 2]
if '--min' in args:
    i = args.index('--min'); floor = float(args[i + 1]); del args[i:i + 2]
runs = []
for d in args:
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, 'rank_*.json')))]
    runs.append((os.path.basename(d.rstrip('/')), recs))
print("%-14s %5s %8s %8s %8s %8s %8s   %s" % ("run", "ranks", "min", "median", "mean", "max", "std", "fine nodes owned/rank, local/owned"))
for name, recs in runs:
    t = np.array([r['setup_seconds'] for r in recs])
    last = 'level%d' % (recs[0]['levels'] - 1)
    owned = np.mean([r['sizes'][last]['nodes_owned'] for r in recs])
    local = np.mean([r['sizes'][last]['nodes_local'] for r in recs])
    print("%-14s %5d %8.2f %8.2f %8.2f %8.2f %8.2f   %.0f, %.2f" % (name, len(recs), t.min(), np.median(t), t.mean(), t.max(), t.std(), owned, local / owned))


def table(recs, key):
    names = set()
    for r in recs:
        names |= set(r[key])
    out = {}
    for nme in names:
        v = np.array([r[key].get(nme, 0.0) for r in recs])
        out[nme] = (v.mean(), v.max())
    return out


first, last = table(runs[0][1], 'inclusive'), table(runs[-1][1], 'inclusive')
rows = []
for nme in set(first) | set(last):
    a, b = first.get(nme, (0, 0)), last.get(nme, (0, 0))
    if max(a[0], b[0]) >= floor:
        rows.append((b[0] - a[0], nme, a, b))
rows.sort(reverse=True)
print("\nInclusive seconds, mean over ranks (max over ranks): %s -> %s, largest growth first" % (runs[0][0], runs[-1][0]))
print("%9s  %-18s %-18s  %s" % ("growth", runs[0][0], runs[-1][0], "function"))
for g, nme, a, b in rows[:top]:
    print("%+9.2f  %7.2f (%6.2f)   %7.2f (%6.2f)   %s" % (g, a[0], a[1], b[0], b[1], nme))
