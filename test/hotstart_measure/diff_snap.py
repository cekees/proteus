"""Diff two snapshots: which state differs after a hotstart, and by how much.

    python diff_snap.py TRUE.pkl HOT.pkl [floor]
"""
import pickle
import sys
import numpy as np

a, b = (pickle.load(open(f, "rb")) for f in sys.argv[1:3])
floor = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-12
same, diff, only_a, only_b = 0, [], [], []
for k in sorted(set(a) | set(b)):
    if k not in b:
        only_a.append(k); continue
    if k not in a:
        only_b.append(k); continue
    x, y = np.asarray(a[k], float), np.asarray(b[k], float)
    if x.shape != y.shape:
        diff.append((k, np.inf, np.inf, "shape %s vs %s" % (x.shape, y.shape))); continue
    d = np.abs(x - y).max() if x.size else 0.0
    sc = max(np.abs(x).max() if x.size else 0.0, 1e-300)
    if d <= floor * max(sc, 1.0):
        same += 1
    else:
        diff.append((k, d, d / sc, ""))
print("identical: %d   differing: %d   only in uninterrupted: %d   only in hotstart: %d"
      % (same, len(diff), len(only_a), len(only_b)))
print("\nDIFFERING (abs, rel to max|true|)")
for k, d, r, note in sorted(diff, key=lambda t: -t[2]):
    print("  %-70s %10.3e %10.3e %s" % (k[:70], d, r, note))
print("\nONLY IN UNINTERRUPTED RUN: %d" % len(only_a)); [print("  ", k) for k in only_a[:40]]
print("ONLY IN HOTSTART: %d" % len(only_b)); [print("  ", k) for k in only_b[:40]]
