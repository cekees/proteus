"""Compare two vortex2D archives step by step: per field, the relative
L2 and Linf difference at each archived step >= from_step, and the step at
which it first exceeds a floor.

    python compare.py A.h5 B.h5 NAME FROM_STEP
"""
import sys
import h5py
import numpy as np

a, b, from_step = h5py.File(sys.argv[1], "r"), h5py.File(sys.argv[2], "r"), int(sys.argv[4])
fields = ["u", "phid", "vof", "phiCorr"]
steps = sorted(int(k[3:]) for k in a if k.startswith("u_t") and k[3:].isdigit())
print("%-6s %-8s %12s %12s" % ("step", "field", "rel_L2", "Linf"))
last = {}
for s in steps:
    if s < from_step:
        continue
    for f in fields:
        k = "%s_t%d" % (f, s)
        if k not in a or k not in b:
            continue
        x, y = np.array(a[k]).ravel(), np.array(b[k]).ravel()
        d = x - y
        l2 = np.linalg.norm(d) / max(np.linalg.norm(x), 1e-300)
        li = np.abs(d).max()
        last[f] = (s, l2, li)
        if s in (from_step, from_step + 1, from_step + 2, from_step + 5, from_step + 10, steps[-1]):
            print("%-6d %-8s %12.3e %12.3e" % (s, f, l2, li))
print("FINAL", sys.argv[3], " ".join("%s:L2=%.2e,Linf=%.2e" % (f, v[1], v[2]) for f, v in last.items()))
