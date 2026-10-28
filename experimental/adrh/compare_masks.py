"""Constexpr sparsity masks versus the runtime-loop kernel: same numbers, less work.

Checks that elementKernelMasked reproduces elementKernel (Poisson, the coupled two-component system, a
five-component chain), then times residual + Jacobian for: the runtime-loop kernel (every block),
the masked kernel with all-true masks (the cost of the unrolled loops themselves), and the masked
kernel with the functor's real masks.   usage: python compare_masks.py [elements]
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adrh_proto as a
from proteus import Quadrature

nE = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
rng = np.random.default_rng(0)
xn = rng.uniform(0, 1, (nE, 4, 3))
xn[:, 1:, :] = xn[:, :1, :] + rng.uniform(0.1, 0.3, (nE, 3, 3)) * np.eye(3)
quad = Quadrature.SimplexGaussQuadrature(3, 4)
qp = np.array([list(p)[:3] for p in quad.points])
qw = np.array(quad.weights)


def rel(x, y):
    return np.abs(x - y).max() / max(np.abs(y).max(), 1e-300)


def best(f, n=3):
    t = []
    for _ in range(n):
        s = time.perf_counter(); f(); t.append(time.perf_counter() - s)
    return min(t)


u1 = rng.uniform(0.2, 1.0, (nE, 1, 4))
R0, J0 = a.poisson3d(xn, u1, qp, qw)
R1, J1 = a.poisson3d_masked(xn, u1, qp, qw)
print("Poisson (nc=1):       masked vs runtime  residual %.1e  jacobian %.1e" % (rel(R1, R0), rel(J1, J0)))
u2 = rng.uniform(0.2, 1.0, (nE, 2, 4)); o2 = rng.uniform(0.2, 1.0, (nE, 2, 4))
R0, J0 = a.coupled2(xn, u2, o2, qp, qw, 0.7, 0.05)
R1, J1 = a.coupled2_masked(xn, u2, o2, qp, qw, 0.7, 0.05)
print("Coupled (nc=2, mass): masked vs runtime  residual %.1e  jacobian %.1e" % (rel(R1, R0), rel(J1, J0)))
u5 = rng.uniform(0.2, 1.0, (nE, 5, 4)); o5 = rng.uniform(0.2, 1.0, (nE, 5, 4))
R0, J0 = a.chain5(xn, u5, o5, qp, qw, 0.7, 0.05, False, False)
R1, J1 = a.chain5(xn, u5, o5, qp, qw, 0.7, 0.05, True, True)
print("Chain (nc=5, mass):   masked vs runtime  residual %.1e  jacobian %.1e" % (rel(R1, R0), rel(J1, J0)))
d = rng.uniform(-1, 1, u5.shape); eps = 1e-6
fd = (a.chain5(xn, u5 + eps * d, o5, qp, qw, 0.7, 0.05, True, True)[0] - a.chain5(xn, u5 - eps * d, o5, qp, qw, 0.7, 0.05, True, True)[0]) / (2 * eps)
print("Chain Jacobian vs central differences (masked, sparse): %.1e" % rel(np.einsum('eiajb,ejb->eia', J1, d), fd))

print("\nelement residual + Jacobian, %d tets, %d points, best of 3 (microseconds per element):" % (nE, len(qw)))
rows = [
    ("Poisson nc=1", "runtime loops", lambda: a.poisson3d(xn, u1, qp, qw)),
    ("Poisson nc=1", "masked", lambda: a.poisson3d_masked(xn, u1, qp, qw)),
    ("Coupled nc=2", "runtime loops", lambda: a.coupled2(xn, u2, o2, qp, qw, 0.7, 0.05)),
    ("Coupled nc=2", "masked", lambda: a.coupled2_masked(xn, u2, o2, qp, qw, 0.7, 0.05)),
    ("Chain nc=5", "runtime loops, all blocks", lambda: a.chain5(xn, u5, o5, qp, qw, 0.7, 0.05, False, False)),
    ("Chain nc=5", "masked, all-true masks", lambda: a.chain5(xn, u5, o5, qp, qw, 0.7, 0.05, True, False)),
    ("Chain nc=5", "masked, real masks", lambda: a.chain5(xn, u5, o5, qp, qw, 0.7, 0.05, True, True)),
]
for case, kind, f in rows:
    print("  %-14s %-28s %8.2f" % (case, kind, 1e6 * best(f) / nE))
