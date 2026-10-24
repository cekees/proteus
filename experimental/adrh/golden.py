"""The ADRH element kernels on fixed inputs, to compare one build of the module with another.

    python golden.py save FILE.npz               # the module named by ADRH_MODULE (default adrh_proto)
    python golden.py compare FILE.npz [module]   # bitwise and relative difference against a saved run
    python golden.py against MODULE              # adrh_proto against MODULE on the functions both have

Used to show that restructuring the host kernel changed no bit (save before, compare after), and that the
Kokkos module (adrh_kokkos, any backend and thread count) reproduces the host kernel.
"""
import importlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def tet_quadrature(nd, order=4):
    from proteus import Quadrature
    q = Quadrature.SimplexGaussQuadrature(nd, order)
    return np.array([list(p)[:nd] for p in q.points]), np.array(q.weights)


def inputs(nE=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = {}
    for nd in (2, 3):
        xn = np.zeros((nE, nd + 1, 3))
        xn[:, 0, :nd] = rng.uniform(0, 1, (nE, nd))
        for k in range(nd):                                  # a perturbed right simplex: well conditioned, not axis aligned
            xn[:, k + 1, :nd] = xn[:, 0, :nd] + 0.2 * np.eye(nd)[k] + rng.uniform(-0.04, 0.04, (nE, nd))
        d[nd] = xn
    return d, rng


def calls(m, nE=2000):
    """name -> (module function name, args). Only functions the module has are run."""
    xs, rng = inputs(nE)
    xn3 = xs[3]
    qp, qw = tet_quadrature(3)
    u1 = rng.uniform(0.2, 1.0, (nE, 1, 4))
    u2 = rng.uniform(0.2, 1.0, (nE, 2, 4)); o2 = rng.uniform(0.2, 1.0, (nE, 2, 4))
    u5 = rng.uniform(0.2, 1.0, (nE, 5, 4)); o5 = rng.uniform(0.2, 1.0, (nE, 5, 4))
    out = {
        "poisson3d": ("poisson3d", (xn3, u1, qp, qw)),
        "poisson3d_masked": ("poisson3d_masked", (xn3, u1, qp, qw)),
        "coupled2": ("coupled2", (xn3, u2, o2, qp, qw, 0.7, 0.05)),
        "coupled2_masked": ("coupled2_masked", (xn3, u2, o2, qp, qw, 0.7, 0.05)),
        "coupled2_steady": ("coupled2_masked", (xn3, u2, None, qp, qw, 0.7, 0.0)),
    }
    for masked in (False, True):
        for sparse in (False, True):
            if sparse and not masked:
                continue
            out["chain5_m%d_s%d" % (masked, sparse)] = ("chain5", (xn3, u5, o5, qp, qw, 0.7, 0.05, masked, sparse))
    b = np.array([1.0, 0.5, 0.25]); omega = np.array([3.0, 2.0, 4.0])
    for nd in (2, 3):
        qpn, qwn = tet_quadrature(nd)
        un = rng.uniform(0.2, 1.0, (nE, 1, nd + 1))
        out["ad_sine_%dd" % nd] = ("advection_diffusion_sine", (nd, b[:nd].copy(), 0.2, omega[:nd].copy(), 0.3, xs[nd], un, qpn, qwn))
        qpf, qwf = tet_quadrature(nd - 1)
        kind = rng.integers(1, 5, nE).astype(np.int32)
        local = rng.integers(0, nd + 1, nE).astype(np.int32)
        out["faces_%dd" % nd] = ("advection_diffusion_sine_faces",
                                 (nd, b[:nd].copy(), 0.2, omega[:nd].copy(), 0.3, local, kind, xs[nd], un, qpf, qwf,
                                  2.0, 1.0, 1.0, 0.7, 0.4))
    return out


def run(m, nE=2000):
    res = {}
    for name, (fn, args) in calls(m, nE).items():
        if not hasattr(m, fn):
            continue
        try:
            r = getattr(m, fn)(*args)
        except NotImplementedError:
            continue
        for k, v in enumerate(r):
            res["%s.%s" % (name, "RJ"[k])] = np.asarray(v)
    return res


def rel(x, y):
    return float(np.abs(x - y).max() / max(np.abs(y).max(), 1e-300))


def load(name):
    return importlib.import_module(name)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "save":
        res = run(load(os.environ.get("ADRH_MODULE", "adrh_proto")))
        np.savez(sys.argv[2], **res)
        print("saved %d arrays" % len(res))
    elif cmd in ("compare", "against"):
        if cmd == "compare":
            ref = dict(np.load(sys.argv[2]))
            cur = run(load(sys.argv[3] if len(sys.argv) > 3 else os.environ.get("ADRH_MODULE", "adrh_proto")))
        else:
            ref = run(load("adrh_proto")); cur = run(load(sys.argv[2]))
        worst = 0.0; same = 0
        for k in sorted(ref):
            if k not in cur:
                continue
            bit = bool(np.array_equal(ref[k], cur[k])); same += bit
            r = rel(cur[k], ref[k]); worst = max(worst, r)
            print("%-34s %s  max rel difference %.1e" % (k, "bitwise equal" if bit else "differs      ", r))
        print("\n%d of %d arrays bitwise equal; worst relative difference %.1e" % (same, len([k for k in ref if k in cur]), worst))
