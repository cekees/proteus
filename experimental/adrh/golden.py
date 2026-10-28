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


def cube_quadrature(nd):
    """Tensor Gauss rule (3 points per axis) on proteus's reference cube [-1, 1]^nd."""
    g = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]); w = np.array([5.0, 8.0, 5.0]) / 9.0
    pts = np.array(np.meshgrid(*([g] * nd), indexing="ij")).reshape(nd, -1).T
    wts = np.prod(np.array(np.meshgrid(*([w] * nd), indexing="ij")).reshape(nd, -1).T, axis=1)
    return pts, wts


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


EDGES = {2: [(0, 1), (1, 2), (0, 2)], 3: [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]}


def space_coords(nd, space, xv, rng):
    """The DOF coordinates (nE, nDOF, 3) of elements of a Lagrange space. P1: the vertices xv (nE, nd+1, 3); P2: plus the edge
    midpoints in proteus's order; Q1: a perturbed cube in proteus's node order (the perturbation keeps the map invertible)."""
    nE = xv.shape[0]
    if space == "p1":
        return xv
    if space == "p2":
        mids = [0.5 * (xv[:, a] + xv[:, b]) for a, b in EDGES[nd]]
        return np.concatenate([xv, np.stack(mids, axis=1)], axis=1)
    sgn = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)[: 2 ** nd]
    X = np.zeros((nE, 2 ** nd, 3))
    base = rng.uniform(0, 1, (nE, 1, nd))
    X[:, :, :nd] = base + 0.1 * (sgn[None, :, :nd] + 1.0) + rng.uniform(-0.01, 0.01, (nE, 2 ** nd, nd))
    return X


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
        for space in ("p2", "q1"):
            qps, qws = tet_quadrature(nd) if space == "p2" else cube_quadrature(nd)
            Xs = space_coords(nd, space, xs[nd], rng)
            us = rng.uniform(0.2, 1.0, (nE, 1, Xs.shape[1]))
            out["ad_sine_%dd_%s" % (nd, space)] = ("advection_diffusion_sine", (nd, space, b[:nd].copy(), 0.2, omega[:nd].copy(), 0.3, Xs, us, qps, qws))
    # faces and the transient entry points, in every space
    x0 = np.array([0.35, 0.4, 0.45])
    for nd in (2, 3):
        for space in ("p1", "p2", "q1"):
            Xs = space_coords(nd, space, xs[nd], rng)
            nDOF = Xs.shape[1]
            qpf_, qwf_ = tet_quadrature(nd - 1) if space != "q1" else cube_quadrature(nd - 1)
            nfaces = 2 * nd if space == "q1" else nd + 1
            kind = rng.integers(1, 5, nE).astype(np.int32)
            local = rng.integers(0, nfaces, nE).astype(np.int32)
            uu = rng.uniform(0.2, 1.0, (nE, 1, nDOF))
            qps_, qws_ = tet_quadrature(nd) if space != "q1" else cube_quadrature(nd)
            if space != "p1":                       # p1 faces and the sine element kernel are covered below
                out["faces_%dd_%s" % (nd, space)] = ("advection_diffusion_sine_faces",
                    (nd, space, b[:nd].copy(), 0.2, omega[:nd].copy(), 0.3, local, kind, Xs, uu, qpf_, qwf_, 2.0, 1.0, 1.0, 0.7, 0.4))
            uo = rng.uniform(0.2, 1.0, (nE, 1, nDOF))
            out["transient_%dd_%s" % (nd, space)] = ("transient_adr",
                (nd, space, b[:nd].copy(), 0.02, 0.3, Xs, uu, uo, qps_, qws_, 0.5, 0.05))
            out["transient_faces_%dd_%s" % (nd, space)] = ("transient_adr_faces",
                (nd, space, b[:nd].copy(), 0.02, 0.3, 0.3, 0.7, x0[:nd].copy(), local, kind, Xs, uu, qpf_, qwf_, 0.5, 2.0, 1.0, 1.0, 0.7, 0.4))
    for nd in (2, 3):
        qpn, qwn = tet_quadrature(nd)
        un = rng.uniform(0.2, 1.0, (nE, 1, nd + 1))
        out["ad_sine_%dd" % nd] = ("advection_diffusion_sine", (nd, "p1", b[:nd].copy(), 0.2, omega[:nd].copy(), 0.3, xs[nd], un, qpn, qwn))
        qpf, qwf = tet_quadrature(nd - 1)
        kind = rng.integers(1, 5, nE).astype(np.int32)
        local = rng.integers(0, nd + 1, nE).astype(np.int32)
        out["faces_%dd" % nd] = ("advection_diffusion_sine_faces",
                                 (nd, "p1", b[:nd].copy(), 0.2, omega[:nd].copy(), 0.3, local, kind, xs[nd], un, qpf, qwf,
                                  2.0, 1.0, 1.0, 0.7, 0.4))
    # the nodal passes and the head-bounded FCT limiter on a 2D triangle mesh's pattern: two materials, a random state
    import scipy.sparse as sp
    nx = ny = 30
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    a = (i * ny + j).ravel(); bb = ((i + 1) * ny + j).ravel(); c = ((i + 1) * ny + j + 1).ravel(); d = (i * ny + j + 1).ravel()
    conn = np.stack([np.stack([a, bb, c], 1), np.stack([a, c, d], 1)], 1).reshape(-1, 3)
    n = nx * ny
    P = sp.coo_matrix((np.ones(conn.size * 3), (np.repeat(conn, 3, 1).ravel(), np.tile(conn, (1, 3)).ravel())), shape=(n, n)).tocsr()
    P.sort_indices()
    indptr, indices = P.indptr.astype(np.int32), P.indices.astype(np.int32)
    tpos = np.asarray(m.transpose_map(indptr, indices)) if hasattr(m, "transpose_map") else None
    mat = rng.integers(0, 2, n).astype(np.int32)
    uh = rng.uniform(-8.0, -0.5, n); uo_ = rng.uniform(-8.0, -0.5, n)
    alpha = np.array([3.35, 1.0]); nvg = np.array([2.0, 1.6]); thr = np.array([0.102, 0.08]); thsr = np.array([0.266, 0.35])
    out["richards_m_nodes"] = ("richards_m_nodes", (alpha, nvg, thr, thsr, 0, mat, uh, 1.0, 0.0))
    out["head_extrema"] = ("head_extrema", (indptr, indices, uh, uo_))
    if tpos is not None:
        Mc = rng.uniform(0.0, 1.0, indices.size); A = rng.normal(size=indices.size); ML = rng.uniform(0.5, 1.5, n)
        pm = np.maximum(uh, uo_) + 0.5; pn = np.minimum(uh, uo_) - 0.5
        mL_ = rng.uniform(0.1, 0.3, n); mo_ = rng.uniform(0.1, 0.3, n); mmax = np.maximum(mL_, mo_) + 0.02; mmin = np.minimum(mL_, mo_) - 0.02
        out["fct_limit_flux_bounded"] = ("fct_limit_flux_bounded", (indptr, indices, tpos, Mc, A, ML, mL_, mo_, mmax, mmin, 0.01))
        # implicit FCT: the prelimiter, and the limited datum from an antisymmetric flux
        At = A - A[tpos]; At[np.repeat(np.arange(n), np.diff(indptr)) == indices] = 0.0
        alpha_nv = np.asarray(m.nodal_variation(indptr, indices, Mc, uh, 3.0)) if hasattr(m, "nodal_variation") else None
        out["nodal_variation"] = ("nodal_variation", (indptr, indices, Mc, uh, 3.0))
        # the edge-difference upwind and antidiffusive passes (compressible flux): S, dphi and its partials, kr
        Sx = -np.abs(rng.normal(size=indices.size)); Sx = 0.5 * (Sx + Sx[tpos])
        dph = rng.normal(size=indices.size); dph = dph - dph[tpos]
        kr_ = rng.uniform(0.0, 1.0, n); dkr_ = rng.uniform(0.0, 1.0, n)
        out["upwind_flux_edges"] = ("upwind_flux_edges", (indptr, indices, Sx, dph, -np.ones(indices.size), np.ones(indices.size), kr_, dkr_))
        out["antidiffusive_flux_edges"] = ("antidiffusive_flux_edges", (indptr, indices, rng.normal(size=indices.size), Sx, dph, kr_))
        if alpha_nv is not None:
            out["implicit_fct_limit"] = ("implicit_fct_limit", (indptr, indices, tpos, Mc, At, ML, mL_, mo_, np.maximum(mL_, mo_) + 0.05, np.minimum(mL_, mo_) - 0.05, alpha_nv, 0.01))
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
        if not isinstance(r, tuple):
            r = (r,)
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
