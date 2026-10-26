"""The K-upwinded low-order operator of Richards' equation (lowOrder.h, richards.h modes 1 and 2) on 3D tetrahedra.

Proteus's low-order and FCT schemes write the flux on the total head phi = psi - g.x and take the relative permeability at the UPWIND node of each
edge, F_ij = kr_up c_ij (phi_j - phi_i), c_ij = max(0, -S_ij), S the stiffness K_s grad phi_i . grad phi_j. This checks, for both retention models
(van Genuchten, Brooks-Corey-Mualem):

  1. the stiffness matrices of the element kernel (mode 1: S; mode 2: K_r frozen at the quadrature points) are symmetric with zero row sums;
  2. upwind_flux against a straightforward edge loop in numpy, its Jacobian against finite differences of the residual, and conservation
     (the residual sums to zero and so does every column of the Jacobian);
  3. the antidiffusive flux is antisymmetric and the FCT built on it keeps the local bounds (over the new and old states, or over the new one alone) and the mass M_L u exactly; with the antidiffusion
     d_ij (v_i - v_j) of the Jacobian-based scheme it gives the same bits as fct_limit.

usage: [ADRH_MODULE=adrh_kokkos] python upwind_test.py
"""
import importlib
import os
import sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
engine = importlib.import_module(os.environ.get('ADRH_MODULE', 'adrh_proto'))
from proteus import Quadrature
from stabilization_test import csr_pattern, scatter
from ad_solve_device import cube_tets

MATERIALS = {
    'van Genuchten': (0, 14.5, 2.68),                  # psk, alpha, n
    'Brooks-Corey-Mualem': (2, 1.0 / 0.072, 0.592),    # psk, 1/p_d, lambda
}


def setup(n_cells):
    X, conn = cube_tets(n_cells)
    n = X.shape[0]
    quad = Quadrature.SimplexGaussQuadrature(3, 4)
    qp = np.array([list(p)[:3] for p in quad.points]); qw = np.array(quad.weights)
    indptr, indices = csr_pattern(conn, n)
    return X, conn, n, qp, qw, indptr, indices, engine.transpose_map(indptr, indices)


def mats(psk, alpha, nvg):
    return (np.array([alpha]), np.array([nvg]), np.array([0.045]), np.array([0.385]), np.array([[5.0, 5.0, 2.0]]), np.array([0.0, 0.0, -1.0]), 1.0, 0.0)


def stiffness(setup_, m, u, mode, psk):
    X, conn, n, qp, qw, indptr, indices, tpos = setup_
    ue = u[conn].reshape(-1, 1, 4)
    _, J = engine.richards_vgm(3, "p1", *m[:4], m[4], m[5], m[6], m[7], np.zeros(conn.shape[0], dtype='i'), X[conn], ue, ue, qp, qw, 0.0, 0.0, psk, mode)
    return scatter(conn, n, np.zeros((conn.shape[0], 4)), J.reshape(-1, 4, 4), indptr, indices)[1]


def numpy_upwind(indptr, indices, S, phi, kr, dkr):
    n = len(phi); r = np.zeros(n); J = np.zeros(S.size)
    for i in range(n):
        for p in range(indptr[i], indptr[i + 1]):
            j = int(indices[p])
            if j == i: continue
            c = max(0.0, -S[p]); d = phi[j] - phi[i]
            if -S[p] * d <= 0.0: F = kr[i] * c * d; dFi = dkr[i] * c * d - kr[i] * c; dFj = kr[i] * c
            else: F = kr[j] * c * d; dFi = -kr[j] * c; dFj = dkr[j] * c * d + kr[j] * c
            r[i] -= F; J[p] = -dFj
            J[np.searchsorted(indices[indptr[i]:indptr[i + 1]], i) + indptr[i]] -= dFi
    return r, J


def test_operator(name, psk, alpha, nvg):
    st = setup(5)
    X, conn, n, qp, qw, indptr, indices, tpos = st
    m = mats(psk, alpha, nvg)
    rng = np.random.default_rng(5)
    u = -rng.uniform(0.02, 6.0, n)                         # heads from wet to dry
    S = stiffness(st, m, u, 1, psk)
    C = stiffness(st, m, u, 2, psk)
    Sm = sp.csr_matrix((S, indices, indptr), shape=(n, n))
    print("\n%s: %d nodes, %d edges" % (name, n, (indices.size - n) // 2))
    print("  stiffness S: symmetry %.1e, row sums %.1e (relative to max |S|); frozen-K matrix C: symmetry %.1e, row sums %.1e" % (
        abs(Sm - Sm.T).max() / abs(Sm).max(), np.abs(Sm.sum(axis=1)).max() / abs(Sm).max(),
        abs(sp.csr_matrix((C, indices, indptr), shape=(n, n)) - sp.csr_matrix((C, indices, indptr), shape=(n, n)).T).max() / abs(C).max(),
        np.abs(sp.csr_matrix((C, indices, indptr), shape=(n, n)).sum(axis=1)).max() / abs(C).max()))
    kr, dkr = engine.richards_kr_nodes(*m[:4], psk, np.zeros(n, dtype='i'), u)
    # derivative of kr against finite differences
    h = 1e-6
    krp, _ = engine.richards_kr_nodes(*m[:4], psk, np.zeros(n, dtype='i'), u + h)
    krm, _ = engine.richards_kr_nodes(*m[:4], psk, np.zeros(n, dtype='i'), u - h)
    print("  nodal kr: d kr/du against central finite differences %.1e (relative)" % (np.abs((krp - krm) / (2 * h) - dkr).max() / np.abs(dkr).max()))
    phi = u + X[:, 2]                                       # psi - g.x with g = (0, 0, -1)
    r, Jv = engine.upwind_flux(indptr, indices, S, phi, kr, dkr)
    rr, Jr = numpy_upwind(indptr, indices, S, phi, kr, dkr)
    print("  upwind_flux against the numpy edge loop: residual %.1e, Jacobian %.1e (relative max)" % (
        np.abs(r - rr).max() / np.abs(rr).max(), np.abs(Jv - Jr).max() / np.abs(Jr).max()))
    J = sp.csr_matrix((Jv, indices, indptr), shape=(n, n))
    print("  conservation: sum of the residual %.1e (of %.1e), largest column sum of the Jacobian %.1e (of %.1e)" % (
        abs(r.sum()), np.abs(r).max(), np.abs(J.sum(axis=0)).max(), abs(J).max()))

    def residual(v):
        kv, dv = engine.richards_kr_nodes(*m[:4], psk, np.zeros(n, dtype='i'), v)
        return engine.upwind_flux(indptr, indices, S, v + X[:, 2], kv, dv)[0]

    Jfd = np.zeros(n)
    cols = rng.choice(n, 8, replace=False)
    err = 0.0
    for j in cols:                                          # a few columns of the finite-difference Jacobian
        e = np.zeros(n); e[j] = 1e-4
        col = (residual(u + e) - residual(u - e)) / 2e-4
        err = max(err, np.abs(col - J[:, j].toarray().ravel()).max() / np.abs(col).max())
    print("  Jacobian against central finite differences of the residual (8 columns, h = 1e-4): %.1e (relative)" % err)
    A = engine.antidiffusive_flux(indptr, indices, C, S, phi, kr)
    Am = sp.csr_matrix((A, indices, indptr), shape=(n, n))
    print("  antidiffusive flux: antisymmetry %.1e (of %.1e)" % (abs(Am + Am.T).max(), abs(Am).max()))
    # FCT: the mass and the local bounds
    Mel = engine.consistent_mass(3, "p1", X[conn], qp, qw)[0]
    Mc = scatter(conn, n, np.zeros((conn.shape[0], 4)), Mel.reshape(-1, 4, 4), indptr, indices)[1]
    ML = np.asarray(sp.csr_matrix((Mc, indices, indptr), shape=(n, n)).sum(axis=1)).ravel()
    mold = rng.uniform(0.1, 0.4, n); mL = np.clip(mold + rng.uniform(-0.05, 0.05, n), 0.05, 0.43); dt = 1e-3
    out = engine.fct_limit_flux(indptr, indices, tpos, Mc, A, ML, mL, mold, dt)
    hi = np.array([max(max(mL[j], mold[j]) for j in indices[indptr[i]:indptr[i + 1]]) for i in range(n)])
    lo = np.array([min(min(mL[j], mold[j]) for j in indices[indptr[i]:indptr[i + 1]]) for i in range(n)])
    viol = max(0.0, (out - hi).max(), (lo - out).max())
    print("  FCT on the antidiffusive flux: local bounds violated by %.1e, mass M_L m changed by %.1e (relative), moved the low-order state by %.2e" % (
        viol, abs(ML @ (out - mL)) / (ML @ np.abs(mL)), np.abs(out - mL).max()))
    assert viol < 1e-13 and abs(ML @ (out - mL)) < 1e-13 * (ML @ np.abs(mL))
    # the bounds over the low-order state alone (proteus's FCTStep): tighter, so the correction is smaller
    outb = engine.fct_limit_flux(indptr, indices, tpos, Mc, A, ML, mL, mold, dt, False)
    hi2 = np.array([max(mL[j] for j in indices[indptr[i]:indptr[i + 1]]) for i in range(n)])
    lo2 = np.array([min(mL[j] for j in indices[indptr[i]:indptr[i + 1]]) for i in range(n)])
    print("  with the bounds over the low-order state alone: violated by %.1e, mass changed by %.1e, moved the state by %.2e" % (
        max(0.0, (outb - hi2).max(), (lo2 - outb).max()), abs(ML @ (outb - mL)) / (ML @ np.abs(mL)), np.abs(outb - mL).max()))
    assert max(0.0, (outb - hi2).max(), (lo2 - outb).max()) < 1e-13 and abs(ML @ (outb - mL)) < 1e-13 * (ML @ np.abs(mL))
    # the same limiter with the Jacobian-based antidiffusion d_ij (v_i - v_j) is fct_limit
    d = engine.dissipation_values(indptr, indices, tpos, Jv)
    v = rng.uniform(0, 1, n)
    Ad = np.zeros(S.size)
    for i in range(n):
        for p in range(indptr[i], indptr[i + 1]):
            Ad[p] = d[p] * (v[i] - v[indices[p]])
    o1 = engine.fct_limit_flux(indptr, indices, tpos, Mc, Ad, ML, mL, mold, dt)
    o2 = engine.fct_limit(indptr, indices, tpos, Mc, d, ML, v, mL, mold, dt)
    print("  fct_limit_flux with d_ij (v_i - v_j) against fct_limit: %.1e (max abs)" % np.abs(o1 - o2).max())
    assert np.abs(o1 - o2).max() < 1e-13
    assert np.abs(r - rr).max() < 1e-12 * np.abs(rr).max() and np.abs(Jv - Jr).max() < 1e-12 * np.abs(Jr).max()
    assert err < 1e-4 and abs(Am + Am.T).max() < 1e-15 * abs(Am).max()
    try:
        host = importlib.import_module('adrh_proto')
        if engine.__name__ != 'adrh_proto':
            r2, J2 = host.upwind_flux(indptr, indices, S, phi, kr, dkr)
            o3 = host.fct_limit_flux(indptr, indices, tpos, Mc, A, ML, mL, mold, dt)
            print("  %s against the host module: residual and Jacobian %s, FCT %s" % (
                engine.__name__, "bitwise equal" if np.array_equal(r, r2) and np.array_equal(Jv, J2) else "differ by %.1e" % max(np.abs(r - r2).max(), np.abs(Jv - J2).max()),
                "bitwise equal" if np.array_equal(out, o3) else "differs by %.1e" % np.abs(out - o3).max()))
    except ImportError:
        pass


if __name__ == '__main__':
    for name, (psk, alpha, nvg) in MATERIALS.items():
        test_operator(name, psk, alpha, nvg)
