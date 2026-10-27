"""The algebraic stabilization pass (lowOrder.h): lumped mass and discrete upwinding of the assembled spatial operator.

  1. the pass against a scipy reference, on the Jacobian of an advection-dominated 3D tetrahedral problem: the result has no
     positive off-diagonal, keeps the row sums, and leaves a constant vector's residual unchanged;
  2. a sharp advected front, u_t + b u_x - kappa u_xx = 0, step initial data, cell Peclet number 25, backward Euler on a 1D
     mesh: consistent-mass Galerkin overshoots and undershoots, the stabilized scheme keeps every value inside the data's bounds
     (the discrete maximum principle), conserves mass, and is compared with the exact (erfc) solution.

  3. the FCT limiter (lowOrder.h) on top: on random bounded data it keeps every node inside the local extrema of the two states and
     the mass M_L u exactly, matches an independent numpy implementation of Zalesak's limiter, and (on the host and Kokkos modules) gives
     the same bits; on the sharp front it removes most of the smearing of the low-order scheme without bringing back the overshoot.

usage: [ADRH_MODULE=adrh_kokkos] python stabilization_test.py
"""
import importlib
import os
import sys
from math import erfc, sqrt

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
engine = importlib.import_module(os.environ.get('ADRH_MODULE', 'adrh_proto'))
from proteus import Quadrature


def csr_pattern(conn, n):
    """Sorted CSR pattern of a nodal mesh and the position of every element block entry in it."""
    rows = np.repeat(conn, conn.shape[1], axis=1).ravel(); cols = np.tile(conn, (1, conn.shape[1])).ravel()
    A = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr(); A.sort_indices()
    return A.indptr.astype('i'), A.indices.astype('i')


def scatter(conn, n, R, J, indptr, indices):
    """Element residuals and Jacobians (nE, nD), (nE, nD, nD) into a vector and CSR values on the pattern."""
    r = np.zeros(n); np.add.at(r, conn.ravel(), R.reshape(conn.shape).ravel())
    rows = np.repeat(conn, conn.shape[1], axis=1).ravel(); cols = np.tile(conn, (1, conn.shape[1])).ravel()
    A = sp.coo_matrix((J.reshape(conn.shape[0], conn.shape[1], conn.shape[1]).ravel(), (rows, cols)), shape=(n, n)).tocsr(); A.sort_indices()
    assert np.array_equal(A.indptr, indptr) and np.array_equal(A.indices, indices)
    return r, A.data.copy()


def test_pass():
    sys.path.insert(0, HERE)
    from ad_solve_device import cube_tets
    X, conn = cube_tets(9)
    n = X.shape[0]
    quad = Quadrature.SimplexGaussQuadrature(3, 4); qp = np.array([list(p)[:3] for p in quad.points]); qw = np.array(quad.weights)
    rng = np.random.default_rng(3)
    u = rng.uniform(0, 1, n)
    b = np.array([40.0, 20.0, 10.0])                      # advection dominated: positive off-diagonals everywhere
    R, J = engine.transient_adr(3, "p1", b, 0.01, 0.0, X[conn], u[conn].reshape(-1, 1, 4), u[conn].reshape(-1, 1, 4), qp, qw, 0.0, 0.0)
    indptr, indices = csr_pattern(conn, n)
    r, Jv = scatter(conn, n, R, J, indptr, indices)
    tpos = engine.transpose_map(indptr, indices)
    assert np.array_equal(indices[tpos], np.repeat(np.arange(n), np.diff(indptr))), "transpose map"
    Jv2, r2 = engine.algebraic_dissipation(indptr, indices, tpos, Jv, u, r)
    # the reference, in scipy
    A = sp.csr_matrix((Jv, indices, indptr), shape=(n, n))
    off = A - sp.diags(A.diagonal())
    D = off.maximum(off.T).maximum(0.0)                   # d_ij = max(0, J_ij, J_ji), i != j
    Aref = A - D + sp.diags(np.asarray(D.sum(axis=1)).ravel())
    rref = r + (sp.diags(np.asarray(D.sum(axis=1)).ravel()) - D) @ u
    Anew = sp.csr_matrix((Jv2, indices, indptr), shape=(n, n))
    offnew = Anew - sp.diags(Anew.diagonal())
    print("pass vs scipy reference: Jacobian %.1e, residual %.1e (relative max)" % (
        abs(Anew - Aref).max() / abs(Aref).max(), np.abs(r2 - rref).max() / np.abs(rref).max()))
    print("  positive off-diagonals before: %d, after: %d (largest %.1e); row sums changed by %.1e; constant vector's residual change %.1e" % (
        (off.data > 0).sum(), (offnew.data > 1e-14 * abs(Anew).max()).sum(), offnew.max(),
        np.abs(np.asarray(Anew.sum(axis=1)).ravel() - np.asarray(A.sum(axis=1)).ravel()).max(),
        np.abs(engine.algebraic_dissipation(indptr, indices, tpos, Jv, np.ones(n), np.zeros(n))[1]).max()))
    assert (offnew.data > 1e-12 * abs(Anew).max()).sum() == 0


def exact(x, t, b, kappa, x0):
    return np.array([0.5 * erfc((xi - x0 - b * t) / (2.0 * sqrt(kappa * t))) for xi in x])


def front(variant, nn=101, b=1.0, kappa=2e-4, dt=0.002, T=0.3, x0=0.2):
    x = np.linspace(0.0, 1.0, nn); n = nn
    conn = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    xn = np.zeros((n - 1, 2, 3)); xn[:, :, 0] = x[conn]
    quad = Quadrature.SimplexGaussQuadrature(1, 3); qp = np.array([list(p)[:1] for p in quad.points]); qw = np.array(quad.weights)
    indptr, indices = csr_pattern(conn, n); tpos = engine.transpose_map(indptr, indices)
    bv = np.array([b])
    u = np.where(x < x0, 1.0, 0.0); u[0] = 1.0
    lo, hi = u.min(), u.max(); steps = int(round(T / dt)); mass0 = None
    wnode = np.zeros(n)
    for step in range(steps):
        uold = u.copy()
        ue = u[conn].reshape(-1, 1, 2); uo = uold[conn].reshape(-1, 1, 2)
        if variant == 'galerkin':                          # consistent mass, as the engine's element kernel has it
            R, J = engine.transient_adr(1, "p1", bv, kappa, 0.0, xn, ue, uo, qp, qw, 0.0, dt)
            r, Jv = scatter(conn, n, R, J, indptr, indices)
        else:
            R, J = engine.transient_adr(1, "p1", bv, kappa, 0.0, xn, ue, uo, qp, qw, 0.0, 0.0)       # the spatial operator alone
            r, Jv = scatter(conn, n, R, J, indptr, indices)
            if variant in ('lumped+upwind', 'fct'):
                Jv, r = engine.algebraic_dissipation(indptr, indices, tpos, Jv, u, r)
            w, m, dm, mold = engine.transient_adr_lumped(1, "p1", bv, kappa, 0.0, xn, ue, uo, qp, qw, 0.0, dt)
            rm = np.zeros(n); np.add.at(rm, conn.ravel(), (w * (m - mold) / dt).ravel()); r = r + rm
            dg = np.zeros(n); np.add.at(dg, conn.ravel(), (w * dm / dt).ravel())
            Jv = Jv.copy(); diag = indptr[:-1] + np.array([np.searchsorted(indices[indptr[i]:indptr[i + 1]], i) for i in range(n)])
            Jv[diag] += dg
            wnode = np.zeros(n); np.add.at(wnode, conn.ravel(), w.ravel())
        A = sp.csr_matrix((Jv, indices, indptr), shape=(n, n)).tolil()
        r = r.copy()
        for i, g in ((0, 1.0), (n - 1, 0.0)):                # strong rows: inflow value, and the far end held at 0
            r[i] = u[i] - g; A.rows[i] = [i]; A.data[i] = [1.0]
        u = u + spl.spsolve(A.tocsc(), -r)
        if variant == 'fct':                                 # the limiter: antidiffusive fluxes at the low-order solution
            ue2 = u[conn].reshape(-1, 1, 2)
            R2, J2 = engine.transient_adr(1, "p1", bv, kappa, 0.0, xn, ue2, ue2, qp, qw, 0.0, 0.0)
            _, Jv2 = scatter(conn, n, R2, J2, indptr, indices)
            d = engine.dissipation_values(indptr, indices, tpos, Jv2)
            Mel = engine.consistent_mass(1, "p1", xn, qp, qw)[0]
            _, Mc = scatter(conn, n, np.zeros((conn.shape[0], 2)), Mel.reshape(-1, 2, 2), indptr, indices)
            u = engine.fct_limit(indptr, indices, tpos, Mc, d, wnode, u, u, uold, dt)
            u[0], u[-1] = 1.0, 0.0                           # the Dirichlet values stay what they are
    if variant != 'galerkin':
        pass
    return x, u, wnode


def width(x, u, lo=0.1, hi=0.9):
    """Distance between where the profile falls through hi and through lo (the front's width), interpolated."""
    def cross(level):
        for i in range(len(x) - 1):
            if u[i] >= level > u[i + 1]:
                return x[i] + (u[i] - level) * (x[i + 1] - x[i]) / (u[i] - u[i + 1])
        return np.nan
    return cross(lo) - cross(hi)


def test_front():
    print("\nsharp front, cell Peclet number 25, backward Euler, CFL 0.2, t = 0.3")
    print("  %-34s %10s %10s %12s %12s %10s" % ("scheme", "min u", "max u", "L1 error", "width 10-90%", "mass"))
    out = {}
    for variant in ('galerkin', 'lumped', 'lumped+upwind', 'fct'):
        x, u, wnode = front(variant)
        ex = exact(x, 0.3, 1.0, 2e-4, 0.2)
        h = x[1] - x[0]
        out[variant] = u
        print("  %-34s %10.2e %10.6f %12.4e %12.4f %10.6f" % (
            {'galerkin': 'Galerkin, consistent mass', 'lumped': 'lumped mass only', 'lumped+upwind': 'lumped mass + discrete upwinding',
             'fct': 'FCT (low-order + Zalesak limiter)'}[variant], u.min(), u.max(), h * np.abs(u - ex).sum(), width(x, u), np.trapezoid(u, x)))
    print("  exact front width %.4f" % width(x, ex))
    for variant in ('lumped+upwind', 'fct'):
        u = out[variant]
        assert u.min() > -1e-12 and u.max() < 1.0 + 1e-12, variant + ": the bounds failed"
    print("  low-order and FCT both stay inside [0, 1]: largest excursions %.1e and %.1e" % (
        max(-out['lumped+upwind'].min(), out['lumped+upwind'].max() - 1.0, 0.0), max(-out['fct'].min(), out['fct'].max() - 1.0, 0.0)))


def zalesak_reference(indptr, indices, Mc, d, ML, uL, uold, dt):
    """The limiter written the plain way, loops over nodes and their neighbours: the oracle for lowOrder.h."""
    n = len(uL); delta = uL - uold; out = uL.copy()
    pos = {(i, int(indices[p])): p for i in range(n) for p in range(indptr[i], indptr[i + 1])}
    G = {}
    for (i, j), p in pos.items():
        if i != j:
            f = Mc[p] * (delta[j] - delta[i]) / dt + d[p] * (uL[j] - uL[i])     # computed once per pair, from the lower-numbered node
            if i < j:
                G[(i, j)] = -f; G[(j, i)] = f
    Rp = np.ones(n); Rm = np.ones(n)
    for i in range(n):
        nb = [int(indices[p]) for p in range(indptr[i], indptr[i + 1])]
        gi = [G[(i, j)] for j in nb if j != i]
        Pp = sum(g for g in gi if g > 0); Pm = sum(g for g in gi if g < 0)
        umax = max(max(uL[j], uold[j]) for j in nb); umin = min(min(uL[j], uold[j]) for j in nb)
        Qp = ML[i] / dt * (umax - uL[i]); Qm = ML[i] / dt * (umin - uL[i])
        Rp[i] = min(1.0, Qp / Pp) if Pp > 0 else 1.0
        Rm[i] = min(1.0, Qm / Pm) if Pm < 0 else 1.0
    for i in range(n):
        acc = 0.0
        for j in [int(indices[p]) for p in range(indptr[i], indptr[i + 1])]:
            if j == i: continue
            g = G[(i, j)]
            acc += (min(Rp[i], Rm[j]) if g > 0 else min(Rm[i], Rp[j]) if g < 0 else 0.0) * g
        out[i] = uL[i] + dt / ML[i] * acc
    return out


def test_fct_properties():
    from ad_solve_device import cube_tets
    X, conn = cube_tets(7)
    n = X.shape[0]
    quad = Quadrature.SimplexGaussQuadrature(3, 4); qp = np.array([list(p)[:3] for p in quad.points]); qw = np.array(quad.weights)
    rng = np.random.default_rng(7)
    indptr, indices = csr_pattern(conn, n); tpos = engine.transpose_map(indptr, indices)
    Mel = engine.consistent_mass(3, "p1", X[conn], qp, qw)[0]
    _, Mc = scatter(conn, n, np.zeros((conn.shape[0], 4)), Mel.reshape(-1, 4, 4), indptr, indices)
    ML = np.asarray(sp.csr_matrix((Mc, indices, indptr), shape=(n, n)).sum(axis=1)).ravel()
    u0 = rng.uniform(0, 1, n)
    R, J = engine.transient_adr(3, "p1", np.array([30.0, 15.0, 8.0]), 0.01, 0.0, X[conn], u0[conn].reshape(-1, 1, 4), u0[conn].reshape(-1, 1, 4), qp, qw, 0.0, 0.0)
    _, Jv = scatter(conn, n, R, J, indptr, indices)
    d = engine.dissipation_values(indptr, indices, tpos, Jv)
    uold = rng.uniform(0, 1, n); uL = np.clip(uold + rng.uniform(-0.2, 0.2, n), 0, 1); dt = 0.05
    out = engine.fct_limit(indptr, indices, tpos, Mc, d, ML, uL, uL, uold, dt)
    ref = zalesak_reference(indptr, indices, Mc, d, ML, uL, uold, dt)
    # local bounds, from the two states over each node and its neighbours
    hi = np.array([max(max(uL[j], uold[j]) for j in indices[indptr[i]:indptr[i + 1]]) for i in range(n)])
    lo = np.array([min(min(uL[j], uold[j]) for j in indices[indptr[i]:indptr[i + 1]]) for i in range(n)])
    print("\nFCT on random bounded data, %d nodes, %d edges" % (n, (indices.size - n) // 2))
    print("  against the numpy implementation of Zalesak's limiter: %.1e (max abs)" % np.abs(out - ref).max())
    print("  local bounds: largest violation %.1e; mass M_L u changed by %.1e (relative to M_L |uL|)" % (
        max(0.0, (out - hi).max(), (lo - out).max()), abs(ML @ (out - uL)) / (ML @ np.abs(uL))))
    print("  the limiter moved the low-order solution by %.2e (max), the unlimited step would move it by %.2e" % (
        np.abs(out - uL).max(), np.abs(zalesak_unlimited(indptr, indices, Mc, d, ML, uL, uold, dt) - uL).max()))
    assert np.abs(out - ref).max() < 1e-13 and max(0.0, (out - hi).max(), (lo - out).max()) < 1e-13
    assert abs(ML @ (out - uL)) < 1e-14 * (ML @ np.abs(uL))
    try:
        host = importlib.import_module('adrh_proto')
        if engine.__name__ != 'adrh_proto':
            o2 = host.fct_limit(indptr, indices, tpos, Mc, d, ML, uL, uL, uold, dt)
            print("  %s against the host module: %s (max abs %.1e)" % (engine.__name__, "bitwise equal" if np.array_equal(o2, out) else "differs", np.abs(o2 - out).max()))
    except ImportError:
        pass


def zalesak_unlimited(indptr, indices, Mc, d, ML, uL, uold, dt):
    n = len(uL); delta = uL - uold; out = uL.copy()
    for i in range(n):
        acc = 0.0
        for p in range(indptr[i], indptr[i + 1]):
            j = int(indices[p])
            if j == i: continue
            acc += -(Mc[p] * (delta[j] - delta[i]) / dt + d[p] * (uL[j] - uL[i]))
        out[i] = uL[i] + dt / ML[i] * acc
    return out


if __name__ == '__main__':
    test_pass()
    test_front()
    test_fct_properties()
