"""The algebraic stabilization pass (lowOrder.h): lumped mass and discrete upwinding of the assembled spatial operator.

  1. the pass against a scipy reference, on the Jacobian of an advection-dominated 3D tetrahedral problem: the result has no
     positive off-diagonal, keeps the row sums, and leaves a constant vector's residual unchanged;
  2. a sharp advected front, u_t + b u_x - kappa u_xx = 0, step initial data, cell Peclet number 25, backward Euler on a 1D
     mesh: consistent-mass Galerkin overshoots and undershoots, the stabilized scheme keeps every value inside the data's bounds
     (the discrete maximum principle), conserves mass, and is compared with the exact (erfc) solution.

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
            if variant == 'lumped+upwind':
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
    if variant != 'galerkin':
        pass
    return x, u, wnode


def test_front():
    print("\nsharp front, cell Peclet number 25, backward Euler, CFL 0.2, t = 0.3")
    print("  %-28s %10s %10s %14s %12s" % ("scheme", "min u", "max u", "L1 error", "mass"))
    for variant in ('galerkin', 'lumped', 'lumped+upwind'):
        x, u, wnode = front(variant)
        ex = exact(x, 0.3, 1.0, 2e-4, 0.2)
        h = x[1] - x[0]
        mass = np.trapezoid(u, x)
        print("  %-28s %10.2e %10.6f %14.4e %12.6f" % ({'galerkin': 'Galerkin, consistent mass', 'lumped': 'lumped mass only',
                                                         'lumped+upwind': 'lumped mass + discrete upwinding'}[variant],
                                                         u.min(), u.max(), h * np.abs(u - ex).sum(), mass))
        if variant == 'lumped+upwind':
            assert u.min() > -1e-12 and u.max() < 1.0 + 1e-12, "the discrete maximum principle failed"
            print("  the stabilized scheme stays inside [0, 1]: largest excursion %.1e" % max(-u.min(), u.max() - 1.0, 0.0))


if __name__ == '__main__':
    test_pass()
    test_front()
