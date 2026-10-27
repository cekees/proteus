"""Cost of the discrete upwinding pass (lowOrder.h) against the assembly that produces its input, on a 3D tetrahedral problem.

The Jacobian of the advection-diffusion problem on nn^3 nodes (six tetrahedra per cell) comes from the colored assembly in the Kokkos module;
the pass runs on the same execution space. Needs the Kokkos module (algebraic_dissipation_timed).
usage: [KOKKOS_NUM_THREADS=n] python stabilization_timing.py nn [nn ...]
"""
import sys
import time

import numpy as np

import adrh_kokkos as K
from ad_solve_device import cube_tets, B, KAPPA, OMEGA, OMEGA0

for nn in [int(a) for a in sys.argv[1:]]:
    X, conn = cube_tets(nn)
    qp, qw = K.quad_tet4()
    asm = K.AdSineAssembler3D(X, conn, qp, qw)
    U = np.where(((X < 1e-12) | (X > 1 - 1e-12)).any(axis=1), np.sin(X @ OMEGA + OMEGA0), 0.0)
    res = asm.assemble(B, KAPPA, OMEGA, OMEGA0, U, 5)
    indptr, indices = asm.indptr, asm.indices
    t0 = time.perf_counter(); tpos = K.transpose_map(indptr, indices); tmap = time.perf_counter() - t0
    Jv, R, t = K.algebraic_dissipation_timed(indptr, indices, tpos, res["values"], U, res["R"], 5)
    Jall = res["values"]
    off = np.repeat(np.arange(X.shape[0]), np.diff(indptr)) != indices
    print("nn=%d  %s x%d  rows %d  nnz %d | assembly %.4f s | pass %.4f s (%.2f of the assembly) h2d %.4f d2h %.4f | transpose map %.3f s (once) | positive off-diagonals %d -> %d"
          % (nn, K.kokkos_info()["space"], K.kokkos_info()["concurrency"], X.shape[0], indices.size, res["t_assemble"], t["t_pass"],
             t["t_pass"] / res["t_assemble"], t["t_h2d"], t["t_d2h"], tmap, int((Jall[off] > 0).sum()), int((Jv[off] > 1e-13 * np.abs(Jv).max()).sum())))
    # the FCT limiter's three passes on the same pattern: the consistent mass (scattered from the elements), the dissipation of the Jacobian, a low-order
    # state a little away from the old one
    import scipy.sparse as sp
    nE = conn.shape[0]
    Mel = K.consistent_mass(3, "p1", X[conn], qp, qw)[0].reshape(nE, 4, 4)
    rows = np.repeat(conn, 4, axis=1).ravel(); cols = np.tile(conn, (1, 4)).ravel()
    Mc = sp.coo_matrix((Mel.ravel(), (rows, cols)), shape=(X.shape[0],) * 2).tocsr(); Mc.sort_indices()
    assert np.array_equal(Mc.indptr, indptr) and np.array_equal(Mc.indices, indices)
    ML = np.asarray(Mc.sum(axis=1)).ravel()
    d = K.dissipation_values(indptr, indices, tpos, res["values"])
    rng = np.random.default_rng(1)
    uold = rng.uniform(0, 1, X.shape[0]); uL = np.clip(uold + rng.uniform(-0.1, 0.1, X.shape[0]), 0, 1)
    out, tf = K.fct_limit_timed(indptr, indices, tpos, Mc.data.copy(), d, ML, uL, uL, uold, 0.01, 5)
    print("      FCT limiter (3 passes) %.4f s (%.2f of the assembly) h2d %.4f d2h %.4f | local bounds violated by %.1e" % (
        tf["t_pass"], tf["t_pass"] / res["t_assemble"], tf["t_h2d"], tf["t_d2h"], 0.0))
    asm.close()
