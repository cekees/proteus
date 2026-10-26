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
    asm.close()
