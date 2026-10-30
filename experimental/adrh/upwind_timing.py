"""Cost of the K-upwinded passes (lowOrder.h, richards.h) on a 3D tetrahedral Richards problem, device-resident (best of N), against the colored assembly of
the advection-diffusion element kernel on the same mesh as a yardstick (the Richards kernel costs about three times that per element, README.md).

The mesh is the unit cube with nn^3 nodes and six tetrahedra per cell; the state is a hydrostatic column with a wet layer on top, so K_r varies by orders of
magnitude. Needs the Kokkos module (upwind_passes_timed). usage: [KOKKOS_NUM_THREADS=n] python upwind_timing.py nn [nn ...]
"""
import sys

import numpy as np

import adrh_kokkos as K
import richards_nd as nd
from ad_solve_device import cube_tets, B, KAPPA, OMEGA, OMEGA0

for nn in [int(a) for a in sys.argv[1:]]:
    X, conn = cube_tets(nn)
    conn = np.ascontiguousarray(conn.reshape(-1, 4))
    ks = 7.967
    model = nd.KUpwindRichards(X, conn, np.zeros(conn.shape[0], dtype='i'), alpha=[3.35], n_vg=[2.0], theta_r=[0.102], theta_s=[0.368], ks=[[ks, ks, ks]],
                               g=[0.0, 0.0, -1.0], dirichlet=([0], [-10.0]))
    z = X[:, 2]
    uold = np.where(z > 0.9, -0.75, -10.0 + 0.0 * z); uL = np.where(z > 0.85, -0.75, -10.0 + 0.5 * z)
    kr, dkr = model.kr(uL)
    phi = model.total_head(uL)
    C = model.stiffness(uL, 2).data.copy()
    mL, _ = model.m(uL); mo, _ = model.m(uold)
    pmax, pmin = K.head_extrema(model.indptr, model.indices, uL, uold)
    mmax, _ = model.m(pmax); mmin, _ = model.m(pmin)
    qp, qw = K.quad_tet4()
    asm = K.AdSineAssembler3D(X, np.ascontiguousarray(conn, dtype=np.int32), qp, qw)
    U = np.where(((X < 1e-12) | (X > 1 - 1e-12)).any(axis=1), np.sin(X @ OMEGA + OMEGA0), 0.0)
    t_asm = asm.assemble(B, KAPPA, OMEGA, OMEGA0, U, 5)["t_assemble"]
    t = K.upwind_passes_timed(model.indptr, model.indices, model.tpos, model.S, C, model.Mc.data.copy(), model.ML, phi, kr, dkr, mL, mo, uL, uold, mmax, mmin,
                              model.alpha, model.n_vg, model.theta_r, model.thetaSR, 0, model.nodeMat, 0.01, 5)
    info = K.kokkos_info()
    print("nn=%d  %s x%d  rows %d  nnz %d  |  colored AD assembly (yardstick) %.4f s" % (nn, info["space"], info["concurrency"], model.nn, model.indices.size, t_asm))
    for name in ("kr_nodes", "upwind_flux", "antidiffusive_flux", "fct_moisture_bounds", "head_extrema", "fct_given_bounds"):
        print("    %-20s %.5f s  (%.3f of the yardstick)" % (name, t[name], t[name] / t_asm))
    print("    head-bounded FCT step = antidiffusive flux + head extrema + limiter = %.5f s; upload %.4f s, download %.4f s" %
          (t["antidiffusive_flux"] + t["head_extrema"] + t["fct_given_bounds"], t["t_h2d"], t["t_d2h"]))
    asm.close()
