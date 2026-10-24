"""Advection-diffusion regression: ADRH engine vs the Proteus model (ad_sine_p/n.py) vs the exact solution.

Steady div(b u - kappa grad u) + d(x) = 0 on the unit square or cube, exact solution sin(omega.x + omega0) (the
verified AnalyticalSolutions.LinearADR_Sine with c = 0), Galerkin with P1 (AD_SPACE=p1, the default) or P2 (p2) on
regular triangles / tetrahedra, or Q1 (q1) on regular quadrilaterals / hexahedra, in AD_ND=2 or 3 (the default)
dimensions. The engine builds the element and exterior-face residual and Jacobian; the driver assembles, imposes any
strong Dirichlet rows and solves with scipy (regression_common.py has the shared parts). Checks:

  1. engine and Proteus solve the same discrete problem: nodal solutions agree to solver round-off.
     AD_BC picks the boundary treatment on both sides (ad_sine_p.py has the Proteus side):
       strong (default)  Dirichlet data everywhere, imposed strongly by the driver (Proteus: numericalFluxType=None)
       mixed             the face x = 1 is a flux face: prescribed diffusive flux, advection out by the interior value
                         (Proteus: StrongDirichletFactory({0: 'outFlow'}); AD_MIXED=1 means this)
       inflow            as mixed, and the inflow face x = 0 has the advective flux (b . n) u and the diffusive flux
                         prescribed (the engine's Flux kind with hasAdv)
       robin             as mixed, but the face x = 1 carries alpha u + beta q.n = g (the engine's Flux kind with an
                         affine diffusive flux, against Proteus's RobinFactory)
       iipg, nipg, sipg  Dirichlet data everywhere, imposed weakly: upwinded advective flux against f(g), diffusive flux
                         with penalty C/h (2, 10, 100) and the NIPG/SIPG adjoint term (the engine's WeakDirichlet kind
                         against Proteus's Advection_DiagonalUpwind_Diffusion_*_exterior);
  2. both converge to the exact solution at the space's rates (P1, Q1: L2 ~ h^2; P2: L2 ~ h^3);
  3. the exact solution fed through the engine's residual is small and shrinks with h (consistency).

usage: [AD_ND=2|3] [AD_SPACE=p1|p2|q1] [AD_BC=mode] python ad_regression.py [nn ...]        default 5 9 17
"""
import sys

import numpy as np

from regression_common import (ND, SPACE, BC, nD, qp, qw, qpf, qwf, WEAK_PARAMS, ROBIN_ALPHA, ROBIN_BETA,
                               adrh_proto, run_proteus, boundary_faces, dirichlet_dofs, assemble,
                               solve_with_dirichlet, l2_error)

b = np.array([1.0, 0.5, 0.25])[:ND]; kappa = 0.2; omega = np.array([3.0, 2.0, 4.0])[:ND]; omega0 = 0.3
# the verified exact solution (LinearADR_Sine with c = 0), on points with any leading shape and 2 or 3 coordinates
exact = lambda X: np.sin(X[..., :ND] @ omega + omega0)


def engine_solution(nodes, conn):
    """nodes: the coordinates of every global dof; conn: the element-to-dof map (nE, nD)."""
    nE, nN = len(conn), len(nodes)
    xn = np.ascontiguousarray(nodes[conn])                                      # (nE, nD, 3): Proteus keeps 3 coordinates
    R, J = adrh_proto.advection_diffusion_sine(ND, SPACE, b, kappa, omega, omega0, xn, np.zeros((nE, 1, nD)), qp, qw)
    K, rhs = assemble(conn, R, J, nN)
    faces = boundary_faces(nodes, conn)
    if faces is not None:
        fe, fk, kind = faces
        face_conn = conn[fe]
        pc, pp, sig = WEAK_PARAMS
        Rf, Jf = adrh_proto.advection_diffusion_sine_faces(ND, SPACE, b, kappa, omega, omega0, fk, kind,
                                                           np.ascontiguousarray(nodes[face_conn]), np.zeros((len(fe), 1, nD)),
                                                           qpf, qwf, pc, pp, sig, ROBIN_ALPHA, ROBIN_BETA)
        Kf, rhs_f = assemble(face_conn, Rf, Jf, nN)
        K = K + Kf; rhs = rhs + rhs_f
    u = solve_with_dirichlet(K, rhs, dirichlet_dofs(nodes), exact(nodes))
    return u, xn, R, J


sizes = [int(a) for a in sys.argv[1:]] or [5, 9, 17]
prev = None
print("AD_ND=%d AD_SPACE=%s AD_BC=%s" % (ND, SPACE, BC))
print("%4s %9s %12s %12s %14s %14s %9s %9s %14s" % ("nn", "elements", "|eng-prot|", "eng-vs-ex L2", "prot-vs-ex L2", "eng max err", "L2 rate", "max rate", "res(exact) inf"))
for nn in sizes:
    up, nodes, conn = run_proteus('ad_sine_p', 'ad_sine_n', 'adsine', nn)   # nodes = every dof's coordinates
    ue, xn, R, J = engine_solution(nodes, conn)
    l2e, l2p = l2_error(ue, conn, xn, exact), l2_error(up, conn, xn, exact)
    mxe = np.abs(ue - exact(nodes)).max()
    # the exact solution through the engine: element residual of u_exact at the interior dofs (no boundary terms)
    uex = exact(nodes)[conn][:, None, :]
    Rx = adrh_proto.advection_diffusion_sine(ND, SPACE, b, kappa, omega, omega0, xn, np.ascontiguousarray(uex), qp, qw)[0]
    X = nodes[:, :ND]
    interior = (X.min(axis=1) > 1e-12) & (X.max(axis=1) < 1 - 1e-12)
    resid = np.abs(np.bincount(conn.ravel(), weights=Rx[:, 0, :].ravel(), minlength=len(nodes))[interior]).max()
    h = 1.0 / (nn - 1)
    rate = ("%9.2f %9.2f" % (np.log(prev[0] / l2e) / np.log(prev[1] / h), np.log(prev[2] / mxe) / np.log(prev[1] / h))) if prev else "%9s %9s" % ("-", "-")
    print("%4d %9d %12.2e %12.3e %14.3e %14.3e %s %14.3e" % (nn, len(conn), np.abs(ue - up).max(), l2e, l2p, mxe, rate, resid))
    prev = (l2e, h, mxe)
