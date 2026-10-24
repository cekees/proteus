"""Advection-diffusion regression: ADRH engine vs the Proteus model (ad_sine_p/n.py) vs the exact solution.

Steady div(b u - kappa grad u) + d(x) = 0 on the unit square or cube, exact solution sin(omega.x + omega0) (the
verified AnalyticalSolutions.LinearADR_Sine with c = 0), Galerkin with P1 (AD_SPACE=p1, the default) or P2 (p2) on
regular triangles / tetrahedra, or Q1 (q1) on regular quadrilaterals / hexahedra, in AD_ND=2 or 3 (the default)
dimensions. The engine builds the element and exterior-face residual and Jacobian; the driver
assembles, imposes any strong Dirichlet rows and solves with scipy. Checks:

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
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adrh_proto
from proteus import Comm
Comm.init()
from proteus import default_s, default_so, NumericalSolution, Quadrature
from proteus.defaults import load_numerics, load_physics
from proteus.iproteus import opts as runopts

ND = int(os.environ.get('AD_ND', '3'))
os.environ['AD_ND'] = str(ND)
SPACE = os.environ.get('AD_SPACE', 'p1')
os.environ['AD_SPACE'] = SPACE
nD, nV, FACE_VERTICES = adrh_proto.space_info(ND, SPACE)          # dofs and vertices per element; each face's vertices
b = np.array([1.0, 0.5, 0.25])[:ND]; kappa = 0.2; omega = np.array([3.0, 2.0, 4.0])[:ND]; omega0 = 0.3
order = 4
os.environ.setdefault('AD_QORDER', str(order))
Gauss = Quadrature.CubeGaussQuadrature if SPACE == 'q1' else Quadrature.SimplexGaussQuadrature   # as ad_sine_n.py
quad = Gauss(ND, order)
qp = np.array([list(p)[:ND] for p in quad.points]); qw = np.array(quad.weights)
quadf = Gauss(ND - 1, order)
qpf = np.array([list(p)[:ND - 1] for p in quadf.points]); qwf = np.array(quadf.weights)
# the verified exact solution (LinearADR_Sine with c = 0), on points with any leading shape and 2 or 3 coordinates
exact = lambda X: np.sin(X[..., :ND] @ omega + omega0)

BC = os.environ.get('AD_BC', 'mixed' if os.environ.get('AD_MIXED', '0') == '1' else 'strong')
WEAK = {'iipg': (2.0, 1.0, 0.0), 'nipg': (10.0, 1.0, -1.0), 'sipg': (100.0, 1.0, 1.0)}   # C, p, sigma as in Proteus
assert BC in ('strong', 'mixed', 'inflow', 'robin') or BC in WEAK, BC
os.environ['AD_BC'] = BC
ROBIN_ALPHA, ROBIN_BETA = 1.0, 2.0                                 # as ad_sine_p.py
# the faces the engine's boundary kernel sees, by plane: (kind code of SineBC, axis, value); the rest is strong Dirichlet
FACES = {'strong': [], 'mixed': [(1, 0, 1.0)], 'inflow': [(2, 0, 0.0), (1, 0, 1.0)], 'robin': [(4, 0, 1.0)]}
for mode in WEAK:
    FACES[mode] = [(3, ax, v) for ax in range(ND) for v in (0.0, 1.0)]


def proteus_solution(nn):
    """Proteus's solution: the dof vector, the coordinates of every global dof (vertices, then P2 edge midpoints),
    the element-to-global dof map (nE, nD) whose first nV columns are the element's vertices."""
    os.environ['AD_NN'] = str(nn)
    for m in ('ad_sine_p', 'ad_sine_n'):
        sys.modules.pop(m, None)
    p = load_physics('ad_sine_p', HERE)
    n = load_numerics('ad_sine_n', HERE)
    so = default_so; so.name = p.name = 'adsine%d' % nn; so.sList = [default_s]
    runopts.logLevel = 1; runopts.verbose = False
    ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runopts)
    ns.calculateSolution('adsine')
    lm = ns.modelList[0].levelModelList[-1]
    dofMap = lm.u[0].femSpace.dofMap
    l2g = np.array(dofMap.l2g)
    dofx = np.array(dofMap.lagrangeNodesArray) if hasattr(dofMap, 'lagrangeNodesArray') else np.array(lm.mesh.nodeArray)
    assert np.array_equal(l2g[:, :nV], lm.mesh.elementNodesArray)
    return np.array(lm.u[0].dof), dofx, l2g


def plane_faces(nodes, l2g, axis, value):
    """Element faces (element, local face index of the space) whose vertices all lie on the plane x[axis] = value.
    Per plane, because a face with its vertices on two different planes is an interior face."""
    on = np.abs(nodes[l2g[:, :nV]][:, :, axis] - value) < 1e-12                # (nE, nV)
    on_face = np.stack([on[:, fv].all(axis=1) for fv in FACE_VERTICES], axis=1)  # (nE, nFaces)
    return np.nonzero(on_face)


def engine_solution(nodes, conn):
    """nodes: the coordinates of every global dof; conn: the element-to-dof map (nE, nD)."""
    nE, nN = len(conn), len(nodes)
    xn = np.ascontiguousarray(nodes[conn])                                      # (nE, nD, 3): Proteus keeps 3 coordinates
    u0 = np.zeros((nE, 1, nD))
    R, J = adrh_proto.advection_diffusion_sine(ND, SPACE, b, kappa, omega, omega0, xn, u0, qp, qw)
    rows = np.repeat(conn, nD, axis=1).ravel(); cols = np.tile(conn, (1, nD)).ravel()
    K = sp.coo_matrix((J[:, 0, :, 0, :].ravel(), (rows, cols)), shape=(nN, nN)).tocsr()
    rhs = -np.bincount(conn.ravel(), weights=R[:, 0, :].ravel(), minlength=nN)
    if FACES[BC]:
        fe, fk, kind = [], [], []
        for code, ax, v in FACES[BC]:
            e, k = plane_faces(nodes, conn, ax, v)
            fe.append(e); fk.append(k); kind.append(np.full(len(e), code))
        fe, fk, kind = (np.concatenate(a) for a in (fe, fk, kind))
        pc, pp, sig = WEAK.get(BC, (2.0, 1.0, 0.0))
        face_conn = conn[fe]
        Rf, Jf = adrh_proto.advection_diffusion_sine_faces(ND, SPACE, b, kappa, omega, omega0,
                                                           np.ascontiguousarray(fk, dtype='i'),
                                                           np.ascontiguousarray(kind, dtype='i'),
                                                           np.ascontiguousarray(nodes[face_conn]), np.zeros((len(fe), 1, nD)),
                                                           qpf, qwf, pc, pp, sig, ROBIN_ALPHA, ROBIN_BETA)
        rows_f = np.repeat(face_conn, nD, axis=1).ravel(); cols_f = np.tile(face_conn, (1, nD)).ravel()
        K = K + sp.coo_matrix((Jf[:, 0, :, 0, :].ravel(), (rows_f, cols_f)), shape=(nN, nN)).tocsr()
        rhs = rhs - np.bincount(face_conn.ravel(), weights=Rf[:, 0, :].ravel(), minlength=nN)
    # the dofs whose rows the driver replaces by the Dirichlet value (Proteus: every Lagrange node getDBC accepts)
    X = nodes[:, :ND]
    lo, hi = X.min(axis=1) < 1e-12, X.max(axis=1) > 1 - 1e-12
    if BC in WEAK:
        onb = np.zeros(nN, bool)
    elif BC in ('mixed', 'robin'):
        onb = lo | (X[:, 1:].max(axis=1) > 1 - 1e-12)
    elif BC == 'inflow':
        onb = (X[:, 1:].min(axis=1) < 1e-12) | (X[:, 1:].max(axis=1) > 1 - 1e-12)
    else:
        onb = lo | hi
    ub = exact(nodes)
    free = ~onb
    Kff = K[free][:, free]
    rhs_f = rhs[free] - K[free][:, onb] @ ub[onb]
    u = ub.copy(); u[free] = spl.spsolve(Kff.tocsc(), rhs_f)
    return u, xn, R, J


W_BASIS = adrh_proto.basis_values(ND, SPACE, qp)[0]                  # (nQ, nD) the space's basis at the quadrature points


def errors(u, nodes, conn, xn):
    # L2 error by the engine's quadrature and geometry, with u_h interpolated by the space's basis at the points
    uh = u[conn] @ W_BASIS.T                                          # (nE, nQ)
    xq, adet = adrh_proto.quadrature_geometry(ND, SPACE, xn, qp)      # (nE, nQ, 3), (nE, nQ)
    l2 = np.sqrt(np.sum(adet * qw[None, :] * (uh - exact(xq)) ** 2))
    return l2, np.abs(u - exact(nodes)).max()


sizes = [int(a) for a in sys.argv[1:]] or [5, 9, 17]
prev = None
print("AD_ND=%d AD_SPACE=%s AD_BC=%s" % (ND, SPACE, BC))
print("%4s %9s %12s %12s %14s %14s %9s %9s %14s" % ("nn", "elements", "|eng-prot|", "eng-vs-ex L2", "prot-vs-ex L2", "eng max err", "L2 rate", "max rate", "res(exact) inf"))
for nn in sizes:
    up, nodes, conn = proteus_solution(nn)                           # nodes = every dof's coordinates
    ue, xn, R, J = engine_solution(nodes, conn)
    l2e, mxe = errors(ue, nodes, conn, xn)
    l2p, mxp = errors(up, nodes, conn, xn)
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
