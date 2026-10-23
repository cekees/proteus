"""Advection-diffusion regression: ADRH engine vs the Proteus model (ad_sine_p/n.py) vs the exact solution.

Steady div(b u - kappa grad u) + d(x) = 0 on the unit cube, exact solution sin(omega.x + omega0) (the
verified AnalyticalSolutions.LinearADR_Sine with c = 0), P1 Galerkin on regular tetrahedra, strong
Dirichlet data from the exact solution. The engine builds the element residual and Jacobian; the driver
assembles, imposes the Dirichlet rows and solves with scipy. Checks:

  1. engine and Proteus solve the same discrete problem: nodal solutions agree to solver round-off.
     AD_BC picks the boundary treatment on both sides (ad_sine_p.py has the Proteus side):
       strong (default)  Dirichlet data everywhere, imposed strongly by the driver (Proteus: numericalFluxType=None)
       mixed             the face x = 1 is a flux face: prescribed diffusive flux, advection out by the interior value
                         (Proteus: StrongDirichletFactory({0: 'outFlow'}); AD_MIXED=1 means this)
       inflow            as mixed, and the inflow face x = 0 has the advective flux (b . n) u and the diffusive flux
                         prescribed (the engine's Flux kind with hasAdv)
       iipg, nipg, sipg  Dirichlet data everywhere, imposed weakly: upwinded advective flux against f(g), diffusive flux
                         with penalty C/h (2, 10, 100) and the NIPG/SIPG adjoint term (the engine's WeakDirichlet kind
                         against Proteus's Advection_DiagonalUpwind_Diffusion_*_exterior);
  2. both converge to the exact solution at the P1 rates (L2 ~ h^2, max-norm ~ h^2);
  3. the exact solution fed through the engine's residual is small and shrinks with h (consistency).

usage: [AD_BC=mode] python ad_regression.py [nn ...]        default 5 9 17
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
from proteus import AnalyticalSolutions

b = np.array([1.0, 0.5, 0.25]); kappa = 0.2; omega = np.array([3.0, 2.0, 4.0]); omega0 = 0.3
order = 4
os.environ.setdefault('AD_QORDER', str(order))
quad = Quadrature.SimplexGaussQuadrature(3, order)
qp = np.array([list(p)[:3] for p in quad.points]); qw = np.array(quad.weights)
# verified exact solution (LinearADR_Sine with c = 0): same u as the functor's source assumes
exact_cls = AnalyticalSolutions.LinearADR_Sine(b=b, a=kappa * np.eye(3), c=0.0, omega=omega, omega0=omega0)
exact = lambda X: np.sin(X @ omega + omega0)


def proteus_solution(nn):
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
    return np.array(lm.u[0].dof), np.array(lm.mesh.nodeArray), np.array(lm.mesh.elementNodesArray)


BC = os.environ.get('AD_BC', 'mixed' if os.environ.get('AD_MIXED', '0') == '1' else 'strong')
WEAK = {'iipg': (2.0, 1.0, 0.0), 'nipg': (10.0, 1.0, -1.0), 'sipg': (100.0, 1.0, 1.0)}   # C, p, sigma as in Proteus
assert BC in ('strong', 'mixed', 'inflow') or BC in WEAK, BC
os.environ['AD_BC'] = BC
quadf = Quadrature.SimplexGaussQuadrature(2, order)
qpf = np.array([list(p)[:2] for p in quadf.points]); qwf = np.array(quadf.weights)
# the faces the engine's boundary kernel sees, by plane: (kind code of SineBC, axis, value); the rest is strong Dirichlet
FACES = {'strong': [], 'mixed': [(1, 0, 1.0)], 'inflow': [(2, 0, 0.0), (1, 0, 1.0)]}
for mode in WEAK:
    FACES[mode] = [(3, ax, v) for ax in range(3) for v in (0.0, 1.0)]


def plane_faces(nodes, conn, axis, value):
    """Element faces (element, local face = the opposite node) whose three nodes lie on the plane x[axis] = value.
    Per plane, because a face with its nodes on two different planes is an interior face."""
    on = np.abs(nodes[conn][:, :, axis] - value) < 1e-12                       # (nE, 4)
    return np.nonzero(on.sum(axis=1)[:, None] - on.astype(int) == 3)


def engine_solution(nodes, conn):
    nE, nN = len(conn), len(nodes)
    xn = np.ascontiguousarray(nodes[conn])
    u0 = np.zeros((nE, 1, 4))
    R, J = adrh_proto.advection_diffusion_sine(b, kappa, omega, omega0, xn, u0, qp, qw)
    rows = np.repeat(conn, 4, axis=1).ravel(); cols = np.tile(conn, (1, 4)).ravel()
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
        Rf, Jf = adrh_proto.advection_diffusion_sine_faces(b, kappa, omega, omega0, np.ascontiguousarray(fk, dtype='i'),
                                                           np.ascontiguousarray(kind, dtype='i'),
                                                           np.ascontiguousarray(nodes[face_conn]), np.zeros((len(fe), 1, 4)),
                                                           qpf, qwf, pc, pp, sig)
        rows_f = np.repeat(face_conn, 4, axis=1).ravel(); cols_f = np.tile(face_conn, (1, 4)).ravel()
        K = K + sp.coo_matrix((Jf[:, 0, :, 0, :].ravel(), (rows_f, cols_f)), shape=(nN, nN)).tocsr()
        rhs = rhs - np.bincount(face_conn.ravel(), weights=Rf[:, 0, :].ravel(), minlength=nN)
    # the nodes whose rows the driver replaces by the Dirichlet value
    lo, hi = nodes.min(axis=1) < 1e-12, nodes.max(axis=1) > 1 - 1e-12
    if BC in WEAK:
        onb = np.zeros(nN, bool)
    elif BC == 'mixed':
        onb = lo | (nodes[:, 1:].max(axis=1) > 1 - 1e-12)
    elif BC == 'inflow':
        onb = (nodes[:, 1:].min(axis=1) < 1e-12) | (nodes[:, 1:].max(axis=1) > 1 - 1e-12)
    else:
        onb = lo | hi
    ub = exact(nodes)
    free = ~onb
    Kff = K[free][:, free]
    rhs_f = rhs[free] - K[free][:, onb] @ ub[onb]
    u = ub.copy(); u[free] = spl.spsolve(Kff.tocsc(), rhs_f)
    return u, xn, R, J


def errors(u, nodes, conn, xn):
    # L2 error by the engine's quadrature, with u_h interpolated barycentrically at the points
    lam = np.column_stack([1 - qp.sum(axis=1), qp])
    uh = u[conn] @ lam.T                                              # (nE, nQ)
    xq = np.einsum('qb,ebi->eqi', lam, xn)
    d = np.einsum('ebi,bj->eij', xn[:, 1:, :] - xn[:, :1, :], np.eye(3))  # edge matrix (nE, 3, 3) rows = edges
    vol = np.abs(np.linalg.det(xn[:, 1:, :] - xn[:, :1, :]))
    l2 = np.sqrt(np.sum(vol[:, None] * qw[None, :] * (uh - exact(xq)) ** 2))
    return l2, np.abs(u - exact(nodes)).max()


sizes = [int(a) for a in sys.argv[1:]] or [5, 9, 17]
prev = None
print("AD_BC=%s" % BC)
print("%4s %9s %12s %12s %14s %14s %9s %9s %14s" % ("nn", "elements", "|eng-prot|", "eng-vs-ex L2", "prot-vs-ex L2", "eng max err", "L2 rate", "max rate", "res(exact) inf"))
for nn in sizes:
    up, nodes, conn = proteus_solution(nn)
    ue, xn, R, J = engine_solution(nodes, conn)
    l2e, mxe = errors(ue, nodes, conn, xn)
    l2p, mxp = errors(up, nodes, conn, xn)
    # the exact solution through the engine: element residual of u_exact (interior elements only, no boundary terms)
    uex = exact(nodes)[conn][:, None, :]
    Rx = adrh_proto.advection_diffusion_sine(b, kappa, omega, omega0, xn, np.ascontiguousarray(uex), qp, qw)[0]
    resid = np.abs(np.bincount(conn.ravel(), weights=Rx[:, 0, :].ravel(), minlength=len(nodes))[(nodes.min(axis=1) > 1e-12) & (nodes.max(axis=1) < 1 - 1e-12)]).max()
    h = 1.0 / (nn - 1)
    rate = ("%9.2f %9.2f" % (np.log(prev[0] / l2e) / np.log(prev[1] / h), np.log(prev[2] / mxe) / np.log(prev[1] / h))) if prev else "%9s %9s" % ("-", "-")
    print("%4d %9d %12.2e %12.3e %14.3e %14.3e %s %14.3e" % (nn, len(conn), np.abs(ue - up).max(), l2e, l2p, mxe, rate, resid))
    prev = (l2e, h, mxe)
