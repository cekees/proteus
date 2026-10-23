"""Advection-diffusion regression: ADRH engine vs the Proteus model (ad_sine_p/n.py) vs the exact solution.

Steady div(b u - kappa grad u) + d(x) = 0 on the unit cube, exact solution sin(omega.x + omega0) (the
verified AnalyticalSolutions.LinearADR_Sine with c = 0), P1 Galerkin on regular tetrahedra, strong
Dirichlet data from the exact solution. The engine builds the element residual and Jacobian; the driver
assembles, imposes the Dirichlet rows and solves with scipy. Checks:

  1. engine and Proteus solve the same discrete problem: nodal solutions agree to solver round-off;
  2. both converge to the exact solution at the P1 rates (L2 ~ h^2, max-norm ~ h^2);
  3. the exact solution fed through the engine's residual is small and shrinks with h (consistency).

usage: python ad_regression.py [nn ...]        default 5 9 17
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


def engine_solution(nodes, conn):
    nE, nN = len(conn), len(nodes)
    xn = np.ascontiguousarray(nodes[conn])
    u0 = np.zeros((nE, 1, 4))
    R, J = adrh_proto.advection_diffusion_sine(b, kappa, omega, omega0, xn, u0, qp, qw)
    rows = np.repeat(conn, 4, axis=1).ravel(); cols = np.tile(conn, (1, 4)).ravel()
    K = sp.coo_matrix((J[:, 0, :, 0, :].ravel(), (rows, cols)), shape=(nN, nN)).tocsr()
    rhs = -np.bincount(conn.ravel(), weights=R[:, 0, :].ravel(), minlength=nN)
    onb = (nodes.min(axis=1) < 1e-12) | (nodes.max(axis=1) > 1 - 1e-12)
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
