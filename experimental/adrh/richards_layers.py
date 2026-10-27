"""Heterogeneous Richards: a two-layer column, the engine against an ODE solution that uses no finite elements.

Steady one-dimensional flow has a constant flux, Q = f - a psi' with f = -K(psi) (gravity points down) and a = K(psi), so each layer obeys
        psi' = -Q / K(psi) - 1,        K(psi) = K_r(psi; alpha, n) K_s,
with the pressure head continuous across the interface. Given psi at the bottom the whole profile follows from Q, and Q is found by
matching psi at the top (shooting, brentq). The engine solves the same problem as a steady Galerkin system, P1 elements, strong
Dirichlet rows, Newton, with RichardsMaterials giving each element its layer's van Genuchten-Mualem parameters through bindElement
(richards.h). The nodal error against the ODE solution should fall at the P1 rate as the mesh is refined, and the host and Kokkos
modules must agree.

usage: [ADRH_MODULE=adrh_kokkos] python richards_layers.py
"""
import importlib
import os
import sys

import numpy as np
import scipy.integrate as si
import scipy.optimize as so
import scipy.sparse as sp
import scipy.sparse.linalg as spl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
engine = importlib.import_module(os.environ.get('ADRH_MODULE', 'adrh_proto'))
from proteus import Quadrature

# material 0 below (a sand), material 1 above (a finer soil, two orders of magnitude less conductive near saturation)
ALPHA = np.array([3.35, 1.0]); N_VG = np.array([2.0, 1.6]); THETA_R = np.array([0.102, 0.08]); THETA_S = np.array([0.368, 0.43])
KS = np.array([7.967, 0.05]); X_INTERFACE = 0.5
PSI_BOTTOM, PSI_TOP = -1.0, -3.0


def kr(psi, alpha, n):
    """Mualem-van Genuchten relative conductivity of the pressure head psi (negative: unsaturated)."""
    m = 1.0 - 1.0 / n
    pc = alpha * max(-psi, 1e-8)
    s = (1.0 + pc ** n) ** (-m)
    return np.sqrt(s) * (1.0 - pc ** (n - 1.0) * s) ** 2 if psi < 0 else 1.0


def profile(Q, x):
    """psi at the points x (sorted) by integrating psi' = -Q / K(psi) - 1 from the bottom through both layers; NaN where the
    integration fails (a trial flux for which no profile reaches that far)."""
    out = np.full(x.shape, np.nan); psi0 = PSI_BOTTOM; x0 = 0.0
    for layer, x1 in ((0, X_INTERFACE), (1, 1.0)):
        rhs = lambda xx, p, L=layer: [-Q / (kr(p[0], ALPHA[L], N_VG[L]) * KS[L]) - 1.0]
        dry = lambda xx, p: p[0] + 400.0               # a trial flux that drives psi below -400 m has no profile worth following
        dry.terminal = True
        sol = si.solve_ivp(rhs, (x0, x1), [psi0], rtol=1e-12, atol=1e-13, method='LSODA', dense_output=True, events=dry)
        if sol.status != 0 or sol.t[-1] < x1:
            return out, np.nan
        sel = (x >= x0) & (x <= x1) if layer == 1 else (x <= x1)
        sel &= np.isnan(out) | (layer == 1)
        out[sel] = sol.sol(x[sel])[0]
        psi0, x0 = sol.y[0][-1], x1
    return out, psi0


def exact_profile(x):
    """The shooting solution: the flux Q for which psi(1) = PSI_TOP, and the profile at x."""
    # a trial flux whose profile dries out before the top is below the target, however far: count it as -1e6
    def f(Q):
        top = profile(Q, np.array([0.0, 1.0]))[1]
        return -1.0e6 if not np.isfinite(top) else top - PSI_TOP
    grid = np.linspace(-0.5, 0.5, 201)
    vals = [f(Q) for Q in grid]
    k = next(i for i in range(len(grid) - 1) if vals[i] * vals[i + 1] < 0)
    Q = so.brentq(f, grid[k], grid[k + 1], xtol=1e-15, rtol=1e-14)
    return Q, profile(Q, x)[0]


class Column:
    def __init__(self, nn):
        self.nn = nn; self.x = np.linspace(0.0, 1.0, nn)
        self.conn = np.stack([np.arange(nn - 1), np.arange(1, nn)], axis=1)
        self.xn = np.zeros((nn - 1, 2, 3)); self.xn[:, :, 0] = self.x[self.conn]
        mid = 0.5 * (self.x[:-1] + self.x[1:])
        self.material = (mid > X_INTERFACE).astype('i')
        quad = Quadrature.SimplexGaussQuadrature(1, 3)
        self.qp = np.array([list(p)[:1] for p in quad.points]); self.qw = np.array(quad.weights)

    def system(self, u):
        ue = u[self.conn].reshape(-1, 1, 2)
        R, J = engine.richards_vgm(1, "p1", ALPHA, N_VG, THETA_R, THETA_S - THETA_R, KS.reshape(2, 1), np.array([-1.0]), 1.0, 0.0,
                                   self.material, self.xn, ue, None, self.qp, self.qw, 0.0, 0.0, 0, 0)
        n = self.nn
        rows = np.repeat(self.conn, 2, axis=1).ravel(); cols = np.tile(self.conn, (1, 2)).ravel()
        A = sp.coo_matrix((J.reshape(-1, 2, 2).ravel(), (rows, cols)), shape=(n, n)).tolil()
        r = np.zeros(n); np.add.at(r, self.conn.ravel(), R.reshape(-1, 2).ravel())
        for i, g in ((0, PSI_BOTTOM), (n - 1, PSI_TOP)):
            r[i] = u[i] - g; A.rows[i] = [i]; A.data[i] = [1.0]
        return r, A.tocsc()

    def solve(self, tol=1e-11):
        u = np.linspace(PSI_BOTTOM, PSI_TOP, self.nn)
        r, A = self.system(u); its = 0
        while np.abs(r).max() > tol and its < 100:
            du = spl.spsolve(A, -r); lam = 1.0; r0 = np.abs(r).max()
            while True:
                un = u + lam * du; rn, An = self.system(un)
                if np.abs(rn).max() < (1 - 1e-4 * lam) * r0 or lam < 1e-5: break
                lam *= 0.5
            u, r, A = un, rn, An; its += 1
        return u, its


def main():
    import signal
    signal.alarm(240)                                   # a hung integration should fail the run, not wait forever
    Q, _ = exact_profile(np.array([0.0, 1.0]))
    print("two layers, interface at x = %.2f: Ks %s m/d, alpha %s, n %s; psi(0) = %.1f, psi(1) = %.1f; steady flux Q = %.6f (ODE, shooting)" %
          (X_INTERFACE, KS, ALPHA, N_VG, PSI_BOTTOM, PSI_TOP, Q))
    print("  %5s %4s  %12s  %6s" % ("nn", "its", "max |error|", "rate"))
    prev = None
    for nn in (11, 21, 41, 81, 161, 321):
        col = Column(nn)
        u, its = col.solve()
        _, ex = exact_profile(col.x)
        err = np.abs(u - ex).max()
        print("  %5d %4d  %12.4e  %6s" % (nn, its, err, "" if prev is None else "%.2f" % np.log2(prev / err)))
        prev = err


if __name__ == '__main__':
    main()
