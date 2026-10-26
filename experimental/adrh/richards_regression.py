"""Richards' equation, the engine against proteus's richards model: test/richards/test_1 (Celia et al. 1990, Fig. 6b).

A 1 m column of sand (Mualem-van Genuchten: alpha 3.35, n 2, theta_r 0.102, theta_s 0.368, K_s 7.967 m/d), initial pressure head
-10 m, held at -10 m at the bottom and -0.75 m at the top, gravity pointing to the bottom; one day in 1000 backward Euler steps on 40
elements, Galerkin with the consistent mass matrix (proteus's STABILIZATION_TYPE=0 on that deck). The engine (richards.h, the kernel
element by element, Newton with a backtracking line search and a direct solve here) is compared with the profiles proteus saved in
test/richards/comparison_files/richards_test_1.csv at 12 h (frame 500) and 24 h (frame 1000), the same files test_richards.py checks
proteus against at 8 decimals.

The deck's numerical flux, Richards_IIPG_exterior, imposes the Dirichlet values WEAKLY: the interior advective flux and the usual
diffusive flux through the face plus a penalty 100 (u - g) (the face diameter of a point is 1), no adjoint term. That is the default here
(--bc weak); --bc strong replaces the two boundary rows instead, which is a different scheme and moves the front by a fraction of an element.

Proteus stops its Newton iteration when the 2-norm of the residual is below nl_atol_res = 1e-8, which at the dry nodes (a tiny capacity)
leaves an error of 1e-7 to 4e-6 in the pressure head. The engine reproduces the saved profiles to round-off (about 2.5e-13) when it stops by the
same rule (the default); converged to 1e-12 it differs from them by exactly that tolerance error, and the second line printed says so.

usage: [ADRH_MODULE=adrh_kokkos] python richards_regression.py [--bc weak|strong] [--stabilize|--fct] [--nn 41] [--steps 1000]

--stabilize replaces the consistent mass by the lumped mass and adds the discrete upwinding of lowOrder.h: a different scheme from every saved
column, so it is checked for its bounds (the pressure head stays between the initial and the boundary data) and compared with them.
"""
import argparse
import csv
import importlib
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
engine = importlib.import_module(os.environ.get('ADRH_MODULE', 'adrh_proto'))
from proteus import Quadrature

CSV = os.environ.get('RICHARDS_CSV', os.path.join(HERE, '..', '..', 'test', 'richards', 'comparison_files', 'richards_test_1.csv'))

# the deck's parameters, computed as the deck computes them
viscosity, density, gravity = 8.9e-4, 998.2, 9.8
m_per_s_by_m_per_d = 1.1574074e-5
permeability = (7.967 * m_per_s_by_m_per_d) * viscosity / (gravity * density)
KS = (1.0 * density * gravity * permeability / (viscosity * 1.0)) / m_per_s_by_m_per_d
ALPHA, N_VG, THETA_S, THETA_R = 3.35, 2.0, 0.368, 0.102
PONDING, BOTTOM, T_END = -0.75, -10.0, 1.0


class Column:
    def __init__(self, nn, bc='weak', stabilize=False, fct=False):
        self.nn = nn
        self.bc = bc
        self.fct = fct
        stabilize = stabilize or fct
        self.stabilize = stabilize
        self.x = np.linspace(0.0, 1.0, nn)
        self.conn = np.stack([np.arange(nn - 1), np.arange(1, nn)], axis=1)
        self.xn = np.zeros((nn - 1, 2, 3)); self.xn[:, :, 0] = self.x[self.conn]
        quad = Quadrature.SimplexGaussQuadrature(1, 3)
        self.qp = np.array([list(p)[:1] for p in quad.points]); self.qw = np.array(quad.weights)
        self.fixed = np.array([0, nn - 1])
        if stabilize:                                           # tridiagonal pattern and its transpose map, once
            P = sp.diags([np.ones(nn - 1), np.ones(nn), np.ones(nn - 1)], [-1, 0, 1], format='csr'); P.sort_indices()
            self.indptr, self.indices = P.indptr.astype('i'), P.indices.astype('i')
            self.tpos = engine.transpose_map(self.indptr, self.indices)
        # the two boundary faces: the bottom point is the left vertex (local face 1) of element 0, the top the right vertex (face 0)
        # of the last element; a face of a segment has no quadrature coordinates
        self.faceElem = np.array([0, nn - 2]); self.faceLocal = np.array([1, 0], dtype='i')
        self.gD = np.array([BOTTOM, PONDING]); self.qpf = np.zeros((1, 0)); self.qwf = np.array([1.0])

    def material_args(self):
        return (np.array([ALPHA]), np.array([N_VG]), np.array([THETA_R]), np.array([THETA_S - THETA_R]), np.array([[KS]]),
                np.array([-1.0]), 1.0, 0.0, np.zeros(self.nn - 1, dtype="i"))

    def spatial(self, u, t, dt=0.0, uold=None):
        """Element and boundary-face residual and Jacobian (CSR) of the spatial operator; with dt > 0 the element kernel adds its consistent mass."""
        mat = self.material_args()
        ue = u[self.conn].reshape(-1, 1, 2); uo = (uold if uold is not None else u)[self.conn].reshape(-1, 1, 2)
        R, J = engine.richards_vgm(1, "p1", *mat[:4], mat[4], mat[5], mat[6], mat[7], mat[8], self.xn, ue, uo, self.qp, self.qw, t, dt)
        n = self.nn
        rows = np.repeat(self.conn, 2, axis=1).ravel(); cols = np.tile(self.conn, (1, 2)).ravel()
        A = sp.coo_matrix((J.reshape(-1, 2, 2).ravel(), (rows, cols)), shape=(n, n)).tocsr()
        r = np.zeros(n); np.add.at(r, self.conn.ravel(), R.reshape(-1, 2).ravel())
        if self.bc == 'weak':
            ef = self.conn[self.faceElem]
            Rf, Jf = engine.richards_vgm_faces(1, "p1", *mat[:4], mat[4], mat[5], mat[6], mat[7], mat[8],
                                               self.faceElem.astype("i"), self.faceLocal, self.gD, self.xn[self.faceElem],
                                               u[ef].reshape(-1, 1, 2), self.qpf, self.qwf, t, 100.0, 1.0)
            np.add.at(r, ef.ravel(), Rf.reshape(-1, 2).ravel())
            rows = np.repeat(ef, 2, axis=1).ravel(); cols = np.tile(ef, (1, 2)).ravel()
            A = A + sp.coo_matrix((Jf.reshape(-1, 2, 2).ravel(), (rows, cols)), shape=(n, n)).tocsr()
        return r, A

    def system(self, u, uold, t, dt):
        n = self.nn; mat = self.material_args()
        r, A = self.spatial(u, t, 0.0 if self.stabilize else dt, uold)
        if self.stabilize:
            ue = u[self.conn].reshape(-1, 1, 2); uo = uold[self.conn].reshape(-1, 1, 2)
            A.sort_indices()
            Jv, r = engine.algebraic_dissipation(self.indptr, self.indices, self.tpos, A.data.copy(), u, r)
            w, m, dm, mold = engine.richards_vgm_lumped(1, "p1", *mat[:4], mat[4], mat[5], mat[6], mat[7], mat[8], self.xn, ue, uo,
                                                        self.qp, self.qw, t, dt)
            rm = np.zeros(n); np.add.at(rm, self.conn.ravel(), (w * (m - mold) / dt).ravel())
            dg = np.zeros(n); np.add.at(dg, self.conn.ravel(), (w * dm / dt).ravel())
            r = r + rm
            A = sp.csr_matrix((Jv, self.indices, self.indptr), shape=(n, n)) + sp.diags(dg)
        return r, A

    def fct_correct(self, uL, uold, t, dt):
        """Flux-corrected transport on the moisture content: the low-order solution uL (lumped mass, discrete upwinding) is moved toward the Galerkin
        solution as far as the local extrema of the low-order and old moisture contents allow, and the pressure head is recovered from the
        limited moisture content with the analytic van Genuchten inverse."""
        n = self.nn; mat = self.material_args()
        ue = uL[self.conn].reshape(-1, 1, 2); uo = uold[self.conn].reshape(-1, 1, 2)
        r, A = self.spatial(uL, t)                           # the spatial operator alone (elements and faces), at the low-order solution
        A.sort_indices()
        d = engine.dissipation_values(self.indptr, self.indices, self.tpos, A.data.copy())
        w, m, dm, mold = engine.richards_vgm_lumped(1, "p1", *mat[:4], mat[4], mat[5], mat[6], mat[7], mat[8], self.xn, ue, uo, self.qp, self.qw, t, dt)
        ML = np.zeros(n); np.add.at(ML, self.conn.ravel(), w.ravel())
        mL = np.zeros(n); np.add.at(mL, self.conn.ravel(), (w * m).ravel()); mL /= ML
        mo = np.zeros(n); np.add.at(mo, self.conn.ravel(), (w * mold).ravel()); mo /= ML
        Mel = engine.consistent_mass(1, "p1", self.xn, self.qp, self.qw)[0]
        rows = np.repeat(self.conn, 2, axis=1).ravel(); cols = np.tile(self.conn, (1, 2)).ravel()
        Mc = sp.coo_matrix((Mel.reshape(-1, 2, 2).ravel(), (rows, cols)), shape=(n, n)).tocsr(); Mc.sort_indices()
        mnew = engine.fct_limit(self.indptr, self.indices, self.tpos, Mc.data.copy(), d, ML, uL, mL, mo, dt)
        se = np.clip((mnew - THETA_R) / (THETA_S - THETA_R), 1e-12, 1.0 - 1e-15)
        mvg = 1.0 - 1.0 / N_VG
        return -((se ** (-1.0 / mvg) - 1.0) ** (1.0 / N_VG)) / ALPHA, mnew, ML

    def strong(self, r, A, u):
        """Strong Dirichlet rows: residual u - g, identity row (a no-op in the weak mode)."""
        if self.bc != 'strong':
            return r, A
        g = np.array([BOTTOM, PONDING])
        r = r.copy(); r[self.fixed] = u[self.fixed] - g
        A = A.tolil()
        for i in self.fixed:
            A.rows[i] = [i]; A.data[i] = [1.0]
        return r, A.tocsr()


def newton(col, uold, t, dt, tol, norm=2, maxit=60):
    """Newton with a backtracking line search; stops when the residual's norm (2: proteus's rule, or 'inf') is below tol."""
    nrm = (lambda v: np.linalg.norm(v)) if norm == 2 else (lambda v: np.abs(v).max())
    u = uold.copy()
    r, A = col.system(u, uold, t, dt); r, A = col.strong(r, A, u)
    its = 0
    while nrm(r) > tol and its < maxit:
        du = spl.spsolve(A.tocsc(), -r)
        lam, r0 = 1.0, nrm(r)
        while True:                                              # backtracking line search on the residual norm
            un = u + lam * du
            rn, An = col.system(un, uold, t, dt); rn, An = col.strong(rn, An, un)
            if nrm(rn) < (1.0 - 1e-4 * lam) * r0 or lam < 1e-4:
                break
            lam *= 0.5
        u, r, A = un, rn, An
        its += 1
    if nrm(r) > tol:
        raise RuntimeError("Newton did not converge at t=%g: |r| = %.2e after %d iterations" % (t, nrm(r), its))
    return u, its


def read_reference():
    with open(CSV) as f:
        rows = list(csv.DictReader(f))
    x = np.array([float(r['x']) for r in rows])
    return x, {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != 'x'}


def run(col, steps, tol, norm):
    u = np.full(col.nn, BOTTOM); u[-1] = PONDING
    dt = T_END / steps
    t0 = time.perf_counter(); frames = {}; total_its = 0
    for step in range(1, steps + 1):
        uold = u
        u, its = newton(col, u, step * dt, dt, tol, norm); total_its += its
        if col.fct:
            u, _, _ = col.fct_correct(u, uold, step * dt, dt)
        if step in (steps // 2, steps): frames[step] = u.copy()
    return frames, total_its, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bc', default='weak', choices=['weak', 'strong'])
    ap.add_argument('--stabilize', action='store_true', help='lumped mass + discrete upwinding (lowOrder.h)')
    ap.add_argument('--fct', action='store_true', help='stabilize, then the FCT limiter on the moisture content')
    ap.add_argument('--nn', type=int, default=41)
    ap.add_argument('--steps', type=int, default=1000)
    a = ap.parse_args()
    col = Column(a.nn, a.bc, a.stabilize, a.fct)
    dt = T_END / a.steps
    ref = None
    if a.nn == 41 and a.steps == 1000:
        x, ref = read_reference()
        assert np.allclose(x, col.x)
    results = {}
    for label, tol, norm in (("proteus's rule (2-norm < 1e-8)", 1e-8, 2), ("converged (max-norm < 1e-12)", 1e-12, 'inf')):
        frames, its, wall = run(col, a.steps, tol, norm)
        results[label] = frames
        print("%s Dirichlet, Newton stopped at %s: %d steps of %.4g d, %d iterations (%.2f per step), %.1f s with %s" %
              (a.bc, label, a.steps, dt, its, its / a.steps, wall, engine.__name__))
        if ref is not None:
            for frame, tag in ((a.steps // 2, 't500'), (a.steps, 't1000')):
                if not (a.stabilize or a.fct):
                    d = np.abs(frames[frame] - ref['stab_0_' + tag])
                    print("    t = %.3f d: against proteus (stab_0_%s) max |difference| = %.3e at x = %.3f" % (frame * dt, tag, d.max(), col.x[d.argmax()]))
                else:                                           # a different scheme from every saved column: compare, and check the bounds
                    u = frames[frame]
                    print("    t = %.3f d: psi in [%.4f, %.4f] (data: [%.2f, %.2f]); against proteus's  Galerkin %.2e  entropy viscosity %.2e  FCT %.2e (max |diff|)" %
                          (frame * dt, u.min(), u.max(), BOTTOM, PONDING, np.abs(u - ref['stab_0_' + tag]).max(),
                           np.abs(u - ref['stab_2_' + tag]).max(), np.abs(u - ref['FCT_' + tag]).max()))
    if ref is None:
        print("  (the saved profiles are for nn = 41 and 1000 steps)")


if __name__ == '__main__':
    main()
