"""Richards' equation, the engine against proteus's richards model: the three 1D decks of test/richards (--case): test_1 (Celia et al. 1990, Fig. 6b),
test_2_HYDRUS and test_3 (Szymkiewicz 2009). The description below is of the Celia column; Case holds the others' parameters. --upwind is proteus's low-order
scheme (K at the upwind node on the total head, lumped mass, strong Dirichlet data), --upwind --fct its FCT correction; see README.md.

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

usage: [ADRH_MODULE=adrh_kokkos] python richards_regression.py [--case celia|hydrus|szymkiewicz] [--bc weak|strong] [--stabilize|--upwind] [--fct] [--nn N] [--steps N]

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

COMPARISON_DIR = os.environ.get('RICHARDS_DIR') or (os.path.dirname(os.environ['RICHARDS_CSV']) if os.environ.get('RICHARDS_CSV')
                                                     else os.path.join(HERE, '..', '..', 'test', 'richards', 'comparison_files'))

# the decks' parameters, computed as the decks compute them
viscosity, density, gravity = 8.9e-4, 998.2, 9.8
m_per_s_by_m_per_d = 1.1574074e-5


def dimensionless_ks(ks_m_per_d):
    permeability = (ks_m_per_d * m_per_s_by_m_per_d) * viscosity / (gravity * density)
    return (1.0 * density * gravity * permeability / (viscosity * 1.0)) / m_per_s_by_m_per_d


class Case:
    """One of proteus's 1D Richards benchmarks (test/richards/<deck>): a column of length L along x (x = 0 the bottom, gravity down), P1 elements,
    backward Euler in `steps` steps to t_end, Dirichlet heads at both ends imposed weakly with penalty `penalty`."""

    def __init__(self, name, csv, L, nn, alpha, n_vg, theta_r, theta_s, ks, psk, bottom, top, t_end, frames, ic, penalty=100.0, steps=1000):
        self.name, self.csv, self.L, self.nn, self.alpha, self.n_vg = name, csv, L, nn, alpha, n_vg
        self.theta_r, self.theta_s, self.ks, self.psk = theta_r, theta_s, dimensionless_ks(ks), psk
        self.bottom, self.top, self.t_end, self.frames, self.ic, self.penalty, self.steps = bottom, top, t_end, frames, ic, penalty, steps

    def initial(self, x):
        return self.ic(self, x)


CASES = {
    # Celia et al. (1990) Fig. 6b: initial head -10 m, ponded -0.75 m at the top
    'celia': Case('celia', 'richards_test_1.csv', 1.0, 41, 3.35, 2.0, 0.102, 0.368, 7.967, 0, -10.0, -0.75, 1.0, (500, 1000),
                  lambda c, x: np.where(x == c.L, c.top, c.bottom)),
    # HYDRUS-1D, a 20 m column of sand (van Genuchten alpha 14.5, n 2.68), hydrostatic about a water table at the bottom, head 0 at both ends
    'hydrus': Case('hydrus', 'richards_test_2_HYDRUS.csv', 20.0, 101, 14.5, 2.68, 0.045, 0.43, 7.128, 0, 0.0, 0.0, 2.0, (104, 229, 1000),
                   lambda c, x: c.bottom - x, penalty=1.0e4),
    # Szymkiewicz (2009) test problem 1: a 5 m dry column of Brooks-Corey sand (p_d = 7.2 cm, lambda 0.592, Mualem conductivity), 11 nodes
    'szymkiewicz': Case('szymkiewicz', 'richards_test_3.csv', 5.0, 11, 1.0 / 0.072, 0.592, 0.045, 0.43, 5.04, 2, -7.5, -0.075, 0.125, (500, 1000),
                        lambda c, x: np.where(x == c.L, c.top, c.bottom)),
}


class Column:
    def __init__(self, nn, bc='weak', stabilize=False, fct=False, case=None, upwind=False, proteus_limiter=False):
        self.case = case = case or CASES['celia']
        self.psk = case.psk
        self.nn = nn
        self.bc = bc
        self.fct = fct
        self.upwind = upwind
        self.proteus_limiter = proteus_limiter
        stabilize = stabilize or fct or upwind
        self.stabilize = stabilize
        if upwind and bc != 'strong':
            raise ValueError('the K-upwinded scheme imposes the Dirichlet values strongly, as proteus does: --bc strong')
        self.x = np.linspace(0.0, case.L, nn)
        self.conn = np.stack([np.arange(nn - 1), np.arange(1, nn)], axis=1)
        self.xn = np.zeros((nn - 1, 2, 3)); self.xn[:, :, 0] = self.x[self.conn]
        quad = Quadrature.SimplexGaussQuadrature(1, 3)
        self.qp = np.array([list(p)[:1] for p in quad.points]); self.qw = np.array(quad.weights)
        self.fixed = np.array([0, nn - 1])
        if stabilize:                                           # tridiagonal pattern and its transpose map, once
            P = sp.diags([np.ones(nn - 1), np.ones(nn), np.ones(nn - 1)], [-1, 0, 1], format='csr'); P.sort_indices()
            self.indptr, self.indices = P.indptr.astype('i'), P.indices.astype('i')
            self.tpos = engine.transpose_map(self.indptr, self.indices)
            if upwind:                                          # the stiffness K_s alone (mode 1) and the nodal materials, once
                self.S = self.stiffness(np.zeros(nn), 1).data.copy()
                self.nodeMat = np.zeros(nn, dtype='i')
        # the two boundary faces: the bottom point is the left vertex (local face 1) of element 0, the top the right vertex (face 0)
        # of the last element; a face of a segment has no quadrature coordinates
        self.faceElem = np.array([0, nn - 2]); self.faceLocal = np.array([1, 0], dtype='i')
        self.gD = np.array([case.bottom, case.top]); self.qpf = np.zeros((1, 0)); self.qwf = np.array([1.0])

    def material_args(self):
        c = self.case
        return (np.array([c.alpha]), np.array([c.n_vg]), np.array([c.theta_r]), np.array([c.theta_s - c.theta_r]), np.array([[c.ks]]),
                np.array([-1.0]), 1.0, 0.0, np.zeros(self.nn - 1, dtype="i"))

    def pattern_matrix(self, R_J):
        """Element matrices (nE, 2, 2) as a CSR on the fixed tridiagonal pattern (explicit zeros kept), sorted columns."""
        n = self.nn
        k = np.arange(n)
        rows = np.concatenate([np.repeat(self.conn, 2, axis=1).ravel(), k, k[:-1], k[1:]])
        cols = np.concatenate([np.tile(self.conn, (1, 2)).ravel(), k, k[1:], k[:-1]])
        vals = np.concatenate([R_J.reshape(-1, 2, 2).ravel(), np.zeros(3 * n - 2)])
        A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr(); A.sort_indices()
        return A

    def stiffness(self, u, mode):
        """mode 1: K_s grad phi_i . grad phi_j; mode 2: the same with the relative permeability at the quadrature points of u (frozen)."""
        mat = self.material_args()
        ue = u[self.conn].reshape(-1, 1, 2)
        _, J = engine.richards_vgm(1, "p1", *mat[:4], mat[4], mat[5], mat[6], mat[7], mat[8], self.xn, ue, ue, self.qp, self.qw, 0.0, 0.0, self.psk, mode)
        return self.pattern_matrix(J)

    def total_head(self, u):
        return u - self.material_args()[5][0] * self.x           # psi - g x (g = -1: psi + x)

    def spatial(self, u, t, dt=0.0, uold=None):
        """Element and boundary-face residual and Jacobian (CSR) of the spatial operator; with dt > 0 the element kernel adds its consistent mass."""
        mat = self.material_args()
        ue = u[self.conn].reshape(-1, 1, 2); uo = (uold if uold is not None else u)[self.conn].reshape(-1, 1, 2)
        R, J = engine.richards_vgm(1, "p1", *mat[:4], mat[4], mat[5], mat[6], mat[7], mat[8], self.xn, ue, uo, self.qp, self.qw, t, dt, self.psk, 0)
        n = self.nn
        rows = [np.repeat(self.conn, 2, axis=1).ravel()]; cols = [np.tile(self.conn, (1, 2)).ravel()]; vals = [J.reshape(-1, 2, 2).ravel()]
        r = np.zeros(n); np.add.at(r, self.conn.ravel(), R.reshape(-1, 2).ravel())
        if self.bc == 'weak':
            ef = self.conn[self.faceElem]
            Rf, Jf = engine.richards_vgm_faces(1, "p1", *mat[:4], mat[4], mat[5], mat[6], mat[7], mat[8],
                                               self.faceElem.astype("i"), self.faceLocal, self.gD, self.xn[self.faceElem],
                                               u[ef].reshape(-1, 1, 2), self.qpf, self.qwf, t, self.case.penalty, 1.0, self.psk, 0)
            np.add.at(r, ef.ravel(), Rf.reshape(-1, 2).ravel())
            rows.append(np.repeat(ef, 2, axis=1).ravel()); cols.append(np.tile(ef, (1, 2)).ravel()); vals.append(Jf.reshape(-1, 2, 2).ravel())
        # explicit zeros on the whole tridiagonal pattern: a conductivity that underflows to 0 in a dry element must not drop entries
        # (scipy's sparse sum eliminates them), because the stabilization works on the fixed pattern
        k = np.arange(n); rows.append(np.concatenate([k, k[:-1], k[1:]])); cols.append(np.concatenate([k, k[1:], k[:-1]]))
        vals.append(np.zeros(3 * n - 2))
        A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
        return r, A

    def system(self, u, uold, t, dt):
        n = self.nn; mat = self.material_args()
        if self.upwind:
            kr, dkr = engine.richards_kr_nodes(*mat[:4], self.psk, self.nodeMat, u)
            rf, Jv = engine.upwind_flux(self.indptr, self.indices, self.S, self.total_head(u), kr, dkr)
            ue = u[self.conn].reshape(-1, 1, 2); uo = uold[self.conn].reshape(-1, 1, 2)
            w, m, dm, mold = engine.richards_vgm_lumped(1, "p1", *mat[:4], mat[4], mat[5], mat[6], mat[7], mat[8], self.xn, ue, uo,
                                                        self.qp, self.qw, t, dt, self.psk, 0)
            rm = np.zeros(n); np.add.at(rm, self.conn.ravel(), (w * (m - mold) / dt).ravel())
            dg = np.zeros(n); np.add.at(dg, self.conn.ravel(), (w * dm / dt).ravel())
            A = sp.csr_matrix((Jv, self.indices, self.indptr), shape=(n, n)); A.setdiag(A.diagonal() + dg)
            return rf + rm, A
        r, A = self.spatial(u, t, 0.0 if self.stabilize else dt, uold)
        if self.stabilize:
            ue = u[self.conn].reshape(-1, 1, 2); uo = uold[self.conn].reshape(-1, 1, 2)
            A.sort_indices()
            Jv, r = engine.algebraic_dissipation(self.indptr, self.indices, self.tpos, A.data.copy(), u, r)
            w, m, dm, mold = engine.richards_vgm_lumped(1, "p1", *mat[:4], mat[4], mat[5], mat[6], mat[7], mat[8], self.xn, ue, uo,
                                                        self.qp, self.qw, t, dt, self.psk, 0)
            rm = np.zeros(n); np.add.at(rm, self.conn.ravel(), (w * (m - mold) / dt).ravel())
            dg = np.zeros(n); np.add.at(dg, self.conn.ravel(), (w * dm / dt).ravel())
            r = r + rm
            A = sp.csr_matrix((Jv, self.indices, self.indptr), shape=(n, n)); A.setdiag(A.diagonal() + dg)
        return r, A

    def fct_correct(self, uL, uold, t, dt):
        """Flux-corrected transport on the moisture content: the low-order solution uL (lumped mass, discrete upwinding) is moved toward the Galerkin
        solution as far as the local extrema of the low-order and old moisture contents allow, and the pressure head is recovered from the
        limited moisture content with the analytic van Genuchten inverse."""
        n = self.nn; mat = self.material_args()
        ue = uL[self.conn].reshape(-1, 1, 2); uo = uold[self.conn].reshape(-1, 1, 2)
        if self.upwind:
            return self.fct_correct_upwind(uL, uold, t, dt)
        r, A = self.spatial(uL, t)                           # the spatial operator alone (elements and faces), at the low-order solution
        A.sort_indices()
        d = engine.dissipation_values(self.indptr, self.indices, self.tpos, A.data.copy())
        w, m, dm, mold = engine.richards_vgm_lumped(1, "p1", *mat[:4], mat[4], mat[5], mat[6], mat[7], mat[8], self.xn, ue, uo, self.qp, self.qw, t, dt, self.psk, 0)
        ML = np.zeros(n); np.add.at(ML, self.conn.ravel(), w.ravel())
        mL = np.zeros(n); np.add.at(mL, self.conn.ravel(), (w * m).ravel()); mL /= ML
        mo = np.zeros(n); np.add.at(mo, self.conn.ravel(), (w * mold).ravel()); mo /= ML
        Mel = engine.consistent_mass(1, "p1", self.xn, self.qp, self.qw)[0]
        rows = np.repeat(self.conn, 2, axis=1).ravel(); cols = np.tile(self.conn, (1, 2)).ravel()
        Mc = sp.coo_matrix((Mel.reshape(-1, 2, 2).ravel(), (rows, cols)), shape=(n, n)).tocsr(); Mc.sort_indices()
        mnew = engine.fct_limit(self.indptr, self.indices, self.tpos, Mc.data.copy(), d, ML, uL, mL, mo, dt)
        return self.invert(mnew, uL), mnew, ML

    def fct_correct_upwind(self, uL, uold, t, dt):
        """FCT on the K-upwinded scheme: the antidiffusive flux is the Galerkin flux (K at the quadrature points) minus the upwinded one."""
        n = self.nn; mat = self.material_args()
        ue = uL[self.conn].reshape(-1, 1, 2); uo = uold[self.conn].reshape(-1, 1, 2)
        kr, _ = engine.richards_kr_nodes(*mat[:4], self.psk, self.nodeMat, uL)
        C = self.stiffness(uL, 2)
        A = engine.antidiffusive_flux(self.indptr, self.indices, C.data.copy(), self.S, self.total_head(uL), kr)
        w, m, dm, mold = engine.richards_vgm_lumped(1, "p1", *mat[:4], mat[4], mat[5], mat[6], mat[7], mat[8], self.xn, ue, uo, self.qp, self.qw, t, dt, self.psk, 0)
        ML = np.zeros(n); np.add.at(ML, self.conn.ravel(), w.ravel())
        mL = np.zeros(n); np.add.at(mL, self.conn.ravel(), (w * m).ravel()); mL /= ML
        mo = np.zeros(n); np.add.at(mo, self.conn.ravel(), (w * mold).ravel()); mo /= ML
        Mel = engine.consistent_mass(1, "p1", self.xn, self.qp, self.qw)[0]
        Mc = self.pattern_matrix(Mel)
        # proteus_limiter: the local extrema over the low-order moisture content alone (proteus's non-monolithic FCTStep), not over the old one too
        mnew = engine.fct_limit_flux(self.indptr, self.indices, self.tpos, Mc.data.copy(), A, ML, mL, mo, dt, not self.proteus_limiter)
        u = self.invert(mnew, uL)
        u[self.fixed] = uL[self.fixed]
        return u, mnew, ML

    def invert(self, theta, uL):
        """The pressure head of a moisture content: the analytic inverse of the retention curve (van Genuchten, or Brooks-Corey, whose flat part
        inside the entry pressure keeps the low-order head when the limited moisture content is saturated)."""
        c = self.case
        se = (theta - c.theta_r) / (c.theta_s - c.theta_r)
        if c.psk == 0:
            se = np.clip(se, 1e-12, 1.0 - 1e-15)
            mvg = 1.0 - 1.0 / c.n_vg
            return -((se ** (-1.0 / mvg) - 1.0) ** (1.0 / c.n_vg)) / c.alpha
        wet = se >= 1.0
        return np.where(wet, np.maximum(uL, -1.0 / c.alpha), -np.clip(se, 1e-300, 1.0) ** (-1.0 / c.n_vg) / c.alpha)

    def strong(self, r, A, u):
        """Strong Dirichlet rows: residual u - g, identity row (a no-op in the weak mode)."""
        if self.bc != 'strong':
            return r, A
        g = np.array([self.case.bottom, self.case.top])
        r = r.copy(); r[self.fixed] = u[self.fixed] - g
        A = A.tolil()
        for i in self.fixed:
            A.rows[i] = [i]; A.data[i] = [1.0]
        return r, A.tocsr()


def newton(col, uold, t, dt, tol, norm=2, maxit=60):
    """Newton with a backtracking line search; stops when the residual's norm (2: proteus's rule, or 'inf') is below tol."""
    nrm = (lambda v: np.linalg.norm(v)) if norm == 2 else (lambda v: np.abs(v).max())
    u = uold.copy()
    if col.bc == 'strong':                                       # Dirichlet values set before the iteration, as proteus does
        u[col.fixed] = [col.case.bottom, col.case.top]
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


def read_reference(case):
    with open(os.path.join(COMPARISON_DIR, case.csv)) as f:
        rows = list(csv.DictReader(f))
    x = np.array([float(r['x']) for r in rows])
    return x, {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != 'x'}


def run(col, steps, tol, norm, frames_at=None):
    case = col.case
    u = col.case.initial(col.x).astype(float)
    dt = case.t_end / steps
    frames_at = frames_at or (steps // 2, steps)
    t0 = time.perf_counter(); frames = {}; total_its = 0
    for step in range(1, steps + 1):
        uold = u
        u, its = newton(col, u, step * dt, dt, tol, norm); total_its += its
        if col.fct:
            u, _, _ = col.fct_correct(u, uold, step * dt, dt)
        if step in frames_at: frames[step] = u.copy()
    return frames, total_its, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='celia', choices=sorted(CASES))
    ap.add_argument('--bc', default='weak', choices=['weak', 'strong'])
    ap.add_argument('--stabilize', action='store_true', help='lumped mass + discrete upwinding (lowOrder.h)')
    ap.add_argument('--upwind', action='store_true', help="proteus's low-order operator: K at the upwind node on the total head, lumped mass, strong Dirichlet (--bc strong)")
    ap.add_argument('--proteus-limiter', action='store_true', help="with --upwind --fct: take the limiter's local bounds over the low-order moisture content alone, as proteus's FCTStep does")
    ap.add_argument('--fct', action='store_true', help='stabilize, then the FCT limiter on the moisture content')
    ap.add_argument('--nn', type=int, default=None)
    ap.add_argument('--steps', type=int, default=None)
    a = ap.parse_args()
    case = CASES[a.case]
    nn = a.nn or case.nn; steps = a.steps or case.steps
    col = Column(nn, a.bc, a.stabilize, a.fct, case, a.upwind, a.proteus_limiter)
    dt = case.t_end / steps
    ref = None
    if nn == case.nn and steps == case.steps:
        x, ref = read_reference(case)
        assert np.allclose(x, col.x)
    frames_at = tuple(sorted(set((steps * f) // case.steps for f in case.frames)))
    scheme = ('K-upwind ' if a.upwind else '') + ('FCT' if a.fct else ('lumped mass + upwinding' if a.stabilize else ('low-order' if a.upwind else 'Galerkin')))
    for label, tol, norm in (("proteus's rule (2-norm < 1e-8)", 1e-8, 2), ("converged (max-norm < 1e-12)", 1e-12, 'inf')):
        frames, its, wall = run(col, steps, tol, norm, frames_at)
        print("%s, %s Dirichlet, Newton stopped at %s: %d steps of %.4g d, %d iterations (%.2f per step), %.1f s with %s" %
              (case.name, a.bc, label, steps, dt, its, its / steps, wall, engine.__name__))
        if ref is not None:
            for frame, f0 in zip(frames_at, case.frames):
                u = frames[frame]; tag = 't%d' % f0
                if not (a.stabilize or a.fct or a.upwind) and ('stab_0_' + tag) in ref:
                    d = np.abs(u - ref['stab_0_' + tag])
                    print("    t = %.3f d: against proteus (stab_0_%s) max |difference| = %.3e at x = %.3f" % (frame * dt, tag, d.max(), col.x[d.argmax()]))
                else:                                           # a different scheme from every saved column: compare, and check the bounds
                    diffs = "  ".join("%s %.2e" % (k, np.abs(u - ref[k + '_' + tag]).max()) for k in ('stab_0', 'stab_2', 'FCT') if (k + '_' + tag) in ref)
                    print("    t = %.3f d: psi in [%.4f, %.4f] (data: [%.2f, %.2f]); %s against proteus's (max |diff|)  %s" %
                          (frame * dt, u.min(), u.max(), min(case.bottom, case.top), max(case.bottom, case.top), scheme, diffs))
    if ref is None:
        print("  (the saved profiles are for nn = %d and %d steps)" % (case.nn, case.steps))


if __name__ == '__main__':
    main()
