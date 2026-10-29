"""Richards' equation in 1, 2 or 3 dimensions on a P1 simplex mesh, with the K-upwinded low-order scheme and its FCT correction, and heterogeneous media.

The 1D column of richards_regression.py (Column) is pinned to proteus's saved profiles and is tridiagonal by construction; this is the same scheme
on a general connectivity, built from the engine's passes (richards.h, lowOrder.h): the element kernel in mode 1 (the stiffness K_s alone, each
element with its own material) and mode 2 (the stiffness with the relative permeability frozen at the quadrature points), the nodal passes
richards_kr_nodes / richards_m_nodes, upwind_flux, antidiffusive_flux and fct_limit_flux. The Newton iteration, the line search and the strong
Dirichlet rows are those of Column.

Heterogeneous media. K_s is the element's (the element kernel's material table), as in proteus. The storage m(psi) and the upwind relative permeability
are NODAL (the lumped mass and the discrete upwinding are nodal), so a node has one retention curve: the first incident element's, which is
proteus's rule too (Richards.py, nodeMaterialTypes_n: "the nodal scheme allows it only one theta(psi)"). For a node on a soil interface that is one of the
two soils, and the upwind K of an edge in the other soil's element is that soil's K_s times the first soil's K_r; richards_nd_test.py measures
what that costs against an ODE solution.

interface='element' (the default is proteus's, interface='node') keeps the nodal storage but takes the upwind relative permeability of an element's
flux from that element's own material: the flux is split by material, F = sum_m kr_m(psi_up) c_m (phi_j - phi_i), c_m the upwinding weight of the
stiffness of the elements of material m alone, so an interface node gives each soil its own K_r(psi) (psi is continuous across the interface). One
material reduces to the same scheme as interface='node'.

Compressibility: rho = rho_t exp(beta psi) (rho_t, beta in the constructor; rho0 = 1, the engine's bindings). The gravity term is then not the gradient of a nodal head, so the
flux of an edge uses the head difference dphi = u_j - u_i - rhoEdge g.(x_j - x_i) with rhoEdge the mean of rho over the edge, and the upwinded coefficient is rho K_r at the upwind
node (upwind_flux_edges, antidiffusive_flux_edges); the storage is rho theta; FCT inverts m = rho theta(psi) for the head with a per-node Newton iteration.
"""
import importlib
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
engine = importlib.import_module(os.environ.get('ADRH_MODULE', 'adrh_proto'))
from proteus import Quadrature


def rectangle_triangles(nx, ny, Lx, Ly):
    """A structured triangulation of [0, Lx] x [0, Ly]: nx x ny nodes, node (i, j) = i * ny + j, every cell cut along its (0,0)-(1,1) diagonal into
    two counter-clockwise triangles. Returns (x (nn, 2), conn (nE, 3))."""
    X, Y = np.meshgrid(np.linspace(0.0, Lx, nx), np.linspace(0.0, Ly, ny), indexing='ij')
    x = np.stack([X.ravel(), Y.ravel()], axis=1)
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing='ij')
    a = (i * ny + j).ravel(); b = ((i + 1) * ny + j).ravel(); c = ((i + 1) * ny + j + 1).ravel(); d = (i * ny + j + 1).ravel()
    conn = np.stack([np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)], axis=1).reshape(-1, 3).astype('i')
    return x, conn


def first_incident_material(conn, material, nNodes):
    """The material of the first element (in element order) containing each node: proteus's nodeMaterialTypes_n."""
    nodes, first = np.unique(conn.ravel(), return_index=True)
    out = np.zeros(nNodes, dtype='i')
    out[nodes] = np.asarray(material)[first // conn.shape[1]]
    return out


class KUpwindRichards:
    """alpha, n_vg, theta_r, theta_s: (nMaterials,); ks: (nMaterials, nd) the diagonal conductivities (dimensionless, as proteus's KsTypes); g: (nd,) the
    gravity direction (proteus's dimensionless_gravity); material: (nE,) the material of each element; dirichlet: (node numbers, values)."""

    def __init__(self, x, conn, material, alpha, n_vg, theta_r, theta_s, ks, g, dirichlet, psk=0, interface='node', rho=1.0, beta=0.0, quadrature_order=3):
        self.x = np.asarray(x, dtype=float); self.conn = np.asarray(conn, dtype='i')
        self.nn, self.nd = self.x.shape
        self.nE, self.nV = self.conn.shape
        assert self.nV == self.nd + 1, "P1 simplices"
        self.material = np.asarray(material, dtype='i')
        self.alpha, self.n_vg, self.theta_r = (np.asarray(a, dtype=float) for a in (alpha, n_vg, theta_r))
        self.theta_s = np.asarray(theta_s, dtype=float); self.thetaSR = self.theta_s - self.theta_r
        self.ks = np.asarray(ks, dtype=float).reshape(len(self.alpha), self.nd)
        self.g = np.asarray(g, dtype=float)
        self.psk = psk
        assert interface in ('node', 'element')
        self.interface = interface
        self.rho, self.beta = float(rho), float(beta)          # rho_t and the compressibility: rho = rho_t exp(beta psi), rho0 = 1 (the engine's bindings)
        self.compressible = self.beta != 0.0 or self.rho != 1.0
        self.fixed = np.asarray(dirichlet[0], dtype='i'); self.gD = np.asarray(dirichlet[1], dtype=float)
        self.nodeMat = first_incident_material(self.conn, self.material, self.nn)
        self.xn = np.zeros((self.nE, self.nV, 3)); self.xn[:, :, :self.nd] = self.x[self.conn]
        quad = Quadrature.SimplexGaussQuadrature(self.nd, quadrature_order)
        self.qp = np.array([list(p)[:self.nd] for p in quad.points]); self.qw = np.array(quad.weights)
        rows = np.repeat(self.conn, self.nV, axis=1).ravel(); cols = np.tile(self.conn, (1, self.nV)).ravel()
        self._rows, self._cols = rows, cols
        P = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(self.nn, self.nn)).tocsr(); P.sort_indices()
        self.indptr, self.indices = P.indptr.astype('i'), P.indices.astype('i')
        self.tpos = engine.transpose_map(self.indptr, self.indices)
        self.prows = np.repeat(np.arange(self.nn), np.diff(self.indptr)); self.pcols = self.indices
        self.G = (self.x[self.pcols] - self.x[self.prows]) @ self.g      # g.(x_j - x_i) on the pattern
        self.S = self.stiffness(np.zeros(self.nn), 1).data.copy()
        self.present = np.unique(self.material)
        self.S_by = {int(k): self.stiffness(np.zeros(self.nn), 1, only=k).data.copy() for k in self.present} if interface == 'element' else None
        w = self._lumped(np.zeros(self.nn), np.zeros(self.nn), 1.0)[0]
        self.ML = np.zeros(self.nn); np.add.at(self.ML, self.conn.ravel(), w.ravel())
        Mel = engine.consistent_mass(self.nd, "p1", self.xn, self.qp, self.qw)[0]
        self.Mc = self.pattern_matrix(Mel)

    # -- the element kernels, on the fixed CSR pattern ------------------------------------------------------------------------------------------
    def _args(self, unit=False):
        """The element kernel's arguments; unit: unit density and no compressibility (the stiffness S of the upwinded flux, which carries no rho)."""
        return (self.alpha, self.n_vg, self.theta_r, self.thetaSR, self.ks, self.g, 1.0 if unit else self.rho, 0.0 if unit else self.beta, self.material)

    def pattern_matrix(self, Rloc):
        """Element matrices (nE, nV, nV) summed into a CSR on the pattern (explicit zeros kept, so a conductivity that underflows to 0 in a dry
        element does not drop entries), sorted columns."""
        vals = np.concatenate([np.asarray(Rloc).reshape(-1), np.zeros(self.indices.size)])
        A = sp.coo_matrix((vals, (np.concatenate([self._rows, self.prows]), np.concatenate([self._cols, self.pcols]))),
                          shape=(self.nn, self.nn)).tocsr()
        A.sort_indices()
        assert np.array_equal(A.indptr, self.indptr) and np.array_equal(A.indices, self.indices)
        return A

    def stiffness(self, u, mode, only=None):
        """mode 1: K_s grad phi_i . grad phi_j; mode 2: the same with the relative permeability at the quadrature points of u (frozen). only: the
        elements of that material alone."""
        ue = u[self.conn].reshape(self.nE, 1, self.nV)
        _, J = engine.richards_vgm(self.nd, "p1", *self._args(unit=(mode == 1))[:8], self.material, self.xn, ue, ue, self.qp, self.qw, 0.0, 0.0, self.psk, mode)
        if only is not None:
            J = np.asarray(J).reshape(self.nE, -1) * (self.material == only)[:, None]
        return self.pattern_matrix(J)

    def _lumped(self, u, uold, dt):
        ue = u[self.conn].reshape(self.nE, 1, self.nV); uo = uold[self.conn].reshape(self.nE, 1, self.nV)
        return engine.richards_vgm_lumped(self.nd, "p1", *self._args(), self.xn, ue, uo, self.qp, self.qw, 0.0, dt, self.psk, 0)

    # -- the nodal passes ---------------------------------------------------------------------------------------------------------------------------
    def total_head(self, u):
        return u - self.x @ self.g                              # phi = psi - g.x

    def kr(self, u, material=None):
        """The nodal relative permeability and its derivative: with each node's material, or (material) with that material at every node."""
        nodeMat = self.nodeMat if material is None else np.full(self.nn, material, dtype='i')
        return engine.richards_kr_nodes(self.alpha, self.n_vg, self.theta_r, self.thetaSR, self.psk, nodeMat, u)

    def kr_eff(self, u, material=None):
        """The upwinded nodal coefficient: K_r, or for a compressible fluid rho K_r with rho = rho_t exp(beta u) at the node (and its derivative in u)."""
        kr, dkr = self.kr(u, material)
        if not self.compressible:
            return kr, dkr
        e = self.rho * np.exp(self.beta * u)
        return e * kr, e * (self.beta * kr + dkr)

    def edge_heads(self, u):
        """The edge head difference of the flux on the pattern and its partials in u_i and u_j. The gravity term of a compressible fluid is rho (rho / rho0) K_r g,
        not the gradient of a nodal head, so the edge difference is dphi = u_j - u_i - rhoEdge g.(x_j - x_i) with rhoEdge = (rho_i + rho_j) / 2 over rho0 = 1
        (proteus's edge loop does the same for its salinity density); with rho = 1 and beta = 0 it is phi_j - phi_i."""
        rhom = self.rho * np.exp(self.beta * u)
        rhohat = 0.5 * (rhom[self.prows] + rhom[self.pcols])
        return (u[self.pcols] - u[self.prows] - rhohat * self.G, -1.0 - 0.5 * self.beta * rhom[self.prows] * self.G,
                1.0 - 0.5 * self.beta * rhom[self.pcols] * self.G)

    def flux(self, u):
        """The flux part of the residual and the values of its Jacobian on the pattern."""
        if self.compressible:
            dphi, ddi, ddj = self.edge_heads(u)
            def one(S, material):
                kr, dkr = self.kr_eff(u, material)
                return engine.upwind_flux_edges(self.indptr, self.indices, S, dphi, ddi, ddj, kr, dkr)
        else:
            phi = self.total_head(u)
            def one(S, material):
                kr, dkr = self.kr(u, material)
                return engine.upwind_flux(self.indptr, self.indices, S, phi, kr, dkr)
        if self.interface == 'node':
            return one(self.S, None)
        r = np.zeros(self.nn); Jv = np.zeros(self.indices.size)
        for k in self.present:
            rk, Jk = one(self.S_by[int(k)], int(k))
            r += rk; Jv += Jk
        return r, Jv

    def m(self, u):
        return engine.richards_m_nodes(self.alpha, self.n_vg, self.theta_r, self.thetaSR, self.psk, self.nodeMat, u, self.rho, self.beta)

    def mass(self, u):
        """The conserved mass, sum_i M_L,i m(u_i): FCT keeps it exactly."""
        return float(self.ML @ self.m(u)[0])

    # -- the nonlinear system -------------------------------------------------------------------------------------------------------------------
    def _strong(self, r, A, u):
        mask = np.ones(self.nn); mask[self.fixed] = 0.0
        r = r * mask; r[self.fixed] = u[self.fixed] - self.gD
        return r, (sp.diags(mask) @ A + sp.diags(1.0 - mask)).tocsc()

    def system(self, u, uold, dt, datum=None):
        """Residual and Jacobian of one backward Euler step of the K-upwinded low-order scheme (dt None: the steady problem), strong Dirichlet rows.
        datum: the moisture content the mass term differences against, in place of m(uold) (implicit FCT's limited datum)."""
        r, Jv = self.flux(u)
        A = sp.csr_matrix((Jv, self.indices, self.indptr), shape=(self.nn, self.nn))
        if dt is not None:
            m, dm = self.m(u)
            mo = self.m(uold)[0] if datum is None else datum
            r = r + self.ML * (m - mo) / dt
            A = A + sp.diags(self.ML * dm / dt)
        return self._strong(r, A, u)

    def galerkin_system(self, u, uold, dt):
        """Residual and Jacobian of one backward Euler step of the Galerkin discretization (K at the quadrature points, the consistent mass matrix), strong
        Dirichlet rows: implicit FCT's high-order predictor."""
        ue = u[self.conn].reshape(self.nE, 1, self.nV); uo = uold[self.conn].reshape(self.nE, 1, self.nV)
        R, J = engine.richards_vgm(self.nd, "p1", *self._args()[:8], self.material, self.xn, ue, uo, self.qp, self.qw, 0.0, dt, self.psk, 0)
        r = np.zeros(self.nn); np.add.at(r, self.conn.ravel(), np.asarray(R).reshape(self.nE, self.nV).ravel())
        return self._strong(r, self.pattern_matrix(J), u)

    def newton(self, u0, uold, dt, tol=1e-12, norm='inf', maxit=60, datum=None, system=None):
        """Newton with a backtracking line search on the residual norm (2: proteus's stopping rule; 'inf'). system(u) -> (r, A) replaces the low-order
        system (the Galerkin predictor); datum is passed to the low-order system."""
        nrm = (lambda v: np.linalg.norm(v)) if norm == 2 else (lambda v: np.abs(v).max())
        system = system or (lambda v: self.system(v, uold, dt, datum))
        u = np.array(u0, dtype=float); u[self.fixed] = self.gD
        r, A = system(u); its = 0
        while nrm(r) > tol and its < maxit:
            du = spl.spsolve(A, -r)
            lam, r0 = 1.0, nrm(r)
            while True:
                un = u + lam * du
                rn, An = system(un)
                if nrm(rn) < (1.0 - 1e-4 * lam) * r0 or lam < 1e-4:
                    break
                lam *= 0.5
            u, r, A = un, rn, An
            its += 1
        if nrm(r) > tol:
            raise RuntimeError("Newton did not converge: |r| = %.2e after %d iterations" % (nrm(r), its))
        return u, its

    # -- FCT -----------------------------------------------------------------------------------------------------------------------------------------
    def invert(self, theta, uL):
        """The pressure head of a moisture content with each node's retention curve (the analytic inverse; Brooks-Corey's flat part inside the entry
        pressure keeps the low-order head when the limited moisture content is saturated). For a compressible fluid m = rho_t exp(beta u) theta(u): the analytic
        inverse of theta = m / rho_t starts a per-node Newton iteration on m(u) = m."""
        if self.compressible:
            if self.beta == 0.0:                                # m = rho theta(u): the analytic inverse of theta = m / rho
                return self._invert_incompressible(theta / self.rho, uL)
            return self._invert_compressible(theta, uL)
        return self._invert_incompressible(theta, uL)

    def _invert_compressible(self, target, u0):
        """The head u with m(u) = rho_t exp(beta u) theta(u) = target at every node, beta > 0: m is increasing in u (also across a Brooks-Corey flat part and
        beyond saturation), but it falls below rho_t theta_r as u -> -infinity, which the analytic inverse cannot reach. A bracket grown from u0 by doubling steps,
        then a safeguarded Newton iteration (a bisection step wherever Newton leaves the bracket)."""
        f = lambda u: self.m(u)[0] - target
        lo = np.array(u0, dtype=float); hi = lo.copy(); step = 1.0
        for _ in range(60):
            below, above = f(lo) > 0.0, f(hi) < 0.0
            if not (below.any() or above.any()):
                break
            lo = np.where(below, lo - step, lo); hi = np.where(above, hi + step, hi); step *= 2.0
        u = 0.5 * (lo + hi)
        for _ in range(200):
            mm, dm = self.m(u); fu = mm - target
            lo = np.where(fu < 0.0, u, lo); hi = np.where(fu > 0.0, u, hi)
            newton = u - fu / np.where(dm > 0.0, dm, np.inf)
            un = np.where((newton > lo) & (newton < hi), newton, 0.5 * (lo + hi))
            done = np.abs(un - u) <= 1e-14 * (1.0 + np.abs(u))
            u = un
            if done.all():
                break
        return u

    def _invert_incompressible(self, theta, uL):
        s = self.nodeMat
        a, n = self.alpha[s], self.n_vg[s]
        se = (theta - self.theta_r[s]) / (self.theta_s[s] - self.theta_r[s])
        if self.psk == 0:
            se = np.clip(se, 1e-12, 1.0 - 1e-15)
            mvg = 1.0 - 1.0 / n
            return -((se ** (-1.0 / mvg) - 1.0) ** (1.0 / n)) / a
        wet = se >= 1.0
        return np.where(wet, np.maximum(uL, -1.0 / a), -np.clip(se, 1e-300, 1.0) ** (-1.0 / n) / a)

    def antidiffusive(self, u):
        """The antidiffusive flux at u on the pattern: the Galerkin flux (K at the quadrature points) minus the K-upwinded one, into i from j."""
        if self.compressible:
            dphi = self.edge_heads(u)[0]
            def one(C, S, material):
                return engine.antidiffusive_flux_edges(self.indptr, self.indices, C, S, dphi, self.kr_eff(u, material)[0])
        else:
            phi = self.total_head(u)
            def one(C, S, material):
                return engine.antidiffusive_flux(self.indptr, self.indices, C, S, phi, self.kr(u, material)[0])
        if self.interface == 'node':
            return one(self.stiffness(u, 2).data.copy(), self.S, None)
        A = np.zeros(self.indices.size)                         # linear in the stiffness and in the upwinded term: the sum over materials
        for k in self.present:
            A += one(self.stiffness(u, 2, only=k).data.copy(), self.S_by[int(k)], int(k))
        return A

    def fct_correct(self, uL, uold, dt, bounds_old=True, bounds='head'):
        """FCT on the moisture content, the antidiffusive flux being the Galerkin flux (K at the quadrature points) minus the upwinded one. Returns the
        corrected head and moisture content.

        bounds: 'head' bounds the head over a node and its neighbours (the low-order and, with bounds_old, the old state) and maps the bounds to moisture
        content with the node's own retention curve; 'moisture' bounds the moisture content itself, as proteus does. They are the same for one
        material. They differ across a soil interface, where the moisture contents of two soils at one head are not comparable: the moisture bound lets
        a node dry below every head around it (psi = -11 in a lens that started at -6 to -10, richards_2d.py --lens --fct)."""
        A = self.antidiffusive(uL)
        mL, _ = self.m(uL); mo, _ = self.m(uold)
        if bounds == 'moisture':
            mnew = engine.fct_limit_flux(self.indptr, self.indices, self.tpos, self.Mc.data.copy(), A, self.ML, mL, mo, dt, bounds_old)
        else:
            pmax, pmin = engine.head_extrema(self.indptr, self.indices, uL, uold if bounds_old else uL)
            mmax, _ = self.m(pmax); mmin, _ = self.m(pmin)
            mnew = engine.fct_limit_flux_bounded(self.indptr, self.indices, self.tpos, self.Mc.data.copy(), A, self.ML, mL, mo, mmax, mmin, dt)
        u = self.invert(mnew, uL)
        u[self.fixed] = uL[self.fixed]
        return u, mnew

    # -- implicit FCT (Dielmann, Kuzmin and Kees; proteus's FCT_TYPE='implicit') -------------------------------------------------------------------------
    def implicit_datum(self, pH, uold, dt, prelimiter_power=3.0, prelimiter='pressure', mask=True, bounds='global'):
        """The limited datum s^{n,FCT}: the antidiffusive fluxes at the predictor pH, the nodal-variation prelimiter on the pressure head (or the total head),
        Zalesak with bounds taken from s^n. bounds: 'global' (proteus's and the paper's) the retention curve's own, theta_r to theta_s of each node; 'local' the
        head's extrema over a node and its neighbours at pH and at the old state, mapped to moisture content with the node's own curve (as fct_correct's
        bounds='head'). mask: keep s^n at the Dirichlet nodes, as proteus's bc_mask does (without it the datum conserves sum M_L s exactly)."""
        A = self.antidiffusive(pH)
        sH, _ = self.m(pH); sn, _ = self.m(uold)
        head = pH if prelimiter == 'pressure' else self.total_head(pH)
        alpha = engine.nodal_variation(self.indptr, self.indices, self.Mc.data.copy(), head, prelimiter_power)
        if bounds == 'global':
            smax = np.maximum(self.rho * self.theta_s[self.nodeMat], sn); smin = np.minimum(self.rho * self.theta_r[self.nodeMat], sn)
        else:
            pmax, pmin = engine.head_extrema(self.indptr, self.indices, pH, uold)
            smax = np.maximum(self.m(pmax)[0], sn); smin = np.minimum(self.m(pmin)[0], sn)
        datum = engine.implicit_fct_limit(self.indptr, self.indices, self.tpos, self.Mc.data.copy(), A, self.ML, sH, sn, smax, smin, alpha, dt)
        if mask:
            datum[self.fixed] = sn[self.fixed]
        return datum, alpha

    def implicit_fct_step(self, uold, dt, prelimiter_power=3.0, prelimiter='pressure', tol=1e-12, norm='inf', fallback=True, bounds='global', predictor_maxit=20):
        """One step of the implicit FCT: (1) the Galerkin predictor p^H (the low-order solution instead, if the Galerkin Newton fails and fallback is set,
        as proteus's FCT_PREDICTOR_FALLBACK), (2) the limited datum, (3) the low-order solve M_L (s(p) - s^{n,FCT}) / dt + A~(p) u(p) = 0 started from p^H.
        predictor_maxit caps the Galerkin Newton so that a predictor that will fail does not cost sixty iterations first. Returns the head and a dict of the Newton
        iteration counts and whether the predictor fell back."""
        fell = False
        try:
            pH, k1 = self.newton(uold, uold, dt, tol, norm, maxit=predictor_maxit, system=lambda v: self.galerkin_system(v, uold, dt))
        except RuntimeError:
            if not fallback:
                raise
            pH, k1 = self.newton(uold, uold, dt, tol, norm); fell = True
        datum, _ = self.implicit_datum(pH, uold, dt, prelimiter_power, prelimiter, bounds=bounds)
        u, k2 = self.newton(pH, uold, dt, tol, norm, datum=datum)
        return u, dict(predictor=k1, final=k2, fallback=fell)

    def run(self, u0, t_end, steps, fct=False, tol=1e-12, norm='inf', bounds='head', implicit=False, prelimiter_power=3.0, implicit_bounds='global', bounds_old=True):
        """steps backward Euler steps of dt = t_end / steps from u0; returns the final head and the Newton iteration count (implicit: the implicit FCT
        step, with both solves' iterations counted)."""
        u = np.array(u0, dtype=float); dt = t_end / steps; its = 0
        for _ in range(steps):
            uold = u
            if implicit:
                u, c = self.implicit_fct_step(uold, dt, prelimiter_power, tol=tol, norm=norm, bounds=implicit_bounds); its += c['predictor'] + c['final']
                continue
            u, k = self.newton(u, uold, dt, tol, norm); its += k
            if fct:
                u, _ = self.fct_correct(u, uold, dt, bounds=bounds, bounds_old=bounds_old)
        return u, its
