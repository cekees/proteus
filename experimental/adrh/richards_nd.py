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

Compressibility (beta != 0) is not in this scheme: the total-head form of the flux needs rho / rho0 = 1.
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

    def __init__(self, x, conn, material, alpha, n_vg, theta_r, theta_s, ks, g, dirichlet, psk=0, interface='node'):
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
        self.fixed = np.asarray(dirichlet[0], dtype='i'); self.gD = np.asarray(dirichlet[1], dtype=float)
        self.nodeMat = first_incident_material(self.conn, self.material, self.nn)
        self.xn = np.zeros((self.nE, self.nV, 3)); self.xn[:, :, :self.nd] = self.x[self.conn]
        quad = Quadrature.SimplexGaussQuadrature(self.nd, 3)
        self.qp = np.array([list(p)[:self.nd] for p in quad.points]); self.qw = np.array(quad.weights)
        rows = np.repeat(self.conn, self.nV, axis=1).ravel(); cols = np.tile(self.conn, (1, self.nV)).ravel()
        self._rows, self._cols = rows, cols
        P = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(self.nn, self.nn)).tocsr(); P.sort_indices()
        self.indptr, self.indices = P.indptr.astype('i'), P.indices.astype('i')
        self.tpos = engine.transpose_map(self.indptr, self.indices)
        self.prows = np.repeat(np.arange(self.nn), np.diff(self.indptr)); self.pcols = self.indices
        self.S = self.stiffness(np.zeros(self.nn), 1).data.copy()
        self.present = np.unique(self.material)
        self.S_by = {int(k): self.stiffness(np.zeros(self.nn), 1, only=k).data.copy() for k in self.present} if interface == 'element' else None
        w = self._lumped(np.zeros(self.nn), np.zeros(self.nn), 1.0)[0]
        self.ML = np.zeros(self.nn); np.add.at(self.ML, self.conn.ravel(), w.ravel())
        Mel = engine.consistent_mass(self.nd, "p1", self.xn, self.qp, self.qw)[0]
        self.Mc = self.pattern_matrix(Mel)

    # -- the element kernels, on the fixed CSR pattern ------------------------------------------------------------------------------------------
    def _args(self):
        return (self.alpha, self.n_vg, self.theta_r, self.thetaSR, self.ks, self.g, 1.0, 0.0, self.material)

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
        _, J = engine.richards_vgm(self.nd, "p1", *self._args()[:8], self.material, self.xn, ue, ue, self.qp, self.qw, 0.0, 0.0, self.psk, mode)
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

    def flux(self, u):
        """The flux part of the residual and the values of its Jacobian on the pattern."""
        phi = self.total_head(u)
        if self.interface == 'node':
            kr, dkr = self.kr(u)
            return engine.upwind_flux(self.indptr, self.indices, self.S, phi, kr, dkr)
        r = np.zeros(self.nn); Jv = np.zeros(self.indices.size)
        for k in self.present:
            kr, dkr = self.kr(u, int(k))
            rk, Jk = engine.upwind_flux(self.indptr, self.indices, self.S_by[int(k)], phi, kr, dkr)
            r += rk; Jv += Jk
        return r, Jv
    def m(self, u):
        return engine.richards_m_nodes(self.alpha, self.n_vg, self.theta_r, self.thetaSR, self.psk, self.nodeMat, u)

    def mass(self, u):
        """The conserved mass, sum_i M_L,i m(u_i): FCT keeps it exactly."""
        return float(self.ML @ self.m(u)[0])

    # -- the nonlinear system -------------------------------------------------------------------------------------------------------------------
    def system(self, u, uold, dt):
        """Residual and Jacobian of one backward Euler step of the K-upwinded low-order scheme (dt None: the steady problem), strong Dirichlet rows."""
        r, Jv = self.flux(u)
        A = sp.csr_matrix((Jv, self.indices, self.indptr), shape=(self.nn, self.nn))
        if dt is not None:
            m, dm = self.m(u); mo, _ = self.m(uold)
            r = r + self.ML * (m - mo) / dt
            A = A + sp.diags(self.ML * dm / dt)
        mask = np.ones(self.nn); mask[self.fixed] = 0.0
        r = r * mask; r[self.fixed] = u[self.fixed] - self.gD
        A = (sp.diags(mask) @ A + sp.diags(1.0 - mask)).tocsc()
        return r, A

    def newton(self, u0, uold, dt, tol=1e-12, norm='inf', maxit=60):
        """Newton with a backtracking line search on the residual norm (2: proteus's stopping rule; 'inf')."""
        nrm = (lambda v: np.linalg.norm(v)) if norm == 2 else (lambda v: np.abs(v).max())
        u = np.array(u0, dtype=float); u[self.fixed] = self.gD
        r, A = self.system(u, uold, dt); its = 0
        while nrm(r) > tol and its < maxit:
            du = spl.spsolve(A, -r)
            lam, r0 = 1.0, nrm(r)
            while True:
                un = u + lam * du
                rn, An = self.system(un, uold, dt)
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
        pressure keeps the low-order head when the limited moisture content is saturated)."""
        s = self.nodeMat
        a, n = self.alpha[s], self.n_vg[s]
        se = (theta - self.theta_r[s]) / (self.theta_s[s] - self.theta_r[s])
        if self.psk == 0:
            se = np.clip(se, 1e-12, 1.0 - 1e-15)
            mvg = 1.0 - 1.0 / n
            return -((se ** (-1.0 / mvg) - 1.0) ** (1.0 / n)) / a
        wet = se >= 1.0
        return np.where(wet, np.maximum(uL, -1.0 / a), -np.clip(se, 1e-300, 1.0) ** (-1.0 / n) / a)

    def fct_correct(self, uL, uold, dt, bounds_old=True, bounds='head'):
        """FCT on the moisture content, the antidiffusive flux being the Galerkin flux (K at the quadrature points) minus the upwinded one. Returns the
        corrected head and moisture content.

        bounds: 'head' bounds the head over a node and its neighbours (the low-order and, with bounds_old, the old state) and maps the bounds to moisture
        content with the node's own retention curve; 'moisture' bounds the moisture content itself, as proteus does. They are the same for one
        material. They differ across a soil interface, where the moisture contents of two soils at one head are not comparable: the moisture bound lets
        a node dry below every head around it (psi = -11 in a lens that started at -6 to -10, richards_2d.py --lens --fct)."""
        phi = self.total_head(uL)
        if self.interface == 'node':
            kr, _ = self.kr(uL)
            A = engine.antidiffusive_flux(self.indptr, self.indices, self.stiffness(uL, 2).data.copy(), self.S, phi, kr)
        else:                                                   # linear in the stiffness and in the upwinded term: the sum over materials
            A = np.zeros(self.indices.size)
            for k in self.present:
                kr, _ = self.kr(uL, int(k))
                A += engine.antidiffusive_flux(self.indptr, self.indices, self.stiffness(uL, 2, only=k).data.copy(), self.S_by[int(k)], phi, kr)
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

    def run(self, u0, t_end, steps, fct=False, tol=1e-12, norm='inf', bounds='head'):
        """steps backward Euler steps of dt = t_end / steps from u0; returns the final head and the Newton iteration count."""
        u = np.array(u0, dtype=float); dt = t_end / steps; its = 0
        for _ in range(steps):
            uold = u
            u, k = self.newton(u, uold, dt, tol, norm); its += k
            if fct:
                u, _ = self.fct_correct(u, uold, dt, bounds=bounds)
        return u, its
