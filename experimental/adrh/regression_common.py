"""What the ADRH regressions share: the dimension, space and boundary mode from the environment, the quadrature
(the same rules the Proteus model uses), running the Proteus model, picking boundary faces, assembling the
engine's element and face contributions, imposing strong Dirichlet rows, and the norms.

Environment: AD_ND (2 or 3), AD_SPACE (p1, p2, q1), AD_BC (strong, mixed, inflow, robin, iipg, nipg, sipg). The
driver that imports this re-exports them so the Proteus model files (ad_*_p.py, ad_*_n.py) see the same choices.
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
SPACE = os.environ.get('AD_SPACE', 'p1')
BC = os.environ.get('AD_BC', 'mixed' if os.environ.get('AD_MIXED', '0') == '1' else 'strong')
os.environ['AD_ND'], os.environ['AD_SPACE'], os.environ['AD_BC'] = str(ND), SPACE, BC
nD, nV, FACE_VERTICES = adrh_proto.space_info(ND, SPACE)          # dofs and vertices per element; each face's vertices

ORDER = 4
os.environ.setdefault('AD_QORDER', str(ORDER))
Gauss = Quadrature.CubeGaussQuadrature if SPACE == 'q1' else Quadrature.SimplexGaussQuadrature   # as the _n file
_quad = Gauss(ND, ORDER)
qp = np.array([list(p)[:ND] for p in _quad.points]); qw = np.array(_quad.weights)
_quadf = Gauss(ND - 1, ORDER)
qpf = np.array([list(p)[:ND - 1] for p in _quadf.points]); qwf = np.array(_quadf.weights)
W_BASIS = adrh_proto.basis_values(ND, SPACE, qp)[0]                  # (nQ, nD) the space's basis at the quadrature points

WEAK = {'iipg': (2.0, 1.0, 0.0), 'nipg': (10.0, 1.0, -1.0), 'sipg': (100.0, 1.0, 1.0)}   # C, p, sigma as in Proteus
assert BC in ('strong', 'mixed', 'inflow', 'robin') or BC in WEAK, BC
ROBIN_ALPHA, ROBIN_BETA = 1.0, 2.0                                 # as the _p files
# the faces the engine's boundary kernel sees, by plane: (kind code of the BC functor, axis, value); the rest is strong
FACES = {'strong': [], 'mixed': [(1, 0, 1.0)], 'inflow': [(2, 0, 0.0), (1, 0, 1.0)], 'robin': [(4, 0, 1.0)]}
for _mode in WEAK:
    FACES[_mode] = [(3, ax, v) for ax in range(ND) for v in (0.0, 1.0)]
WEAK_PARAMS = WEAK.get(BC, (2.0, 1.0, 0.0))
_DEFAULT_TNLIST = default_so.tnList                                   # None: a steady run


def run_proteus(pname, nname, name, nn):
    """Run the Proteus model (pname, nname beside this file) on the nn^ND regular mesh: the final dof vector, the
    coordinates of every global dof (vertices, then P2 edge midpoints), the element-to-global dof map (nE, nD)
    whose first nV columns are the element's vertices."""
    os.environ['AD_NN'] = str(nn)
    for m in (pname, nname):
        sys.modules.pop(m, None)
    p = load_physics(pname, HERE)
    n = load_numerics(nname, HERE)
    so = default_so; so.name = p.name = '%s%d' % (name, nn); so.sList = [default_s]
    so.tnList = getattr(n, 'tnList', _DEFAULT_TNLIST)
    runopts.logLevel = 1; runopts.verbose = False
    ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runopts)
    ns.calculateSolution(name)
    lm = ns.modelList[0].levelModelList[-1]
    dofMap = lm.u[0].femSpace.dofMap
    l2g = np.array(dofMap.l2g)
    dofx = np.array(dofMap.lagrangeNodesArray) if hasattr(dofMap, 'lagrangeNodesArray') else np.array(lm.mesh.nodeArray)
    assert np.array_equal(l2g[:, :nV], lm.mesh.elementNodesArray)
    # the regressions place their boundary data on the unit domain; proteus reads p.x0 and p.L for its regular mesh
    assert abs(dofx[:, :ND].min()) < 1e-12 and abs(dofx[:, :ND].max() - 1.0) < 1e-12, "the mesh is not the unit domain"
    return np.array(lm.u[0].dof), dofx, l2g


def plane_faces(nodes, l2g, axis, value):
    """Element faces (element, local face index of the space) whose vertices all lie on the plane x[axis] = value.
    Per plane, because a face with its vertices on two different planes is an interior face."""
    on = np.abs(nodes[l2g[:, :nV]][:, :, axis] - value) < 1e-12                # (nE, nV)
    on_face = np.stack([on[:, fv].all(axis=1) for fv in FACE_VERTICES], axis=1)  # (nE, nFaces)
    return np.nonzero(on_face)


def boundary_faces(nodes, l2g):
    """(elements, local faces, kind codes) of the faces the boundary kernel gets in this mode, or None."""
    if not FACES[BC]:
        return None
    fe, fk, kind = [], [], []
    for code, ax, v in FACES[BC]:
        e, k = plane_faces(nodes, l2g, ax, v)
        fe.append(e); fk.append(k); kind.append(np.full(len(e), code))
    fe, fk, kind = (np.concatenate(a) for a in (fe, fk, kind))
    return fe, np.ascontiguousarray(fk, dtype='i'), np.ascontiguousarray(kind, dtype='i')


def dirichlet_dofs(nodes):
    """The dofs whose rows the driver replaces by the Dirichlet value: every Lagrange node getDBC accepts."""
    X = nodes[:, :ND]
    lo, hi = X.min(axis=1) < 1e-12, X.max(axis=1) > 1 - 1e-12
    if BC in WEAK:
        return np.zeros(len(nodes), bool)
    if BC in ('mixed', 'robin'):
        return lo | (X[:, 1:].max(axis=1) > 1 - 1e-12)
    if BC == 'inflow':
        return (X[:, 1:].min(axis=1) < 1e-12) | (X[:, 1:].max(axis=1) > 1 - 1e-12)
    return lo | hi


def assemble(conn, R, J, nN):
    """Global matrix from J (nE, 1, nD, 1, nD) and right-hand side -R from R (nE, 1, nD), scattered by conn (nE, nD)."""
    rows = np.repeat(conn, nD, axis=1).ravel(); cols = np.tile(conn, (1, nD)).ravel()
    K = sp.coo_matrix((J[:, 0, :, 0, :].ravel(), (rows, cols)), shape=(nN, nN)).tocsr()
    rhs = -np.bincount(conn.ravel(), weights=R[:, 0, :].ravel(), minlength=nN)
    return K, rhs


def solve_with_dirichlet(K, rhs, onb, ub):
    """Solve K u = rhs with u = ub on the dofs onb (the linear problem: one solve)."""
    free = ~onb
    rhs_f = rhs[free] - K[free][:, onb] @ ub[onb]
    u = ub.copy()
    u[free] = spl.spsolve(K[free][:, free].tocsc(), rhs_f)
    return u


def l2_error(u, conn, xn, exact_at):
    """L2 error by the engine's quadrature and geometry, u_h interpolated by the space's basis at the points;
    exact_at(X) evaluates the exact solution on points X (..., 3)."""
    uh = u[conn] @ W_BASIS.T                                          # (nE, nQ)
    xq, adet = adrh_proto.quadrature_geometry(ND, SPACE, xn, qp)      # (nE, nQ, 3), (nE, nQ)
    return np.sqrt(np.sum(adet * qw[None, :] * (uh - exact_at(xq)) ** 2))
