"""Check the prototype engine against proteus's Python path, and time both.

1. tet P1 Poisson (test/ci/poisson_3d_p + c0p1): element residual and Jacobian of the engine
   equal OneLevelTransport's elementResidual / elementJacobian.
2. a two-component nonlinear system with no Python reference: the element Jacobian is the
   derivative of the element residual (central differences), including the mass term and its
   old-time evaluation.
usage: python compare_with_python.py [nn]
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib
adrh_proto = importlib.import_module(os.environ.get("ADRH_MODULE", "adrh_proto"))   # or adrh_kokkos

from proteus import Comm
Comm.init()
from proteus import default_s, default_so, NumericalSolution
from proteus.defaults import load_numerics, load_physics
from proteus.iproteus import opts as runopts

nn = int(sys.argv[1]) if len(sys.argv) > 1 else 15
TESTCI = os.path.join(HERE, '..', '..', 'test', 'ci')
p = load_physics('poisson_3d_p', TESTCI)
n = load_numerics('poisson_3d_c0p1_n', TESTCI)
n.nnx = n.nny = n.nnz = nn
n.nLevels = 1
so = default_so
so.name = p.name = 'adrhproto'
so.sList = [default_s]
runopts.logLevel = 1
runopts.verbose = False
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runopts)
lm = ns.modelList[0].levelModelList[-1]
mesh = lm.mesh
nE = mesh.nElements_global
xn = np.ascontiguousarray(mesh.nodeArray[mesh.elementNodesArray])           # (nE, 4, 3)
l2g = lm.u[0].femSpace.dofMap.l2g
qp = np.array([list(pt)[:3] for pt in n.elementQuadrature.points])
qw = np.array(n.elementQuadrature.weights)

# --- 1. against the Python path ---------------------------------------------------------
rng = np.random.default_rng(1)
ufree = rng.uniform(-1, 1, lm.dim)
r = np.zeros(lm.dim)
lm.getResidual(ufree, r)             # sets the dof (free unknowns + Dirichlet values) and fills elementResidual
uel = np.ascontiguousarray(lm.u[0].dof[l2g])[:, None, :]                  # (nE, 1, 4)
tpy = time.time()
lm.calculateElementResidual()
lm.calculateElementJacobian()
tpy = time.time() - tpy
t0 = time.time()
R, J = adrh_proto.poisson3d(xn, uel, qp, qw)
teng = time.time() - t0
Rp, Jp = lm.elementResidual[0], lm.elementJacobian[0][0]
# OneLevelTransport adds the exterior boundary-flux terms into elementResidual, and the
# prototype is the interior (Galerkin) kernel only, so compare on elements with no exterior face
exterior = np.zeros(mesh.nElementBoundaries_global, dtype=bool)
exterior[mesh.exteriorElementBoundariesArray] = True
interior = ~exterior[mesh.elementBoundariesArray].any(axis=1)
errR = np.abs(R[interior, 0, :] - Rp[interior]).max() / np.abs(Rp[interior]).max()
errJ = np.abs(J[:, 0, :, 0, :] - Jp).max() / np.abs(Jp).max()
print("tet P1 Poisson, %d elements (%d without an exterior face): relative max difference  residual %.2e  jacobian %.2e"
      % (nE, interior.sum(), errR, errJ))
print("  Python path (element residual + Jacobian; quadrature arrays already filled) %.4f s; engine (everything) %.4f s"
      % (tpy, teng))

# --- 2. finite-difference Jacobian check on the coupled system ---------------------------
u = rng.uniform(0.2, 1.0, (nE, 2, 4))
uo = rng.uniform(0.2, 1.0, (nE, 2, 4))
t, dt = 0.7, 0.05
R0, J0 = adrh_proto.coupled2(xn, u, uo, qp, qw, t, dt)
d = rng.uniform(-1, 1, u.shape)
eps = 1e-6
Rp_ = adrh_proto.coupled2(xn, u + eps * d, uo, qp, qw, t, dt)[0]
Rm_ = adrh_proto.coupled2(xn, u - eps * d, uo, qp, qw, t, dt)[0]
fd = (Rp_ - Rm_) / (2 * eps)
jd = np.einsum('eiajb,ejb->eia', J0, d)
print("coupled 2-component system: Jacobian vs central differences, relative max difference %.2e"
      % (np.abs(fd - jd).max() / np.abs(jd).max()))
R2 = adrh_proto.coupled2(xn, u, None, qp, qw, t, 0.0)[0]
print("  steady (no mass term) residual finite: %s" % np.isfinite(R2).all())
assert errR < 1e-12 and errJ < 1e-12
assert np.abs(fd - jd).max() / np.abs(jd).max() < 1e-7
print("OK")
