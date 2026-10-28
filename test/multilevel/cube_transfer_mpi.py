"""Helper run under mpiexec by test_parallel_space_multigrid.py; not collected.

Checks the distributed transfer operators of spaces whose dofs are not mesh
vertices against a function they must reproduce: interpolate f on the coarse
level, prolong, and compare with f interpolated directly on the fine level. For
f in the coarse space the two agree exactly, so any error in the numbering, the
ownership or the ghost refresh shows up as a nonzero difference -- and this does
not depend on the discretization being assembled correctly in parallel.

Usage: cube_transfer_mpi.py KIND LEVELS ORDER   (KIND quad or hex; ORDER 1 or 2)
"""
import mpi_failfast  # noqa: F401  (an uncaught exception aborts every rank)
import os
import sys

import numpy as np
from proteus import Comm
comm = Comm.init()
from mpi4py import MPI
from proteus.iproteus import opts as runopts
from proteus import (MeshTools, defaults, default_n, default_s, default_so,
                     NumericalSolution, LinearSolvers, FemTools)

rank, size = comm.rank(), comm.size()
mpi = comm.comm.tompi4py()
testroot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# a control: partition every level independently, as proteus used to
if os.environ.get('PROTEUS_TEST_INDEPENDENT_PARTITIONS'):
    MeshTools.MultilevelMesh.levelConsistentPartitioning = False
kind = sys.argv[1]
nLevels = int(sys.argv[2])
order = int(sys.argv[3])
degree = order     # the functions below lie in the space: Q1 holds x*y(*z), Q2 more
if kind == 'hex':
    TESTDIR = os.path.join(testroot, "ci")
    p = defaults.load_physics('poisson_3d_p', TESTDIR)
    n = defaults.load_numerics('poisson_3d_c0q1_n', TESTDIR)
    n.nnx = n.nny = n.nnz = 3
    n.linearSolverConvergenceTest = 'r'
else:
    TESTDIR = os.path.join(testroot, "poisson_2d")
    p = defaults.load_physics('poisson_het_2d_p', TESTDIR)
    n = defaults.load_numerics('poisson_het_2d_c0pk_n', TESTDIR)
    p.x0 = (0., 0., 0.)
    p.L = (1., 1., 1.)
    p.meshfile = None
    n.quad = True
    n.nn = 3
    n.femSpaces[0] = default_n.C0_AffineLinearOnCubeWithNodalBasis
    n.elementQuadrature = default_n.CubeGaussQuadrature(p.nd, 3)
    n.elementBoundaryQuadrature = default_n.CubeGaussQuadrature(p.nd - 1, 3)
if order == 2:
    n.femSpaces[0] = default_n.C0_AffineQuadraticOnCubeWithNodalBasis
    n.elementQuadrature = default_n.CubeGaussQuadrature(p.nd, 5)
    n.elementBoundaryQuadrature = default_n.CubeGaussQuadrature(p.nd - 1, 5)
n.nLevels = nLevels
n.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.node
n.nLayersOfOverlapForParallel = 1
n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
n.levelLinearSolver = LinearSolvers.KSP_petsc4py
n.linearSmoother = None
so = default_so
so.name = p.name = "cubetransfer_%s_%d_%d_%d" % (kind, nLevels, order, size)
so.sList = [default_s]
runopts.logLevel = 1
runopts.verbose = False
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runopts)
model = ns.modelList[0]
transfers = model.mlTransport.meshTransfers if hasattr(model, 'mlTransport') else None
if transfers is None:
    transfers = model.meshTransfers if hasattr(model, 'meshTransfers') else None
assert transfers is not None, "cannot find the mesh transfers on the model"


def f(x):
    # in Q1: contains the bilinear and trilinear terms, which every weight of a
    # cell-centre or body-centre node is needed to reproduce; in Q2 also squares
    g = 1.0 + 2.0 * x[..., 0] - 3.0 * x[..., 1] + 0.5 * x[..., 0] * x[..., 1]
    if kind == 'hex':
        g = g + x[..., 2] - 0.7 * x[..., 0] * x[..., 2] + 0.3 * x[..., 0] * x[..., 1] * x[..., 2]
    if order == 2:
        g = g + x[..., 0] ** 2 - 2.0 * x[..., 1] ** 2 + 0.4 * x[..., 0] ** 2 * x[..., 1]
        if kind == 'hex':
            g = g + 0.6 * x[..., 2] ** 2 * x[..., 1]
    return g


def interpolate(levelModel):
    """f at the interpolation points, scattered to the dof array; owned dofs only."""
    fs = levelModel.u[0].femSpace
    pts = fs.updateInterpolationPoints()
    dm = fs.dofMap
    ic = fs.referenceFiniteElement.interpolationConditions
    u = np.zeros(dm.nDOF_subdomain)
    for eN in range(pts.shape[0]):
        for k in range(ic.nQuadraturePoints):
            u[dm.l2g[eN, ic.quadrature2DOF_element(k)]] = f(pts[eN, k])
    return u, dm


worst = 0.0
scale = 0.0
for l in range(nLevels - 1):
    uc, dmc = interpolate(model.levelModelList[l])
    uf, dmf = interpolate(model.levelModelList[l + 1])
    result = np.zeros_like(uf)
    # a dof array's ghost entries are not valid before the operator refreshes them
    transfers.prolong_bcListDict[0][l + 1].matvec(uc, result)
    off = np.asarray(dmf.dof_offsets_subdomain_owned)
    nown = int(off[rank + 1] - off[rank])
    err = np.abs(result[:nown] - uf[:nown]).max() if nown else 0.0
    worst = max(worst, err)
    scale = max(scale, np.abs(uf[:nown]).max() if nown else 0.0)
worst = mpi.allreduce(worst, op=MPI.MAX)
scale = mpi.allreduce(scale, op=MPI.MAX)
if rank == 0:
    print("RESULT kind=%s levels=%d order=%d ranks=%d maxerr=%.3e scale=%.3e"
          % (kind, nLevels, order, size, worst, scale))
    sys.stdout.flush()
