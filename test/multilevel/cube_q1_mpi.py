"""Helper run under mpiexec by test_parallel_cube_meshes.py; not collected.

Solves a Poisson problem with Q1 on a quadrilateral or hexahedral mesh and
prints a checksum of the globally assembled solution, so ranks can be compared.
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
                     NumericalSolution, LinearSolvers)

rank, size = comm.rank(), comm.size()
mpi = comm.comm.tompi4py()
testroot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kind = sys.argv[1]
nn = int(sys.argv[2])
if kind == 'hex':
    TESTDIR = os.path.join(testroot, "ci")
    p = defaults.load_physics('poisson_3d_p', TESTDIR)
    n = defaults.load_numerics('poisson_3d_c0q1_n', TESTDIR)
    n.nnx = n.nny = n.nnz = nn
else:
    TESTDIR = os.path.join(testroot, "poisson_2d")
    p = defaults.load_physics('poisson_het_2d_p', TESTDIR)
    n = defaults.load_numerics('poisson_het_2d_c0pk_n', TESTDIR)
    p.x0 = (0., 0., 0.)
    p.L = (1., 1., 1.)
    p.meshfile = None
    n.quad = True
    n.nn = nn
    n.femSpaces[0] = default_n.C0_AffineLinearOnCubeWithNodalBasis
    n.elementQuadrature = default_n.CubeGaussQuadrature(p.nd, 3)
    n.elementBoundaryQuadrature = default_n.CubeGaussQuadrature(p.nd - 1, 3)
n.nLevels = 1
if size > 1:
    n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
    n.levelLinearSolver = LinearSolvers.KSP_petsc4py
    # an exact solve, so the comparison is about the mesh and not the tolerance
    n.petscOptions = {'ksp_type': 'preonly', 'pc_type': 'redundant', 'redundant_pc_type': 'lu'}
else:
    n.multilevelLinearSolver = LinearSolvers.LU
    n.levelLinearSolver = LinearSolvers.LU
n.linearSmoother = None
n.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.node
n.nLayersOfOverlapForParallel = 1
so = default_so
so.name = p.name = "cubeq1_%s_%d_%d" % (kind, nn, size)
so.sList = [default_s]
runopts.logLevel = 1
runopts.verbose = False
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runopts)
ns.calculateSolution("cubeq1")
lm = ns.modelList[0].levelModelList[-1]
dm = lm.u[0].femSpace.dofMap
N = dm.nDOF_all_processes
g = np.zeros(N)
s2g = np.asarray(dm.subdomain2global)
off = np.asarray(dm.dof_offsets_subdomain_owned)
for i in range(int(off[rank + 1] - off[rank])):
    g[int(s2g[i])] = lm.u[0].dof[i]
G = np.zeros(N)
mpi.Allreduce(g, G, op=MPI.SUM)
if rank == 0:
    print("RESULT kind=%s n=%d ranks=%d N=%d checksum=%.12e" % (kind, nn, size, N, G.sum()))
    sys.stdout.flush()
