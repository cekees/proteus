"""Helper run under mpiexec by test_cube_multigrid.py; not collected.

Multigrid for Q1 on a regular quadrilateral or hexahedral grid, in the modes
pcmg_parallel_mpi.py offers for simplices: proteus's MGM, PETSc's PCMG over the
same hierarchy, PCMG-preconditioned GMRES, and a direct solve to hold the
converged modes to.

Usage: cube_mg_mpi.py KIND NN LEVELS MODE [ORDER [COARSE]]   (KIND is quad or hex; ORDER 1 or 2, default 1;
MODE ksplu is a direct solve that also runs in parallel; COARSE is redundant, direct or amg)
"""
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
nLevels = int(sys.argv[3])
mode = sys.argv[4]
assert mode in ('mgm', 'pcmg', 'pcmg_its', 'direct', 'ksplu'), mode
Q = int(sys.argv[5]) if len(sys.argv) > 5 else 1
coarse = sys.argv[6] if len(sys.argv) > 6 else 'direct'
if kind == 'hex':
    TESTDIR = os.path.join(testroot, "ci")
    p = defaults.load_physics('poisson_3d_p', TESTDIR)
    n = defaults.load_numerics('poisson_3d_c0q1_n', TESTDIR)
    n.nnx = n.nny = n.nnz = nn
    if Q == 2:
        n.femSpaces[0] = default_n.C0_AffineQuadraticOnCubeWithNodalBasis
        n.elementQuadrature = default_n.CubeGaussQuadrature(p.nd, 5)
        n.elementBoundaryQuadrature = default_n.CubeGaussQuadrature(p.nd - 1, 5)
    n.linearSolverConvergenceTest = 'r'
else:
    TESTDIR = os.path.join(testroot, "poisson_2d")
    p = defaults.load_physics('poisson_het_2d_p', TESTDIR)
    n = defaults.load_numerics('poisson_het_2d_c0pk_n', TESTDIR)
    # the problem's Dirichlet conditions are written for the unit square, so
    # this has to be it; a shifted domain would have no Dirichlet boundary at
    # all and the problem would be singular. The shifted-origin case is covered
    # by the mesh-only tests.
    p.x0 = (0., 0., 0.)
    p.L = (1., 1., 1.)
    p.meshfile = None
    n.quad = True
    n.nn = nn
    n.femSpaces[0] = (default_n.C0_AffineQuadraticOnCubeWithNodalBasis if Q == 2
                      else default_n.C0_AffineLinearOnCubeWithNodalBasis)
    n.elementQuadrature = default_n.CubeGaussQuadrature(p.nd, 3)
    n.elementBoundaryQuadrature = default_n.CubeGaussQuadrature(p.nd - 1, 3)
n.nLevels = nLevels
n.linearCoarseSolver = coarse
n.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.node
n.nLayersOfOverlapForParallel = 1
if mode == 'direct':
    n.multilevelLinearSolver = LinearSolvers.LU
    n.levelLinearSolver = LinearSolvers.LU
    n.linearSmoother = None
elif mode == 'ksplu':
    n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
    n.levelLinearSolver = LinearSolvers.KSP_petsc4py
    n.linearSmoother = None
    n.petscOptions = {'ksp_type': 'preonly', 'pc_type': 'redundant', 'redundant_pc_type': 'lu'}
elif mode == 'mgm':
    n.multilevelLinearSolver = LinearSolvers.MGM
    n.levelLinearSolver = LinearSolvers.MGM
    n.linearSmoother = LinearSolvers.Jacobi
    n.linearWCycles = 1
    n.linearPreSmooths = n.linearPostSmooths = 3
else:
    n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
    n.levelLinearSolver = LinearSolvers.KSP_petsc4py
    n.linearSmoother = LinearSolvers.petsc_MG
    options = {'pc_type': 'mg', 'pc_mg_cycle_type': 'v',
               'mg_levels_ksp_type': 'richardson', 'mg_levels_ksp_richardson_scale': '0.8',
               'mg_levels_ksp_max_it': '3', 'mg_levels_pc_type': 'jacobi'}
    if mode == 'pcmg':
        options['ksp_type'] = 'preonly'
    else:
        options.update({'ksp_type': 'gmres', 'ksp_rtol': '1e-14', 'ksp_atol': '1e-14'})
    n.petscOptions = options
so = default_so
so.name = p.name = "cubemg_%s_%d_%d_%s_%d_q%d" % (kind, nn, nLevels, mode, size, Q)
so.sList = [default_s]
runopts.logLevel = 1
runopts.verbose = False
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runopts)
ns.calculateSolution("cubemg")
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
    print("RESULT kind=%s nn=%d levels=%d mode=%s ranks=%d N=%d checksum=%.12e"
          % (kind, nn, nLevels, mode, size, N, G.sum()))
    sys.stdout.flush()
