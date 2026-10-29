"""Helper run under mpiexec by test_parallel_space_multigrid.py; not collected.

Multigrid for spaces whose degrees of freedom are not mesh vertices -- quadratic
Lagrange, non-conforming P1, discontinuous linear -- on a 2D simplex hierarchy,
in the modes the vertex-based spaces get: proteus's MGM, PETSc's PCMG over the
same hierarchy, PCMG-preconditioned GMRES, and a direct solve to hold the
converged modes to. With Jacobi smoothing and exact transfers the parallel MGM
cycle is the serial one, so the checksum must not depend on the rank count.

Usage: space_mg_mpi.py SPACE LEVELS MODE   (SPACE is p2, ncp1 or dgp1)
"""
import mpi_failfast  # noqa: F401  (an uncaught exception aborts every rank)
import os
import sys

import numpy as np
from proteus import Comm
comm = Comm.init()
from mpi4py import MPI
from proteus.iproteus import opts as runopts
from multilevel_problems import make_uncoupled_components
from proteus import (MeshTools, defaults, default_n, default_s, default_so,
                     NumericalSolution, LinearSolvers, FemTools)

rank, size = comm.rank(), comm.size()
mpi = comm.comm.tompi4py()
testroot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spaceName = sys.argv[1]
nLevels = int(sys.argv[2])
mode = sys.argv[3]
NC = int(sys.argv[4]) if len(sys.argv) > 4 else 1
assert mode in ('mgm', 'pcmg', 'pcmg_its', 'direct', 'ksplu'), mode
space = {'p2': FemTools.C0_AffineQuadraticOnSimplexWithNodalBasis,
         'ncp1': FemTools.NC_AffineLinearOnSimplexWithNodalBasis,
         'dgp1': FemTools.DG_AffineLinearOnSimplexWithNodalBasis}[spaceName]
TESTDIR = os.path.join(testroot, "poisson_2d")
p = defaults.load_physics('poisson_het_2d_p', TESTDIR)
n = defaults.load_numerics('poisson_het_2d_c0pk_n', TESTDIR)
p.meshfile = os.path.join(TESTDIR, 'square4x4')
p.x0 = (0., 0., 0.)
p.L = (1., 1., 1.)
n.femSpaces[0] = space
if NC > 1:
    make_uncoupled_components(p, n, NC, space)
# a discontinuous space has no coupling between elements unless the flux supplies
# it on the interior faces, which the _exterior variant leaves out
n.numericalFluxType = (default_n.Advection_DiagonalUpwind_Diffusion_SIPG if spaceName.startswith('dg')
                       else default_n.Advection_DiagonalUpwind_Diffusion_SIPG_exterior)
n.nLevels = nLevels
# a discontinuous space couples across element faces, so each rank must own whole
# elements; node partitioning leaves the interface elements' face fluxes
# incomplete and even a direct solve then depends on the rank count
n.parallelPartitioningType = (MeshTools.MeshParallelPartitioningTypes.element
                              if spaceName.startswith('dg')
                              else MeshTools.MeshParallelPartitioningTypes.node)
n.nLayersOfOverlapForParallel = 1
if mode == 'direct':
    n.multilevelLinearSolver = LinearSolvers.LU
    n.levelLinearSolver = LinearSolvers.LU
    n.linearSmoother = None
elif mode == 'ksplu':
    # a direct solve that also runs in parallel, to hold the hierarchy modes to
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
so.name = p.name = "spacemg_%s_%d_%s_%dc_%d" % (spaceName, nLevels, mode, NC, size)
so.sList = [default_s]
runopts.logLevel = 1
runopts.verbose = False
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runopts)
ns.calculateSolution("spacemg")
lm = ns.modelList[0].levelModelList[-1]
sums = []
for cj in range(NC):
    dm = lm.u[cj].femSpace.dofMap
    N = dm.nDOF_all_processes
    g = np.zeros(N)
    s2g = np.asarray(dm.subdomain2global)
    off = np.asarray(dm.dof_offsets_subdomain_owned)
    for i in range(int(off[rank + 1] - off[rank])):
        g[int(s2g[i])] = lm.u[cj].dof[i]
    G = np.zeros(N)
    mpi.Allreduce(g, G, op=MPI.SUM)
    sums.append(G.sum())
if rank == 0:
    print("RESULT space=%s levels=%d mode=%s ranks=%d N=%d checksum=%.12e checksums=%s"
          % (spaceName, nLevels, mode, size, N, sums[0], ','.join('%.12e' % c for c in sums)))
    sys.stdout.flush()
