"""Multigrid setup and solve cost, by phase; run under mpiexec, not collected.

Poisson on a regular hexahedral or tetrahedral grid (Q1/P1), solved by proteus's MGM,
by PETSc PCMG-preconditioned CG, or -- as the single-level baseline whose iteration
count grows with the mesh -- by ASM-preconditioned CG. Reports the wall-clock time of
each phase (maximum over ranks), the iteration count, and peak resident memory.

Usage: mg_scaling_mpi.py KIND NN LEVELS MODE [independent] [pieces|global]

KIND     hex or tet
NN       nodes per axis of the coarsest level (the finest has (NN-1)*2**(LEVELS-1)+1)
MODE     mgm | pcmg_cg | asm_cg
independent  partition every level on its own instead of inheriting the coarser one's
         (MultilevelMesh.levelConsistentPartitioning = False), to measure what
         level-consistency buys at scale
global   build the global mesh and partition it (regularGrid='global') instead of
         generating each rank's piece

Phases: mesh = hierarchy generation + partitioning (MultilevelMesh construction),
transfers = building the distributed transfer operators, setup = everything else
NumericalSolution does before the solve (spaces, quadrature, Jacobian structures,
solver objects), solve = calculateSolution (assembly + linear solve).

Prints ``RESULT kind=.. ranks=.. nodes=<finest> per_rank=.. t_mesh=.. t_transfers=..
t_setup=.. t_solve=.. its=.. rss_max_mb=.. rss_mean_mb=.. residual=..``.
"""
import os
import resource
import sys
import time

import numpy as np
from proteus import Comm
comm = Comm.init()
from mpi4py import MPI
from proteus.iproteus import opts as runopts
from proteus import (MeshTools, Transport, defaults, default_n, default_s, default_so,
                     NumericalSolution, LinearSolvers)

rank, size = comm.rank(), comm.size()
mpi = comm.comm.tompi4py()
testroot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kind = sys.argv[1]
nn = int(sys.argv[2])
nLevels = int(sys.argv[3])
mode = sys.argv[4]
flags = sys.argv[5:]
assert kind in ('hex', 'tet') and mode in ('mgm', 'pcmg_cg', 'asm_cg'), (kind, mode)
# PETSc's PCMG takes its transfers as PETSc matrices, so a one-rank run needs the
# distributed ones too; proteus's own MGM uses the serial SparseMatrix ones on one rank
if mode == 'pcmg_cg':
    Transport.MultilevelTransport.distributedTransfers = True
if mode == 'mgm':
    # undamped (or 0.8) Jacobi diverges as a stand-alone iteration on Q1/P1 in 3D
    import functools
    LinearSolvers.multilevelLinearSolverChooser = functools.partial(
        LinearSolvers.multilevelLinearSolverChooser, relaxationFactor=0.5)
if 'independent' in flags:
    MeshTools.MultilevelMesh.levelConsistentPartitioning = False

spent = {'mesh': 0.0, 'transfers': 0.0}


def timed(owner, name, key):
    original = getattr(owner, name)

    def wrapper(*args, **kwargs):
        t0 = time.time()
        try:
            return original(*args, **kwargs)
        finally:
            spent[key] += time.time() - t0
    setattr(owner, name, wrapper)


# the mesh phase is the hierarchy constructor; the transfers are one method
meshClass = MeshTools.MultilevelHexahedralMesh if kind == 'hex' else MeshTools.MultilevelTetrahedralMesh
timed(meshClass, '__init__', 'mesh')
timed(Transport.MultilevelTransport, 'useParallelTransfers', 'transfers')

TESTDIR = os.path.join(testroot, "ci")
p = defaults.load_physics('poisson_3d_p', TESTDIR)
n = defaults.load_numerics('poisson_3d_c0q1_n' if kind == 'hex' else 'poisson_3d_c0p1_n', TESTDIR)
n.nnx = n.nny = n.nnz = nn
n.linearSolverConvergenceTest = 'r'
n.nLevels = nLevels
n.regularGrid = 'global' if 'global' in flags else 'generated'
n.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.node
n.nLayersOfOverlapForParallel = 1
n.l_atol_res = 1.0e-30
n.linTolFac = 1.0e-8          # relative residual reduction asked of the linear solve
if mode == 'mgm':
    # one V-cycle per linear solve; the Newton loop around it, which re-forms the residual
    # every time, is what makes it an iteration to a tolerance
    n.maxNonlinearIts = 200
    n.tolFac = 1.0e-8
    n.nl_atol_res = 1.0e-30
    n.multilevelLinearSolver = LinearSolvers.MGM
    n.levelLinearSolver = LinearSolvers.MGM
    n.linearSmoother = LinearSolvers.Jacobi
    n.linearWCycles = 1
    n.linearPreSmooths = n.linearPostSmooths = 3
    n.maxLinearSolverIts = 200
else:
    n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
    n.levelLinearSolver = LinearSolvers.KSP_petsc4py
    if mode == 'pcmg_cg':
        n.linearSmoother = LinearSolvers.petsc_MG
        n.petscOptions = {'ksp_type': 'cg', 'pc_type': 'mg', 'pc_mg_cycle_type': 'v',
                          'mg_levels_ksp_type': 'chebyshev', 'mg_levels_ksp_max_it': '2',
                          'mg_levels_pc_type': 'jacobi',
                          'ksp_rtol': '1e-8', 'ksp_atol': '1e-30', 'ksp_max_it': '500'}
    else:
        n.linearSmoother = None
        n.petscOptions = {'ksp_type': 'cg', 'pc_type': 'asm', 'pc_asm_overlap': '1',
                          'sub_ksp_type': 'preonly', 'sub_pc_type': 'ilu',
                          'ksp_rtol': '1e-8', 'ksp_atol': '1e-30', 'ksp_max_it': '5000'}
so = default_so
so.name = p.name = "mgscale_%s_%d_%d_%s_%d" % (kind, nn, nLevels, mode, size)
so.sList = [default_s]
runopts.logLevel = 1
runopts.verbose = False
mpi.Barrier()
t0 = time.time()
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runopts)
mpi.Barrier()
t_build = time.time() - t0
t0 = time.time()
ns.calculateSolution("mgscale")
mpi.Barrier()
t_solve = time.time() - t0
lm = ns.modelList[0].levelModelList[-1]
dm = lm.u[0].femSpace.dofMap
nodes = dm.nDOF_all_processes
solver = ns.lsList[0].solverList[-1]
reason = 0
if mode == 'mgm':
    its = ns.nlsList[0].solverList[-1].its
else:
    its = int(solver.ksp.getIterationNumber())
    reason = int(solver.ksp.getConvergedReason())
# the true residual of the discrete problem, relative to the one at u = 0: the
# solvers' own norms (preconditioned, scaled) are not comparable across modes
nOwned = int(dm.dof_offsets_subdomain_owned[rank + 1] - dm.dof_offsets_subdomain_owned[rank])
def residualNorm(u):
    r = np.zeros_like(u)
    lm.getResidual(u, r)
    return np.sqrt(mpi.allreduce(float(np.dot(r[:nOwned], r[:nOwned])), op=MPI.SUM))
uSolution = np.array(lm.u[0].dof, copy=True)
uZero = np.zeros_like(uSolution)
residual = residualNorm(uSolution) / residualNorm(uZero)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 if sys.platform != 'darwin' else 1024.0 ** 2)
peakMax = mpi.allreduce(peak, op=MPI.MAX)
peakMean = mpi.allreduce(peak, op=MPI.SUM) / size
t_mesh = mpi.allreduce(spent['mesh'], op=MPI.MAX)
t_transfers = mpi.allreduce(spent['transfers'], op=MPI.MAX)
t_build = mpi.allreduce(t_build, op=MPI.MAX)
t_solve = mpi.allreduce(t_solve, op=MPI.MAX)
if rank == 0:
    print("RESULT kind=%s mode=%s levels=%d ranks=%d nodes=%d per_rank=%d t_mesh=%.3f t_transfers=%.3f "
          "t_setup=%.3f t_solve=%.3f its=%d rss_max_mb=%.0f rss_mean_mb=%.0f residual=%.3e reason=%d flags=%s"
          % (kind, mode, nLevels, size, nodes, nodes // size, t_mesh, t_transfers,
             t_build - t_mesh - t_transfers, t_solve, its, peakMax, peakMean, residual,
             reason, ','.join(flags) or '-'))
    sys.stdout.flush()
