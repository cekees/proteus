"""Multigrid setup and solve cost, by phase; run under mpiexec, not collected.

Poisson on a regular hexahedral or tetrahedral grid (Q1/P1), solved by proteus's MGM,
by PETSc PCMG-preconditioned CG, or -- as the single-level baseline whose iteration
count grows with the mesh -- by ASM-preconditioned CG. Reports the wall-clock time of
each phase (maximum over ranks), the iteration count, and peak resident memory.

Usage: mg_scaling_mpi.py KIND NN LEVELS MODE [independent] [global] [coarse=redundant|direct|amg] [space=p1|dg|nc]

space=dg or nc (tetrahedra only) solves discontinuous P1 (SIPG) or non-conforming P1 instead of C0 P1,
partitioned by elements: their dofs are not vertices, so they take the general transfer route.

KIND     hex or tet
NN       nodes per axis of the coarsest level (the finest has (NN-1)*2**(LEVELS-1)+1)
MODE     mgm | pcmg_cg | asm_cg
independent  partition every level on its own instead of inheriting the coarser one's
         (MultilevelMesh.levelConsistentPartitioning = False), to measure what
         level-consistency buys at scale
coarse=  how the coarsest level is solved in parallel (n.linearCoarseSolver; default redundant)
global   build the global mesh and partition it (regularGrid='global') instead of
         generating each rank's piece

Phases: mesh = hierarchy generation + partitioning (MultilevelMesh construction),
transfers = building the distributed transfer operators, setup = everything else
NumericalSolution does before the solve (spaces, quadrature, Jacobian structures,
solver objects), solve = calculateSolution (assembly + linear solve), of which prepare = the linear solver's
setup (PCMG construction, factorizations) and linear = all of its solve calls, both inside
the solve time; jac and res are the element-loop kernels (time, calls) that assemble
the Jacobian and residual, on all levels.

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
if os.environ.get('LOG_VIEW'):
    from petsc4py import PETSc as _PETSc
    _PETSc.Log.begin()
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

spent = {'mesh': 0.0, 'transfers': 0.0, 'prepare': 0.0, 'linear': 0.0, 'jacobian': 0.0, 'residual': 0.0}
calls = {'jacobian': 0, 'residual': 0}


depth = {}


def timed(owner, name, key):
    """Add the wall-clock time of owner.name to spent[key], counting a call made
    from inside another timed call under the same key once (a V-cycle's levels
    call each other)."""
    original = getattr(owner, name)

    def wrapper(*args, **kwargs):
        outer = depth.get(key, 0) == 0
        depth[key] = depth.get(key, 0) + 1
        t0 = time.time()
        try:
            return original(*args, **kwargs)
        finally:
            depth[key] -= 1
            if outer:
                spent[key] += time.time() - t0
    setattr(owner, name, wrapper)


def counted(owner, name, key):
    original = getattr(owner, name)

    def wrapper(*args, **kwargs):
        calls[key] += 1
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
counted(Transport.OneLevelTransport, 'calculateElementJacobian', 'jacobian')
counted(Transport.OneLevelTransport, 'calculateElementResidual', 'residual')
# the linear solver's own time: its prepare (factorizations, PCMG setup) and its solves
for cls in (LinearSolvers.KSP_petsc4py, LinearSolvers.TwoLevel):
    if 'prepare' in cls.__dict__:
        timed(cls, 'prepare', 'prepare')
    timed(cls, 'solve', 'linear')

TESTDIR = os.path.join(testroot, "ci")
p = defaults.load_physics('poisson_3d_p', TESTDIR)
n = defaults.load_numerics('poisson_3d_c0q1_n' if kind == 'hex' else 'poisson_3d_c0p1_n', TESTDIR)
n.nnx = n.nny = n.nnz = nn
n.linearSolverConvergenceTest = 'r'
n.nLevels = nLevels
space = 'p1'
for flag in flags:
    if flag.startswith('coarse='):
        n.linearCoarseSolver = flag.split('=', 1)[1]
    elif flag.startswith('space='):
        space = flag.split('=', 1)[1]
if space != 'p1':
    assert kind == 'tet', "dg and nc are simplex spaces"
    n.femSpaces[0] = (default_n.DG_AffineLinearOnSimplexWithNodalBasis if space == 'dg'
                      else default_n.NC_AffineLinearOnSimplexWithNodalBasis)
    # a discontinuous space has no coupling between elements unless the flux supplies it on the
    # interior faces; each rank must own whole elements for those fluxes to be complete
    n.numericalFluxType = (default_n.Advection_DiagonalUpwind_Diffusion_SIPG if space == 'dg'
                           else default_n.Advection_DiagonalUpwind_Diffusion_SIPG_exterior)
n.regularGrid = 'global' if 'global' in flags else 'generated'
n.parallelPartitioningType = (MeshTools.MeshParallelPartitioningTypes.element if 'space=dg' in flags or 'space=nc' in flags
                              else MeshTools.MeshParallelPartitioningTypes.node)
n.nLayersOfOverlapForParallel = 1
n.l_atol_res = 1.0e-30
n.linTolFac = 1.0e-8          # relative residual reduction asked of the linear solve
# The problem is linear, so one Newton step with an accurate linear solve is the whole
# solve. Eisenstat-Walker (on by default) asks the linear solver for a loose tolerance,
# the Newton loop then repeats -- ten Jacobian assemblies and factorizations per level
# -- and the cost measured is mostly the repetition.
n.useEisenstatWalker = False
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
    # One Newton iteration per level, whatever the residual. With the default absolute
    # tolerance (1e-8) a linear solve to a relative 1e-8 does not reach it, so the step
    # was declared failed and retried with a smaller dt -- ten times, 10x the assembly
    # and the factorizations -- and the cost measured was mostly the retries.
    n.nonlinearSolverConvergenceTest = 'its'
    n.levelNonlinearSolverConvergenceTest = 'its'
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
newton = ','.join(str(sl.its) for sl in ns.nlsList[0].solverList)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 if sys.platform != 'darwin' else 1024.0 ** 2)
peakMax = mpi.allreduce(peak, op=MPI.MAX)
peakMean = mpi.allreduce(peak, op=MPI.SUM) / size
t_mesh = mpi.allreduce(spent['mesh'], op=MPI.MAX)
def spread(key):
    return (mpi.allreduce(spent[key], op=MPI.MIN), mpi.allreduce(spent[key], op=MPI.SUM) / size,
            mpi.allreduce(spent[key], op=MPI.MAX))
# A rank that reaches a collective early waits there for the others, and the wait is
# charged to whatever phase it is in. Assembly is the imbalanced phase, so the linear
# solver's phases absorb it: the maximum over ranks is mostly waiting, and the minimum
# -- the rank that assembled last, and so waited least -- is the solver's own cost.
t_prepare_min, t_prepare_mean, t_prepare = spread('prepare')
t_linear_min, t_linear_mean, t_linear = spread('linear')
t_jac_min, t_jac_mean, t_jac = spread('jacobian')
t_res = mpi.allreduce(spent['residual'], op=MPI.MAX)
n_jac = calls['jacobian']
n_res = calls['residual']
t_transfers = mpi.allreduce(spent['transfers'], op=MPI.MAX)
t_build = mpi.allreduce(t_build, op=MPI.MAX)
t_solve = mpi.allreduce(t_solve, op=MPI.MAX)
if os.environ.get('LOG_VIEW'):
    _PETSc.Log.view()
if os.environ.get('PER_RANK'):
    sd = lm.mesh
    print("RANK %d elements=%d nodes=%d t_jac=%.2f t_res=%.2f t_linear=%.2f cpu=%d"
          % (rank, sd.nElements_global, sd.nNodes_global, spent['jacobian'], spent['residual'],
             spent['linear'], os.sched_getaffinity(0).__iter__().__next__() if hasattr(os, 'sched_getaffinity') else -1),
          flush=True)
if rank == 0:
    print("RESULT kind=%s mode=%s levels=%d ranks=%d nodes=%d per_rank=%d t_mesh=%.3f t_transfers=%.3f "
          "t_setup=%.3f t_solve=%.3f t_prepare=%.3f t_prepare_min=%.3f t_linear=%.3f t_linear_min=%.3f t_jac=%.3f t_jac_mean=%.3f n_jac=%d t_res=%.3f n_res=%d its=%d rss_max_mb=%.0f rss_mean_mb=%.0f residual=%.3e reason=%d newton_its=%s flags=%s"
          % (kind, mode, nLevels, size, nodes, nodes // size, t_mesh, t_transfers,
             t_build - t_mesh - t_transfers, t_solve, t_prepare, t_prepare_min, t_linear, t_linear_min, t_jac, t_jac_mean, n_jac, t_res, n_res, its, peakMax, peakMean, residual,
             reason, newton, ','.join(flags) or '-'))
    sys.stdout.flush()
