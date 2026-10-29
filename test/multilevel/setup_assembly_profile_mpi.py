"""Where setup and assembly spend their time and memory; run under mpiexec, not collected.

The evidence base of the C++ refactor campaign (CXX_REFACTOR_setup_and_assembly_SCOPING.md).
Poisson on a regular hexahedral or tetrahedral grid, one Newton step per level, timing with
plain wall clocks (no profiler: cProfile and tracemalloc inflate per-call costs a thousandfold
and distort the ratios) the pieces of NumericalSolution's construction and of the solve, as
the maximum over ranks, and reporting where the memory of the finest level goes.

Usage: setup_assembly_profile_mpi.py KIND NN LEVELS [quad=N]

quad=N overrides the element quadrature order the test numerics ask for (CubeGaussQuadrature(nd,3)
is 64 points on a hexahedron, where a trilinear Poisson problem needs 8); the memory per node is
proportional to it, so it is the first thing to look at.

Prints PHASE lines (seconds, max over ranks), the memory by array group, and one RESULT line.
"""
import mpi_failfast  # noqa: F401  (an uncaught exception aborts every rank)
import os
import resource
import sys
import time

import numpy as np
from proteus import Comm
comm = Comm.init()
from mpi4py import MPI
from proteus import (MeshTools, defaults, default_n, default_s, default_so, NumericalSolution,
                     LinearSolvers, Transport, TransportCoefficients, NumericalFlux, FemTools)
from proteus.iproteus import opts as runopts

rank, size = comm.rank(), comm.size()
mpi = comm.comm.tompi4py()
kind, nn, levels = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
quad = None
for flag in sys.argv[4:]:
    if flag.startswith('quad='):
        quad = int(flag.split('=')[1])
spent = {}


def timed(owner, name, label):
    f = getattr(owner, name)

    def wrapper(*args, **kwargs):
        t = time.perf_counter()
        try:
            return f(*args, **kwargs)
        finally:
            spent[label] = spent.get(label, 0.0) + time.perf_counter() - t
    setattr(owner, name, wrapper)


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2 if sys.platform == 'darwin' else 1024.0)


meshClass = MeshTools.MultilevelHexahedralMesh if kind == 'hex' else MeshTools.MultilevelTetrahedralMesh
timed(meshClass, '__init__', '1 mesh: generate + partition the hierarchy')
timed(Transport.MultilevelTransport, '__init__', '2 models: MultilevelTransport.__init__ (spaces, arrays, quadrature, transfers)')
timed(Transport.OneLevelTransport, '__init__', '3   of which OneLevelTransport.__init__ (every level)')
timed(TransportCoefficients.PoissonEquationCoefficients, 'initializeElementQuadrature', '4     coefficients at element quadrature (Python callbacks per point)')
timed(TransportCoefficients.PoissonEquationCoefficients, 'initializeGlobalExteriorElementBoundaryQuadrature', '5     coefficients at boundary quadrature (Python callbacks per point)')
timed(Transport.MultilevelTransport, 'useParallelTransfers', '6   transfer operators')
timed(Transport.OneLevelTransport, 'calculateElementResidual', 'A element residual kernel wrapper')
timed(Transport.OneLevelTransport, 'calculateElementJacobian', 'B element Jacobian kernel wrapper')
timed(Transport.OneLevelTransport, 'calculateExteriorElementBoundaryCoefficients', 'C boundary coefficients (Python callbacks per point)')
timed(Transport.OneLevelTransport, 'getJacobian', 'D getJacobian (all of it)')
timed(Transport.OneLevelTransport, 'getResidual', 'E getResidual (all of it)')
timed(NumericalSolution.NS_base, 'allocateModels', '0 NS_base.allocateModels (everything above, plus solvers)')

testroot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = defaults.load_physics('poisson_3d_p', os.path.join(testroot, 'ci'))
n = defaults.load_numerics('poisson_3d_c0q1_n' if kind == 'hex' else 'poisson_3d_c0p1_n', os.path.join(testroot, 'ci'))
n.nnx = n.nny = n.nnz = nn
n.nLevels = levels
if quad is not None:
    if kind == 'hex':
        n.elementQuadrature = default_n.CubeGaussQuadrature(p.nd, quad)
        n.elementBoundaryQuadrature = default_n.CubeGaussQuadrature(p.nd - 1, quad)
    else:
        n.elementQuadrature = default_n.SimplexGaussQuadrature(p.nd, quad)
        n.elementBoundaryQuadrature = default_n.SimplexGaussQuadrature(p.nd - 1, quad)
n.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.node
n.nLayersOfOverlapForParallel = 1
n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
n.levelLinearSolver = LinearSolvers.KSP_petsc4py
n.linearSmoother = LinearSolvers.petsc_MG
n.petscOptions = {'ksp_type': 'cg', 'pc_type': 'mg', 'mg_levels_ksp_type': 'chebyshev', 'mg_levels_ksp_max_it': '2',
                  'mg_levels_pc_type': 'jacobi', 'ksp_rtol': '1e-8'}
n.nonlinearSolverConvergenceTest = 'its'
n.levelNonlinearSolverConvergenceTest = 'its'
n.useEisenstatWalker = False
Transport.MultilevelTransport.distributedTransfers = True
so = default_so
so.name = p.name = 'setupprofile'
so.sList = [default_s]
runopts.logLevel = 1
runopts.verbose = False
mpi.Barrier()
rss0 = rss()
t0 = time.perf_counter()
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runopts)
mpi.Barrier()
tBuild = time.perf_counter() - t0
rssBuild = rss()
t0 = time.perf_counter()
ns.calculateSolution('setupprofile')
mpi.Barrier()
tSolve = time.perf_counter() - t0
lm = ns.modelList[0].levelModelList[-1]
nodes = lm.u[0].femSpace.dofMap.nDOF_all_processes
groups = {}


def nbytes(obj):
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(nbytes(v) for v in obj.values())
    return 0


for name in ('q', 'ebqe', 'ebq', 'ebq_global', 'phi_ip', 'elementResidual', 'elementJacobian', 'elementJacobian_eb'):
    groups[name] = nbytes(getattr(lm, name, None))
qRows = sorted(((v.nbytes, k, v.shape) for k, v in lm.q.items() if isinstance(v, np.ndarray)), key=lambda r: -r[0])
nq = lm.q[('v', 0)].shape[1] if ('v', 0) in lm.q else 0
reduced = {k: mpi.allreduce(v, op=MPI.MAX) for k, v in spent.items()}
tBuild = mpi.allreduce(tBuild, op=MPI.MAX)
tSolve = mpi.allreduce(tSolve, op=MPI.MAX)
peak = mpi.allreduce(rssBuild, op=MPI.MAX)
final = mpi.allreduce(rss(), op=MPI.MAX)
nqMax = mpi.allreduce(nq, op=MPI.MAX)
groupsMax = {k: mpi.allreduce(v, op=MPI.MAX) for k, v in groups.items()}
qTop = [(mpi.allreduce(b, op=MPI.MAX), k, s) for (b, k, s) in qRows[:8]]
if rank == 0:
    print("RESULT kind=%s ranks=%d levels=%d fine_nodes=%d quad_points=%d build=%.3f solve=%.3f rss_import_mb=%.0f "
          "rss_after_build_mb=%.0f rss_final_mb=%.0f fine_q_mb=%.1f fine_q_bytes_per_node=%.0f"
          % (kind, size, levels, nodes, nqMax, tBuild, tSolve, rss0, peak, final, groupsMax['q'] / 1e6,
             groupsMax['q'] / max(nodes / size, 1)))
    for label in sorted(reduced):
        print("PHASE %-82s %8.3f s" % (label, reduced[label]))
    for k, v in groupsMax.items():
        print("MEMORY %-22s %9.2f MB (finest level, max over ranks)" % (k, v / 1e6))
    for b, k, s in qTop:
        print("QARRAY %-26s %9.2f MB %s" % (k, b / 1e6, s))
    sys.stdout.flush()
