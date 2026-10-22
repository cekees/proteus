"""Helper run under mpiexec by test_parallel_fas.py; not collected.

The paper's degenerate advection-diffusion problem (kees2006_adr_1d_p) on the
unit square, solved by nested iteration with either Newton (the reference) or
FAS, and a checksum of the globally assembled fine-level solution.

Usage: fas_parallel_mpi.py MODE SMOOTHER LEVELS [R] [CYCLES] [SMOOTHS] [NN] [TOLFAC] [NC]
  NC        uncoupled copies of the problem (default 1); each must come out as the single one,
            which is what a stacked system's transfer operator has to deliver
  TOLFAC    the relative tolerance each level's solve stops at; a loose one (0.3) leaves
            the solution unconverged, so it measures the cycles taken, not the answer
  MODE      newton | fas | newton_pcmg (Newton with PETSc PCMG-preconditioned GMRES, for comparison)
  SMOOTHER  NLJacobi | NLGaussSeidel | NLStarILU (ignored for newton)
"""
import os
import sys
import zlib
import time

import numpy as np
from proteus import Comm
comm = Comm.init()
from mpi4py import MPI
from proteus import (Context, LinearSolvers, MeshTools, NonlinearSolvers, NumericalSolution,
                     default_s, default_so, defaults)
from proteus.iproteus import opts as runopts

rank, size = comm.rank(), comm.size()
mpi = comm.comm.tompi4py()
mode, smootherName, nLevels = sys.argv[1], sys.argv[2], int(sys.argv[3])
r = int(sys.argv[4]) if len(sys.argv) > 4 else 2
cycles = int(sys.argv[5]) if len(sys.argv) > 5 else 10
smooths = int(sys.argv[6]) if len(sys.argv) > 6 else 3
nn = int(sys.argv[7]) if len(sys.argv) > 7 else 9
tolFac = float(sys.argv[8]) if len(sys.argv) > 8 and sys.argv[8] != '-' else None
NC = int(sys.argv[9]) if len(sys.argv) > 9 else 1
here = os.path.dirname(os.path.abspath(__file__))

Context.contextOptionsString = "r=%d nd=2 nn=%d" % (r, nn)
p = defaults.load_physics('kees2006_adr_1d_p', here)
n = defaults.load_numerics('kees2006_adr_1d_n', here)
if NC > 1:
    import copy
    from proteus.TransportCoefficients import NonlinearVADR_pqrst
    comps = range(NC)
    p.nc = NC
    p.coefficients = NonlinearVADR_pqrst(nc=NC,
                                         M=dict((i, 0.0) for i in comps),
                                         A=dict((i, p.a * np.eye(p.nd)) for i in comps),
                                         B=dict((i, np.array([p.b] + [0.0] * (p.nd - 1))) for i in comps),
                                         C=dict((i, 0.0) for i in comps),
                                         p=dict((i, 1.0) for i in comps),
                                         q=dict((i, float(p.q)) for i in comps),
                                         r=dict((i, float(p.r)) for i in comps),
                                         s=dict((i, 1.0) for i in comps),
                                         t=dict((i, 0.0) for i in comps))
    p.coefficients.variableNames = ['u%d' % i for i in comps]
    p.analyticalSolution = dict((i, p.analyticalSolution[0]) for i in comps)
    p.dirichletConditions = dict((i, p.getDBC) for i in comps)
    p.initialConditions = dict((i, p.initialConditions[0]) for i in comps)
    p.fluxBoundaryConditions = dict((i, 'noFlow') for i in comps)
    p.diffusiveFluxBoundaryConditions = dict((i, {}) for i in comps)
    n.femSpaces = dict((i, n.femSpaces[0]) for i in comps)
    from proteus.SubgridError import AdvectionDiffusionReaction_ASGS
    n.subgridError = AdvectionDiffusionReaction_ASGS(p.coefficients, p.nd, lag=False)
n.nLevels = nLevels
n.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.node
n.nLayersOfOverlapForParallel = 1
n.maxNonlinearIts = 50
if tolFac is not None:
    n.tolFac = tolFac
    n.nl_atol_res = 1.0e-12
n.multilevelNonlinearSolver = NonlinearSolvers.NLNI
if mode == 'newton_pcmg':
    from proteus import Transport
    Transport.MultilevelTransport.distributedTransfers = True   # PCMG needs PETSc transfer matrices
    n.levelNonlinearSolver = NonlinearSolvers.Newton
    n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
    n.levelLinearSolver = LinearSolvers.KSP_petsc4py
    n.linearSmoother = LinearSolvers.petsc_MG
    n.petscOptions = {'ksp_type': 'gmres', 'pc_type': 'mg', 'pc_mg_cycle_type': 'v',
                      'mg_levels_ksp_type': 'chebyshev', 'mg_levels_ksp_max_it': '2',
                      'mg_levels_pc_type': 'jacobi', 'ksp_rtol': '1e-8'}
elif mode == 'newton':
    n.levelNonlinearSolver = NonlinearSolvers.Newton
    n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
    n.levelLinearSolver = LinearSolvers.KSP_petsc4py
    n.linearSmoother = None
    n.petscOptions = {'ksp_type': 'preonly', 'pc_type': 'redundant', 'redundant_pc_type': 'lu'}
else:
    n.levelNonlinearSolver = NonlinearSolvers.FAS
    n.nonlinearSmoother = getattr(NonlinearSolvers, smootherName)
    n.nonlinearWCycles = cycles
    n.nonlinearPreSmooths = smooths
    n.nonlinearPostSmooths = smooths
    n.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
    n.levelLinearSolver = LinearSolvers.KSP_petsc4py
    n.linearSmoother = None
if os.environ.get('FAS_RELAX'):
    # an experiment knob: the smoothers' damping, which the numerics files cannot reach
    import functools
    NonlinearSolvers.multilevelNonlinearSolverChooser = functools.partial(
        NonlinearSolvers.multilevelNonlinearSolverChooser,
        relaxationFactor=float(os.environ['FAS_RELAX']))
    NumericalSolution.NonlinearSolvers.multilevelNonlinearSolverChooser = NonlinearSolvers.multilevelNonlinearSolverChooser
so = default_so
#the archive is named after every argument (a checksum of them): tests that differ in any of
#them run side by side (pytest-xdist) in this directory, and two jobs opening one HDF5 file hang
so.name = p.name = "fasp_%s_%s_%d_%08x_%d" % (mode, smootherName, nLevels, zlib.crc32(" ".join(sys.argv[1:]).encode()), size)
so.sList = [default_s]
runopts.logLevel = 1
runopts.verbose = False
mpi.Barrier()
t0 = time.time()
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runopts)
mpi.Barrier()
t_build = time.time() - t0
t0 = time.time()
ns.calculateSolution(p.name)
mpi.Barrier()
t_solve = time.time() - t0
solverIts = ','.join(str(sl.its) for sl in ns.nlsList[0].solverList)
lm = ns.modelList[0].levelModelList[-1]
x = np.asarray(lm.mesh.nodeArray)
exact = p.analyticalSolution[0]
sums = []
errs = []
for cj in range(NC):
    dm = lm.u[cj].femSpace.dofMap
    N = dm.nDOF_all_processes
    off = np.asarray(dm.dof_offsets_subdomain_owned)
    nown = int(off[rank + 1] - off[rank])
    g = np.zeros(N)
    s2g = np.asarray(dm.subdomain2global)
    err = 0.0
    for i in range(nown):
        g[int(s2g[i])] = lm.u[cj].dof[i]
        err = max(err, abs(lm.u[cj].dof[i] - exact.uOfX(x[i])))
    G = np.zeros(N)
    mpi.Allreduce(g, G, op=MPI.SUM)
    sums.append(G.sum())
    errs.append(mpi.allreduce(err, op=MPI.MAX))
t_build = mpi.allreduce(t_build, op=MPI.MAX)
t_solve = mpi.allreduce(t_solve, op=MPI.MAX)
if rank == 0:
    print("RESULT mode=%s smoother=%s levels=%d ranks=%d N=%d checksum=%.12e maxerr=%.6e checksums=%s t_build=%.3f t_solve=%.3f its=%s"
          % (mode, smootherName, nLevels, size, N, sums[0], max(errs), ','.join('%.12e' % c for c in sums),
             t_build, t_solve, solverIts))
    sys.stdout.flush()
