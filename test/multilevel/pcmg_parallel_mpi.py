"""Helper run under mpiexec by test_petsc_pcmg.py; not collected by pytest.

Solves the 2D Poisson problem with either proteus's MGM or PETSc's PCMG over
the same hierarchy, and prints a checksum of the globally assembled fine-level
solution.

The two are configured to be the same algorithm: one V-cycle, three sweeps of
Jacobi damped by 0.8 before and after, re-discretized coarse matrices, an exact
coarse solve. Then they must agree to roundoff, which is a check of everything
the two share -- the transfer operators, the level matrices, the vector layouts
-- and of nothing else, because the cycle itself is implemented twice
independently. Any difference is therefore a defect in what we feed PETSc.

Usage: pcmg_parallel_mpi.py LEVELS MODE [DIM]   (DIM is 2 or 3, default 2)

Modes:
  mgm       proteus MGM, linearWCycles=1
  pcmg      PETSc PCMG, KSP preonly (one application of the cycle)
  pcmg_its  PETSc PCMG preconditioning GMRES to a tight tolerance, which
            converges to the discrete solution rather than reproducing a cycle
  direct    a direct solve, the reference the converged mode is held to
"""
import os
import sys

import numpy as np
from proteus import Comm
comm = Comm.init()
from mpi4py import MPI
from proteus.iproteus import opts as runopts
from multilevel_problems import make_uncoupled_components
from proteus import (MeshTools, defaults, default_n, default_s, default_so,
                     NumericalSolution, LinearSolvers)

rank, size = comm.rank(), comm.size()
mpi = comm.comm.tompi4py()
nLevels = int(sys.argv[1]) if len(sys.argv) > 1 else 3
mode = sys.argv[2] if len(sys.argv) > 2 else 'pcmg'
assert mode in ('mgm', 'pcmg', 'pcmg_its', 'direct'), mode
dim = int(sys.argv[3]) if len(sys.argv) > 3 else 2
assert dim in (2, 3), dim
NC = int(sys.argv[4]) if len(sys.argv) > 4 else 1
assert NC == 1 or dim == 2, 'components are only wired up for the 2D problem'
testroot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if dim == 2:
    TESTDIR = os.path.join(testroot, "poisson_2d")
    p = defaults.load_physics('poisson_het_2d_p', TESTDIR)
    n = defaults.load_numerics('poisson_het_2d_c0pk_n', TESTDIR)
    p.meshfile = os.path.join(TESTDIR, 'square4x4')
    p.x0 = (0., 0., 0.)
    p.L = (1., 1., 1.)
    n.femSpaces[0] = default_n.C0_AffineLinearOnSimplexWithNodalBasis
    n.numericalFluxType = default_n.Advection_DiagonalUpwind_Diffusion_SIPG_exterior
    if NC > 1:
        make_uncoupled_components(p, n, NC, default_n.C0_AffineLinearOnSimplexWithNodalBasis)
else:
    # a regular tetrahedral mesh on a domain that does not start at the
    # origin, which is what first exposed finer levels being left unshifted
    TESTDIR = os.path.join(testroot, "ci")
    p = defaults.load_physics('poisson_3d_p', TESTDIR)
    n = defaults.load_numerics('poisson_3d_c0p1_n', TESTDIR)
    n.nnx = n.nny = n.nnz = 3
    n.linearSolverConvergenceTest = 'r'
n.nLevels = nLevels
n.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.node
n.nLayersOfOverlapForParallel = 1
if mode == 'direct':
    n.multilevelLinearSolver = LinearSolvers.LU
    n.levelLinearSolver = LinearSolvers.LU
    n.linearSmoother = None
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
    # the numerics module puts -pc_type asm in the global options database when
    # `parallel` is set, and setFromOptions would honor it over the PC the solver
    # was given, so say what we mean
    options = {
        'pc_type': 'mg',
        'pc_mg_cycle_type': 'v',
        'mg_levels_ksp_type': 'richardson',
        'mg_levels_ksp_richardson_scale': '0.8',
        'mg_levels_ksp_max_it': '3',
        'mg_levels_pc_type': 'jacobi',
    }
    if mode == 'pcmg':
        options['ksp_type'] = 'preonly'
    else:
        options['ksp_type'] = 'gmres'
        options['ksp_rtol'] = '1e-14'
        options['ksp_atol'] = '1e-14'
    n.petscOptions = options
so = default_so
so.name = p.name = "pcmg_%s_%d_%dd_%dc_%d" % (mode, nLevels, dim, NC, size)
so.sList = [default_s]
runopts.logLevel = 1
runopts.verbose = False
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runopts)
ns.calculateSolution("pcmg")
lm = ns.modelList[0].levelModelList[-1]
sums = []
for cj in range(NC):
    dm = lm.u[cj].femSpace.dofMap
    N = dm.nDOF_all_processes
    g = np.zeros(N)
    s2g = np.asarray(dm.subdomain2global)
    off = np.asarray(dm.dof_offsets_subdomain_owned)
    nown = int(off[rank + 1] - off[rank])
    dof = np.asarray(lm.u[cj].dof)
    for i in range(nown):
        g[int(s2g[i])] = dof[i]
    G = np.zeros(N)
    mpi.Allreduce(g, G, op=MPI.SUM)
    sums.append(G.sum())
if rank == 0:
    print(f"RESULT mode={mode} dim={dim} levels={nLevels} ranks={size} N={N} "
          f"checksum={sums[0]:.12e} checksums={','.join('%.12e' % c for c in sums)}")
    sys.stdout.flush()
