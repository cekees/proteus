"""Helper run under mpiexec by test_parallel_mgm.py; not collected by pytest.

Solves the 2D Poisson problem with proteus's own geometric multigrid and prints
a checksum of the globally assembled fine-level solution, so one run at
different rank counts can be compared.

Jacobi smoothing is pointwise and the transfer operators are exact, so when the
coarse solve is exact too the parallel cycle is the serial cycle and the
checksum must agree at any rank count to roundoff. A coarse solve that is only
block-Jacobi across the ranks breaks that, which is how this is detected.
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
                     NumericalSolution, LinearSolvers)

rank, size = comm.rank(), comm.size()
mpi = comm.comm.tompi4py()
nLevels = int(sys.argv[1]) if len(sys.argv) > 1 else 3
smoother = sys.argv[2] if len(sys.argv) > 2 else 'Jacobi'
cycles = int(sys.argv[3]) if len(sys.argv) > 3 else 3
smooths = int(sys.argv[4]) if len(sys.argv) > 4 else 3
NC = int(sys.argv[5]) if len(sys.argv) > 5 else 1
TESTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "poisson_2d")

p = defaults.load_physics('poisson_het_2d_p', TESTDIR)
n = defaults.load_numerics('poisson_het_2d_c0pk_n', TESTDIR)
p.meshfile = os.path.join(TESTDIR, 'square4x4')
p.x0 = (0., 0., 0.)
p.L = (1., 1., 1.)
n.nLevels = nLevels
n.femSpaces[0] = default_n.C0_AffineLinearOnSimplexWithNodalBasis
if NC > 1:
    make_uncoupled_components(p, n, NC, default_n.C0_AffineLinearOnSimplexWithNodalBasis)
n.numericalFluxType = default_n.Advection_DiagonalUpwind_Diffusion_SIPG_exterior
n.multilevelLinearSolver = LinearSolvers.MGM
n.levelLinearSolver = LinearSolvers.MGM
n.linearSmoother = getattr(LinearSolvers, smoother)
n.linearWCycles = cycles
n.linearPreSmooths = n.linearPostSmooths = smooths
n.parallelPartitioningType = MeshTools.MeshParallelPartitioningTypes.node
n.nLayersOfOverlapForParallel = 1
so = default_so
#every argument is in the name: the archive file is named after it, and tests that differ
#only in cycles, smooths or components run side by side (pytest-xdist) in this directory,
#where two jobs opening the same HDF5 file hang each other
so.name = p.name = "mgm_%d_%s_%d_%d_%dc_%d" % (nLevels, smoother, cycles, smooths, NC, size)
so.sList = [default_s]
runopts.logLevel = 1
runopts.verbose = False
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, runopts)
ns.calculateSolution("mgm")
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
    print(f"RESULT levels={nLevels} smoother={smoother} ranks={size} N={N} "
          f"checksum={sums[0]:.12e} checksums={','.join('%.12e' % c for c in sums)}")
    sys.stdout.flush()
