"""Helper run under mpiexec by test_parallel_smoothers.py; not collected by pytest.

Solves the 2D Poisson problem with one of proteus's own smoothers as the
linear solver and prints a checksum of the globally assembled solution, so the
same run at different rank counts can be compared.  Jacobi is pointwise, so
with the ghost entries kept current its iterates are the serial ones; the
others become their block variants in parallel and need only converge to the
same solution.
"""
import mpi_failfast  # noqa: F401  (an uncaught exception aborts every rank)
import os
import sys

import numpy as np
from proteus import Comm
comm = Comm.init()
from mpi4py import MPI
from proteus.iproteus import opts as runopts
from proteus import MeshTools, defaults, default_n, default_s, default_so, NumericalSolution, LinearSolvers
rank, size = comm.rank(), comm.size()
mpi = comm.comm.tompi4py()
TESTDIR=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"poisson_2d")
smoother = sys.argv[1] if len(sys.argv)>1 else "Jacobi"
S = getattr(LinearSolvers, smoother)
p=defaults.load_physics('poisson_het_2d_p',TESTDIR); n=defaults.load_numerics('poisson_het_2d_c0pk_n',TESTDIR)
p.meshfile=os.path.join(TESTDIR,'square4x4'); p.x0=(0.,0.,0.); p.L=(1.,1.,1.)
n.nLevels=1
n.femSpaces[0]=default_n.C0_AffineLinearOnSimplexWithNodalBasis
n.numericalFluxType=default_n.Advection_DiagonalUpwind_Diffusion_SIPG_exterior
n.multilevelLinearSolver=S; n.levelLinearSolver=S
n.linearSolverMaxIts=200; n.linearSolverConvergenceTest='r'
n.parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.node
n.nLayersOfOverlapForParallel=1
so=default_so; so.name=p.name="sm_%s_%d"%(smoother,size); so.sList=[default_s]
runopts.logLevel=1; runopts.verbose=False
ns=NumericalSolution.NS_base(so,[p],[n],so.sList,runopts)
ns.calculateSolution("sm")
lm=ns.modelList[0].levelModelList[-1]
dm=lm.u[0].femSpace.dofMap
N=dm.nDOF_all_processes
g=np.zeros(N)
s2g=np.asarray(dm.subdomain2global)
off=np.asarray(dm.dof_offsets_subdomain_owned)
nown=int(off[rank+1]-off[rank])
dof=np.asarray(lm.u[0].dof)
for i in range(nown):
    g[int(s2g[i])] = dof[i]
G=np.zeros(N); mpi.Allreduce(g,G,op=MPI.SUM)
if rank==0:
    print(f"RESULT smoother={smoother} ranks={size} N={N} checksum={G.sum():.12e} norm={np.linalg.norm(G):.12e}")

