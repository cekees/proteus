"""Helper run under mpiexec by test_parallel_partitioning.py; not collected.

Solves the discontinuous P1 Poisson problem with a direct parallel solve and
prints the partitioning the mesh actually got and a rank-independent checksum
of the solution (the sum of the owned dofs over all ranks).

Usage: dg_partition_mpi.py NAME
"""
import os
import sys

import numpy as np
from proteus import Comm
comm = Comm.init()
from mpi4py import MPI
from importlib import reload
from proteus.iproteus import opts
from proteus import default_n, default_s, default_so, NumericalSolution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poisson_het_2d_p
import poisson_het_2d_dgpk_n

rank, size = comm.rank(), comm.size()
reload(poisson_het_2d_p)
reload(poisson_het_2d_dgpk_n)
p = poisson_het_2d_p
n = poisson_het_2d_dgpk_n
n.femSpaces[0] = default_n.DG_AffineLinearOnSimplexWithNodalBasis
n.nnx = n.nny = n.nn = 11
# the numerics file asks for element partitioning; a discontinuous space needs
# it, because node partitioning leaves out face neighbours of owned elements
n.multilevelLinearSolver = n.levelLinearSolver = default_n.KSP_petsc4py
n.OptDB.clear()
n.OptDB.setValue('ksp_type', 'preonly')
n.OptDB.setValue('pc_type', 'redundant')
n.OptDB.setValue('redundant_pc_type', 'lu')
reload(default_so)
so = default_so
so.name = p.name = "%s_pe%d" % (sys.argv[1], size)
reload(default_s)
so.sList = [default_s]
opts.logLevel = 1
opts.verbose = False
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, opts)
ns.calculateSolution(so.name)
mesh = ns.mlMesh_nList[0].meshList[-1]
u = ns.modelList[0].levelModelList[-1].u[0]
dm = u.femSpace.dofMap
nOwned = int(dm.dof_offsets_subdomain_owned[rank + 1] - dm.dof_offsets_subdomain_owned[rank])
checksum = comm.comm.tompi4py().allreduce(float(np.sum(u.dof[:nOwned])), op=MPI.SUM)
if rank == 0:
    print("RESULT ranks=%d partitioning=%s overlap=%s checksum=%.12e"
          % (size, mesh.parallelPartitioningType, mesh.nLayersOfOverlap, checksum))
    sys.stdout.flush()
for ext in ('h5', 'xmf'):
    try:
        if rank == 0 and os.path.exists(so.name + '.' + ext):
            os.remove(so.name + '.' + ext)
    except OSError:
        pass
