#!/usr/bin/env python
"""CLSVOF (test/CLSVOF/pure_level_set, rotation case) for a few steps; prints the sum and the sum of squares of u.

Run under mpiexec by test_rank_interface_flux.py. PART_TYPE / NLAYERS pick the partition (mesh and assembly
alike). CLSVOF's lumped projections are not summed across ranks, so it only works with a node partition or
overlap; the other is refused.
"""
import os, sys
from proteus import Comm
comm = Comm.init()
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proteus
from proteus import default_s, default_so, MeshTools
from proteus.iproteus import opts
from CLSVOF.pure_level_set import parameters
parameters.ct.test_case = int(os.environ.get("CASE", "2"))
from CLSVOF.pure_level_set import clsvof as C
C.parallel = True
ptype = getattr(MeshTools.MeshParallelPartitioningTypes, os.environ.get("PART_TYPE", "element"))
nl = int(os.environ.get("NLAYERS", "0"))
C.partitioningType = ptype
C.domain.MeshOptions.parallelPartitioningType = ptype
C.domain.MeshOptions.nLayersOfOverlapForParallel = nl
from CLSVOF.pure_level_set import clsvof_p, clsvof_n
clsvof_n.parallelPartitioningType = ptype
clsvof_n.nLayersOfOverlapForParallel = nl
so = default_so
so.tnList = clsvof_n.tnList[:int(os.environ.get("NSTEPS", "6"))]
clsvof_p.name = clsvof_p.name or "clsvof_p"
so.name = "clsvof_r"
ns = proteus.NumericalSolution.NS_base(so, [clsvof_p], [clsvof_n], [default_s], opts)
ns.calculateSolution("r")
m = ns.modelList[0].levelModelList[-1]
from mpi4py import MPI
c = comm.comm.tompi4py()
own = m.mesh.nElements_owned
u = m.q[('u', 0)][:own]; dV = m.q['dV'][:own]
print("RESULT", "%.14e" % c.allreduce(float((u*dV).sum()), op=MPI.SUM), "%.14e" % c.allreduce(float((u**2*dV).sum()), op=MPI.SUM), "own", own, flush=True)
mesh = m.mesh

