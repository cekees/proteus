#!/usr/bin/env python
"""The 2D periodic duct (duct.py) on one or two ranks; prints sums of the velocity.

Run by ``test_periodic_parallel.py``. In parallel the periodic numbering needs an
element partition without overlap, and that has to be asked of the mesh as well
as of n. SOLVER_OPTIONS names the PETSc options file; a serial run uses the
superlu one and a parallel run the same with superlu_dist.
"""
import os
import sys

from proteus import Comm, Context, NumericalSolution, default_s, defaults
import proteus.test_utils.TestTools as TestTools

HERE = os.path.dirname(os.path.abspath(__file__))
Context.contextOptionsString = ("periodic=True grid=True nd=2 nnx=42 triangles=False spaceOrder=1 "
                                "weak=True coord=True pc_type='selfp_petsc'")
Comm.argv = TestTools.fixture_set_petsc_options_from_file(os.environ["SOLVER_OPTIONS"])
comm = Comm.init()

import numpy as np

from proteus.iproteus import opts

defaults.load_source("duct", os.path.join(HERE, "duct.py"))
so = defaults.load_system("duct", path=HERE)
pList, nList = [], []
for (p, n) in so.pnList:
    if not isinstance(p, defaults.Physics_base):
        pList.append(defaults.load_physics(p))
        nList.append(defaults.load_numerics(n))
    else:
        pList.append(p)
        nList.append(n)
if os.environ.get("TIGHT"):
    for n in nList:
        n.nl_atol_res = 1.0e-13
        n.l_atol_res = 1.0e-15
        n.tolFac = 0.0
        n.linTolFac = 0.0
sList = [default_s for _ in so.pnList] if so.sList == [] else so.sList
ns = NumericalSolution.NS_base(so, pList, nList, sList, opts)
ns.calculateSolution("duct_part")
model = ns.modelList[0].levelModelList[-1]
mesh = model.mesh
owned = mesh.nNodes_owned
from mpi4py import MPI
c = comm.comm.tompi4py()
result = {}
for i, name in enumerate(("p", "u", "v")):
    dof = model.u[i].dof[:owned]
    result[name + "2"] = c.allreduce(float((dof ** 2).sum()), op=MPI.SUM)
    result[name + "s"] = c.allreduce(float(dof.sum()), op=MPI.SUM)
print("RANK rank=%d elements=%d owned=%d" % (comm.rank(), mesh.nElements_global, mesh.nElements_owned), flush=True)
if comm.rank() == 0:
    print("RESULT ranks=%d " % comm.size() + " ".join("%s=%.17e" % kv for kv in result.items()), flush=True)
