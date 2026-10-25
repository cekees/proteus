#!/usr/bin/env python
"""Steady advection-diffusion with an exact solution; prints its L2 error.

Run by ``test_adr_problem.py`` under ``mpiexec`` to compare a partitioned run
against a serial one.  The partition, the overlap and the direction of the
advection all come from the environment, because the bug this guards against
only shows when the cut is across the flow:

    -div(a grad u) + div(b u) = r   on the unit square, u = x^2 + y^2, a = I,
    b = (1, 0) or (0, 1), Dirichlet all round so the solution is pinned.

SPACE=DG runs the same problem in DG (with the partition ADRProblem picks).

With ``b = (1, 0)`` and the contiguous-index partition that
``-mat_partitioning_type current`` gives, the cut runs along the flow, every
interface face has ``b.n == 0``, and a spurious boundary flux there is zero
whether or not anything removes it.  ``BDIR=y`` turns the flow across the cut,
where the spurious flux is largest.
"""
import os
import sys

from proteus import Comm

comm = Comm.init()

from proteus import ADRProblem, MeshTools, NumericalSolution, default_s, defaults
from proteus.iproteus import opts

cells = int(os.environ.get("CELLS", "16"))
bdir = os.environ.get("BDIR", "y")
f = ["u", "0.0"] if bdir == "x" else ["0.0", "u"]
df = ["1.0", "0.0"] if bdir == "x" else ["0.0", "1.0"]
# residual form is  div(f - a grad u) + r = 0, so r = 4 - 2*(the advected coordinate)
r = "4.0 - 2.0*x" if bdir == "x" else "4.0 - 2.0*y"

if os.environ.get("ADVONLY"):
    # pure advection, b = (0, 1): Dirichlet on the inflow side only, free outflow,
    # so the cut sees both inflow and outflow faces
    f, df, r = ["0.0", "u"], ["0.0", "1.0"], "-2.0*y"
problem = {
    "adr": {"dim": 2, "components": ["u"],
            "equations": [{
                "component": "u",
                "advection": {"f": f, "df": {"u": df},
                              "depends_on": {"u": "nonlinear"}},
                **({} if os.environ.get("ADVONLY") else {"diffusion": {"u": {"rowptr": [0, 1, 2], "colind": [0, 1],
                                    "a": ["1.0", "1.0"], "da": {}, "depends_on": {}}}}),
                "reaction": {"r": r, "dr": {}, "depends_on": {}},
            }]},
    "domain": {"lower": [0.0, 0.0], "upper": [1.0, 1.0]},
    "dirichlet": [{"component": "u", "region": "all",
                   "where": "y <= 1e-9" if os.environ.get("ADVONLY") else None,
                   "value": "x**2 + y**2"}],
    "exact": {"u": "x**2 + y**2"},
}

_, p, n, system = ADRProblem.setup(problem, {"u": (os.environ.get("SPACE", "CG"), 1)}, cells, name="adr_adv_part")
if os.environ.get("FLUX"):
    from proteus import NumericalFlux
    n.numericalFluxType = getattr(NumericalFlux, os.environ["FLUX"])
# an empty PART_TYPE / NLAYERS leaves what ADRProblem.setup chose
if os.environ.get("PART_TYPE", "element"):
    n.parallelPartitioningType = getattr(MeshTools.MeshParallelPartitioningTypes,
                                         os.environ.get("PART_TYPE", "element"))
if os.environ.get("NLAYERS", "0"):
    n.nLayersOfOverlapForParallel = int(os.environ.get("NLAYERS", "0"))
so = defaults.System_base(name="adr_adv_part", pnList=[(p, n)], sList=[default_s])
so.tnList = system["tnList"]
ns = NumericalSolution.NS_base(so, [p], [n], so.sList, opts)
ns.calculateSolution("adr_adv_part")
model = ns.modelList[0].levelModelList[-1]
errors = ADRProblem.l2_errors(model, problem, so.tnList[-1])

mesh = model.mesh
interfaces = int((mesh.elementBoundaryMaterialTypes[
    mesh.exteriorElementBoundariesArray] == 0).sum())
print("RANK rank=%d exterior=%d interfaces=%d"
      % (comm.rank(), len(mesh.exteriorElementBoundariesArray), interfaces), flush=True)
if comm.rank() == 0:
    print("RESULT ranks=%d L2_u=%.17e" % (comm.size(), errors["u"]), flush=True)
