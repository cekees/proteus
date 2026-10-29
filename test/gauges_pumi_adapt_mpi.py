#!/usr/bin/env python
"""Runs point, line and line integral gauges on a PUMI mesh that has just been
adapted, and prints what they measure as JSON.

Run by ``test_gauges_pumi_mpi.py`` under ``mpiexec``, on 2 or 4 ranks.  The unit
square, stored pre-split for 1, 2 and 4 ranks (see
``MeshAdaptPUMI/make_adapt_meshes.py``), is adapted to a circle level set, which
refines a band around the circle and, under MPI, balances the parts with ParMA, so
the partition the gauges see is not one anyone wrote down.  Then
``convertFromPUMI`` and ``generatePartitionedMeshFromPUMI`` build the Proteus mesh
the way a run does after each adapt, and gauges are attached to it.  The field is
u = x + 10 y, so every value has a closed form and the answer must not depend on
the partition.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from proteus import Comm

comm = Comm.init()

import test_gauges as T
from proteus import Domain, MeshTools, cmeshTools
from proteus.MeshAdaptPUMI import MeshAdapt
from proteus.Gauges import PointGauges, LineGauges, LineIntegralGauges
from proteus.MeshTools import MeshParallelPartitioningTypes

PUMI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MeshAdaptPUMI")


def adapted_square():
    """The stored unit square, adapted to a circle of radius 0.3 about its centre."""
    adapter = MeshAdapt.MeshAdapt()
    domain = Domain.PUMIDomain(dim=2, manager=MeshAdapt.AdaptManager(adapter))
    manager = domain.AdaptManager
    manager.sizeInputs = [b"interface"]
    manager.adapt = 1
    manager.hmax, manager.hmin, manager.hphi, manager.numIterations = 0.1, 0.0125, 0.025, 5
    adapter.loadModelAndMesh(os.path.join(PUMI_DIR, "adapt_square.dmg").encode("ascii"),
                             (os.path.join(PUMI_DIR, "adapt_square_%ip.smb" % comm.size())).encode("ascii"))
    adapter.setAdaptProperties(manager)

    def convert():
        mesh = MeshTools.TriangularMesh()
        mesh.cmesh = cmeshTools.CMesh()
        mesh.convertFromPUMI(domain, adapter, domain.faceList, domain.regList,
                             parallel=comm.size() > 1, dim=2)
        return mesh

    local = convert()
    local = local.subdomainMesh if comm.size() > 1 else local
    adapter.transferPropertiesToPUMI(np.array([998.2, 1.205]), np.array([1.004e-6, 1.5e-5]),
                                     np.zeros(3), 1.0, 1.0, 1.0, 2.0)
    phi = np.linalg.norm(local.nodeArray[:, :2] - 0.5, axis=1) - 0.3
    adapter.transferFieldToPUMI(b"phi", np.ascontiguousarray(phi.reshape(-1, 1)))
    adapter.adaptPUMIMesh(b"")

    mesh = convert()
    ml = MeshTools.MultilevelTriangularMesh(0, 0, 0, skipInit=True, nLayersOfOverlap=0,
                                            parallelPartitioningType=MeshParallelPartitioningTypes.element)
    ml.generatePartitionedMeshFromPUMI(mesh, 1, nLayersOfOverlap=0)
    return ml


def run(gauges, name):
    filename = "gauges_pumi_adapt_%s.csv" % name
    T.silent_rm(filename)
    comm.barrier()
    gauge = gauges(fileName=filename)
    T.run_gauge(gauge, [0.0], nd=2, mesh_builder=adapted_square)
    names, data = T.parse_gauge_output(filename)
    T.delete_file(filename)
    return {"names": names, "values": np.atleast_2d(data)[0, 1:].tolist()}


result = {"size": comm.size()}

# points on the circle's band, near the corners, on the cuts a 4-way split of the
# square is likely to make (x or y = 0.5), and a roundoff outside the square
points = ((0.5, 0.2, 0), (0.8, 0.5, 0), (0.5, 0.8, 0), (0.2, 0.5, 0), (0.37, 0.22, 0),
          (0.05, 0.05, 0), (0.95, 0.95, 0), (0.5, 0.5, 0), (0.25, 0.75, 0), (1 + 1e-12, 0.5, 0))
result["points"] = run(lambda fileName: PointGauges(gauges=((("u0",), points),), fileName=fileName),
                       "points")
# a line through the refined band, leaving the mesh at both ends
line = ((-1, 0.25, 0), (2, 0.25, 0))
result["line"] = run(lambda fileName: LineGauges(gauges=((("u0",), (line,)),), fileName=fileName), "line")
result["integral"] = run(lambda fileName: LineIntegralGauges(gauges=((("u0",), (line,)),), fileName=fileName),
                         "integral")
# along the vertical through the centre, which the circle's band crosses twice
vertical = ((0.5, -0.5, 0), (0.5, 1.5, 0))
result["vertical_integral"] = run(
    lambda fileName: LineIntegralGauges(gauges=((("u0",), (vertical,)),), fileName=fileName),
    "vertical")

# erdc#426: the element containing a point is not always in the star of the nearest node.  On this
# graded mesh it happens for about 1 point in 200; the ring search must find the element anyway.
from scipy import spatial
from proteus.Gauges import PointGauges as _PG
model, ic = T.gauge_setup(2, mesh_builder=adapted_square)
mesh = model.levelModelList[-1].mesh
femSpace = model.levelModelList[-1].u[0].femSpace
probe = _PG(gauges=((("u0",), ((0.5, 0.5, 0),)),), fileName="gauges_pumi_adapt_probe.csv")
tree = spatial.cKDTree(mesh.nodeArray)
rng = np.random.default_rng(1 + comm.rank())
elements = rng.integers(0, mesh.nElements_global, 3000)
points = np.einsum("ni,nij->nj", rng.dirichlet(np.ones(3), len(elements)),
                   mesh.nodeArray[mesh.elementNodesArray[elements]])
outside_star = located = 0
for p in points:
    node = int(tree.query(p)[1])
    star = mesh.nodeElementsArray[mesh.nodeElementOffsets[node]:mesh.nodeElementOffsets[node + 1]]
    element = probe.getLocalElement(femSpace, p, node)
    inside_found = False
    if element is not None:
        X = mesh.nodeArray[mesh.elementNodesArray[element]][:, :2]
        lam = np.linalg.solve(np.vstack([X.T, np.ones(3)]), np.append(p[:2], 1.0))
        inside_found = bool(np.all(lam >= -1e-9))
    containing_in_star = False
    for e in star:
        X = mesh.nodeArray[mesh.elementNodesArray[e]][:, :2]
        containing_in_star |= bool(np.all(np.linalg.solve(np.vstack([X.T, np.ones(3)]), np.append(p[:2], 1.0)) >= -1e-9))
    outside_star += int(not containing_in_star)
    located += int(inside_found)
totals = comm.comm.tompi4py().allgather((len(points), outside_star, located))
result["location"] = {"points": sum(t[0] for t in totals), "outside_star": sum(t[1] for t in totals),
                      "located": sum(t[2] for t in totals)}

if comm.rank() == 0:
    print("OBSERVED_SIZE=%d" % comm.size())
    print("GAUGE_RESULT=" + json.dumps(result))
