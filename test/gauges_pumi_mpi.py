#!/usr/bin/env python
"""Runs point, line and line integral gauges on a PUMI mesh and prints what
they measure as JSON.

Run by ``test_gauges_pumi_mpi.py`` under ``mpiexec``.  The mesh is loaded
through PUMI, so its partition is the one the model files were written with
(``squarepar``/``cubepar`` have one part per rank, ``square``/``cube`` a single
part), and the element partition is what ``convertFromPUMI`` sees.  The field
is u = x + 10 y in 2D (u = x + 10 y + 100 z in 3D) at t = 0, so every value has
a closed form and the answer must not depend on the partition.
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


def build_pumi_mesh(name, nd):
    """The PUMI model ``name``, converted and partitioned by elements the way
    ``Domain`` builds a PUMI mesh for a run.  PUMI appends each rank's part
    index to ``name``, so ``squarepar.smb`` is read as ``squarepar0.smb`` on
    rank 0 and ``squarepar1.smb`` on rank 1."""
    base = os.path.join(PUMI_DIR, name)
    domain = Domain.PUMIDomain(dim=nd, manager=MeshAdapt.AdaptManager())
    domain.AdaptManager.PUMIAdapter.loadModelAndMesh((base + ".dmg").encode("ascii"),
                                                     (base + ".smb").encode("ascii"))
    if nd == 3:
        mesh = MeshTools.TetrahedralMesh()
        mlMesh = MeshTools.MultilevelTetrahedralMesh
    else:
        mesh = MeshTools.TriangularMesh()
        mlMesh = MeshTools.MultilevelTriangularMesh
    mesh.cmesh = cmeshTools.CMesh()
    mesh.convertFromPUMI(domain, domain.AdaptManager.PUMIAdapter, domain.faceList,
                         domain.regList, parallel=comm.size() > 1, dim=nd)
    ml = mlMesh(0, 0, 0, skipInit=True, nLayersOfOverlap=0,
                parallelPartitioningType=MeshParallelPartitioningTypes.element)
    ml.generatePartitionedMeshFromPUMI(mesh, 1, nLayersOfOverlap=0)
    return ml


def run(gauges, name, nd=2, **kw):
    filename = "gauges_pumi_%s.csv" % name
    T.silent_rm(filename)
    comm.barrier()
    gauge = gauges(fileName=filename)
    T.run_gauge(gauge, [0.0], nd=nd, **kw)
    names, data = T.parse_gauge_output(filename)
    T.delete_file(filename)
    return {"names": names, "values": np.atleast_2d(data)[0, 1:].tolist()}


result = {"size": comm.size()}

# the unit square in two parts: points on and next to the cuts, the last two a roundoff outside
square = lambda: build_pumi_mesh("squarepar", 2)
points = ((0.37, 0.22, 0), (0.5, 0.5, 0), (0.25, 0.75, 0), (0.5, -1e-12, 0), (1 + 1e-12, 0.5, 0))
result["points"] = run(lambda fileName: PointGauges(gauges=((("u0",), points),), fileName=fileName),
                       "points", mesh_builder=square)
line = ((-1, 0.25, 0), (2, 0.25, 0))
result["line"] = run(lambda fileName: LineGauges(gauges=((("u0",), (line,)),), fileName=fileName),
                     "line", mesh_builder=square)
result["integral"] = run(lambda fileName: LineIntegralGauges(gauges=((("u0",), (line,)),), fileName=fileName),
                         "integral", mesh_builder=square)
vertical = ((0.31, -0.5, 0), (0.31, 1.5, 0))
result["vertical_integral"] = run(
    lambda fileName: LineIntegralGauges(gauges=((("u0",), (vertical,)),), fileName=fileName),
    "vertical", mesh_builder=square)

# the unit cube in two parts: a diagonal through both
cube = lambda: build_pumi_mesh("cubepar", 3)
diagonal = ((0, 0, 0), (1, 1, 1))
result["diagonal_3d"] = run(lambda fileName: LineGauges(gauges=((("u0",), (diagonal,)),), fileName=fileName),
                            "diagonal", nd=3, mesh_builder=cube)

if comm.rank() == 0:
    print("OBSERVED_SIZE=%d" % comm.size())
    print("GAUGE_RESULT=" + json.dumps(result))
