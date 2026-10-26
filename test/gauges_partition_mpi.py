#!/usr/bin/env python
"""Runs point, line and line integral gauges on a partitioned mesh and prints
what they measure as JSON.

Run by ``test_gauges_mpi.py`` under ``mpiexec`` with several numbers of ranks.
The field is u = x + 10 y at t = 0, so every value has a closed form and the
answer must not depend on how the mesh is cut: a gauge point on the cut, a line
along it, and a line whose segments are found on one rank while the points they
end at belong to another all have to come out the same as in serial.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from proteus import Comm

comm = Comm.init()

import test_gauges as T
from proteus.Gauges import PointGauges, LineGauges, LineIntegralGauges


def run(gauges, name, nd=2, **kw):
    filename = "gauges_partition_%s.csv" % name
    T.silent_rm(filename)
    comm.barrier()
    gauge = gauges(fileName=filename)
    T.run_gauge(gauge, [0.0], nd=nd, **kw)
    names, data = T.parse_gauge_output(filename)
    T.delete_file(filename)
    return {"names": names, "values": np.atleast_2d(data)[0, 1:].tolist()}


result = {"size": comm.size()}
# the points sit on and next to the cuts of a 10 x 10 mesh, the last two a
# roundoff outside the domain
points = ((0.37, 0.22, 0), (0.5, 0.5, 0), (0.25, 0.75, 0), (0.5, -1e-12, 0), (1 + 1e-12, 0.5, 0))
result["points"] = run(lambda fileName: PointGauges(gauges=((("u0",), points),), fileName=fileName),
                       "points", total_nodes=100)
line = ((-1, 0.25, 0), (2, 0.25, 0))
result["line"] = run(lambda fileName: LineGauges(gauges=((("u0",), (line,)),), fileName=fileName),
                     "line", total_nodes=100)
result["integral"] = run(lambda fileName: LineIntegralGauges(gauges=((("u0",), (line,)),), fileName=fileName),
                         "integral", total_nodes=100)
# a vertical line, so the cuts run across it
vertical = ((0.31, -0.5, 0), (0.31, 1.5, 0))
result["vertical_integral"] = run(lambda fileName: LineIntegralGauges(gauges=((("u0",), (vertical,)),), fileName=fileName),
                                  "vertical", total_nodes=100)
# the Turek cylinder mesh: a hole, with both ends a roundoff outside
result["turek_line"] = run(lambda fileName: LineGauges(gauges=((("u0",), (T.TUREK_LINE,)),), fileName=fileName),
                           "turek_line", mesh_builder=T.build_turek_cylinder_mesh)
result["turek_integral"] = run(lambda fileName: LineIntegralGauges(gauges=((("u0",), (T.TUREK_LINE,)),), fileName=fileName),
                               "turek_integral", mesh_builder=T.build_turek_cylinder_mesh)
# a 3D line along the edges of the tetrahedra: the points a rank finds as the start
# of its segments are the ends of segments found on another rank
diagonal = ((0, 0, 0), (1, 1, 1))
result["diagonal_3d"] = run(lambda fileName: LineGauges(gauges=((("u0",), (diagonal,)),), fileName=fileName),
                            "diagonal", nd=3, total_nodes=2048)
if comm.rank() == 0:
    print("OBSERVED_SIZE=%d" % comm.size())
    print("GAUGE_RESULT=" + json.dumps(result))
