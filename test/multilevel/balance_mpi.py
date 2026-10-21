"""Helper run under mpiexec by test_partition_balance.py; not collected.

Builds a regular multilevel mesh -- generated in pieces, or the global hierarchy partitioned
(graph partitioner, or Cartesian owners) -- and prints MultilevelMesh.partitionBalance for
every level: max/mean over ranks of owned nodes and of subdomain elements, the surface cost
(largest ghost/owned node ratio) and the subdomain size.

Usage: balance_mpi.py KIND NN LEVELS [global|generated]
"""
import sys

from proteus import Comm
comm = Comm.init()
from proteus import MeshTools

kind = sys.argv[1]
nn = int(sys.argv[2])
levels = int(sys.argv[3])
how = sys.argv[4] if len(sys.argv) > 4 else 'generated'
part = MeshTools.MeshParallelPartitioningTypes.node
common = dict(refinementLevels=levels, nLayersOfOverlap=1, parallelPartitioningType=part,
              regularGrid=how)
if kind == 'hex':
    ml = MeshTools.MultilevelHexahedralMesh(nn, nn, nn, **common)
elif kind == 'tet':
    ml = MeshTools.MultilevelTetrahedralMesh(nn, nn, nn, **common)
elif kind == 'quad':
    ml = MeshTools.MultilevelQuadrilateralMesh(nn, nn, 1, **common)
else:
    ml = MeshTools.MultilevelTriangularMesh(nn, nn, 1, **common)
balance = ml.partitionBalance()
if comm.rank() == 0:
    print("RESULT kind=%s nn=%d levels=%d ranks=%d how=%s %s"
          % (kind, nn, levels, comm.size(), how,
             ' '.join("L%d.nodes=%.4f L%d.elements=%.4f L%d.ghost=%.4f"
                      % (l, b['nodes'], l, b['elements'], l, b['ghostFraction'])
                      for l, b in enumerate(balance))))
    sys.stdout.flush()
