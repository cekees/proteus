"""Helper run under mpiexec by test_partition_balance.py; not collected.

Partitions a triangular mesh by nodes with the graph partitioner, with and without node
weights, and prints each rank's share of the nodes. Nodes in the left half (x < 0.5) weigh
`HEAVY` times the others, so a partitioner that honours the weights gives the rank that gets
the left half far fewer nodes.

Usage: weighted_partition_mpi.py HEAVY
"""
import sys

import numpy as np
from proteus import Comm
comm = Comm.init()
from proteus import MeshTools

heavy = int(sys.argv[1])
shares = {}
for label, weighted in (('plain', False), ('weighted', True)):
    mesh = MeshTools.TriangularMesh()
    mesh.generateTriangularMeshFromRectangularGrid(33, 33, 1.0, 1.0)
    x = np.asarray(mesh.nodeArray)[:, 0]
    weights = np.where(x < 0.5, heavy, 1).astype('i') if weighted else None
    mesh.partitionMesh(nLayersOfOverlap=1,
                       parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.node,
                       nodeWeights=weights)
    off = np.asarray(mesh.nodeOffsets_subdomain_owned)
    shares[label] = np.diff(off) / float(off[-1])
if comm.rank() == 0:
    print("RESULT ranks=%d plain_min=%.4f plain_max=%.4f weighted_min=%.4f weighted_max=%.4f"
          % (comm.size(), shares['plain'].min(), shares['plain'].max(),
             shares['weighted'].min(), shares['weighted'].max()))
    sys.stdout.flush()
