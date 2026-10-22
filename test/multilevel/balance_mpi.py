"""Helper run under mpiexec by test_partition_balance.py; not collected.

Builds a regular multilevel mesh -- generated in pieces, or the global hierarchy partitioned
(graph partitioner, or Cartesian owners) -- and prints MultilevelMesh.partitionBalance for
every level: max/mean over ranks of owned nodes and of subdomain elements, the surface cost
(largest ghost/owned node ratio) and the subdomain size.

Usage: balance_mpi.py KIND NN LEVELS [global|generated] [unweighted|rcb|independent] [alternate] [element]

With element partitioning L<l>.nodes is the owned *elements* and L<l>.ghost the ghost-element
fraction.

unweighted   the graph partitioner cuts the coarsest level counting nodes (before the weighting)
rcb          recursive coordinate bisection of the coarsest level, weighted by descendants
independent  every level partitioned on its own, the baseline for balance
"""
import sys

from proteus import Comm
comm = Comm.init()
import os
from proteus import MeshTools
if os.environ.get('PARTITIONER'):
    from petsc4py import PETSc
    for item in os.environ['PARTITIONER'].split(','):
        k, v = item.split('=')
        PETSc.Options()[k] = v

kind = sys.argv[1]
nn = int(sys.argv[2])
levels = int(sys.argv[3])
how = sys.argv[4] if len(sys.argv) > 4 else 'generated'
variant = sys.argv[5] if len(sys.argv) > 5 else 'weighted'
if 'alternate' in sys.argv[6:]:
    MeshTools.MultilevelMesh.parentRule = 'alternate'
if variant == 'unweighted':
    MeshTools.MultilevelMesh.weightedCoarsePartition = False
elif variant == 'rcb':
    MeshTools.MultilevelMesh.coarsePartitioner = 'rcb'
elif variant == 'independent':
    MeshTools.MultilevelMesh.levelConsistentPartitioning = False
by = 'element' if 'element' in sys.argv[6:] else 'node'
part = (MeshTools.MeshParallelPartitioningTypes.element if by == 'element'
        else MeshTools.MeshParallelPartitioningTypes.node)
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
    print("RESULT kind=%s nn=%d levels=%d ranks=%d how=%s variant=%s %s"
          % (kind, nn, levels, comm.size(), how, variant,
             ' '.join("L%d.nodes=%.4f L%d.elements=%.4f L%d.ghost=%.4f"
                      % (l, b['ownedElements'] if by == 'element' else b['nodes'], l, b['elements'],
                         l, b['ghostElementFraction'] if by == 'element' else b['ghostFraction'])
                      for l, b in enumerate(balance))))
    sys.stdout.flush()
