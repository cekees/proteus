"""Time and memory of regular-mesh generation and partitioning; run under mpiexec, not collected.

The baseline of the Cartesian partitioner campaign (CARTESIAN_PARTITIONER_CAMPAIGN.md,
Phase 0) and its repeat on the new path (Phase 6). Every rank builds the
global mesh and partitions it, as proteus does today, and this reports the
wall-clock time of each step (the maximum over ranks) and the peak resident
memory per rank.

Usage: partition_scaling_mpi.py KIND NNX NNY NNZ LEVELS [graph|cartesian|generated]

KIND     tri, tet, quad or hex; NNX NNY NNZ are coarsest-level node counts
         (NNZ is ignored in 2D)
LEVELS   1: generate one mesh and partition it by nodes, timed separately;
         >1: the MultilevelXMesh constructor, with generation, hierarchy
         construction and partitioning of the levels timed separately
owners   graph (default): the graph partitioner; cartesian: node bricks from
         MeshTools.cartesianNodeOwners on the coarsest level; generated: no
         global mesh, MeshTools.cartesianSubdomain (LEVELS 1), timed as part

Set PROTEUS_PARTITION_TIMING=1 to also get the partitioner's own stage timings
(PARTITION_TIMING lines, one per partitioned mesh).

Prints ``RESULT kind=.. ranks=.. nodes=<finest level> elements=.. gen=..
hier=.. part=.. total=.. rss_max_mb=.. rss_min_mb=..``.
"""
import resource
import sys
import time

import numpy as np
from proteus import Comm
comm = Comm.init()
from mpi4py import MPI
from proteus import MeshTools, cmeshTools

rank, size = comm.rank(), comm.size()
mpi = comm.comm.tompi4py()
kind = sys.argv[1]
nnx, nny, nnz = (int(a) for a in sys.argv[2:5])
nLevels = int(sys.argv[5])
ownersKind = sys.argv[6] if len(sys.argv) > 6 else 'graph'
if kind in ('tri', 'quad'):
    nnz = 1
L = (1.0, 1.0, 1.0)
fastest = 'y' if kind == 'quad' else 'x'
spent = {'gen': 0.0, 'hier': 0.0, 'part': 0.0}


def timed(key, f):
    def wrapper(*args, **kwargs):
        mpi.Barrier()
        start = time.perf_counter()
        try:
            return f(*args, **kwargs)
        finally:
            spent[key] += time.perf_counter() - start
    return wrapper


def peakRSS_mb():
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024.0 if sys.platform != 'darwin' else kb / 1024.0 ** 2


if ownersKind == 'cartesian':
    MeshTools.MultilevelMesh.coarsestLevelOwners = staticmethod(
        lambda mesh, partitioningType: MeshTools.cartesianNodeOwners(nnx, nny, nnz, fastest=fastest))
start = time.perf_counter()
if ownersKind == 'generated':
    assert nLevels == 1, "generated: one level"
    flat = kind in ('quad', 'tri')
    finest = timed('part', MeshTools.cartesianSubdomain)(
        kind, (nnx, nny, nnz)[:2 if flat else 3], L[:2 if flat else 3])
elif nLevels == 1:
    mesh = {'tri': MeshTools.TriangularMesh, 'quad': MeshTools.QuadrilateralMesh,
            'tet': MeshTools.TetrahedralMesh, 'hex': MeshTools.HexahedralMesh}[kind]()
    gen = {'tri': lambda: mesh.generateTriangularMeshFromRectangularGrid(nnx, nny, L[0], L[1]),
           'quad': lambda: mesh.generateQuadrilateralMeshFromRectangularGrid(nnx, nny, L[0], L[1]),
           'tet': lambda: mesh.generateTetrahedralMeshFromRectangularGrid(nnx, nny, nnz, *L),
           'hex': lambda: mesh.generateHexahedralMeshFromRectangularGrid(nnx, nny, nnz, *L)}[kind]
    timed('gen', gen)()
    owners = (MeshTools.cartesianNodeOwners(nnx, nny, nnz, fastest=fastest)
              if ownersKind == 'cartesian' else None)
    timed('part', mesh.partitionMesh)(nLayersOfOverlap=1,
                                      parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.node,
                                      nodeOwners=owners)
    finest = mesh
else:
    #the constructors look these up at call time, so wrapping them times each step
    for name in ('generateTriangularMeshFromRectangularGrid', 'generateTetrahedralMeshFromRectangularGrid',
                 'generateQuadrilateralMeshFromRectangularGrid', 'generateHexahedralMeshFromRectangularGrid'):
        setattr(cmeshTools, name, timed('gen', getattr(cmeshTools, name)))
    cmeshTools.CMultilevelMesh = timed('hier', cmeshTools.CMultilevelMesh)
    MeshTools.MultilevelMesh.partitionLevels = timed('part', MeshTools.MultilevelMesh.partitionLevels)
    if hasattr(MeshTools.MultilevelMesh, 'generateStructuredHierarchy'):
        #regular quad/hex hierarchies generate every level: count it as generation
        #and hierarchy together, less the partitioning inside it
        MeshTools.MultilevelMesh.generateStructuredHierarchy = timed(
            'hier', MeshTools.MultilevelMesh.generateStructuredHierarchy)
    part = MeshTools.MeshParallelPartitioningTypes.node
    if kind == 'tri':
        ml = MeshTools.MultilevelTriangularMesh(nnx, nny, 1, Lx=L[0], Ly=L[1], Lz=L[2], refinementLevels=nLevels,
                                                nLayersOfOverlap=1, parallelPartitioningType=part)
    elif kind == 'quad':
        ml = MeshTools.MultilevelQuadrilateralMesh(nnx, nny, 1, Lx=L[0], Ly=L[1], Lz=L[2], refinementLevels=nLevels,
                                                   nLayersOfOverlap=1, parallelPartitioningType=part)
    elif kind == 'tet':
        ml = MeshTools.MultilevelTetrahedralMesh(nnx, nny, nnz, Lx=L[0], Ly=L[1], Lz=L[2], refinementLevels=nLevels,
                                                 nLayersOfOverlap=1, parallelPartitioningType=part)
    else:
        ml = MeshTools.MultilevelHexahedralMesh(nnx, nny, nnz, Lx=L[0], Ly=L[1], Lz=L[2], refinementLevels=nLevels,
                                                nLayersOfOverlap=1, parallelPartitioningType=part)
    finest = ml.meshList[-1]
    #the structured hierarchy partitions inside generateStructuredHierarchy
    if spent['hier'] > spent['part'] and kind in ('quad', 'hex'):
        spent['hier'] -= spent['part']
mpi.Barrier()
total = time.perf_counter() - start
worst = {k: mpi.allreduce(v, op=MPI.MAX) for k, v in spent.items()}
rss = peakRSS_mb()
rssMax = mpi.allreduce(rss, op=MPI.MAX)
rssMin = mpi.allreduce(rss, op=MPI.MIN)
if rank == 0:
    print("RESULT kind=%s owners=%s levels=%d ranks=%d nodes=%d elements=%d gen=%.3f hier=%.3f part=%.3f "
          "total=%.3f rss_max_mb=%.1f rss_min_mb=%.1f"
          % (kind, ownersKind, nLevels, size, finest.nNodes_global, finest.nElements_global,
             worst['gen'], worst['hier'], worst['part'], total, rssMax, rssMin))
    sys.stdout.flush()
