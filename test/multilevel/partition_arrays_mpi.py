"""Helper run under mpiexec by test_partition_arrays.py; not collected.

Partitions one regular mesh with prescribed owners and digests every array
the partitioner produces on every rank: the eight offset/numbering arrays
partitionNodes and partitionElements return, the old-to-new permutations, the
subdomain mesh's connectivity and material arrays, and -- for element
partitioning, which renumbers the global mesh in place -- the global mesh's.
Two partitioners that produce the same digest produced the same arrays, entry
for entry, on every rank.

Usage: partition_arrays_mpi.py KIND BY OWNERS [DUMPDIR]

KIND    tri, tet, quad or hex (odd node counts, so bricks are uneven)
BY      node or element
OWNERS  brick    a px x py x pz grid of index bricks, computed here from the
                 node coordinates (independent of MeshTools.cartesianNodeOwners)
        scatter  a deterministic hash of the index
        empty    the same with rank 1 given nothing when there are more than two
                 ranks (the partitioner before 2026-10 crashed on this)
        closedup the empty partition of one more rank, without its empty rank:
                 rank r here is rank r+1 there for r > 0, and the global
                 numbering is the same, so every array must be too
        graph    none: the graph partitioner (MatPartitioning) decides; it is
                 not reproducible from run to run, so neither is this digest
        graphreplay  graph, then a fresh mesh partitioned with the owners the
                 graph partitioner chose; mismatches counts the arrays that
                 differ between the two
DUMPDIR if given, each rank also writes its arrays to DUMPDIR/rank<r>.npz

Prints ``RESULT ... digest=<sha256 of the integer arrays, all ranks>
fulldigest=<same, floating-point arrays included> petsc=<0|1> mismatches=<n>``,
where petsc says whether the PETSc C library had been initialized by the end
(-1: this build cannot tell).
"""
import mpi_failfast  # noqa: F401  (an uncaught exception aborts every rank)
import hashlib
import os
import sys

import numpy as np
from proteus import Comm
comm = Comm.init()
from mpi4py import MPI
from proteus import MeshTools, cmeshTools, cpartitioning

rank, size = comm.rank(), comm.size()
mpi = comm.comm.tompi4py()
kind, by, ownersKind = sys.argv[1], sys.argv[2], sys.argv[3]
dumpdir = sys.argv[4] if len(sys.argv) > 4 else None

NN = {'tri': (9, 7, 1), 'quad': (9, 7, 1), 'tet': (9, 7, 6), 'hex': (9, 7, 6)}[kind]
L = (1.0, 0.7, 1.3)


def makeMesh():
    mesh = {'tri': MeshTools.TriangularMesh, 'quad': MeshTools.QuadrilateralMesh,
            'tet': MeshTools.TetrahedralMesh, 'hex': MeshTools.HexahedralMesh}[kind]()
    if kind == 'tri':
        mesh.generateTriangularMeshFromRectangularGrid(NN[0], NN[1], L[0], L[1])
    elif kind == 'quad':
        mesh.generateQuadrilateralMeshFromRectangularGrid(NN[0], NN[1], L[0], L[1])
    elif kind == 'tet':
        mesh.generateTetrahedralMeshFromRectangularGrid(NN[0], NN[1], NN[2], L[0], L[1], L[2])
    else:
        mesh.generateHexahedralMeshFromRectangularGrid(NN[0], NN[1], NN[2], L[0], L[1], L[2])
    return mesh


mesh = makeMesh()


def rankGrid(n):
    """px, py, pz with px*py*pz == n: the largest factor along x, the rest along y."""
    dim = 2 if NN[2] == 1 else 3
    px = n
    for f in range(int(n ** 0.5), 0, -1):
        if n % f == 0:
            px = n // f
            break
    py = n // px
    pz = 1
    if dim == 3 and py > 1:
        for f in range(int(py ** 0.5), 0, -1):
            if py % f == 0:
                pz = f
                py //= f
                break
    return px, py, pz


def chunk(i, nn, p):
    """The chunk of 0..nn-1 holding index i when split into p near-equal chunks."""
    starts = np.cumsum([0] + [nn // p + (nn % p > a) for a in range(p)])
    return np.searchsorted(starts[1:], i, side='right')


def brickOwners(points, cells):
    """Node bricks, or for elements (cells=True) cell bricks: either way no rank is empty."""
    grid = rankGrid(size)
    nn = [NN[d] - 1 if cells and NN[d] > 1 else NN[d] for d in range(3)]
    #an element centroid lies inside its cell: floor gives the cell's lower node
    idx = [np.floor(points[:, d] / (L[d] / (NN[d] - 1)) + 1.0e-8).astype(int) if NN[d] > 1
           else np.zeros(points.shape[0], int) for d in range(3)]
    a, b, c = (chunk(idx[d], nn[d], grid[d]) for d in range(3))
    return (a + grid[0] * (b + grid[1] * c)).astype('i')


def scatterOwners(n, empty, ranks=size):
    i = np.arange(n, dtype=np.int64)
    owners = ((i * 2654435761) >> 7) % ranks
    if empty and ranks > 2:
        owners[owners == 1] = 0
    return owners.astype('i')


def closedUpOwners(n):
    """The 'empty' partition for one more rank, with the empty rank removed."""
    owners = scatterOwners(n, True, size + 1)
    owners[owners > 1] -= 1
    return owners


if by == 'node':
    part = MeshTools.MeshParallelPartitioningTypes.node
    points = np.asarray(mesh.nodeArray)
    n = mesh.nNodes_global
else:
    part = MeshTools.MeshParallelPartitioningTypes.element
    #not every generator fills elementBarycentersArray
    nodes = np.asarray(mesh.nodeArray)
    points = nodes[np.asarray(mesh.elementNodesArray)].mean(axis=1)
    n = mesh.nElements_global
owners = {'brick': lambda: brickOwners(points, by == 'element'),
          'scatter': lambda: scatterOwners(n, False), 'empty': lambda: scatterOwners(n, True),
          'closedup': lambda: closedUpOwners(n),
          'graph': lambda: None, 'graphreplay': lambda: None}[ownersKind]()

INTERFACE = ('nElements_global nNodes_global nNodes_element nNodes_elementBoundary '
             'nElementBoundaries_element nElementBoundaries_global '
             'nInteriorElementBoundaries_global nExteriorElementBoundaries_global '
             'max_nElements_node nEdges_global max_nNodeNeighbors_node elementNodesArray '
             'nodeElementsArray nodeElementOffsets elementNeighborsArray elementBoundariesArray '
             'elementBoundaryNodesArray elementBoundaryElementsArray '
             'elementBoundaryLocalElementBoundariesArray interiorElementBoundariesArray '
             'exteriorElementBoundariesArray edgeNodesArray nodeStarArray nodeStarOffsets '
             'elementMaterialTypes elementBoundaryMaterialTypes nodeMaterialTypes nodeArray '
             'nx ny nz px py pz elementIJK weights U_KNOT V_KNOT W_KNOT elementDiametersArray '
             'elementInnerDiametersArray elementBoundaryDiametersArray elementBarycentersArray '
             'elementBoundaryBarycentersArray nodeDiametersArray nodeSupportArray h hMin '
             'sigmaMax volume').split()


#computeGeometricInfo_hexahedron never fills these: they are uninitialized memory
UNSET = {'elementBarycentersArray', 'elementBoundaryBarycentersArray'} if kind == 'hex' else set()


def meshArrays(prefix, cmesh):
    out = {}
    for name, value in zip(INTERFACE, cmeshTools.buildPythonMeshInterface(cmesh)):
        if value is None or name in UNSET:
            continue
        out[prefix + name] = np.array(value, copy=True)
    return out


def partition(mesh, owners):
    """Partition mesh with the given owners (None: the graph partitioner) and
    return every array that results, copied."""
    mesh.partitionMesh(nLayersOfOverlap=1, parallelPartitioningType=part,
                       nodeOwners=owners if by == 'node' else None,
                       elementOwners=owners if by == 'element' else None)
    arrays = {}
    for name in ('elementOffsets_subdomain_owned', 'elementNumbering_subdomain2global',
                 'nodeOffsets_subdomain_owned', 'nodeNumbering_subdomain2global',
                 'elementBoundaryOffsets_subdomain_owned', 'elementBoundaryNumbering_subdomain2global',
                 'edgeOffsets_subdomain_owned', 'edgeNumbering_subdomain2global',
                 'nodeNumbering_global_old2new', 'elementNumbering_global_old2new'):
        #element partitioning does not keep the node permutation
        if getattr(mesh, name, None) is not None:
            arrays[name] = np.array(getattr(mesh, name), copy=True)
    arrays.update(meshArrays('subdomain.', mesh.subdomainMesh.cmesh))
    if by == 'element':
        arrays.update(meshArrays('global.', mesh.cmesh))
    return arrays


arrays = partition(mesh, owners)
mismatches = 0
if ownersKind == 'graphreplay':
    #The graph partitioner is not reproducible from run to run (ParMETIS through
    #PETSc), so its output cannot be compared between builds. Instead read the
    #partition it chose back out of the numbering and partition a fresh mesh with
    #those owners: everything must come out the same as from the graph partitioner.
    entity = 'node' if by == 'node' else 'element'
    old2new = arrays[entity + 'Numbering_global_old2new']
    offsets = arrays[entity + 'Offsets_subdomain_owned']
    chosen = np.searchsorted(offsets[1:], old2new, side='right').astype('i')
    graphArrays = arrays
    arrays = partition(makeMesh(), chosen)
    for name in sorted(set(arrays) | set(graphArrays)):
        a, b = arrays.get(name), graphArrays.get(name)
        if a is None or b is None or a.shape != b.shape or not np.array_equal(a, b):
            mismatches += 1
            print("rank %d: %s differs between the graph partitioner and its replay" % (rank, name))
mismatches = mpi.allreduce(mismatches, op=MPI.SUM)


def digest(include_float):
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = arrays[name]
        if a.dtype.kind == 'f' and not include_float:
            continue
        if a.dtype.kind in 'iu':
            a = a.astype(np.int64)
        h.update(name.encode())
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


if dumpdir:
    os.makedirs(dumpdir, exist_ok=True)
    np.savez(os.path.join(dumpdir, 'rank%d.npz' % rank), **arrays)
digests = mpi.gather((digest(False), digest(True)), root=0)
petsc = int(cpartitioning.petscInitialized()) if hasattr(cpartitioning, 'petscInitialized') else -1
petsc = mpi.allreduce(petsc, op=MPI.MAX)
if rank == 0:
    combine = lambda k: hashlib.sha256(''.join(d[k] for d in digests).encode()).hexdigest()
    print("RESULT kind=%s by=%s owners=%s ranks=%d digest=%s fulldigest=%s petsc=%d mismatches=%d"
          % (kind, by, ownersKind, size, combine(0), combine(1), petsc, mismatches))
    sys.stdout.flush()
