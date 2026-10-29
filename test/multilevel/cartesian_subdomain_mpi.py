"""Helper run under mpiexec by test_cartesian_subdomain.py; not collected.

Builds one rank's piece of a regular grid twice:

* Phase 1's way -- the global mesh from the rectangular-grid generator,
  partitioned by partitionMesh with MeshTools.cartesianNodeOwners for a rank
  grid -- and
* Phase 2's way -- MeshTools.cartesianSubdomain, which generates the piece
  without the global mesh --

and compares every array the two produce, exactly: the offsets and
subdomain-to-global maps, the old-to-new permutations (arrays on one side,
functions on the other), the global counts, and every array of the subdomain
mesh, floating-point ones included.

Usage: cartesian_subdomain_mpi.py KIND NNX NNY [NNZ] [PX PY [PZ]] [level=L]

KIND is quad, hex, tet, or tri0, tri1 or tri2 (triangles split with that
triangleFlag). The rank grid defaults to MeshTools.cartesianRankGrid. With
level=L both sides are the grid refined L times (NN.. stay the coarsest
counts): the global one from the generator at that resolution, partitioned with
the scaled bricks cartesianNodeOwners(level=L) gives, against
cartesianSubdomain(level=L), as a hierarchy generated in pieces builds its
levels.

Prints ``RESULT kind=.. nn=.. grid=.. ranks=.. mismatches=<arrays that differ,
summed over ranks> digest=<sha256 of the generated arrays> petsc=<0|1>``.
"""
import mpi_failfast  # noqa: F401  (an uncaught exception aborts every rank)
import hashlib
import sys

import numpy as np
from proteus import Comm
comm = Comm.init()
from mpi4py import MPI
from proteus import MeshTools, cmeshTools, cpartitioning

rank, size = comm.rank(), comm.size()
mpi = comm.comm.tompi4py()
kind = sys.argv[1]
triangleFlag = 1
if kind.startswith('tri'):
    kind, triangleFlag = 'tri', int(kind[3:])
dim = 2 if kind in ('quad', 'tri') else 3
nn = tuple(int(a) for a in sys.argv[2:2 + dim]) + (1,) * (3 - dim)
options = dict(a.split('=') for a in sys.argv[2:] if '=' in a)
level = int(options.get('level', 0))
rest = [int(a) for a in sys.argv[2 + dim:] if '=' not in a]
grid = tuple(rest) + (1,) * (3 - len(rest)) if rest else MeshTools.cartesianRankGrid(nn[0], nn[1], nn[2], size)
L = (1.0, 0.7, 1.3)
#the refined grid's node counts
nnL = tuple(((n - 1) << level) + 1 if d < dim else 1 for d, n in enumerate(nn))

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
MAPS = ('elementOffsets_subdomain_owned', 'elementNumbering_subdomain2global',
        'nodeOffsets_subdomain_owned', 'nodeNumbering_subdomain2global',
        'elementBoundaryOffsets_subdomain_owned', 'elementBoundaryNumbering_subdomain2global',
        'edgeOffsets_subdomain_owned', 'edgeNumbering_subdomain2global')


def arraysOf(mesh, old2new):
    arrays = {name: np.array(getattr(mesh, name), copy=True) for name in MAPS}
    arrays['nodeNumbering_global_old2new'], arrays['elementNumbering_global_old2new'] = old2new
    for name in ('nNodes_global', 'nElements_global', 'nElementBoundaries_global', 'nEdges_global'):
        arrays['global.' + name] = np.array(getattr(mesh, name))
    for name, value in zip(INTERFACE, cmeshTools.buildPythonMeshInterface(mesh.subdomainMesh.cmesh)):
        if value is not None and name not in UNSET:
            arrays['subdomain.' + name] = np.array(value, copy=True)
    return arrays


#Phase 1: the global mesh, partitioned with Cartesian owners
glob = {'quad': MeshTools.QuadrilateralMesh, 'hex': MeshTools.HexahedralMesh,
        'tri': MeshTools.TriangularMesh, 'tet': MeshTools.TetrahedralMesh}[kind]()
if kind == 'quad':
    glob.generateQuadrilateralMeshFromRectangularGrid(nnL[0], nnL[1], L[0], L[1])
elif kind == 'hex':
    glob.generateHexahedralMeshFromRectangularGrid(nnL[0], nnL[1], nnL[2], L[0], L[1], L[2])
elif kind == 'tri':
    glob.generateTriangularMeshFromRectangularGrid(nnL[0], nnL[1], L[0], L[1], triangleFlag=triangleFlag)
else:
    glob.generateTetrahedralMeshFromRectangularGrid(nnL[0], nnL[1], nnL[2], L[0], L[1], L[2])
owners = MeshTools.cartesianNodeOwners(nn[0], nn[1], nn[2], rankGrid=grid, fastest='y' if kind == 'quad' else 'x',
                                       level=level)
glob.partitionMesh(nLayersOfOverlap=1, parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.node,
                   nodeOwners=owners)
reference = arraysOf(glob, (np.asarray(glob.nodeNumbering_global_old2new),
                            np.asarray(glob.elementNumbering_global_old2new)))

#Phase 2: generated
gen = MeshTools.cartesianSubdomain(kind, nn[:dim], L[:dim], rankGrid=grid, triangleFlag=triangleFlag, level=level)
generated = arraysOf(gen, (gen.nodeOld2New(np.arange(gen.nNodes_global)),
                           gen.elementOld2New(np.arange(gen.nElements_global))))

mismatches = 0
for name in sorted(set(reference) | set(generated)):
    a, b = reference.get(name), generated.get(name)
    if a is None or b is None or a.shape != b.shape or not np.array_equal(a, b):
        mismatches += 1
        detail = ''
        if a is not None and b is not None and a.shape == b.shape:
            bad = np.argwhere(a != b)
            detail = ' first at %s: %s vs %s' % (bad[0].tolist(), a[tuple(bad[0])], b[tuple(bad[0])])
        elif a is not None and b is not None:
            detail = ' shapes %s vs %s' % (a.shape, b.shape)
        print("rank %d: %s differs%s" % (rank, name, detail))
        sys.stdout.flush()
mismatches = mpi.allreduce(mismatches, op=MPI.SUM)
h = hashlib.sha256()
for name in sorted(generated):
    a = generated[name]
    h.update(name.encode() + str(a.shape).encode() + np.ascontiguousarray(a).tobytes())
digests = mpi.gather(h.hexdigest(), root=0)
petsc = mpi.allreduce(int(cpartitioning.petscInitialized()), op=MPI.MAX)
if rank == 0:
    print("RESULT kind=%s nn=%s grid=%s level=%d ranks=%d mismatches=%d digest=%s petsc=%d"
          % (sys.argv[1], 'x'.join(map(str, nn[:dim])), 'x'.join(map(str, grid)), level, size, mismatches,
             hashlib.sha256(''.join(digests).encode()).hexdigest(), petsc))
    sys.stdout.flush()
