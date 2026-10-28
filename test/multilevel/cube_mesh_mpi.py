"""Helper run under mpiexec by test_parallel_cube_meshes.py; not collected.

Partitions a regular quadrilateral or hexahedral mesh by nodes and checks the
subdomain mesh on every rank against what a regular grid must look like.

The check is geometric rather than numerical on purpose. A subdomain element
that points at the wrong node still has some nodes, so counts and indices look
plausible; but in a regular grid every element has the same extent along each
axis, so one that does not has been handed a node from somewhere else.
"""
import sys

import numpy as np
from proteus import Comm
comm = Comm.init()
from mpi4py import MPI
from proteus import MeshTools

rank, size = comm.rank(), comm.size()
mpi = comm.comm.tompi4py()
kind = sys.argv[1]
n = int(sys.argv[2])
# 'global' builds the global mesh and partitions it (partitionNodes); the default
# generates each rank's piece directly. Both must give a valid partition.
if len(sys.argv) > 3 and sys.argv[3] == 'global':
    MeshTools.MultilevelMesh.regularGrid = 'global'
part = MeshTools.MeshParallelPartitioningTypes.node
L = 2.0
if kind == 'hex':
    ml = MeshTools.MultilevelHexahedralMesh(n, n, n, Lx=L, Ly=L, Lz=L, refinementLevels=1,
                                            nLayersOfOverlap=1, parallelPartitioningType=part)
    dim = 3
else:
    ml = MeshTools.MultilevelQuadrilateralMesh(n, n, 1, Lx=L, Ly=L, refinementLevels=1,
                                               nLayersOfOverlap=1, parallelPartitioningType=part)
    dim = 2
h = L / (n - 1)
mesh = ml.meshList[0]
sd = mesh.subdomainMesh
x = np.asarray(sd.nodeArray)
e = np.asarray(sd.elementNodesArray)

problems = []
if e.size and (e.min() < 0 or e.max() >= sd.nNodes_global):
    problems.append("element node index out of range")
extent = x[e].max(axis=1) - x[e].min(axis=1)           # (nElements, 3)
bad = np.abs(extent[:, :dim] - h) > 1e-12
if bad.any():
    problems.append("%d of %d elements are not a cell of the grid (extent %s, expected %g)"
                    % (bad.any(axis=1).sum(), e.shape[0], extent[bad.any(axis=1)][0, :dim], h))
used = np.unique(e)
if used.size != sd.nNodes_global:
    problems.append("subdomain holds %d nodes but its elements use %d" % (sd.nNodes_global, used.size))
# every node of the subdomain lies on the grid
grid = np.round(x[:, :dim] / h)
if np.abs(x[:, :dim] - grid * h).max() > 1e-12:
    problems.append("a node is off the grid")
# the owned nodes partition the global node set
offsets = np.asarray(mesh.nodeOffsets_subdomain_owned)
if offsets[-1] != mesh.nNodes_global:
    problems.append("owned node counts sum to %d, not %d" % (offsets[-1], mesh.nNodes_global))

# An owned element boundary must have every element that contains it on its owner:
# the owner assembles that boundary's row of a space whose dofs sit on boundaries
# (Q2's edge dofs, the non-conforming P1 dofs), and a row missing an element is
# wrong. For quadrilaterals the partitioner used the simplex rule to choose an
# element boundary's owner, which handed edges to ranks holding neither end point.
gmesh = mesh.globalMesh if getattr(mesh, 'globalMesh', None) is not None else mesh
ebOff = np.asarray(mesh.elementBoundaryOffsets_subdomain_owned)
ebNumber = np.asarray(mesh.elementBoundaryNumbering_subdomain2global)
sharedCount = {int(ebNumber[i]): int((np.asarray(sd.elementBoundaryElementsArray[i]) >= 0).sum())
               for i in range(ebNumber.size)}
bestCount = {}
for view in mpi.allgather(sharedCount):
    for key, c in view.items():
        bestCount[key] = max(bestCount.get(key, 0), c)
nOwnedEb = int(ebOff[rank + 1] - ebOff[rank])
short = [int(ebNumber[i]) for i in range(nOwnedEb) if sharedCount[int(ebNumber[i])] < bestCount[int(ebNumber[i])]]
if short:
    problems.append("%d owned element boundaries lack an element that contains them on their owner (e.g. %s)"
                    % (len(short), short[:3]))

allproblems = mpi.allgather(problems)
if rank == 0:
    flat = [(r, p) for r, ps in enumerate(allproblems) for p in ps]
    if flat:
        for r, p in flat:
            print("CUBE_MESH_PROBLEM rank %d: %s" % (r, p))
    else:
        print("CUBE_MESH_OK kind=%s n=%d ranks=%d" % (kind, n, size))
    sys.stdout.flush()
