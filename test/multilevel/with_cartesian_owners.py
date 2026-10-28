"""Run a helper with the coarsest mesh level cut into Cartesian bricks; not collected by pytest.

Usage: with_cartesian_owners.py HELPER [ARGS...]

Sets MultilevelMesh.coarsestLevelOwners so that the coarsest level of every
multilevel mesh the helper builds is cut into Cartesian bricks -- of nodes, or
of cells for element partitioning -- instead of by the graph partitioner; the
finer levels inherit (level-consistent partitioning). Nothing then needs PETSc to partition, and with
without_petsc4py.py in front of this, nothing initializes it: the line
``PETSC_INITIALIZED=<0|1>`` printed by rank 0 at the end says whether anything did.

The coarsest mesh must have its nodes on a regular grid; the bricks are
computed from the coordinates, so any node numbering will do. When the numbering
is one of the rectangular-grid generators' own, the analytic owners from
cartesianNodeOwners are checked against those bricks, so a wrong axis order
fails here rather than silently giving a poor partition.
"""
import os
import runpy
import sys

import numpy as np
from mpi4py import MPI
from proteus import Comm, MeshTools, cpartitioning


def ownersFromCoordinates(mesh, parallelPartitioningType):
    size = Comm.get().size()
    x = np.asarray(mesh.nodeArray)
    axes = [np.unique(x[:, d]) for d in range(3)]
    nn = [a.size for a in axes]
    index = [np.searchsorted(axes[d], x[:, d]) for d in range(3)]
    assert np.prod(nn) == mesh.nNodes_global, ("not a regular grid", nn, mesh.nNodes_global)
    grid = MeshTools.cartesianRankGrid(nn[0], nn[1], nn[2], size)
    if parallelPartitioningType != MeshTools.MeshParallelPartitioningTypes.node:
        #element bricks: the cell an element lies in, cut along each axis as the
        #nodes would be
        centroid = x[np.asarray(mesh.elementNodesArray)].mean(axis=1)
        cell = [np.searchsorted(axes[d], centroid[:, d]) - 1 if nn[d] > 1 else
                np.zeros(mesh.nElements_global, int) for d in range(3)]
        chunk = [np.searchsorted(MeshTools._cartesianChunkStarts(max(nn[d] - 1, 1), grid[d])[1:], cell[d],
                                 side='right') for d in range(3)]
        return (chunk[0] + grid[0] * (chunk[1] + grid[1] * chunk[2])).astype(np.intc)
    chunk = [np.searchsorted(MeshTools._cartesianChunkStarts(nn[d], grid[d])[1:], index[d], side='right')
             for d in range(3)]
    bricks = (chunk[0] + grid[0] * (chunk[1] + grid[1] * chunk[2])).astype(np.intc)
    #a generator's own numbering: the analytic owners must be these bricks
    xFastest = index[0] + nn[0] * (index[1] + nn[1] * index[2])
    yFastest = index[1] + nn[1] * index[0]
    for fastest, numbering in (('x', xFastest), ('y', yFastest)):
        if fastest == 'y' and nn[2] > 1:
            continue
        if np.array_equal(numbering, np.arange(mesh.nNodes_global)):
            owners = MeshTools.cartesianNodeOwners(nn[0], nn[1], nn[2], grid, fastest=fastest)
            assert np.array_equal(owners, bricks), "cartesianNodeOwners disagrees with the coordinates"
    return bricks


MeshTools.MultilevelMesh.coarsestLevelOwners = staticmethod(ownersFromCoordinates)
# a helper named without a directory is one of this directory's, wherever the run starts
helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[1])
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(helper)))
runpy.run_path(helper, run_name='__main__')
initialized = Comm.get().comm.tompi4py().allreduce(int(cpartitioning.petscInitialized()), op=MPI.MAX)
if Comm.get().rank() == 0:
    print("PETSC_INITIALIZED=%d" % initialized)
    sys.stdout.flush()
