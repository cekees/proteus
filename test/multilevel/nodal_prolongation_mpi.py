"""Helper run under mpiexec by test_hierarchical_mesh.py; not collected by pytest.

Builds the C0 nodal prolongation as a distributed PETSc Mat straight from the
node-parent array and checks three things on every level:

* a linear field prolongs exactly -- the operator really is the linear
  interpolant, checked without reference to any serially-built operator;
* every row this rank owns matches the serial CSR that the equivalence test
  already pins against proteus's geometric construction;
* the restriction row sums used for restricting solutions are positive, which
  is what the normalization divides by.
"""
import sys

import numpy as np
from mpi4py import MPI

from proteus import Comm

comm = Comm.init()
from proteus import MeshTools
from proteus.LinearAlgebraTools import nodalProlongationMat, nodalRestrictionScaling

rank = comm.rank()
mpi = comm.comm.tompi4py()

mlMesh = MeshTools.MultilevelTriangularMesh(
    5, 5, 1, refinementLevels=3, nLayersOfOverlap=1,
    parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.node)
mlMesh.calculateNodeParents()


def linear_field(x):
    return 3.0 * x[:, 0] - 2.0 * x[:, 1] + 0.75


for l in range(len(mlMesh.meshList) - 1):
    coarse = mlMesh.meshList[l]
    fine = mlMesh.meshList[l + 1]
    prolong = nodalProlongationMat(
        mlMesh.nodeParentsArrayList[l + 1],
        np.asarray(coarse.nodeOffsets_subdomain_owned),
        np.asarray(fine.nodeOffsets_subdomain_owned))

    nrows, ncols = prolong.getSize()
    assert nrows == fine.nNodes_global, (nrows, fine.nNodes_global)
    assert ncols == coarse.nNodes_global, (ncols, coarse.nNodes_global)

    coarse_vec = prolong.createVecRight()
    fine_vec = prolong.createVecLeft()
    clo, chi = coarse_vec.getOwnershipRange()
    coarse_vec.setArray(linear_field(np.asarray(coarse.nodeArray)[clo:chi]))
    coarse_vec.assemble()
    prolong.mult(coarse_vec, fine_vec)

    flo, fhi = fine_vec.getOwnershipRange()
    expected = linear_field(np.asarray(fine.nodeArray)[flo:fhi])
    local_err = float(np.abs(fine_vec.getArray() - expected).max()) if fhi > flo else 0.0
    err = mpi.allreduce(local_err, op=MPI.MAX)
    assert err == 0.0, "linear field did not prolong exactly: %r" % err

    rowptr, colind, nzval = mlMesh.nodalProlongation(l)
    disagreeing = 0
    for row in range(flo, fhi):
        cols, vals = prolong.getRow(row)
        reference = sorted(zip(colind[rowptr[row]:rowptr[row + 1]].tolist(),
                               nzval[rowptr[row]:rowptr[row + 1]].tolist()))
        if sorted(zip(cols, vals)) != reference:
            disagreeing += 1
    disagreeing = mpi.allreduce(disagreeing, op=MPI.SUM)
    assert disagreeing == 0, "%i rows differ from the serial operator" % disagreeing

    scaling = nodalRestrictionScaling(prolong)
    assert scaling.min()[1] > 0.0, "a restriction row sum is not positive"

print("NODAL_PROLONGATION_OK rank=%i of %i" % (rank, comm.size()))
sys.stdout.flush()
