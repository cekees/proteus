"""Helper run under mpiexec by test_hierarchical_mesh.py; not collected by pytest.

Builds the C0 nodal prolongation as a distributed PETSc Mat from the node
parents and checks it against the *subdomain* meshes -- that is, against the
numbering the solvers' vectors actually use.

That distinction is the whole point of this test. calculateNodeParents works in
the numbering the hierarchy was built with, and partitioning renumbers the
global nodes so each rank's owned ones are contiguous. An earlier version of
this test compared the operator against a serial CSR built from the same parent
array and prolonged a field sampled from the same unpartitioned node array:
every side shared one numbering, so it agreed with itself while being unusable
with a partitioned vector. Sampling the field on the subdomain meshes is what
catches that.
"""
import mpi_failfast  # noqa: F401  (an uncaught exception aborts every rank)
import sys

import numpy as np
from mpi4py import MPI

from proteus import Comm

comm = Comm.init()
from proteus import MeshTools
#this tests the global hierarchy builder, not hierarchies generated in pieces
MeshTools.MultilevelMesh.regularGrid = 'global'
from proteus.LinearAlgebraTools import (ParTransferOperator,
                                        ParTransposeTransferOperator,
                                        ParVec_petsc4py, nodalProlongationMat)

rank = comm.rank()
mpi = comm.comm.tompi4py()

mlMesh = MeshTools.MultilevelTriangularMesh(
    5, 5, 1, refinementLevels=3, nLayersOfOverlap=1,
    parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.node)
mlMesh.calculateNodeParents()


def linear_field(x):
    """Prolongation is linear interpolation, so this must survive it exactly."""
    return 3.0 * x[:, 0] - 2.0 * x[:, 1] + 0.75


def par_vec_for(mesh, array):
    offsets = np.asarray(mesh.nodeOffsets_subdomain_owned)
    owned = int(offsets[rank + 1] - offsets[rank])
    subdomain2global = np.asarray(mesh.nodeNumbering_subdomain2global)
    return ParVec_petsc4py(array, 1, owned, int(offsets[-1]),
                           subdomain2global.shape[0] - owned, subdomain2global)


for l in range(len(mlMesh.meshList) - 1):
    coarse = mlMesh.meshList[l]
    fine = mlMesh.meshList[l + 1]

    prolongMat = nodalProlongationMat(
        mlMesh.partitionedNodeParents(l),
        np.asarray(coarse.nodeOffsets_subdomain_owned),
        np.asarray(fine.nodeOffsets_subdomain_owned))

    coarse_array = np.zeros(coarse.subdomainMesh.nNodes_global)
    fine_array = np.zeros(fine.subdomainMesh.nNodes_global)
    par_coarse = par_vec_for(coarse, coarse_array)
    par_fine = par_vec_for(fine, fine_array)

    prolong = ParTransferOperator(prolongMat, par_coarse, par_fine)
    restrict = ParTransposeTransferOperator(prolongMat, par_coarse, par_fine)

    assert prolong.shape == [fine_array.size, coarse_array.size], prolong.shape
    assert restrict.shape == [coarse_array.size, fine_array.size], restrict.shape

    # a linear field, sampled on the subdomain meshes, must prolong exactly --
    # ghost entries included, since the solvers read them straight away
    source = linear_field(np.asarray(coarse.subdomainMesh.nodeArray))
    expected = linear_field(np.asarray(fine.subdomainMesh.nodeArray))
    result = np.zeros_like(fine_array)
    prolong.matvec(source, result)
    error = mpi.allreduce(float(np.abs(result - expected).max()), op=MPI.MAX)
    assert error < 1.0e-12, (
        "level %i: a linear field did not prolong exactly, max error %r"
        % (l, error))

    # restriction is the adjoint of prolongation, over the owned entries
    rng = np.random.default_rng(11 + l)
    c = rng.random(coarse_array.size)
    f = rng.random(fine_array.size)
    Pc = np.zeros(fine_array.size)
    PTf = np.zeros(coarse_array.size)
    prolong.matvec(c, Pc)
    restrict.matvec(f, PTf)
    lhs = mpi.allreduce(float(np.dot(Pc[:par_fine.dim_proc], f[:par_fine.dim_proc])), op=MPI.SUM)
    rhs = mpi.allreduce(float(np.dot(c[:par_coarse.dim_proc], PTf[:par_coarse.dim_proc])), op=MPI.SUM)
    assert abs(lhs - rhs) <= 1.0e-12 * max(1.0, abs(lhs)), (
        "level %i: restriction is not the adjoint of prolongation, %r vs %r"
        % (l, lhs, rhs))

print("NODAL_PROLONGATION_OK rank=%i of %i" % (rank, comm.size()))
sys.stdout.flush()
