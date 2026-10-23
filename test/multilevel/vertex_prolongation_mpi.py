"""Helper run under mpiexec by test_vertex_interpolation.py; not collected.

Builds the vertex prolongation for a hierarchy of any of the four vertex-based
element types as a distributed PETSc Mat and checks it against the *subdomain*
meshes, in the numbering the solvers' vectors use, ghost entries included.

Q1 and P1 spaces contain different functions -- Q1 reproduces the multilinear
x*y and x*y*z, P1 only linears -- so each field is checked where its
interpolation is exact. A weight that is off anywhere shows up as an error
in one of them.
"""
import sys

import numpy as np
from mpi4py import MPI

from proteus import Comm
comm = Comm.init()
from proteus import MeshTools
#this tests the global hierarchy builder, not hierarchies generated in pieces
MeshTools.MultilevelMesh.regularGrid = 'global'
from proteus.LinearAlgebraTools import (ParTransferOperator, ParTransposeTransferOperator,
                                        ParVec_petsc4py, vertexProlongationMat)

rank = comm.rank()
mpi = comm.comm.tompi4py()
kind = sys.argv[1]
part = MeshTools.MeshParallelPartitioningTypes.node
if kind == 'tri':
    ml = MeshTools.MultilevelTriangularMesh(3, 3, 1, x=1.0, y=2.0, refinementLevels=3,
                                            nLayersOfOverlap=1, parallelPartitioningType=part)
    dim = 2
elif kind == 'tet':
    ml = MeshTools.MultilevelTetrahedralMesh(3, 3, 3, x=-1.0, y=-2.0, z=0.5, refinementLevels=3,
                                             nLayersOfOverlap=1, parallelPartitioningType=part)
    dim = 3
elif kind == 'quad':
    ml = MeshTools.MultilevelQuadrilateralMesh(3, 3, 1, x=1.0, y=2.0, Lx=2.0, Ly=3.0, refinementLevels=3,
                                               nLayersOfOverlap=1, parallelPartitioningType=part)
    dim = 2
elif kind == 'hex':
    ml = MeshTools.MultilevelHexahedralMesh(3, 3, 3, x=-3.0, y=-3.0, z=-3.0, Lx=6.0, Ly=6.0, Lz=6.0,
                                            refinementLevels=3, nLayersOfOverlap=1,
                                            parallelPartitioningType=part)
    dim = 3
else:
    raise ValueError(kind)
assert ml.hasVertexInterpolation(), "the hierarchy has no vertex interpolation"
cube = kind in ('quad', 'hex')


def fields(x):
    """Fields the interpolation must reproduce exactly."""
    out = {'constant': np.ones(x.shape[0]),
           'linear': 3.0*x[:, 0] - 2.0*x[:, 1] + (0.5*x[:, 2] if dim == 3 else 0.0) + 0.75}
    if cube:
        out['bilinear'] = x[:, 0]*x[:, 1] + 0.3*x[:, 0] - x[:, 1]
        if dim == 3:
            out['trilinear'] = x[:, 0]*x[:, 1]*x[:, 2] + x[:, 0]*x[:, 2] + 0.1
    return out


def par_vec_for(mesh, array):
    offsets = np.asarray(mesh.nodeOffsets_subdomain_owned)
    owned = int(offsets[rank+1] - offsets[rank])
    s2g = np.asarray(mesh.nodeNumbering_subdomain2global)
    return ParVec_petsc4py(array, 1, owned, int(offsets[-1]), s2g.shape[0] - owned, s2g)


for l in range(len(ml.meshList) - 1):
    coarse, fine = ml.meshList[l], ml.meshList[l+1]
    cOff = np.asarray(coarse.nodeOffsets_subdomain_owned)
    fOff = np.asarray(fine.nodeOffsets_subdomain_owned)
    P = vertexProlongationMat(*ml.partitionedVertexInterpolation(l),
                              coarseNodeOffsets=cOff, fineNodeOffsets=fOff)
    carr = np.zeros(coarse.subdomainMesh.nNodes_global)
    farr = np.zeros(fine.subdomainMesh.nNodes_global)
    pc, pf = par_vec_for(coarse, carr), par_vec_for(fine, farr)
    prolong = ParTransferOperator(P, pc, pf)
    restrict = ParTransposeTransferOperator(P, pc, pf)
    xc = np.asarray(coarse.subdomainMesh.nodeArray)
    xf = np.asarray(fine.subdomainMesh.nodeArray)
    fc, ff = fields(xc), fields(xf)
    for name in fc:
        out = np.zeros_like(farr)
        prolong.matvec(fc[name], out)
        err = mpi.allreduce(float(np.abs(out - ff[name]).max()), op=MPI.MAX)
        assert err < 1.0e-12, "%s level %i: %s field off by %r" % (kind, l, name, err)
    # restriction is the adjoint, over the owned entries
    rng = np.random.default_rng(5 + l)
    c, f = rng.random(carr.size), rng.random(farr.size)
    Pc, Rf = np.zeros(farr.size), np.zeros(carr.size)
    prolong.matvec(c, Pc)
    restrict.matvec(f, Rf)
    lhs = mpi.allreduce(float(np.dot(Pc[:pf.dim_proc], f[:pf.dim_proc])), op=MPI.SUM)
    rhs = mpi.allreduce(float(np.dot(c[:pc.dim_proc], Rf[:pc.dim_proc])), op=MPI.SUM)
    assert abs(lhs - rhs) <= 1.0e-12*max(1.0, abs(lhs)), (kind, l, lhs, rhs)

print("VERTEX_PROLONGATION_OK kind=%s rank=%i of %i" % (kind, rank, comm.size()))
sys.stdout.flush()
