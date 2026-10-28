#!/usr/bin/env python
"""Proteus mesh -> PUMI mesh at any rank count; run under mpiexec by
test_reconstructParallel.py.

Partitions a unit square (SHAPE=tri) or cube (SHAPE=tet) with Proteus, gathers
it on rank 0, builds and migrates the PUMI mesh by Proteus's own element
partition, then reloads Reconstructed.smb/.dmg and prints one RANK line per
rank: the elements Proteus gave the rank, the elements PUMI gave it, and the
global counts, and the model's entity counts (regions faces edges vertices).
"""
import itertools
import os
from proteus import Comm
comm = Comm.init()
import numpy as np
from mpi4py import MPI
from proteus import Domain, MeshTools, cmeshTools
from proteus.MeshAdaptPUMI import MeshAdapt, AdaptHelper

shape = os.environ.get("SHAPE", "tri")
nd = 3 if shape == "tet" else 2
if nd == 3:
    mesh = MeshTools.TetrahedralMesh()
    mesh.generateTetrahedralMeshFromRectangularGrid(4, 4, 4, 1.0, 1.0, 1.0)
else:
    mesh = MeshTools.TriangularMesh()
    mesh.generateTriangularMeshFromRectangularGrid(9, 9, 1.0, 1.0)
nGlobal = mesh.nElements_global
def boundary_tags(m, faces):
    # {(rounded face centre): tag} for the given exterior faces of a mesh
    centre = m.nodeArray[m.elementBoundaryNodesArray[faces]].mean(axis=1)
    return {tuple(np.round(c, 6)): int(t) for c, t in zip(centre, np.asarray(m.elementBoundaryMaterialTypes)[faces])}
expected = boundary_tags(mesh, mesh.exteriorElementBoundariesArray)
mesh.partitionMesh(nLayersOfOverlap=int(os.environ.get("NLAYERS", "1")),
                   parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.element)
assert comm.size() > 1, 'run under mpiexec -n 2 or more'
sub = mesh.subdomainMesh
corners = [list(c) for c in itertools.product([0.0, 1.0], repeat=nd)]

adapter = MeshAdapt.MeshAdapt()
args = AdaptHelper.gatherGlobalMesh(sub, MPI.COMM_WORLD, nd, corners)
n = adapter.reconstructFromProteusGlobal(nd, *args)

domain = Domain.PUMIDomain(dim=nd, manager=MeshAdapt.AdaptManager(MeshAdapt.MeshAdapt()))
adapter2 = domain.AdaptManager.PUMIAdapter
adapter2.loadModelAndMesh(b"Reconstructed.dmg", b"Reconstructed.smb")
pm = MeshTools.TetrahedralMesh() if nd == 3 else MeshTools.TriangularMesh()
pm.cmesh = cmeshTools.CMesh()
pm.convertFromPUMI(domain, adapter2, domain.faceList, domain.regList,
                   parallel=True, dim=nd)
psub = pm.subdomainMesh
# every true-boundary face of the PUMI subdomains (tag > 0) must carry the tag the
# Proteus mesh gave the face at the same place, and together they must cover all of them
ext = np.asarray(psub.exteriorElementBoundariesArray)
found = boundary_tags(psub, ext)
wrong = sum(1 for c, t in found.items() if t > 0 and expected.get(c) != t)
allfound = {}
for part in MPI.COMM_WORLD.allgather({c: t for c, t in found.items() if t > 0}):
    allfound.update(part)
missing = len(expected) - len(allfound)
print("RANK rank=%d proteus_owned=%d pumi_owned=%d returned=%d global=%d pumi_global=%d "
      "model=%s tagwrong=%d tagmissing=%d"
      % (comm.rank(), sub.nElements_owned, psub.nElements_owned, n, nGlobal,
         pm.nElements_global, ",".join(open("Reconstructed.dmg").readline().split()), wrong, missing), flush=True)
