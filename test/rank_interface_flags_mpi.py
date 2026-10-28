#!/usr/bin/env python
"""Tags on the exterior faces of a partitioned subdomain; run under mpiexec by
test_rank_interface_flags.py.

A rank's exterior faces are the real boundary of the domain plus the faces it
shares with its neighbours. The partitioner marks the second kind -1; the
domain's own tags (0 and positive) are kept. TAGS=real keeps the generated
boundary flags 1/2/3/4; TAGS=zero sets every real boundary face to 0, the
unflagged-domain case that used to look exactly like an interface.

Environment: PART_TYPE (element|node), NLAYERS, TAGS (real|zero), SHAPE (tri|tet).
"""
import os
from proteus import Comm
comm = Comm.init()
import numpy as np
from proteus import MeshTools

part = {"element": MeshTools.MeshParallelPartitioningTypes.element,
        "node": MeshTools.MeshParallelPartitioningTypes.node}[os.environ.get("PART_TYPE", "element")]
layers = int(os.environ.get("NLAYERS", "0"))
shape = os.environ.get("SHAPE", "tri")

if shape == "tet":
    mesh = MeshTools.TetrahedralMesh()
    mesh.generateTetrahedralMeshFromRectangularGrid(5, 5, 5, 1.0, 1.0, 1.0)
else:
    mesh = MeshTools.TriangularMesh()
    mesh.generateTriangularMeshFromRectangularGrid(17, 17, 1.0, 1.0)
if os.environ.get("TAGS", "real") == "zero":
    mesh.elementBoundaryMaterialTypes[mesh.exteriorElementBoundariesArray] = 0
nReal = len(mesh.exteriorElementBoundariesArray)
mesh.partitionMesh(nLayersOfOverlap=layers, parallelPartitioningType=part)

sub = mesh.subdomainMesh
tags = np.asarray(sub.elementBoundaryMaterialTypes)[sub.exteriorElementBoundariesArray]
hist = {int(t): int((tags == t).sum()) for t in np.unique(tags)}
#ground truth from the geometry: a real boundary face has its centre on the faces of the unit box
centre = sub.nodeArray[sub.elementBoundaryNodesArray[sub.exteriorElementBoundariesArray]].mean(axis=1)
onBox = np.any(np.isclose(centre, 0.0) | np.isclose(centre, 1.0), axis=1)
if shape != "tet":
    onBox = np.any((np.isclose(centre, 0.0) | np.isclose(centre, 1.0))[:, :2], axis=1)
wrong = int(((tags == -1) & onBox).sum() + ((tags == 0) & ~onBox).sum())
print("RANK rank=%d exterior=%d real=%d wrong=%d tags=%s"
      % (comm.rank(), len(tags), nReal, wrong, ",".join("%d:%d" % kv for kv in sorted(hist.items()))),
      flush=True)
