"""Helper run under mpiexec by test_generated_hierarchy.py; not collected.

Checks the inter-level data of a hierarchy generated in pieces
(MultilevelMesh.generateCartesianHierarchy), level by level and rank by rank,
without any global mesh:

* parents (cartesianSubdomainParents): every fine element this rank owns has
  its parent in the coarse subdomain, and every fine element with a parent lies
  in it (its centroid is inside the parent, by barycentric coordinates);
* vertex interpolation (cartesianVertexInterpolation): one row per owned fine
  node, every column a node of the coarse subdomain (so the operator is local),
  rows summing to one, and the interpolant of a linear function (Q1: the
  trilinear x*y*z as well) equal to the function at every owned fine node.

Usage: generated_hierarchy_mpi.py KIND NN LEVELS     (KIND tri, quad, tet, hex)

Prints ``RESULT kind=.. ranks=.. failures=<count over ranks and levels>``.
"""
import sys

import numpy as np
from proteus import Comm
comm = Comm.init()
from mpi4py import MPI
from proteus import MeshTools, cpartitioning

rank, size = comm.rank(), comm.size()
mpi = comm.comm.tompi4py()
kind = sys.argv[1]
nn = int(sys.argv[2])
nLevels = int(sys.argv[3])
dim = 2 if kind in ('tri', 'quad') else 3
origin = (-0.5, 0.25, 1.0)[:dim] + (0.0,) * (3 - dim)
L = (1.0, 0.7, 1.3)
ml = MeshTools.MultilevelMesh()
ml.generateCartesianHierarchy(kind, (nn,) * dim, origin, L, nLevels, triangleFlag=0)
failures = 0
messages = []


def fail(message):
    global failures
    failures += 1
    if len(messages) < 5:
        messages.append("rank %d: %s" % (rank, message))


hierarchy = ml.subdomainHierarchy()
for l in range(nLevels - 1):
    coarse, fine = ml.meshList[l], ml.meshList[l + 1]
    cs, fs = coarse.subdomainMesh, fine.subdomainMesh
    parents = np.asarray(hierarchy.elementParentsArrayList[l + 1])
    if parents.shape != (fs.nElements_global,):
        fail("level %d: %s parents for %d fine elements" % (l + 1, parents.shape, fs.nElements_global))
        continue
    if (parents[:fs.nElements_owned] < 0).any():
        fail("level %d: an owned fine element has no parent here" % (l + 1))
    cx = np.asarray(cs.nodeArray)[:, :dim]
    fx = np.asarray(fs.nodeArray)[:, :dim]
    for e in np.flatnonzero(parents >= 0):
        centroid = fx[np.asarray(fs.elementNodesArray)[e]].mean(axis=0)
        corners = cx[np.asarray(cs.elementNodesArray)[parents[e]]]
        if kind in ('quad', 'hex'):
            inside = (centroid > corners.min(axis=0)).all() and (centroid < corners.max(axis=0)).all()
        else:
            A = (corners[1:] - corners[0]).T
            lam = np.linalg.solve(A, centroid - corners[0])
            inside = (lam > 1.0e-12).all() and lam.sum() < 1.0 - 1.0e-12
        if not inside:
            fail("level %d: fine element %d is not inside its parent" % (l + 1, e))
            break
    rowptr, colind, nzval = ml.partitionedVertexInterpolation(l)
    if rowptr.shape[0] != fs.nNodes_owned + 1:
        fail("level %d: %d rows for %d owned fine nodes" % (l + 1, rowptr.shape[0] - 1, fs.nNodes_owned))
        continue
    coarseLocal = dict((int(g), i) for i, g in enumerate(np.asarray(coarse.nodeNumbering_subdomain2global)))
    missing = [int(c) for c in colind if int(c) not in coarseLocal]
    if missing:
        fail("level %d: %d prolongation columns are not coarse subdomain nodes" % (l + 1, len(missing)))
        continue
    sums = np.add.reduceat(nzval, rowptr[:-1]) if nzval.size else np.zeros(0)
    if np.abs(sums - 1.0).max(initial=0.0) > 1.0e-14:
        fail("level %d: a prolongation row does not sum to one" % (l + 1))
    tests = [lambda x: 1.0 + x @ np.arange(1.0, dim + 1.0)]
    if kind in ('quad', 'hex'):
        tests.append(lambda x: np.prod(x, axis=1))
    for f in tests:
        coarseValues = f(cx)
        local = np.array([coarseLocal[int(c)] for c in colind], dtype=int)
        interpolated = np.add.reduceat(nzval * coarseValues[local], rowptr[:-1]) if nzval.size else np.zeros(0)
        exact = f(fx[:fs.nNodes_owned])
        if np.abs(interpolated - exact).max(initial=0.0) > 1.0e-12 * max(1.0, np.abs(exact).max(initial=0.0)):
            fail("level %d: the prolongation does not reproduce a function it should" % (l + 1))
for m in messages:
    print(m)
failures = mpi.allreduce(failures, op=MPI.SUM)
if rank == 0:
    print("RESULT kind=%s ranks=%d levels=%d failures=%d" % (kind, size, nLevels, failures))
    sys.stdout.flush()
