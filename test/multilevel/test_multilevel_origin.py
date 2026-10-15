#!/usr/bin/env python
"""A multilevel mesh must not depend on where its domain starts.

The C-path constructors of MultilevelEdgeMesh, MultilevelTriangularMesh and
MultilevelTetrahedralMesh generated the coarse mesh, refined it in C -- which
copies the coarse nodes into every finer level -- and only then added the
origin (x, y, z) to level 0's node array. The finer levels were left at the
origin-based coordinates. So with a nonzero origin the levels lived in
different coordinate frames:

* the geometric transfer operators between level 0 and level 1 were wrong (a
  linear field prolonged with an error of 31.5 on the 3D Poisson problem with
  its origin at (-3,-3,-3));
* the finest level, the one that is solved on, did not cover the domain; and
* the nodal transfer operators, which require each level's nodes to be a prefix
  of the next level's, correctly refused the hierarchy.

Nothing flagged it: with the origin at zero the frames coincide, and multilevel
runs on shifted domains were rare because multigrid had been broken since 2019.
"""
import numpy as np
import pytest

from proteus import Comm
Comm.init()
from proteus import MeshTools

CASES = {
    'edge': lambda: MeshTools.MultilevelEdgeMesh(
        5, 1, 1, x=2.0, Lx=3.0, refinementLevels=3),
    'triangle': lambda: MeshTools.MultilevelTriangularMesh(
        3, 3, 1, x=1.0, y=2.0, Lx=2.0, Ly=3.0, refinementLevels=3),
    'tetrahedron': lambda: MeshTools.MultilevelTetrahedralMesh(
        3, 3, 3, x=-3.0, y=-3.0, z=-3.0, Lx=6.0, Ly=6.0, Lz=6.0, refinementLevels=3),
}


@pytest.mark.parametrize('kind', sorted(CASES))
def test_every_level_shares_the_coarse_nodes(kind):
    mlMesh = CASES[kind]()
    for l in range(len(mlMesh.meshList) - 1):
        coarse = np.asarray(mlMesh.meshList[l].nodeArray)
        fine = np.asarray(mlMesh.meshList[l + 1].nodeArray)
        assert np.array_equal(fine[:coarse.shape[0]], coarse), (
            "%s level %i: the nodes of the next level do not begin with this "
            "level's, so the levels are in different coordinate frames" % (kind, l))


@pytest.mark.parametrize('kind', sorted(CASES))
def test_every_level_covers_the_domain(kind):
    mlMesh = CASES[kind]()
    lo = np.asarray(mlMesh.meshList[0].nodeArray).min(axis=0)
    hi = np.asarray(mlMesh.meshList[0].nodeArray).max(axis=0)
    for l, mesh in enumerate(mlMesh.meshList):
        nodes = np.asarray(mesh.nodeArray)
        assert np.allclose(nodes.min(axis=0), lo) and np.allclose(nodes.max(axis=0), hi), (
            "%s level %i spans %s..%s, not the domain %s..%s"
            % (kind, l, nodes.min(axis=0), nodes.max(axis=0), lo, hi))


@pytest.mark.parametrize('kind', sorted(CASES))
def test_a_shifted_hierarchy_is_still_nodally_nested(kind):
    assert CASES[kind]().isNodallyNested()
