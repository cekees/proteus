#!/usr/bin/env python
"""The combinatorial nodal transfers are a specialization, not a replacement.

They apply only when the hierarchy is nodally nested *and* every space puts its
degrees of freedom on the mesh vertices.  Both conditions fail in cases worth
supporting -- weakly non-hierarchical refinements whose new vertices miss the
parent edges, hierarchies built by coarsening a given fine mesh, and every
space above C0 P1 -- and in those cases the operators have to be built the
general way, by evaluating coarse basis functions against the fine space's
interpolation conditions.

Taking the cheap path where it does not apply would not raise; it would return
a plausible, wrong operator.  So the predicate guarding it is tested from the
negative side here, not just the positive one.
"""
import numpy as np
import pytest

from proteus import FemTools, MeshTools


def _hierarchy(refinementLevels=2):
    return MeshTools.MultilevelTriangularMesh(5, 5, 1,
                                              refinementLevels=refinementLevels)


def test_uniform_refinement_is_nodally_nested():
    assert _hierarchy().isNodallyNested()


def test_a_fine_node_off_its_parent_edge_is_rejected():
    """Weakly non-hierarchical: new vertices that do not bisect a parent edge."""
    mlMesh = _hierarchy()
    nNodes_coarse = mlMesh.meshList[0].nNodes_global
    # move a node the refinement added, so it no longer bisects any coarse edge
    np.asarray(mlMesh.meshList[1].nodeArray)[nNodes_coarse] += 0.01

    assert not mlMesh.isNodallyNested()
    with pytest.raises(ValueError, match='midpoint'):
        mlMesh.calculateNodeParents(recalculate=True)


def test_levels_that_do_not_share_a_node_numbering_are_rejected():
    """Coarsening a fine mesh need not preserve the fine numbering at all."""
    mlMesh = _hierarchy()
    # disturb an inherited node, so the coarse nodes are no longer a prefix
    np.asarray(mlMesh.meshList[0].nodeArray)[0] += 0.01

    assert not mlMesh.isNodallyNested()
    with pytest.raises(ValueError, match='prefix'):
        mlMesh.calculateNodeParents(recalculate=True)


def test_a_rejected_hierarchy_does_not_cache_a_partial_answer():
    """The check raises partway through; nothing half-built may be kept."""
    mlMesh = _hierarchy(refinementLevels=3)
    nNodes_coarse = mlMesh.meshList[1].nNodes_global
    np.asarray(mlMesh.meshList[2].nodeArray)[nNodes_coarse] += 0.01

    assert not mlMesh.isNodallyNested()
    assert mlMesh.nodeParentsArrayList is None, (
        "a partial parent list was cached, and would be returned as complete")


@pytest.mark.parametrize('spaceType,applies', [
    (FemTools.C0_AffineLinearOnSimplexWithNodalBasis, True),
    (FemTools.C0_AffineQuadraticOnSimplexWithNodalBasis, False),
    (FemTools.DG_AffineLinearOnSimplexWithNodalBasis, False),
    (FemTools.NC_AffineLinearOnSimplexWithNodalBasis, False),
])
def test_only_vertex_dof_spaces_take_the_nodal_path(spaceType, applies):
    """P2, DG and non-conforming P1 have dofs that are not mesh vertices.

    Naming a fine degree of freedom by a pair of coarse vertices is meaningless
    for them, however nested the mesh is.
    """
    mlMesh = _hierarchy()
    spaces = [{0: spaceType(mesh.subdomainMesh, 2)} for mesh in mlMesh.meshList]
    assert FemTools.nodalTransfersApply(mlMesh, spaces) is applies


def test_a_nodal_space_on_a_non_nested_hierarchy_still_falls_back():
    """Both halves of the condition are required, not either."""
    mlMesh = _hierarchy()
    nNodes_coarse = mlMesh.meshList[0].nNodes_global
    np.asarray(mlMesh.meshList[1].nodeArray)[nNodes_coarse] += 0.01
    spaces = [{0: FemTools.C0_AffineLinearOnSimplexWithNodalBasis(mesh.subdomainMesh, 2)}
              for mesh in mlMesh.meshList]
    assert FemTools.nodalTransfersApply(mlMesh, spaces) is False
