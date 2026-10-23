#!/usr/bin/env python
"""Element maps and transfer operators on quadrilaterals and hexahedra.

The affine element map's inverse computed J^{-1}(x - x_0), the position relative
to the cell's first node. That is the reference coordinate only when the first
node is the reference origin, which it is on the reference simplex and is not on
the reference cube, [-1,1]^d with its first node at (-1,...,-1). Every
reference coordinate came out one too large, so a point at a cell's first corner
mapped to the cell centre.

Anything built on it was wrong for cubes: the geometric transfer operators
(coarse basis evaluated at fine interpolation points) gave row sums of 0.25 on
quadrilaterals and 0.125 on hexahedra, so they did not even preserve constants,
and so did point probes, wave boundary conditions and the trace evaluations.

The affine inverse is also exact only for parallelogram cells. A general
quadrilateral or hexahedron has a multilinear map, so the inverse is refined by
Newton's method from the affine estimate.
"""
import itertools

import numpy as np
import pytest

from proteus import Comm
Comm.init()
from proteus import FemTools, MeshTools

QUAD_FACES = [[0, 1], [1, 2], [2, 3], [3, 0]]
HEX_FACES = [[0, 1, 2, 3], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7]]


def _space(kind, skew):
    if kind == 'quad':
        ml = MeshTools.MultilevelQuadrilateralMesh(4, 4, 1, Lx=1.5, Ly=2.0, x=0.3, y=-0.7,
                                                   refinementLevels=1)
        dim = 2
    else:
        ml = MeshTools.MultilevelHexahedralMesh(3, 3, 3, Lx=1.5, Ly=2.0, Lz=1.0, x=0.3, y=-0.7, z=0.2,
                                                refinementLevels=1)
        dim = 3
    mesh = ml.meshList[0].subdomainMesh
    if skew:
        x = np.asarray(mesh.nodeArray)
        x[:, :dim] += 0.06*np.sin(5.0*x[:, :dim][:, ::-1] + 0.4)
    space = FemTools.C0_AffineLinearOnCubeWithNodalBasis(mesh, dim)
    return mesh, space, dim


def _inverse_jacobian_at_centre(space, mesh, dim, nP):
    J = np.zeros((mesh.nElements_global, nP, dim, dim))
    invJ = np.zeros_like(J)
    detJ = np.zeros((mesh.nElements_global, nP))
    space.elementMaps.getJacobianValues(np.zeros((nP, 3)), J, invJ, detJ)
    return invJ


@pytest.mark.parametrize('kind', ['quad', 'hex'])
@pytest.mark.parametrize('skew', [False, True], ids=['parallelogram', 'skewed'])
def test_inverse_map_recovers_the_reference_point(kind, skew):
    mesh, space, dim = _space(kind, skew)
    rng = np.random.default_rng(3)
    nP = 6
    xi_true = np.zeros((nP, 3))
    xi_true[:, :dim] = rng.uniform(-0.95, 0.95, size=(nP, dim))
    x = np.zeros((mesh.nElements_global, nP, 3))
    space.elementMaps.getValues(xi_true, x)
    invJ = _inverse_jacobian_at_centre(space, mesh, dim, nP)
    xi = np.zeros_like(x)
    space.elementMaps.getInverseValues(invJ, x, xi)
    err = np.abs(xi[:, :, :dim] - xi_true[None, :, :dim]).max()
    assert err < 1e-10, "inverse map is off by %g (%s cells)" % (err, 'skewed' if skew else 'regular')
    # and the single-point form agrees
    for eN in (0, mesh.nElements_global - 1):
        single = space.elementMaps.getInverseValue(eN, x[eN, 0])
        assert np.abs(single - xi_true[0, :dim]).max() < 1e-10


@pytest.mark.parametrize('kind', ['quad', 'hex'])
def test_a_cells_first_corner_maps_to_the_reference_corner(kind):
    """The shift in one line: node 0 of every cell is the corner (-1,...,-1)."""
    mesh, space, dim = _space(kind, skew=False)
    nodes = np.asarray(mesh.nodeArray)[np.asarray(mesh.elementNodesArray)[:, 0]]
    x = nodes[:, None, :].copy()
    xi = np.zeros_like(x)
    invJ = _inverse_jacobian_at_centre(space, mesh, dim, 1)
    space.elementMaps.getInverseValues(invJ, x, xi)
    assert np.abs(xi[:, 0, :dim] + 1.0).max() < 1e-12


@pytest.mark.parametrize('kind', ['quad', 'hex'])
def test_trace_inverse_puts_face_centres_on_the_reference_boundary(kind):
    """On a face centre exactly one reference coordinate is +-1 and the rest are 0."""
    mesh, space, dim = _space(kind, skew=False)
    faces = QUAD_FACES if kind == 'quad' else HEX_FACES
    nE = mesh.nElements_global
    nodes = np.asarray(mesh.nodeArray)[np.asarray(mesh.elementNodesArray)]
    x = np.zeros((nE, len(faces), 1, 3))
    for f, ids in enumerate(faces):
        x[:, f, 0, :] = nodes[:, ids, :].mean(axis=1)
    invJ0 = _inverse_jacobian_at_centre(space, mesh, dim, 1)
    invJ = np.repeat(invJ0[:, None, :, :, :], len(faces), axis=1)
    xi = np.zeros_like(x)
    space.elementMaps.getInverseValuesTrace(invJ, x, xi)
    mags = np.sort(np.abs(xi[:, :, 0, :dim]), axis=2)
    assert np.abs(mags[:, :, -1] - 1.0).max() < 1e-12, "a face centre is not on the reference boundary"
    assert np.abs(mags[:, :, :-1]).max() < 1e-12, "a face centre is off its face's centre"


@pytest.mark.parametrize('kind', ['quad', 'hex'])
def test_exterior_trace_inverse_agrees(kind):
    mesh, space, dim = _space(kind, skew=False)
    nExt = mesh.nExteriorElementBoundaries_global
    ext = np.asarray(mesh.exteriorElementBoundariesArray)
    ebn = np.asarray(mesh.elementBoundaryNodesArray)
    coords = np.asarray(mesh.nodeArray)
    x = np.zeros((nExt, 1, 3))
    for k, ebN in enumerate(ext):
        x[k, 0, :] = coords[ebn[ebN]].mean(axis=0)
    invJ_e = _inverse_jacobian_at_centre(space, mesh, dim, 1)
    left = np.asarray(mesh.elementBoundaryElementsArray)[ext, 0]
    invJ = invJ_e[left]
    xi = np.zeros_like(x)
    space.elementMaps.getInverseValuesGlobalExteriorTrace(invJ, x, xi)
    mags = np.sort(np.abs(xi[:, 0, :dim]), axis=1)
    assert np.abs(mags[:, -1] - 1.0).max() < 1e-12
    assert np.abs(mags[:, :-1]).max() < 1e-12


def test_simplex_inverse_map_is_unchanged():
    """The reference simplex has its first node at the origin: no shift there."""
    ml = MeshTools.MultilevelTriangularMesh(3, 3, 1, refinementLevels=1)
    mesh = ml.meshList[0].subdomainMesh
    space = FemTools.C0_AffineLinearOnSimplexWithNodalBasis(mesh, 2)
    rng = np.random.default_rng(1)
    xi_true = np.zeros((4, 3))
    xi_true[:, :2] = rng.uniform(0.05, 0.45, size=(4, 2))
    x = np.zeros((mesh.nElements_global, 4, 3))
    space.elementMaps.getValues(xi_true, x)
    J = np.zeros((mesh.nElements_global, 4, 2, 2))
    invJ = np.zeros_like(J)
    space.elementMaps.getJacobianValues(np.zeros((4, 3)), J, invJ, np.zeros((mesh.nElements_global, 4)))
    xi = np.zeros_like(x)
    space.elementMaps.getInverseValues(invJ, x, xi)
    assert np.abs(xi[:, :, :2] - xi_true[None, :, :2]).max() < 1e-12
