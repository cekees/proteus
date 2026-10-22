"""MeshTools owns meshing policy; Domain owns the domain.

A domain says what a region *is*. How fine the elements are, which generator
runs, how the mesh gets partitioned -- none of that is part of the region, so
none of it is state on a Domain object. It lives on the MeshOptions that
MeshTools attaches to a domain on first use.

These tests pin that split, and the compatibility spellings that keep the
rest of Proteus reading through to the same one owner.
"""
import copy
import os
import pickle

import pytest

from proteus import Domain, MeshTools


class TestMeshOptionsOwnership(object):

    def test_a_fresh_domain_carries_no_meshing_state(self):
        """Constructing a domain must not bring meshing policy with it."""
        for domain in (Domain.RectangularDomain(L=[1.0, 2.0]),
                       Domain.PlanarStraightLineGraphDomain(
                           vertices=[[0., 0.], [1., 0.], [0., 1.]],
                           segments=[[0, 1], [1, 2], [2, 0]]),
                       Domain.PiecewiseLinearComplexDomain(
                           vertices=[[0., 0., 0.], [1., 0., 0.],
                                     [0., 1., 0.], [0., 0., 1.]],
                           facets=[[[0, 2, 3]], [[0, 1, 2]],
                                   [[0, 1, 3]], [[1, 2, 3]]]),
                       Domain.Mesh3DMDomain("some_mesh")):
            leaked = [name for name in vars(domain)
                      if name in ('MeshOptions', 'use_gmsh', 'use_plex', 'he')]
            assert leaked == [], (type(domain).__name__, leaked)

    def test_options_are_created_once_with_the_domain_dimension(self):
        domain = Domain.RectangularDomain(L=[1.0, 2.0, 3.0])
        meshOptions = MeshTools.getMeshOptions(domain)
        assert meshOptions.nd == 3
        assert MeshTools.getMeshOptions(domain) is meshOptions

    def test_set_replaces_them(self):
        domain = Domain.RectangularDomain(L=[1.0, 2.0])
        replacement = MeshTools.MeshOptions(2)
        MeshTools.setMeshOptions(domain, replacement)
        assert MeshTools.getMeshOptions(domain) is replacement
        assert domain.MeshOptions is replacement

    def test_the_old_spellings_reach_the_one_owner(self):
        domain = Domain.RectangularDomain(L=[1.0, 2.0])
        assert domain.MeshOptions is MeshTools.getMeshOptions(domain)

        domain.MeshOptions = MeshTools.MeshOptions(2)
        assert domain.MeshOptions is MeshTools.getMeshOptions(domain)

    @pytest.mark.parametrize("name", ["use_gmsh", "use_plex"])
    def test_the_generator_flags_are_not_shadowed(self, name):
        """domain.<flag> and the options' <flag> used to be separate copies.

        _generateMesh read the domain's, SpatialTools wrote the options' --
        so asking for gmsh never reached the branch that runs it.
        """
        domain = Domain.RectangularDomain(L=[1.0, 2.0])
        meshOptions = MeshTools.getMeshOptions(domain)

        setattr(domain, name, True)
        assert getattr(meshOptions, name) is True

        setattr(meshOptions, name, False)
        assert getattr(domain, name) is False

    def test_gmsh_characteristic_length_is_policy_not_domain(self):
        domain = Domain.GMSH_3D_Domain(geofile="flume", he=0.1)
        assert MeshTools.getMeshOptions(domain).he == 0.1
        assert "he" not in vars(domain)

    def test_gmsh_domain_needs_no_characteristic_length(self):
        domain = Domain.GMSH_3D_Domain(geofile="flume")
        assert domain.geofile == "flume.geo"
        assert "he" not in vars(domain)

    def test_region_constraints_are_policy(self):
        domain = Domain.PlanarStraightLineGraphDomain(
            vertices=[[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
            segments=[[0, 1], [1, 2], [2, 3], [3, 0]],
            segmentFlags=[1, 2, 3, 4],
            regions=[[0.5, 0.5]],
            regionFlags=[1],
            regionConstraints=[0.125])
        assert MeshTools.getMeshOptions(domain).regionConstraints == [0.125]
        assert "regionConstraints" not in vars(domain)
        assert domain.regionConstraints == [0.125]

    def test_region_constraints_still_reach_the_poly_file(self, tmpdir):
        """They are policy, but triangle/tetgen read them out of the .poly."""
        domain = Domain.PlanarStraightLineGraphDomain(
            vertices=[[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
            segments=[[0, 1], [1, 2], [2, 3], [3, 0]],
            segmentFlags=[1, 2, 3, 4],
            regions=[[0.5, 0.5]],
            regionFlags=[1],
            regionConstraints=[0.125])
        fileprefix = os.path.join(str(tmpdir), "constrained")
        domain.writePoly(fileprefix)
        with open(fileprefix + ".poly") as f:
            written = f.read()
        # last region line: index, x, y, flag, constraint
        region = written.strip().split("\n")[-1].split()
        assert int(region[3]) == 1
        assert float(region[4]) == 0.125

    def test_options_survive_copying_the_domain(self):
        """They are instance state, so a copied or unpickled domain keeps them."""
        domain = Domain.RectangularDomain(L=[1.0, 2.0])
        MeshTools.getMeshOptions(domain).he = 0.03125

        assert MeshTools.getMeshOptions(copy.deepcopy(domain)).he == 0.03125
        assert MeshTools.getMeshOptions(
            pickle.loads(pickle.dumps(domain))).he == 0.03125
