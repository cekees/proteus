#!/usr/bin/env python
"""Multigrid on hierarchies generated in pieces: no global mesh on any rank.

MultilevelMesh.generateCartesianHierarchy (the default, MeshOptions.regularGrid
= 'generated'; 'global' selects the global mesh, partitioned) builds
each level of a regular-grid hierarchy as every rank's own piece
(MeshTools.cartesianSubdomain(level=l)), cut into the coarsest level's bricks
scaled so that the partition is level-consistent by construction, and computes
the transfer operators' ingredients per rank: the owned rows of the vertex
interpolation and the parents of the subdomain's elements.

What is pinned here:

* each generated level is, array for array on every rank, the global grid at
  that resolution partitioned with the same scaled bricks;
* the inter-level data is local and right: every parent of an owned fine
  element or node is on the rank, every fine element lies inside its parent,
  and the prolongation reproduces linear (and, for Q1, multilinear) functions;
* proteus's multigrid on a generated hierarchy gives the same answer on every
  rank count, the same answer as the global path where the two hierarchies are
  the same meshes (quadrilaterals, hexahedra, right-leaning triangles; the
  global tetrahedral hierarchy refines differently, CARTESIAN_PARTITIONER_CAMPAIGN.md
  Phase 4 decisions), and with petsc4py blocked it never initializes PETSc.
"""
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ('mpiexec', 'mpirun'):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which('mpiexec') or shutil.which('mpirun')


def _launch(args, n_ranks, generate=True, blocked=False, timeout=600):
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip('no MPI launcher available')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)
    # CI runners have two cores; let Open MPI (4.x and 5.x/PRRTE) start more
    # ranks than that. MPICH oversubscribes by default and ignores these.
    env.setdefault('OMPI_MCA_rmaps_base_oversubscribe', '1')
    env.setdefault('PRTE_MCA_rmaps_default_mapping_policy', ':oversubscribe')
    script = [os.path.join(HERE, 'without_petsc4py.py')] if blocked else []
    if not generate:
        args = list(args) + ['global']
    result = subprocess.run([mpiexec, '-n', str(n_ranks), sys.executable] + script +
                            [os.path.join(HERE, args[0])] + [str(a) for a in args[1:]],
                            capture_output=True, text=True, env=env, timeout=timeout, cwd=HERE)
    assert result.returncode == 0, result.stdout + result.stderr
    for line in result.stdout.splitlines():
        if line.startswith('RESULT'):
            return dict(item.split('=', 1) for item in line.split()[1:]), result.stdout
    raise AssertionError(result.stdout + result.stderr)


@pytest.mark.parametrize('n_ranks', [1, 3, 4])
@pytest.mark.parametrize('level', [1, 2])
@pytest.mark.parametrize('kind,nn', [('quad', (5, 4)), ('hex', (3, 3, 4)), ('tri0', (5, 4)), ('tet', (3, 3, 4))])
def test_each_level_is_the_partitioned_level(kind, nn, level, n_ranks):
    got, out = _launch(['cartesian_subdomain_mpi.py', kind] + list(nn) + ['level=%d' % level], n_ranks)
    assert int(got['mismatches']) == 0, out


@pytest.mark.parametrize('n_ranks', [1, 2, 3, 4])
@pytest.mark.parametrize('kind', ['tri', 'quad', 'tet', 'hex'])
def test_inter_level_data_is_local_and_right(kind, n_ranks):
    got, out = _launch(['generated_hierarchy_mpi.py', kind, 4, 3], n_ranks)
    assert int(got['failures']) == 0, out


def _close(a, b):
    a, b = float(a), float(b)
    return abs(a - b) <= 1.0e-11 * abs(a)


@pytest.mark.parametrize('kind', ['tri', 'quad', 'tet', 'hex'])
def test_multigrid_on_a_generated_hierarchy(kind):
    serial, out = _launch(['regular_mg_mpi.py', kind, 3, 3], 1)
    assert serial['generated'] == '1', out
    for n_ranks in (2, 3, 4):
        got, out = _launch(['regular_mg_mpi.py', kind, 3, 3], n_ranks)
        assert got['generated'] == '1', out
        assert _close(serial['checksum'], got['checksum']), (serial, got)
        assert _close(serial['sumsq'], got['sumsq']), (serial, got)
    if kind != 'tet':
        reference, out = _launch(['regular_mg_mpi.py', kind, 3, 3], 3, generate=False)
        assert reference['generated'] == '0', out
        assert _close(reference['checksum'], serial['checksum']), (reference, serial)
        assert _close(reference['sumsq'], serial['sumsq']), (reference, serial)


@pytest.mark.parametrize('kind', ['quad', 'tet'])
def test_a_generated_hierarchy_needs_no_petsc(kind):
    pytest.importorskip('petsc4py')
    reference, out = _launch(['regular_mg_mpi.py', kind, 3, 3], 3)
    got, out = _launch(['regular_mg_mpi.py', kind, 3, 3], 3, blocked=True)
    assert got['generated'] == '1' and got['petsc'] == '0', out
    assert _close(reference['checksum'], got['checksum']), (reference, got)


@pytest.mark.parametrize('kind,options', [
    ('tri', dict(triangleFlag=1)),
    ('tri', dict(triangleFlag=2)),
    ('quad', dict(nLayersOfOverlap=2)),
    ('hex', dict(parallelPartitioningType=1)),
    ('tet', dict(regularGrid='global')),
])
def test_the_cases_it_does_not_cover_take_the_global_path(kind, options):
    """Union-jack and left-leaning triangles, overlap other than one layer,
    element partitioning and regularGrid='global' build the global mesh and
    partition it; the default builds the same grid in pieces."""
    from proteus import MeshTools
    cls = {'tri': MeshTools.MultilevelTriangularMesh, 'quad': MeshTools.MultilevelQuadrilateralMesh,
           'tet': MeshTools.MultilevelTetrahedralMesh, 'hex': MeshTools.MultilevelHexahedralMesh}[kind]
    nz = 3 if kind in ('tet', 'hex') else 1
    if 'parallelPartitioningType' in options:
        options = dict(options, parallelPartitioningType=MeshTools.MeshParallelPartitioningTypes.element)
    fallback = cls(3, 3, nz, refinementLevels=2, **options)
    assert not fallback.generated
    assert fallback.meshList[-1].nNodes_global == fallback.meshList[-1].subdomainMesh.nNodes_global
    default = cls(3, 3, nz, refinementLevels=2, **(dict(triangleFlag=0) if kind == 'tri' else {}))
    assert default.generated
    assert default.meshList[-1].nNodes_global == fallback.meshList[-1].nNodes_global
