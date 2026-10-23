#!/usr/bin/env python
"""Generating a regular grid in pieces gives what partitioning the whole grid gives.

MeshTools.cartesianSubdomain generates each rank's piece of a regular
quadrilateral, hexahedral, triangular (every triangleFlag) or tetrahedral grid
directly (cartesianPartitioning.cpp): no global mesh, no graph partitioner, no
PETSc. Phase 1 of the Cartesian partitioner
campaign made partitioning the global grid with prescribed owners PETSc-free and
pinned its arrays to the old partitioner's; this pins the generated piece to
that, entity by entity: the offsets and subdomain-to-global maps, the old-to-new
permutations (which become functions), the global counts, and every array of the
subdomain mesh, floating-point ones included -- on every rank, for rank grids
from 1x1 to 3x5, uneven bricks, and rank grids that leave ranks empty. The
union-jack triangles are the hard case: their diagonals alternate with the
cell's parity, and some triangles miss their cell's lowest corner, so their
owners and numbers are not those of a box of cells.
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


def _launch(args, n_ranks, blocked=False, timeout=300):
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
    result = subprocess.run([mpiexec, '-n', str(n_ranks), sys.executable] + script +
                            [os.path.join(HERE, 'cartesian_subdomain_mpi.py')] + [str(a) for a in args],
                            capture_output=True, text=True, env=env, timeout=timeout, cwd=HERE)
    assert result.returncode == 0, result.stdout + result.stderr
    for line in result.stdout.splitlines():
        if line.startswith('RESULT'):
            return dict(item.split('=', 1) for item in line.split()[1:]), result.stdout
    raise AssertionError(result.stdout + result.stderr)


# triN: triangles split with triangleFlag N (0 right-leaning, 1 union jack, 2 left-leaning)
KINDS = [('quad', (9, 7)), ('hex', (5, 4, 6)), ('tri0', (9, 7)), ('tri1', (9, 7)), ('tri2', (9, 7)),
         ('tet', (5, 4, 6))]


def _ranks(grid):
    n = 1
    for p in grid:
        n *= p
    return n


@pytest.mark.parametrize('n_ranks', [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize('kind,nn', KINDS)
def test_generated_piece_is_the_partitioned_piece(kind, nn, n_ranks):
    got, out = _launch([kind] + list(nn), n_ranks)
    assert int(got['mismatches']) == 0, out


@pytest.mark.parametrize('kind,nn,grid', [('quad', (9, 7), (1, 5)),
                                          ('quad', (9, 7), (5, 1)),
                                          ('quad', (16, 11), (3, 5)),
                                          ('tri1', (16, 11), (3, 5)),
                                          ('tri1', (8, 6), (3, 2)),
                                          ('tri2', (9, 7), (1, 5)),
                                          ('hex', (5, 4, 6), (1, 2, 3)),
                                          ('hex', (6, 5, 4), (3, 1, 2)),
                                          ('tet', (6, 5, 4), (3, 1, 2)),
                                          ('tet', (4, 5, 6), (1, 1, 5))])
def test_odd_rank_grids(kind, nn, grid):
    got, out = _launch([kind] + list(nn) + list(grid), _ranks(grid))
    assert int(got['mismatches']) == 0, out


@pytest.mark.parametrize('kind,nn,grid', [('quad', (3, 3), (5, 1)),
                                          ('quad', (2, 2), (3, 2)),
                                          ('tri1', (3, 3), (5, 1)),
                                          ('tri2', (2, 3), (3, 2)),
                                          ('hex', (3, 2, 2), (4, 1, 1)),
                                          ('tet', (3, 2, 2), (4, 1, 1))])
def test_rank_grids_with_empty_ranks(kind, nn, grid):
    got, out = _launch([kind] + list(nn) + list(grid), _ranks(grid))
    assert int(got['mismatches']) == 0, out


@pytest.mark.parametrize('kind,nn', KINDS)
def test_generation_needs_no_petsc(kind, nn):
    got, out = _launch([kind] + list(nn), 4, blocked=True)
    assert int(got['mismatches']) == 0, out
    assert got['petsc'] == '0', out
