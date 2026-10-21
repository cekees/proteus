#!/usr/bin/env python
"""How evenly a generated hierarchy's bricks spread the work.

Level-consistent partitioning has every finer level inherit the coarser one's cut, so the
cut is chosen once, on the coarsest grid, and what it costs shows on the finest level, where
nearly all the work is. The generated Cartesian bricks used to be cut on coarse *nodes*:
11 coarse nodes over 2 ranks is 6|5, scaled up three levels it is 24|17 fine nodes per axis
(the trailing fine layers going to the lower brick), a factor 2.8 in elements in 3D. They are
now cut on coarse *cells* -- 10 cells over 2 ranks is 5|5, fine nodes 20|21 -- so the cut is
off by one fine node per axis when the cells divide evenly, and by at most one coarse cell
otherwise.

Pinned here: the node counts per rank of the owner arrays against the rule they replaced
(no MPI needed), and the same measured on real generated hierarchies through
MultilevelMesh.partitionBalance (max/mean over ranks of owned nodes and of subdomain
elements, the assembly work).
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from proteus import MeshTools


def _old_starts(nn, p, level):
    """The brick starts before the change: the default node chunks, scaled up."""
    chunks = np.cumsum([0] + [nn // p + (nn % p > a) for a in range(p)])
    return np.minimum(chunks << level, ((nn - 1) << level) + 1)


def _imbalance(counts):
    return counts.max() / counts.mean()


def _axis_imbalance(nn, p, level, starts):
    return _imbalance(np.diff(starts(nn, p, level)).astype(float))


@pytest.mark.parametrize('nn,p,level', [(11, 2, 2), (12, 2, 2), (18, 3, 2), (24, 4, 2),
                                        (13, 2, 3), (21, 4, 2), (9, 2, 1)])
def test_the_cell_cut_is_never_worse_than_the_node_cut(nn, p, level):
    new = _axis_imbalance(nn, p, level, lambda n, q, l: MeshTools._cartesianBrickStarts(n, q, l))
    old = _axis_imbalance(nn, p, level, _old_starts)
    assert new <= old + 1e-12, (nn, p, level, new, old)


@pytest.mark.parametrize('nn,p,level', [(11, 2, 2), (13, 2, 3), (13, 3, 2), (25, 4, 2), (9, 4, 1)])
def test_the_cell_cut_is_exact_to_one_fine_node_when_the_cells_divide(nn, p, level):
    assert (nn - 1) % p == 0
    starts = MeshTools._cartesianBrickStarts(nn, p, level)
    widths = np.diff(starts)
    assert widths.max() - widths.min() <= 1, (nn, p, level, widths)


@pytest.mark.parametrize('nn,p,level', [(12, 2, 2), (22, 4, 2), (14, 3, 2)])
def test_otherwise_it_is_off_by_one_coarse_cell(nn, p, level):
    widths = np.diff(MeshTools._cartesianBrickStarts(nn, p, level))
    assert widths.max() - widths.min() <= (1 << level) + 1, (nn, p, level, widths)


def test_the_old_rule_was_as_unbalanced_as_the_notes_say():
    # the control: 11 coarse nodes over 2 ranks, three levels
    assert np.diff(_old_starts(11, 2, 2)).tolist() == [24, 17]
    assert np.diff(MeshTools._cartesianBrickStarts(11, 2, 2)).tolist() == [20, 21]


def test_an_axis_with_fewer_cells_than_ranks_keeps_the_node_chunks():
    assert MeshTools._cartesianBrickStarts(3, 3, 0).tolist() == [0, 1, 2, 3]
    assert MeshTools._cartesianBrickStarts(2, 2, 2).tolist() == [0, 4, 5]
    assert MeshTools._cartesianBrickStarts(1, 3, 2).tolist() == [0, 1, 1, 1]


@pytest.mark.parametrize('nn,size', [(11, 4), (12, 8), (13, 6)])
def test_every_node_has_one_owner_and_the_owners_are_the_bricks(nn, size):
    owners = MeshTools.cartesianNodeOwners(nn, nn, nn, size=size, level=2)
    assert owners.size == (4 * (nn - 1) + 1) ** 3
    assert set(np.unique(owners).tolist()) == set(range(size))


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ('mpiexec', 'mpirun'):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which('mpiexec') or shutil.which('mpirun')


def _balance(kind, nn, levels, n_ranks):
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip('no MPI launcher available')
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'balance_mpi.py')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)
    result = subprocess.run([mpiexec, '-n', str(n_ranks), sys.executable, helper, kind, str(nn), str(levels),
                             'generated'], capture_output=True, text=True, env=env, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    line = [l for l in result.stdout.splitlines() if l.startswith('RESULT')][0]
    return dict((k, float(v)) for k, v in (item.split('=') for item in line.split()[6:]))


@pytest.mark.parametrize('kind', ['hex', 'quad', 'tet', 'tri'])
@pytest.mark.parametrize('n_ranks', [2, 4])
def test_the_finest_level_of_a_generated_hierarchy_is_balanced(kind, n_ranks):
    got = _balance(kind, 11, 3, n_ranks)
    # 10 cells split evenly: the finest level differs by one node per axis
    assert got['L2.nodes'] < 1.06, got
    assert got['L2.elements'] < 1.07, got
    # the surface cost of the cut is the one a brick has, not something worse
    assert got['L2.ghost'] < 0.2, got
