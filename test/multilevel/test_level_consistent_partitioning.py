#!/usr/bin/env python
"""Level-consistent partitioning: a finer level inherits the coarser one's partition.

Partitioning every level independently gives each its own graph cut, so the
rank that owns a fine node has nothing to do with the ranks that own its coarse
parents. The prolongation between them is then an all-to-all, and -- more to the
point for building operators without gathering anything -- a rank cannot form
its rows of the prolongation from what it holds. With inheritance a fine node
goes to the owner of one of its parents (a fine element to the owner of its
parent), and then every parent of an owned fine node is in the rank's coarse
subdomain.

What is pinned here:

* with inheritance no owned fine node has a parent outside the coarse subdomain
  (``missing`` is zero), on simplices, quadrilaterals and hexahedra, for node and
  element partitioning; and the off-process share of the prolongation is small;
* the control is real: without inheritance ``missing`` is large, so the zero
  above is the partitioner's doing and not a property of the small meshes;
* the element table the subdomain hierarchy relies on agrees with matching
  elements through their nodes, which does not depend on how the partitioner
  numbers anything;
* an operator build that would need a missing parent refuses, on every rank,
  rather than hanging the ranks that did not notice (the Mat is assembled
  collectively).
"""
import os
import shutil
import subprocess
import sys

import pytest

# these test the global hierarchy builder (see conftest.py), not hierarchies
# generated in pieces, which test_generated_hierarchy.py covers
pytestmark = pytest.mark.global_hierarchy


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ('mpiexec', 'mpirun'):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which('mpiexec') or shutil.which('mpirun')


def _launch(helper, args, n_ranks, env_extra=None, check=True, timeout=300):
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip('no MPI launcher available')
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), helper)
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)
    env.update(env_extra or {})
    result = subprocess.run([mpiexec, '-n', str(n_ranks), sys.executable, helper] + args,
                            capture_output=True, text=True, env=env, timeout=timeout)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def _measure(kind, consistent, by, n_ranks, levels=3):
    result = _launch('level_consistent_mpi.py', [kind, str(levels), str(int(consistent)), by],
                     n_ranks)
    line = [l for l in result.stdout.splitlines() if l.startswith('RESULT')][0]
    return dict(item.split('=') for item in line.split()[1:])


OFFPROC_BOUND = 0.25
# the control: independent cuts are random, 0.24-0.99 over a few runs on a 4-rank quad mesh, so
# only a bound well below the worst luck can be asserted of them
INDEPENDENT_OFFPROC_FLOOR = 0.15


@pytest.mark.parametrize('kind', ['tri', 'tet', 'quad', 'hex'])
@pytest.mark.parametrize('by', ['node', 'element'])
@pytest.mark.parametrize('n_ranks', [2, 4])
def test_every_parent_of_an_owned_node_is_on_the_rank(kind, by, n_ranks):
    got = _measure(kind, True, by, n_ranks)
    assert int(got['missing']) == 0, got
    assert int(got['lookup_disagreements']) == 0, got
    if by == 'node':
        # PETSc then communicates with neighbours only; independent cuts put most
        # of the prolongation off process (0.4-0.99 on these meshes)
        assert float(got['offproc']) < OFFPROC_BOUND, got


@pytest.mark.parametrize('kind', ['tri', 'tet', 'quad', 'hex'])
@pytest.mark.parametrize('by', ['node', 'element'])
def test_independent_partitions_do_not_have_this_property(kind, by):
    got = _measure(kind, False, by, 4)
    assert int(got['missing']) > 0, got
    if by == 'node':
        # independent cuts of the finer levels vary from run to run (hex on 4
        # ranks: 0.40-0.99 over ten runs, quad 0.24-0.90, against a fixed ~0.12 inherited),
        # so the control is only that they stay well above what inheritance gets
        assert float(got['offproc']) > INDEPENDENT_OFFPROC_FLOOR, got


@pytest.mark.parametrize('space,degree', [('p2', 2), ('ncp1', 1)])
def test_operators_without_level_consistency_are_refused_not_wrong(space, degree):
    result = _launch('space_transfer_mpi.py', [space, '3', str(degree)], 4,
                     env_extra={'PROTEUS_TEST_INDEPENDENT_PARTITIONS': '1'},
                     check=False)
    assert result.returncode != 0, result.stdout
    assert 'level-consistent' in result.stdout + result.stderr, result.stdout + result.stderr
