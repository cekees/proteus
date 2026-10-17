#!/usr/bin/env python
"""The numerics file's partitioning options reach the mesh, in parallel.

A p/n problem without a domain gets one built by NumericalSolution, which copies
the numerics file's meshing options onto domain.MeshOptions. It used to leave out
parallelPartitioningType and nLayersOfOverlapForParallel, so such problems were
always node-partitioned with one layer of overlap whatever the numerics file
said. For a discontinuous space that changes the answer: node partitioning does
not put every face neighbour of an owned element on the rank, so those faces are
assembled as domain boundary and the solution depends on the rank count.
"""
import os
import shutil
import subprocess
import sys

import pytest

from proteus import MeshTools


def _mpiexec():
    # the launcher next to the interpreter belongs to the MPI petsc4py was built
    # against; another one on PATH can start every rank as a size-1 world
    bindir = os.path.dirname(sys.executable)
    for name in ('mpiexec', 'mpirun'):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which('mpiexec') or shutil.which('mpirun')


def _run(n_ranks, tmp_path):
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip('no MPI launcher available')
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dg_partition_mpi.py')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)
    # CI runners have two cores; let Open MPI (4.x and 5.x/PRRTE) start more
    # ranks than that. MPICH oversubscribes by default and ignores these.
    env.setdefault('OMPI_MCA_rmaps_base_oversubscribe', '1')
    env.setdefault('PRTE_MCA_rmaps_default_mapping_policy', ':oversubscribe')
    result = subprocess.run([mpiexec, '-n', str(n_ranks), sys.executable, helper, 'dg_partition'],
                            capture_output=True, text=True, env=env, timeout=300,
                            cwd=str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr
    line = [l for l in result.stdout.splitlines() if l.startswith('RESULT')][0]
    return dict(item.split('=') for item in line.split()[1:])


@pytest.mark.parametrize('n_ranks', [2, 3])
def test_numerics_partitioning_reaches_the_mesh(n_ranks, tmp_path):
    got = _run(n_ranks, tmp_path)
    assert int(got['partitioning']) == MeshTools.MeshParallelPartitioningTypes.element, got
    assert int(got['overlap']) == 1, got


@pytest.mark.parametrize('n_ranks', [2, 3])
def test_discontinuous_solution_does_not_depend_on_rank_count(n_ranks, tmp_path):
    serial = float(_run(1, tmp_path)['checksum'])
    parallel = float(_run(n_ranks, tmp_path)['checksum'])
    assert abs(parallel - serial) < 1.0e-10 * abs(serial), (n_ranks, parallel, serial)
