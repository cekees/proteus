#!/usr/bin/env python
"""Q2 (tensor-product quadratic Lagrange) on regular quadrilateral and hexahedral grids, in parallel.

The 3D map came from a partition-aware C builder and worked; the 2D one built the
numbering from the *global* mesh and so ran only in serial. It now follows from
the partitioner's own vertex, edge and cell numberings
(QuadraticLagrangeCubeDOFMap._buildPartitionedQuadraticQuad).

Getting it to run exposed a partitioner defect that Q1 could not show: a quadrilateral's
edge was assigned to a rank by the *simplex* rule ("every node of the element but the one
across from the face"), which for a quad names nodes the edge does not contain. The owner
could then hold neither end point, and so not both cells that share the edge; its row of a
space with dofs on edges was assembled from one cell and the solution was wrong
(0.39 off on a 4x4 grid, differently on every rank count). Q1 has no edge dofs and never
noticed. test_parallel_cube_meshes.py pins the partitioner side; here are the consequences:

* transfer operators reproduce a function of the coarse space exactly (Q1 and Q2);
* a single-level Q2 solve is the serial one on any number of ranks, and exact for the
  polynomial problem it solves;
* with Jacobi smoothing proteus's MGM cycle is the serial cycle, PETSc's PCMG is that
  cycle, and PCMG-preconditioned GMRES reaches the direct solution.
"""
import os
import shutil
import subprocess
import sys

import pytest


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ('mpiexec', 'mpirun'):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which('mpiexec') or shutil.which('mpirun')


def _launch(helper, args, n_ranks, timeout=600):
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip('no MPI launcher available')
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), helper)
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)
    result = subprocess.run([mpiexec, '-n', str(n_ranks), sys.executable, path] + [str(a) for a in args],
                            capture_output=True, text=True, env=env, timeout=timeout)
    assert result.returncode == 0, result.stdout + result.stderr
    for line in result.stdout.splitlines():
        if line.startswith('RESULT'):
            return line
    raise AssertionError(result.stdout + result.stderr)


def _checksum(kind, nn, levels, mode, n_ranks, order=2):
    line = _launch('cube_mg_mpi.py', [kind, nn, levels, mode, order], n_ranks)
    return float(line.split('checksum=')[1].split()[0])


@pytest.mark.parametrize('kind', ['quad', 'hex'])
@pytest.mark.parametrize('order', [1, 2])
@pytest.mark.parametrize('n_ranks', [1, 2, 3])
def test_transfers_reproduce_a_coarse_space_function(kind, order, n_ranks):
    line = _launch('cube_transfer_mpi.py', [kind, 3, order], n_ranks)
    err = float(line.split('maxerr=')[1].split()[0])
    scale = float(line.split('scale=')[1].split()[0])
    assert scale > 1.0, line
    assert err < 1.0e-12 * scale, line


@pytest.mark.parametrize('n_ranks', [2, 3, 4])
def test_single_level_q2_quad_is_the_serial_solve(n_ranks):
    serial = _checksum('quad', 4, 1, 'ksplu', 1)
    parallel = _checksum('quad', 4, 1, 'ksplu', n_ranks)
    assert abs(parallel - serial) < 1.0e-9 * abs(serial), (n_ranks, parallel, serial)


@pytest.mark.parametrize('kind', ['quad', 'hex'])
@pytest.mark.parametrize('n_ranks', [2, 3])
def test_parallel_q2_cycle_is_the_serial_cycle(kind, n_ranks):
    serial = _checksum(kind, 3, 3, 'mgm', 1)
    parallel = _checksum(kind, 3, 3, 'mgm', n_ranks)
    assert abs(parallel - serial) < 1.0e-11 * abs(serial), (kind, n_ranks, parallel, serial)


@pytest.mark.parametrize('kind', ['quad', 'hex'])
@pytest.mark.parametrize('n_ranks', [2, 3])
def test_q2_pcmg_cycle_is_the_mgm_cycle(kind, n_ranks):
    mgm = _checksum(kind, 3, 3, 'mgm', 1)
    pcmg = _checksum(kind, 3, 3, 'pcmg', n_ranks)
    assert abs(pcmg - mgm) < 1.0e-11 * abs(mgm), (kind, n_ranks, pcmg, mgm)


@pytest.mark.parametrize('kind', ['quad', 'hex'])
def test_q2_pcmg_preconditioned_gmres_reaches_the_direct_solution(kind):
    direct = _checksum(kind, 3, 3, 'direct', 1)
    converged = _checksum(kind, 3, 3, 'pcmg_its', 3)
    assert abs(converged - direct) < 1.0e-9 * abs(direct), (kind, converged, direct)
