#!/usr/bin/env python
"""How the coarsest level of a parallel multigrid cycle is solved.

The default is a parallel sparse direct solver ('direct': MUMPS, else SuperLU_DIST). The
alternative is to gather the coarse matrix onto every rank and factor it there ('redundant'):
exact, and cheap while the coarse level is small -- but replicated, so with weak scaling it
grows with the rank count (measured: 91% of the flops and 8.7 of 12.6 seconds at 64 ranks on
a 3-level hexahedral hierarchy; MULTILEVEL_MG_CAMPAIGN.md 3.21). ``n.linearCoarseSolver``:

* 'direct' -- exact and distributed, so a cycle is the same cycle as with 'redundant' to
  roundoff, for proteus's MGM and for PETSc's PCMG.
* 'amg' -- one algebraic multigrid V-cycle. Scalable but *not* exact, so a single cycle is
  different (the control below), and what is left to check is that it is still a good
  enough coarse solver for the Krylov iteration around it to reach the direct solution.
"""
import os
import shutil
import subprocess
import sys

import pytest

from proteus import LinearSolvers


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ('mpiexec', 'mpirun'):
        candidate = os.path.join(bindir, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which('mpiexec') or shutil.which('mpirun')


def _checksum(kind, mode, n_ranks, coarse='direct', levels=3):
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip('no MPI launcher available')
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cube_mg_mpi.py')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)
    result = subprocess.run([mpiexec, '-n', str(n_ranks), sys.executable, helper, kind, '3', str(levels),
                             mode, '1', coarse],
                            capture_output=True, text=True, env=env, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    for line in result.stdout.splitlines():
        if line.startswith('RESULT'):
            return float(line.split('checksum=')[1].split()[0])
    raise AssertionError(result.stdout + result.stderr)


@pytest.mark.parametrize('kind', ['quad', 'hex'])
@pytest.mark.parametrize('mode', ['mgm', 'pcmg'])
@pytest.mark.parametrize('n_ranks', [2, 3])
def test_a_distributed_direct_coarse_solve_is_the_redundant_one(kind, mode, n_ranks):
    redundant = _checksum(kind, mode, n_ranks, 'redundant')
    direct = _checksum(kind, mode, n_ranks)      # the default
    assert abs(direct - redundant) < 1.0e-11 * abs(redundant), (kind, mode, n_ranks, direct, redundant)


@pytest.mark.parametrize('kind', ['quad', 'hex'])
def test_an_amg_coarse_solve_is_not_exact(kind):
    # the control for the next test: with one cycle the answer does differ
    exact = _checksum(kind, 'mgm', 3)
    amg = _checksum(kind, 'mgm', 3, 'amg')
    assert abs(amg - exact) > 1.0e-6 * abs(exact), (kind, amg, exact)


@pytest.mark.parametrize('kind', ['quad', 'hex'])
@pytest.mark.parametrize('coarse', ['direct', 'amg'])
def test_krylov_over_pcmg_reaches_the_direct_solution_with_either(kind, coarse):
    direct = _checksum(kind, 'direct', 1)
    converged = _checksum(kind, 'pcmg_its', 3, coarse)
    assert abs(converged - direct) < 1.0e-9 * abs(direct), (kind, coarse, converged, direct)


def test_an_unknown_coarse_solver_is_refused_not_ignored():
    with pytest.raises(ValueError, match='coarseSolver'):
        LinearSolvers.coarseSolverOptions('not-a-solver')
    assert [name for name, _ in LinearSolvers.coarseSolverOptions('redundant')] == [
        'ksp_type', 'pc_type', 'redundant_pc_type']
