#!/usr/bin/env python
"""FAS, the paper's nonlinear multigrid, in parallel.

The paper's degenerate advection-diffusion problem (test_fas.py's, extended to the
unit square) solved by nested iteration whose levels are FAS cycles. Serial FAS is
pinned by test_fas.py; what is pinned here is that parallel runs are the same
algorithm:

* With the pointwise smoother (NLJacobi) a parallel cycle *is* the serial cycle: the
  transfers are exact, the coarse "solver" is more smoothing, and nothing is
  decomposed. So with each level stopped early the answer -- not near the converged
  one but the unconverged iterate -- must agree to roundoff at any rank count.
* Gauss-Seidel and the Schwarz smoother are processor-block methods in parallel, so
  they are not rank-independent per cycle; they are held to the property that matters,
  reaching the Newton answer on any rank count.
* A stacked system (two uncoupled copies) gives each component the single-component
  answer, which is what the stacked restriction and its row sums have to deliver.

What parallel FAS needed beyond the transfer operators: the coarse level's work
vectors as ParVecs (the FAS coarse solve refreshes its ghosts), the smoothers
refreshing ghost entries after each update, the stacked restriction row sums
(`restrictSumList`) the paper's I-tilde divides a restricted solution by, a Schwarz
sweep over *owned* stars only -- which must run in ascending node order, since the
serial sweep runs downward and "the first ownedNodes" of that order are the ghosts --
and Gauss-Seidel reading the owners' residual in ghost rows, without which the sweep
has no fixed point at the solution and stalls short of it.
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


def _run(args, n_ranks, timeout=600):
    mpiexec = _mpiexec()
    if mpiexec is None:
        pytest.skip('no MPI launcher available')
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fas_parallel_mpi.py')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [p for p in sys.path if p] + [env.get('PYTHONPATH', '')]).strip(os.pathsep)
    result = subprocess.run([mpiexec, '-n', str(n_ranks), sys.executable, helper] + [str(a) for a in args],
                            capture_output=True, text=True, env=env, timeout=timeout)
    assert result.returncode == 0, result.stdout + result.stderr
    for line in result.stdout.splitlines():
        if line.startswith('RESULT'):
            return dict(item.split('=', 1) for item in line.split()[1:])
    raise AssertionError(result.stdout + result.stderr)


def _checksum(got):
    return float(got['checksum'])


@pytest.fixture(scope='module')
def newton_reference():
    return _checksum(_run(['newton', 'NLJacobi', 3], 1))


@pytest.mark.parametrize('n_ranks', [2, 3, 4])
def test_jacobi_fas_cycle_is_the_serial_cycle(n_ranks):
    # tolerance 0.3 stops every level early: this is the iterate after a few cycles
    args = ['fas', 'NLJacobi', 3, 2, 10, 3, 9, 0.3]
    serial = _checksum(_run(args, 1))
    parallel = _checksum(_run(args, n_ranks))
    assert abs(parallel - serial) < 1.0e-11 * abs(serial), (n_ranks, parallel, serial)


def test_the_unconverged_iterate_is_not_the_converged_answer(newton_reference):
    # the control for the test above: it can only fail if the cycle differs
    early = _checksum(_run(['fas', 'NLJacobi', 3, 2, 10, 3, 9, 0.3], 1))
    assert abs(early - newton_reference) > 1.0e-3 * abs(newton_reference), (early, newton_reference)


@pytest.mark.parametrize('smoother', ['NLGaussSeidel', 'NLStarILU'])
@pytest.mark.parametrize('n_ranks', [2, 3, 4, 6])
def test_block_smoothers_reach_the_newton_answer(smoother, n_ranks, newton_reference):
    got = _run(['fas', smoother, 3, 2, 10, 3], n_ranks)
    assert abs(_checksum(got) - newton_reference) < 1.0e-9 * abs(newton_reference), (got, newton_reference)


@pytest.mark.parametrize('smoother', ['NLJacobi', 'NLStarILU'])
@pytest.mark.parametrize('n_ranks', [1, 2, 3])
def test_a_stacked_system_gives_each_component_the_single_answer(smoother, n_ranks):
    tolFac = 0.3 if smoother == 'NLJacobi' else '-'
    single = _checksum(_run(['fas', smoother, 3, 2, 10, 3, 9, tolFac], 1))
    got = _run(['fas', smoother, 3, 2, 10, 3, 9, tolFac, 2], n_ranks)
    for value in got['checksums'].split(','):
        assert abs(float(value) - single) < 1.0e-9 * abs(single), (got, single)
